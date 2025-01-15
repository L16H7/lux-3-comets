# It throws error if I don't put it.
import absl.logging
import jax.random
import jax.random
absl.logging.set_verbosity(absl.logging.ERROR)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import jax
import jax.numpy as jnp
import orbax.checkpoint
import numpy as np

from constants import Constants
from representation import create_representations, transform_coordinates
from rnn import Actor, ScannedRNN


class DotDict:
    """A class that recursively converts dictionaries to objects 
    accessible with dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                value = DotDict(value)
            setattr(self, key, value)

expand_repeat = jax.jit(lambda x: jnp.expand_dims(x.repeat(16, axis=0), axis=0))

def reshape_observation(obs):
    new_obs = dict(obs)
    new_obs["units"] = dict(new_obs["units"]) # copy so we can modify
    new_obs["units"]["position"] = new_obs["units"]["position"][None, :]
    new_obs["units"]["energy"] = new_obs["units"]["energy"][None, :]

    new_obs["units_mask"] = new_obs["units_mask"][None, :]
    new_obs["relic_nodes"] = new_obs["relic_nodes"][None, :]
    new_obs["relic_nodes_mask"] = new_obs["relic_nodes_mask"][None, :]
    new_obs["sensor_mask"] = new_obs["sensor_mask"][None, :]

    new_obs["map_features"] = dict(new_obs["map_features"]) # copy so we can modify
    new_obs["map_features"]["energy"] = new_obs["map_features"]["energy"][None, :]
    new_obs["map_features"]["tile_type"] = new_obs["map_features"]["tile_type"][None, :]

    new_obs["team_points"] = new_obs["team_points"][None, :]

    new_obs["steps"] = jnp.atleast_1d(new_obs["steps"])
    new_obs["match_steps"] = jnp.atleast_1d(new_obs["match_steps"])

    return new_obs

directions = jnp.array(
    [
        [0, -1],    # Move up
        [1, 0],     # Move right
        [0, 1],     # Move down
        [-1, 0],    # Move left
    ],
    dtype=jnp.int16,
)

def vectorized_transform_actions(actions):
    # Create a JAX array that maps each action index to its new action
    # Index:      0  1  2  3  4 . 5
    # Action map: 0  2  1  4  3 . 5
    action_map = jnp.array([0, 2, 1, 4, 3, 5])

    # Vectorized mapping
    transformed_actions = action_map[actions]
    return transformed_actions

def transform_observation(obs):
    # Horizontal flip across the last dimension (24, 24 grids)
    flipped = jnp.flip(obs, axis=2)
    
    # Rotate 90 degrees clockwise after flip, across the last two dimensions (24x24)
    rotated = jnp.rot90(flipped, k=-1, axes=(1, 2))
    
    return rotated

def generate_attack_masks(agent_positions, target_positions, x_range=8, y_range=8, choose_y=False, chosen_x=None,):
    """
    Generate attack masks for agents based on both x and y distances to targets.
    Targets outside the range are filtered out before mask generation.
    
    Args:
        agent_positions (jnp.ndarray): Shape (num_agents, 2) array of agent positions
        target_positions (jnp.ndarray): Shape (num_targets, 2) array of target positions
        x_range (int): Maximum x-distance range (default 8)
        y_range (int): Maximum y-distance range (default 8)
    
    Returns:
        attack_masks (jnp.ndarray): Shape (num_agents, 17, 17) boolean array
                                   True indicates valid attack position at that offset
    """
    # Pre-filter invalid targets (marked as -1)
    valid_targets = target_positions != -1
    valid_targets = jnp.all(valid_targets, axis=-1)
    target_positions = jnp.where(valid_targets[:, None], target_positions, 1000)
    
    # Calculate x and y distances from agent to each target
    x_distances = target_positions[None, :, 0] - agent_positions[:, None, 0]
    y_distances = target_positions[None, :, 1] - agent_positions[:, None, 1]
    
    # Create range mask for targets
    targets_in_range = (jnp.abs(x_distances) <= x_range) & (jnp.abs(y_distances) <= y_range)
    targets_in_range = targets_in_range & valid_targets[None, :]

    target_positions = jnp.where(target_positions == -1, 1000, target_positions)

    x_distances = jnp.where(
        targets_in_range,
        x_distances,
        -100, 
    )
    y_distances = jnp.where(
        targets_in_range,
        y_distances,
        -100,
    )
    
    x_offsets = jnp.arange(-8, 9)
    y_offsets = jnp.arange(-8, 9)
    
    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Check valid positions for x and y separately
    valid_x = (x_distances == x_offsets)
    valid_x = jnp.any(valid_x, axis=-1)

    if choose_y:
        x_distances = x_distances[:, None, :]
        y_distances = y_distances[:, None, :]
        x_offsets = x_offsets[None, :, None]
        y_offsets = y_offsets[None, :, None]
        
        # Filter targets based on chosen x
        chosen_x = chosen_x[:, None, None]  # Shape: (num_agents, 1, 1)
        valid_targets_for_x = (x_distances == chosen_x)
        
        # Apply the x-based filter to y distances
        y_distances = jnp.where(valid_targets_for_x, y_distances, -100)
        
        # Generate y masks only for valid targets based on chosen x
        valid_y = (y_distances == y_offsets)
        valid_y = jnp.any(valid_y, axis=-1)

        indices = jnp.arange(valid_y.shape[1])

        # Use advanced indexing to extract the desired slices
        final_filter = valid_y[0, indices, indices, :]
        return final_filter

    valid_y = (y_distances == y_offsets)
    valid_y = jnp.any(valid_y, axis=-1)

    return valid_x, valid_y
    
def get_actions(rng, team_idx: int, opponent_idx: int, logits, observations, sap_ranges):
    n_envs = observations.units.position.shape[0]
    
    agent_positions = observations.units.position[:, team_idx, ..., None, :] 
    agent_positions = agent_positions if team_idx == 0 else transform_coordinates(agent_positions)

    new_positions = agent_positions + directions

    in_bounds = (
        (new_positions[..., 0] >= 0) & (new_positions[..., 0] <= Constants.MAP_WIDTH - 1) &
        (new_positions[..., 1] >= 0) & (new_positions[..., 1] <= Constants.MAP_HEIGHT - 1)
    )

    asteroid_tiles = observations.map_features.tile_type == Constants.ASTEROID_TILE
    asteroid_tiles = asteroid_tiles if team_idx == 0 else transform_observation(asteroid_tiles)

    is_asteroid = asteroid_tiles[
        0, 
        new_positions[..., 0].clip(0, Constants.MAP_WIDTH - 1),
        new_positions[..., 1].clip(0, Constants.MAP_HEIGHT - 1),
    ]
    valid_movements = in_bounds & (~is_asteroid)

    adjacent_offsets = jnp.array(
        [
            [0, 0],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ], dtype=jnp.int16
    )

    transformed_agent_positions = transform_coordinates(agent_positions)
    transformed_agent_positions = filter_targets_with_sensor(
        transformed_agent_positions,
        observations.sensor_mask
    )
    opponent_positions = observations.units.position[:, opponent_idx, ..., None, :] 
    opponent_positions = opponent_positions if team_idx == 0 else transform_coordinates(opponent_positions)

    opponent_positions = jnp.concat([
        opponent_positions,
        transformed_agent_positions, # to attack mirror positions
    ], axis=1)
    
    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions,
    )

    opponent_positions = jnp.where(
        opponent_positions == 24,
        -100,
        opponent_positions,
    )

    target_positions = opponent_positions + adjacent_offsets
    target_x, _ = generate_attack_masks(
        agent_positions=agent_positions.reshape(-1, 2),
        target_positions=target_positions.reshape(-1, 2),
        x_range=sap_ranges,
        y_range=sap_ranges
    )
    logits2_mask = target_x

    logits1_mask = jnp.concat(
        [ 
            jnp.ones((1, 16, 1)),
            valid_movements.reshape(1, -1, 4),
            target_x.sum(axis=-1).reshape(1, 16, 1)
        ],
        axis=-1
    )

    logits1, logits2, logits3 = logits

    logits1_mask = logits1_mask.reshape(logits1.shape)
    logits2_mask = logits2_mask.reshape(logits2.shape)
    large_negative = -1e9
    masked_logits1 = jnp.where(logits1_mask.reshape(logits1.shape), logits1, large_negative)
    masked_logits2 = jnp.where(logits2_mask.reshape(logits2.shape), logits2, large_negative)

    rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
    action1 = jax.random.categorical(rng1, masked_logits1, axis=-1)
    action2 = jax.random.categorical(rng2, masked_logits2, axis=-1)

    target_y = generate_attack_masks(
        agent_positions=agent_positions.reshape(-1, 2),
        target_positions=target_positions.reshape(-1, 2),
        x_range=sap_ranges,
        y_range=sap_ranges,
        choose_y=True,
        chosen_x=action2.reshape(-1) - Constants.MAX_SAP_RANGE
    )

    logits3_mask = target_y
    logits3_mask = logits3_mask.reshape(logits3.shape)
    masked_logits3 = jnp.where(logits3_mask.reshape(logits3.shape), logits3, large_negative)

    action3 = jax.random.categorical(rng3, masked_logits3, axis=-1)

    # action1 = np.argmax(masked_logits1, axis=-1)
    # action2 = np.argmax(masked_logits2, axis=-1)
    # action3 = np.argmax(masked_logits3, axis=-1)

    return [action1, action2, action3]


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player

        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.unit_move_cost = jnp.array([[[env_cfg["unit_move_cost"]]]]).repeat(16, 1) / 8.0
        self.unit_sap_cost = (jnp.array([[[env_cfg["unit_sap_cost"]]]]).repeat(16, 1) - 30.0) / 20.0
        self.unit_sap_range = jnp.array([[[env_cfg["unit_sap_range"]]]]).repeat(16, 1) / 8.0
        self.unit_sensor_range = jnp.array([[[env_cfg["unit_sensor_range"]]]]).repeat(16, 1) / 8.0

        checkpoint_path = os.path.join(script_dir, 'checkpoint')
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.params = orbax_checkpointer.restore(checkpoint_path)
        # BATCH = 16
        # SEQ = 1
        # self.params = self.actor.init(self.rng, self.actor_hstates, {
        #     "observations": jnp.zeros((SEQ, BATCH, 9, 24, 24)),
        #     "match_steps": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
        #     "positions": jnp.zeros((SEQ, BATCH, 2)),
        #     "team_points": jnp.zeros((SEQ, BATCH, 1)),
        #     "opponent_points": jnp.zeros((SEQ, BATCH, 1)),
        # })['params']

        self.rng = jax.random.PRNGKey(42)
        self.actor = Actor(n_actions=6)
        self.actor_hstates = ScannedRNN.initialize_carry(16, 128)

        self.inference_fn = jax.jit(lambda x, x1, x2: self.actor.apply(x, x1, x2))

        self.discovered_relic_nodes = np.ones((1, 6, 2)) * -1
        self.prev_team_points = 0
        self.points_map = jnp.zeros((1, 24, 24), dtype=jnp.float32)
        self.points_gained = 0
        self.prev_agent_positions = jnp.ones((1, 16, 2), dtype=jnp.int32) * -1


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        observation = DotDict(reshape_observation(obs))

        relic_mask = observation.relic_nodes != -1
        self.discovered_relic_nodes[relic_mask] = observation.relic_nodes[relic_mask]

        representations = create_representations(
            obs=observation,
            discovered_relic_nodes=self.discovered_relic_nodes,
            max_steps_in_match=100,
            prev_agent_positions=self.prev_agent_positions,
            points_map=self.points_map,
            points_gained=jnp.array([self.points_gained]),
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
        )
        
        (
            state,
            observations,
            episode_info,
            points_map,
            agent_positions,
            _,
            _,
        ) = representations
        self.points_map = points_map

        agent_states = jnp.expand_dims(state, axis=0).repeat(16, axis=1) # 1, N_TOTAL_AGENTS, 9, 24, 24
        agent_observations = observations.reshape(1, -1, 10, 17, 17)
        agent_episode_info = episode_info.repeat(16, axis=0)

        logits, self.actor_hstates = self.inference_fn(
            { "params": self.params },
            self.actor_hstates,
            {
                "states": agent_states,
                "observations": agent_observations,
                "match_steps": jnp.expand_dims(agent_episode_info[:, 0], axis=[0, -1]),
                "matches": jnp.expand_dims(agent_episode_info[:, 1], axis=[0, -1]),
                "positions": agent_positions,
                "team_points": jnp.expand_dims(agent_episode_info[:, 2], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(agent_episode_info[:, 3], axis=[0, -1]),
                "unit_move_cost": self.unit_move_cost,
                "unit_sap_cost": self.unit_sap_cost,
                "unit_sap_range": self.unit_sap_range,
                "unit_sensor_range": self.unit_sensor_range,
            }
        )

        self.rng, action_rng = jax.random.split(self.rng)

        actions = get_actions(
            rng=action_rng,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
            logits=logits,
            observations=observation,
            sap_ranges=self.env_cfg["unit_sap_range"],
        )

        actions = jnp.squeeze(jnp.stack(actions), axis=1).T

        transformed_targets = transform_coordinates(actions[:, 1:], 17, 17)

        transformed_p1_actions = jnp.zeros_like(actions)
        transformed_p1_actions = transformed_p1_actions.at[:, 0].set(vectorized_transform_actions(actions[:, 0]))
        transformed_p1_actions = transformed_p1_actions.at[:, 1].set(transformed_targets[:, 0])
        transformed_p1_actions = transformed_p1_actions.at[:, 2].set(transformed_targets[:, 1])
        actions = actions if self.team_id == 0 else transformed_p1_actions

        actions = actions.at[:, 1:].set(actions[:, 1:] - 8)

        team_points = obs['team_points'][self.team_id]

        self.points_gained = team_points - self.prev_team_points
        self.prev_team_points = team_points
        self.prev_agent_positions = agent_positions

        if step == 90:
            a = True
        
        return actions


def filter_targets_with_sensor(target_positions, sensor_map):
    """
    Filter target positions, replacing with (-1, -1) if sensor is True at that position.
    
    Args:
        target_positions (jnp.ndarray): Shape (n_envs, 16, 2) array of target positions
        sensor_map (jnp.ndarray): Shape (n_envs, 24, 24) boolean array where True means sensor
        
    Returns:
        jnp.ndarray: Shape (n_envs, 16, 2) filtered target positions
    """
    def process_single_env(targets, sensors):
        def filter_single_target(pos):
            # Check if position is already invalid
            is_valid = jnp.all(pos != -1)
            
            # Get sensor value at target position
            sensor_value = jnp.where(
                is_valid,
                sensors[pos[0], pos[1]],  # Only check sensor if position is valid
                True  # If position was already invalid, keep it invalid
            )
            
            # Replace with -1,-1 if sensor is True
            return jnp.where(sensor_value, jnp.array([-1, -1]), pos)
        
        # Apply to each target position
        return jax.vmap(filter_single_target)(targets)
    
    # Apply to each environment
    return jax.vmap(process_single_env)(target_positions, sensor_map)
