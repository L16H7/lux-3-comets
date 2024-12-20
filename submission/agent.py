# It throws error if I don't put it.
import absl.logging
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


    team_positions = observations.units.position[:, team_idx, ...]
    team_positions = team_positions if team_idx == 0 else transform_coordinates(team_positions)

    opponent_positions = observations.units.position[:, opponent_idx, ...]
    opponent_positions = opponent_positions if team_idx == 0 else transform_coordinates(opponent_positions)
    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions
    )
    opponent_positions = jnp.where(
        opponent_positions == 24,
        -100,
        opponent_positions
    )

    opponent_positions = opponent_positions + Constants.MAX_SAP_RANGE
    diff = -team_positions[:, :, None, :] + opponent_positions[:, None, :, :]
    diff = jnp.where(diff < 0, -100, diff)

    # Function to set True for one row given indices
    def set_true_row(bool_array, indices):
        return bool_array.at[indices].set(True)

    # Vectorize the function across rows using vmap
    def update_bool_array(bool_array, turn_ons):
        # vmap across the first axis (rows of turn_ons and bool_array)
        return jax.vmap(set_true_row, in_axes=(0, 0), out_axes=0)(bool_array, turn_ons)

    # Use JIT compilation for performance
    update_bool_array_jit = jax.jit(update_bool_array)

    bool_array = jnp.zeros_like(
        jnp.squeeze(logits[1], axis=0),
        dtype=bool
    )

    diff = diff.reshape(-1, 16, 2)
    x = diff[..., 0]
    attack_x = update_bool_array_jit(bool_array, x)

    y = diff[..., 1]
    attack_y = update_bool_array_jit(bool_array, y)

    attack_available = attack_x.sum(-1) & attack_y.sum(-1) & ((diff.sum(-1) < (4 * Constants.MAX_SAP_RANGE)).sum(-1) > 0)

    logits1_mask = jnp.concat(
        [ 
            jnp.ones((1, attack_available.shape[0], 1)),
            valid_movements.reshape(1, -1, 4),
            attack_available.reshape(1, -1, 1) 
        ],
        axis=-1
    )

    logits2_mask = attack_x
    logits3_mask = attack_y

    logits2_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits2_mask
    )
    logits2_mask = jnp.expand_dims(logits2_mask, axis=0)

    logits3_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits3_mask
    )
    logits3_mask = jnp.expand_dims(logits3_mask, axis=0)

    logits1, logits2, logits3 = logits
    large_negative = -1e9
    masked_logits1 = jnp.where(logits1_mask, logits1, large_negative)
    masked_logits2 = jnp.where(logits2_mask, logits2, large_negative)
    masked_logits3 = jnp.where(logits3_mask, logits3, large_negative)

    '''
    sap_range_clip = Constants.MAX_SAP_RANGE - 1
    logits2 = logits2.at[..., : sap_range_clip].set(-100)
    logits2 = logits2.at[..., -sap_range_clip:].set(-100)

    logits3 = logits3.at[..., : sap_range_clip].set(-100)
    logits3 = logits3.at[..., -sap_range_clip:].set(-100)
    '''

    masked_logits2 = masked_logits2.at[..., : sap_ranges].set(large_negative)
    masked_logits2 = masked_logits2.at[..., -sap_ranges:].set(large_negative)

    masked_logits3 = masked_logits3.at[..., : sap_ranges].set(large_negative)
    masked_logits3 = masked_logits3.at[..., -sap_ranges:].set(large_negative)

    # action1 = np.argmax(masked_logits1, axis=-1)
    # action2 = np.argmax(masked_logits2, axis=-1)
    # action3 = np.argmax(masked_logits3, axis=-1)
    action1 = jax.random.categorical(rng, masked_logits1, axis=-1)
    action2 = jax.random.categorical(rng, masked_logits2, axis=-1)
    action3 = jax.random.categorical(rng, masked_logits3, axis=-1)

    return [action1, action2, action3]


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player

        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg
        self.sap_range = env_cfg["unit_sap_range"]

        checkpoint_path = os.path.join(script_dir, 'checkpoint')
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.params = orbax_checkpointer.restore(checkpoint_path)

        self.rng = jax.random.PRNGKey(42)
        
        self.actor = Actor(n_actions=6)

        BATCH = 16
        SEQ = 1
        self.actor_hstates = ScannedRNN.initialize_carry(16, 256)
        # self.params = self.actor.init(self.rng, self.actor_hstates, {
        #     "observations": jnp.zeros((SEQ, BATCH, 9, 24, 24)),
        #     "prev_actions": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
        #     "match_phases": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
        #     "positions": jnp.zeros((SEQ, BATCH, 2)),
        #     "prev_points": jnp.zeros((SEQ, BATCH, 1)),
        #     "team_points": jnp.zeros((SEQ, BATCH, 1)),
        #     "opponent_points": jnp.zeros((SEQ, BATCH, 1)),
        # })['params']
        self.inference_fn = jax.jit(lambda x, x1, x2: self.actor.apply(x, x1, x2))

        self.discovered_relic_nodes = np.ones((1, 6, 2)) * -1

        self.prev_actions = jnp.zeros((1, 16), dtype=jnp.int32)
        self.prev_points = jnp.zeros((1, 16, 1))
        self.prev_team_points = 0
        self.prev_opponent_points = 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        observation = DotDict(reshape_observation(obs))

        relic_mask = observation.relic_nodes != -1
        self.discovered_relic_nodes[relic_mask] = observation.relic_nodes[relic_mask]

        discovered_relic_nodes = self.discovered_relic_nodes
        if self.team_id == 1:
            discovered_relic_nodes = jnp.concatenate(
                (self.discovered_relic_nodes[:, 3:, :], self.discovered_relic_nodes[:, :3, :]),
                axis=1
            )

        representations = create_representations(
            obs=observation,
            discovered_relic_nodes=discovered_relic_nodes,
            max_steps_in_match=100,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id
        )
        
        (
            state,
            episode_info,
            agent_positions,
            _,
        ) = representations

        agent_observations = jnp.expand_dims(state, axis=0).repeat(16, axis=1) # 1, N_TOTAL_AGENTS, 9, 24, 24
        agent_episode_info = episode_info.repeat(16, axis=0)

        BATCH = 16
        SEQ = 1
 
        logits, self.actor_hstates = self.inference_fn(
            { "params": self.params },
            self.actor_hstates,
            {
                "observations": agent_observations,
                "prev_actions": self.prev_actions,
                "match_phases": jnp.expand_dims(agent_episode_info[:, 0].astype(jnp.int32), axis=0),
                "positions": agent_positions,
                "prev_points": self.prev_points,
                "team_points": jnp.expand_dims(agent_episode_info[:, 2], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(agent_episode_info[:, 3], axis=[0, -1]),

            }
        )

        self.rng, action_rng = jax.random.split(self.rng)
        actions = get_actions(
            rng=action_rng,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
            logits=logits,
            observations=observation,
            sap_ranges=self.sap_range,
        )

        # previous action doesn't need to modified for agent1 because we only transform actions
        # when we submit to the engine
        self.prev_actions = actions[0].reshape(1, 16)

        # if self.team_id == 1:
            # actions[1], actions[2] = actions[2], actions[1]

        actions = jnp.squeeze(jnp.stack(actions), axis=1).T

        transformed_p1_actions = jnp.zeros_like(actions)
        transformed_p1_actions = transformed_p1_actions.at[:, 0].set(vectorized_transform_actions(actions[:, 0]))
        transformed_p1_actions = transformed_p1_actions.at[:, 1].set(actions[:, 2])
        transformed_p1_actions = transformed_p1_actions.at[:, 2].set(actions[:, 1])
        actions = actions if self.team_id == 0 else transformed_p1_actions

        actions = actions.at[:, 1:].set(actions[:, 1:] - 8)

        team_points = obs['team_points'][self.team_id]
        points_gained = (team_points - self.prev_team_points) / 16.0
        points_gained = jnp.expand_dims(points_gained.repeat(16), axis=[0, -1])

        self.prev_team_points = team_points
        self.prev_points = points_gained
        
        return actions
        # return jnp.zeros((16, 3), jnp.int32)
