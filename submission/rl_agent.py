# It throws error if I don't put it.
import absl.logging
import jax.random
absl.logging.set_verbosity(absl.logging.ERROR)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import jax
import jax.numpy as jnp
import orbax.checkpoint
import numpy as np

from agent import get_actions, vectorized_transform_actions
from representation import create_representations, transform_coordinates, reconcile_positions, transform_observation_3dim
from model import Actor
from points import update_points_map, filter_by_proximity, mark_duplicates_single


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

class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player

        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        self.unit_move_cost = jnp.array(env_cfg["unit_move_cost"]).repeat(16) / 8.0
        self.unit_sap_cost = (jnp.array(env_cfg["unit_sap_cost"]).repeat(16) - 30.0) / 20.0
        self.unit_sap_range = jnp.array(env_cfg["unit_sap_range"]).repeat(16) / 8.0
        self.unit_sensor_range = jnp.array(env_cfg["unit_sensor_range"]).repeat(16) / 8.0

        checkpoint_path = os.path.join(script_dir, 'checkpoint')
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.params = orbax_checkpointer.restore(checkpoint_path)

        self.rng = jax.random.PRNGKey(42)
        self.actor = Actor(n_actions=6)

        self.inference_fn = jax.jit(lambda x, x2: self.actor.apply(x, x2))

        self.discovered_relic_nodes = np.ones((1, 6, 2), dtype=jnp.int32) * -1
        self.prev_team_points = 0
        self.points_map = jnp.zeros((1, 24, 24), dtype=jnp.float32)
        self.search_map = jnp.zeros((1, 24, 24), dtype=jnp.float32)
        self.temporal_states = jnp.zeros((1, 6, 24, 24), dtype=jnp.float32)
        self.points_gained = 0
        self.prev_agent_positions = jnp.ones((1, 16, 2), dtype=jnp.int32) * -1
        self.points_gained_history = []
        self.positions_explored_history = []


    def act(self, step: int, obs, remainingOverageTime: int = 60):
        observation = DotDict(reshape_observation(obs))

        relic_mask = observation.relic_nodes != -1
        relic_nodes_count_before = (self.discovered_relic_nodes[..., 0] > 0).sum(axis=-1)
        self.discovered_relic_nodes[relic_mask] = observation.relic_nodes[relic_mask]
        relic_nodes_count_after = (self.discovered_relic_nodes[..., 0] > 0).sum(axis=-1)

        # reset points map if new relic node is found
        if (relic_nodes_count_after - relic_nodes_count_before)[0] > 0:
            self.points_map = jnp.maximum(self.points_map, 0)
            self.points_gained_history = []
            self.positions_explored_history = []

        if step == 101 or step == 202:
            self.points_gained_history = []
            self.positions_explored_history = []
            self.points_map = jnp.maximum(self.points_map, 0)

        team_points = obs['team_points'][self.team_id]
        self.points_gained = team_points - self.prev_team_points
        self.prev_team_points = team_points

        energy_agent_positions = jnp.where(
            observation.units.energy[0, self.team_id, :, None].repeat(2, axis=-1) > -1,
            observation.units.position[:, self.team_id, ...],
            -1
        )

        representations = create_representations(
            obs=observation,
            temporal_states=self.temporal_states,
            discovered_relic_nodes=self.discovered_relic_nodes,
            max_steps_in_match=100,
            prev_agent_positions=energy_agent_positions,
            points_map=self.points_map,
            search_map=self.search_map,
            points_gained=jnp.array([self.points_gained]),
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
        )
        
        (
            _,
            temporal_states,
            agent_observations,
            episode_info,
            points_map,
            search_map,
            agent_positions,
            agent_energies,
            _,
        ) = representations

        self.points_map = points_map
        self.search_map = search_map
        self.temporal_states = temporal_states

        agent_observations = jnp.squeeze(agent_observations, axis=0)
        agent_episode_info = episode_info.repeat(16, axis=0)
        agent_positions = agent_positions.reshape(-1, 2)

        # if step == 123 and self.team_id == 0:
        #     jnp.save('agent_1_team_0', agent_observations[5])
        #     jnp.save('team_0_points_map2', self.points_map)
        #     a = True

        # if step == 123 and self.team_id == 1:
        #     jnp.save('agent_1_team_1', agent_observations[5])
        #     jnp.save('team_1_points_map2', self.points_map)
        #     a = True

        logits = self.inference_fn(
            { "params": self.params },
            {
                "observations": agent_observations,
                "positions": agent_positions,
                "match_steps": agent_episode_info[:, 0],
                "matches": agent_episode_info[:, 1],
                "team_points": agent_episode_info[:, 2],
                "opponent_points": agent_episode_info[:, 3],
                "unit_move_cost": self.unit_move_cost,
                "unit_sap_cost": self.unit_sap_cost,
                "unit_sap_range": self.unit_sap_range,
                "unit_sensor_range": self.unit_sensor_range,
                "energies": agent_energies,
            }
        )

        self.rng, action_rng = jax.random.split(self.rng)

        actions = get_actions(
            rng=action_rng,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
            logits=logits,
            observations=observation,
            sap_ranges=jnp.array([self.env_cfg["unit_sap_range"]]),
            relic_nodes=self.discovered_relic_nodes,
        )

        transformed_targets = transform_coordinates(actions[..., 1:], 17, 17)
        transformed_p1_actions = jnp.zeros_like(actions)
        transformed_p1_actions = transformed_p1_actions.at[..., 0].set(vectorized_transform_actions(actions[:, :, 0]))
        transformed_p1_actions = transformed_p1_actions.at[..., 1].set(transformed_targets[..., 0])
        transformed_p1_actions = transformed_p1_actions.at[..., 2].set(transformed_targets[..., 1])

        actions = actions if self.team_id == 0 else transformed_p1_actions

        actions = actions.at[..., 1:].set(actions[..., 1:] - 8)


        updated_points_map = jnp.squeeze(self.points_map, axis=0)
        for i, history in enumerate(self.positions_explored_history):
            updated_points_map = update_points_map(
                updated_points_map,
                history,
                self.points_gained_history[i]
            )
            transformed_history = transform_coordinates(history)
            transformed_history = jnp.where(
                transformed_history == 24,
                -1,
                transformed_history
            )
            updated_points_map = update_points_map(
                updated_points_map,
                transformed_history,
                self.points_gained_history[i]
            )


        self.points_map = jnp.expand_dims(updated_points_map, axis=0)

        if self.points_gained > 0:
            self.points_gained_history.append(self.points_gained)
            agent_positions = jnp.where(
                observation.units.energy[0, self.team_id, :, None].repeat(2, axis=-1) > -1,
                agent_positions,
                -1
            )

            # Update points map
            proximity_positions = filter_by_proximity(
                agent_positions,
                jnp.squeeze(reconcile_positions(self.discovered_relic_nodes)),
            )

            self.positions_explored_history.append(
                mark_duplicates_single(proximity_positions)
            )

        
        return jnp.squeeze(actions, axis=0)
