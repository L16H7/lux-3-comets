# import sys
# # Get the parent directory of the current file
# parent_dir = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(parent_dir)

import jax
import optax
import orbax.checkpoint
import os
import wandb
import jax.numpy as jnp
import jax.tree_util as jtu

from collections import OrderedDict
from dataclasses import asdict
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from functools import partial
from time import time
from typing import NamedTuple

from agent import get_actions, vectorized_transform_actions
from config import Config
from constants import Constants
from evaluate import evaluate
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from make_states import make_states, make_actor_state
from opponent import get_actions as get_opponent_actions
from ppo import Transition, calculate_gae, ppo_update
from representation import create_agent_representations, transform_coordinates, get_env_info, combined_states_info, reconcile_positions, teacher_get_env_info
from teacher.model import make_actor_state as make_teacher_actor_state


class RunnerState(NamedTuple):
    rng: jnp.ndarray
    actor_train_state: TrainState
    critic_train_state: TrainState
    p0_representations: jnp.ndarray
    p1_representations: jnp.ndarray
    observations: jnp.ndarray
    states: jnp.ndarray


def make_train(config: Config):
    env = LuxAIS3Env(auto_reset=False)

    def reset_fn(key, params):
        return env.reset(key, params)

    def v_reset(meta_keys, meta_env_params):
        observations, states = jax.vmap(reset_fn)(meta_keys, meta_env_params)
        n_envs = observations['player_0'].relic_nodes.shape[0]
        p0_representations, p1_representations = create_agent_representations(
            observations=observations,
            p0_temporal_states=jnp.zeros((n_envs, 14, 24, 24)),
            p1_temporal_states=jnp.zeros((n_envs, 14, 24, 24)),
            p0_discovered_relic_nodes=observations['player_0'].relic_nodes,
            p1_discovered_relic_nodes=observations['player_1'].relic_nodes,
            p0_points_map=jnp.zeros((n_envs, config.map_width, config.map_height), dtype=jnp.float32),
            p1_points_map=jnp.zeros((n_envs, config.map_width, config.map_height), dtype=jnp.float32),
            p0_search_map=jnp.zeros((n_envs, config.map_width, config.map_height), dtype=jnp.int32),
            p1_search_map=jnp.zeros((n_envs, config.map_width, config.map_height), dtype=jnp.int32),
            p0_points_gained=jnp.zeros((n_envs)),
            p1_points_gained=jnp.zeros((n_envs)),
            p0_prev_agent_energies=jnp.zeros_like(states.units.energy[:, 0, ...]),
            p1_prev_agent_energies=jnp.zeros_like(states.units.energy[:, 1, ...]),
            p0_points_history_positions=(jnp.ones((101, n_envs, 16, 2), dtype=jnp.int32) * -1),
            p1_points_history_positions=(jnp.ones((101, n_envs, 16, 2), dtype=jnp.int32) * -1),
            p0_points_history=jnp.zeros((101, n_envs), dtype=jnp.int32),
            p1_points_history=jnp.zeros((101, n_envs), dtype=jnp.int32),
            unit_move_cost=meta_env_params.unit_move_cost,
            sensor_range=meta_env_params.unit_sensor_range,
            nebula_info=jnp.zeros((n_envs, 2)), # [nebula_energy_deduction, is_updated]
            p0_sapped_units_mask=jnp.zeros((n_envs, 16)),
            p1_sapped_units_mask=jnp.zeros((n_envs, 16)),
        )

        return p0_representations, p1_representations, observations, states

    def step_fn(state, action, key, params):
        return env.step(key, state, action, params)
   
    def v_step(
        states,
        actions,
        p0_temporal_states,
        p1_temporal_states,
        p0_discovered_relic_nodes,
        p1_discovered_relic_nodes,
        p0_agent_positions,
        p1_agent_positions,
        p0_points_map,
        p1_points_map,
        p0_search_map,
        p1_search_map,
        p0_points_history_positions,
        p1_points_history_positions,
        p0_points_history,
        p1_points_history,
        updated_nebula_info,
        p0_sapped_units_mask,
        p1_sapped_units_mask,
        meta_keys,
        meta_env_params,
    ):
        next_observations, next_states, rewards, terminated, truncated, envinfo = jax.vmap(step_fn)(
            states,
            actions,
            meta_keys,
            meta_env_params,
        )

        p0_relic_nodes_before = (p0_discovered_relic_nodes[..., 0] > -1).sum(axis=-1)
        p0_relic_mask = next_observations['player_0'].relic_nodes != -1
        p0_new_discovered_relic_nodes = jnp.where(
            p0_relic_mask, 
            next_observations['player_0'].relic_nodes, 
            p0_discovered_relic_nodes
        )
        p0_new_discovered_relic_nodes = reconcile_positions(p0_new_discovered_relic_nodes)
        p0_relic_nodes_after = (p0_new_discovered_relic_nodes[..., 0] > -1).sum(axis=-1)
        p0_relic_nodes_diff = p0_relic_nodes_after - p0_relic_nodes_before
        p0_relic_diff_mask = p0_relic_nodes_diff > 0

        p0_points_map = jnp.where(
            jnp.broadcast_to(
                p0_relic_diff_mask[:, None, None],
                p0_points_map.shape
            ), 
            jnp.maximum(p0_points_map, 0),
            p0_points_map,
        )

        p0_points_history_positions = jnp.where(
            jnp.broadcast_to(
                p0_relic_diff_mask[None, :, None, None],
                p0_points_history_positions.shape
            ), 
            (jnp.ones_like(p0_points_history_positions) * -1),
            p0_points_history_positions
        )

        p0_points_history = jnp.where(
            jnp.broadcast_to(
                p0_relic_diff_mask[None, :],
                p0_points_history.shape
            ),
            jnp.zeros_like(p0_points_history),
            p0_points_history,
        )

        p1_relic_nodes_before = (p1_discovered_relic_nodes[..., 0] > -1).sum(axis=-1)
        p1_relic_mask = next_observations['player_1'].relic_nodes != -1
        p1_new_discovered_relic_nodes = jnp.where(
            p1_relic_mask, 
            next_observations['player_1'].relic_nodes, 
            p1_discovered_relic_nodes
        )

        p1_new_discovered_relic_nodes = reconcile_positions(p1_new_discovered_relic_nodes)
        p1_relic_nodes_after = (p1_new_discovered_relic_nodes[..., 0] > -1).sum(axis=-1)
        p1_relic_nodes_diff = p1_relic_nodes_after - p1_relic_nodes_before
        p1_relic_diff_mask = p1_relic_nodes_diff > 0

        p1_points_map = jnp.where(
            jnp.broadcast_to(
                p1_relic_diff_mask[:, None, None],
                p1_points_map.shape
            ), 
            jnp.maximum(p1_points_map, 0),
            p1_points_map,
        )

        p1_points_history_positions = jnp.where(
            jnp.broadcast_to(
                p1_relic_diff_mask[None, :, None, None],
                p1_points_history_positions.shape
            ), 
            (jnp.ones_like(p1_points_history_positions) * -1),
            p1_points_history_positions
        )

        p1_points_history = jnp.where(
            jnp.broadcast_to(
                p1_relic_diff_mask[None, :],
                p1_points_history.shape
            ),
            jnp.zeros_like(p1_points_history),
            p1_points_history,
        )

        team_points = next_observations["player_0"].team_points
        team_wins = next_observations["player_0"].team_wins
        info = {
            "p0_points_mean": team_points[:, 0].mean(),
            "p1_points_mean": team_points[:, 1].mean(),   
            "p0_points_std": team_points[:, 0].std(),   
            "p1_points_std": team_points[:, 1].std(),   
            "p0_match_wins": team_wins[:, 0].mean(),
            "p1_match_wins": team_wins[:, 1].mean(),
            "p0_episode_wins": (team_wins[:, 0] > team_wins[:, 1]).mean(),
            "p1_episode_wins": (team_wins[:, 1] > team_wins[:, 0]).mean(),
            "p0_energy_depletions": jnp.sum(next_observations["player_0"].units.energy[:, 0, :] == 0) / len(meta_keys),
            "p1_energy_depletions": jnp.sum(next_observations["player_1"].units.energy[:, 1, :] == 0) / len(meta_keys),
            "p0_sap_units_destroyed": envinfo["sap_destroyed_units"][:, 0].mean(),
            "p1_sap_units_destroyed": envinfo["sap_destroyed_units"][:, 1].mean(),
            "p0_collision_units_destroyed": envinfo["collision_destroyed_units"][:, 0].mean(),
            "p1_collision_units_destroyed": envinfo["collision_destroyed_units"][:, 1].mean(),
            "p0_net_energy_of_sap_loss": envinfo["net_energy_of_sap_loss"][:, 0].mean(),
            "p1_net_energy_of_sap_loss": envinfo["net_energy_of_sap_loss"][:, 1].mean(),
        }

        p0_next_representations, p1_next_representations = create_agent_representations(
            observations=next_observations,
            p0_temporal_states=p0_temporal_states,
            p1_temporal_states=p1_temporal_states,
            p0_discovered_relic_nodes=p0_new_discovered_relic_nodes,
            p1_discovered_relic_nodes=p1_new_discovered_relic_nodes,
            p0_points_map=p0_points_map,
            p1_points_map=p1_points_map,
            p0_search_map=p0_search_map,
            p1_search_map=p1_search_map,
            p0_points_gained=envinfo["points_gained"][..., 0],
            p1_points_gained=envinfo["points_gained"][..., 1],
            p0_prev_agent_energies=states.units.energy[:, 0, ...],
            p1_prev_agent_energies=states.units.energy[:, 1, ...],
            p0_points_history_positions=p0_points_history_positions,
            p1_points_history_positions=p1_points_history_positions,
            p0_points_history=p0_points_history,
            p1_points_history=p1_points_history,
            unit_move_cost=meta_env_params.unit_move_cost,
            sensor_range=meta_env_params.unit_sensor_range,
            nebula_info=updated_nebula_info,
            p0_sapped_units_mask=p0_sapped_units_mask,
            p1_sapped_units_mask=p1_sapped_units_mask,
        )

        return p0_next_representations, p1_next_representations, next_observations, next_states, rewards, terminated, truncated, info
        

    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        actor_train_state: TrainState,
        critic_train_state: TrainState,
        eval_opponent_state: TrainState,
        teacher_train_state: TrainState,
    ):
        def _meta_step(meta_state, _):
            def sample_params(rng_key):
                randomized_game_params = dict()
                for k, v in env_params_ranges.items():
                    rng_key, subkey = jax.random.split(rng_key)
                    if isinstance(v[0], int):
                        randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.int16))
                    else:
                        randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.float32))
                params = EnvParams(**randomized_game_params)
                return params

            rng, actor_train_state, critic_train_state = meta_state

            rng, meta_key_rng, meta_env_params_rng, eval_rng, _ = jax.random.split(rng, num=5)
            meta_keys = jax.random.split(meta_key_rng, config.n_envs)
            meta_env_params = jax.vmap(sample_params)(
                jax.random.split(meta_env_params_rng, config.n_envs)
            )

            env_info = get_env_info(meta_env_params)
            teacher_env_info = teacher_get_env_info(meta_env_params)

            def _update_step(runner_state: RunnerState, _):
                def _env_step(runner_state: RunnerState, _):
                    (
                        rng,
                        actor_train_state,
                        critic_train_state,
                        p0_representations,
                        p1_representations,
                        observations,
                        states,
                    ) = runner_state

                    (
                        p0_states,
                        p0_temporal_states,
                        p0_agent_observations,
                        teacher_p0_agent_observations,
                        p0_episode_info,
                        p0_points_map,
                        p0_search_map,
                        p0_agent_positions,
                        p0_energies,
                        p0_energies_gained,
                        p0_units_mask,
                        p0_discovered_relic_nodes,
                        p0_points_history_positions,
                        p0_points_history,
                        updated_nebula_info,
                    ) = p0_representations
                    (
                        p1_states,
                        p1_temporal_states,
                        p1_agent_observations,
                        teacher_p1_agent_observations,
                        p1_episode_info,
                        p1_points_map,
                        p1_search_map,
                        p1_agent_positions,
                        p1_energies,
                        p1_energies_gained,
                        p1_units_mask,
                        p1_discovered_relic_nodes,
                        p1_points_history_positions,
                        p1_points_history,
                        _,
                    ) = p1_representations

                    p0_agent_episode_info = p0_episode_info.repeat(config.n_agents, axis=0)
                    p0_agent_states = p0_states.repeat(config.n_agents, axis=0) # N_TOTAL_AGENTS, 10, 24, 24
                    p0_agent_observations = p0_agent_observations.reshape(-1, 19, 47, 47) 
                    p0_agent_positions = p0_agent_positions.reshape(-1, 2)

                    p0_logits = actor_train_state.apply_fn(
                        actor_train_state.params,
                        {
                            "states": p0_agent_states,
                            "observations": p0_agent_observations,
                            "positions": p0_agent_positions,
                            "match_steps": p0_agent_episode_info[:, 0],
                            "matches": p0_agent_episode_info[:, 1],
                            "team_points": p0_agent_episode_info[:, 2],
                            "opponent_points": p0_agent_episode_info[:, 3],
                            "unit_move_cost": env_info[:, 0],
                            "unit_sap_cost": env_info[:, 1],
                            "unit_sap_range": env_info[:, 2],
                            "unit_sensor_range": env_info[:, 3],
                            "energies": p0_energies,
                            "energies_gained": p0_energies_gained,
                            "points_gained_history": p0_agent_episode_info[:, 4:],
                        },
                        rngs={"dropout": rng},
                    )

                    rng, p0_action_rng, p1_action_rng = jax.random.split(rng, num=3)
                    p0_actions, p0_log_probs, p0_logits_mask = get_actions(
                        rng=p0_action_rng,
                        team_idx=0,
                        opponent_idx=1,
                        logits=p0_logits,
                        observations=observations['player_0'],
                        sap_ranges=meta_env_params.unit_sap_range,
                        sap_costs=meta_env_params.unit_sap_cost,
                        relic_nodes=p0_discovered_relic_nodes,
                        points_map=p0_points_map,
                    )

                    p0_combined_states = combined_states_info(
                        p0_states,
                        p1_states,
                        p0_temporal_states,
                        p1_temporal_states,
                    )

                    p0_values = critic_train_state.apply_fn(
                        critic_train_state.params,
                        {
                            "states": p0_combined_states,
                            "match_steps": p0_episode_info[:, 0],
                            "matches": p0_episode_info[:, 1],
                            "team_points": p0_episode_info[:, 2],
                            "opponent_points": p0_episode_info[:, 3],
                            "points_gained_history": p0_episode_info[:, 4:],
                        },
                        rngs={ "dropout": rng },
                    )

                    p1_agent_episode_info = p1_episode_info.repeat(config.n_agents, axis=0)
                    p1_agent_states = p1_states.repeat(16, axis=0) # N_TOTAL_AGENTS, 10, 24, 24
                    p1_agent_observations = p1_agent_observations.reshape(-1, 19, 47, 47)
                    p1_agent_positions = p1_agent_positions.reshape(-1, 2)

                    p1_logits = actor_train_state.apply_fn(
                        actor_train_state.params,
                        {
                            "states": p1_agent_states,
                            "observations": p1_agent_observations,
                            "positions": p1_agent_positions,
                            "match_steps": p1_agent_episode_info[:, 0],
                            "matches": p1_agent_episode_info[:, 1],
                            "team_points": p1_agent_episode_info[:, 2],
                            "opponent_points": p1_agent_episode_info[:, 3],
                            "unit_move_cost": env_info[:, 0],
                            "unit_sap_cost": env_info[:, 1],
                            "unit_sap_range": env_info[:, 2],
                            "unit_sensor_range": env_info[:, 3],
                            "energies": p1_energies,
                            "energies_gained": p1_energies_gained,
                            "points_gained_history": p1_agent_episode_info[:, 4:],

                        },
                        rngs={ "dropout": rng },
                    )

                    p1_actions, p1_log_probs, p1_logits_mask = get_actions(
                        rng=p1_action_rng,
                        team_idx=1,
                        opponent_idx=0,
                        logits=p1_logits,
                        observations=observations['player_1'],
                        sap_ranges=meta_env_params.unit_sap_range,
                        sap_costs=meta_env_params.unit_sap_cost,
                        relic_nodes=p1_discovered_relic_nodes,
                        points_map=p1_points_map,
                    )

                    p1_combined_states = combined_states_info(
                        p1_states,
                        p0_states,
                        p1_temporal_states,
                        p0_temporal_states,
                    )
                    p1_values = critic_train_state.apply_fn(
                        critic_train_state.params,
                        {
                            "states": p1_combined_states,
                            "match_steps": p1_episode_info[:, 0],
                            "matches": p1_episode_info[:, 1],
                            "team_points": p1_episode_info[:, 2],
                            "opponent_points": p1_episode_info[:, 3],
                            "points_gained_history": p1_episode_info[:, 4:],
                        },
                        rngs={ "dropout": rng },
                    )

                    transformed_targets = transform_coordinates(p1_actions[..., 1:], 17, 17)

                    transformed_p1_actions = jnp.zeros_like(p1_actions)
                    transformed_p1_actions = transformed_p1_actions.at[..., 0].set(vectorized_transform_actions(p1_actions[:, :, 0]))
                    transformed_p1_actions = transformed_p1_actions.at[..., 1].set(transformed_targets[..., 0])
                    transformed_p1_actions = transformed_p1_actions.at[..., 2].set(transformed_targets[..., 1])

                    p0_sapped_units_mask = p0_actions[..., 0] == 5
                    p1_sapped_units_mask = p1_actions[..., 0] == 5
                    p0_next_representations, p1_next_representations, next_observations, next_states, rewards, terminated, truncated, _ = v_step(
                        states,
                        OrderedDict({
                            "player_0": p0_actions.at[:, :, 1:].set(p0_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
                            "player_1": transformed_p1_actions.at[:, :, 1:].set(transformed_p1_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
                        }),
                        p0_temporal_states,
                        p1_temporal_states,
                        p0_discovered_relic_nodes,
                        p1_discovered_relic_nodes,
                        p0_agent_positions.reshape(config.n_envs, -1, 2),
                        p1_agent_positions.reshape(config.n_envs, -1, 2),
                        p0_points_map,
                        p1_points_map,
                        p0_search_map,
                        p1_search_map,
                        p0_points_history_positions,
                        p1_points_history_positions,
                        p0_points_history,
                        p1_points_history,
                        updated_nebula_info,
                        p0_sapped_units_mask,
                        p1_sapped_units_mask,
                        meta_keys,
                        meta_env_params,
                    )

                    p0_rewards = rewards[:, 0, :].reshape(-1)
                    p1_rewards = rewards[:, 1, :].reshape(-1)

                    p0_values = p0_values.reshape(-1).repeat(16)
                    p1_values = p1_values.reshape(-1).repeat(16)

                    # COMMENT FOR FIXED OPPONENT
                    transition = Transition(
                        agent_states=jnp.concat([p0_agent_states, p1_agent_states], axis=0),
                        observations=jnp.concat([p0_agent_observations, p1_agent_observations], axis=0),
                        teacher_observations=jnp.concat([teacher_p0_agent_observations, teacher_p1_agent_observations], axis=0),
                        states=jnp.concat([p0_combined_states, p1_combined_states], axis=0),
                        episode_info=jnp.concat([p0_episode_info, p1_episode_info], axis=0),
                        agent_episode_info=jnp.concat([p0_agent_episode_info, p1_agent_episode_info], axis=0),
                        actions=jnp.concat([p0_actions.reshape(-1, 3), p1_actions.reshape(-1, 3)], axis=0),
                        log_probs=jnp.concat([jax.lax.stop_gradient(p0_log_probs), jax.lax.stop_gradient(p1_log_probs)], axis=0),
                        values=jnp.concat([jax.lax.stop_gradient(p0_values), jax.lax.stop_gradient(p1_values)], axis=0),
                        agent_positions=jnp.concat([p0_agent_positions, p1_agent_positions], axis=0),
                        energies=jnp.concat([p0_energies, p1_energies], axis=0),
                        energies_gained=jnp.concat([p0_energies_gained, p1_energies_gained], axis=0),
                        rewards=jnp.concat([p0_rewards, p1_rewards], axis=0),
                        dones=jnp.logical_or(terminated["player_0"], truncated["player_0"]).repeat(2 * config.n_agents),
                        units_mask=jnp.concat([p0_units_mask.reshape(-1), p1_units_mask.reshape(-1)], axis=0),
                        logits1_mask=jnp.concat([p0_logits_mask[0], p1_logits_mask[0]], axis=0),
                        logits2_mask=jnp.concat([p0_logits_mask[1], p1_logits_mask[1]], axis=0),
                        logits3_mask=jnp.concat([p0_logits_mask[2], p1_logits_mask[2]], axis=0),
                        env_information=env_info.repeat(2, axis=0),
                        teacher_env_information=teacher_env_info.repeat(2, axis=0),
                    )

                    runner_state = RunnerState(
                        rng,
                        actor_train_state,
                        critic_train_state,
                        p0_next_representations,
                        p1_next_representations,
                        next_observations,
                        next_states,
                    )

                    return runner_state, transition

                runner_state, transitions = jax.lax.scan(
                    _env_step,
                    runner_state,
                    None,
                    config.n_actor_steps
                )

                (
                    rng,
                    actor_train_state,
                    critic_train_state,
                    p0_representations,
                    p1_representations,
                    observations,
                    states,
                ) = runner_state

                (
                    p0_states,
                    p0_temporal_states,
                    _, _,
                    p0_episode_info,
                    _, _, _, _, _, _, _, _, _, _
                ) = p0_representations

                (
                    p1_states,
                    p1_temporal_states,
                    _, _,
                    p1_episode_info,
                    _, _, _, _, _, _, _, _, _, _
                ) = p1_representations

                p0_combined_states = combined_states_info(
                    p0_states,
                    p1_states,
                    p0_temporal_states,
                    p1_temporal_states,
                )

                p0_last_values = critic_train_state.apply_fn(
                    critic_train_state.params,
                    {
                        "states": p0_combined_states,
                        "match_steps": p0_episode_info[:, 0],
                        "matches": p0_episode_info[:, 1],
                        "team_points": p0_episode_info[:, 2],
                        "opponent_points": p0_episode_info[:, 3],
                        "points_gained_history": p0_episode_info[:, 4:],
                    },
                    rngs={ "dropout": rng },
                )

                p1_combined_states = combined_states_info(
                    p1_states,
                    p0_states,
                    p1_temporal_states,
                    p0_temporal_states,
                )

                p1_last_values = critic_train_state.apply_fn(
                    critic_train_state.params,
                    {
                        "states": p1_combined_states,
                        "match_steps": p1_episode_info[:, 0],
                        "matches": p1_episode_info[:, 1],
                        "team_points": p1_episode_info[:, 2],
                        "opponent_points": p1_episode_info[:, 3],
                        "points_gained_history": p1_episode_info[:, 4:],
                    },
                    rngs={ "dropout": rng },
                )

                advantages, targets = calculate_gae(
                    transitions,
                    jnp.concat([p0_last_values, p1_last_values], axis=0).repeat(config.n_agents, axis=1).reshape(-1),
                    config.gamma,
                    config.gae_lambda
                )
                advantages = jax.lax.stop_gradient(advantages)
                targets = jax.lax.stop_gradient(targets)

                def _update_epoch(update_state, _):
                    def _update_minibatch(train_state, batch_info):
                        rng, actor_train_state, critic_train_state = train_state
                        transitions, advantages, targets = batch_info
                        updated_actor_train_state, updated_critic_train_state, minibatch_info = ppo_update(
                            rng=rng,
                            actor_train_state=actor_train_state,
                            critic_train_state=critic_train_state,
                            teacher_train_state=teacher_train_state,
                            transitions=transitions,
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.policy_clip,
                            vf_coef=config.value_coeff,
                            ent_coef=config.entropy_coeff,
                        )
                        return (rng, updated_actor_train_state, updated_critic_train_state), minibatch_info

                    (
                        rng,
                        (actor_train_states, critic_train_states),
                        transitions,
                        advantages,
                        targets
                    ) = update_state

                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.n_minibatches)

                    batch = (
                        transitions,
                        advantages,
                        targets,
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.reshape(
                            x,
                            [config.n_minibatches, -1]
                            + list(x.shape[2:]),
                        ),
                        batch,
                    )

                    shuffled_minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), minibatches
                    )

                    (rng, updated_actor_train_state, updated_critic_train_state), loss_info = jax.lax.scan(
                        _update_minibatch,
                        (rng, actor_train_states, critic_train_states),
                        shuffled_minibatches
                    ) 

                    updated_state = (
                        rng,
                        (updated_actor_train_state, updated_critic_train_state),
                        transitions,
                        advantages,
                        targets
                    )

                    return updated_state, loss_info

                update_state = (
                    rng,
                    (actor_train_state, critic_train_state),
                    transitions,
                    advantages,
                    targets
                )

                updated_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.n_epochs)

                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                update_step_info = {
                    "actor_loss": loss_info["actor_loss"],
                    "value_loss": loss_info["value_loss"],
                    "entropy": loss_info["entropy"],
                    "loss": loss_info["loss"],
                    "approx_kl": loss_info["approx_kl"],
                    "clip_frac": loss_info["clip_frac"],
                    "explained_var": loss_info["explained_var"],
                    "adv_mean": loss_info["adv_mean"],
                    "adv_std": loss_info["adv_std"],
                    "value_mean": loss_info["value_mean"],
                    "value_std": loss_info["value_std"],
                    "actor_dense6_mean": loss_info["actor_dense6_mean"],
                    "actor_dense6_std": loss_info["actor_dense6_std"],
                    "reward_mean": transitions.rewards.mean(),
                    "reward_std": transitions.rewards.std(),
                }

                _, (updated_actor_train_states, updated_critic_train_states), _, _, _  = updated_state

                updated_runner_state = RunnerState(
                    rng,
                    updated_actor_train_states,
                    updated_critic_train_states,
                    p0_representations,
                    p1_representations,
                    observations,
                    states,
                )
                return updated_runner_state, update_step_info

            p0_representations, p1_representations, observations, states = v_reset(meta_keys, meta_env_params)

            runner_state = RunnerState(
                rng=rng,
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                p0_representations=p0_representations,
                p1_representations=p1_representations,
                observations=observations,
                states=states,
            )

            updated_runner_state, update_step_info = jax.lax.scan(_update_step, runner_state, None, config.n_update_steps)

            rng, eval_meta_key_rng, eval_meta_env_params_rng, _ = jax.random.split(rng, num=4)
            eval_meta_keys = jax.random.split(eval_meta_key_rng, config.n_eval_envs)
            eval_meta_env_params = jax.vmap(sample_params)(
                jax.random.split(eval_meta_env_params_rng, config.n_eval_envs)
            )

            eval_info = evaluate(
                eval_rng,
                eval_meta_keys,
                eval_meta_env_params,
                updated_runner_state.actor_train_state,
                eval_opponent_state,
                'self',
                config.n_eval_envs,
                config.n_agents,
                v_reset,
                v_step
            )
            eval_info = jax.lax.pmean(eval_info, axis_name="devices")

            teacher_eval_info = evaluate(
                eval_rng,
                eval_meta_keys,
                eval_meta_env_params,
                updated_runner_state.actor_train_state,
                teacher_train_state,
                'self2',
                config.n_eval_envs,
                config.n_agents,
                v_reset,
                v_step
            )
            teacher_eval_info = jax.lax.pmean(teacher_eval_info, axis_name="devices")

            meta_step_info = {
                "update_step_info": update_step_info,
                "eval_info": {
                    **eval_info,
                    **teacher_eval_info
                },
            }

            meta_state = (rng, updated_runner_state.actor_train_state, updated_runner_state.critic_train_state)
            return meta_state, meta_step_info


        meta_state = (rng, actor_train_state, critic_train_state)
        meta_state, train_info = jax.lax.scan(_meta_step, meta_state, None, config.n_meta_steps)

        (_, updated_actor_train_state, updated_critic_train_state) = meta_state

        return {
            "actor_state": updated_actor_train_state,
            "critic_state": updated_critic_train_state,
            "train_info": train_info,
        }

    return train

def train(config: Config):
    run = wandb.init(
        project=config.wandb_project,
        config={**asdict(config)}
    )

    rng = jax.random.key(config.train_seed)
    actor_train_state, critic_train_state = make_states(config=config)
    train_device_rngs = jax.random.split(rng, num=jax.local_device_count())
    actor_train_state = replicate(actor_train_state, jax.local_devices())
    critic_train_state = replicate(critic_train_state, jax.local_devices())

    eval_opponent_state = make_actor_state('/root/26100_actor')
    eval_opponent_state = replicate(eval_opponent_state, jax.local_devices())

    teacher_state = make_teacher_actor_state('/root/19000_actor')
    teacher_state = replicate(teacher_state, jax.local_devices())

    print("Compiling...")
    t = time()
    train_fn = make_train(
        config=config,
    )
    train_fn = train_fn.lower(
        train_device_rngs,
        actor_train_state,
        critic_train_state,
        eval_opponent_state,
        teacher_state,
    ).compile()

    elapsed_time = time() - t
    print(f"Done in {elapsed_time:.2f}s.") 

    print("Training...")

    loop = 0
    total_transitions = 0
    meta_step = 0
    update_step = 0
    while True:
        rng, train_rng, _, _ = jax.random.split(rng, num=4)
        train_device_rngs = jax.random.split(train_rng, num=jax.local_device_count())
        loop += 1
        t = time()
        train_summary = jax.block_until_ready(
            train_fn(
                train_device_rngs,
                actor_train_state,
                critic_train_state,
                eval_opponent_state,
                teacher_state,
            )
        )
        elapsed_time = time() - t
        print(f"Done in {elapsed_time:.4f}s.")
        print("Logginig...")
        train_info = train_summary["train_info"]
        update_step_info = unreplicate(train_info["update_step_info"])
        eval_info = unreplicate(train_info["eval_info"])
        actor_train_state = train_summary["actor_state"]
        critic_train_state = train_summary["critic_state"]

        for i in range(config.n_meta_steps):
            meta_step += 1
            meta_info = jtu.tree_map(lambda x: x[i], eval_info)
            meta_info["meta_steps"] = meta_step
            wandb.log(meta_info)
            for j in range(config.n_update_steps):
                update_step += config.n_minibatches * config.n_epochs * jax.local_device_count()
                total_transitions += config.n_envs_per_device * jax.local_device_count() * config.n_actor_steps
                info = jtu.tree_map(lambda x: x[i, j], update_step_info)
                info["transitions"] = total_transitions
                info["update_steps"] = update_step
                wandb.log(info)

        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        orbax_checkpointer.save(
            os.path.abspath(f"{config.checkpoint_path}/{meta_step}_actor"),
            unreplicate(train_summary["actor_state"]).params
        )

        orbax_checkpointer.save(
            os.path.abspath(f"{config.checkpoint_path}/{meta_step}_critic"),
            unreplicate(train_summary["critic_state"]).params
        )


if __name__ == "__main__":
    config = Config(
        n_meta_steps=20,
        n_actor_steps=14,
        n_update_steps=36,
        n_envs=64,
        n_envs_per_device=64,
        n_eval_envs=64,
        n_minibatches=32,
        n_epochs=1,
        actor_learning_rate=8e-5,
        critic_learning_rate=1e-4,
        wandb_project="Optimus",
        train_seed=20250401,
        entropy_coeff=0.01,
        gae_lambda=0.95,
        gamma=0.99,
    )
    train(config=config)
