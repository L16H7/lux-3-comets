import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from collections import OrderedDict

from agent import get_actions, vectorized_transform_actions, transform_coordinates
from constants import Constants
from opponent import get_actions as get_opponent_actions
from representation import get_env_info
from rnn import ScannedRNN


def evaluate(
    rng,
    meta_keys,
    meta_env_params,
    actor_train_state,
    opponent_state,
    n_envs,
    n_agents,
    v_reset,
    v_step,
):
    N_TOTAL_AGENTS = n_envs * n_agents

    p0_discovered_relic_nodes = jnp.ones((n_envs, 6, 2)) * -1
    p1_discovered_relic_nodes = jnp.ones((n_envs, 6, 2)) * -1

    unit_move_cost, unit_sap_cost, unit_sap_range, unit_sensor_range = get_env_info(meta_env_params)

    def _env_step(runner_state, _):
        (
            rng,
            actor_train_state,
            (p0_representations, p1_representations),
            observations,
            states,
            (p0_discovered_relic_nodes, p1_discovered_relic_nodes),
            p0_prev_actor_hstates,
            p1_prev_actor_hstates,
        ) = runner_state

        (
            p0_states,
            p0_observations,
            p0_episode_info,
            p0_points_map,
            p0_team_positions,
            p0_agent_ids,
            p0_units_mask,
        ) = p0_representations

        p0_agent_episode_info = p0_episode_info.repeat(n_agents, axis=0)
        p0_agent_states = jnp.expand_dims(p0_states, axis=0).repeat(16, axis=1)
        p0_agent_observations = p0_observations.reshape(1, -1, 10, 17, 17)
        p0_agent_positions = jnp.reshape(p0_team_positions, (1, N_TOTAL_AGENTS, 2))

        p0_logits, p0_new_actor_hstates = actor_train_state.apply_fn(
            actor_train_state.params,
            p0_prev_actor_hstates,
            {
                "states": p0_agent_states,
                "observations": p0_agent_observations,
                "positions": p0_agent_positions,
                "match_steps": jnp.expand_dims(p0_agent_episode_info[:, 0], axis=[0, -1]),
                "matches": jnp.expand_dims(p0_agent_episode_info[:, 1], axis=[0, -1]),
                "team_points": jnp.expand_dims(p0_agent_episode_info[:, 2], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(p0_agent_episode_info[:, 3], axis=[0, -1]),
                "unit_move_cost": unit_move_cost,
                "unit_sap_cost": unit_sap_cost,
                "unit_sap_range": unit_sap_range,
                "unit_sensor_range": unit_sensor_range,
                "agent_ids": jnp.expand_dims(p0_agent_ids, axis=0),
            }
        )

        rng, p0_action_rng, p1_action_rng = jax.random.split(rng, num=3)
        p0_actions, _, _ = get_actions(
            rng=p0_action_rng,
            team_idx=0,
            opponent_idx=1,
            logits=p0_logits,
            observations=observations['player_0'],
            sap_ranges=meta_env_params.unit_sap_range,
        )

        (
            p1_states,
            p1_observations,
            p1_episode_info,
            p1_points_map,
            p1_team_positions,
            p1_agent_ids,
            p1_units_mask,
        ) = p1_representations

        p1_agent_episode_info = p1_episode_info.repeat(n_agents, axis=0)

        p1_agent_states = jnp.expand_dims(p1_states, axis=0).repeat(16, axis=1)
        p1_agent_observations = p1_observations.reshape(1, -1, 10, 17, 17)
        p1_agent_positions = jnp.reshape(p1_team_positions, (1, N_TOTAL_AGENTS, 2))

        p1_logits, p1_new_actor_hstates = opponent_state.apply_fn(
            opponent_state.params,
            p1_prev_actor_hstates,
            {
                "states": p1_agent_states,
                "observations": p1_agent_observations,
                "positions": p1_agent_positions,
                "match_steps": jnp.expand_dims(p1_agent_episode_info[:, 0], axis=[0, -1]),
                "matches": jnp.expand_dims(p1_agent_episode_info[:, 1], axis=[0, -1]),
                "team_points": jnp.expand_dims(p1_agent_episode_info[:, 2], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(p1_agent_episode_info[:, 3], axis=[0, -1]),
                "unit_move_cost": unit_move_cost,
                "unit_sap_cost": unit_sap_cost,
                "unit_sap_range": unit_sap_range,
                "unit_sensor_range": unit_sensor_range,
                "agent_ids": jnp.expand_dims(p1_agent_ids, axis=0),
            }
        )

        p1_actions, _, _ = get_actions(
            rng=p1_action_rng,
            team_idx=1,
            opponent_idx=0,
            logits=p1_logits,
            observations=observations['player_1'],
            sap_ranges=meta_env_params.unit_sap_range,
        )

        transformed_targets = transform_coordinates(p1_actions[..., 1:], 17, 17)

        transformed_p1_actions = jnp.zeros_like(p1_actions)
        transformed_p1_actions = transformed_p1_actions.at[..., 0].set(vectorized_transform_actions(p1_actions[:, :, 0]))
        transformed_p1_actions = transformed_p1_actions.at[..., 1].set(transformed_targets[..., 0])
        transformed_p1_actions = transformed_p1_actions.at[..., 2].set(transformed_targets[..., 1])

        p0_relic_mask = observations['player_0'].relic_nodes != -1
        p0_new_discovered_relic_nodes = jnp.where(
            p0_relic_mask, 
            observations['player_0'].relic_nodes, 
            p0_discovered_relic_nodes
        )

        p1_relic_mask = observations['player_1'].relic_nodes != -1
        p1_new_discovered_relic_nodes = jnp.where(
            p1_relic_mask, 
            observations['player_1'].relic_nodes, 
            p1_discovered_relic_nodes
        )

        p0_next_representations, p1_next_representations, next_observations, next_states, rewards, _, _, info = v_step(
            states,
            OrderedDict({
                "player_0": p0_actions.at[:, :, 1:].set(p0_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
                "player_1": transformed_p1_actions.at[:, :, 1:].set(transformed_p1_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
            }),
            p0_new_discovered_relic_nodes,
            p1_new_discovered_relic_nodes,
            p0_team_positions,
            p1_team_positions,
            p0_points_map,
            p1_points_map,
            meta_keys,
            meta_env_params,
        )

        runner_state = (
            rng,
            actor_train_state,
            (p0_next_representations, p1_next_representations),
            next_observations,
            next_states,
            (p0_new_discovered_relic_nodes, p1_new_discovered_relic_nodes),
            p0_new_actor_hstates,
            p1_new_actor_hstates,
        )
    
        return runner_state, info

    p0_representations, p1_representations, observations, states = v_reset(meta_keys, meta_env_params)

    p0_actor_init_hstates = ScannedRNN.initialize_carry(n_envs * n_agents, 128)

    p1_actor_init_hstates = ScannedRNN.initialize_carry(n_envs * n_agents, 128)

    runner_state = (
        rng,
        actor_train_state,
        (p0_representations, p1_representations),
        observations,
        states,
        (p0_discovered_relic_nodes, p1_discovered_relic_nodes),
        p0_actor_init_hstates,
        p1_actor_init_hstates,
    )

    runner_state, info = jax.lax.scan(_env_step, runner_state, None, 505)
    last_match_steps = jtu.tree_map(lambda x: jnp.take(x, jnp.array([99, 200, 301, 402, 503]), axis=0), info)

    info_ = {
        "p0_energy_depletions": last_match_steps["p0_energy_depletions"],
        "p1_energy_depletions": last_match_steps["p1_energy_depletions"],
        "p0_points_mean": last_match_steps["p0_points_mean"],
        "p1_points_mean": last_match_steps["p1_points_mean"],
        "p0_points_std": last_match_steps["p0_points_std"],
        "p1_points_std": last_match_steps["p1_points_std"],
    }
    info2_ = {
        "eval/p0_wins": info["p0_wins"][-1],
        "eval/p1_wins": info["p1_wins"][-1],
        "eval/p0_sap_destroyed_units": info["p0_sap_units_destroyed"].sum(),
        "eval/p1_sap_destroyed_units": info["p1_sap_units_destroyed"].sum(),
        "eval/p0_collision_destroyed_units": info["p0_collision_units_destroyed"].sum(),
        "eval/p1_collision_destroyed_units": info["p1_collision_units_destroyed"].sum(),
        "eval/p0_net_energy_of_sap_loss": info["p0_net_energy_of_sap_loss"].sum(),
        "eval/p1_net_energy_of_sap_loss": info["p1_net_energy_of_sap_loss"].sum(),
    }
    info_dict = {f"eval/{key}_ep{i+1}": value for key, array in info_.items() for i, value in enumerate(array)}

    return {
        **info_dict,
        **info2_,
    }
