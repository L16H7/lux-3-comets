import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from collections import OrderedDict

from agent import get_actions, vectorized_transform_actions, transform_coordinates
from constants import Constants
from opponent import get_actions as get_opponent_actions
from representation import get_env_info, teacher_get_env_info


def evaluate(
    rng,
    meta_keys,
    meta_env_params,
    actor_train_state,
    opponent_state,
    label,
    n_envs,
    n_agents,
    v_reset,
    v_step,
):
    env_info = get_env_info(meta_env_params)
    teacher_env_info = teacher_get_env_info(meta_env_params)
    p1_env_info = env_info if label != Constants.TEACHER_LABEL else teacher_env_info

    def _env_step(runner_state, _):
        (
            rng,
            actor_train_state,
            (p0_representations, p1_representations),
            observations,
            states,
        ) = runner_state

        (
            p0_states,
            p0_temporal_states,
            p0_agent_observations,
            _,
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
            updated_nebula_info
        ) = p0_representations

        p0_agent_episode_info = p0_episode_info.repeat(n_agents, axis=0)
        p0_agent_states = p0_states.repeat(16, axis=0)
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
            rngs={ "dropout": rng }
        )

        rng, p0_action_rng, p1_action_rng = jax.random.split(rng, num=3)
        p0_actions, _, _ = get_actions(
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
            _
        ) = p1_representations

        p1_agent_episode_info = p1_episode_info.repeat(n_agents, axis=0)
        p1_agent_states = p1_states.repeat(16, axis=0)
        p1_agent_observations = p1_agent_observations.reshape(-1, 19, 47, 47)
        p1_agent_positions = p1_agent_positions.reshape(-1, 2)

        p1_logits = opponent_state.apply_fn(
            opponent_state.params,
            {
                "observations": p1_agent_observations if label != Constants.TEACHER_LABEL else teacher_p1_agent_observations,
                "positions": p1_agent_positions,
                "match_steps": p1_agent_episode_info[:, 0],
                "matches": p1_agent_episode_info[:, 1],
                "team_points": p1_agent_episode_info[:, 2],
                "opponent_points": p1_agent_episode_info[:, 3],
                "unit_move_cost": p1_env_info[:, 0],
                "unit_sap_cost": p1_env_info[:, 1],
                "unit_sap_range": p1_env_info[:, 2],
                "unit_sensor_range": p1_env_info[:, 3],
                "energies": p1_energies if label != Constants.TEACHER_LABEL else p1_energies_gained,
                "energies_gained": p1_energies_gained,
                "points_gained_history": p1_agent_episode_info[:, 4:],
            },
            rngs={ "dropout": rng }
        )

        p1_actions, _, _ = get_actions(
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

        p0_sapped_units_mask = p0_actions[..., 0] == 5
        p1_sapped_units_mask = p1_actions[..., 0] == 5

        p0_next_representations, p1_next_representations, next_observations, next_states, rewards, _, _, info = v_step(
            states,
            OrderedDict({
                "player_0": p0_actions.at[:, :, 1:].set(p0_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
                "player_1": transformed_p1_actions.at[:, :, 1:].set(transformed_p1_actions[:, :, 1:] - Constants.MAX_SAP_RANGE),
            }),
            p0_temporal_states,
            p1_temporal_states,
            p0_new_discovered_relic_nodes,
            p1_new_discovered_relic_nodes,
            p0_agent_positions.reshape(n_envs, -1, 2),
            p1_agent_positions.reshape(n_envs, -1, 2),
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

        runner_state = (
            rng,
            actor_train_state,
            (p0_next_representations, p1_next_representations),
            next_observations,
            next_states,
        )
        # POINTS MAP
        ground_truth = runner_state[4].relic_nodes_map_weights.transpose(0, 2, 1)  # shape: (n_envs, 24, 24)
        max_relic_nodes = runner_state[4].relic_nodes_mask.sum(axis=-1) // 2
        ground_truth = (ground_truth > 0) & (ground_truth <= max_relic_nodes[:, None, None])

        prediction = runner_state[2][0][5]  # shape: (n_envs, 24, 24)


        # Calculate true positives
        true_positives = jnp.sum((ground_truth > 0) & (prediction == 1), axis=(1, 2))
        # Calculate false positives
        false_positives = jnp.sum((ground_truth == 0) & (prediction == 1), axis=(1, 2))

        false_negative = jnp.sum((ground_truth > 0) & (prediction == -1), axis=(1, 2))

        # Calculate total number of positive labels in ground truth
        total_positives = jnp.sum(ground_truth > 0, axis=(1, 2))

        # Calculate true positive percentage
        true_positive_percentage = (true_positives / total_positives) * 100

        # Calculate false positive percentage
        false_positive_percentage = (false_positives / total_positives) * 100

        false_negative_percentage = (false_negative / total_positives) * 100

        info = {
            **info,
            "points_map_coverage_mean": jnp.mean(true_positive_percentage),
            "points_map_coverage_std": jnp.std(true_positive_percentage),
            "points_map_false_flags_mean": jnp.mean(false_positive_percentage),
            "points_map_false_flags_std": jnp.std(false_positive_percentage),
            "points_map_false_negative_mean": jnp.mean(false_negative_percentage),
            "points_map_false_negative_std": jnp.std(false_negative_percentage),
        }
        return runner_state, info

    p0_representations, p1_representations, observations, states = v_reset(meta_keys, meta_env_params)

    runner_state = (
        rng,
        actor_train_state,
        (p0_representations, p1_representations),
        observations,
        states,
    )

    runner_state, info = jax.lax.scan(_env_step, runner_state, None, 505)

    nebula_energy_reduction_calculation_success_rate = (abs(runner_state[2][0][-1][:, 0]) == meta_env_params.nebula_tile_energy_reduction).sum() / n_envs

    last_match_steps = jtu.tree_map(lambda x: jnp.take(x, jnp.array([99, 200, 301, 402, 502]), axis=0), info)

    info_ = {
        f"eval_{label}_stats/p0_energy_depletions": last_match_steps["p0_energy_depletions"],
        f"eval_{label}_stats/p1_energy_depletions": last_match_steps["p1_energy_depletions"],
        f"eval_{label}/p0_points_mean": last_match_steps["p0_points_mean"],
        f"eval_{label}/p1_points_mean": last_match_steps["p1_points_mean"],
        f"eval_{label}/p0_points_std": last_match_steps["p0_points_std"],
        f"eval_{label}/p1_points_std": last_match_steps["p1_points_std"],
        f"eval_{label}_debug/points_map_coverage_mean": last_match_steps["points_map_coverage_mean"],
        f"eval_{label}_debug/points_map_coverage_std": last_match_steps["points_map_coverage_std"],
        f"eval_{label}_debug/points_map_false_flags_mean": last_match_steps["points_map_false_flags_mean"],
        f"eval_{label}_debug/points_map_false_flags_std": last_match_steps["points_map_false_flags_std"],
        f"eval_{label}_debug/points_map_false_negative_mean": last_match_steps["points_map_false_negative_mean"],
        f"eval_{label}_debug/points_map_false_negative_std": last_match_steps["points_map_false_negative_std"],
    }
    info2_ = {
        f"eval_{label}/p0_match_wins": info["p0_match_wins"][-1],
        f"eval_{label}/p1_match_wins": info["p1_match_wins"][-1],
        f"eval_{label}/p0_episode_wins": info["p0_episode_wins"][-1],
        f"eval_{label}/p1_episode_wins": info["p1_episode_wins"][-1],
        f"eval_{label}_stats/p0_sap_destroyed_units": info["p0_sap_units_destroyed"].sum(),
        f"eval_{label}_stats/p1_sap_destroyed_units": info["p1_sap_units_destroyed"].sum(),
        f"eval_{label}_stats/p0_collision_destroyed_units": info["p0_collision_units_destroyed"].sum(),
        f"eval_{label}_stats/p1_collision_destroyed_units": info["p1_collision_units_destroyed"].sum(),
        f"eval_{label}_stats/p0_net_energy_of_sap_loss": info["p0_net_energy_of_sap_loss"].sum(),
        f"eval_{label}_stats/p1_net_energy_of_sap_loss": info["p1_net_energy_of_sap_loss"].sum(),
        f"eval_{label}_debug/nebula_energy_reduction_calculation_success_rate": nebula_energy_reduction_calculation_success_rate,
    }

    info_dict = {f"{key}_ep{i+1}": value for key, array in info_.items() for i, value in enumerate(array)}

    return {
        **info_dict,
        **info2_,
    }
