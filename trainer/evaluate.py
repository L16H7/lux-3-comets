import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from collections import OrderedDict

from agent import get_actions
from rnn import ScannedRNN
from utils import calculate_sapping_stats


def analyse_stats(
    observations,
    next_observations,
    p0_actions,
    p1_actions,
):
    p0_sap_info = calculate_sapping_stats(
        actions=p0_actions.reshape(-1, 3) * observations['player_0'].units_mask[:, 0, :].reshape(-1, 1),
        units_position=observations['player_0'].units.position[:, 0, ...].reshape(-1, 2),
        units_mask=observations['player_0'].units_mask[:, 0, :].reshape(-1),
        opponent_units_position=next_observations['player_1'].units.position[:, 1, ...].reshape(-1, 2),
        opponent_units_mask=next_observations['player_1'].units_mask[:, 1, :].reshape(-1),
    )

    p1_sap_info = calculate_sapping_stats(
        actions=p1_actions.reshape(-1, 3) * observations['player_1'].units_mask[:, 1, :].reshape(-1, 1),
        units_position=observations['player_1'].units.position[:, 1, ...].reshape(-1, 2),
        units_mask=observations['player_1'].units_mask[:, 1, :].reshape(-1),
        opponent_units_position=next_observations['player_0'].units.position[:, 0, ...].reshape(-1, 2),
        opponent_units_mask=next_observations['player_0'].units_mask[:, 0, :].reshape(-1),
    )
 
    return {
        "p0_total_direct_hits": p0_sap_info["total_direct_hits"],
        "p0_total_indirect_hits": p0_sap_info["total_indirect_hits"],
        "p0_total_sapped_actions": p0_sap_info["total_sapped_actions"],
        "p1_total_direct_hits": p1_sap_info["total_direct_hits"],
        "p1_total_indirect_hits": p1_sap_info["total_indirect_hits"],
        "p1_total_sapped_actions": p1_sap_info["total_sapped_actions"],
    }


def evaluate(
    rng,
    actor_train_state,
    n_envs,
    n_agents,
    sample_params,
    v_reset,
    v_step,
):
    N_TOTAL_AGENTS = n_envs * n_agents

    rng, meta_key_rng, meta_env_params_rng, _ = jax.random.split(rng, num=4)
    meta_keys = jax.random.split(meta_key_rng, n_envs)
    meta_env_params = jax.vmap(sample_params)(
        jax.random.split(meta_env_params_rng, n_envs)
    )

    p0_discovered_relic_nodes = jnp.ones((n_envs, 6, 2)) * -1
    p1_discovered_relic_nodes = jnp.ones((n_envs, 6, 2)) * -1

    def _env_step(runner_state, _):
        (
            rng,
            actor_train_state,
            (p0_representations, p1_representations),
            (p0_prev_actions, p1_prev_actions),
            (p0_prev_rewards, p1_prev_rewards),
            observations,
            states,
            (p0_discovered_relic_nodes, p1_discovered_relic_nodes),
            p0_prev_actor_hstates,
            p1_prev_actor_hstates,
        ) = runner_state

        (
            _,
            p0_agent_observations,
            p0_episode_info,
            p0_team_positions,
            p0_opponent_positions,
            p0_relic_nodes_positions,
            _,
        ) = p0_representations

        p0_agent_episode_info = p0_episode_info.repeat(n_agents, axis=0)

        p0_agent_observations = jnp.reshape(p0_agent_observations, (1, N_TOTAL_AGENTS, 9, 9, 9))
        p0_agent_positions = jnp.reshape(p0_team_positions, (1, N_TOTAL_AGENTS, 2))
        p0_relic_nodes_positions = jnp.expand_dims(p0_relic_nodes_positions.repeat(n_agents, axis=0), axis=0)
        p0_team_positions = jnp.expand_dims(jnp.repeat(p0_team_positions, n_agents, axis=0), axis=0)
        p0_opponent_positions = jnp.expand_dims(jnp.repeat(p0_opponent_positions, n_agents, axis=0), axis=0)

        p0_logits, p0_actor_hstates = actor_train_state.apply_fn(
            actor_train_state.params,
            p0_prev_actor_hstates,
            {
                "observations": p0_agent_observations,
                "prev_actions": p0_prev_actions,
                "positions": p0_agent_positions,
                "relic_nodes_positions": p0_relic_nodes_positions,
                "team_positions": p0_team_positions,
                "opponent_positions": p0_opponent_positions,
                "prev_rewards": p0_prev_rewards,
                "teams": jnp.expand_dims(p0_agent_episode_info[:, 0].astype(jnp.int32), axis=0),
                "match_phases": jnp.expand_dims(p0_agent_episode_info[:, 1].astype(jnp.int32), axis=0),
                "matches": jnp.expand_dims(p0_agent_episode_info[:, 2].astype(jnp.int32), axis=0),
                "team_points": jnp.expand_dims(p0_agent_episode_info[:, 3], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(p0_agent_episode_info[:, 4], axis=[0, -1]),
            }
        )

        rng, p0_action_rng, p1_action_rng = jax.random.split(rng, num=3)
        p0_actions, _ = get_actions(
            rng=p0_action_rng,
            team_idx=0,
            opponent_idx=1,
            logits=p0_logits,
            observations=observations['player_0'],
            sap_ranges=meta_env_params.unit_sap_range,
        )
        p0_actions = jnp.squeeze(jnp.stack(p0_actions), axis=1)
        p0_actions = p0_actions.T.reshape(n_envs, n_agents, -1)

        (
            _,
            p1_agent_observations,
            p1_episode_info,
            p1_team_positions,
            p1_opponent_positions,
            p1_relic_nodes_positions,
            _,
        ) = p1_representations

        p1_agent_episode_info = p1_episode_info.repeat(n_agents, axis=0)

        p1_agent_observations = jnp.reshape(p1_agent_observations, (1, N_TOTAL_AGENTS, 9, 9, 9))
        p1_agent_positions = jnp.reshape(p1_team_positions, (1, N_TOTAL_AGENTS, 2))
        p1_relic_nodes_positions = jnp.expand_dims(p1_relic_nodes_positions.repeat(n_agents, axis=0), axis=0)
        p1_team_positions = jnp.expand_dims(jnp.repeat(p1_team_positions, n_agents, axis=0), axis=0)
        p1_opponent_positions = jnp.expand_dims(jnp.repeat(p1_opponent_positions, n_agents, axis=0), axis=0)

        p1_logits, p1_actor_hstates = actor_train_state.apply_fn(
            actor_train_state.params,
            p1_prev_actor_hstates,
            {
                "observations": p1_agent_observations,
                "prev_actions": p1_prev_actions,
                "positions": p1_agent_positions,
                "relic_nodes_positions": p1_relic_nodes_positions,
                "team_positions": p1_team_positions,
                "opponent_positions": p1_opponent_positions,
                "prev_rewards": p1_prev_rewards,
                "teams": jnp.expand_dims(p1_agent_episode_info[:, 0].astype(jnp.int32), axis=0),
                "match_phases": jnp.expand_dims(p1_agent_episode_info[:, 1].astype(jnp.int32), axis=0),
                "matches": jnp.expand_dims(p1_agent_episode_info[:, 2].astype(jnp.int32), axis=0),
                "team_points": jnp.expand_dims(p1_agent_episode_info[:, 3], axis=[0, -1]),
                "opponent_points": jnp.expand_dims(p1_agent_episode_info[:, 4], axis=[0, -1]),
            }
        )

        p1_actions, _ = get_actions(
            rng=p1_action_rng,
            team_idx=1,
            opponent_idx=0,
            logits=p1_logits,
            observations=observations['player_1'],
            sap_ranges=meta_env_params.unit_sap_range,
        )
        p1_actions = jnp.squeeze(jnp.stack(p1_actions), axis=1)
        p1_actions = p1_actions.T.reshape(n_envs, n_agents, -1)

        p0_actions = p0_actions.at[:, :, 1:].set(p0_actions[:, :, 1:] - 4)
        p1_actions = p1_actions.at[:, :, 1:].set(p1_actions[:, :, 1:] - 4)

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
                "player_0": p0_actions,
                "player_1": p1_actions,
            }),
            p0_new_discovered_relic_nodes,
            p1_new_discovered_relic_nodes,
            meta_keys,
            meta_env_params,
        )

        # stats = analyse_stats(
        #     observations=observations,
        #     next_observations=next_observations,
        #     p0_actions=p0_actions,
        #     p1_actions=p1_actions,
        # )

        p0_rewards = rewards[:, 0, :].reshape(1, -1, 1)
        p1_rewards = rewards[:, 1, :].reshape(1, -1, 1)

        runner_state = (
            rng,
            actor_train_state,
            (p0_next_representations, p1_next_representations),
            (p0_actions[:, :, 0].reshape(1, -1), p1_actions[:, :, 0].reshape(1, -1)),
            (p0_rewards, p1_rewards),
            next_observations,
            next_states,
            (p0_new_discovered_relic_nodes, p1_new_discovered_relic_nodes),
            p0_actor_hstates,
            p1_actor_hstates,
        )
    
        return runner_state, info

    p0_representations, p1_representations, observations, states = v_reset(meta_keys, meta_env_params)

    p0_actor_init_hstates = ScannedRNN.initialize_carry(n_envs * n_agents, 256)

    p1_actor_init_hstates = ScannedRNN.initialize_carry(n_envs * n_agents, 256)

    p0_prev_actions = jnp.zeros((1, n_envs * n_agents), dtype=jnp.int32)
    p1_prev_actions = jnp.zeros((1, n_envs * n_agents), dtype=jnp.int32)

    p0_prev_rewards = jnp.zeros((1, n_envs * n_agents, 1))
    p1_prev_rewards = jnp.zeros((1, n_envs * n_agents, 1))

    runner_state = (
        rng,
        actor_train_state,
        (p0_representations, p1_representations),
        (p0_prev_actions, p1_prev_actions),
        (p0_prev_rewards, p1_prev_rewards),
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
    }
    info_dict = {f"eval/{key}_ep{i+1}": value for key, array in info_.items() for i, value in enumerate(array)}

    return {
        **info_dict,
        **info2_,
    }
