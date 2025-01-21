import jax
import jax.numpy as jnp

from agent import get_actions


def get_opponent_actions(
    rng,
    observations,
    unit_sap_range,
    self_state,
    opponent1_state,
    opponent2_state,
    agent_states,
    agent_observations,
    agent_positions,
    agent_episode_info,
    env_info
):
    n_envs = agent_states.shape[0]
    frac = n_envs // 8
    self_envs = frac * 3
    opponent1_envs = frac * 3
    opponent2_envs = frac * 2

    assert n_envs == (self_envs + opponent1_envs + opponent2_envs)
    self_logits = self_state.apply_fn(
        self_state.params,
        {
            "states": agent_states[: self_envs, ...],
            "observations": agent_observations[: self_envs, ...],
            "positions": agent_positions[: self_envs, ...],
            "match_steps": agent_episode_info[: self_envs, 0],
            "matches": agent_episode_info[: self_envs, 1],
            "team_points": agent_episode_info[: self_envs, 2],
            "opponent_points": agent_episode_info[: self_envs, 3],
            "unit_move_cost": env_info[: self_envs, 0],
            "unit_sap_cost": env_info[: self_envs, 1],
            "unit_sap_range": env_info[: self_envs, 2],
            "unit_sensor_range": env_info[: self_envs, 3],
        }
    )

    opponent1_envs_ = self_envs + opponent1_envs
    opponent1_logits = opponent1_state.apply_fn(
        opponent1_state.params,
        {
            "states": agent_states[self_envs: opponent1_envs_, ...],
            "observations": agent_observations[self_envs: opponent1_envs_, ...],
            "positions": agent_positions[self_envs: opponent1_envs_, ...],
            "match_steps": agent_episode_info[self_envs: opponent1_envs_, 0],
            "matches": agent_episode_info[self_envs: opponent1_envs_, 1],
            "team_points": agent_episode_info[self_envs: opponent1_envs_, 2],
            "opponent_points": agent_episode_info[self_envs: opponent1_envs_, 3],
            "unit_move_cost": env_info[self_envs: opponent1_envs_, 0],
            "unit_sap_cost": env_info[self_envs: opponent1_envs_, 1],
            "unit_sap_range": env_info[self_envs: opponent1_envs_, 2],
            "unit_sensor_range": env_info[self_envs: opponent1_envs_, 3],
        }
    )

    opponent2_logits = opponent2_state.apply_fn(
        opponent2_state.params,
        {
            "states": agent_states[opponent1_envs_:, ...],
            "observations": agent_observations[opponent1_envs_:, ...],
            "positions": agent_positions[opponent1_envs_:, ...],
            "match_steps": agent_episode_info[opponent1_envs_:, 0],
            "matches": agent_episode_info[opponent1_envs_:, 1],
            "team_points": agent_episode_info[opponent1_envs_:, 2],
            "opponent_points": agent_episode_info[opponent1_envs_:, 3],
            "unit_move_cost": env_info[opponent1_envs_:, 0],
            "unit_sap_cost": env_info[opponent1_envs_:, 1],
            "unit_sap_range": env_info[opponent1_envs_:, 2],
            "unit_sensor_range": env_info[opponent1_envs_:, 3],
        }
    )
    
    opponent_logits = [
        jnp.concatenate([self_logits[0], opponent1_logits[0], opponent2_logits[0]], axis=0),
        jnp.concatenate([self_logits[1], opponent1_logits[1], opponent2_logits[1]], axis=0),
        jnp.concatenate([self_logits[2], opponent1_logits[2], opponent2_logits[2]], axis=0),
    ]

    oppnent_actions, _, _ =get_actions(
        rng=rng,
        team_idx=1,
        opponent_idx=0,
        logits=opponent_logits,
        observations=observations,
        sap_ranges=unit_sap_range,
    ) 
    return oppnent_actions
