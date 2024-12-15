from collections import OrderedDict
# import sys
import os

# # Get the parent directory of the current file
# parent_dir = os.path.dirname(os.path.dirname(__file__))
# sys.path.append(parent_dir)

import jax
import orbax.checkpoint
import wandb
import jax.numpy as jnp
import jax.tree_util as jtu

from dataclasses import asdict
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from functools import partial
from time import time
from typing import NamedTuple

from agent import get_actions
from config import Config
from evaluate import evaluate
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams, env_params_ranges
from make_states import make_states
from ppo import Transition, calculate_gae, ppo_update
from representation import create_representations
from rnn import ScannedRNN


class RunnerState(NamedTuple):
    rng: jnp.ndarray
    actor_train_state: TrainState
    critic_train_state: TrainState
    p0_representations: jnp.ndarray
    p1_representations: jnp.ndarray
    p0_prev_actions: jnp.ndarray
    p1_prev_actions: jnp.ndarray
    p0_prev_rewards: jnp.ndarray
    p1_prev_rewards: jnp.ndarray
    observations: jnp.ndarray
    states: jnp.ndarray
    p0_discovered_relic_nodes: jnp.ndarray
    p1_discovered_relic_nodes: jnp.ndarray
    p0_actor_hstate: jnp.ndarray
    p0_critic_hstate: jnp.ndarray
    p1_actor_hstate: jnp.ndarray
    p1_critic_hstate: jnp.ndarray


def create_agent_representations(observations, p0_discovered_relic_nodes, p1_discovered_relic_nodes):
    p0_observations = observations["player_0"]
    p0_representations = create_representations(
        obs=p0_observations,
        discovered_relic_nodes=p0_discovered_relic_nodes,
        map_width=config.map_width,
        map_height=config.map_height,
        team_idx=0,
        enemy_idx=1,
    )

    p1_observations = observations["player_1"]
    p1_representations = create_representations(
        obs=p1_observations,
        discovered_relic_nodes=p1_discovered_relic_nodes,
        map_width=config.map_width,
        map_height=config.map_height,
        team_idx=1,
        enemy_idx=0,
    )
    return p0_representations, p1_representations


def make_train(config: Config):
    env = LuxAIS3Env(auto_reset=False)

    def reset_fn(key, params):
        return env.reset(key, params)

    def v_reset(meta_keys, meta_env_params):
        observations, states = jax.vmap(reset_fn)(meta_keys, meta_env_params)
        p0_representations, p1_representations = create_agent_representations(
            observations=observations,
            p0_discovered_relic_nodes=observations['player_0'].relic_nodes,
            p1_discovered_relic_nodes=observations['player_1'].relic_nodes,
        )
        return p0_representations, p1_representations, observations, states

    def step_fn(state, action, key, params):
        return env.step(key, state, action, params)
   
    def v_step(
        states,
        actions,
        p0_discovered_relic_nodes,
        p1_discovered_relic_nodes,
        meta_keys,
        meta_env_params,
    ):
        observations, states, rewards, terminated, truncated, _ = jax.vmap(step_fn)(
            states,
            actions,
            meta_keys,
            meta_env_params,
        )

        team_points = observations["player_0"].team_points
        team_wins = observations["player_0"].team_wins
        info = {
            "p0_points_mean": team_points[:, 0].mean(),
            "p1_points_mean": team_points[:, 1].mean(),   
            "p0_points_std": team_points[:, 0].std(),   
            "p1_points_std": team_points[:, 1].std(),   
            "p0_wins": team_wins[:, 0].mean(),
            "p1_wins": team_wins[:, 1].mean(),
            "p0_energy_depletions": jnp.sum(observations["player_0"].units.energy[:, 0, :] == 0) / len(meta_keys),
            "p1_energy_depletions": jnp.sum(observations["player_1"].units.energy[:, 1, :] == 0) / len(meta_keys),
        }

        p0_representations, p1_representations = create_agent_representations(
            observations=observations,
            p0_discovered_relic_nodes=p0_discovered_relic_nodes,
            p1_discovered_relic_nodes=p1_discovered_relic_nodes,
        )
        return p0_representations, p1_representations, observations, states, rewards, terminated, truncated, info
        

    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        actor_train_state: TrainState,
        critic_train_state: TrainState,
    ):
        N_TOTAL_AGENTS = config.n_envs * config.n_agents

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

            p0_discovered_relic_nodes = jnp.ones((config.n_envs, 6, 2)) * -1
            p1_discovered_relic_nodes = jnp.ones((config.n_envs, 6, 2)) * -1

            def _update_step(runner_state: RunnerState, _):
                def _env_step(runner_state: RunnerState, _):
                    (
                        rng,
                        actor_train_state,
                        critic_train_state,
                        p0_representations,
                        p1_representations,
                        p0_prev_actions,
                        p1_prev_actions,
                        p0_prev_rewards,
                        p1_prev_rewards,
                        observations,
                        states,
                        p0_discovered_relic_nodes,
                        p1_discovered_relic_nodes,
                        p0_prev_actor_hstates,
                        p0_prev_critic_hstates,
                        p1_prev_actor_hstates,
                        p1_prev_critic_hstates,
                    ) = runner_state

                    (
                        p0_states,
                        p0_agent_observations,
                        p0_episode_info,
                        p0_team_positions,
                        p0_opponent_positions,
                        p0_relic_nodes_positions,
                        p0_units_mask,
                    ) = p0_representations

                    p0_agent_episode_info = p0_episode_info.repeat(config.n_agents, axis=0)

                    p0_agent_observations = jnp.reshape(p0_agent_observations, (1, N_TOTAL_AGENTS, 9, 9, 9))
                    p0_agent_positions = jnp.reshape(p0_team_positions, (1, N_TOTAL_AGENTS, 2))
                    p0_relic_nodes_positions = jnp.expand_dims(p0_relic_nodes_positions.repeat(config.n_agents, axis=0), axis=0)
                    p0_team_positions = jnp.expand_dims(jnp.repeat(p0_team_positions, config.n_agents, axis=0), axis=0)
                    p0_opponent_positions = jnp.expand_dims(jnp.repeat(p0_opponent_positions, config.n_agents, axis=0), axis=0)

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
                    p0_actions, p0_log_probs = get_actions(
                        rng=p0_action_rng,
                        team_idx=0,
                        opponent_idx=1,
                        logits=p0_logits,
                        observations=observations['player_0']
                    )
                    p0_actions = jnp.squeeze(jnp.stack(p0_actions), axis=1)
                    p0_actions = p0_actions.T.reshape(config.n_envs, config.n_agents, -1)

                    p0_values, p0_critic_hstates = critic_train_state.apply_fn(
                        critic_train_state.params,
                        p0_prev_critic_hstates,
                        {
                            "states": jnp.expand_dims(p0_states, axis=0),
                            "teams": jnp.expand_dims(p0_episode_info[:, 0].astype(jnp.int32), axis=0),
                            "match_phases": jnp.expand_dims(p0_episode_info[:, 1].astype(jnp.int32), axis=0),
                            "matches": jnp.expand_dims(p0_episode_info[:, 2].astype(jnp.int32), axis=0),
                            "team_points": jnp.expand_dims(p0_episode_info[:, 3], axis=[0, -1]),
                            "opponent_points": jnp.expand_dims(p0_episode_info[:, 4], axis=[0, -1]),
                        }
                    )


                    (
                        p1_states,
                        p1_agent_observations,
                        p1_episode_info,
                        p1_team_positions,
                        p1_opponent_positions,
                        p1_relic_nodes_positions,
                        p1_units_mask,
                    ) = p1_representations

                    p1_agent_episode_info = p1_episode_info.repeat(config.n_agents, axis=0)

                    p1_agent_observations = jnp.reshape(p1_agent_observations, (1, N_TOTAL_AGENTS, 9, 9, 9))
                    p1_agent_positions = jnp.reshape(p1_team_positions, (1, N_TOTAL_AGENTS, 2))
                    p1_relic_nodes_positions = jnp.expand_dims(p1_relic_nodes_positions.repeat(config.n_agents, axis=0), axis=0)
                    p1_team_positions = jnp.expand_dims(jnp.repeat(p1_team_positions, config.n_agents, axis=0), axis=0)
                    p1_opponent_positions = jnp.expand_dims(jnp.repeat(p1_opponent_positions, config.n_agents, axis=0), axis=0)

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

                    p1_actions, p1_log_probs = get_actions(
                        rng=p1_action_rng,
                        team_idx=1,
                        opponent_idx=0,
                        logits=p1_logits,
                        observations=observations['player_1']
                    )
                    p1_actions = jnp.squeeze(jnp.stack(p1_actions), axis=1)
                    p1_actions = p1_actions.T.reshape(config.n_envs, config.n_agents, -1)

                    p1_values, p1_critic_hstates = critic_train_state.apply_fn(
                        critic_train_state.params,
                        p1_prev_critic_hstates,
                        {
                            "states": jnp.expand_dims(p1_states, axis=0),
                            "teams": jnp.expand_dims(p1_episode_info[:, 0].astype(jnp.int32), axis=0),
                            "match_phases": jnp.expand_dims(p1_episode_info[:, 1].astype(jnp.int32), axis=0),
                            "matches": jnp.expand_dims(p1_episode_info[:, 2].astype(jnp.int32), axis=0),
                            "team_points": jnp.expand_dims(p1_episode_info[:, 3], axis=[0, -1]),
                            "opponent_points": jnp.expand_dims(p1_episode_info[:, 4], axis=[0, -1]),
                        }
                    )

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

                    p0_next_representations, p1_next_representations, next_observations, next_states, rewards, terminated, truncated, _ = v_step(
                        states,
                        OrderedDict({
                            "player_0": p0_actions.at[:, :, 1:].set(p0_actions[:, :, 1:] - 4),
                            "player_1": p1_actions.at[:, :, 1:].set(p1_actions[:, :, 1:] - 4),
                        }),
                        p0_new_discovered_relic_nodes,
                        p1_new_discovered_relic_nodes,
                        meta_keys,
                        meta_env_params,
                    )

                    p0_rewards = rewards[:, 0, :].reshape(1, -1, 1)
                    p1_rewards = rewards[:, 1, :].reshape(1, -1, 1)

                    transition = Transition(
                        observations=jnp.squeeze(jnp.concat([p0_agent_observations, p1_agent_observations], axis=1), axis=0),
                        states=jnp.concat([p0_states, p1_states], axis=0),
                        episode_info=jnp.concat([p0_episode_info, p1_episode_info], axis=0),
                        agent_episode_info=jnp.concat([p0_agent_episode_info, p1_agent_episode_info], axis=0),
                        actions=jnp.concat([p0_actions, p1_actions], axis=0),
                        prev_actions=jnp.squeeze(jnp.concat([p0_prev_actions, p1_prev_actions], axis=1), axis=0),
                        prev_rewards=jnp.squeeze(jnp.concat([p0_prev_rewards, p1_prev_rewards], axis=1), axis=[0, 2]),
                        log_probs=jnp.squeeze(jnp.concat([p0_log_probs, p1_log_probs], axis=1), axis=0),
                        values=jnp.squeeze(jnp.concat([p0_values, p1_values], axis=1), axis=[0, 2]),
                        agent_positions=jnp.squeeze(jnp.concat([p0_agent_positions, p1_agent_positions], axis=1), axis=0),
                        team_positions=jnp.squeeze(jnp.concat([p0_team_positions, p1_team_positions], axis=1), axis=0),
                        opponent_positions=jnp.squeeze(jnp.concat([p0_opponent_positions, p1_opponent_positions], axis=1), axis=0),
                        rewards=jnp.squeeze(jnp.concat([p0_rewards, p1_rewards], axis=1), axis=[0, 2]),
                        relic_nodes_positions=jnp.squeeze(jnp.concat([p0_relic_nodes_positions, p1_relic_nodes_positions], axis=1), axis=0),
                        dones=jnp.logical_or(terminated["player_0"], truncated["player_0"]).repeat(2 * config.n_agents),
                        units_mask=jnp.concat([p0_units_mask.reshape(-1), p1_units_mask.reshape(-1)], axis=0),
                    )

                    runner_state = RunnerState(
                        rng,
                        actor_train_state,
                        critic_train_state,
                        p0_next_representations,
                        p1_next_representations,
                        p0_actions[:, :, 0].reshape(1, -1),
                        p1_actions[:, :, 0].reshape(1, -1),
                        p0_rewards,
                        p1_rewards,
                        next_observations,
                        next_states,
                        p0_new_discovered_relic_nodes,
                        p1_new_discovered_relic_nodes,
                        p0_actor_hstates,
                        p0_critic_hstates,
                        p1_actor_hstates,
                        p1_critic_hstates,
                    )

                    return runner_state, transition

                # KEEPING hstates to be used in update step
                p0_actor_init_hstates = runner_state.p0_actor_hstate
                p0_critic_init_hstates = runner_state.p0_critic_hstate
                p1_actor_init_hstates = runner_state.p1_actor_hstate
                p1_critic_init_hstates = runner_state.p1_critic_hstate
                
                runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.n_actor_steps)

                (
                    rng,
                    actor_train_state,
                    critic_train_state,
                    p0_representations,
                    p1_representations,
                    p0_prev_actions,
                    p1_prev_actions,
                    p0_prev_rewards,
                    p1_prev_rewards,
                    observations,
                    states,
                    p0_discovered_relic_nodes,
                    p1_discovered_relic_nodes,
                    p0_prev_actor_hstates,
                    p0_prev_critic_hstates,
                    p1_prev_actor_hstates,
                    p1_prev_critic_hstates,
                ) = runner_state

                (
                    p0_states,
                    _,
                    p0_episode_info,
                    _,
                    _,
                    _,
                    _,
                ) = p0_representations

                (
                    p1_states,
                    _,
                    p1_episode_info,
                    _,
                    _,
                    _,
                    _,
                ) = p1_representations

                p0_last_values, _ = critic_train_state.apply_fn(
                    critic_train_state.params,
                    p0_prev_critic_hstates,
                    {
                        "states": jnp.expand_dims(p0_states, axis=0),
                        "teams": jnp.expand_dims(p0_episode_info[:, 0].astype(jnp.int32), axis=0),
                        "match_phases": jnp.expand_dims(p0_episode_info[:, 1].astype(jnp.int32), axis=0),
                        "matches": jnp.expand_dims(p0_episode_info[:, 2].astype(jnp.int32), axis=0),
                        "team_points": jnp.expand_dims(p0_episode_info[:, 3].astype(jnp.int32), axis=[0, -1]),
                        "opponent_points": jnp.expand_dims(p0_episode_info[:, 4].astype(jnp.int32), axis=[0, -1]),
                    }
                )

                p1_last_values, _ = critic_train_state.apply_fn(
                    critic_train_state.params,
                    p1_prev_critic_hstates,
                    {
                        "states": jnp.expand_dims(p1_states, axis=0),
                        "teams": jnp.expand_dims(p1_episode_info[:, 0].astype(jnp.int32), axis=0),
                        "match_phases": jnp.expand_dims(p1_episode_info[:, 1].astype(jnp.int32), axis=0),
                        "matches": jnp.expand_dims(p1_episode_info[:, 2].astype(jnp.int32), axis=0),
                        "team_points": jnp.expand_dims(p1_episode_info[:, 3].astype(jnp.int32), axis=[0, -1]),
                        "opponent_points": jnp.expand_dims(p1_episode_info[:, 4].astype(jnp.int32), axis=[0, -1]),
                    }
                )

                advantages, targets = calculate_gae(
                    transitions,
                    jnp.concat([p0_last_values, p1_last_values], axis=0).repeat(config.n_agents, axis=1).reshape(-1),
                    config.gamma,
                    config.gae_lambda
                )

                def _update_epoch(update_state, _):
                    def _update_minibatch(train_state, batch_info):
                        actor_train_state, critic_train_state = train_state
                        actor_hstates, critic_hstates, transitions, advantages, targets = batch_info
                        updated_actor_train_state, updated_critic_train_state, minibatch_info = ppo_update(
                            actor_train_state=actor_train_state,
                            critic_train_state=critic_train_state,
                            actor_hstates=actor_hstates,
                            critic_hstates=critic_hstates,
                            transitions=transitions,
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.policy_clip,
                            vf_coef=config.value_coeff,
                            ent_coef=config.entropy_coeff,
                        )
                        return (updated_actor_train_state, updated_critic_train_state), minibatch_info

                    (
                        rng,
                        (actor_train_states, critic_train_states),
                        (p0_actor_init_hstates, p0_critic_init_hstates),
                        (p1_actor_init_hstates, p1_critic_init_hstates),
                        transitions,
                        advantages,
                        targets
                    ) = update_state

                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.n_minibatches)

                    batch = (
                        jnp.expand_dims(jnp.concat([p0_actor_init_hstates, p1_actor_init_hstates], axis=0), axis=0),
                        jnp.expand_dims(jnp.concat([p0_critic_init_hstates, p1_critic_init_hstates], axis=0), axis=0),
                        transitions,
                        advantages,
                        targets,
                    )

                    minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.swapaxes(
                            jnp.reshape(
                                x,
                                [x.shape[0], config.n_minibatches, -1]
                                + list(x.shape[2:]),
                            ),
                            1,
                            0,
                        ),
                        batch,
                    )
                    shuffled_minibatches = jax.tree_util.tree_map(
                        lambda x: jnp.take(x, permutation, axis=0), minibatches
                    )

                    (updated_actor_train_state, updated_critic_train_state), loss_info = jax.lax.scan(
                        _update_minibatch,
                        (actor_train_states, critic_train_states),
                        shuffled_minibatches
                    ) 

                    updated_state = (
                        rng,
                        (updated_actor_train_state, updated_critic_train_state),
                        (p0_actor_init_hstates, p0_critic_init_hstates),
                        (p1_actor_init_hstates, p1_critic_init_hstates),
                        transitions,
                        advantages,
                        targets
                    )

                    return updated_state, loss_info

                update_state = (
                    rng,
                    (actor_train_state, critic_train_state),
                    (p0_actor_init_hstates, p0_critic_init_hstates),
                    (p1_actor_init_hstates, p1_critic_init_hstates),
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
                    "reward_mean": transitions.rewards.mean(),
                    "reward_std": transitions.rewards.std(),
                }

                _, (updated_actor_train_states, updated_critic_train_states), _, _, _, _, _ = updated_state

                updated_runner_state = RunnerState(
                    rng,
                    updated_actor_train_states,
                    updated_critic_train_states,
                    p0_representations,
                    p1_representations,
                    p0_prev_actions,
                    p1_prev_actions,
                    p0_prev_rewards,
                    p1_prev_rewards,
                    observations,
                    states,
                    p0_discovered_relic_nodes,
                    p1_discovered_relic_nodes,
                    p0_prev_actor_hstates,
                    p0_prev_critic_hstates,
                    p1_prev_actor_hstates,
                    p1_prev_critic_hstates,
                )
                return updated_runner_state, update_step_info

            p0_representations, p1_representations, observations, states = v_reset(meta_keys, meta_env_params)

            p0_actor_init_hstates = ScannedRNN.initialize_carry(config.n_envs * config.n_agents, 256)
            p0_critic_init_hstates = ScannedRNN.initialize_carry(config.n_envs, 512)

            p1_actor_init_hstates = ScannedRNN.initialize_carry(config.n_envs * config.n_agents, 256)
            p1_critic_init_hstates = ScannedRNN.initialize_carry(config.n_envs, 512)

            p0_prev_actions = jnp.zeros((1, config.n_envs * config.n_agents), dtype=jnp.int32)
            p1_prev_actions = jnp.zeros((1, config.n_envs * config.n_agents), dtype=jnp.int32)

            p0_prev_rewards = jnp.zeros((1, config.n_envs * config.n_agents, 1))
            p1_prev_rewards = jnp.zeros((1, config.n_envs * config.n_agents, 1))

            runner_state = RunnerState(
                rng=rng,
                actor_train_state=actor_train_state,
                critic_train_state=critic_train_state,
                p0_representations=p0_representations,
                p1_representations=p1_representations,
                p0_prev_actions=p0_prev_actions,
                p1_prev_actions=p1_prev_actions,
                p0_prev_rewards=p0_prev_rewards,
                p1_prev_rewards=p1_prev_rewards,
                observations=observations,
                states=states,
                p0_discovered_relic_nodes=p0_discovered_relic_nodes,
                p1_discovered_relic_nodes=p1_discovered_relic_nodes,
                p0_actor_hstate=p0_actor_init_hstates,
                p0_critic_hstate=p0_critic_init_hstates,
                p1_actor_hstate=p1_actor_init_hstates,
                p1_critic_hstate=p1_critic_init_hstates,
            )

            updated_runner_state, update_step_info = jax.lax.scan(_update_step, runner_state, None, config.n_update_steps)

            eval_info = evaluate(eval_rng, updated_runner_state.actor_train_state, config.n_eval_envs, config.n_agents, sample_params, v_reset, v_step)

            meta_step_info = {
                "update_step_info": update_step_info,
                "eval_info": eval_info,
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
    # run = wandb.init(
    #     project=config.wandb_project,
    #     config={**asdict(config)}
    # )
    rng = jax.random.key(config.train_seed)
    actor_train_state, critic_train_state = make_states(config=config)
    train_device_rngs = jax.random.split(rng, num=jax.local_device_count())
    actor_train_state = replicate(actor_train_state, jax.local_devices())
    critic_train_state = replicate(critic_train_state, jax.local_devices())
    print("Compiling...")
    t = time()
    train_fn = make_train(
        config=config,
    )
    train_fn = train_fn.lower(train_device_rngs, actor_train_state, critic_train_state).compile()
    elapsed_time = time() - t
    print(f"Done in {elapsed_time:.2f}s.") 

    print("Training...")

    loop = 0
    total_transitions = 0
    meta_step = 0
    update_step = 0
    while True:
        rng, train_rng = jax.random.split(rng)
        train_device_rngs = jax.random.split(train_rng, num=jax.local_device_count())
        loop += 1
        t = time()
        train_summary = jax.block_until_ready(train_fn(train_device_rngs, actor_train_state, critic_train_state))
        elapsed_time = time() - t
        print(f"Done in {elapsed_time:.4f}s.")
        print("Logginig...")
        train_info = train_summary["train_info"]
        update_step_info = unreplicate(train_info["update_step_info"])
        eval_info = unreplicate(train_info["eval_info"])
        actor_train_state = train_summary["actor_state"]
        critic_train_state = train_summary["critic_state"]

        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        orbax_checkpointer.save(
            os.path.abspath(f"{config.checkpoint_path}/{loop}_actor"),
            unreplicate(train_summary["actor_state"]).params
        )

        orbax_checkpointer.save(
            os.path.abspath(f"{config.checkpoint_path}/{loop}_critic"),
            unreplicate(train_summary["critic_state"]).params
        )

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


if __name__ == "__main__":
    config = Config(
        n_meta_steps=1,
        n_actor_steps=8,
        n_update_steps=1,
        # n_update_steps=4,
        n_envs=2,
        n_envs_per_device=2,
        n_eval_envs=1,
        n_minibatches=1,
    )
    train(config=config)
