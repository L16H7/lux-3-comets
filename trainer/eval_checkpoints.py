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
from flax.training import train_state
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
from make_states import make_states
from model import Actor
from opponent import get_actions as get_opponent_actions
from ppo import Transition, calculate_gae, ppo_update
from representation import create_agent_representations, transform_coordinates, get_env_info, combined_states_info, reconcile_positions, teacher_get_env_info
from teacher.model import make_teacher_state

class TrainState(train_state.TrainState):
    key: jax.Array

class RunnerState(NamedTuple):
    rng: jnp.ndarray
    actor_train_state: TrainState
    critic_train_state: TrainState
    p0_representations: jnp.ndarray
    p1_representations: jnp.ndarray
    observations: jnp.ndarray
    states: jnp.ndarray


def make_eval_checkpoints(config: Config):
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
    def eval_checkpoints(
        rng: jax.Array,
        player_0_train_state: TrainState,
        player_1_train_state: TrainState,
    ):
        def _eval_step(meta_state, _):
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

            rng, _, _ = meta_state
            rng, meta_key_rng, meta_env_params_rng, eval_rng, _ = jax.random.split(rng, num=5)
            meta_keys = jax.random.split(meta_key_rng, config.n_envs)
            meta_env_params = jax.vmap(sample_params)(
                jax.random.split(meta_env_params_rng, config.n_envs)
            )

            rng, eval_meta_key_rng, eval_meta_env_params_rng, _ = jax.random.split(rng, num=4)
            eval_meta_keys = jax.random.split(eval_meta_key_rng, config.n_eval_envs)
            eval_meta_env_params = jax.vmap(sample_params)(
                jax.random.split(eval_meta_env_params_rng, config.n_eval_envs)
            )

            eval_info = evaluate(
                eval_rng,
                eval_meta_keys,
                eval_meta_env_params,
                player_0_train_state,
                player_1_train_state,
                'self',
                config.n_eval_envs,
                config.n_agents,
                v_reset,
                v_step
            )
            eval_info = jax.lax.pmean(eval_info, axis_name="devices")

            meta_step_info = {
                "eval_info": {
                    **eval_info,
                },
            }

            return meta_state, meta_step_info


        meta_state = (rng, player_0_train_state, player_1_train_state)
        meta_state, eval_checkpoints_info = jax.lax.scan(_eval_step, meta_state, None, config.n_meta_steps)

        return {
            "eval_checkpoints_info": eval_checkpoints_info,
        }

    return eval_checkpoints

def train(config: Config):
    run = wandb.init(
        project=config.wandb_project,
        config={**asdict(config)}
    )

    rng = jax.random.key(config.train_seed)

    actor = Actor()
    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.actor_learning_rate),
    )

    actor_checkpoint_path = '/root/lux-3-comets/checkpoints/80_actor'
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    actor_network_params = orbax_checkpointer.restore(actor_checkpoint_path)

    player_0_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_network_params,
        tx=actor_tx,
        key=rng,
    )
    train_device_rngs = jax.random.split(rng, num=jax.local_device_count())
    player_0_train_state = replicate(player_0_train_state, jax.local_devices())

    p1_actor_checkpoint_path = '/root/lux-3-comets/checkpoints/600_actor'
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    p1_actor_network_params = orbax_checkpointer.restore(p1_actor_checkpoint_path)

    player_1_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=p1_actor_network_params,
        tx=actor_tx,
        key=rng,
    )
    player_1_train_state = replicate(player_1_train_state, jax.local_devices())


    print("Compiling...")
    t = time()
    train_fn = make_eval_checkpoints(
        config=config,
    )
    train_fn = train_fn.lower(
        train_device_rngs,
        player_0_train_state,
        player_1_train_state,
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
                player_0_train_state,
                player_0_train_state,
            )
        )
        elapsed_time = time() - t
        print(f"Done in {elapsed_time:.4f}s.")
        print("Logginig...")
        eval_checkpoints_info = train_summary["eval_checkpoints_info"]
        eval_info = unreplicate(eval_checkpoints_info["eval_info"])

        for i in range(config.n_meta_steps):
            meta_step += 1
            meta_info = jtu.tree_map(lambda x: x[i], eval_info)
            meta_info["meta_steps"] = meta_step
            for j in range(config.n_update_steps):
                update_step += config.n_minibatches * config.n_epochs * jax.local_device_count()
                total_transitions += config.n_envs_per_device * jax.local_device_count() * config.n_actor_steps
                info = jtu.tree_map(lambda x: x[i, j], eval_checkpoints_info)
                wandb.log(info)


if __name__ == "__main__":
    config = Config(
        n_meta_steps=1,
        n_actor_steps=14,
        n_update_steps=36,
        n_envs=32,
        n_envs_per_device=32,
        n_eval_envs=32,
        n_minibatches=16,
        n_epochs=1,
        actor_learning_rate=8e-5,
        critic_learning_rate=1e-4,
        wandb_project="Bench",
        train_seed=42,
        entropy_coeff=0.01,
        gae_lambda=0.98,
        gamma=0.995,
    )
    train(config=config)
