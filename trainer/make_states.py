import jax
import orbax.checkpoint
import optax
from flax.training.train_state import TrainState
import jax.numpy as jnp

from config import Config
from model import Actor, Critic


def make_states(config: Config):
    rng = jax.random.PRNGKey(config.train_seed)
    actor = Actor(n_actions=6)
    BATCH = 16
    actor_network_params = actor.init(rng, {
        "states": jnp.zeros((BATCH, 11, 24, 24)),
        "observations": jnp.zeros((BATCH, 16, 47, 47)),
        "match_steps": jnp.zeros((BATCH,), dtype=jnp.float32),
        "matches": jnp.zeros((BATCH,), dtype=jnp.float32),
        "positions": jnp.zeros((BATCH, 2), dtype=jnp.int32),
        "team_points": jnp.zeros((BATCH,)),
        "opponent_points": jnp.zeros((BATCH,)),
        "unit_move_cost": jnp.zeros((BATCH,)),
        "unit_sap_cost": jnp.zeros((BATCH,)),
        "unit_sap_range": jnp.zeros((BATCH,)),
        "unit_sensor_range": jnp.zeros((BATCH,)),
        "energies": jnp.zeros((BATCH,)),
    })

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(actor_network_params))
    print(f"Number of actor parameters: {num_params:,}")

    critic = Critic()
    critic_network_params = critic.init(rng, {
        "states": jnp.zeros((BATCH, 10, 24, 24)),
        "match_steps": jnp.zeros((BATCH,)),
        "matches": jnp.zeros((BATCH,)),
        "team_points": jnp.zeros((BATCH,)),
        "opponent_points": jnp.zeros((BATCH,)),
    })
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(critic_network_params))
    print(f"Number of critic parameters: {num_params:,}")

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.actor_learning_rate),
    )

    critic_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.critic_learning_rate),
    )

    ### --------- RESUME HERE --------- ###
    '''
    actor_checkpoint_path = '/root/lux-3-comets/checkpoints_old/35_actor'
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    actor_network_params = orbax_checkpointer.restore(actor_checkpoint_path)

    critic_checkpoint_path = '/root/lux-3-comets/checkpoints_old/35_critic'
    critic_network_params = orbax_checkpointer.restore(critic_checkpoint_path)
    print('resumed from', actor_checkpoint_path, critic_checkpoint_path)
    '''
    ### ------------------------------- ###


    actor_train_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_network_params,
        tx=actor_tx,
    )
    critic_train_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_network_params,
        tx=critic_tx,
    )

    return actor_train_state, critic_train_state
