import jax
import orbax.checkpoint
import optax
from flax.training import train_state
import jax.numpy as jnp

from config import Config
from model import Actor, Critic

class TrainState(train_state.TrainState):
        key: jax.Array

def make_states(config: Config):
    rng = jax.random.PRNGKey(config.train_seed)
    rng, dropout_rng = jax.random.split(rng)
    actor = Actor(n_actions=6)
    BATCH = 16
    actor_input = {
        "states": jnp.zeros((BATCH, 11, 24, 24)),
        "observations": jnp.zeros((BATCH, 19, 47, 47)),
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
        "energies_gained": jnp.zeros((BATCH,)),
        "points_gained_history": jnp.zeros((BATCH, 4)),
    }
    actor_network_params = actor.init({
        "params": rng,
        "dropout": dropout_rng,
    }, actor_input)

    print(actor.tabulate(rng, actor_input))

    critic = Critic()
    critic_input = {
        "states": jnp.zeros((BATCH, 24, 24, 16)),
        "match_steps": jnp.zeros((BATCH,)),
        "matches": jnp.zeros((BATCH,)),
        "team_points": jnp.zeros((BATCH,)),
        "opponent_points": jnp.zeros((BATCH,)),
        "points_gained_history": jnp.zeros((BATCH, 4)),
    }
    critic_network_params = critic.init(rng, critic_input)
    print(critic.tabulate(rng, critic_input))

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
        key=rng,
    )
    critic_train_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_network_params,
        tx=critic_tx,
        key=rng,
    )

    return actor_train_state, critic_train_state
