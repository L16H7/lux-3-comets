import jax
import orbax.checkpoint
import optax
from flax.training.train_state import TrainState
import jax.numpy as jnp

from config import Config
from model import ActorCritic


def make_states(config: Config):
    rng = jax.random.PRNGKey(config.train_seed)
    model = ActorCritic(n_actions=6)

    BATCH = 16
    model_input = {
        "states": jnp.zeros((BATCH, 24, 24, 10)),
        "match_steps": jnp.zeros((BATCH,), dtype=jnp.float32),
        "matches": jnp.zeros((BATCH,), dtype=jnp.float32),
        "positions": jnp.zeros((BATCH, 16, 2), dtype=jnp.int32),
        "team_points": jnp.zeros((BATCH,)),
        "opponent_points": jnp.zeros((BATCH,)),
        "unit_move_cost": jnp.zeros((BATCH,)),
        "unit_sap_cost": jnp.zeros((BATCH,)),
        "unit_sap_range": jnp.zeros((BATCH,)),
        "unit_sensor_range": jnp.zeros((BATCH,)),
        "agent_ids": jnp.zeros((BATCH,)),
    }
    network_params = model.init(rng, model_input)

    num_params = sum(x.size for x in jax.tree_util.tree_leaves(network_params))
    print(f"Number of actor parameters: {num_params:,}")

    actor_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(config.actor_learning_rate),
    )

    ### --------- RESUME HERE --------- ###
    '''
    actor_checkpoint_path = '/root/lux-3-comets/checkpoints_old/35_actor'
    orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
    actor_network_params = orbax_checkpointer.restore(actor_checkpoint_path)
    '''
    ### ------------------------------- ###

    train_state = TrainState.create(
        apply_fn=model.apply,
        params=network_params,
        tx=actor_tx,
    )


    return train_state
