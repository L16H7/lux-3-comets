import jax
import orbax
import os
import orbax.checkpoint
import jax.numpy as jnp

from flax.training.train_state import TrainState
from rnn import Actor, ScannedRNN


jax.config.update('jax_platform_name', 'cpu')
device = jax.local_devices()[0] 
checkpoint_path = os.path.abspath("/Users/light/research/neurips/rl_practice/checkpoints/_172_actor_cpu")
# Create a new checkpointer instance
orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
params = orbax_checkpointer.restore(checkpoint_path)

actor = Actor(n_actions=6)
SEQ = 1
BATCH = 16

actor_init_hstate = ScannedRNN.initialize_carry(BATCH, 256)
actor_input = {
    "observations": jnp.zeros((SEQ, BATCH, 9, 9, 9)),
    "teams": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "matches": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "match_steps": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "positions": jnp.zeros((SEQ, BATCH, 2)),
    "relic_nodes_positions": jnp.zeros((SEQ, BATCH, 6, 2),),
    "team_positions": jnp.zeros((SEQ, BATCH, 16, 2)),
    "opponent_positions": jnp.zeros((SEQ, BATCH, 16, 2)),
    "team_points": jnp.zeros((SEQ, BATCH, 1)),
    "opponent_points": jnp.zeros((SEQ, BATCH, 1)),
}

rng = jax.random.PRNGKey(42)
actor_network_params = actor.init(rng, actor_init_hstate, {
    "observations": jnp.zeros((SEQ, BATCH, 9, 9, 9)),
    "teams": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "matches": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "match_steps": jnp.zeros((SEQ, BATCH,), dtype=jnp.int32),
    "positions": jnp.zeros((SEQ, BATCH, 2)),
    "relic_nodes_positions": jnp.zeros((SEQ, BATCH, 6, 2),),
    "team_positions": jnp.zeros((SEQ, BATCH, 16, 2)),
    "opponent_positions": jnp.zeros((SEQ, BATCH, 16, 2)),
    "team_points": jnp.zeros((SEQ, BATCH, 1)),
    "opponent_points": jnp.zeros((SEQ, BATCH, 1)),
})
jax.debug.breakpoint()

inff = actor.apply({ "params": params }, actor_init_hstate, actor_input)
jax.debug.breakpoint()
