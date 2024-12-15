import jax
import distrax

import jax.numpy as jnp

from constants import Constants


directions = jnp.array(
    [
        [0, -1],  # Move up
        [1, 0],  # Move right
        [0, 1],  # Move down
        [-1, 0],  # Move left
    ],
    dtype=jnp.int16,
)

@jax.jit
def get_actions(rng, logits, observations):
    new_positions = observations.units.position[:, 0, ..., None, :] + directions

    in_bounds = (
        (new_positions[..., 0] >= 0) & (new_positions[..., 0] <= Constants.MAP_WIDTH - 1) &
        (new_positions[..., 1] >= 0) & (new_positions[..., 1] <= Constants.MAP_HEIGHT - 1)
    )

    is_asteroid = (observations.map_features.tile_type == Constants.ASTEROID_TILE)[
        0, 
        new_positions[..., 1].clip(0, Constants.MAP_HEIGHT - 1),
        new_positions[..., 0].clip(0, Constants.MAP_WIDTH - 1)
    ]
    valid_movements = in_bounds & (~is_asteroid)
    jax.debug.breakpoint()

    logits1, logits2, logits3 = logits
    dist1 = distrax.Categorical(logits=logits1)
    dist2 = distrax.Categorical(logits=logits2)
    dist3 = distrax.Categorical(logits=logits3)
    dist = distrax.Joint([dist1, dist2, dist3])

    actions, log_probs = dist.sample_and_log_prob(seed=rng)
    return actions, log_probs
