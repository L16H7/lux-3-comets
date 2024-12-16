import jax
import distrax

import jax.numpy as jnp

from constants import Constants


directions = jnp.array(
    [
        [0, 0],     # NO-OP
        [0, -1],    # Move up
        [1, 0],     # Move right
        [0, 1],     # Move down
        [-1, 0],    # Move left
    ],
    dtype=jnp.int16,
)

def get_actions(rng, team_idx, opponent_idx, logits, observations, sap_range=3):
    new_positions = observations.units.position[:, team_idx, ..., None, :] + directions

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

    team_positions = observations.units.position[:, team_idx, ...]
    opponent_positions = observations.units.position[:, opponent_idx, ...]
    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions
    )

    opponent_positions = opponent_positions + Constants.MAX_SAP_RANGE
    diff = -team_positions[:, :, None, :] + opponent_positions[:, None, :, :]
    diff = jnp.where(diff < 0, -100, diff)

    # Function to set True for one row given indices
    def set_true_row(bool_array, indices):
        return bool_array.at[indices].set(True)

    # Vectorize the function across rows using vmap
    def update_bool_array(bool_array, turn_ons):
        # vmap across the first axis (rows of turn_ons and bool_array)
        return jax.vmap(set_true_row, in_axes=(0, 0), out_axes=0)(bool_array, turn_ons)

    # Use JIT compilation for performance
    update_bool_array_jit = jax.jit(update_bool_array)

    bool_array = jnp.zeros_like(
        jnp.squeeze(logits[1], axis=0),
        dtype=bool
    )

    diff = diff.reshape(-1, 16, 2)
    x = diff[..., 0]
    attack_x = update_bool_array_jit(bool_array, x)

    y = diff[..., 1]
    attack_y = update_bool_array_jit(bool_array, y)

    attack_available = attack_x.sum(-1) & attack_y.sum(-1)

    action1_mask = jnp.concat(
        [ 
            valid_movements.reshape(1, -1, 5),
            attack_available.reshape(1, -1, 1) 
        ],
        axis=-1
    )

    logits1, logits2, logits3 = logits
    logits1 = jnp.where(action1_mask, logits1, -jnp.inf)
    logits2 = jnp.where(jnp.expand_dims(attack_x, axis=0), logits2, -jnp.inf)
    logits3 = jnp.where(jnp.expand_dims(attack_y, axis=0), logits3, -jnp.inf)

    dist1 = distrax.Categorical(logits=logits1)
    dist2 = distrax.Categorical(logits=logits2)
    dist3 = distrax.Categorical(logits=logits3)
    dist = distrax.Joint([dist1, dist2, dist3])

    actions, log_probs = dist.sample_and_log_prob(seed=rng)
    return actions, log_probs
