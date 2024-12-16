import jax
import distrax

import jax.numpy as jnp

from constants import Constants


directions = jnp.array(
    [
        [0, -1],    # Move up
        [1, 0],     # Move right
        [0, 1],     # Move down
        [-1, 0],    # Move left
    ],
    dtype=jnp.int16,
)

@jax.jit
def get_actions(rng, team_idx: int, opponent_idx: int, logits, observations, sap_ranges):
    n_envs = observations.units.position.shape[0]
    
    new_positions = observations.units.position[:, team_idx, ..., None, :] + directions

    in_bounds = (
        (new_positions[..., 0] >= 0) & (new_positions[..., 0] <= Constants.MAP_WIDTH - 1) &
        (new_positions[..., 1] >= 0) & (new_positions[..., 1] <= Constants.MAP_HEIGHT - 1)
    )

    is_asteroid = (observations.map_features.tile_type == Constants.ASTEROID_TILE)[
        0, 
        new_positions[..., 0].clip(0, Constants.MAP_WIDTH - 1),
        new_positions[..., 1].clip(0, Constants.MAP_HEIGHT - 1),
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

    attack_available = attack_x.sum(-1) & attack_y.sum(-1) & ((diff.sum(-1) < (4 * Constants.MAX_SAP_RANGE)).sum(-1) > 0)

    logits1_mask = jnp.concat(
        [ 
            jnp.ones((1, attack_available.shape[0], 1)),
            valid_movements.reshape(1, -1, 4),
            attack_available.reshape(1, -1, 1) 
        ],
        axis=-1
    )

    def mask_sap_range(logits_slice, sap_range):
        cols = logits_slice.shape[1]
        mask = jnp.arange(cols) < sap_range
        logits_slice = jnp.where(mask[None, :], 0, logits_slice)
        
        mask2 = jnp.arange(cols) > (16 - sap_range)
        logits_slice = jnp.where(mask2[None, :], 0, logits_slice)
        return logits_slice

    sap_range_mask = jnp.ones((n_envs, 16, 17), sap_ranges)
    sap_range_mask = jax.vmap(mask_sap_range, in_axes=(0, 0))(sap_range_mask, sap_ranges)
    sap_range_mask = sap_range_mask.reshape(-1, 17)

    logits2_mask = attack_x & sap_range_mask
    logits3_mask = attack_y & sap_range_mask

    logits2_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits2_mask
    )
    logits2_mask = jnp.expand_dims(logits2_mask, axis=0)

    logits3_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits3_mask
    )
    logits3_mask = jnp.expand_dims(logits3_mask, axis=0)

    logits1, logits2, logits3 = logits
    jax.debug.breakpoint()
    masked_logits1 = jnp.where(logits1_mask, logits1, -jnp.inf)
    masked_logits2 = jnp.where(logits2_mask, logits2, -jnp.inf)
    masked_logits3 = jnp.where(logits3_mask, logits3, -jnp.inf)

    '''
    sap_range_clip = Constants.MAX_SAP_RANGE - 1
    logits2 = logits2.at[..., : sap_range_clip].set(-100)
    logits2 = logits2.at[..., -sap_range_clip:].set(-100)

    logits3 = logits3.at[..., : sap_range_clip].set(-100)
    logits3 = logits3.at[..., -sap_range_clip:].set(-100)
    '''

    dist1 = distrax.Categorical(logits=masked_logits1)
    dist2 = distrax.Categorical(logits=masked_logits2.reshape(1, -1, 17))
    dist3 = distrax.Categorical(logits=masked_logits3.reshape(1, -1, 17))
    dist = distrax.Joint([dist1, dist2, dist3])

    actions, log_probs = dist.sample_and_log_prob(seed=rng)
    jax.debug.breakpoint()
    return actions, log_probs
