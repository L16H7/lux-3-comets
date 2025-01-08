import jax
import distrax

import jax.numpy as jnp

from constants import Constants
from representation import transform_coordinates


directions = jnp.array(
    [
        [0, -1],    # Move up
        [1, 0],     # Move right
        [0, 1],     # Move down
        [-1, 0],    # Move left
    ],
    dtype=jnp.int16,
)

def transform_observation(obs):
    # Horizontal flip across the last dimension (24, 24 grids)
    flipped = jnp.flip(obs, axis=2)
    
    # Rotate 90 degrees clockwise after flip, across the last two dimensions (24x24)
    rotated = jnp.rot90(flipped, k=-1, axes=(1, 2))
    
    return rotated

def vectorized_transform_actions(actions):
    # Create a JAX array that maps each action index to its new action
    # Index:      0  1  2  3  4 . 5
    # Action map: 0  2  1  4  3 . 5
    action_map = jnp.array([0, 2, 1, 4, 3, 5])

    # Vectorized mapping
    transformed_actions = action_map[actions]
    return transformed_actions

# @jax.jit
def get_actions(rng, team_idx: int, opponent_idx: int, logits, observations, sap_ranges):
    n_envs = observations.units.position.shape[0]
    
    agent_positions = observations.units.position[:, team_idx, ..., None, :] 
    agent_positions = agent_positions if team_idx == 0 else transform_coordinates(agent_positions)

    new_positions = agent_positions + directions

    in_bounds = (
        (new_positions[..., 0] >= 0) & (new_positions[..., 0] <= Constants.MAP_WIDTH - 1) &
        (new_positions[..., 1] >= 0) & (new_positions[..., 1] <= Constants.MAP_HEIGHT - 1)
    )

    asteroid_tiles = observations.map_features.tile_type == Constants.ASTEROID_TILE
    asteroid_tiles = asteroid_tiles if team_idx == 0 else transform_observation(asteroid_tiles)

    is_asteroid = asteroid_tiles[
        0, 
        new_positions[..., 0].clip(0, Constants.MAP_WIDTH - 1),
        new_positions[..., 1].clip(0, Constants.MAP_HEIGHT - 1),
    ]
    valid_movements = in_bounds & (~is_asteroid)

    team_positions = observations.units.position[:, team_idx, ...]
    team_positions = team_positions if team_idx == 0 else transform_coordinates(team_positions)

    opponent_positions = observations.units.position[:, opponent_idx, ...]
    opponent_positions = opponent_positions if team_idx == 0 else transform_coordinates(opponent_positions)
    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions
    )
    opponent_positions = jnp.where(
        opponent_positions == 24,
        -100,
        opponent_positions
    )

    opponent_positions = opponent_positions + Constants.MAX_SAP_RANGE

    neighboring_positions = jnp.expand_dims(opponent_positions, axis=2) + directions
    neighboring_positions = jnp.where(
        neighboring_positions == -1,
        -100,
        neighboring_positions
    )
    neighboring_positions = jnp.where(
        neighboring_positions == 24,
        -100,
        neighboring_positions
    )

    target_positions = jnp.concatenate([
        opponent_positions,
        neighboring_positions.reshape(neighboring_positions.shape[0], -1, 2)
    ], axis=1)

    diff = -team_positions[:, :, None, :] + target_positions[:, None, :, :]
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
        jnp.squeeze(logits[1].reshape(1, -1, 17), axis=0),
        dtype=bool
    )

    diff = diff.reshape(-1, 80, 2)
    x = diff[..., 0]
    attack_x = update_bool_array_jit(bool_array, x)

    y = diff[..., 1]
    attack_y = update_bool_array_jit(bool_array, y)

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

    attack_available = logits2_mask.sum(-1) & logits3_mask.sum(-1)

    logits1_mask = jnp.concat(
        [ 
            jnp.ones((1, attack_available.shape[0], 1)),
            valid_movements.reshape(1, -1, 4),
            attack_available.reshape(1, -1, 1) 
        ],
        axis=-1
    )

    logits2_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits2_mask
    )

    logits3_mask = jnp.where(
        jnp.expand_dims(attack_available, axis=-1).repeat(17, axis=-1) == 0,
        1,
        logits3_mask
    )

    logits1, logits2, logits3 = logits

    logits1_mask = logits1_mask.reshape(logits1.shape)
    logits2_mask = logits2_mask.reshape(logits2.shape)
    logits3_mask = logits3_mask.reshape(logits3.shape)

    large_negative = -1e9
    masked_logits1 = jnp.where(logits1_mask.reshape(logits1.shape), logits1, large_negative)
    masked_logits2 = jnp.where(logits2_mask.reshape(logits2.shape), logits2, large_negative)
    masked_logits3 = jnp.where(logits3_mask.reshape(logits3.shape), logits3, large_negative)

    '''
    sap_range_clip = Constants.MAX_SAP_RANGE - 1
    logits2 = logits2.at[..., : sap_range_clip].set(-100)
    logits2 = logits2.at[..., -sap_range_clip:].set(-100)

    logits3 = logits3.at[..., : sap_range_clip].set(-100)
    logits3 = logits3.at[..., -sap_range_clip:].set(-100)
    '''

    dist1 = distrax.Categorical(logits=masked_logits1)
    dist2 = distrax.Categorical(logits=masked_logits2)
    dist3 = distrax.Categorical(logits=masked_logits3)

    rng, action_rng1, action_rng2, action_rng3 = jax.random.split(rng, num=4)
    actions1, log_probs1 = dist1.sample_and_log_prob(seed=action_rng1)
    actions2, log_probs2 = dist2.sample_and_log_prob(seed=action_rng2)
    actions3, log_probs3 = dist3.sample_and_log_prob(seed=action_rng3)
    actions = [actions1, actions2, actions3]

    log_probs = log_probs1 + log_probs2 + log_probs3

    actions = jnp.stack([actions1.reshape(-1), actions2.reshape(-1), actions3.reshape(-1)])
    actions = actions.T.reshape(n_envs, 16, -1)

    logits_mask = [
        logits1_mask,
        logits2_mask,
        logits3_mask
    ]

    return actions, log_probs, logits_mask
