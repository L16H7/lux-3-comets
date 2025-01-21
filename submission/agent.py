import jax

import jax.numpy as jnp
# import numpy as np

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

def mask_sap_range(logits_slice, cutoff_range):
    cols = logits_slice.shape[1]
    mask = jnp.arange(cols) < cutoff_range
    logits_slice = jnp.where(mask[None, :], 0, logits_slice)
    
    mask2 = jnp.arange(cols) > (16 - cutoff_range)
    logits_slice = jnp.where(mask2[None, :], 0, logits_slice)
    return logits_slice

mask_sap_range_vmap = jax.vmap(
    mask_sap_range,
    in_axes=(0, 0)
)

def mask_out_of_bounds(agent_positions):
    target_coods = jnp.arange(-8, 9)
    target_x = agent_positions.reshape(-1, 2)[:, 0][:, None] + target_coods[None, :]
    target_x = (target_x >= 0) & (target_x < Constants.MAP_WIDTH)

    target_y = agent_positions.reshape(-1, 2)[:, 1][:, None] + target_coods[None, :]
    target_y = (target_y >= 0) & (target_y < Constants.MAP_HEIGHT)
    return target_x, target_y

def filter_targets_with_sensor(target_positions, sensor_map):
    """
    Filter target positions, replacing with (-1, -1) if sensor is True at that position.
    
    Args:
        target_positions (jnp.ndarray): Shape (n_envs, 16, 2) array of target positions
        sensor_map (jnp.ndarray): Shape (n_envs, 24, 24) boolean array where True means sensor
        
    Returns:
        jnp.ndarray: Shape (n_envs, 16, 2) filtered target positions
    """
    def process_single_env(targets, sensors):
        def filter_single_target(pos):
            # Check if position is already invalid
            is_valid = jnp.all(pos != -1)
            
            # Get sensor value at target position
            sensor_value = jnp.where(
                is_valid,
                sensors[pos[0], pos[1]],  # Only check sensor if position is valid
                True  # If position was already invalid, keep it invalid
            )
            
            # Replace with -1,-1 if sensor is True
            return jnp.where(sensor_value, jnp.array([-1, -1]), pos)
        
        # Apply to each target position
        return jax.vmap(filter_single_target)(targets)
    
    # Apply to each environment
    return jax.vmap(process_single_env)(target_positions, sensor_map)

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

    adjacent_offsets = jnp.array(
        [
            [0, 0],
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ], dtype=jnp.int16
    )

    sensor_mask = observations.sensor_mask
    sensor_mask = sensor_mask if team_idx == 0 else transform_observation(sensor_mask)

    transformed_agent_positions = transform_coordinates(agent_positions)
    transformed_agent_positions = filter_targets_with_sensor(
        transformed_agent_positions,
        observations.sensor_mask
    )
    opponent_positions = observations.units.position[:, opponent_idx, ..., None, :] 
    opponent_positions = opponent_positions if team_idx == 0 else transform_coordinates(opponent_positions)

    opponent_positions = jnp.concat([
        opponent_positions,
        transformed_agent_positions, # to attack mirror positions
    ], axis=1)
    
    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions,
    )

    opponent_positions = jnp.where(
        opponent_positions == 24,
        -100,
        opponent_positions,
    )

    target_positions = opponent_positions + adjacent_offsets
    target_x, _ = generate_attack_masks_batch(
        agent_positions.reshape(n_envs, -1, 2),
        target_positions.reshape(n_envs, -1, 2),
        sap_ranges,
        sap_ranges
    )
    logits2_mask = target_x

    logits1_mask = jnp.concat(
        [ 
            jnp.ones((1, n_envs * 16, 1)),
            valid_movements.reshape(1, -1, 4),
            target_x.sum(axis=-1).reshape(1, n_envs * 16, 1)
        ],
        axis=-1
    )

    logits1, logits2, logits3 = logits

    logits1_mask = logits1_mask.reshape(logits1.shape)
    logits2_mask = logits2_mask.reshape(logits2.shape)
    large_negative = -1e9
    masked_logits1 = jnp.where(logits1_mask.reshape(logits1.shape), logits1, large_negative)
    masked_logits2 = jnp.where(logits2_mask.reshape(logits2.shape), logits2, large_negative)

    rng, rng1, rng2, rng3 = jax.random.split(rng, num=4)
    actions1 = jax.random.categorical(rng1, masked_logits1, axis=-1)
    actions2 = jax.random.categorical(rng2, masked_logits2, axis=-1)

    # actions1 = np.argmax(masked_logits1, axis=-1)
    # actions2 = np.argmax(masked_logits2, axis=-1)

    target_y = jax.vmap(
        generate_attack_masks_y,
        in_axes=(0, 0, 0, 0, 0)
    )(
        agent_positions.reshape(n_envs, -1, 2),
        target_positions.reshape(n_envs, -1, 2),
        sap_ranges,
        sap_ranges,
        actions2.reshape(n_envs, -1) - Constants.MAX_SAP_RANGE
    )

    logits3_mask = target_y
    logits3_mask = logits3_mask.reshape(logits3.shape)
    masked_logits3 = jnp.where(logits3_mask.reshape(logits3.shape), logits3, large_negative)

    actions3 = jax.random.categorical(rng3, masked_logits3, axis=-1)
    # actions3 = np.argmax(masked_logits3, axis=-1)

    actions = jnp.stack([actions1, actions2, actions3]).T
    return actions

def generate_attack_masks(agent_positions, target_positions, x_range=8, y_range=8, choose_y=False, chosen_x=None,):
    """
    Generate attack masks for agents based on both x and y distances to targets.
    Targets outside the range are filtered out before mask generation.
    
    Args:
        agent_positions (jnp.ndarray): Shape (num_agents, 2) array of agent positions
        target_positions (jnp.ndarray): Shape (num_targets, 2) array of target positions
        x_range (int): Maximum x-distance range (default 8)
        y_range (int): Maximum y-distance range (default 8)
    
    Returns:
        attack_masks (jnp.ndarray): Shape (num_agents, 17, 17) boolean array
                                   True indicates valid attack position at that offset
    """
    # Pre-filter invalid targets (marked as -1)
    valid_targets = target_positions != -1
    valid_targets = jnp.all(valid_targets, axis=-1)
    target_positions = jnp.where(valid_targets[:, None], target_positions, 1000)
    
    # Calculate x and y distances from agent to each target
    x_distances = target_positions[None, :, 0] - agent_positions[:, None, 0]
    y_distances = target_positions[None, :, 1] - agent_positions[:, None, 1]
    
    # Create range mask for targets
    targets_in_range = (jnp.abs(x_distances) <= x_range) & (jnp.abs(y_distances) <= y_range)
    targets_in_range = targets_in_range & valid_targets[None, :]

    target_positions = jnp.where(target_positions == -1, 1000, target_positions)

    x_distances = jnp.where(
        targets_in_range,
        x_distances,
        -100, 
    )
    y_distances = jnp.where(
        targets_in_range,
        y_distances,
        -100,
    )
    
    x_offsets = jnp.arange(-8, 9)
    y_offsets = jnp.arange(-8, 9)
    
    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Check valid positions for x and y separately
    valid_x = (x_distances == x_offsets)
    valid_x = jnp.any(valid_x, axis=-1)

    if choose_y:
        x_distances = x_distances[:, None, :]
        y_distances = y_distances[:, None, :]
        x_offsets = x_offsets[None, :, None]
        y_offsets = y_offsets[None, :, None]
        
        # Filter targets based on chosen x
        chosen_x = chosen_x[:, None, None]  # Shape: (num_agents, 1, 1)
        valid_targets_for_x = (x_distances == chosen_x)
        
        # Apply the x-based filter to y distances
        y_distances = jnp.where(valid_targets_for_x, y_distances, -100)
        
        # Generate y masks only for valid targets based on chosen x
        valid_y = (y_distances == y_offsets)
        valid_y = jnp.any(valid_y, axis=-1)

        indices = jnp.arange(valid_y.shape[1])

        # Use advanced indexing to extract the desired slices
        final_filter = valid_y[0, indices, indices, :]
        return final_filter

    valid_y = (y_distances == y_offsets)
    valid_y = jnp.any(valid_y, axis=-1)

    return valid_x, valid_y


generate_attack_masks_batch = jax.vmap(
    generate_attack_masks,
    in_axes=(0, 0, 0, 0)
)

def generate_attack_masks_y(agent_positions, target_positions, x_range=8, y_range=8, chosen_x=None):
    """
    Generate attack masks for agents based on both x and y distances to targets.
    Targets outside the range are filtered out before mask generation.
    
    Args:
        agent_positions (jnp.ndarray): Shape (num_agents, 2) array of agent positions
        target_positions (jnp.ndarray): Shape (num_targets, 2) array of target positions
        x_range (int): Maximum x-distance range (default 8)
        y_range (int): Maximum y-distance range (default 8)
    
    Returns:
        attack_masks (jnp.ndarray): Shape (num_agents, 17, 17) boolean array
                                   True indicates valid attack position at that offset
    """
    # Pre-filter invalid targets (marked as -1)
    valid_targets = target_positions != -1
    valid_targets = jnp.all(valid_targets, axis=-1)
    target_positions = jnp.where(valid_targets[:, None], target_positions, 1000)
    
    # Calculate x and y distances from agent to each target
    x_distances = target_positions[None, :, 0] - agent_positions[:, None, 0]
    y_distances = target_positions[None, :, 1] - agent_positions[:, None, 1]
    
    # Create range mask for targets
    targets_in_range = (jnp.abs(x_distances) <= x_range) & (jnp.abs(y_distances) <= y_range)
    targets_in_range = targets_in_range & valid_targets[None, :]

    target_positions = jnp.where(target_positions == -1, 1000, target_positions)

    x_distances = jnp.where(
        targets_in_range,
        x_distances,
        -100, 
    )
    y_distances = jnp.where(
        targets_in_range,
        y_distances,
        -100,
    )
    
    x_offsets = jnp.arange(-8, 9)
    y_offsets = jnp.arange(-8, 9)
    
    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Check valid positions for x and y separately
    valid_x = (x_distances == x_offsets)
    valid_x = jnp.any(valid_x, axis=-1)

    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Filter targets based on chosen x
    chosen_x = chosen_x[:, None, None]  # Shape: (num_agents, 1, 1)
    valid_targets_for_x = (x_distances == chosen_x)
    
    # Apply the x-based filter to y distances
    y_distances = jnp.where(valid_targets_for_x, y_distances, -100)
    
    # Generate y masks only for valid targets based on chosen x
    valid_y = (y_distances == y_offsets)
    valid_y = jnp.any(valid_y, axis=-1)

    indices = jnp.arange(valid_y.shape[1])

    # Use advanced indexing to extract the desired slices
    final_filter = valid_y[0, indices, indices, :]
    return final_filter
