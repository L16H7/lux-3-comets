import jax
import jax.numpy as jnp

from utils import transform_coordinates


def filter_by_proximity(positions, relic_nodes, x_threshold=2, y_threshold=2):
    """
    Filter positions by proximity to relic nodes considering a rectangular 5x5 aura.
    
    Args:
        positions (jnp.ndarray): Shape (num, 2) array of positions to filter
        relic_nodes (jnp.ndarray): Shape (num_relics, 2) array of relic node positions
        x_threshold (int): Maximum allowed x-distance (default 2 for 5x5 aura)
        y_threshold (int): Maximum allowed y-distance (default 2 for 5x5 aura)
    
    Returns:
        filtered_positions (jnp.ndarray): Shape (num, 2) with positions outside aura replaced by [-1, -1]
    """
    # Handle invalid relic nodes (those marked as -1)
    relic_nodes = jnp.where(relic_nodes == -1, 100, relic_nodes)
    
    # Calculate x and y distances separately
    # positions[:, None] - relic_nodes[None, :] creates a (num_positions, num_relics, 2) array
    # where axis 2 contains [x_diff, y_diff]
    distances = positions[:, None] - relic_nodes[None, :]
    
    # Get absolute distances for x and y separately
    x_distances = jnp.abs(distances[..., 0])  # Shape: (num_positions, num_relics)
    y_distances = jnp.abs(distances[..., 1])  # Shape: (num_positions, num_relics)
    
    # Check if positions are within both x and y thresholds for any relic
    within_x = x_distances <= x_threshold
    within_y = y_distances <= y_threshold
    is_within_aura = jnp.any(within_x & within_y, axis=-1)
    
    # Replace positions outside the aura with [-1, -1]
    filtered_positions = jnp.where(is_within_aura[:, None], positions, jnp.array([-1, -1]))
    
    return filtered_positions

# Vectorize the function to operate on batches
filter_by_proximity_batch = jax.vmap(filter_by_proximity, in_axes=(0, 0), out_axes=0)


def mark_duplicates_single(positions):
    """
    Replace duplicate positions with [-1, -1] for a single batch.
    
    Args:
        positions: jnp array of shape (N, 2) containing (row, col) positions
    
    Returns:
        jnp array of same shape with duplicates replaced by [-1, -1]
    """
    position_keys = positions[:, 0] * 100 + positions[:, 1]
    sorted_indices = jnp.argsort(position_keys)
    sorted_keys = position_keys[sorted_indices]
    is_duplicate = jnp.roll(sorted_keys, -1) == sorted_keys
    
    duplicate_mask = jnp.zeros_like(position_keys, dtype=bool)
    duplicate_mask = duplicate_mask.at[sorted_indices].set(is_duplicate)
    
    result = jnp.where(
        duplicate_mask[:, None],
        jnp.array([-1, -1]),
        positions
    )
    
    return result

# Vectorized version for batch processing
mark_duplicates_batched = jax.vmap(mark_duplicates_single)

@jax.jit
def update_points_map(points_map, positions, points_gained):
    """
    Rules:
    1. Ignore [-1, -1] positions completely.
    2. No points gained
        - Update all positions with -1. Confirmed negatives
    3. Points gained:
        - Calculate points per unit for unconfirmed positions
        - Preserve existing confirmed cells (value -1 and 1)
        - Update cells based on probability
            - e.g., 2 points gained for 3 unconfirmed cells ~= 0.6667 per cell
            - 2 points gained for 2 unconfirmed cells = 1 per cell. Confirmed positives
    
    Args:
        points_map: 2D array of current cell values (-1 <= 1)
        positions: Array of shape (N, 2) containing (x, y) coordinates, with (-1, -1) for invalid positions
        points_gained: Total points gained in current step
    
    Returns:
        - updated_map: New points map with updated cell values
    """
    # Use where instead of boolean indexing
    valid_positions = (positions != -1).all(axis=1)
    
    # Get all rows and cols, but mask their effect using valid_positions
    rows = positions[:, 1]
    cols = positions[:, 0]
    
    # Get current values for all positions
    current_values = points_map[rows, cols]
    
    # Mask operations based on valid positions
    confirmed_positive = (current_values == 1.0) & valid_positions
    confirmed_negative = (current_values == -1.0) & valid_positions
    confirmed_mask = confirmed_positive | confirmed_negative
    unconfirmed_mask = ~confirmed_mask & valid_positions  # Only unconfirmed valid positions
    
    # Count how many are unconfirmed
    unconfirmed_count = jnp.sum(unconfirmed_mask)
    
    # Already-confirmed positives count
    confirmed_total = jnp.sum(confirmed_positive).astype(points_gained.dtype)
    
    # Subtract out those already-confirmed positives
    remaining_points = points_gained - confirmed_total
    
    # Distribute leftover points among unconfirmed
    points_per_unit = jnp.where(
        unconfirmed_count > 0,
        jnp.round(remaining_points / unconfirmed_count, 4),
        0.0
    )
    
    # Calculate new values for unconfirmed positions
    new_unconfirmed_values = jnp.where(
        points_per_unit > 0.0,
        points_per_unit,   # partial fraction
        -1.0               # no leftover => confirmed negative
    )
    new_unconfirmed_values = jnp.minimum(new_unconfirmed_values, 1)
    
    # Build the final updated values for each position
    updated_values = jnp.where(
        confirmed_positive,
        1.0,  # Keep positives as is
        jnp.where(
            confirmed_negative,
            -1.0,  # Keep negatives as is
            jnp.where(
                unconfirmed_mask,
                new_unconfirmed_values,
                current_values  # Keep original values for invalid positions
            )
        )
    )
    
    # Write the updated values back into the grid
    updated_map = points_map.at[rows, cols].set(updated_values)
    return updated_map

update_points_map_batch = jax.jit(jax.vmap(update_points_map, in_axes=(0, 0, 0)))


def update_points_map_with_relic_nodes(points_map, relic_nodes, positions, points_gained):
    proximity_positions = filter_by_proximity_batch(
        positions,
        relic_nodes
    )
    proximity_positions = mark_duplicates_batched(proximity_positions)
    transformed_proximity_positions = transform_coordinates(proximity_positions)

    updated_points_map = update_points_map_batch(
        points_map,
        jnp.concatenate([
            proximity_positions,
            transformed_proximity_positions,
        ], axis=1),
        points_gained * 2,
    )

    return updated_points_map

vmap_update_points_map_with_relic_nodes = jax.vmap(
    update_points_map_with_relic_nodes,
    in_axes=(None, None, 0, 0)
)
