import jax
import jax.numpy as jnp


def filter_by_proximity(positions, relic_nodes, max_distance=4):
    """
    Filter positions by proximity to relic nodes.

    Args:
    positions (jnp.ndarray): Shape (num, 2)
    relic_nodes (jnp.ndarray): Shape (num_relics, 2)
    max_distance (int): Maximum Manhattan distance

    Returns:
    filtered_positions (jnp.ndarray): Shape (num, 2)
    """
    distances = jnp.abs(positions[:, None] - relic_nodes[None, :]).sum(axis=-1)
    is_close = jnp.any(distances <= max_distance, axis=-1)
    filtered_positions = jnp.where(is_close[:, None], positions, jnp.array([-1, -1]))
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
    Enhanced points map update with probability distribution and cell state preservation.
    
    Rules:
    1. Default state: All cells start as 0 (unexplored)
    2. No points scenario: Mark as 0 (visited but no points) unless already marked as 1
    3. Points gained scenario:
        - Calculate points per unit for unconfirmed positions
        - Preserve existing confirmed cells (value -1 and 1)
        - Update cells based on probability e.g., 2 points gained for 3 unconfirmed cells ~= 0.67 per cell
    
    Args:
        points_map: 2D array of current cell values (-1 <= 1)
        positions: Array of shape (N, 2) containing (x, y) coordinates
        points_gained: Total points gained in current step
    
    Returns:
        - updated_map: New points map with updated cell values
    """
    rows = positions[:, 1]
    cols = positions[:, 0]

    current_values = points_map[rows, cols]

    confirmed_positive = (current_values == 1.0)
    confirmed_negative = (current_values == -1.0)

    confirmed_mask = confirmed_positive | confirmed_negative
    unconfirmed_mask = ~confirmed_mask

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
    # If leftover is negative or zero, it means no positive fraction is assigned
    # => those unconfirmed become -1 (negative)
    # Otherwise, they get the fractional points_per_unit
    new_unconfirmed_values = jnp.where(
        points_per_unit > 0.0,
        points_per_unit,   # partial fraction
        -1.0               # no leftover => confirmed negative
    )

    # Build the final updated values for each position
    updated_values = jnp.where(
        confirmed_positive,
        1.0,  # Keep positives as is
        jnp.where(
            confirmed_negative,
            -1.0,  # Keep negatives as is
            new_unconfirmed_values
        )
    )

    # Write the updated values back into the grid
    updated_map = points_map.at[rows, cols].set(updated_values)
    return updated_map

update_points_map_batch = jax.jit(jax.vmap(update_points_map, in_axes=(0, 0, 0)))
