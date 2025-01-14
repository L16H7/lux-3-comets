import jax
import jax.numpy as jnp


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
