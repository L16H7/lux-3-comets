import jax
import jax.numpy as jnp


@jax.jit
def update_points_map(points_map, positions, points_gained):
    """
    Enhanced points map update with probability distribution and cell state preservation.
    
    Rules:
    1. Default state: All cells start as -1 (unexplored)
    2. No points scenario: Mark as 0 (visited but no points) unless already marked as 1
    3. Points gained scenario:
        - Calculate points per unit for unconfirmed positions
        - Preserve existing confirmed cells (value 0 and 1)
        - Update cells based on probability e.g., 2 points gained for 3 unconfirmed cells ~= 0.67 per cell
    
    Args:
        points_map: 2D array of current cell values (-1, 0, or 1)
        positions: Array of shape (N, 2) containing (x, y) coordinates
        points_gained: Total points gained in current step
    
    Returns:
        - updated_map: New points map with updated cell values
    """
    rows = positions[:, 1]
    cols = positions[:, 0]

    # Gather the current values (for just the positions).
    current_values = points_map[rows, cols]

    # Identify which positions are already “confirmed negative” = 0, 
    # and which are “confirmed positive” = 1.
    confirmed_positive = (current_values == 1.0)
    confirmed_negative = (current_values == 0.0)

    # Confirmed means 0.0 or 1.0
    confirmed_mask = confirmed_negative | confirmed_positive
    unconfirmed_mask = ~confirmed_mask  # i.e. the ones that are still -1.0

    # Number of unconfirmed positions
    unconfirmed_count = jnp.sum(unconfirmed_mask)

    # Already-confirmed positives count
    confirmed_total = jnp.sum(confirmed_positive)

    # We first “use up” points to confirm existing positives.
    # E.g. if you gained 2 points, but 1 cell was already known to be “1.0”,
    # we interpret that as 1 leftover point to distribute.
    remaining_points = points_gained - confirmed_total

    # Points to assign equally among unconfirmed positions
    points_per_unit = jnp.where(
        unconfirmed_count > 0,
        remaining_points / unconfirmed_count,
        0.0
    )
    # If we ended up with a negative leftover, clamp to 0
    points_per_unit = jnp.maximum(
        jnp.round(points_per_unit, 4),
        0.0
    )

    # We create new values for the positions we’re updating:
    #  (1) preserve the ones that are already 1 or 0
    #  (2) for the unconfirmed (-1.0), set them to the fraction
    #      if points_gained == 0 => fraction = 0, effectively marking them as visited but no points
    updated_values = jnp.where(
        confirmed_positive,
        1.0,  # keep them as 1
        jnp.where(
            confirmed_negative,
            0.0,  # keep them as 0
            points_per_unit  # unconfirmed -> partial fraction
        )
    )

    # Insert these updated values back into points_map
    updated_map = points_map.at[rows, cols].set(updated_values)

    return updated_map

update_points_map_batch = jax.jit(jax.vmap(update_points_map, in_axes=(0, 0, 0)))
