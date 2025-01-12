import jax
import jax.numpy as jnp


@jax.jit
def update_points_map(points_map, positions, points_gained):
    """
    Updates a points map based on agent positions and points gained.
    
    The function maintains a map of cell values that follows these rules:
    - All cells start with a default value of 0
    - Cells marked as -1 are permanently locked at that value
    - When points_gained is 0, newly visited positions are marked as -1
    - When points_gained > 0, newly visited positions are marked as 1, unless already -1
    
    Args:
        points_map: 2D array representing the current state of the map
        positions: Array of shape (16, 2) containing (x, y) coordinates of positions to update
        points_gained: Integer indicating points gained in current step
        
    Returns:
        Updated points map with new values set according to rules
    """
    rows = positions[:, 1]
    cols = positions[:, 0]
    
    # Create mask for cells that are not permanently marked (-1)
    updatable_mask = points_map[rows, cols] != -1
    
    # Determine new values based on points_gained
    new_values = jnp.where(points_gained > 0, 1, -1)
    
    # Only update cells that aren't permanently marked
    update_values = jnp.where(updatable_mask, new_values, -1)
    
    # Update the map
    updated_map = points_map.at[rows, cols].set(update_values)
    
    return updated_map

update_points_map_batch = jax.vmap(update_points_map, in_axes=(0, 0, 0))
