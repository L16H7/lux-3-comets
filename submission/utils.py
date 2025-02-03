import jax
import jax.numpy as jnp


@jax.jit
def transform_coordinates(coordinates, map_width=24, map_height=24):
    # Adjust for horizontal flip: (x, y) -> (MAP_WIDTH - 1 - x, y)
    flipped_positions = jnp.stack([map_width - 1 - coordinates[..., 0], coordinates[..., 1]], axis=-1)
    
    # Adjust for 90-degree rotation clockwise: (MAP_WIDTH - 1 - x, y) -> (y, MAP_WIDTH - 1 - x)
    rotated_positions = jnp.stack([map_height - 1 - flipped_positions[..., 1], flipped_positions[..., 0]], axis=-1)
    
    return rotated_positions
