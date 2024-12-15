## Movements
```python
# 0 is do nothing, 1 is move up, 2 is move right, 3 is move down, 4 is move left, 5 is sap
# Define movement directions
directions = jnp.array(
    [
        [0, 0],  # Do nothing
        [0, -1],  # Move up
        [1, 0],  # Move right
        [0, 1],  # Move down
        [-1, 0],  # Move left
    ],
    dtype=jnp.int16,
)
```

## Action masking
Action masking improves the learning. https://arxiv.org/abs/2006.14171

### Movment action masking
- Agents must not move outside of the map.
- Agents must not move into the asteroids.

### Sap action masking
- Agent must not sap if there is no other agent, either friendly or foe, inside 9x9 grid with the target in the middle.