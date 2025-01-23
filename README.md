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
- Agent must not sap if target coordinates are out of sap range or out of the map.

## Point cells calulation
- By default, all cells will be marked as 0.
- If agents do not score any points, all prev positions will be marked as -1 permanently.
- If agents score points, all prev positions will be marked as 1 except for -1 cells.

# League training
- There will be 3 opponents.
    1. self
    2. best checkpoint
    3. exploiter

- self plays 3/8 of the n_envs
- best checkpoint plays 3/8
- exploiter plays 2/8
