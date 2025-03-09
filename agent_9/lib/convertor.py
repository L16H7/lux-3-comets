# Set of functions to convert Rule Base info to RL representation
import numpy as np
import jax.numpy as jnp
from lib.data_classes import TerrainTile, Fragment, Relic
from lib.helper import mark_nxn
from typing import Set


def convert_terrain(terrain: np.ndarray, nebula_energy_reduction: int) -> jnp.ndarray:
    """
    Nebula is converted to (nebula_energy_reduction - 1) * 0.05
    Asteroid is -2
    Unseen and empty are both 0
    """
    terrain = terrain.copy()
    terrain = terrain.astype(float)

    # change all unknown to 0
    terrain[terrain == TerrainTile.UNKNOWN.value] = 0

    # Change nebula values
    # In my case it is negative and -10 is default when we do not know the answer
    nebula = -(0 if nebula_energy_reduction == -10 else nebula_energy_reduction)
    nebula = (nebula - 1) * 0.05
    terrain[terrain == TerrainTile.NEBULA.value] = nebula

    terrain[terrain == TerrainTile.ASTEROID.value] = -2

    # RL uses 3 dims
    terrain = np.expand_dims(terrain, axis=0)  # Shape becomes (1, H, W)
    return jnp.asarray(terrain)


def convert_points_map(relics_possible: np.ndarray, fragments: Set[Fragment]) -> jnp.ndarray:
    """My representation is
        False - relic is not possible
        True - Possible but not seen yet or already seen and guaranteed

    RL representation is
        -1  for positions where fragment is not possible
         1  where fragment is guaranteed
        0.5 where you are unsure. I am not sure about it and maybe you have probability of being unsure
         0  for everything else
    """
    res = relics_possible.copy().astype(float)
    for f in fragments:
        res[f.y, f.x] = 2

    res[res == 1] = 0.5
    res[res == 2] = 1.0
    # RL uses 3 dims
    res = np.expand_dims(res, axis=0)  # Shape becomes (1, H, W)
    return jnp.asarray(res)


def convert_points_map_original(relics_possible: np.ndarray, fragments: Set[Fragment], relics: Set[Relic]) -> jnp.ndarray:
    """My representation is
        False - relic is not possible
        True - Possible but not seen yet or already seen and guaranteed

    RL representation is
        -1  for positions where fragment is not possible
         1  where fragment is guaranteed
        0.5 where you are unsure. I am not sure about it and maybe you have probability of being unsure
         0  for everything else
    """
    res = relics_possible.copy().astype(float)
    for f in fragments:
        res[f.y, f.x] = 2

    relics_data = np.zeros((24, 24), dtype=bool)
    for r in relics:
        mark_nxn(relics_data, r.pos())

    res[(res == 0) & relics_data] = -1
    res[res == 1] = 0
    res[res == 2] = 1

    # RL uses 3 dims
    res = np.expand_dims(res, axis=0)  # Shape becomes (1, H, W)
    return jnp.asarray(res)