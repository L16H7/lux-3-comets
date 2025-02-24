import numpy as np
from typing import List, Set, Tuple

from lib.data_classes import Position, Relic, Fragment, Unit, TerrainTile


def mark_nxn(X: np.ndarray[bool], p: Position, n: int = 5):
    """Takes a boolean matrix and marks an NxN area around the position with True"""
    if n % 2 == 0:
        raise Exception('N should be odd')

    if n == 1:
        X[p.y, p.x] = True
        return

    l1, l2 = n // 2, n - n // 2
    y_s, x_s = max(0, p.y - l1), max(0, p.x - l1)
    y_e, x_e = min(24, p.y + l2), min(24, p.x + l2)
    X[y_s:y_e, x_s:x_e] = True

def is_world_changed(step: int, terrain_speed: float) -> bool:
    return (step - 2) * terrain_speed % 1 > (step - 1) * terrain_speed % 1

def distance(p1: Position, p2: Position) -> int:
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

def is_shot_possible(u: Unit, p: Position, sap_range: int) -> bool:
    v1 = abs(u.y - p.y) <= sap_range
    v2 = abs(u.x - p.x) <= sap_range
    return v1 and v2

def is_shift_in_dir(V1: np.ndarray[bool], V2: np.ndarray[bool], T1: np.ndarray[int], T2: np.ndarray[int], is_up_right: bool = True) -> float:
    # Returns the probability that the shift was done in that direction
    M1_ = -np.ones_like(T1)
    if is_up_right:
        M1_[:-1, 1:] = T1[1:, :-1]
    else:  # then down-left
        M1_[1:, :-1] = T1[:-1, 1:]

    mask = (~V1) | (~V2) | (M1_ == -1) | (T2 == -1)  # only look at values seen to both
    M2_ = T2.copy()
    M1_[mask] = -1
    M2_[mask] = -1

    valid_rows = ~(np.all(M1_ == -1, axis=1) & np.all(M2_ == -1, axis=1))
    return sum(np.all(M1_[valid_rows] == M2_[valid_rows], axis=1)) / sum(valid_rows)

def is_move_detected(V1: np.ndarray[bool], V2: np.ndarray[bool], T1: np.ndarray[int], T2: np.ndarray[int]) -> bool:
    mask = V1 & V2
    return not np.array_equal(T1 * mask, T2 * mask)

def bool_matrix_to_strings(matrix: np.ndarray[bool]) -> List[str]:
    mapping = {
        False: ' ',
        True : '.',
    }
    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    char_matrix = vectorized_mapping(matrix)
    return ["".join(row) for row in char_matrix]

def terrain_matrix_to_strings(terrain: np.ndarray[int]) -> List[str]:
    mapping = {
        -1: '?',
        0: ' ',
        1: '#',
        2: '█'
    }
    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    char_matrix = vectorized_mapping(terrain)

    res = ['_' * 24]
    res.extend(["".join(row) for row in char_matrix])
    res.append('‾' * 24)
    return res

def relics_matrix_to_strings(relics_possible: np.ndarray[int], relics: Set[Relic], units: Set[Unit], vision: np.ndarray[bool]) -> List[str]:
    mapping = {
        0: ' ',     # Nothing
        1: '?',     # Do not know
        2: '.',     # I have vision
        3: '■',     # Unit is located
        4: 'X',     # Relic
    }
    matrix = relics_possible.astype(int)
    matrix[vision] = 2

    for u in units:
        matrix[u.y, u.x] = 3

    for r in relics:
        matrix[r.y, r.x] = 4

    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    char_matrix = vectorized_mapping(matrix)

    res = ['_' * 24]
    res.extend(["".join(row) for row in char_matrix])
    res.append('‾' * 24)
    return res

def fragment_matrix_to_string(fragments_possible: np.ndarray[bool], relics: Set[Relic], fragments: Set[Fragment]) -> List[str]:
    mapping = {
        0: ' ', # Cant have a relic/fragment here
        1: '.', # Possible to have a fragment
        2: '□', # Fragment probability is 0.5
        3: '■', # Fragment is guaranteed to be here
        4: 'X', # Relic is here
    }
    matrix = fragments_possible.astype(int)
    for f in fragments:
        if f.probability == 1:
            matrix[f.y, f.x] = 3
        elif f.probability == 0.5:
            matrix[f.y, f.x] = 2

    for r in relics:
        matrix[r.y, r.x] = 4

    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    char_matrix = vectorized_mapping(matrix)

    res = ['_' * 24]
    res.extend(["".join(row) for row in char_matrix])
    res.append('‾' * 24)
    return res

def mask_other_side(team_id: int) -> np.ndarray[bool]:
    indices = np.arange(24)
    if team_id == 0:
        return indices[:, None] + indices > 24 - 1  # Below anti-diagonal

    return indices[:, None] + indices < 24 - 1  # Above anti-diagonal

def change_terrain(map_info_terrain: np.ndarray[int], terrain: np.ndarray[int], move_dir: int) -> np.ndarray[int]:
    M = -np.ones_like(map_info_terrain)
    if move_dir == 1:
        M[:-1, 1:] = map_info_terrain[1:, :-1]
        return M
    if move_dir == -1:
        M[1:, :-1] = map_info_terrain[:-1, 1:]
        return M

    return terrain

def best_in_kxk_region(matrix: np.ndarray[int], p: Position, k: int = 5, min_energy: int = 3) -> List[Position]:
    if k % 2 == 0:
        raise Exception('Only odd value possible')
    k1, k2 = k // 2, k // 2 + 1

    y_min, y_max = max(0, p.y - k1), min(24, p.y + k2)
    x_min, x_max = max(0, p.x - k1), min(24, p.x + k2)
    region = matrix[y_min:y_max, x_min:x_max]

    flat_indices = np.argsort(region.ravel())[::-1]  # Sort in descending order

    vals = region.ravel()[flat_indices]
    positions = np.column_stack(np.unravel_index(flat_indices, region.shape))
    return [Position(y=positions[i][0] + y_min, x=positions[i][1] + x_min) for i in range(sum(vals >= min_energy))]

def max_in_kxk_region(matrix: np.ndarray[int], p: Position, k: int = 5) -> Position:
    if k % 2 == 0:
        raise Exception('Only odd value possible')
    k1, k2 = k // 2, k // 2 + 1

    y_min, y_max = max(0, p.y - k1), min(24, p.y + k2)
    x_min, x_max = max(0, p.x - k1), min(24, p.x + k2)
    region = matrix[y_min:y_max, x_min:x_max]

    max_value = region.max()

    # Get all coordinates of max_value
    max_positions = np.argwhere(region == max_value)

    # Convert local indices to global indices
    global_positions = [(y_min + r, x_min + c) for r, c in max_positions]

    y, x = min(global_positions, key=lambda pos: ((pos[0] - p.y) ** 2 + (pos[1] - p.x) ** 2, pos[0], pos[1]))
    return Position(y=y, x=x)

def word_predictions(terrain: np.ndarray[int], step: int,terrain_speed: int, terrain_dir: int) -> List[Tuple[int, np.ndarray[int]]]:
    # Knowing the terrain map of the current turn it does calculations how the map will look like next turns until
    # the end of the match. It does not show all maps in all steps but only during the change of the step
    end_turn = [102, 203, 304, 405, 506]
    res = [(step, terrain)]
    if terrain_speed == 0 or terrain_dir == 0:
        for i in range(step + 1, step + 101):
            if i in end_turn:
                res.append((i, terrain))

        return res

    matrix = terrain.copy()
    for i in range(step + 1, step + 101):
        if i in end_turn:
            res.append((i, matrix))
            break

        if is_world_changed(i, terrain_speed):
            matrix = change_terrain(matrix, matrix, terrain_dir)
            res.append((i, matrix))

    return res

def future_tiles(wp: List[Tuple[int, np.ndarray[int]]], p: Position) -> List[int]:
    """Knowing `word_predictions`, this function tells the future values of tiles in that position
    steps are relative to staring step in word_prediction
    """
    data = [(s, m[p.y, p.x]) for s, m in wp[::-1]]

    res = []
    prev_step, prev_value = data.pop()
    while data:
        curr_step, curr_value = data.pop()
        res.extend([prev_value] * (curr_step - prev_step))
        prev_step, prev_value = curr_step, curr_value

    res.append(prev_value)
    return res

def energy_map_total(mi_energy: np.ndarray[int], mi_terrain: np.ndarray[int], nebula_reduction: int, val: int = 3) -> np.ndarray[int]:
    energy = mi_energy.copy()

    energy[mi_energy == 50] = val  # To prioritize moving. 50 is hardcoded default value. See map_maker

    # Reduce the energy of Nebula tiles
    energy[mi_terrain == TerrainTile.NEBULA.value] -= nebula_reduction

    # Penalize Asteroid tiles. Not really needed in most cases, but sometimes I am not checking terrain.
    energy[mi_terrain == TerrainTile.ASTEROID.value] = -99

    return energy

# ------------------
def _apply_kernel(y: int, x: int, kernel: np.ndarray[int], vision: np.ndarray[int]):
    # This function is only helper to do iteration
    n = vision.shape[0]
    k = kernel.shape[0]
    assert k % 2 == 1, "Kernel size must be odd"

    half_k = k // 2
    y1, y2 = max(0, y - half_k), min(n, y + half_k + 1)
    x1, x2 = max(0, x - half_k), min(n, x + half_k + 1)

    k_y_start, k_y_end = half_k - (y - y1), half_k + (y2 - y)
    k_x_start, k_x_end = half_k - (x - x1), half_k + (x2 - x)

    res = np.zeros_like(vision)
    res[y1:y2, x1:x2] = vision[y1:y2, x1:x2] + kernel[k_y_start:k_y_end, k_x_start:k_x_end]
    return res > 0

def build_vision_kernel(sensor_range: int) -> np.ndarray[int]:
    """Builds a kernel of vision for a unit based on the sensor range"""
    k = 2 * sensor_range + 1
    kernel = np.zeros((k, k), dtype=int)
    for n in range(1, sensor_range + 2):
        l1, l2 = n - 1, n
        y_s, x_s = max(0, sensor_range - l1), max(0, sensor_range - l1)
        y_e, x_e = min(24, sensor_range + l2), min(24, sensor_range + l2)
        kernel[y_s:y_e, x_s:x_e] += 1

    kernel[sensor_range, sensor_range] = 10
    return kernel


def iteration(is_relics_possible: np.ndarray[bool], nebula_vision: np.ndarray[int], vision_kernel: np.ndarray[int]) -> Tuple[int, Position]:
    """Knowing the map of possible relics, information about nebula (boolean matrix multiplied by vision reduction)
    and vision kernel, it does convolution at every point taking into an account the lost vision from the nebula
    and select the position which will allow you to see the most number of relics (and how many of them).
    This also modifies `is_relics_possible` may by removing currently seen positions
    """
    data, n = [], nebula_vision.shape[0]
    best_num, y_, x_ = 0, 0, 0
    for y in range(n):
        for x in range(n):
            mask = _apply_kernel(y, x, vision_kernel, nebula_vision)
            num = is_relics_possible[mask].sum()
            if num > best_num:
                best_num, y_, x_ = num, y, x
            data.append(num)

    mask = _apply_kernel(y_, x_, vision_kernel, nebula_vision)
    is_relics_possible[mask] = False
    return best_num, Position(y=y_, x=x_)