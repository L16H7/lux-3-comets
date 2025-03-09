from collections import defaultdict
from logging import Logger
from typing import Tuple, List, Set, Dict

import numpy as np

from lib.data_classes import TerrainTile, Move, MoveResult, MapInfo, Position, GameParam, Unit, EnergyPos
from lib.helper import distance, energy_map_total, mark_nxn


# TODO: this works only for unique positions in `pos_finish`. So you can't have same position with multiple different energies
def _best_path_multiple_targets(
        pos_start: EnergyPos,
        pos_finish: List[EnergyPos],
        energy_map: np.ndarray[int],
        obstacle_map: np.ndarray[bool],
        fragments: np.ndarray[float],
        move_cost: int,
        approximate_after: int = 102,  # The turn will end in this number of steps. No need to find the path if we can't reach
) -> Dict[EnergyPos, Tuple[List[EnergyPos], float]]:
    """Have a matrix of energy and obstacles and starting position of the robot with its energy, it finds the
    best actions a robot has to take to reach end position with at least energy_end_required.
    If no, returns empty path. This path has position and energy at each step.

    Energy_map should have energy information at each position (even the ones you do not see). This is outside
    of this function to decide how do you impute them. Information about NEBULA should already be merged in it.
    Obstacle_map is binary map which marks ASTEROID as true. In similar approach, it is outside of this function
    how you impute unseen values

    This function returns output to all `pos_finish` to which you can reach. If you can't reach a position, it is not
    in the list.
    To speed-up the execution you can approximate the movement after some position. Speedup is significant for long
    series of moves

    Takes O(n^2 * max_energy)
    """
    def backtrack(e: int, p: Position) -> List[Position]:
        y_, x_, e_, path = p.y, p.x, e, []
        while (y_, x_, e_) in backtracking:
            path.append(EnergyPos(y=y_, x=x_, e=e_))
            y_, x_, e_ = backtracking[(y_, x_, e_)]
        path.append(pos_start)
        return path[::-1]

    Y, X = energy_map.shape
    directions, max_energy = [0, 1, 0, -1, 0], 400
    frontier = [(pos_start.e, 0, pos_start)]  # (energy, points, position)
    backtracking = {}

    destinations: Dict[Position, int] = {ep.pos(): ep.e for ep in pos_finish}
    results: Dict[EnergyPos, Tuple[List[EnergyPos], int]] = {}

    pos_to_energy = np.full_like(energy_map, -100)
    pos_to_energy[pos_start.y, pos_start.x] = pos_start.e
    is_start = True # Is first turn. To be able to move if you already stand on the obstacle

    gamma = 1  # To penalize uncertainty of moving to a fragment later
    while frontier:
        gamma *= 0.9
        new_frontier_map, pos_to_points = defaultdict(lambda: -100), {}
        frontier = sorted(frontier, key=lambda v: (-v[0], -v[1]))
        for energy, points, p in frontier:
            if p.pos() in destinations and energy >= destinations[p.pos()]:
                results[EnergyPos(y=p.y, x=p.x, e=destinations[p.pos()])] = (backtrack(energy, p), points)

                del destinations[p.pos()]
                if not destinations:
                    return results

            if obstacle_map[p.y, p.x] and not is_start:
                # I use it after the destination check to be able to reach position even if it is blocked at current
                # time (as might be unblocked later)
                continue

            if energy >= move_cost:
                for i in range(4):
                    p_nxt = Position(y=p.y + directions[i], x=p.x + directions[i + 1])
                    if 0 <= p_nxt.y < Y and 0 <= p_nxt.x < X:
                        new_energy = min(max_energy, energy - move_cost + int(energy_map[p_nxt.y, p_nxt.x]))
                        if new_energy > pos_to_energy[p_nxt.y, p_nxt.x]:
                            pos_to_energy[p_nxt.y, p_nxt.x] = new_energy
                            new_frontier_map[p_nxt] = max(new_frontier_map[p_nxt], new_energy)
                            pos_to_points[p_nxt] = points + fragments[p_nxt.y, p_nxt.x] * gamma
                            backtracking[(p_nxt.y, p_nxt.x, new_energy)] = (p.y, p.x, energy)

            if energy < max_energy and energy_map[p.y, p.x] > 0:
                new_energy = min(max_energy, energy + int(energy_map[p.y, p.x]))
                if new_energy > pos_to_energy[p.y, p.x]:
                    pos_to_energy[p.y, p.x] = new_energy
                    new_frontier_map[p] = max(new_frontier_map[p], new_energy)
                    backtracking[(p.y, p.x, new_energy)] = (p.y, p.x, energy)

        frontier, is_start = [(e, pos_to_points.get(p, 0), p) for p, e in new_frontier_map.items()], False
        if not frontier:
            return results

        approximate_after -= 1
        if approximate_after == -1:
            # Reached a step after which you do not want to calculate exact solution and approximation is enough
            # This is needed to reduce the time. And it dramatically reduces it even for a pretty high approximation
            # like 50. For long sequences it can be like 10x faster
            for pos_want, energy_want in destinations.items():
                min_estimate, best_pos, best_points, best_energy = float('inf'), None, 0, 0
                for energy, points, pos in frontier:
                    d = distance(pos_want, pos)  # number of turns to reach if you can move anyway you want

                    # You might also need to recharge for:
                    # energy - d * move_cost + recharge_num * 5 >= energy_want
                    # Here I assume that I will find tile with energy at least 5 to recharge
                    recharge_num = (energy_want - energy + d * move_cost) // 5
                    recharge_num = max(0, recharge_num)

                    estimate_num = int((d + recharge_num) * 1.2)
                    # have done some evaluations and this estimate is pretty accurate (diff < 5 turns on 50 horizon)
                    if estimate_num < min_estimate:
                        min_estimate = estimate_num
                        best_pos = pos
                        best_points = points
                        best_energy = energy

                curr_path = backtrack(best_energy, best_pos)
                curr_path.extend([EnergyPos(y=-1, x=-1, e=energy_want)] * min_estimate)
                results[EnergyPos(y=pos_want.y, x=pos_want.x, e=energy_want)] = (curr_path, best_points)

            return results

    return results


def _path_to_moves(move_path: List[EnergyPos]) -> List[Move]:
    res = []
    for i in range(1, len(move_path)):
        p1, p2 = move_path[i - 1].pos(), move_path[i].pos()
        if p2.y == -1 and p2.x == -1:
            move = Move.UNKNOWN    # If we just estimated the move
        elif p1.x == p2.x and p1.y == p2.y:
            move = Move.NONE
        elif p2.x - p1.x == 1:
            move = Move.R
        elif p1.x - p2.x == 1:
            move = Move.L
        elif p2.y - p1.y == 1:
            move = Move.D
        elif p1.y - p2.y == 1:
            move = Move.U
        else:
            raise Exception(f'Not possible: {p1} {p2}')
        res.append(move)
    return res


"""Responsible for finding the best move for a unit if he wants to move from one position to another"""
class PathFinder:

    def __init__(self, cfg: GameParam):
        self.logger: Logger = None

        self._obstacle = np.zeros((24, 24), dtype=bool)
        self._energy = np.zeros((24, 24))
        self._units_enemy: Set[Unit] = set()

        # Should not worry about unoccupied. Occupied is fine as well.
        # Maybe make unoccupied worth slightly more
        self._fragments_cost: np.ndarray[bool] = np.zeros((24, 24), bool)

        self._cfg = cfg

    def _add_enemy_to_energy(self, pos_from: EnergyPos):
        """For every enemy unit which can hit me next turn, penalize the positions that unit can be. Unit can hit me if:
        - his energy >= my energy
        - he is located at d<=2
        To penalize, I put 50% of my energy as a penalty on that tile.
        """
        penalty_map = np.zeros((24, 24), dtype=bool)
        for u in self._units_enemy:
            if u.energy >= pos_from.e and distance(pos_from.pos(), u.pos()) <= 2:
                mark_nxn(penalty_map, u.pos(), 3)

        return self._energy.copy() - penalty_map.astype(int) * (pos_from.e // 2)

    def update_map(self, m: MapInfo, units_enemy: Set[Unit], units_my: Set[Unit], logger: Logger):
        self.logger = logger
        self._units_enemy = units_enemy

        my_positions = {u.pos() for u in units_my}
        self._fragments_cost = np.zeros((24, 24), float)
        for f in m.fragments:
            if f.probability == 0:
                continue
            if f.pos() not in my_positions:
                self._fragments_cost[f.y, f.x] = 1
            else:
                self._fragments_cost[f.y, f.x] = 0.8  # Slightly penalize occupied positions

        self._energy = energy_map_total(m.energy, m.terrain, self._cfg.nebula_energy_reduction)
        self._obstacle = (m.terrain == TerrainTile.ASTEROID.value)

        self.logger.debug('My energy understanding')
        self.logger.debug(f"\n{self._energy}")

    def move_from_to(self, pos_1: EnergyPos, pos_2: List[EnergyPos], approximate_after: int = 102, is_debug: bool = True) -> Dict[EnergyPos, MoveResult]:
        paths = _best_path_multiple_targets(
            pos_start=pos_1,
            pos_finish=pos_2,
            energy_map=self._add_enemy_to_energy(pos_1),
            obstacle_map=self._obstacle,
            fragments=self._fragments_cost,
            move_cost=self._cfg.unit_move_cost,
            approximate_after=approximate_after,
        )

        res = {}
        for p in pos_2:
            debug_pref = f'Move from {pos_1} to {p}:'
            if p not in paths:
                # Can't reach there
                res[p] = MoveResult(move=Move.NONE, num_moves=105, energy=-10, points=-5, path=[], path_str='')
                if is_debug:
                    self.logger.debug(f'{debug_pref} not possible')
            else:
                path, points = paths[p]
                num_moves = len(path) - 1
                if num_moves == 1 and self._obstacle[p.y, p.x]:
                    # Standing right next to asteroid and need to go there. Just wait
                    if is_debug:
                        self.logger.debug(f'{debug_pref} need to wait as standing next to asteroid')
                    res[p] = MoveResult(move=Move.NONE, num_moves=1, energy=max(0, pos_1.e), points=points, path=[], path_str='')
                elif num_moves == 0:
                    if is_debug:
                        self.logger.debug(f'{debug_pref} already at position)')
                    res[p] = MoveResult(move=Move.NONE, num_moves=0, energy=max(0, pos_1.e), points=points, path=[], path_str='')
                else:
                    moves = _path_to_moves(path)
                    moves_str = ''.join(str(m) for m in moves)
                    if is_debug:
                        self.logger.debug(f'{debug_pref} need {num_moves} moves. Will have ~{path[-1].e} energy and get {points} points. Moves: `{moves_str}`')
                    res[p] = MoveResult(
                        move=moves[0],
                        num_moves=num_moves,
                        energy=max(0, path[-1].e),
                        points=points,
                        path=path,
                        path_str=moves_str,
                    )
        return res
