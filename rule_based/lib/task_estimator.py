from typing import List

import numpy as np

from lib.data_classes import Unit, Task, TaskType, Position, EnergyPos, ValueEstimate, TerrainTile, GameParam
from lib.mapmaker import MapMaker
from lib.path_finder import PathFinder
from lib.helper import word_predictions, future_tiles, energy_map_total

FRAGMENT_PROBABILITY = 0.7  # P(relic position) has a fragment = 1/5 (based on analysis). But I boost it to 0.7
VALUE_ZERO = ValueEstimate(points=0, other=0, energy=0, debug='Can not reach')


def _estimate_to_value(ve: ValueEstimate) -> int:
    return 1000000 * ve.points + 1000 * ve.other + ve.energy

def matrix_of_value_estimate_to_int(cost: List[List[ValueEstimate]]) -> np.ndarray[int]:
    return np.array([[_estimate_to_value(ve) for ve in row] for row in cost])


class TaskEstimator:
    def __init__(
            self,
            map_maker: MapMaker,
            path_finder: PathFinder,
            cfg: GameParam,
            step: int,
            step_in_match: int,
            unit: Unit,
            positions: List[EnergyPos]
    ):
        self.map_maker: MapMaker = map_maker
        self.path_finder: PathFinder = path_finder
        self.turns_left: int = 100 - step_in_match
        self.cfg = cfg
        self.step: int = step
        self.step_in_match: int = step_in_match
        self.unit: Unit = unit

        # TODO: micro optimization. Now this calculates 16 times for each player
        self.word_predictions = word_predictions(map_maker.map_info.terrain, step, self.cfg.terrain_speed, self.cfg.terrain_dir)

        self.energy_map = energy_map_total(map_maker.map_info.energy, map_maker.map_info.terrain, cfg.nebula_energy_reduction)

        self.move_results = self.path_finder.move_from_to(
            pos_1=EnergyPos(y=unit.y, x=unit.x, e=unit.energy),
            pos_2=positions,
            approximate_after=min(self.turns_left, 25),
            is_debug=False
        )

    def _steps_in_pos(self, num_moves: int, pos: Position) -> int:
        # Check when you can actually reach the position. `Num_moves` assumes that the word is not changing
        # So here we check if at time we reached the position we can move there. If it is occupied, we check
        # when will it be unoccupied
        if num_moves == 0:  # I am already there. Does not matter that position is occupied
            return self.turns_left

        tiles_in_future = future_tiles(self.word_predictions, pos)
        if num_moves > len(tiles_in_future):
            return self.turns_left - num_moves

        for i in range(num_moves, len(tiles_in_future)):
            if tiles_in_future[i] == TerrainTile.ASTEROID.value:
                continue

            if tiles_in_future[i] == TerrainTile.UNKNOWN.value:
                return self.turns_left - i - 5  # To somehow tell that we are not sure

            return self.turns_left - i

        return self.turns_left - len(tiles_in_future) - 1

    def _expected_energy(self, move_e: int, task_pos: Position, steps_stay: int) -> int:
        # Energy from staying on the tile. Do not look that much in the future as the terrain will change
        time_collect = min(10, steps_stay)
        e = self.energy_map[task_pos.y, task_pos.x] * time_collect
        e += move_e  # move_e is guaranteed to be [0, 400]
        return min(max(0, e), 400)  # to make it in [0, 400] range

    # TaskType.WAIT
    def _estimate_wait(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        return ValueEstimate(
            points=0,
            other=steps_in_pos * task.score,
            energy=self._expected_energy(move.energy, task.pos(), steps_in_pos)
        )

    # TaskType.SEARCH_RELICS
    def _estimate_search_relics(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        prob_relic = task.score  # Probability that you will find a relic in that area
        num_stays_in_pos = steps_in_pos + move.points
        return ValueEstimate(
            points=num_stays_in_pos * prob_relic * FRAGMENT_PROBABILITY,
            other=0,
            energy=move.energy,
            debug=f'Expected points {num_stays_in_pos}, prob relic {prob_relic}'
        )

    # TaskType.SEARCH_FRAGMENTS
    def _estimate_search_fragments(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        num_stays_in_pos = steps_in_pos + move.points
        return ValueEstimate(
            points=num_stays_in_pos * FRAGMENT_PROBABILITY,
            other=0,
            energy=move.energy,
            debug=f'move_point {move.points}'
        )

    # TaskType.COLLECT_FRAGMENT_MY
    def _estimate_collect_fragment_my(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        return ValueEstimate(
            points=steps_in_pos + move.points,
            other=0,
            energy=self._expected_energy(move.energy, task.pos(), steps_in_pos),
            debug=f'Turns left {self.turns_left}, num_moves {move.num_moves}, move_point {move.points}'
        )

    # TaskType.ATTACK
    def _estimate_attack(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        num_shots = self._expected_energy(move.energy, task.pos(), steps_in_pos) / self.cfg.unit_sap_cost
        return ValueEstimate(
            points=min(num_shots, steps_in_pos),
            other=0,
            energy=self._expected_energy(move.energy, task.pos(), 0)
        )

    # TaskType.COLLECT_FRAGMENT_ENEMY
    def _estimate_collect_fragment_enemy(self, task: Task) -> ValueEstimate:
        move = self.move_results[EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        steps_in_pos = self._steps_in_pos(move.num_moves, task.pos())
        if steps_in_pos <= 0:
            return VALUE_ZERO

        # Probability based on how many shots can units around fire. task.score is this count. Always > 0
        prob = 1 / (1 + np.exp((1 - task.score) / 2))
        return ValueEstimate(
            points=steps_in_pos * prob,
            other=0,
            energy=self._expected_energy(0, task.pos(), steps_in_pos)
        )

    def estimate_task(self, task: Task):
        if task.type == TaskType.COLLECT_FRAGMENT_MY:
            return self._estimate_collect_fragment_my(task)

        if task.type == TaskType.COLLECT_FRAGMENT_ENEMY:
            return self._estimate_collect_fragment_enemy(task)

        if task.type == TaskType.SEARCH_FRAGMENTS:
            return self._estimate_search_fragments(task)

        if task.type == TaskType.SEARCH_RELICS:
            return self._estimate_search_relics(task)

        if task.type == TaskType.ATTACK:
            return self._estimate_attack(task)

        if task.type == TaskType.RETAKE_FRAGMENT_MY:
            return self._estimate_attack(task)

        if task.type == TaskType.WAIT:
            return self._estimate_wait(task)

        raise Exception(f"Do not know how to estimate {task.type}")