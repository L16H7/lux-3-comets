import time
from logging import Logger
from typing import List, Dict, Set, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from lib.data_classes import Unit, Task, GameParam, EnergyPos, MoveResult, Position, TaskType
from lib.mapmaker import MapMaker
from lib.path_finder import PathFinder
from lib.task_estimator import TaskEstimator
from lib.task_estimator import matrix_of_value_estimate_to_int
from lib.task_generator import TaskGenerator
from lib.attacker import Attacker


class TaskAssigner:
    def __init__(
            self,
            map_maker: MapMaker,
            path_finder: PathFinder,
            cfg: GameParam,
            team_id: int,
            step: int,
            step_in_match: int,
            logger: Logger
    ):
        self.map_maker = map_maker
        self.path_finder = path_finder
        self.cfg = cfg
        self.step = step
        self.step_in_match = step_in_match

        self.TG = TaskGenerator(self.map_maker.map_info, team_id, cfg, step_in_match, logger)
        self.shots: List[Position] = []
        self.logger = logger

    def add_attacks(self, uid_to_task: Dict[int, Task], attacker: Attacker) -> Dict[int, Task]:
        attacks = attacker.adjust_attack(attacker.get_attack_suggestions(1), uid_to_task)
        if not attacks:
            return uid_to_task

        self.logger.info("Attacks")
        for uid, val in attacks.items():
            self.logger.info(f'   {uid:2d} -> {val}')

        new_uid_to_task = {}
        for uid, task in uid_to_task.items():
            if uid not in attacks:
                new_uid_to_task[uid] = task
                continue

            pos_attack = attacks[uid]
            task = task.add_attack(pos_attack)
            new_uid_to_task[uid] = task
            self.shots.append(Position(y=task.y_sap, x=task.x_sap))

        return new_uid_to_task

    def assign(self, units_my: Set[Unit], units_enemy: Set[Unit], attacker: Attacker) -> np.ndarray[int]:
        start_time = time.perf_counter()
        self.shots = []

        if not units_my:
            return np.zeros((16, 3), dtype=int)

        self.logger.debug(f"=============GENERATE TASKS=====================")
        tasks = self.TG.generate_tasks(units_my, units_enemy)
        if not tasks:
            return np.zeros((16, 3), dtype=int)

        units_my = sorted(list(units_my), key=lambda u: u.id)
        pos_to = [EnergyPos(y=t.y, x=t.x, e=t.suggested_energy) for t in tasks]

        matrix_estimates = []
        move_results: List[Dict[EnergyPos, MoveResult]] = [None] * 16

        for u in units_my:
            TE = TaskEstimator(
                map_maker=self.map_maker,
                path_finder=self.path_finder,
                cfg=self.cfg,
                step=self.step,
                step_in_match=self.step_in_match,
                unit=u,
                positions=pos_to,
            )
            move_results[u.id] = TE.move_results
            matrix_estimates.append([TE.estimate_task(t) for t in tasks])

        cost = matrix_of_value_estimate_to_int(matrix_estimates)
        row_ind, col_ind = linear_sum_assignment(-cost)

        uid_to_units = {u.id: u for u in units_my}
        uid_to_task = {units_my[int(u_ind)].id: tasks[int(t_ind)] for u_ind, t_ind in zip(row_ind, col_ind)}
        uid_to_estimate = {units_my[int(u_ind)].id: matrix_estimates[int(u_ind)][int(t_ind)] for u_ind, t_ind in zip(row_ind, col_ind)}

        uid_to_task = self.add_attacks(uid_to_task, attacker)

        self.logger.debug("Unit to tasks assignments:")
        res = [self.generate_cmd(uid, uid_to_task, uid_to_units, move_results, uid_to_estimate) for uid in range(16)]
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        self.logger.debug(f"Took {int(execution_time_ms)} ms to generate assignment")
        self.logger.debug(f"{res}")
        return np.array(res)

    def generate_cmd(self, uid: int, uid_to_task: Dict[int, Task], uid_to_units: Dict[int, Unit], move_results, estimates):
        if uid not in uid_to_task:
            return 0, 0, 0

        task = uid_to_task[uid]
        move = move_results[uid][EnergyPos(y=task.y, x=task.x, e=task.suggested_energy)]
        self.logger.debug(f"  UID {uid:2d}, task {task}. Will get {estimates[uid]}. Move info {move}")
        return self._generate_command(task, move, uid_to_units[uid])

    def _generate_command(self, task: Task, move: MoveResult, unit: Unit) -> Tuple[int, int, int]:
        if task.x_sap is not None:
            # calculate where to shoot based on task.x/y and unit x/y
            return 5, task.x_sap - unit.x, task.y_sap - unit.y

        unit_pos, task_pos = unit.pos(), task.pos()
        if unit_pos == task_pos:
            return 0, 0, 0  # already at location

        if move.move > 5:
            self.logger.critical(f'UID {unit.id} got {move}')
            return 0, 0, 0

        return move.move, 0, 0
