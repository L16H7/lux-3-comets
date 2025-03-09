from logging import Logger
from typing import List, Set
import random
import numpy as np
from collections import defaultdict

from lib.data_classes import Unit, Task, TaskType, GameParam, Position, MapInfo, EnergyPos as EP, TerrainTile
from lib.helper import mask_other_side, best_in_kxk_region, iteration, build_vision_kernel, energy_map_total, is_shot_possible

np.set_printoptions(threshold=10**9)
np.set_printoptions(linewidth=10**9)
np.set_printoptions(precision=3, suppress=True)


class TaskGenerator:

    def __init__(self, map_info: MapInfo, team_id: int, cfg: GameParam, step_in_match: int, logger: Logger):
        self.map_info = map_info
        self.team_id = team_id
        self.cfg = cfg
        self.step_in_match = step_in_match
        self.logger = logger
        self.energy_combined = energy_map_total(self.map_info.energy, self.map_info.terrain, self.cfg.nebula_energy_reduction)

    @staticmethod
    def _get_waiting_positions(team_id: int, vision: int) -> List[EP]:
        # Here in energy position instead of energy I pass importance. This is just to not create another class
        # Importance here is just to prioritize going to positions which are further (and thus you can instigate more)
        if vision == 1:
            return [
                EP(2, 20, 5), EP(20, 2, 5), EP(8, 14, 4), EP(14, 8, 4),
                EP(2, 15, 3), EP(14, 3, 3), EP(8, 9, 3), EP(15, 5, 3), EP(3, 17, 3), EP(9, 11, 3),
                EP(2, 9, 2), EP(8, 3, 2), EP(8, 6, 2), EP(2, 12, 2), EP(2, 3, 1),
            ] if team_id == 0 else [
                EP(3, 21, 5), EP(21, 3, 5), EP(9, 15, 4), EP(15, 9, 4),
                EP(8, 21, 3), EP(20, 9, 3), EP(14, 15, 3), EP(18, 8, 3), EP(6, 20, 3), EP(12, 14, 3),
                EP(14, 21, 2), EP(20, 15, 2), EP(17, 15, 2), EP(11, 21, 2), EP(20, 21, 1)
            ]
        elif vision == 2:
            return [
                EP(20, 2, 5), EP(2, 20, 5),
                EP(14, 7, 4), EP(7, 14, 4), EP(15, 2, 4), EP(2, 15, 4),
                EP(9, 9, 3), EP(10, 2, 3), EP(2, 10, 3),
                EP(7, 4, 2), EP(4, 7, 2), EP(2, 2, 1)
            ] if team_id == 0 else [
                EP(21, 3, 5), EP(3, 21, 5),
                EP(16, 9, 4), EP(9, 16, 4), EP(21, 8, 4), EP(8, 21, 4),
                EP(14, 14, 3), EP(21, 13, 3), EP(13, 21, 3),
                EP(19, 16, 2), EP(16, 19, 2), EP(21, 21, 1)
            ]
        elif vision == 3:
            return [
                EP(19, 3, 5), EP(3, 19, 5), EP(13, 7, 4), EP(7, 13, 4), EP(13, 3, 4), EP(3, 13, 4),
                EP(8, 8, 3), EP(7, 3, 3), EP(3, 7, 3), EP(3, 3, 2)
            ] if team_id == 0 else [
                EP(20, 4, 5), EP(4, 20, 5), EP(16, 10, 4), EP(10, 16, 4), EP(20, 10, 4), EP(10, 20, 4),
                EP(15, 15, 3), EP(20, 16, 3), EP(16, 20, 3), EP(20, 20, 2),
            ]
        else:
            return [
                EP(19, 4, 5), EP(4, 19, 5), EP(11, 7, 4), EP(7, 11, 4), EP(10, 3, 3), EP(3, 10, 3), EP(3, 3, 2)
            ] if team_id == 0 else [
                EP(19, 4, 5), EP(4, 19, 5), EP(16, 12, 4), EP(12, 16, 4), EP(20, 13, 3), EP(13, 20, 3), EP(20, 20, 2),
            ]

    @staticmethod
    def _get_default_attacking_pos(team_id: int) -> List[Position]:
        if team_id == 0:
            return [Position(y=14, x=8), Position(y=8, x=14), Position(y=20, x=2), Position(y=2, x=20)]
        return [Position(y=15, x=9), Position(y=9, x=15), Position(y=21, x=3), Position(y=3, x=21)]

    # TaskType.WAIT
    def _gen_wait(self) -> List[Task]:
        if self.map_info.is_relic_new_found or not self.map_info.is_relic_search_this_match or self.step_in_match >= 50:
            return []

        tasks = []
        for p in self._get_waiting_positions(self.team_id, self.cfg.unit_sensor_range):
            positions = best_in_kxk_region(self.energy_combined, p, k=5)
            if not positions:
                continue

            tasks.append(Task(
                type=TaskType.WAIT,
                y=positions[0].y, x=positions[0].x,
                suggested_energy=self.cfg.unit_move_cost,
                score=p.e * 10,     # This is not an energy, but rather importance here
                y_real=p.y,
                x_real=p.x,
            ))
        return tasks

    # TaskType.SEARCH_RELICS
    def _gen_search_relics(self, max_tasks: int) -> List[Task]:
        can_search_for_relics = self.step_in_match >= 50 and not self.map_info.is_relic_new_found
        if not can_search_for_relics:
            return []   # Does not make sense to search for them, so do not create any tasks

        tasks, is_relics_possible, cnt = [], self.map_info.relics_possible.copy(), self.map_info.relics_possible.sum()

        nebular_reduction = -((self.map_info.terrain == TerrainTile.NEBULA.value) * self.cfg.nebula_vision_reduction)
        is_relics_possible[mask_other_side(self.team_id)] = 0
        kernel = build_vision_kernel(self.cfg.unit_sensor_range)

        for _ in range(max_tasks):
            score, p = iteration(is_relics_possible, nebular_reduction, kernel)
            if score == 0:
                break

            tasks.append(Task(
                type=TaskType.SEARCH_RELICS,
                y=p.y, x=p.x,
                suggested_energy=self.cfg.unit_move_cost * 3,
                score=score / cnt,  # Probability that you will find a relic in that area
            ))

        return tasks

    # TaskType.SEARCH_FRAGMENTS
    def _gen_search_fragments(self) -> List[Task]:
        data = self.map_info.fragments_possible.copy()
        data[mask_other_side(self.team_id)] = 0
        for f in self.map_info.fragments:
            data[f.y, f.x] = 0

        y_indices, x_indices = np.where(data)
        return [Task(
            type=TaskType.SEARCH_FRAGMENTS,
            y=y, x=x,
            suggested_energy=self.cfg.unit_move_cost * 3,
        ) for y, x in zip(y_indices, x_indices)]

    # TaskType.COLLECT_FRAGMENT_MY
    def _gen_collect_fragments_my(self, units: Set[Unit]) -> List[Task]:
        # Also ignore positions which are occupied by enemy units
        tasks, units_pos = [], {u.pos() for u in units}
        for f in self.map_info.fragments:
            if not f.is_team(self.team_id):
                continue

            if f.probability == 1:
                tasks.append(Task(
                    type=TaskType.COLLECT_FRAGMENT_MY,
                    y=f.y, x=f.x,
                    suggested_energy=-100
                ))
            elif f.probability != 1:
                # If you are not sure if this is a fragment
                if f.pos() in units_pos:
                    # Unit is already on the fragment then with some probability ignore this task which will force
                    # unit to move from there and do something else which will make some fluctuations in positions
                    if random.random() < 0.3:
                        tasks.append(Task(
                            type=TaskType.COLLECT_FRAGMENT_MY,
                            y=f.y, x=f.x,
                            suggested_energy=self.cfg.unit_move_cost * 3
                        ))
                else:
                    tasks.append(Task(
                        type=TaskType.COLLECT_FRAGMENT_MY,
                        y=f.y, x=f.x,
                        suggested_energy=self.cfg.unit_move_cost * 3
                    ))

        return tasks

    # TaskType.ATTACK
    def _gen_attack(self, units_enemy: Set[Unit]) -> List[Task]:
        tasks, k = [], 2 * self.cfg.unit_sap_range + 1

        # Add default tasks just in case you do not have what to do
        energy_need = self.cfg.unit_move_cost * 5 + self.cfg.unit_sap_cost
        energy_copy = self.energy_combined.copy()  # To not go there
        for u in units_enemy:
            energy_copy[u.y, u.x] = -1

        all_positions = {}
        for f in self.map_info.fragments:
            if f.team_id != self.team_id:
                for p_ in best_in_kxk_region(energy_copy, f.pos(), k=k):
                    all_positions[p_] = f

        for p_, f in all_positions.items():
            tasks.append(Task(
                type=TaskType.ATTACK,
                y=p_.y, x=p_.x,
                suggested_energy=energy_need,
                score=energy_copy[p_.y, p_.x],
                y_real=f.y,
                x_real=f.x,
            ))

        return tasks

    # TaskType.COLLECT_FRAGMENT_ENEMY
    def _gen_collect_fragments_enemy(self, units_my: Set[Unit], enemy_units: Set[Unit]) -> List[Task]:
        positions = {u.pos(): 0 for u in enemy_units}
        for u in enemy_units:
            positions[u.pos()] += u.energy

        fragments = []
        for f in self.map_info.fragments:
            if f.is_team(self.team_id):  # Fragment is our
                continue

            # Count how many shots my units which are in shooting distance to fragments
            cnt = 0
            for u in units_my:
                if is_shot_possible(u, f.pos(), self.cfg.unit_sap_range):
                    cnt += u.energy // self.cfg.unit_sap_cost

            if cnt > 0:
                fragments.append((f, cnt))

        tasks = []
        for f, cnt in fragments:
            energy = self.cfg.unit_sap_cost if f.pos() not in positions else positions[f.pos()]
            tasks.append(Task(
                type=TaskType.COLLECT_FRAGMENT_ENEMY,
                y=f.y, x=f.x,
                suggested_energy=min(100, energy),
                score=cnt,
            ))

        return tasks

    # TODO: NOT DONE. generate tasks only if my fragments are taken
    @staticmethod
    def _gen_retake_fragments_my() -> List[Task]:
        return []

    def generate_tasks(self,  units_my: Set[Unit], units_enemy: Set[Unit]):
        def log_tasks(new_tasks: List[Task], name: str):
            if new_tasks:
                self.logger.info(f'{new_tasks[0].type}  Generated tasks:')
                for t in new_tasks:
                    s = f', Real pos {Position(y=t.y_real, x=t.x_real)}' if t.x_real else ''
                    self.logger.info(f'   Pos: {t.pos()}{s}. Energy: {t.suggested_energy}. Score: {t.score}')
            else:
                self.logger.info(f'{name}  No tasks found')

        tasks = []

        add_tasks = self._gen_wait()
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.WAIT')

        add_tasks = self._gen_search_relics(max_tasks=16)
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.SEARCH_RELICS')

        add_tasks = self._gen_search_fragments()
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.SEARCH_FRAGMENTS')

        add_tasks = self._gen_collect_fragments_my(units_my)
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.COLLECT_FRAGMENT_MY')

        add_tasks = self._gen_attack(units_enemy)
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.ATTACK')

        add_tasks = self._gen_collect_fragments_enemy(units_my, units_enemy)
        tasks.extend(add_tasks)
        log_tasks(add_tasks, 'TaskType.COLLECT_FRAGMENT_ENEMY')

        #
        # add_tasks = self._gen_retake_fragments_my()
        # tasks.extend(add_tasks)
        # log_tasks(add_tasks, 'NOT IMPLEMENTED')

        # PathFinder does not allow tasks with the same position but different energy. So you get maximum energy
        pos_to_energy = defaultdict(int)
        for t in tasks:
            p = Position(y=t.y, x=t.x)
            pos_to_energy[p] = max(pos_to_energy[p], t.suggested_energy)

        return [t.update_energy(pos_to_energy[Position(y=t.y, x=t.x)]) for t in tasks]
