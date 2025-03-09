import logging
from typing import Dict, Any, List, Set

import numpy as np

from lib.data_classes import GameParam, Unit, TerrainTile, MapInfo, Position
from lib.helper import mark_nxn, is_shift_in_dir, is_move_detected, terrain_matrix_to_strings


def print_matrices(M1, M2) -> List[str]:
    s1, s2 = terrain_matrix_to_strings(M1), terrain_matrix_to_strings(M2)
    res = ['          PREV                            CURR            ']
    res.extend([f'{l1}          {l2}' for l1, l2 in zip(s1, s2)])
    return res



"""
Responsible for learning and maintaining information about the config of the word
This does not include map, units. and include both information which is known and learnt
https://discord.com/channels/753650408806809672/799501005377110026/1331328100407050322
"""
class GameConfig:

    def __init__(self, cfg: Dict[str, Any]):
        # Cfg is some parts of config shared with you
        # https://discord.com/channels/753650408806809672/860995476250034186/1318829350937952379
        self.cfg = GameParam(
            max_units=cfg['max_units'],
            unit_move_cost=cfg['unit_move_cost'],
            unit_sap_cost=cfg['unit_sap_cost'],
            unit_sap_range=cfg['unit_sap_range'],
            unit_sensor_range=cfg['unit_sensor_range'],
            spawn_rate=3,

            # Learning
            nebula_vision_reduction=0,      # This is the lower range of the estimate (real_value >= this one)
            nebula_energy_reduction=-10,    # To encourage estimation. So you really want to go and try
            terrain_speed=0,                # 0 means you do not know
            terrain_dir=0,                  # 0 means you do not know, 1 and -1 actual directions
            unit_sap_dropoff_factor=1.1,    # To encourage first shoot and get the estimate
        )

        self._logger: logging.Logger = None
        self._map_info: MapInfo = None      # Here do not use terrain and always use terrain_tmp
        self._prev_units_my: Dict[int, Unit] = {}  # Needed only for nebula energy estimation
        self._prev_units_enemy: Dict[int, Unit] = {}  # Needed only for sap dropoff estimation
        self._prev_shots: List[Position] = []  # Needed only for sap dropoff estimation

        self._prev_vision_single: np.ndarray[bool] = np.zeros((24, 24), bool)
        self._prev_terrain: np.ndarray[int] = np.zeros((24, 24), int)

    def is_nebula_energy_reduction_estimated(self) -> bool:
        return self.cfg.nebula_energy_reduction in [0, 1, 2, 3, 5, 25]

    def is_unit_sap_dropoff_factor_estimated(self) -> bool:
        return self.cfg.unit_sap_dropoff_factor in [0.25, 0.5, 1]

    def _vision_power(self, my_units: Set[Unit]) -> np.ndarray[int]:
        matrix = np.zeros((24, 24), dtype=int)
        for p in my_units:
            for n in range(1, self.cfg.unit_sensor_range + 2):
                l1, l2 = n - 1, n
                y_s, x_s = max(0, p.y - l1), max(0, p.x - l1)
                y_e, x_e = min(24, p.y + l2), min(24, p.x + l2)
                matrix[y_s:y_e, x_s:x_e] += 1
        return matrix

    def _vision_mask(self, my_units: Set[Unit]) -> np.ndarray[bool]:
        matrix = np.zeros((24, 24), dtype=bool)
        for p in my_units:
            l1, l2 = self.cfg.unit_sensor_range, self.cfg.unit_sensor_range + 1
            y_s, x_s = max(0, p.y - l1), max(0, p.x - l1)
            y_e, x_e = min(24, p.y + l2), min(24, p.x + l2)
            matrix[y_s:y_e, x_s:x_e] = True
        return matrix

    def _sap_dropoff_update(self, e_prv: int, e_nxt: int, tile_e: int, cnt: int = 1):
        if self.is_unit_sap_dropoff_factor_estimated():
            return

        val_estimate, res = self._calculate_sap_dropoff(e_prv=e_prv, e_nxt=e_nxt, tile_e=tile_e) / cnt, None
        for val in [0.25, 0.5, 1]:
            if abs(val_estimate - val) / val < 0.2:
                res = val

        if res:
            self._logger.info(f'Estimated sap dropoff factor for first time: {res}')
            self.cfg.unit_sap_dropoff_factor = res
        else:
            self._logger.info(f'Tried to estimate sap dropoff, but was way off: {val_estimate}')

    def _calculate_sap_dropoff(self, e_prv: int, e_nxt: int, tile_e: int) -> float:
        # Using math, you can calculate the following formula
        dx = e_prv - e_nxt + tile_e - self.cfg.unit_move_cost
        return float(dx) / self.cfg.unit_sap_cost

    def _learn_nebula_energy_reduction(self, units_my: List[Unit]) -> None:
        # TODO: The only thing it does not currently count is when enemy ships comes super close to you
        prev_units = {}

        for u in units_my:
            if self.is_nebula_energy_reduction_estimated():
                return

            prev_units[u.id] = u
            if self._map_info.terrain_tmp[u.y, u.x] == TerrainTile.NEBULA.value and u.id in self._prev_units_my:
                # Unit existed previously and now it is on NEBULA
                move_cost = self.cfg.unit_move_cost if u.pos() != self._prev_units_my[u.id].pos() else 0

                # Previously it was possible that you do not see value on the tile so logic was more complicated
                # now it is no longer the case.
                # equation is the following:  energy_prev + energy[u.y, u.x] - unit_move_cost - X = energy_curr
                # If z = energy_prev + energy[u.y, u.x] - unit_move_cost
                # max(z - X, 0) = energy_curr
                z = self._prev_units_my[u.id].energy + self._map_info.energy[u.y, u.x] - move_cost
                if u.energy > 0 and (z >= u.energy): # second is needed when nebula suddenly moves to you
                    self.cfg.nebula_energy_reduction = z - u.energy
                    self._logger.info(f'Nebular energy reduction updated to: {self.cfg.nebula_energy_reduction}')

        self._prev_units_my = prev_units

    def _learn_unit_sap_dropoff_factor(self, units_enemy: List[Unit]) -> None:
        if self.is_unit_sap_dropoff_factor_estimated():
            return

        # See all positions where I shot and mark areas around it (and count). Also mark direct hits.
        # I ignore direct hits. It is possible to make a formula for it as well, but probably no need to.
        map_non_direct_shots = np.zeros((24, 24), int)
        map_direct_shots = np.zeros((24, 24), bool)
        for p in self._prev_shots:
            shot_curr = np.zeros((24, 24), int)
            mark_nxn(shot_curr, p, 3)
            map_non_direct_shots += shot_curr.astype(int)
            map_direct_shots[p.y, p.x] = True

        for u in units_enemy:
            if u.id not in self._prev_units_enemy:
                continue  # Can't do anything as I have no idea what was his energy before

            cnt = map_non_direct_shots[u.y, u.x]
            if cnt == 0:
                continue  # Unit was not affected by non-direct shots

            if map_direct_shots[u.y, u.x]:
                continue  # We ignore direct shots

            e_prv = self._prev_units_enemy[u.id].energy
            e_nxt = u.energy
            tile_e = self._map_info.energy[u.y, u.x]

            if e_nxt <= 3:
                # The order of actions is:
                # move all units, do all saps, update energy based on the tiles. So there can be incorrect calculation
                # Example: your e_prev 50 and become 0. Tile_e is -24 (due to nebula) and move_cost is 2. Your sap
                # cost is 43. You have no idea whether your sap mult is 0.5 or 1.
                continue

            if self._map_info.terrain_tmp[u.y, u.x] == TerrainTile.NEBULA.value:
                if not self.is_nebula_energy_reduction_estimated():
                    # Do not know how much enemy ship lost from nebular, so can't estimate this
                    continue

                tile_e -= self.cfg.nebula_energy_reduction

            self._sap_dropoff_update(e_prv, e_nxt, tile_e, cnt)

    def _terrain_speed_helper(self, step: int):
        is_move = is_move_detected(self._prev_vision_single, self._map_info.vision_single, self._prev_terrain, self._map_info.terrain_tmp)
        if is_move:
            if step in [8, 15]:
                self.cfg.terrain_speed = 0.15
            elif step in [11, 31]:
                self.cfg.terrain_speed = 0.1
            elif step == 21:
                self.cfg.terrain_speed = 0.05
            elif step == 41:
                self.cfg.terrain_speed = 0.025

            for v in print_matrices(self._prev_terrain, self._map_info.terrain_tmp):
                self._logger.debug(v)

            p1 = is_shift_in_dir(self._prev_vision_single, self._map_info.vision_single, self._prev_terrain, self._map_info.terrain_tmp, True)
            p2 = is_shift_in_dir(self._prev_vision_single, self._map_info.vision_single, self._prev_terrain, self._map_info.terrain_tmp, False)
            self._logger.info(f"Probability of up/right {p1:.3f}, of down/left {p2:.3f}")

            if max(p1, p2) < 0.8:
                self._logger.critical(f"Very wrong belief of move direction")
                return

            if abs(p1 - p2) < 0.1:
                self._logger.warning(f"Both directions are probable. Can not decide what is better")
                return

            if p1 > p2:
                self.cfg.terrain_dir = 1
            else:
                self.cfg.terrain_dir = -1

    def _learn_terrain_speed(self, step: int):
        # Official numbers are +-[0.025, 0.05, 0.1, 0.15].
        if self.cfg.terrain_dir != 0:
            return

        if step > 3:
            self._terrain_speed_helper(step)

        self._prev_vision_single = self._map_info.vision_single.copy()
        self._prev_terrain = self._map_info.terrain_tmp.copy()

    def _learn_nebula_vision_reduction(self, my_units: Set[Unit], vision: np.ndarray[bool]):
        mask_vision = self._vision_mask(my_units) & vision
        power_vision = self._vision_power(my_units)
        curr_min_estimate = np.where(mask_vision, 0, power_vision).max()

        if curr_min_estimate > self.cfg.nebula_vision_reduction:
            self.cfg.nebula_vision_reduction = curr_min_estimate
            self._logger.info(f"Estimate of nebula vision reduction was increased to {self.cfg.nebula_vision_reduction}")

    def add_shots(self, shots: List[Position]):
        # This should be called at the end of the turn. Needed only for sap dropoff estimate
        self._prev_shots = shots

    def update(self, units_my: List[Unit], units_enemy: List[Unit], map_info: MapInfo, step: int, logger: logging.Logger):
        self._logger = logger
        self._map_info = map_info

        self._learn_nebula_energy_reduction(units_my)
        self._learn_unit_sap_dropoff_factor(units_enemy)
        self._learn_terrain_speed(step)
        self._learn_nebula_vision_reduction(units_my, map_info.vision)

        self._prev_units_enemy = {u.id: u for u in units_enemy}

        speed = 'unknown' if self.cfg.terrain_speed == 0 else str(self.cfg.terrain_speed)
        move_dir = 'Up-Right' if self.cfg.terrain_dir == 1 else 'Down-Left' if self.cfg.terrain_dir == -1 else 'unknown'

        self._logger.info(f"Current estimate of `nebula_vision_reduction`: {self.cfg.nebula_vision_reduction}")
        self._logger.info(f"Current estimate of `nebula_energy_reduction`: {self.cfg.nebula_energy_reduction}")
        self._logger.info(f"Current estimate of `terrain_speed`          : {speed}")
        self._logger.info(f"Current estimate of `terrain_dir`            : {move_dir}")
        self._logger.info(f"Current estimate of `unit_sap_dropoff_factor`: {self.cfg.unit_sap_dropoff_factor}")
