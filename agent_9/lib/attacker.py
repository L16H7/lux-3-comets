import logging
from typing import List, Dict, Tuple, Set

import numpy as np
from scipy.ndimage import convolve

from lib.data_classes import Unit, Fragment, TerrainTile, Position, Task, TaskType
from lib.game_config import GameConfig

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={'float': '{:.2f}'.format})


class Attacker:

    def __init__(self, units_my: List[Unit], units_enemy: List[Unit], terrain: np.ndarray[int], energy: np.ndarray[int], fragments: Set[Fragment], cfg: GameConfig, team_id: int, logger: logging.Logger):
        self.units_my: List[Unit] = units_my
        self.units_enemy: List[Unit] = units_enemy
        self.terrain_map: np.ndarray[int] = terrain
        self.energy_map: np.ndarray[int] = energy       # Energy already incorporated info from nebula
        self.fragments: Set[Fragment] = fragments

        self.cfg: GameConfig = cfg
        self.logger = logger
        self.team_id = team_id

        self.utility = self._get_attack_utility(self._get_probability_next_move())
        logger.info("Attacking utility")
        logger.info(f"\n{self.utility}")

    def _tile_importance(self, y: int, x: int, fragments: Set[Position], is_dir_enemy: bool) -> float:
        # For a tile position, calculates how important is it based on some parameters of that tile:
        #  - whether it has a fragment
        #  - energy on the tile
        #  - whether this moves you in direction of the enemy
        fragment_add        = 5  # to increase the quality of the fragment (as units collect from them)
        dir_to_enemy_mult   = 1.05  # to bump the probability that the unit would like to go in direction of enemy
        energy_range_mult   = [0.5, 2]  # So the smallest energy tile will be weighted 4x less than the highest energy

        e = self.energy_map[y, x]
        if self.terrain_map[y, x] == TerrainTile.NEBULA:
            e -= self.cfg.cfg.cfgnebula_energy_reduction

        min_e, max_e = -10, 10
        e = max(min_e, min(max_e, e))    # clip in -10, 10 range
        e = energy_range_mult[0] + (e - min_e) * (energy_range_mult[1] - energy_range_mult[0]) / (max_e - min_e)

        if is_dir_enemy:
            e *= dir_to_enemy_mult

        if Position(y=y, x=x) in fragments:
            e += fragment_add

        return e

    def _unit_probability_distribution(self, heatmap: np.ndarray[float], unit: Unit, fragments_pos: Set[Position]) -> np.ndarray[float]:
        # Updates the heatmap with the probability distribution of the next move of the unit
        # Modifies the heatmap in place and returns it for convenience
        if unit.energy < 0:  # Units with negative energy will be removed. They are dead, but still shown
            return heatmap

        is_move_possible = unit.energy >= self.cfg.cfg.unit_move_cost
        weights, n = [], heatmap.shape[0]
        dirs = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
        for dy, dx in dirs:
            y_, x_ = unit.y + dy, unit.x + dx
            if y_ < 0 or x_ < 0 or y_ == n or x_ == n:
                weights.append(0)  # if you move outside the range, ignore it
            elif dx == dy == 0:
                v = self._tile_importance(y_, x_, fragments_pos, False)
                weights.append(v)  # Already here
            elif self.terrain_map[y_, x_] != TerrainTile.ASTEROID.value and is_move_possible:
                # Is not obstructed and have energy to move there
                v = self._tile_importance(y_, x_, fragments_pos, False)
                weights.append(v)
            else:
                weights.append(0)  # can't move

        if len(weights) != 5:
            raise Exception('Wrong number of moves')

        s = sum(weights)
        weights = [w / s for w in weights]  # Normalizing weights

        for i, (dy, dx) in enumerate(dirs):
            if weights[i] != 0:
                heatmap[unit.y + dy, unit.x + dx] += weights[i]
        return heatmap

    def _get_probability_next_move(self) -> np.ndarray[float]:
        # For every unit generate probability where the unit will be in one turn. This will depend on the units
        # energy, whether it is on fragment and whether it can move in some directions. As you will sum, some
        # values will be > 1 (multiple units can come to some location).
        fragments_pos = {f.pos() for f in self.fragments}
        heatmap = np.zeros_like(self.terrain_map, dtype=float)
        for u in self.units_enemy:
            heatmap = self._unit_probability_distribution(heatmap, u, fragments_pos)

        return heatmap

    def _get_attack_utility(self, heatmap: np.ndarray[float]):
        # You will need to compare this value with 1 to figure out how valuable is to shoot somewhere.
        #
        # Having a heat-matrix of probability score, you can calculate the utility of attacking
        # each point on the map. If `s = unit_sap_cost` and `a = unit_sap_dropoff_factor` the value on the
        # [p1, p2, p3]
        # [p4, p5, p6]
        # [p7, p8, p8]
        # will be `s` multiplied by `p5 + a * (p1 + p2 + p3 + p4 + p6 + p7 + p8 + p9).
        # If K = sum of that window, this is `res = p5 + a * (K - p5) = a * K + p5 * (1 - a)
        # As you see, `s` is not important here as it cost you to attack `s` energy, and you kill expected `s * res`
        # So all you need is to compare this result against 1 to see how much more utility will you kill
        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        utility = convolve(heatmap, kernel, mode='constant', cval=0) * self.cfg.cfg.unit_sap_dropoff_factor
        return utility + heatmap * (1 - self.cfg.cfg.unit_sap_dropoff_factor)

    def _attacking_mask(self, unit: Unit):
        """Given a unit, calculate a mask which it can attack"""
        n, y, x = 24, unit.y, unit.x
        mask = np.zeros((24, 24), dtype=bool)

        # Compute bounds, ensuring they are within matrix limits
        y_min = max(y - self.cfg.cfg.unit_sap_range, 0)
        y_max = min(y + self.cfg.cfg.unit_sap_range + 1, n)
        x_min = max(x - self.cfg.cfg.unit_sap_range, 0)
        x_max = min(x + self.cfg.cfg.unit_sap_range + 1, n)

        # Set the mask within the bounds to True
        mask[y_min:y_max, x_min:x_max] = True
        return mask

    def _attack_hit(self, p: Position) -> np.ndarray[int]:
        """If unit will attack at position, how much energy will be taken"""
        attack_matrix = np.zeros((24, 24), dtype=int)
        y_min, y_max = max(0, p.y - 1), min(24, p.y + 2)
        x_min, x_max = max(0, p.x - 1), min(24, p.x + 2)
        attack_matrix[y_min:y_max, x_min:x_max] = int(self.cfg.cfg.unit_sap_cost * self.cfg.cfg.unit_sap_dropoff_factor)
        attack_matrix[p.y, p.x] = self.cfg.cfg.unit_sap_cost
        return attack_matrix

    @staticmethod
    def _find_top_positions(utility: np.ndarray[float], mask: np.ndarray[bool], top_k: int) -> List[Tuple[Position, float]]:
        """Returns at most `top_k` positions which provide the best utility of shooting there. Ignores zeros"""
        masked_values = utility * mask

        # Get the indices of the top `top_k` values
        flat_indices = np.argsort(masked_values.flatten())[::-1]  # Sort in descending order
        top_indices = flat_indices[:top_k]  # Take the first `top_k` indices

        res = []
        for idx in top_indices:
            y, x = np.unravel_index(idx, utility.shape)
            val = float(masked_values[y, x])
            if val < 0.1:
                break

            res.append((Position(y=int(y), x=int(x)), val))

        return res

    def get_attack_suggestions(self, top_k: int) -> Dict[int, Tuple[Position, float]]:
        """This function does not tell whom to attack. It just tells a positions which a unit can attack
        Later will decide who will attack whom as it knows the bigger picture of the situation (is unit idle,
        is it in risk, etc.)
        """
        units_can_attack = [u for u in self.units_my if u.energy >= self.cfg.cfg.unit_sap_cost]

        uid_action: Dict[int, List[Tuple[Position, float]]] = {}
        is_estimated_shoot_fired = False
        for u in units_can_attack:
            positions = self._find_top_positions(self.utility, self._attacking_mask(u), top_k)
            if positions:
                if not self.cfg.is_unit_sap_dropoff_factor_estimated() and not is_estimated_shoot_fired:
                    self.logger.warning(f"SAP dropoff not estimated. Force {u} to attack {positions[0][0]}")

                    # artificially put super high value to force attack. And do this only once
                    is_estimated_shoot_fired = True
                    uid_action[u.id] = (positions[0][0], float('inf'))
                else:
                    uid_action[u.id] = positions[0]

        return uid_action

    def is_unit_shoot_based_on_task(self, damage: float, unit: Unit, task: Task) -> bool:
        """Given the amount of damage the unit is dealing, and it's task, figure out whether it makes sense to shoot"""
        if damage >= 2:
            return True

        is_in_position = unit.pos() == task.pos()
        can_move_after_shot = unit.energy - self.cfg.cfg.unit_sap_cost > self.cfg.cfg.unit_move_cost * 2
        # If unit is in position or can move or good tile, then you can attack. Otherwise, too risky
        if not (is_in_position or can_move_after_shot or self.energy_map[unit.y, unit.x] >= 5):
            return False

        if damage >= 1.5:
            return True

        if damage > 0.85 and (unit.energy > 200 or is_in_position or task.type == TaskType.ATTACK):
            return True

        if damage > 0.65 and task.type in [TaskType.WAIT, TaskType.ATTACK, TaskType.COLLECT_FRAGMENT_ENEMY] and is_in_position:
            return True

        return False

    def is_unit_shoot_based_on_damage(
            self, uid: int,
            enemy_energy: Dict[int, int],
            enemy_distribution: Dict[int, np.ndarray[float]],
            p_attack: Position
    ) -> Tuple[bool, Dict[int, int]]:
        attack_matrix = self._attack_hit(p_attack)

        total_dmg, enemy_energy_copy = 0, {id: e for id, e in enemy_energy.items()}
        for enemy_u in self.units_enemy:
            if enemy_energy_copy[enemy_u.id] <= 0:
                continue

            expected_dmg = int((enemy_distribution[enemy_u.id] * attack_matrix).sum())
            dmg = min(expected_dmg, enemy_energy_copy[enemy_u.id])

            total_dmg += dmg
            enemy_energy_copy[enemy_u.id] -= dmg

        if total_dmg < self.cfg.cfg.unit_sap_cost * 0.2:
            return False, enemy_energy

        self.logger.info(f"  Unit {uid} expects to deal {total_dmg}. Cost {self.cfg.cfg.unit_sap_cost}")
        return True, enemy_energy_copy

    def adjust_attack(self, attack_suggestions: Dict[int, Tuple[Position, float]], uid_to_task: Dict[int, Task]) -> Dict[int, Position]:
        self.logger.info("Shooting adjustment")

        uid_to_units = {u.id: u for u in self.units_my}
        my_attacking_unit_ids = []
        for uid, (p_attack, dmg) in attack_suggestions.items():
            if uid not in uid_to_task:
                continue

            if self.is_unit_shoot_based_on_task(dmg, uid_to_units[uid], uid_to_task[uid]):
                my_attacking_unit_ids.append(uid)
                self.logger.info(f"  uid {uid} with task {uid_to_task[uid]} will     shoot {p_attack}, {dmg}")
            else:
                self.logger.info(f"  uid {uid} with task {uid_to_task[uid]} will NOT shoot {p_attack}, {dmg}")

        # TODO: In reality will iterate over different order of my units
        enemy_distribution: Dict[int, np.ndarray[float]] = {}
        enemy_energy: Dict[int, int] = {}
        for u in self.units_enemy:
            enemy_distribution[u.id] = self._unit_probability_distribution(np.zeros((24, 24), dtype=float), u, self.fragments)
            enemy_energy[u.id] = u.energy

        new_attacking_unit_ids = []
        for uid in my_attacking_unit_ids:
            self.logger.info(f"  Enemy energy before attack of {uid}: {enemy_energy}")
            is_ok, enemy_energy = self.is_unit_shoot_based_on_damage(uid, enemy_energy, enemy_distribution, attack_suggestions[uid][0])
            if is_ok:
                new_attacking_unit_ids.append(uid)

        self.logger.info(f"  Expect to kill {sum(int(u <= 0) for u in enemy_energy.values())} units")
        return {uid: attack_suggestions[uid][0] for uid in new_attacking_unit_ids}


