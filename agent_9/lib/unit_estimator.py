import numpy as np
from scipy.ndimage import convolve

from typing import Optional, Tuple
from lib.data_classes import Unit, MapInfo, Set, TerrainTile

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(formatter={'float': '{:.3f}'.format})


K = 24   # Size of the environment
MAX_NUM_UNITS = 16


class ProbabilityNext:

    def __init__(self, map_info: MapInfo):
        self.map_info = map_info

    @staticmethod
    def _uniform(y: int, x: int) -> Tuple[float, float, float, float, float]:
        # This defines the easiest uniform function
        res = [0.2] * 5
        dirs = [0, 0, 1, 0, -1, 0]
        for i in range(5):
            can_move = 0 <= y + dirs[i] < K and 0 <= x + dirs[i + 1] < K
            if not can_move:
                res[i] = 0

        s = sum(res)
        return tuple(v / s for v in res)

    def _uniform_with_obstacles(self, y: int, x: int) -> Tuple[float, float, float, float, float]:
        # Improvement over `_uniform` method but here you do not allow to go to locations where there are obstacles
        res = [0.2] * 5
        dirs = [0, 0, 1, 0, -1, 0]
        for i in range(5):
            can_move = 0 <= y + dirs[i] < K and 0 <= x + dirs[i + 1] < K
            if not can_move:
                res[i] = 0
            elif i != 0 and self.map_info.terrain[y + dirs[i], x + dirs[i + 1]] == TerrainTile.ASTEROID.value:
                # Unit is moving but there is an obstacle
                res[i] = 0

        s = sum(res)
        return tuple(v / s for v in res)

    def get_prob(self, y: int, x: int) -> Tuple[float, float, float, float, float]:
        """For a given position of the unit, gives the probabilities that the unit will move to the next positions
        probabilities are given as a vector of size 5 which means: Stay, Right, Down, Left, Up.
        This function is also responsible to:
         - normalize probabilities (sum is guaranteed to be one)
         - if you can't move in some direction (either due to obstacle or due to out of bound), set this prob to zero
        """
        return self._uniform_with_obstacles(y, x)


class UnitEstimator:

    def __init__(self):
        self.estimation = np.zeros((K, K), dtype=float)

    @staticmethod
    def _normalize(M: np.ndarray[float]):
        s = M.sum()
        return M if s == 0 else M / s

    def update(self, vision_single: np.ndarray[bool], unit: Optional[Unit]) -> None:
        """Updates the previous estimation with the current seen knowledge of the world
        vision is a boolean matrix where True means that we see the tile, and false - do not see
        """

        # If we know location of the unit for sure, then all our previous knowledge is irrelevant
        if unit:
            self.estimation = np.zeros((K, K), dtype=float)
            self.estimation[unit.y, unit.x] = 1
            return

        # Now we have some estimates from previous turn but this turn we know that on some of them unit is not
        # possible, because we see it and it is empty. In this case, re-normalize the values
        self.estimation[vision_single == True] = 0
        self.estimation = self._normalize(self.estimation)

    def estimate(self, map_info: MapInfo) -> None:
        """Estimate the probability of the unit being in each position in the map"""
        next_estimation = np.zeros((K, K), dtype=float)
        Prob = ProbabilityNext(map_info)

        dirs = [0, 0, 1, 0, -1, 0]
        for y in range(K):
            for x in range(K):
                probs = Prob.get_prob(y, x)
                for i in range(5):
                    if probs[i] != 0:
                        next_estimation[y + dirs[i], x + dirs[i + 1]] += self.estimation[y, x] * probs[i]

        # Technically can normalize here, but should be unnecessary as this is guaranteed to be probability matrix
        self.estimation = next_estimation

    def kill_unit(self) -> None:
        # If the unit is killed, his matrix became zeros
        self.estimation = np.zeros((K, K), dtype=float)

    def is_unit_dead(self) -> bool:
        return self.estimation.sum() == 0


class AllEstimator:

    def __init__(self, team_id: int):
        self._max_num_units = MAX_NUM_UNITS
        self._time_to_unit_spawn = 0
        self._spawn_rate = 3
        self._p = K - 1 if team_id == 0 else 0
        self._team_id = team_id

        self.unit_estimators = [UnitEstimator() for _ in range(self._max_num_units)]

    def _spawn_unit_if_needed(self) -> None:
        # New unit can be spawn only if the:
        #  - `time_to_unit_spawn` is zero
        #  - some of the units are missing
        if self._time_to_unit_spawn > 0:
            self._time_to_unit_spawn -= 1
            return

        # Technically we can spawn unit. Check which ID is free to be spawned
        for uid, estimator in enumerate(self.unit_estimators):
            if estimator.is_unit_dead():
                # This unit is dead, so spawn the new one
                estimator = UnitEstimator()
                estimator.update(np.array([]), Unit(  # When unit is passed everything else is not used
                    y=self._p, x=self._p, team_id=self._team_id,
                    id=uid, energy=100
                ))
                self.unit_estimators[uid] = estimator

                # and reset the timer
                self._time_to_unit_spawn = self._spawn_rate - 1
                return

    def step_update(self, map_info: MapInfo, units_enemy: Set[Unit]):
        self._spawn_unit_if_needed()

        uid_to_unit = {u.id: u for u in units_enemy}
        for uid in range(self._max_num_units):
            self.unit_estimators[uid].update(map_info.vision_single, uid_to_unit.get(uid, None))
            self.unit_estimators[uid].estimate(map_info)

    def probability_sum(self) -> np.ndarray[float]:
        # Here we sum all 16 matrices. If unit exist that we a probability matrix, otherwise it will be zero
        res = np.zeros((K, K), dtype=float)
        for e in self.unit_estimators:
            res += e.estimation

        return res

    def get_attack_utility(self, prob_sum: np.ndarray[float], sap_dropoff_factor):
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
        utility = convolve(prob_sum, kernel, mode='constant', cval=0) * sap_dropoff_factor
        return utility + prob_sum * (1 - sap_dropoff_factor)
