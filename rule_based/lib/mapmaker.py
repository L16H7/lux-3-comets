from logging import Logger
from typing import Set, Dict

import numpy as np

from lib.data_classes import MapInfo, Relic, GameState, TerrainTile, Unit, Position, Fragment, GameParam
from lib.fragment_finder import FragmentFinder
from lib.helper import terrain_matrix_to_strings, fragment_matrix_to_string, relics_matrix_to_strings, is_world_changed, mark_nxn, change_terrain
from lib.solver import Solver

"""
Responsible for maintaining the map of the world.
It includes:
 - area you see
 - area you explored
 - terrain (space, asteroids, nebula)
 - energy
 - location relics
 - location of points next to relics which give winning points 

Map of the world does not include:
 - Player's units
 - rules of the world
"""

class MapMaker:

    def __init__(self, team_id: int):
        self.n = 24
        self.logger: Logger = None
        self.team_id: int = team_id

        # This information is needed to detect how many points I got this turn, which is used to detect fragments
        self.points_prev: int = 0   # points from the previous turn
        self.points_curr: int = 0   # points currently
        # Equations which will be used for binary optimization solver to find fragments
        self.FragmentFinder: FragmentFinder = FragmentFinder()
        self.Solver: Solver = Solver()

        self.map_info = MapInfo(
            vision_single=np.zeros((self.n, self.n), dtype=bool),
            vision=np.zeros((self.n, self.n), dtype=bool),
            explored=np.zeros((self.n, self.n), dtype=bool),
            visited=np.zeros((self.n, self.n), dtype=bool),
            energy=50 * np.ones((self.n, self.n), dtype=int),
            terrain=-np.ones((self.n, self.n), dtype=int),
            terrain_tmp=-np.ones((self.n, self.n), dtype=int),

            relics=set(),
            fragments=set(),

            # This holds all possible position where a fragment can be. With every new step we zero-out some position
            # where we found out that fragments are not possible.
            fragments_possible=np.zeros((self.n, self.n), dtype=bool),

            relics_possible=self._get_start_relics(),
            is_relic_search_this_match=True,
            is_relic_search_next_match=True,
            is_relic_new_found=False,
        )

    def _get_start_relics(self):
        try:
            return np.load('lib/data/relics.npy') > 0
        except:
            return np.array([
                [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0]
            ]) > 0

    def _expand_matrix_int(self, matrix: np.ndarray[int]) -> np.ndarray[int]:
        matrix = np.copy(matrix)
        for x in range(self.n):
            for y in range(self.n):
                if matrix[x, y] != TerrainTile.UNKNOWN.value:
                    matrix[self.n - 1 - y, self.n - 1 - x] = matrix[x, y]

        return matrix

    def _expand_terrain(self, matrix: np.ndarray[int]) -> np.ndarray[int]:
        return self._expand_matrix_int(matrix)

    def _expand_energy(self, matrix: np.ndarray[int]) -> np.ndarray[int]:
        # Energy matrix has positive and negative values for energy, but also -1 means unknown
        # One might think that because we can't differentiate between -1 (unknown) and -1 (energy)
        # this will cause the issue with symmetry, but in reality we can use the same approach
        return self._expand_matrix_int(matrix)

    def _merge_terrain(self, terrain: np.ndarray[int]) -> None:
        mask = terrain != TerrainTile.UNKNOWN.value
        self.map_info.terrain[mask] = terrain[mask]

    def _expand_relics(self, relics: set[Relic]):
        # Update information about the relics with DEBUG_DATA from symmetric relics
        new_relics = set()
        for r in relics:
            new_relics.add(r)
            new_relics.add(r.symmetric())
        return new_relics

    def _log_fragments(self) -> None:
        # This is just for logging. It does not do anything and not needed
        if not self.map_info.fragments:
            return

        frags = self.map_info.fragments
        self.logger.info(f"Know about {len(frags)} fragments")

        frags_1, frags_0 = [], []
        for f in frags:
            if f.is_team(self.team_id):
                if f.probability == 1:
                    frags_1.append(f.pos())
                else:
                    frags_0.append(f)

        if frags_1:
            self.logger.info(f"  {len(frags_1)} my guaranteed: {sorted(frags_1, key=lambda f: (f.y, f.x))}")

        if frags_0:
            frag_0 = sorted(frags_0, key=lambda f: (-f.probability, f.y, f.x))
            self.logger.info(f"  {len(frags_0)} my unclear   : {[f.pos() for f in frag_0]}")

    def _update_relics(self, gs_relics: Set[Relic]) -> Set[Relic]:
        new_relics = []
        for r in self._expand_relics(gs_relics):
            if r not in self.map_info.relics:
                new_relics.append(r)
                self.map_info.relics.add(r)

        if new_relics:
            self.logger.debug(f"Found additional relics: {new_relics}")
            return new_relics

        return set()

    def _update_fragments_from_equation(self, relics_solutions: Dict[Position, float]):
        if not relics_solutions:
            self.logger.warning("No solutions found")
            return

        fragments = set()
        for pos, prob in relics_solutions.items():
            if prob < 0:
                continue

            f1 = Fragment(y=pos.y, x=pos.x, probability=prob)
            f2 = f1.symmetric()

            for f in [f1, f2]:
                if prob == 0:
                    self.map_info.fragments_possible[f.y, f.x] = False
                else:
                    fragments.add(f)

        self.logger.info(f"Got {len(fragments)} fragments: {sorted(fragments, key=lambda x: (x.team_id, -x.probability))}")
        self.map_info.fragments = fragments

    def _vision_mask(self, my_units: Set[Unit], unit_sensor_range: int):
        # Mask of the area my units are supposed to see. This will be contrasted with real data the unit sees.
        mask = np.zeros((self.n, self.n), dtype=bool)
        for p in my_units:
            l1, l2 = unit_sensor_range, unit_sensor_range + 1
            y_s, x_s = max(0, p.y - l1), max(0, p.x - l1)
            y_e, x_e = min(self.n, p.y + l2), min(self.n, p.x + l2)
            mask[y_s:y_e, x_s:x_e] = True
        return mask

    def _guessing_nebula(self, map_vision: np.ndarray[bool], map_terrain: np.ndarray[int], my_units: Set[Unit], unit_sensor_range: int) -> np.ndarray[int]:
        vision_expected = self._vision_mask(my_units, unit_sensor_range)

        nebula = vision_expected & ~map_vision
        map_terrain[nebula] = TerrainTile.NEBULA.value
        return map_terrain

    def update_vision(self, gs_vision: np.ndarray[bool]) -> None:
        # Vision is whatever you see right now
        self.logger.debug(f"See {gs_vision.sum()} tiles")
        self.map_info.vision_single = gs_vision.copy()

        # Vision matrix is just a boolean matrix, so do | on symmetric info
        self.map_info.vision = gs_vision | np.transpose(gs_vision)[::-1, ::-1]

    def update_explored(self):
        # Unexplored is whatever you have not seen in any iterations
        c1 = self.map_info.explored.sum()
        self.map_info.explored |= self.map_info.vision
        c2 = self.map_info.explored.sum()

        examples = np.where(self.map_info.explored == False)
        examples = [Position(y=y_, x=x_) for y_, x_ in list(zip(examples[0][:5], examples[1][:5]))]

        self.logger.debug(f"Explored {(c2 - c1):3d} new tiles. Unexplored {(576 - c2):3d}. Example: {examples}")

    def update_visited(self, units_my: Set[Unit]):
        # Visited tiles are tiles where a unit was exactly
        c1 = self.map_info.visited.sum()
        for u in units_my:
            self.map_info.visited[u.y, u.x] = True
        c2 = self.map_info.visited.sum()

        self.logger.debug(f"Visited  {(c2 - c1):3d} new tiles. Unvisited  {(576 - c2):3d}")

    def update_energy(self, gs_energy: np.ndarray[int]) -> None:
        # Here I can't use only energy as -1 there means both -1 energy or unseen. So I need to check if I see this map
        energy = self._expand_energy(gs_energy)
        self.map_info.energy[self.map_info.vision] = energy[self.map_info.vision]

    def update_terrain(self, gs: GameState, cfg: GameParam, step: int):
        terrain = self._guessing_nebula(gs.map.vision, gs.map.terrain, gs.units_my(), cfg.unit_sensor_range)
        terrain = self._expand_terrain(terrain)

        if is_world_changed(step, cfg.terrain_speed):  # World moved
            dir_str = 'but do not know which direction'
            if cfg.terrain_dir == 1:
                dir_str = 'World changed Up-Right'
            elif cfg.terrain_dir == -1:
                dir_str = 'World changed Down-Left'
            self.logger.info(dir_str)

            self.map_info.terrain = change_terrain(self.map_info.terrain, terrain, cfg.terrain_dir)
        else:
            self.logger.info("World stayed the same")
        self._merge_terrain(terrain)

        self.logger.debug("My current terrain understanding")
        self.logger.debug("   █ - asteroid, # - nebula, ? - unknown ")
        for l in terrain_matrix_to_strings(self.map_info.terrain):
            self.logger.debug(f'|{l}|')

    def update_terrain_tmp(self, gs: GameState, unit_sensor_range: int):
        terrain = self._guessing_nebula(gs.map.vision, gs.map.terrain, gs.units_my(), unit_sensor_range)
        terrain = self._expand_terrain(terrain)

        mask = terrain != TerrainTile.UNKNOWN.value
        self.map_info.terrain_tmp[mask] = terrain[mask]

    def update_relics_fragments(self, gs_relics: Set[Relic], step_in_match: int):
        if step_in_match == 0:
            self.map_info.relics_possible = self._get_start_relics()

        if step_in_match > 50:
            self.map_info.relics_possible &= ~self.map_info.vision

        new_relics = self._update_relics(gs_relics)
        if not new_relics:
            return

        self.map_info.is_relic_search_this_match = False
        self.map_info.is_relic_search_next_match = True
        self.map_info.is_relic_new_found = True
        for r in new_relics:
            mark_nxn(self.map_info.fragments_possible, r)

    def update(self, gs: GameState, my_points: int, cfg: GameParam, logger):
        # Update my knowledge with current observation
        self.logger = logger

        self.update_vision(gs.map.vision)
        self.update_explored()
        self.update_visited(gs.units_my())
        self.update_energy(gs.map.energy)
        self.update_terrain_tmp(gs, cfg.unit_sensor_range)
        self.update_relics_fragments(gs.relics, gs.stats.match_steps)

        self.logger.debug("My current relics search progress")
        self.logger.debug("   X - relic, ■ - unit, . - my vision, ? - unclear")
        for l in relics_matrix_to_strings(self.map_info.relics_possible, self.map_info.relics, gs.units_my(), self.map_info.vision):
            self.logger.debug(f'|{l}|')

        # update information about winning points. And use them to detect fragments.
        self.points_prev = self.points_curr
        self.points_curr = my_points
        fragment_solutions, is_ok = self.FragmentFinder.solve(
            units=gs.units_my(),
            points_delta=self.points_curr - self.points_prev,
            fragments_guaranteed_pos={f.pos() for f in self.map_info.fragments if f.probability == 1},
            fragments_possible=self.map_info.fragments_possible,
            solver=self.Solver,
            step=gs.stats.steps,
            logger=logger
        )
        if not is_ok:
            self.map_info.fragments_possible = np.zeros((self.n, self.n), dtype=bool)
            self.map_info.fragments = {}
            for r in self.map_info.relics:
                mark_nxn(self.map_info.fragments_possible, r)
        else:
            self._update_fragments_from_equation(fragment_solutions)

        if self.map_info.relics:
            self.logger.debug("My current fragment/relic understanding. After solving equations")
            self.logger.debug("    X - relic, ■ - guaranteed fragment, □ - unclear fragment, . - possible to have fragment")
            for l in fragment_matrix_to_string(self.map_info.fragments_possible, self.map_info.relics, self.map_info.fragments):
                self.logger.debug(f'|{l}|')

        # Log information about fragments
        self._log_fragments()

