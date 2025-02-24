import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

from lib.attacker import Attacker
from lib.data_classes import GameState
from lib.game_config import GameConfig
from lib.game_state import Observations
from lib.logs import logging_init
from lib.mapmaker import MapMaker
from lib.path_finder import PathFinder
from lib.task_assigner import TaskAssigner
from lib.enemy_tracker import EnemyTracker
from lib.helper import energy_map_total

__DEBUG_PATH__ = '_DATA'
__IS_LOGGING__ = False
__IS_STORE_PARAM__ = False
__IS_STORE_MAP__ = False


np.random.seed(0)


"""
You have 3sec per turn and 60sec in total
"""


class Agent:

    def _first_dir(self, path: str) -> str:
        if not __IS_LOGGING__:
            return

        for i in range(1000):
            dir_name = f"{path}_{i:03d}"
            if not os.path.exists(dir_name):
                Path(dir_name).mkdir(parents=True, exist_ok=True)
                return dir_name

    def _debug_map(self, step: int, my_units):
        if not __IS_STORE_MAP__:
            return

        directory = Path(f"{self.logging_dir}/map_info/step_{step:03d}")
        Path(directory).mkdir(parents=True, exist_ok=True)
        np.save(directory / "vision_single.npy", self.map_maker.map_info.vision_single)
        np.save(directory / "terrain.npy", self.map_maker.map_info.terrain)
        np.save(directory / "energy.npy", self.map_maker.map_info.energy)
        np.save(directory / "vision.npy", self.map_maker.map_info.vision)
        np.save(directory / "visited.npy", self.map_maker.map_info.visited)
        np.save(directory / "explored.npy", self.map_maker.map_info.explored)
        np.save(directory / "fragments_possible_my.npy", self.map_maker.map_info.fragments_possible_my)

        with open(directory / "equations.pkl", "wb") as f:
            pickle.dump(self.map_maker.FragmentFinder.equations, f)

        with open(directory / "my_units.pkl", "wb") as f:
            pickle.dump(my_units, f)

        with open(directory / "relics.pkl", "wb") as f:
            pickle.dump(self.map_maker.map_info.relics, f)

    def _save_game_params(self, step: int, player: int):
        if not __IS_STORE_PARAM__:
            return

        if step != 499:
            return

        data = {
            'nebula_vision_reduction': self.cfg.cfg.nebula_vision_reduction,
            'nebula_energy_reduction': self.cfg.cfg.nebula_energy_reduction,
            'terrain_speed': self.cfg.cfg.terrain_speed,
            'terrain_dir': self.cfg.cfg.terrain_dir,
            'unit_sap_dropoff_factor': self.cfg.cfg.unit_sap_dropoff_factor,
        }
        os.makedirs(__DEBUG_PATH__, exist_ok=True)
        with open(f"{__DEBUG_PATH__}/game_params_{player}.txt", 'a') as file:
            file.write(f"{data}\n".replace("'", '"'))

    def __init__(self, player: str, env_cfg) -> None:
        self.player: str = player
        self.team_id: int = 0 if self.player == "player_0" else 1
        self.env_cfg: Dict[str, Any] = env_cfg

        self.debug_data = {}

        self.cfg = GameConfig(env_cfg)
        self.map_maker = MapMaker(self.team_id)
        self.pathfinder = PathFinder(self.cfg.cfg)
        self.enemy_tracker = EnemyTracker()

        self.logging_dir = self._first_dir(f"{__DEBUG_PATH__}/game")


    def act(self, step: int, obs: Dict[str, Any], remainingOverageTime: int = 60):
        start_time = time.perf_counter()
        logger = logging_init(team_id=self.team_id, step=step, logs_dir=f"{self.logging_dir}/logs/", is_logging=__IS_LOGGING__)
        game_state: GameState = Observations(obs, self.team_id).read_state()
        units_enemy, units_my = game_state.units_enemy(), game_state.units_my()

        if game_state.stats.match_steps == 0:
            logger.info("New match starts")
            if step >= 400:
                self.map_maker.map_info.is_relic_search_this_match = False
                self.map_maker.map_info.is_relic_search_next_match = False
                self.map_maker.map_info.is_relic_new_found = True
            else:
                self.map_maker.map_info.is_relic_search_this_match = True
                self.map_maker.map_info.is_relic_search_next_match = True
                self.map_maker.map_info.is_relic_new_found = False

        if game_state.stats.match_steps < 3:
            # There is a bug in Lux AI on MatchStep = 0 you are given info from the last step of previous match
            self.enemy_tracker = EnemyTracker()  # Reload the enemy tracker

        logger.info(f'Enemy units {units_enemy}')

        logger.info(f'--------- MATCH STEP--- {game_state.stats.match_steps}')
        logger.info(f'is_relic_search_this_match: {self.map_maker.map_info.is_relic_search_this_match}')
        logger.info(f'is_relic_search_next_match: {self.map_maker.map_info.is_relic_search_next_match}')
        logger.info(f'is_relic_new_found        : {self.map_maker.map_info.is_relic_new_found}')

        self.map_maker.update(
            gs=game_state,
            my_points=game_state.stats.points[self.team_id],
            cfg= self.cfg.cfg,
            logger=logger,
        )

        self.cfg.update(
            units_my=units_my,
            units_enemy=units_enemy,
            map_info=self.map_maker.map_info,
            step=step,
            logger=logger
        )
        self.map_maker.update_terrain(game_state, self.cfg.cfg, step)

        self.pathfinder.update_map(
            m=self.map_maker.map_info,
            units_enemy=units_enemy,
            units_my=units_my,
            logger=logger
        )

        self.enemy_tracker.update(
            mapinfo_fragments=self.map_maker.map_info.fragments,
            mapinfo_vision_single=self.map_maker.map_info.vision_single,
            enemy_units=units_enemy
        )

        logger.warning(f'Original enemy units {units_enemy}')
        units_enemy = self.enemy_tracker.get_enemy()
        logger.warning(f'Adjusted enemy units {units_enemy}')

        attacker = Attacker(
            units_my=units_my,
            units_enemy=units_enemy,
            terrain=self.map_maker.map_info.terrain,
            energy=energy_map_total(self.map_maker.map_info.energy, self.map_maker.map_info.terrain, self.cfg.cfg.nebula_energy_reduction),
            fragments=self.map_maker.map_info.fragments,
            cfg=self.cfg,
            team_id=self.team_id,
            logger=logger,
        )
        TA = TaskAssigner(
            map_maker=self.map_maker,
            path_finder=self.pathfinder,
            cfg=self.cfg.cfg,
            team_id=self.team_id,
            step=step,
            step_in_match=game_state.stats.match_steps,
            logger=logger,
        )
        res = TA.assign(units_my, units_enemy, attacker)
        self.cfg.add_shots(TA.shots)
        logger.info(f'Shooting at positions: {TA.shots}')

        # All debugging
        self._save_game_params(step, self.team_id)
        self._debug_map(step, units_my)

        execution_time_ms = int((time.perf_counter() - start_time) * 1000)
        logger.debug(f"Whole turn took {execution_time_ms} ms")     # You are allowed to use 3sec
        return res