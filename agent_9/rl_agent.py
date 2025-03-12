# It throws error if I don't put it.
import absl.logging
import jax.random
absl.logging.set_verbosity(absl.logging.ERROR)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import jax
import jax.numpy as jnp
import orbax.checkpoint

from agent import get_actions, vectorized_transform_actions
from representation import create_representations, transform_coordinates, reconcile_positions, transform_observation_3dim
from model import Actor


#  From Salvador agent
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
from lib.unit_estimator import AllEstimator
from lib.helper import energy_map_total
from lib.convertor import rl_shots_to_positions


__DEBUG_PATH__ = '_DATA'
__IS_LOGGING__ = True
__IS_STORE_PARAM__ = False
__IS_STORE_MAP__ = False


class DotDict:
    """A class that recursively converts dictionaries to objects 
    accessible with dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                value = DotDict(value)
            setattr(self, key, value)

expand_repeat = jax.jit(lambda x: jnp.expand_dims(x.repeat(16, axis=0), axis=0))

def reshape_observation(obs):
    new_obs = dict(obs)
    new_obs["units"] = dict(new_obs["units"]) # copy so we can modify
    new_obs["units"]["position"] = new_obs["units"]["position"][None, :]
    new_obs["units"]["energy"] = new_obs["units"]["energy"][None, :]

    new_obs["units_mask"] = new_obs["units_mask"][None, :]
    new_obs["relic_nodes"] = new_obs["relic_nodes"][None, :]
    new_obs["relic_nodes_mask"] = new_obs["relic_nodes_mask"][None, :]
    new_obs["sensor_mask"] = new_obs["sensor_mask"][None, :]

    new_obs["map_features"] = dict(new_obs["map_features"]) # copy so we can modify
    new_obs["map_features"]["energy"] = new_obs["map_features"]["energy"][None, :]
    new_obs["map_features"]["tile_type"] = new_obs["map_features"]["tile_type"][None, :]

    new_obs["team_points"] = new_obs["team_points"][None, :]

    new_obs["steps"] = jnp.atleast_1d(new_obs["steps"])
    new_obs["match_steps"] = jnp.atleast_1d(new_obs["match_steps"])

    return new_obs

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player

        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        self.unit_move_cost = jnp.array(env_cfg["unit_move_cost"]).repeat(16) / 4
        self.unit_sap_cost = jnp.array(env_cfg["unit_sap_cost"]).repeat(16) / 40
        self.unit_sap_range = jnp.array(env_cfg["unit_sap_range"] - 2).repeat(16) / 4
        self.unit_sensor_range = jnp.array(env_cfg["unit_sensor_range"]).repeat(16) / 4

        checkpoint_path = os.path.join(script_dir, 'checkpoint')
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.params = orbax_checkpointer.restore(checkpoint_path)

        self.rng = jax.random.PRNGKey(20)
        self.actor = Actor(n_actions=6)

        self.inference_fn = jax.jit(lambda x, x2: self.actor.apply(x, x2, rngs=self.rng))

        self.discovered_relic_nodes = np.ones((1, 6, 2), dtype=jnp.int32) * -1
        self.prev_team_points = 0
        self.points_map = jnp.zeros((1, 24, 24), dtype=jnp.float32)
        self.search_map = jnp.zeros((1, 24, 24), dtype=jnp.float32)
        self.temporal_states = jnp.zeros((1, 8, 24, 24), dtype=jnp.float32)
        self.points_gained = 0

        self.points_history_positions = (jnp.ones((101, 1, 16, 2), dtype=jnp.int32) * -1)
        self.points_history = jnp.zeros((101, 1), dtype=jnp.int32)
        self.nebula_info = jnp.zeros((1, 2))
        self.prev_agent_energies = jnp.zeros((1, 16, 1))

        self.prev_opponent_points = 0
        self.opponent_points_gained = 0
        self.prev_opponent_points_gained = 0
        self.sapped_units_mask = jnp.zeros((1, 16))

        # From Salvador
        self.debug_data = {}

        self.cfg = GameConfig(env_cfg)
        self.map_maker = MapMaker(self.team_id)
        self.pathfinder = PathFinder(self.cfg.cfg)
        self.enemy_tracker = EnemyTracker()
        self.unit_estimator = AllEstimator(self.team_id)

        self.logging_dir = self._first_dir(f"{__DEBUG_PATH__}/game")
 

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # TODO: how to do this better?
        # Do all analysis and estimation in every step. It is usually very cheap so does not matter
        units_my, units_enemy, game_state, logger, attack_matrix = self.salvador_collect_info(step, obs, remainingOverageTime)

        observation = DotDict(reshape_observation(obs))

        relic_mask = observation.relic_nodes != -1
        relic_nodes_count_before = (self.discovered_relic_nodes[..., 0] > -1).sum(axis=-1)
        self.discovered_relic_nodes = jnp.where(
            relic_mask,
            observation.relic_nodes, 
            self.discovered_relic_nodes,
        )
        self.discovered_relic_nodes = reconcile_positions(self.discovered_relic_nodes)
        relic_nodes_count_after = (self.discovered_relic_nodes[..., 0] > -1).sum(axis=-1)

        # reset points map if new relic node is found
        if (relic_nodes_count_after - relic_nodes_count_before)[0] > 0:
            self.points_map = jnp.maximum(self.points_map, 0)
            self.points_history = jnp.zeros((101, 1), dtype=jnp.int32)
            self.points_history_positions = (jnp.ones((101, 1, 16, 2), dtype=jnp.int32) * -1)

        if step == 101 or step == 202:
            self.points_map = jnp.maximum(self.points_map, 0)
            self.points_history = jnp.zeros((101, 1), dtype=jnp.int32)
            self.points_history_positions = (jnp.ones((101, 1, 16, 2), dtype=jnp.int32) * -1)

        team_points = obs['team_points'][self.team_id]
        self.points_gained = team_points - self.prev_team_points
        self.prev_team_points = team_points

        self.prev_opponent_points_gained = self.opponent_points_gained
        opponent_points = obs['team_points'][self.opponent_team_id]
        self.opponent_points_gained = opponent_points - self.prev_opponent_points
        self.prev_opponent_points = opponent_points

        representations = create_representations(
            obs=observation,
            temporal_states=self.temporal_states,
            relic_nodes=self.discovered_relic_nodes,
            prev_agent_energies=self.prev_agent_energies,
            points_map=self.points_map,
            search_map=self.search_map,
            points_gained=jnp.array([self.points_gained]),
            opponent_points_gained=jnp.array([self.opponent_points_gained]),
            points_history_positions=self.points_history_positions,
            points_history=self.points_history,
            prev_opponent_points_gained=jnp.array([self.prev_opponent_points_gained]),
            unit_move_cost=jnp.array([self.env_cfg["unit_move_cost"]]),
            sensor_range=jnp.array([self.env_cfg["unit_sensor_range"]]),
            nebula_info=self.nebula_info,
            sapped_units_mask=self.sapped_units_mask,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
            map_maker=self.map_maker,
            cfg=self.cfg,
            logger=logger,
        )
        
        (
            states,
            temporal_states,
            agent_observations,
            episode_info,
            points_map,
            search_map,
            agent_positions,
            agent_energies,
            agent_energies_gained,
            _,
            discovered_relic_nodes,
            points_history_positions,
            points_history,
            updated_nebula_info,
        ) = representations

        self.points_map = points_map
        self.search_map = search_map
        self.temporal_states = temporal_states
        self.discovered_relic_nodes = discovered_relic_nodes
        self.points_history_positions = points_history_positions
        self.points_history = points_history
        self.nebula_info = updated_nebula_info

        self.prev_agent_energies = observation.units.energy[:, self.team_id, :, None]

        agent_observations = jnp.squeeze(agent_observations, axis=0)
        agent_episode_info = episode_info.repeat(16, axis=0)
        agent_positions = agent_positions.reshape(-1, 2)

        logits = self.inference_fn(
            { "params": self.params },
            {
                "observations": agent_observations,
                "positions": agent_positions,
                "match_steps": agent_episode_info[:, 0],
                "matches": agent_episode_info[:, 1],
                "team_points": agent_episode_info[:, 2],
                "opponent_points": agent_episode_info[:, 3],
                "unit_move_cost": self.unit_move_cost,
                "unit_sap_cost": self.unit_sap_cost,
                "unit_sap_range": self.unit_sap_range,
                "unit_sensor_range": self.unit_sensor_range,
                "energies": agent_energies,
                "energies_gained": agent_energies_gained,
                "points_gained_history": agent_episode_info[:, 4:],
            },
        )

        self.rng, action_rng = jax.random.split(self.rng)

        actions = get_actions(
            rng=action_rng,
            team_idx=self.team_id,
            opponent_idx=self.opponent_team_id,
            logits=logits,
            observations=observation,
            sap_ranges=jnp.array([self.env_cfg["unit_sap_range"]]),
            relic_nodes=self.discovered_relic_nodes,
            points_map=self.points_map,
            attack_matrix=attack_matrix,
        )

        transformed_targets = transform_coordinates(actions[..., 1:], 17, 17)
        transformed_p1_actions = jnp.zeros_like(actions)
        transformed_p1_actions = transformed_p1_actions.at[..., 0].set(vectorized_transform_actions(actions[:, :, 0]))
        transformed_p1_actions = transformed_p1_actions.at[..., 1].set(transformed_targets[..., 0])
        transformed_p1_actions = transformed_p1_actions.at[..., 2].set(transformed_targets[..., 1])

        actions = actions if self.team_id == 0 else transformed_p1_actions

        actions = actions.at[..., 1:].set(actions[..., 1:] - 8)
        self.sapped_units_mask = actions[..., 0] == 5

        # TODO: how to do this better? This is to highjack some units with my actions
        # actions_salvador = self.salvador_act(step, units_my, units_enemy, game_state, logger)
        # if actions_salvador is not None:  # You get none only when you do not have any SEARCH_FRAGMENT
        #     logger.info(f'USE RULE-BASED AGENT')
        #     return actions_salvador
        #
        # logger.info(f'USE RL AGENT')
        shots = rl_shots_to_positions(np.array(actions), units_my)
        logger.info('------------------------------')
        logger.info(shots)
        logger.info('------------------------------')

        self.cfg.add_shots(shots)
        return jnp.squeeze(actions, axis=0)

    # ===========================
    # Starting from here is rule-based agent
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

    def salvador_collect_info(self, step: int, obs: Dict[str, Any], remainingOverageTime: int = 60):
        start_time = time.perf_counter()
        logger = logging_init(team_id=self.team_id, step=step, logs_dir=f"{self.logging_dir}/logs/",
                              is_logging=__IS_LOGGING__)
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

        if game_state.stats.match_steps <= 1:
            # Units start to appear on T1
            self.unit_estimator = AllEstimator(self.team_id)

        logger.info(f'Enemy units {units_enemy}')

        logger.info(f'--------- MATCH STEP--- {game_state.stats.match_steps}')
        logger.info(f'is_relic_search_this_match: {self.map_maker.map_info.is_relic_search_this_match}')
        logger.info(f'is_relic_search_next_match: {self.map_maker.map_info.is_relic_search_next_match}')
        logger.info(f'is_relic_new_found        : {self.map_maker.map_info.is_relic_new_found}')

        self.map_maker.update(
            gs=game_state,
            my_points=game_state.stats.points[self.team_id],
            cfg=self.cfg.cfg,
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

        self.unit_estimator.step_update(self.map_maker.map_info, units_enemy)
        P = self.unit_estimator.probability_sum()
        A = self.unit_estimator.get_attack_utility(P, sap_dropoff_factor=self.cfg.cfg.unit_sap_dropoff_factor)
        logger.info(f'=============UNIT PROBABILITY==================')
        logger.info(f'Number of units: {P.sum()}')
        logger.info(f'Total estimated probability:\n{P}')
        logger.info(f'Attack utility:\n{A}')
        logger.info(f'=============UNIT PROBABILITY==================')

        return units_my, units_enemy, game_state, logger, A

    def salvador_act(self, step: int, units_my, units_enemy, game_state, logger):
        attacker = Attacker(
            units_my=units_my,
            units_enemy=units_enemy,
            terrain=self.map_maker.map_info.terrain,
            energy=energy_map_total(self.map_maker.map_info.energy, self.map_maker.map_info.terrain,
                                    self.cfg.cfg.nebula_energy_reduction),
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
        if not np.any(res):
            logger.info(f'No tasks so I will go for RL agent')
            return None

        self.cfg.add_shots(TA.shots)
        logger.info(f'Shooting at positions: {TA.shots}')

        # All debugging
        self._save_game_params(step, self.team_id)
        self._debug_map(step, units_my)

        return res
