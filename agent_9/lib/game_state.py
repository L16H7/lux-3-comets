from typing import List, Set

from lib.data_classes import Unit, MapObservations, Relic, Stats, GameState

"""
Responsible for parsing `obs` which is passed to Agent and converting it to
state which can be usable by other functions.
"""
class Observations:

    def __init__(self, obs, team_id: int):
        self.obs = obs
        self.team_id = team_id

    def _get_units(self) -> List[Set[Unit]]:
        units = [set(), set()]
        for player_curr in [0, 1]:
            tmp_p = self.obs['units']['position'][player_curr].tolist()
            tmp_e = self.obs['units']['energy'][player_curr].tolist()
            for i in range(len(tmp_p)):
                if tmp_e[i] != -1:
                    units[player_curr].add(Unit(
                        team_id=player_curr,
                        id=i,
                        x=tmp_p[i][0],
                        y=tmp_p[i][1],
                        energy=tmp_e[i]
                    ))

        return units

    def _get_map(self) -> MapObservations:
        return MapObservations(
            vision=self.obs['sensor_mask'].T,
            energy=self.obs['map_features']['energy'].T,
            terrain=self.obs['map_features']['tile_type'].T
        )

    def _whom_belongs(self, x: int, y: int) -> int:
        d0 = x + y
        d1 = 23 - x + 23 - y
        if d0 == d1:
            # If equal distance, be aggressive and claim this tile
            return self.team_id

        return 0 if d0 < d1 else 1

    def _get_relics(self) -> Set[Relic]:
        relics = set()
        for el in self.obs['relic_nodes'].tolist():
            if el != [-1, -1]:
                relics.add(Relic(x=el[0], y=el[1]))
        return relics

    def _get_stats(self) -> Stats:
        return Stats(
            points=(int(self.obs['team_points'][0]), int(self.obs['team_points'][1])),
            wins=(int(self.obs['team_wins'][0]), int(self.obs['team_wins'][0])),
            steps=self.obs['steps'],
            match_steps=self.obs['match_steps'],
            num_units=len(self.obs['units_mask'][0]),
        )

    def update_obs(self, obs):
        self.obs = obs

    def read_state(self):
        units = self._get_units()
        return GameState(
            team_id=self.team_id,
            units_0=units[0],
            units_1=units[1],
            map=self._get_map(),
            relics=self._get_relics(),
            stats=self._get_stats()
        )