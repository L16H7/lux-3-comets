from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Set, Tuple, List, Any

import numpy as np

"""Stores all Classes for various objects"""

class Move(IntEnum):
    NONE = 0
    U = 1
    R = 2
    D = 3
    L = 4
    SAP = 5
    UNKNOWN = 6     # This is only for pathfinder when you estimate the number of turns

    def __str__(self):
        # Map each enum member to its desired short string
        return {
            Move.NONE: '-',
            Move.U: '↑',
            Move.R: '→',
            Move.D: '↓',
            Move.L: '←',
            Move.SAP: 'X',
            Move.UNKNOWN: '?'
        }.get(self, super().__str__())

# IMPORTANT. You have to use `.value` to get this information
class TerrainTile(Enum):
    SPACE = 0
    NEBULA = 1
    ASTEROID = 2
    UNKNOWN = -1

class TaskType(IntEnum):
    # We do not have action SAP. Each of those tasks can attack

    # Before relics appear (and therefore can't search for them) you have nothing to do. The best thing is to
    # cover the whole map with units to be able to observe the most of the map and be able to quickly find relics
    # when they appear. You just go to some position and find the position with maximum energy in that vicinity.
    # And you stay and wait there
    WAIT                    = 0

    # Once relics appear, you move through the map (unmarking `relics_possible`) until you have found the relic
    SEARCH_RELICS           = 1

    # Once new relic found, you move through 5x5 area around relic and solve LP to find positions of the fragments.
    SEARCH_FRAGMENTS        = 2

    # Moves to fragments (both 100% and unclear) and stays on it.
    COLLECT_FRAGMENT_MY     = 3

    # Move to the position close to the attacking tile, find the tile with the highest energy and stay there
    # From this position we will be shooting or moving to the target.
    ATTACK                  = 4

    COLLECT_FRAGMENT_ENEMY  = 5  # Collect points from relic fragment which belongs to the enemy

    RETAKE_FRAGMENT_MY      = 6  # My fragment was stolen by enemy. This task retakes it

    def __str__(self):
        # Map each enum member to its desired short string
        return {
            TaskType.WAIT:                  'WAIT',
            TaskType.SEARCH_RELICS:         'SEARCH_RELICS',
            TaskType.SEARCH_FRAGMENTS:      'SEARCH_FRAGMENTS',
            TaskType.COLLECT_FRAGMENT_MY:   'COLLECT_FRAGMENTS_MY',

            TaskType.COLLECT_FRAGMENT_ENEMY:'COLLECT_FRAGMENTS_EN',
            TaskType.ATTACK:                'ATTACK',
            TaskType.RETAKE_FRAGMENT_MY:    'RETAKE_FRAGMENT_MY',
        }.get(self, super().__str__())

# Information about Observation of the map
@dataclass
class MapObservations:
    # All are Y, X
    vision: np.ndarray[bool]    # they call it mask
    energy: np.ndarray[int]
    terrain: np.ndarray[int]

@dataclass(frozen=True)
class Position:
    y: int
    x: int

    def pos(self):
        return Position(y=self.y, x=self.x)

    def whom_belongs(self) -> int:
        d0 = self.x + self.y
        d1 = 46 - d0
        if d0 == d1:
            return -1

        return 0 if d0 < d1 else 1

    def normalize(self):
        if self.x + self.y <= 23:
            return Position(y=self.y, x=self.x)
        return Position(y=23 - self.x, x=23 - self.y)

    def __str__(self):
        return f"P(x={self.x:2d}, y={self.y:2d})"

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class EnergyPos(Position):
    e: int

    def __str__(self):
        return f"P(x={self.x:2d}, y={self.y:2d}, e={self.e})"

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class Unit(Position):
    team_id: int
    id: int
    energy: int

    def __str__(self):
        return f"UID(x={self.x:2d}, y={self.y:2d}, id={self.id:2d}, t={self.team_id}, e={self.energy})"

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class Relic(Position):
    team_id: int  = -1  # On whose part of the map it is located. -1 means that it is on diagonal

    def __eq__(self, o):  # Relic can be shared if on the diagonal
        return self.x == o.x and self.y == o.y

    def __post_init__(self):
        object.__setattr__(self, 'team_id', self.whom_belongs())

    def symmetric(self):
        return Relic(x=23 - self.y, y=23 - self.x)

    def is_team(self, team_id: int) -> bool:
        if self.team_id == -1:  # On diagonal. Belongs to both
            return True

        return self.team_id == team_id

    def __str__(self):
        return f"R(x={self.x:2d}, y={self.y:2d}, t={self.team_id})"

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class Fragment(Position):
    probability: float  # You do not always know the position with 100%
    team_id: int  = -1  # On whose part of the map it is located. -1 means that it is on diagonal

    def __eq__(self, o):  # Fragment can be shared if on the diagonal
        return self.x == o.x and self.y == o.y

    def __post_init__(self):
        object.__setattr__(self, 'team_id', self.whom_belongs())

    def symmetric(self):
        return Fragment(x=23 - self.y, y=23 - self.x, probability=self.probability)

    def is_team(self, team_id: int) -> bool:
        if self.team_id == -1:  # On diagonal. Belongs to both
            return True

        return self.team_id == team_id

    def __str__(self):
        return f"F(x={self.x:2d}, y={self.y:2d}, t={self.team_id}, p={1 if self.probability == 1 else self.probability:.3f})"

    def __repr__(self):
        return self.__str__()

@dataclass
class Stats:
    points: (int, int)
    wins: (int, int)
    steps: int
    match_steps: int
    num_units: int

@dataclass
class GameState:
    team_id: int
    units_0: Set[Unit]
    units_1: Set[Unit]
    map: MapObservations
    relics: Set[Relic]
    stats: Stats

    def units_my(self) -> Set[Unit]:
        return self.units_0 if self.team_id == 0 else self.units_1

    def units_enemy(self) -> Set[Unit]:
        return self.units_0 if self.team_id != 0 else self.units_1

@dataclass
class MapInfo:
    vision_single: np.ndarray[bool] # What I currently see
    vision: np.ndarray[bool]        # What I currently see + symmetric
    explored: np.ndarray[bool]      # Positions where I have seen the tile at least once during the game
    visited: np.ndarray[bool]       # Locations where a unit moved. Helpful to find fragments
    energy: np.ndarray[int]         # Current best knowledge of energy states (some might be obsolete)
    terrain: np.ndarray[int]        # Current best knowledge of terrain.
    terrain_tmp: np.ndarray[int]    # Do not use. This is a fix to find terrain speed. Everywhere else use terrain

    relics: Set[Relic]  # All relics I have ever seen so far
    fragments: Set[Fragment]  # Positions of Fragments I have found
    fragments_possible: np.ndarray[bool]   # Locations where fragments can be located (Is not guaranteed)

    relics_possible: np.ndarray[bool]   # Locations where relics can be located (Is not guaranteed)
    is_relic_search_this_match: bool    # Do I need to look for relic this match
    is_relic_search_next_match: bool    # Do I need to search for relic next match
    is_relic_new_found: bool            # Was relic found during this match

@dataclass
class GameParam:
    # This information is shared with you
    max_units: int          # total number of units you can have
    unit_move_cost: int     # cost to move a unit
    unit_sap_cost: int      # cost to sap
    unit_sap_range: int     # from which distance can you sap
    unit_sensor_range: int  # how far can you see by default
    spawn_rate: int         # how quickly new units appear. Every X-th turn

    # Some things might not make sense to learn. Here is what makes sense
    nebula_vision_reduction: int    # how much you can't see
    nebula_energy_reduction: int    # Additional energy you lose when in nebula. So if 10, you will lose 10 energy
    terrain_speed: float            # how frequently Nebula/Asteroids moves
    terrain_dir: int                # Here 1 mean up right -1 mean down left
    unit_sap_dropoff_factor: float  # how much you take from tiles around

@dataclass
class MoveResult:
    move: Move              # which move unit should take
    num_moves: int          # in how many moves will you reach it. -1 means unreachable
    energy: int             # how much energy will you have at the end (not after this one move but at all)
    points: float           # how much points will you collect while moving. Can have none integers to estimate the uncertainty
    path: List[EnergyPos]   # First few positions of the moves
    path_str: str           # Paths of arrows for debugging

    def __str__(self):
        return f"M({self.move}, moves={self.num_moves}, path={self.path_str})"

    def __repr__(self):
        return self.__str__()

@dataclass(frozen=True)
class Task(Position):
    type: TaskType
    suggested_energy: int = 0
    y_sap: int = None  # if unit needs to attack this turn, those values will be set
    x_sap: int = None
    score: float = 0
    y_real: int = None
    x_real: int = None

    def __str__(self):
        part_sap = f", x_={self.x_sap}, y_={self.y_sap}" if self.y_sap is not None else ""
        part_pos = f" X={self.x_real}, Y={self.y_real}" if self.y_real is not None else ""
        res = f"T(x={self.x:2d}, y={self.y:2d}{part_pos} t={self.type}, E={self.suggested_energy}{part_sap})"
        return f"{res:<65}"

    def add_attack(self, attack_pos: Position):
        # Task is frozen, so you need to create a new one
        return Task(
            y=self.y,
            x=self.x,
            type=self.type,
            suggested_energy=self.suggested_energy,
            y_sap=attack_pos.y,
            x_sap=attack_pos.x,
            score=self.score,
            y_real=self.y_real,
            x_real=self.x_real,
        )

    def remove_attack(self):
        # Sometimes previously a task was to attack something but this turn you do not need to attack
        return Task(
            y=self.y,
            x=self.x,
            type=self.type,
            suggested_energy=self.suggested_energy,
            y_sap=None,
            x_sap=None,
            score=self.score,
            y_real=self.y_real,
            x_real=self.x_real,
        )

    def update_energy(self, energy: int):
        return Task(
            y=self.y,
            x=self.x,
            type=self.type,
            suggested_energy=energy,
            y_sap=self.y_sap,
            x_sap=self.x_sap,
            score=self.score,
            y_real=self.y_real,
            x_real=self.x_real,
        )

@dataclass(frozen=True)
class Equation:
    equal: int
    points: Tuple[Position]

@dataclass
class ValueEstimate:
    points: float # how many points you are expected to collect. Non integers are due to uncertainty
    energy: int # how much energy are you expected to get
    other: int  # some other things like how much you see, fragments expected to find
    debug: Any = ''  # Not used anywhere in calculations, only for logging

    def __str__(self):
        res = f"Estimate(p={self.points:4.2f}, e={self.energy:3d}, o={self.other:3d}, d={self.debug})"
        return f"{res:<99}"

    def __repr__(self):
        return self.__str__()