from collections import defaultdict
from typing import List, Set, Dict

import numpy as np

from lib.data_classes import Position, Unit, Fragment


class EnemyTracker:

    def __init__(self):
        # Positions where we expect units to stay. Those positions are: fragments or high-energy positions
        self.pos: Set[Position] = set()

        # Stores uid -> unit for every enemy unit which stays at important positions
        self.pos_units: Dict[Position, List[Unit]] = defaultdict(list)

        # Positions of current enemy units
        self.curr_enemy_pos: Set[Unit] = set()

    def update(self, mapinfo_fragments: Set[Fragment], mapinfo_vision_single: np.ndarray[bool], enemy_units: Set[Unit]):
        self.pos = {f.pos() for f in mapinfo_fragments}

        # Do not store information for anything that is not important
        pos_remove = [p for p in self.pos_units.keys() if p not in self.pos]
        for p in pos_remove:
            self.pos_units.pop(p, None)

        # Check important positions. If I see them, clear them out as either:
        #  - there are no units there or
        #  - units will be visible and I will update them later
        for p in self.pos:
            if mapinfo_vision_single[p.y, p.x]:
                self.pos_units.pop(p, None)

        # Add all units which are in important positions
        for u in enemy_units:
            if u.pos() in self.pos:
                self.pos_units[u.pos()].append(u)

        self.curr_enemy_pos = enemy_units

    def get_enemy(self) -> Set[Unit]:
        for units in self.pos_units.values():
            for u in units:
                self.curr_enemy_pos.add(u)
        return self.curr_enemy_pos
