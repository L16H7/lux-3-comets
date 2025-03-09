import unittest

import numpy as np

from lib.data_classes import Unit, Fragment, Position
from lib.enemy_tracker import EnemyTracker


def U(y: int, x: int, id: int = 1) -> Unit:
    return Unit(
        team_id=0,
        id=id,
        energy=100,
        y=y, x=x
    )

def F(y: int, x: int) -> Fragment:
    return Fragment(
        team_id=0,
        probability=1,
        y=y, x=x
    )

class TestEnemyTracker(unittest.TestCase):
    def test_enemy_tracker(self):
        ET = EnemyTracker()
        ET.update(
            mapinfo_fragments={F(0, 0), F(0, 2), F(1, 1)},
            enemy_units={U(y=1, x=2), U(y=1, x=1), U(y=2, x=2)},
            mapinfo_vision_single=np.array([
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
            ])
        )
        self.assertEqual(ET.pos_units, {
            Position(x=1, y=1): [U(x= 1, y= 1)]
        })
        self.assertEqual(ET.get_enemy(), {U(y=1, x=2), U(y=1, x=1), U(y=2, x=2)})


        ET.update(
            mapinfo_fragments={F(0, 0), F(0, 2), F(1, 1)},
            enemy_units={U(y=0, x=0), U(y=0, x=1), U(y=0, x=2)},
            mapinfo_vision_single=np.array([
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [1, 1, 1, 0],
                [0, 0, 0, 0],
            ])
        )
        self.assertEqual(ET.pos_units, {
            Position(x= 0, y=0): [U(x= 0, y= 0)],
            Position(x= 2, y=0): [U(x= 2, y= 0)]
        })
        self.assertEqual(ET.get_enemy(), {U(y=0, x=0), U(y=0, x=2), U(y=0, x=1)})

        ET.update(
            mapinfo_fragments={F(0, 0), F(0, 2), F(2, 2)},
            enemy_units={U(y=2, x=2), U(y=2, x=2, id=2)},
            mapinfo_vision_single=np.array([
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ])
        )
        self.assertEqual(ET.pos_units, {
            Position(x=0, y=0): [U(x=0, y=0)],
            Position(x=2, y=0): [U(x=2, y=0)],
            Position(x=2, y=2): [U(x=2, y=2, id=2), U(x=2, y=2)]
        })
        self.assertEqual(ET.get_enemy(), {U(x=0, y=0), U(x=2, y=0), U(x=2, y=2), U(x=2, y=2, id=2)})


r = np.load("data/relics.npy")
print(np.array2string((r > 0).astype(int), separator=', '))