from logging import Logger
from typing import List, Dict, Tuple, Set

import numpy as np

from lib.data_classes import Position, Unit, Equation, MapInfo
from lib.solver import Solver


class FragmentFinder:

    def __init__(self):
        self.fragments_guaranteed_pos: Set[Position] = set()
        self.fragments_possible: np.ndarray[bool] = np.array((24, 24), dtype=bool)
        self.equations: List[Equation] = []
        self.last_solutions: Dict[Position, float] = {}
        self.logger = None

    def _simplify_equation(self, unique_pos: Tuple[Position], points_delta: int) -> Equation:
        """Takes an equation and removes all the guaranteed information about it
        So let's assume our equation is `p1 + p2 + p3 + p4 + p5 = 2`, but we know that `p1 = 1` and `p2 = p3 = 0`.
        Then equation will be `p4 + p5 = 1`. It also sorts the points so we can compare equations
        """
        if not unique_pos:
            return Equation(points=tuple([]), equal=0)

        equation_points: List[Position] = []
        for p in unique_pos:
            p = p.normalize()
            if p in self.fragments_guaranteed_pos:
                points_delta -= 1
            elif self.fragments_possible[p.y, p.x]:
                equation_points.append(p)

        if points_delta < 0:
            # Can happen on change of matches. T101, T202, ...
            return Equation(points=tuple([]), equal=0)

        return Equation(
            points=tuple(sorted(equation_points, key=lambda p: (p.y, p.x))),
            equal=points_delta,
        )

    def _compact_equations(self, equations: List[Equation]) -> List[Equation]:
        """Takes the system of equations and simplifies the whole system by
        - add all equations `p_i = 0` and `p_i = 1` from `fragment_map`
        - for all other equations it simplifies it and add non-zero equations

        Then it removes all duplicates
        """
        eq_new = [Equation(equal=1, points=tuple([Position(y=p.y, x=p.x)])) for p in self.fragments_guaranteed_pos]
        for e in equations:
            eq = self._simplify_equation(e.points, e.equal)
            if eq.points:
              eq_new.append(eq)
        return list(set(eq_new))

    def solve(self, units: List[Unit], points_delta: int, fragments_guaranteed_pos: Set[Position], fragments_possible: np.ndarray[bool], solver: Solver, step: int, logger: Logger) -> Tuple[Dict[Position, float], bool]:
        self.logger = logger
        self.fragments_guaranteed_pos = fragments_guaranteed_pos
        self.fragments_possible = fragments_possible

        eq = self._simplify_equation(tuple({u.pos() for u in units if u.energy >= 0}), points_delta)
        if not eq.points:
            self.logger.debug(f"No new equations were added")
            return self.last_solutions, True

        self.equations.append(eq)
        self.equations = self._compact_equations(self.equations)

        solutions, time_ms = solver.solve_new(self.equations)
        self.logger.info(f"Solved {len(self.equations)} equations. Found probabilities in {time_ms:.2f} ms")

        if len(solutions) == 0:
            self.logger.critical(f"Somehow lost solutions on T{step}. Recovering")
            self.equations = []
            self.last_solutions = {}
            return self.last_solutions, False

        self.last_solutions = solutions
        return self.last_solutions, True

