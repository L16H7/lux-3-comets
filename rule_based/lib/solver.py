import time
from typing import List, Dict, Set, Tuple

import numpy as np
import scipy.sparse as sp
import sympy
from scipy.optimize import milp, Bounds, LinearConstraint

from lib.data_classes import Position, Equation


class Solver:

    def __init__(self):
        self.n: int = 0
        self.unique_pos: Set[Position] = set()
        self.pos_to_id: Dict[Position: int] = {}
        self.id_to_pos: List[Position] = []

    def prepare_equations(self, eq: List[Equation]) -> Tuple[List, List]:
        self.unique_pos = sorted(list({p for e in eq for p in e.points}), key=lambda p: (p.x, p.y))
        self.n = len(self.unique_pos)
        self.pos_to_id = {}
        self.id_to_pos = [Position(y=-1, x=-1)] * self.n
        for i, p in enumerate(self.unique_pos):
            self.pos_to_id[p] = i
            self.id_to_pos[i] = p

        A_eq, b_eq = [], []
        for e in eq:
            row = [0] * self.n
            for p in e.points:
                row[self.pos_to_id[p]] += 1
            b_eq.append(e.equal)
            A_eq.append(row)

        return A_eq, b_eq

    def solve(self, equations: List[Equation]) -> Tuple[Dict[Position, float], float]:
        start_time = time.perf_counter()
        A, B = self.prepare_equations(equations)

        result = sympy.linsolve((sympy.Matrix(A), sympy.Matrix(B)))
        if not result:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return {}, execution_time_ms

        res = {self.id_to_pos[i]: x for r in result for i, x in enumerate(r) if x.is_constant()}
        for p in self.pos_to_id.keys():
            if p not in res:
                res[p] = 0.5

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return res, execution_time_ms

    def solve_new(self, equations: List[Equation]) -> Tuple[Dict[Position, float], float]:
        start_time = time.perf_counter()
        A, B = self.prepare_equations(equations)

        lhs = np.array(A)
        rhs = np.array(B)
        m, n = lhs.shape

        # Variables: n * 2 (minimize, maximize) * n
        c = sp.kron(
            sp.eye_array(n),
            np.array(((+1,), (-1,))),
        )

        b = np.tile(rhs, 2 * n)
        system_constraint = LinearConstraint(A=sp.kron(sp.eye_array(2 * n), lhs, format='csc'), lb=b, ub=b)

        result = milp(
            c=c.toarray().ravel(),  # must be dense
            integrality=0,
            bounds=Bounds(lb=0, ub=1),
            constraints=system_constraint,
        )
        if not result.success:
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            return {}, execution_time_ms

        extrema = result.x.reshape((n, 2, n))
        v_min = np.diag(extrema[:, 0])
        v_max = np.diag(extrema[:, 1])

        res = {}
        for i in range(len(v_max)):
            if v_max[i] < 0.5:
                res[self.id_to_pos[i]] = 0  # Guaranteed 0
            elif v_min[i] > 0.5:
                res[self.id_to_pos[i]] = 1  # Guaranteed 1
            else:
                res[self.id_to_pos[i]] = 0.5  # Can be both

        execution_time_ms = (time.perf_counter() - start_time) * 1000
        return res, execution_time_ms
