"""
track_b_discrete/discrete_solvers.py
Complete Chapter 6 Discrete Optimization Implementations from Scratch
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Algorithms Covered:
1. Greedy Heuristic (0/1 Knapsack)
2. Exact Dynamic Programming (0/1 Knapsack DP Table)
3. Exact Branch-and-Bound (with fractional LP upper-bounding & tree pruning)
4. Integer Linear Programming (ILP via Branch-and-Cut / SciPy MILP)
5. QUBO & Ising Model Formulations:
   - Binary-to-spin transformation: x_i in {0,1} <-> s_i in {-1, +1}
   - Constraint quadratic penalty: lambda * (sum A_i x_i - B)^2
   - Max-Cut Problem formulated as Ising Hamiltonian (Track C Baseline)
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from scipy.optimize import milp, LinearConstraint, Bounds


# ---------------------------------------------------------------------------
# 0/1 Knapsack Solvers
# ---------------------------------------------------------------------------

def solve_knapsack_greedy(values: np.ndarray, weights: np.ndarray, capacity: float) -> Dict:
    """Greedy heuristic based on value-to-weight ratio."""
    t0 = time.time()
    n = len(values)
    ratios = values / weights
    order = np.argsort(-ratios)

    total_val = 0.0
    total_wt = 0.0
    selected = np.zeros(n, dtype=int)

    for idx in order:
        if total_wt + weights[idx] <= capacity:
            selected[idx] = 1
            total_val += values[idx]
            total_wt += weights[idx]

    dur = time.time() - t0
    return {
        "method": "Greedy Ratio",
        "value": total_val,
        "weight": total_wt,
        "selected": selected,
        "runtime_s": dur
    }


def solve_knapsack_dp(values: np.ndarray, weights: np.ndarray, capacity: int) -> Dict:
    """Exact Dynamic Programming solver: O(n * W). Capacity and weights must be integer."""
    t0 = time.time()
    n = len(values)
    W = int(capacity)
    w_int = weights.astype(int)
    v_arr = values.astype(float)

    # DP Table: (n + 1, W + 1)
    dp = np.zeros((n + 1, W + 1), dtype=float)

    for i in range(1, n + 1):
        wi = w_int[i - 1]
        vi = v_arr[i - 1]
        for w in range(W + 1):
            if wi <= w:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - wi] + vi)
            else:
                dp[i, w] = dp[i - 1, w]

    # Backtracking to reconstruct solution
    selected = np.zeros(n, dtype=int)
    w_curr = W
    for i in range(n, 0, -1):
        if dp[i, w_curr] != dp[i - 1, w_curr]:
            selected[i - 1] = 1
            w_curr -= w_int[i - 1]

    dur = time.time() - t0
    return {
        "method": "Dynamic Programming",
        "value": dp[n, W],
        "weight": np.sum(selected * weights),
        "selected": selected,
        "runtime_s": dur
    }


class BranchAndBoundKnapsack:
    """Exact Branch-and-Bound with fractional knapsack upper bound bounding."""
    def __init__(self, values: np.ndarray, weights: np.ndarray, capacity: float):
        self.n = len(values)
        self.capacity = capacity
        # Sort items by value-to-weight ratio descending
        ratios = values / weights
        self.order = np.argsort(-ratios)
        self.values = values[self.order]
        self.weights = weights[self.order]

        self.best_value = 0.0
        self.best_selection = np.zeros(self.n, dtype=int)
        self.nodes_explored = 0

    def fractional_upper_bound(self, level: int, curr_val: float, curr_wt: float) -> float:
        if curr_wt >= self.capacity:
            return 0.0
        bound = curr_val
        w = curr_wt
        for i in range(level, self.n):
            if w + self.weights[i] <= self.capacity:
                w += self.weights[i]
                bound += self.values[i]
            else:
                bound += (self.capacity - w) * (self.values[i] / self.weights[i])
                break
        return bound

    def search(self, level: int, curr_val: float, curr_wt: float, selection: np.ndarray):
        self.nodes_explored += 1
        if curr_wt <= self.capacity and curr_val > self.best_value:
            self.best_value = curr_val
            self.best_selection = np.copy(selection)

        if level == self.n or curr_wt >= self.capacity:
            return

        # Prune if upper bound <= current best known solution
        ub = self.fractional_upper_bound(level, curr_val, curr_wt)
        if ub <= self.best_value:
            return

        # Branch 1: Include item at 'level'
        if curr_wt + self.weights[level] <= self.capacity:
            selection[level] = 1
            self.search(level + 1, curr_val + self.values[level], curr_wt + self.weights[level], selection)
            selection[level] = 0

        # Branch 2: Exclude item at 'level'
        self.search(level + 1, curr_val, curr_wt, selection)

    def solve(self) -> Dict:
        t0 = time.time()
        self.search(0, 0.0, 0.0, np.zeros(self.n, dtype=int))
        dur = time.time() - t0

        # Un-sort selection
        final_sel = np.zeros(self.n, dtype=int)
        for i, idx in enumerate(self.order):
            final_sel[idx] = self.best_selection[i]

        return {
            "method": "Branch-and-Bound",
            "value": self.best_value,
            "weight": np.sum(final_sel * (self.values / (self.values / self.weights))),
            "selected": final_sel,
            "nodes_explored": self.nodes_explored,
            "runtime_s": dur
        }


def solve_knapsack_ilp(values: np.ndarray, weights: np.ndarray, capacity: float) -> Dict:
    """Exact Integer Linear Programming (ILP) using Branch-and-Cut."""
    t0 = time.time()
    n = len(values)
    # Objective: maximize values @ x <=> minimize -values @ x
    c = -values
    # Constraint: weights @ x <= capacity
    constraints = LinearConstraint(weights.reshape(1, -1), lb=0, ub=capacity)
    # Integrality: 1 = integer variable in [0, 1]
    integrality = np.ones(n)
    bounds = Bounds(lb=0.0, ub=1.0)

    res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
    dur = time.time() - t0

    selected = np.round(res.x).astype(int)
    return {
        "method": "ILP (Branch-and-Cut)",
        "value": float(-res.fun),
        "weight": float(weights @ selected),
        "selected": selected,
        "runtime_s": dur,
        "status": res.status
    }


# ---------------------------------------------------------------------------
# QUBO & Ising Model Formulations
# ---------------------------------------------------------------------------

class MaxCutProblem:
    """
    Max-Cut graph problem formulated as both QUBO and Ising model.
    Graph: G = (V, E) with weights w_uv.
    Cut value = sum_{(u,v) in E} w_uv * [x_u (1 - x_v) + x_v (1 - x_u)]
    Binary-to-spin mapping: x_u in {0, 1} <-> s_u in {-1, +1} via x_u = (1 - s_u)/2.
    Ising Hamiltonian: H_Ising = -1/2 sum_{(u,v) in E} w_uv * (1 - s_u s_v).
    Minimizing H_Ising minimizes energy <=> maximizes cut value!
    """
    def __init__(self, n_nodes: int, edges: List[Tuple[int, int, float]]):
        self.n = n_nodes
        self.edges = edges
        self.adj = np.zeros((n_nodes, n_nodes))
        for u, v, w in edges:
            self.adj[u, v] = w
            self.adj[v, u] = w

    def get_qubo_matrix(self) -> np.ndarray:
        """Constructs QUBO matrix Q such that Cut = x^T Q x."""
        Q = np.zeros((self.n, self.n))
        for u, v, w in self.edges:
            # -w_uv * (x_u + x_v - 2 x_u x_v) for minimization
            Q[u, u] += w
            Q[v, v] += w
            Q[u, v] -= 2.0 * w
            Q[v, u] -= 2.0 * w
        return Q

    def get_ising_hamiltonian(self) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """
        Returns constant offset C and two-body interaction couplings J_uv:
        H(s) = C + sum_{u < v} J_uv s_u s_v
        """
        total_w = sum(w for _, _, w in self.edges)
        C = -0.5 * total_w
        J = {}
        for u, v, w in self.edges:
            pair = (min(u, v), max(u, v))
            J[pair] = 0.5 * w
        return C, J

    def evaluate_cut(self, partition: np.ndarray) -> float:
        """partition in {0, 1}^n."""
        cut = 0.0
        for u, v, w in self.edges:
            if partition[u] != partition[v]:
                cut += w
        return cut

    def solve_exact_brute_force(self) -> Dict:
        """Exact classical solver enumerating all 2^n configurations."""
        t0 = time.time()
        best_cut = -1.0
        best_part = None
        for i in range(1 << self.n):
            part = np.array([(i >> bit) & 1 for bit in range(self.n)])
            cut = self.evaluate_cut(part)
            if cut > best_cut:
                best_cut = cut
                best_part = part
        dur = time.time() - t0
        return {
            "method": "Exact Classical Brute Force",
            "max_cut": best_cut,
            "partition": best_part,
            "runtime_s": dur
        }


def formulate_knapsack_qubo(values: np.ndarray, weights: np.ndarray, capacity: float, penalty_lambda: float = 10.0) -> np.ndarray:
    """
    Formulates 0/1 Knapsack with penalty term as QUBO:
    min x^T Q x = - sum v_i x_i + lambda * (sum w_i x_i - capacity)^2.
    """
    n = len(values)
    Q = np.zeros((n, n))

    # Linear term from objective: -v_i x_i
    for i in range(n):
        Q[i, i] -= values[i]

    # Penalty expansion: lambda * (sum w_i x_i - C)^2
    # = lambda * [ sum w_i^2 x_i + 2 sum_{i < j} w_i w_j x_i x_j - 2 C sum w_i x_i + C^2 ]
    for i in range(n):
        Q[i, i] += penalty_lambda * (weights[i]**2 - 2.0 * capacity * weights[i])
        for j in range(i + 1, n):
            Q[i, j] += penalty_lambda * 2.0 * weights[i] * weights[j]
            Q[j, i] += penalty_lambda * 2.0 * weights[i] * weights[j]

    return Q
