# Vibe-Coding Prompt Log: Track B — Discrete Optimization
**Assignment:** Optimization Techniques Lab — Assignment 8  
**Student:** Siyam Bubere (Reg. Id: 241070012) | VJTI Mumbai  
**Track:** Track B (Discrete Optimization)  
**Tools Used:** Antigravity AI, Claude 3.7 Sonnet  
**Date:** November 2026  

---

## 1. Initial Prompt (Problem Formulation & Implementation)
```text
Task: Implement from scratch the Chapter 6 discrete optimization suite:
1. 0/1 Knapsack problem comparing:
   - Greedy value-to-weight heuristic
   - Exact Dynamic Programming (DP table)
   - Exact Branch-and-Bound (with fractional LP upper-bound pruning)
   - Integer Linear Programming (ILP via scipy.optimize.milp)
2. Measure execution runtime scaling and optimality gap across problem sizes N in [6, 10, 14, 18, 22, 26].
3. QUBO & Ising Model Formulations:
   - Derive binary-to-spin mapping x_i in {0,1} <-> s_i in {-1, +1}
   - Formulate constraint penalty terms lambda * (sum A_i x_i - B)^2
   - Formulate the 8-node Max-Cut graph problem as an Ising Hamiltonian H = -0.5 * sum w_uv (1 - s_u s_v)
   - Solve Max-Cut classically with an exact ground-truth solver to establish the baseline for Track C QAOA.
```

## 2. AI Response Summary & Code Generation
- Generated `discrete_solvers.py` with 0/1 Knapsack solvers and `MaxCutProblem` class.
- Formulated the Ising Hamiltonian $H = C + \sum_{u < v} J_{uv} s_u s_v$ where $J_{uv} = 0.5 w_{uv}$ and $C = -0.5 \sum w_{uv}$.
- Generated the scaling benchmark across $N \in [6, 26]$.

## 3. Human Critical Review & AI Failure Rectification (Documented Bug 1)
- **AI Tool Error Detected:**
  - The AI assistant passed `bounds=[(0, 1) for _ in range(n)]` to `scipy.optimize.milp`.
  - **Crash:** `ValueError: bounds must be convertible into an instance of scipy.optimize.Bounds`. In modern SciPy, `milp` expects `Bounds(lb=0.0, ub=1.0)` rather than a list of tuples.
  - **Correction:** Replaced with `from scipy.optimize import Bounds; bounds = Bounds(lb=0.0, ub=1.0)`.
  - **Lesson:** Always verify API version contracts for modern SciPy optimization routines rather than relying on legacy `linprog` syntax.

## 4. Benchmark Results & Traceability
- **Exact Max-Cut Value:** 14.00 achieved on partition $[1, 1, 0, 0, 1, 1, 0, 0]$.
- **Greedy Approximation:** Maintained between 96.2% and 100.0% of the true DP/ILP optimum across all problem sizes.
- **Runtime Comparison:** DP and B&B solved within $<10$ ms; ILP Branch-and-Cut solved within $<45$ ms.
- **Artifacts Generated:**
  - `track_b_discrete/discrete_solvers.py`
  - `track_b_discrete/run_track_b.py`
  - `results/track_b_discrete_metrics.csv`
  - `results/track_b_maxcut_ground_truth.npz`
  - `plots/track_b_scaling_runtime.png`
  - `plots/track_b_maxcut_graph.png`
