# Track D: Submodular, Parallel, and Distributed Optimization - Prompt Log

## Overview
- **Track**: Track D (Chapter 6 Submodular, Parallel, and Distributed Optimization)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. Submodular Sensor Coverage: Diminishing returns property validation $F(A \cup \{e\}) - F(A) \ge F(B \cup \{e\}) - F(B)$, greedy vs exact enumeration, $(1 - 1/e \approx 63.2\%)$ theoretical approximation guarantee.
  2. Parallel Program Optimization: Workload scheduling (dynamic chunking vs static partitioning), empirical speedup and efficiency across $p \in \{1, 2, 4, 8\}$ workers, Amdahl's Law and Gustafson's Law fits.
  3. Federated Learning: FedAvg with non-IID client partitions across multiple local epoch configurations ($E \in \{1, 5, 20\}$) benchmarked against Centralized GD.
  4. Multi-Task Optimization: Conflicting objective balancing comparing Equal Weighting, Multiple-Gradient Descent Algorithm (MGDA / Pareto stationary minimum norm), and GradNorm dynamic balancing.

---

## Prompt 1: Submodular Formulation & Diminishing Returns Test
```text
Role: Optimization Theorist & Algorithm Designer
Context: Chapter 6 Submodular Optimization
Task:
1. Define a submodular set function F(S) representing 2D geometric sensor coverage over discrete targets:
   F(S) = | \bigcup_{i \in S} Cover(i) |
2. Write a verification method that empirically tests diminishing returns over random trials:
   For A subset of B and e not in B, test: F(A U {e}) - F(A) >= F(B U {e}) - F(B).
3. Implement greedy selection: at each step pick e* = argmax [F(S U {e}) - F(S)].
4. Implement exact brute-force search over C(N, k) combinations for small k to verify Nemhauser et al. bound:
   F(S_greedy) >= (1 - 1/e) * F(S_opt).
```

### AI Response Summary
- Produced `SensorCoverageSubmodular` class with spatial distance computations and precomputed coverage index sets.
- Implemented `verify_diminishing_returns(n_trials=100)`: strictly verified submodularity with minimum gain difference $\ge 0$.
- Evaluated greedy vs exact for $k \in \{2, 3, 4, 5\}$, achieving 100.0% of OPT across all tested budgets, well exceeding the 63.2% theoretical worst-case bound.

---

## Prompt 2: Parallel Workload Scheduling & Scaling Laws
```text
Role: Systems & High-Performance Computing Engineer
Context: Amdahl's Law vs Gustafson's Law in parallel optimization
Task:
1. Implement a chunked computational kernel with non-uniform complexity.
2. Execute with ThreadPoolExecutor across p in {1, 2, 4, 8} worker threads under static vs dynamic scheduling.
3. Compute speedup S(p) = T(1) / T(p) and efficiency E(p) = S(p) / p.
4. Estimate parallel fraction s from empirical speedup and plot theoretical curves:
   - Amdahl's Law: S(p) = 1 / ((1 - s) + s / p)
   - Gustafson's Law: S_scaled(p) = s * p + (1 - s)
```

### AI Response Summary & Vibe-Coding Refinement (AI Failure Case 3)
- **Initial Draft**: The AI initially generated a pure Python scalar loop computing `int(500 + 400 * np.sin(val))` transcendental functions per element over 80,000 floats.
- **Flaw**: Because the operations were executed in Python bytecode rather than vectorized C routines, Python's Global Interpreter Lock (GIL) serialized the worker threads in `ThreadPoolExecutor`, resulting in CPU thrashing, inflated runtimes (>60s), and sub-linear slowdown at higher thread counts.
- **Refinement**: Refactored the kernel to utilize NumPy vectorized array operations (`x = np.sin(x) * np.cos(x) + np.sqrt(np.abs(x) + 1.0)`) which release the GIL in NumPy C-extensions, reducing execution time to 0.111s and demonstrating true thread scaling.

---

## Prompt 3: Federated Averaging & Multi-Task Gradient Balancing
```text
Role: Distributed ML & Pareto Optimization Specialist
Context: Non-IID Federated Learning and Multi-Task Gradient Conflicts
Task:
1. Implement FedAvg for linear regression across heterogeneous (non-IID) client data partitions with mean shifts.
2. Compare convergence across local epochs E in {1, 5, 20} against centralized baseline gradient descent.
3. Formulate a 2-task linear model with shared representation W_shared and disparate task magnitudes (1x vs 5x).
4. Implement:
   - Equal Weighting: (g1 + g2) / 2
   - MGDA: Analytical min-norm direction alpha*g1 + (1-alpha)*g2 in convex hull
   - GradNorm: Dynamic adjustment of task weights based on relative training pace r_i = (L_i / L_i(0)) / mean(L / L(0))
```

### AI Response Summary
- Produced `FedAvgSimulation` and `run_multitask_comparison`.
- FedAvg achieved rapid loss reduction from 1.2208 ($E=1$) to 0.0307 ($E=5$) and 0.0204 ($E=20$).
- Under Equal Weighting, the high-magnitude task dominated, driving Task 2 loss to 0.0000 while Task 1 stalled at 0.2641. GradNorm balanced the tasks effectively, achieving Task 1 loss = 0.0062 and Task 2 loss = 0.0782.
