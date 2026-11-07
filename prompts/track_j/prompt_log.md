# Track J: Software-Engineering Optimization - Prompt Log

## Overview
- **Track**: Track J (Chapter 6 Software-Engineering Process Optimization)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. Software Codebase & Test Suite Simulation (1,200 statements, 40 tests, 25 synthetic injected mutants/faults).
  2. Test-Suite Minimization (TSM) via 0/1 Integer Linear Programming (Set Cover):
     $$\min \sum_{j=1}^m t_j x_j \quad \text{s.t.} \sum_{j: s \in \text{Cov}(j)} x_j \ge 1 \quad \forall s$$
  3. Test-Suite Prioritization (TSP) via Additional Statement Coverage heuristic.
  4. Average Percentage of Faults Detected (APFD) evaluation comparing Random vs Shortest-Job-First (SJF) vs Optimized Additional Coverage.

---

## Prompt 1: Test-Suite Minimization via 0/1 ILP Set Cover
```text
Role: Software Quality & DevOps Optimization Engineer
Context: CI/CD Pipeline Acceleration & Test-Suite Minimization
Task:
1. Model a software project with 1,200 executable statements and 40 unit/integration tests with non-uniform runtimes.
2. Ingest test coverage matrix A in {0, 1}^(1200 x 40).
3. Formulate Test-Suite Minimization as a 0/1 Integer Linear Program (ILP):
   - Objective: Minimize total execution runtime sum_j (t_j * x_j).
   - Constraints: Enforce every statement s in {1..1200} is covered by at least one selected test.
4. Solve via SciPy MILP and report runtime reduction while verifying 100% statement coverage.
```

### AI Response Summary
- Produced `solve_test_suite_minimization_ilp`.
- The ILP solver pruned 26 redundant tests (65.0% reduction), reducing CI execution time from 2,226.4 ms down to 630.6 ms (71.7% faster) while mathematically guaranteeing 100.0% statement coverage.

---

## Prompt 2: Test Prioritization & APFD Metric
```text
Role: Testing & Reliability Specialist
Context: Test Prioritization for Rapid Feedback
Task:
1. Inject 25 synthetic fault mutants across diverse code modules.
2. Implement Additional Statement Coverage prioritization: iteratively pick the test covering the highest number of currently uncovered statements.
3. Implement the standard IEEE APFD (Average Percentage of Faults Detected) metric:
   APFD = 1 - (sum_{i=1}^m TF_i) / (n * m) + 1 / (2 * n)
4. Compare fault detection velocity across Random, SJF, and Additional Coverage orderings.
```

### AI Response Summary
- Implemented `prioritize_tests_greedy_additional_coverage` and `compute_apfd`.
- Additional Coverage Prioritization achieved an outstanding APFD score of 0.926 (92.6%), discovering 80% of faults in the first 6 tests, significantly outperforming random execution (APFD = 0.835) and SJF (APFD = 0.739).
