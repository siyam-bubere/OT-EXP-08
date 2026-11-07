"""
track_j_swe_opt/test_suite_optimizer.py
Track J: Software-Engineering Optimization (Test-Suite Minimization & Prioritization)
Optimization Techniques Lab - Assignment 8 (Chapter 6 SWE Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Components:
1. Target Codebase Simulator (Realistic 1,200+ statement numerical & data processing library)
2. Coverage Instrumentation & Execution Profiler
3. Test-Suite Minimization (TSM) via Integer Linear Programming (ILP) Set Cover:
   - Preserves 100% statement coverage while minimizing CI execution time
4. Test-Suite Prioritization (TSP) & APFD Metric:
   - Additional Statement Coverage greedy prioritization
   - Average Percentage of Faults Detected (APFD) evaluation on mutant fault injections
"""

import numpy as np
import time
from typing import List, Dict, Set, Tuple, Optional
from scipy.optimize import milp, LinearConstraint, Bounds


class TestCase:
    def __init__(self, test_id: int, name: str, execution_time_ms: float, covered_statements: Set[int], detected_faults: Set[int]):
        self.test_id = test_id
        self.name = name
        self.execution_time_ms = execution_time_ms
        self.covered_statements = covered_statements
        self.detected_faults = detected_faults


def generate_synthetic_software_project(n_statements: int = 1200, n_tests: int = 40, n_faults: int = 25, seed: int = 42) -> Tuple[List[TestCase], Set[int], Set[int]]:
    """
    Generates a realistic software test suite covering a 1,200+ line codebase:
    - Statements partitioned into core algorithms, utility parsers, error handlers, and math kernels.
    - Tests have varying runtimes (unit tests: 5-15 ms, integration tests: 80-250 ms).
    - Injected synthetic faults represent boundary errors, null checks, and logic mutants.
    """
    rng = np.random.default_rng(seed)
    all_statements = set(range(1, n_statements + 1))
    all_faults = set(range(1, n_faults + 1))

    # Map faults to specific statement locations
    fault_locations = {f: int(rng.integers(1, n_statements + 1)) for f in all_faults}

    tests = []
    # Ensure full coverage across the entire 1200 statements
    covered_so_far = set()

    for t_id in range(1, n_tests + 1):
        # Varying test types: unit (fast, focused) vs integration (slow, broad)
        is_integration = (t_id % 4 == 0)
        if is_integration:
            time_ms = float(rng.uniform(70.0, 220.0))
            n_cov = int(rng.integers(250, 500))
        else:
            time_ms = float(rng.uniform(8.0, 35.0))
            n_cov = int(rng.integers(50, 160))

        # Sample covered statements
        cov_sample = set(rng.choice(list(all_statements), size=n_cov, replace=False))

        # Prioritize uncovered statements to ensure 100% suite coverage
        uncovered = list(all_statements - covered_so_far)
        if uncovered and t_id <= 35:
            force_cov = rng.choice(uncovered, size=min(len(uncovered), 35), replace=False)
            cov_sample.update(force_cov)

        covered_so_far.update(cov_sample)

        # Faults detected by this test: faults whose location is covered by cov_sample
        detected = {f for f, loc in fault_locations.items() if loc in cov_sample and rng.random() > 0.15}

        tests.append(TestCase(
            test_id=t_id,
            name=f"test_{'integration' if is_integration else 'unit'}_{t_id}",
            execution_time_ms=time_ms,
            covered_statements=cov_sample,
            detected_faults=detected
        ))

    # Ensure any remaining uncovered statements are covered by the final tests
    remaining = all_statements - covered_so_far
    if remaining:
        tests[0].covered_statements.update(remaining)

    return tests, all_statements, all_faults


def solve_test_suite_minimization_ilp(tests: List[TestCase], all_statements: Set[int]) -> Tuple[List[TestCase], float, float]:
    """
    Formulates Test-Suite Minimization as a 0/1 Integer Linear Program (Set Cover):
      min sum_j (t_j * x_j)
      s.t. sum_{j: s in Cov(j)} x_j >= 1 for all statements s in all_statements
           x_j in {0, 1}
    """
    n_tests = len(tests)
    n_stmts = len(all_statements)
    stmt_list = sorted(list(all_statements))
    stmt_to_idx = {s: i for i, s in enumerate(stmt_list)}

    # Objective: minimize execution time
    c = np.array([t.execution_time_ms for t in tests])

    # Constraint matrix A: (n_stmts x n_tests)
    # A[i, j] = 1 if test j covers statement i
    A = np.zeros((n_stmts, n_tests))
    for j, test in enumerate(tests):
        for s in test.covered_statements:
            if s in stmt_to_idx:
                A[stmt_to_idx[s], j] = 1.0

    # Bounds: x_j in {0, 1}
    bounds = Bounds(lb=0.0, ub=1.0)
    constraints = LinearConstraint(A, lb=1.0, ub=np.inf)
    integrality = np.ones(n_tests)

    res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)

    if not res.success:
        raise RuntimeError(f"Test suite minimization ILP failed: {res.status}")

    x_sol = np.round(res.x).astype(int)
    selected_tests = [tests[j] for j in range(n_tests) if x_sol[j] == 1]

    # Verify coverage
    minimized_coverage = set()
    for t in selected_tests:
        minimized_coverage.update(t.covered_statements)

    assert len(minimized_coverage) == len(all_statements), "Minimization compromised statement coverage!"

    total_orig_time = sum(t.execution_time_ms for t in tests)
    total_min_time = sum(t.execution_time_ms for t in selected_tests)

    return selected_tests, total_orig_time, total_min_time


def prioritize_tests_greedy_additional_coverage(tests: List[TestCase], all_statements: Set[int]) -> List[TestCase]:
    """
    Test-Suite Prioritization: Additional Statement Coverage Heuristic.
    Repeatedly picks the test that covers the maximum number of currently uncovered statements.
    """
    uncovered = set(all_statements)
    remaining_tests = list(tests)
    ordered = []

    while remaining_tests:
        best_test = None
        best_gain = -1

        for t in remaining_tests:
            gain = len(t.covered_statements.intersection(uncovered))
            if gain > best_gain:
                best_gain = gain
                best_test = t

        if best_gain > 0 and best_test is not None:
            ordered.append(best_test)
            uncovered -= best_test.covered_statements
            remaining_tests.remove(best_test)
        else:
            # If all covered, append remaining tests by shortest execution time
            remaining_tests.sort(key=lambda t: t.execution_time_ms)
            ordered.extend(remaining_tests)
            break

    return ordered


def compute_apfd(ordered_tests: List[TestCase], all_faults: Set[int]) -> Tuple[float, List[float]]:
    """
    Computes Average Percentage of Faults Detected (APFD):
      APFD = 1 - (sum_{i=1}^m TF_i) / (n * m) + 1 / (2 * n)
    where:
      n = number of tests
      m = number of faults
      TF_i = position (1-based) of the first test that detects fault i
    """
    n = len(ordered_tests)
    m = len(all_faults)

    first_detected = {}
    cumulative_faults = set()
    faults_detected_curve = []

    for rank, test in enumerate(ordered_tests, start=1):
        for f in test.detected_faults:
            if f not in first_detected and f in all_faults:
                first_detected[f] = rank
        cumulative_faults.update(test.detected_faults)
        faults_detected_curve.append(len(cumulative_faults.intersection(all_faults)) / float(m) * 100.0)

    # For any undetected fault, assign position n + 1
    tf_sum = sum(first_detected.get(f, n + 1) for f in all_faults)
    apfd = 1.0 - (tf_sum / float(n * m)) + (1.0 / (2.0 * n))

    return float(apfd), faults_detected_curve
