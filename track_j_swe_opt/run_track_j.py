"""
track_j_swe_opt/run_track_j.py - Benchmarks & Visualizations for Track J (SWE Optimization)
Optimization Techniques Lab - Assignment 8 (Chapter 6 SWE Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from track_j_swe_opt.test_suite_optimizer import (
    generate_synthetic_software_project,
    solve_test_suite_minimization_ilp,
    prioritize_tests_greedy_additional_coverage,
    compute_apfd
)


def main():
    print("================================================================================")
    print("EXECUTING TRACK J: SOFTWARE ENGINEERING OPTIMIZATION (TSMP)")
    print("================================================================================")

    # 1. Generate Software Project Benchmark (1,200 statements, 40 tests, 25 faults)
    n_stmts = 1200
    n_tests = 40
    n_faults = 25
    tests, all_stmts, all_faults = generate_synthetic_software_project(
        n_statements=n_stmts, n_tests=n_tests, n_faults=n_faults, seed=42
    )
    total_orig_time = sum(t.execution_time_ms for t in tests)
    print(f"\n--- Codebase & Test Suite Profile ---")
    print(f"  Target Codebase Statements : {len(all_stmts):,}")
    print(f"  Total Test Cases in Suite  : {len(tests)}")
    print(f"  Total CI Execution Time    : {total_orig_time:,.1f} ms")
    print(f"  Injected Synthetic Faults  : {len(all_faults)}")

    # 2. Test-Suite Minimization via ILP (Set Cover)
    print(f"\n--- 1. Test-Suite Minimization via 0/1 Integer Linear Programming ---")
    min_tests, orig_time, min_time = solve_test_suite_minimization_ilp(tests, all_stmts)
    time_saved = orig_time - min_time
    time_saved_pct = (time_saved / orig_time) * 100.0
    tests_pruned = len(tests) - len(min_tests)
    tests_pruned_pct = (tests_pruned / len(tests)) * 100.0

    print(f"  Selected Tests for CI      : {len(min_tests)} of {len(tests)} ({tests_pruned} redundant tests pruned)")
    print(f"  Pruned Test Ratio          : {tests_pruned_pct:.1f}% reduction")
    print(f"  Minimized CI Runtime       : {min_time:,.1f} ms (Saved: {time_saved:,.1f} ms, {time_saved_pct:.1f}% faster)")
    print(f"  Statement Coverage         : 100.0% PRESERVED ({len(all_stmts)}/{len(all_stmts)} statements)")

    # 3. Test-Suite Prioritization (TSP) & APFD Evaluation
    print(f"\n--- 2. Test-Suite Prioritization (APFD Optimization) ---")

    # Order A: Random Ordering (Standard developer / default execution)
    rng = np.random.default_rng(42)
    tests_random = list(tests)
    rng.shuffle(tests_random)
    apfd_random, curve_random = compute_apfd(tests_random, all_faults)

    # Order B: Shortest Job First (Execution Time Ascending)
    tests_sjf = sorted(tests, key=lambda t: t.execution_time_ms)
    apfd_sjf, curve_sjf = compute_apfd(tests_sjf, all_faults)

    # Order C: Optimized Additional Coverage Prioritization
    tests_opt = prioritize_tests_greedy_additional_coverage(tests, all_stmts)
    apfd_opt, curve_opt = compute_apfd(tests_opt, all_faults)

    print(f"  Random Ordering APFD                   : {apfd_random:.3f} ({apfd_random*100.0:.1f}%)")
    print(f"  Shortest-Job-First (SJF) APFD          : {apfd_sjf:.3f} ({apfd_sjf*100.0:.1f}%)")
    print(f"  Optimized Additional Coverage APFD     : {apfd_opt:.3f} ({apfd_opt*100.0:.1f}%)")
    print(f"  -> APFD Improvement over Random        : +{(apfd_opt - apfd_random)*100.0:.1f}% points")

    # Save Metrics CSV
    records = [
        {"Optimization_Phase": "Test Suite Minimization", "Metric": "Test Count", "Baseline": len(tests), "Optimized": len(min_tests), "Improvement_%": tests_pruned_pct},
        {"Optimization_Phase": "Test Suite Minimization", "Metric": "CI Runtime (ms)", "Baseline": orig_time, "Optimized": min_time, "Improvement_%": time_saved_pct},
        {"Optimization_Phase": "Test Suite Minimization", "Metric": "Statement Coverage (%)", "Baseline": 100.0, "Optimized": 100.0, "Improvement_%": 0.0},
        {"Optimization_Phase": "Test Prioritization (APFD)", "Metric": "Random vs Optimized APFD", "Baseline": apfd_random, "Optimized": apfd_opt, "Improvement_%": (apfd_opt - apfd_random)*100.0},
        {"Optimization_Phase": "Test Prioritization (APFD)", "Metric": "SJF vs Optimized APFD", "Baseline": apfd_sjf, "Optimized": apfd_opt, "Improvement_%": (apfd_opt - apfd_sjf)*100.0}
    ]
    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv("results/track_j_swe_metrics.csv", index=False)
    print("\n  Saved: results/track_j_swe_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Cumulative Fault Detection Curves (APFD Comparison)
    plt.figure(figsize=(7, 4.5))
    x_axis = range(1, len(tests) + 1)
    plt.plot(x_axis, curve_opt, 'g-o', linewidth=2.5, markersize=5, label=f"Optimized Additional Coverage (APFD = {apfd_opt:.3f})")
    plt.plot(x_axis, curve_sjf, 'b--', linewidth=2, label=f"Shortest Job First (APFD = {apfd_sjf:.3f})")
    plt.plot(x_axis, curve_random, 'r:', linewidth=2, label=f"Random Default Order (APFD = {apfd_random:.3f})")

    plt.xlabel("Number of Tests Executed in CI")
    plt.ylabel("Cumulative Faults Detected (%)")
    plt.title("Track J: Test Prioritization & Fault Detection Velocity (APFD)", fontweight='bold')
    plt.xlim(1, len(tests))
    plt.ylim(0, 105)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_j_apfd_fault_curves.png", dpi=200)
    plt.close()

    # Plot 2: Test Suite Minimization (Tests & Runtime Reduction)
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    ax2 = ax1.twinx()

    categories = ['Original Suite', 'ILP Minimized']
    test_counts = [len(tests), len(min_tests)]
    runtimes = [orig_time, min_time]

    x = np.arange(len(categories))
    w = 0.35

    b1 = ax1.bar(x - w/2, test_counts, width=w, color='steelblue', label="Active Tests in Suite", alpha=0.85)
    b2 = ax2.bar(x + w/2, runtimes, width=w, color='coral', label="CI Execution Time (ms)", alpha=0.85)

    ax1.set_ylabel("Number of Tests", color='steelblue', fontweight='bold')
    ax2.set_ylabel("CI Execution Time (ms)", color='coral', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight='bold')
    ax1.set_ylim(0, max(test_counts) * 1.3)
    ax2.set_ylim(0, max(runtimes) * 1.3)

    plt.title(f"Track J: CI Test Minimization ({time_saved_pct:.1f}% Faster, 100% Coverage)", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/track_j_test_minimization.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_j_apfd_fault_curves.png, track_j_test_minimization.png")
    print("\n[DONE] Track J Execution Complete!\n")


if __name__ == "__main__":
    main()
