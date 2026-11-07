"""
track_g_xv6/run_track_g.py - Benchmarks & Visualizations for Track G (xv6 Scheduler)
Optimization Techniques Lab - Assignment 8 (Chapter 6 OS Optimization)
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
from track_g_xv6.xv6_scheduler_sim import (
    create_workload_1_cpu_bound,
    create_workload_2_io_bound,
    create_workload_3_mixed,
    simulate_round_robin,
    simulate_mlfq_priority
)


def main():
    print("================================================================================")
    print("EXECUTING TRACK G: XV6 OPERATING SYSTEM SCHEDULER OPTIMIZATION")
    print("================================================================================")

    workloads = [
        ("Workload 1 (CPU-Bound)", create_workload_1_cpu_bound),
        ("Workload 2 (I/O-Bound)", create_workload_2_io_bound),
        ("Workload 3 (Mixed Batch/Interactive)", create_workload_3_mixed)
    ]

    records = []

    for name, builder in workloads:
        print(f"\n--- Benchmarking: {name} ---")
        procs = builder()
        res_rr = simulate_round_robin(procs)
        res_mlfq = simulate_mlfq_priority(procs)

        print(f"  Standard Round-Robin : Turnaround={res_rr['avg_turnaround_ticks']:.1f}t, Response={res_rr['avg_response_ticks']:.1f}t, Wait={res_rr['avg_wait_ticks']:.1f}t")
        print(f"  Optimized MLFQ+Aging : Turnaround={res_mlfq['avg_turnaround_ticks']:.1f}t, Response={res_mlfq['avg_response_ticks']:.1f}t, Wait={res_mlfq['avg_wait_ticks']:.1f}t")

        resp_impr = (res_rr['avg_response_ticks'] - res_mlfq['avg_response_ticks']) / max(0.1, res_rr['avg_response_ticks']) * 100.0
        turn_impr = (res_rr['avg_turnaround_ticks'] - res_mlfq['avg_turnaround_ticks']) / max(0.1, res_rr['avg_turnaround_ticks']) * 100.0
        print(f"  -> Response Time Improvement : {resp_impr:.1f}%")
        print(f"  -> Turnaround Improvement    : {turn_impr:.1f}%")

        records.append({
            "Workload": name,
            "RR_Avg_Turnaround": res_rr['avg_turnaround_ticks'],
            "MLFQ_Avg_Turnaround": res_mlfq['avg_turnaround_ticks'],
            "RR_Avg_Response": res_rr['avg_response_ticks'],
            "MLFQ_Avg_Response": res_mlfq['avg_response_ticks'],
            "RR_Avg_Wait": res_rr['avg_wait_ticks'],
            "MLFQ_Avg_Wait": res_mlfq['avg_wait_ticks'],
            "Response_Improvement_%": resp_impr,
            "Turnaround_Improvement_%": turn_impr
        })

    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv("results/track_g_xv6_metrics.csv", index=False)
    print("\n  Saved: results/track_g_xv6_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Response Time Comparison Across Workloads
    plt.figure(figsize=(7, 4.5))
    x = np.arange(len(records))
    w = 0.35
    w_names = ["CPU-Bound", "I/O-Bound", "Mixed Workload"]
    rr_resp = [r["RR_Avg_Response"] for r in records]
    mlfq_resp = [r["MLFQ_Avg_Response"] for r in records]

    plt.bar(x - w/2, rr_resp, width=w, label="Standard xv6 Round-Robin", color='crimson', alpha=0.8)
    plt.bar(x + w/2, mlfq_resp, width=w, label="Optimized xv6 MLFQ + Aging", color='forestgreen', alpha=0.85)

    for i in range(len(records)):
        plt.text(x[i] - w/2, rr_resp[i] + 0.3, f"{rr_resp[i]:.1f}t", ha='center', fontweight='bold', fontsize=8.5)
        plt.text(x[i] + w/2, mlfq_resp[i] + 0.3, f"{mlfq_resp[i]:.1f}t", ha='center', fontweight='bold', fontsize=8.5)

    plt.xticks(x, w_names, fontweight='bold')
    plt.ylabel("Average Response Time (Kernel Ticks)")
    plt.title("Track G: xv6 Process Response Latency Comparison", fontweight='bold')
    plt.ylim(0, max(rr_resp) * 1.3)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_g_scheduler_response_time.png", dpi=200)
    plt.close()

    # Plot 2: Average Turnaround Time Comparison
    plt.figure(figsize=(7, 4.5))
    rr_turn = [r["RR_Avg_Turnaround"] for r in records]
    mlfq_turn = [r["MLFQ_Avg_Turnaround"] for r in records]

    plt.bar(x - w/2, rr_turn, width=w, label="Standard xv6 Round-Robin", color='indianred', alpha=0.8)
    plt.bar(x + w/2, mlfq_turn, width=w, label="Optimized xv6 MLFQ + Aging", color='teal', alpha=0.85)

    for i in range(len(records)):
        plt.text(x[i] - w/2, rr_turn[i] + 2, f"{rr_turn[i]:.1f}t", ha='center', fontweight='bold', fontsize=8.5)
        plt.text(x[i] + w/2, mlfq_turn[i] + 2, f"{mlfq_turn[i]:.1f}t", ha='center', fontweight='bold', fontsize=8.5)

    plt.xticks(x, w_names, fontweight='bold')
    plt.ylabel("Average Turnaround Time (Kernel Ticks)")
    plt.title("Track G: xv6 Job Turnaround Time Across Workloads", fontweight='bold')
    plt.ylim(0, max(rr_turn) * 1.25)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_g_workload_metrics.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_g_scheduler_response_time.png, track_g_workload_metrics.png")
    print("\n[DONE] Track G Execution Complete!\n")


if __name__ == "__main__":
    main()
