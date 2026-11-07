"""
track_e_case_study/run_track_e.py - Benchmarking & Plotting for Track E Case Study
Optimization Techniques Lab - Assignment 8 (Chapter 6 Open-Domain Case Study)
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
from track_e_case_study.microgrid_dispatch import MicrogridSystem, solve_heuristic_baseline, solve_optimal_dispatch


def main():
    print("================================================================================")
    print("EXECUTING TRACK E: SMART MICROGRID REAL-PROCESS DISPATCH OPTIMIZATION")
    print("================================================================================")

    mg = MicrogridSystem(seed=42)

    # 1. Solve Baseline Heuristic
    res_base = solve_heuristic_baseline(mg)
    print("\n--- Baseline Heuristic Priority Dispatch ---")
    print(f"  Total Daily Cost:      ${res_base['total_cost']:.2f}")
    print(f"  Grid Import Cost:      ${res_base['cost_import']:.2f}")
    print(f"  Export Revenue:        ${res_base['rev_export']:.2f}")
    print(f"  Peak Demand Charge:    ${res_base['cost_peak']:.2f} (Peak: {res_base['peak_demand_kw']:.1f} kW)")
    print(f"  Battery Wear Cost:     ${res_base['cost_degradation']:.2f}")
    print(f"  Diesel Fuel Cost:      ${res_base['cost_diesel']:.2f}")

    # 2. Solve Optimal Dispatch
    res_opt = solve_optimal_dispatch(mg)
    print("\n--- Optimal Dispatch (Convex LP/MILP with Epigraph Peak Shaving) ---")
    print(f"  Total Daily Cost:      ${res_opt['total_cost']:.2f}")
    print(f"  Grid Import Cost:      ${res_opt['cost_import']:.2f}")
    print(f"  Export Revenue:        ${res_opt['rev_export']:.2f}")
    print(f"  Peak Demand Charge:    ${res_opt['cost_peak']:.2f} (Peak: {res_opt['peak_demand_kw']:.1f} kW)")
    print(f"  Battery Wear Cost:     ${res_opt['cost_degradation']:.2f}")
    print(f"  Diesel Fuel Cost:      ${res_opt['cost_diesel']:.2f}")
    print(f"  Solver Runtime:        {res_opt['runtime_sec']:.4f} s")

    savings = res_base['total_cost'] - res_opt['total_cost']
    pct_savings = (savings / res_base['total_cost']) * 100.0
    peak_reduction = (res_base['peak_demand_kw'] - res_opt['peak_demand_kw'])
    pct_peak_reduction = (peak_reduction / res_base['peak_demand_kw']) * 100.0

    print("\n================================================================================")
    print(f"OPTIMIZATION VERDICT: Proven Cost Reduction of ${savings:.2f} ({pct_savings:.1f}%)")
    print(f"Peak Demand Reduced by {peak_reduction:.1f} kW ({pct_peak_reduction:.1f}%)")
    print("================================================================================")

    # Save Metrics CSV
    metrics = [
        {
            "Metric": "Total Daily Cost ($)",
            "Baseline_Heuristic": res_base['total_cost'],
            "Optimized_LP_MILP": res_opt['total_cost'],
            "Absolute_Delta": -savings,
            "Percentage_Improvement": pct_savings
        },
        {
            "Metric": "Grid Energy Import Cost ($)",
            "Baseline_Heuristic": res_base['cost_import'],
            "Optimized_LP_MILP": res_opt['cost_import'],
            "Absolute_Delta": res_opt['cost_import'] - res_base['cost_import'],
            "Percentage_Improvement": (res_base['cost_import'] - res_opt['cost_import']) / res_base['cost_import'] * 100.0
        },
        {
            "Metric": "Peak Grid Demand (kW)",
            "Baseline_Heuristic": res_base['peak_demand_kw'],
            "Optimized_LP_MILP": res_opt['peak_demand_kw'],
            "Absolute_Delta": -peak_reduction,
            "Percentage_Improvement": pct_peak_reduction
        },
        {
            "Metric": "Peak Demand Charge ($)",
            "Baseline_Heuristic": res_base['cost_peak'],
            "Optimized_LP_MILP": res_opt['cost_peak'],
            "Absolute_Delta": res_opt['cost_peak'] - res_base['cost_peak'],
            "Percentage_Improvement": (res_base['cost_peak'] - res_opt['cost_peak']) / res_base['cost_peak'] * 100.0
        },
        {
            "Metric": "Solar Feed-In Export Revenue ($)",
            "Baseline_Heuristic": res_base['rev_export'],
            "Optimized_LP_MILP": res_opt['rev_export'],
            "Absolute_Delta": res_opt['rev_export'] - res_base['rev_export'],
            "Percentage_Improvement": (res_opt['rev_export'] - res_base['rev_export']) / (res_base['rev_export'] + 1e-6) * 100.0
        },
        {
            "Metric": "Battery Degradation Cost ($)",
            "Baseline_Heuristic": res_base['cost_degradation'],
            "Optimized_LP_MILP": res_opt['cost_degradation'],
            "Absolute_Delta": res_opt['cost_degradation'] - res_base['cost_degradation'],
            "Percentage_Improvement": (res_base['cost_degradation'] - res_opt['cost_degradation']) / res_base['cost_degradation'] * 100.0
        },
        {
            "Metric": "Solver Runtime (seconds)",
            "Baseline_Heuristic": res_base['runtime_sec'],
            "Optimized_LP_MILP": res_opt['runtime_sec'],
            "Absolute_Delta": res_opt['runtime_sec'] - res_base['runtime_sec'],
            "Percentage_Improvement": 0.0
        }
    ]

    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("results/track_e_case_study_metrics.csv", index=False)
    print("  Saved: results/track_e_case_study_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    t = np.arange(mg.T)

    # Plot 1: 24-Hour Dispatch Profiles Comparison
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # Baseline Dispatch
    axes[0].plot(t, mg.p_load, 'k--', label="Facility Base Load", linewidth=2)
    axes[0].plot(t, mg.p_pv, 'y-', label="Solar PV Generation", linewidth=2.5)
    axes[0].bar(t, res_base['p_grid_import'], color='salmon', alpha=0.7, label="Grid Import")
    axes[0].bar(t, -res_base['p_grid_export'], color='green', alpha=0.5, label="Grid Export")
    axes[0].bar(t, res_base['p_bat_dis'], bottom=res_base['p_grid_import'], color='purple', alpha=0.6, label="Battery Discharge")
    axes[0].plot(t, res_base['p_ev'], 'c-o', label="EV Fleet Load", markersize=4)
    axes[0].set_title("Track E: Baseline Heuristic Priority Dispatch (Uncoordinated EV & Solar)", fontweight='bold')
    axes[0].set_ylabel("Power (kW)")
    axes[0].grid(True, linestyle=':', alpha=0.6)
    axes[0].legend(loc="upper left", ncol=3, fontsize=8)

    # Optimized Dispatch
    axes[1].plot(t, mg.p_load, 'k--', label="Facility Base Load", linewidth=2)
    axes[1].plot(t, mg.p_pv, 'y-', label="Solar PV Generation", linewidth=2.5)
    axes[1].bar(t, res_opt['p_grid_import'], color='salmon', alpha=0.7, label="Grid Import")
    axes[1].bar(t, -res_opt['p_grid_export'], color='green', alpha=0.5, label="Grid Export")
    axes[1].bar(t, res_opt['p_bat_dis'], bottom=res_opt['p_grid_import'], color='purple', alpha=0.6, label="Battery Discharge")
    axes[1].plot(t, res_opt['p_ev'], 'c-o', label="EV Fleet Load (Smart Shifted)", markersize=4)
    axes[1].axhline(res_opt['peak_demand_kw'], color='darkred', linestyle='--', linewidth=1.8, label=f"Optimal Shaved Peak ({res_opt['peak_demand_kw']:.1f} kW)")
    axes[1].set_title(f"Track E: Optimal Dispatch (Peak Shaved by {pct_peak_reduction:.1f}%, Cost Cut by {pct_savings:.1f}%)", fontweight='bold')
    axes[1].set_xlabel("Hour of the Day (0:00 - 23:00)")
    axes[1].set_ylabel("Power (kW)")
    axes[1].set_xticks(np.arange(0, 24, 2))
    axes[1].grid(True, linestyle=':', alpha=0.6)
    axes[1].legend(loc="upper left", ncol=3, fontsize=8)

    plt.tight_layout()
    plt.savefig("plots/track_e_dispatch_profiles.png", dpi=200)
    plt.close()

    # Plot 2: Battery State of Charge & Tariff Overlay
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    t_soc = np.arange(mg.T + 1)
    ax1.plot(t_soc, res_base['soc'] * 100.0, 'r--', linewidth=2.2, label="Baseline SoC (%)")
    ax1.plot(t_soc, res_opt['soc'] * 100.0, 'b-', linewidth=2.5, label="Optimized SoC (%)")
    ax1.axhline(mg.soc_max * 100.0, color='gray', linestyle=':', label="SoC Bounds (15% - 95%)")
    ax1.axhline(mg.soc_min * 100.0, color='gray', linestyle=':')
    ax1.set_xlabel("Hour of the Day")
    ax1.set_ylabel("Battery State of Charge (%)", color='b')
    ax1.set_ylim(0, 105)

    ax2.step(np.arange(mg.T), mg.tariff_import, color='orange', alpha=0.5, where='post', linewidth=2, label="Grid Tariff ($/kWh)")
    ax2.set_ylabel("ToU Tariff ($/kWh)", color='orange')
    ax2.set_ylim(0, 0.45)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
    plt.title("Track E: Battery Energy Arbitrage & Peak Shaving vs Electricity Tariff", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_e_battery_soc.png", dpi=200)
    plt.close()

    # Plot 3: Cost Breakdown Bar Chart
    plt.figure(figsize=(7, 4.5))
    categories = ['Import Cost', 'Peak Demand', 'Bat Wear', 'Total Cost']
    base_vals = [res_base['cost_import'], res_base['cost_peak'], res_base['cost_degradation'], res_base['total_cost']]
    opt_vals = [res_opt['cost_import'], res_opt['cost_peak'], res_opt['cost_degradation'], res_opt['total_cost']]

    x = np.arange(len(categories))
    w = 0.35
    plt.bar(x - w/2, base_vals, width=w, label="Rule-Based Baseline", color='indianred', alpha=0.85)
    plt.bar(x + w/2, opt_vals, width=w, label="Optimized (LP/MILP)", color='seagreen', alpha=0.85)

    for i in range(len(categories)):
        plt.text(x[i] - w/2, base_vals[i] + 15, f"${base_vals[i]:.0f}", ha='center', fontsize=8, fontweight='bold')
        plt.text(x[i] + w/2, opt_vals[i] + 15, f"${opt_vals[i]:.0f}", ha='center', fontsize=8, fontweight='bold')

    plt.xticks(x, categories, fontweight='bold')
    plt.ylabel("Cost ($ / Day)")
    plt.title(f"Track E: Microgrid Economic Comparison (${savings:.1f} Saved / Day)", fontweight='bold')
    plt.ylim(0, max(base_vals) * 1.25)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/track_e_cost_comparison.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_e_dispatch_profiles.png, track_e_battery_soc.png, track_e_cost_comparison.png")
    print("\n[DONE] Track E Execution Complete!\n")


if __name__ == "__main__":
    main()
