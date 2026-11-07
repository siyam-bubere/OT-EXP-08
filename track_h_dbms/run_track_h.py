"""
track_h_dbms/run_track_h.py - Benchmarks & Visualizations for Track H (DBMS Optimization)
Optimization Techniques Lab - Assignment 8 (Chapter 6 DBMS Optimization)
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
from track_h_dbms.dbms_optimizer import (
    TableCatalog,
    JoinOptimizerDP,
    CandidateIndex,
    solve_index_selection_knapsack,
    create_sqlite_benchmark_db,
    benchmark_query_workload
)


def main():
    print("================================================================================")
    print("EXECUTING TRACK H: DBMS QUERY & STORAGE OPTIMIZATION")
    print("================================================================================")

    # -------------------------------------------------------------------------
    # 1. Selinger DP Join-Order Enumeration
    # -------------------------------------------------------------------------
    print("\n--- 1. Dynamic Programming Join Order Enumeration (Selinger Algorithm) ---")
    tables = {
        "Customers": TableCatalog("Customers", 2000),
        "Orders": TableCatalog("Orders", 5000),
        "LineItem": TableCatalog("LineItem", 12000),
        "Products": TableCatalog("Products", 500)
    }

    # Predicate selectivities
    selectivities = {
        ("Customers", "Orders"): 1.0 / 2000.0,       # Foreign key join
        ("LineItem", "Orders"): 1.0 / 5000.0,        # Foreign key join
        ("LineItem", "Products"): 1.0 / 500.0        # Foreign key join
    }

    join_opt = JoinOptimizerDP(tables, selectivities)
    rel_names = ["Customers", "Orders", "LineItem", "Products"]
    cost_dp, card_dp, plan_dp = join_opt.solve(rel_names)

    # Compare with arbitrary unoptimized left-to-right order (Customers x Products x LineItem x Orders)
    unopt_plan = "(((Customers JOIN Products) JOIN LineItem) JOIN Orders)"
    unopt_cost = 2000 * 500 * 0.05 + 1000000 * 12000 * 0.05  # Cross product explosion

    print(f"  Optimal DP Join Plan   : {plan_dp}")
    print(f"  Optimal DP Join Cost   : {cost_dp:,.1f} cost units (Result Cardinality: {card_dp})")
    print(f"  Unoptimized Plan Cost  : {unopt_cost:,.1f} cost units")
    print(f"  Join Cost Reduction    : {(1.0 - cost_dp / unopt_cost)*100.0:.2f}%")

    # -------------------------------------------------------------------------
    # 2. Automated Index Selection (0/1 Knapsack under Storage Budget)
    # -------------------------------------------------------------------------
    print("\n--- 2. Automated Physical Index Selection (0/1 Knapsack Optimization) ---")
    candidates = [
        CandidateIndex("idx_orders_cust", "orders", "cust_id", 0.8, 18.0),
        CandidateIndex("idx_orders_status", "orders", "status", 0.5, 8.0),
        CandidateIndex("idx_lineitem_order", "lineitem", "order_id", 1.4, 32.0),
        CandidateIndex("idx_lineitem_prod", "lineitem", "prod_id", 1.2, 24.0),
        CandidateIndex("idx_cust_nation", "customers", "nation_id", 0.3, 7.0),
        CandidateIndex("idx_prod_cat", "products", "category", 0.2, 5.0)
    ]

    budget_mb = 2.5
    chosen, chosen_storage, total_gain = solve_index_selection_knapsack(candidates, budget_mb)

    print(f"  Storage Budget: {budget_mb:.1f} MB")
    print(f"  Selected {len(chosen)} Indexes (Total Storage: {chosen_storage:.1f} MB, Expected Gain: {total_gain:.1f} ms):")
    for idx in chosen:
        print(f"    - {idx.index_name} on {idx.table}({idx.column}) [Size: {idx.storage_mb} MB, Gain: {idx.latency_gain_ms} ms]")

    # -------------------------------------------------------------------------
    # 3. Empirical SQLite Execution Verification & EXPLAIN Analysis
    # -------------------------------------------------------------------------
    print("\n--- 3. Empirical SQLite Workload Latency & Plan Verification ---")
    db_file = "benchmark.db"
    create_sqlite_benchmark_db(db_file, seed=42)

    # Baseline: no indexes
    lat_baseline, plans_base = benchmark_query_workload(db_file)
    print(f"  Baseline Workload Latency (Full Table Scans) : {lat_baseline:.2f} ms")
    print(f"  Baseline Sample Plan                         : {plans_base[0]}")

    # Optimized: apply knapsack-selected indexes
    index_ddls = [f"CREATE INDEX IF NOT EXISTS {idx.index_name} ON {idx.table}({idx.column})" for idx in chosen]
    lat_opt, plans_opt = benchmark_query_workload(db_file, index_ddls)
    print(f"  Optimized Workload Latency (Index Searches)  : {lat_opt:.2f} ms")
    print(f"  Optimized Sample Plan                        : {plans_opt[0]}")

    speedup = lat_baseline / max(0.01, lat_opt)
    impr_pct = (lat_baseline - lat_opt) / lat_baseline * 100.0
    print(f"  -> Physical Speedup: {speedup:.2f}x ({impr_pct:.1f}% Latency Reduction)")

    # Clean up db file
    try:
        os.remove(db_file)
    except Exception:
        pass

    # Save Metrics CSV
    records = [
        {"Optimization": "Selinger DP Join Tree", "Metric": "Cost Units", "Baseline": unopt_cost, "Optimized": cost_dp, "Improvement_%": (1.0 - cost_dp/unopt_cost)*100.0},
        {"Optimization": "Index Selection Knapsack", "Metric": "Storage (MB)", "Baseline": sum(c.storage_mb for c in candidates), "Optimized": chosen_storage, "Improvement_%": (1.0 - chosen_storage/budget_mb)*100.0},
        {"Optimization": "Physical Query Latency", "Metric": "Execution Time (ms)", "Baseline": lat_baseline, "Optimized": lat_opt, "Improvement_%": impr_pct}
    ]
    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv("results/track_h_dbms_metrics.csv", index=False)
    print("\n  Saved: results/track_h_dbms_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Query Latency Before vs After Index Selection
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(["No Indexes (Table Scans)", "Optimized (Knapsack Indexes)"], [lat_baseline, lat_opt], color=['salmon', 'forestgreen'], width=0.45)
    plt.text(0, lat_baseline + 0.5, f"{lat_baseline:.2f} ms", ha='center', fontweight='bold', fontsize=9.5)
    plt.text(1, lat_opt + 0.5, f"{lat_opt:.2f} ms\n({speedup:.2f}x Faster)", ha='center', fontweight='bold', fontsize=9.5)
    plt.ylabel("Multi-Query Execution Time (ms)")
    plt.title(f"Track H: DBMS Workload Latency ({impr_pct:.1f}% Faster)", fontweight='bold')
    plt.ylim(0, lat_baseline * 1.3)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_h_query_latency.png", dpi=200)
    plt.close()

    # Plot 2: Index Knapsack Benefit vs Storage Frontier
    plt.figure(figsize=(7, 4.5))
    all_stor = [c.storage_mb for c in candidates]
    all_gain = [c.latency_gain_ms for c in candidates]
    all_names = [c.index_name.replace("idx_", "") for c in candidates]

    chosen_set = {c.index_name for c in chosen}
    colors = ['green' if c.index_name in chosen_set else 'gray' for c in candidates]

    plt.scatter(all_stor, all_gain, c=colors, s=160, edgecolors='black', linewidth=1.5, zorder=3)
    for i, name in enumerate(all_names):
        status = " (Chosen)" if candidates[i].index_name in chosen_set else " (Omitted)"
        plt.annotate(f"{name}{status}", (all_stor[i] + 0.03, all_gain[i] + 0.5), fontsize=8.5, fontweight='bold')

    plt.axvline(budget_mb, color='red', linestyle='--', linewidth=2, label=f"Storage Budget ({budget_mb} MB)")
    plt.xlabel("Index Storage Overhead (MB)")
    plt.ylabel("Workload Latency Reduction (ms)")
    plt.title("Track H: Automated Index Selection (0/1 Knapsack Frontier)", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_h_index_selection.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_h_query_latency.png, track_h_index_selection.png")
    print("\n[DONE] Track H Execution Complete!\n")


if __name__ == "__main__":
    main()
