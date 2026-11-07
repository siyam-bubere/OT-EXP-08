"""
track_b_discrete/run_track_b.py - Experimental Benchmarking Suite for Track B
Optimization Techniques Lab - Assignment 8 (Chapter 6 Coverage)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from track_b_discrete.discrete_solvers import (
    solve_knapsack_greedy,
    solve_knapsack_dp,
    BranchAndBoundKnapsack,
    solve_knapsack_ilp,
    MaxCutProblem,
    formulate_knapsack_qubo
)


def run_knapsack_scaling_benchmark():
    print("\n--- Running Track B.1: Knapsack Scaling Benchmark (Greedy vs DP vs B&B vs ILP) ---")
    sizes = [6, 10, 14, 18, 22, 26]
    records = []

    for n in sizes:
        rng = np.random.default_rng(n + 100)
        weights = rng.integers(5, 50, size=n)
        values = weights * rng.uniform(1.1, 3.5, size=n) + rng.integers(1, 20, size=n)
        capacity = int(np.sum(weights) * 0.45)

        # 1. Greedy
        res_gr = solve_knapsack_greedy(values, weights, capacity)
        # 2. DP
        res_dp = solve_knapsack_dp(values, weights, capacity)
        # 3. Branch-and-Bound
        bb_solver = BranchAndBoundKnapsack(values, weights, capacity)
        res_bb = bb_solver.solve()
        # 4. ILP
        res_ilp = solve_knapsack_ilp(values, weights, capacity)

        opt_val = res_dp["value"]
        records.append({
            "N_Items": n,
            "Greedy_Val": res_gr["value"], "Greedy_Time": res_gr["runtime_s"],
            "Greedy_Ratio": res_gr["value"] / opt_val,
            "DP_Val": res_dp["value"], "DP_Time": res_dp["runtime_s"],
            "BB_Val": res_bb["value"], "BB_Time": res_bb["runtime_s"],
            "BB_Nodes": res_bb["nodes_explored"],
            "ILP_Val": res_ilp["value"], "ILP_Time": res_ilp["runtime_s"],
            "Optimal_Val": opt_val
        })

        print(f"  N={n:2d} | OptVal: {opt_val:.1f} | Greedy: {res_gr['value']:.1f} ({res_gr['value']/opt_val:.1%}) | "
              f"Time DP: {res_dp['runtime_s']*1e3:.2f}ms, BB: {res_bb['runtime_s']*1e3:.2f}ms, ILP: {res_ilp['runtime_s']*1e3:.2f}ms")

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/track_b_discrete_metrics.csv", index=False)

    # Plot Runtime Comparison (Log scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))

    ax1.plot(df["N_Items"], df["Greedy_Time"] * 1e3, 'o-', label="Greedy Heuristic", linewidth=2)
    ax1.plot(df["N_Items"], df["DP_Time"] * 1e3, 's-', label="Dynamic Programming", linewidth=2)
    ax1.plot(df["N_Items"], df["BB_Time"] * 1e3, '^-', label="Branch-and-Bound", linewidth=2)
    ax1.plot(df["N_Items"], df["ILP_Time"] * 1e3, 'd-', label="ILP (Branch-and-Cut)", linewidth=2)

    ax1.set_yscale('log')
    ax1.set_title("Runtime Scaling vs Problem Size (N)", fontweight='bold')
    ax1.set_xlabel("Number of Items (N)")
    ax1.set_ylabel("Execution Time (ms, log scale)")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle=':')

    # Optimality Gap (Greedy approximation ratio)
    ax2.plot(df["N_Items"], df["Greedy_Ratio"] * 100.0, 'ro-', linewidth=2.5, label="Greedy / Optimal (%)")
    ax2.axhline(100.0, color='g', linestyle='--', label="Exact Optimal (100%)")
    ax2.set_title("Greedy Solution Quality Ratio (%)", fontweight='bold')
    ax2.set_xlabel("Number of Items (N)")
    ax2.set_ylabel("Optimality Ratio (%)")
    ax2.set_ylim(80, 105)
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_b_scaling_runtime.png", dpi=200)
    plt.close()

    return df


def run_qubo_maxcut_benchmark():
    print("\n--- Running Track B.2: QUBO & Ising Model Max-Cut Benchmark ---")
    # Define an 8-node benchmark graph with 12 weighted edges
    # Serves as the direct target problem for Track C QAOA!
    edges = [
        (0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0), (1, 3, 2.0),
        (2, 4, 2.0), (3, 4, 1.0), (3, 5, 1.5), (4, 6, 1.5),
        (5, 6, 1.0), (5, 7, 2.0), (6, 7, 2.0), (0, 7, 1.0)
    ]
    n_nodes = 8
    maxcut = MaxCutProblem(n_nodes, edges)

    # 1. Exact classical brute-force ground state
    res_bf = maxcut.solve_exact_brute_force()
    max_cut_val = res_bf["max_cut"]
    opt_partition = res_bf["partition"]

    print(f"  Max-Cut Graph: 8 Nodes, 12 Edges")
    print(f"  Exact Maximum Cut Value: {max_cut_val:.2f}")
    print(f"  Optimal Node Partition:  {opt_partition}")
    print(f"  Classical Exact Solver Time: {res_bf['runtime_s']*1e3:.2f} ms")

    # 2. Extract Ising Hamiltonian
    C_offset, J_couplings = maxcut.get_ising_hamiltonian()
    print(f"  Ising Hamiltonian Constant Offset: {C_offset:.2f}")
    print(f"  Couplings Count: {len(J_couplings)} non-zero J_uv terms")

    # Visualize graph & optimal cut
    fig, ax = plt.subplots(figsize=(7, 5))
    # Circular layout
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    pos = {i: np.array([np.cos(angles[i]), np.sin(angles[i])]) for i in range(n_nodes)}

    # Draw edges
    for u, v, w in edges:
        p1, p2 = pos[u], pos[v]
        is_cut = opt_partition[u] != opt_partition[v]
        color = 'crimson' if is_cut else 'gray'
        style = '-' if is_cut else ':'
        lw = 2.5 if is_cut else 1.0
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linestyle=style, linewidth=lw, alpha=0.85)

    # Draw nodes colored by partition
    for i in range(n_nodes):
        c = '#2b5c8f' if opt_partition[i] == 1 else '#d95f02'
        ax.scatter(pos[i][0], pos[i][1], s=400, color=c, zorder=5, edgecolors='black', linewidth=1.5)
        ax.text(pos[i][0], pos[i][1], str(i), color='white', ha='center', va='center', fontweight='bold', zorder=6)

    ax.set_title(f"Track B: Max-Cut Graph (Max Cut = {max_cut_val:.1f}, Cut Edges in Red)", fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("plots/track_b_maxcut_graph.png", dpi=200)
    plt.close()

    # Save Max-Cut specifications for Track C
    np.savez(
        "results/track_b_maxcut_ground_truth.npz",
        n_nodes=n_nodes,
        edges=np.array(edges),
        max_cut=max_cut_val,
        opt_partition=opt_partition,
        C_offset=C_offset
    )
    print("  Saved Max-Cut ground truth to results/track_b_maxcut_ground_truth.npz for Track C QAOA!")

    return res_bf


def run_all_track_b():
    print("================================================================================")
    print("EXECUTING TRACK B: DISCRETE OPTIMIZATION BENCHMARK")
    print("================================================================================")
    t0 = time.time()
    run_knapsack_scaling_benchmark()
    run_qubo_maxcut_benchmark()
    print(f"\n[DONE] Track B Completed in {time.time() - t0:.2f} seconds!")


if __name__ == "__main__":
    run_all_track_b()
