"""
track_c_quantum_opt/run_track_c.py - Experimental Benchmarking Suite for Track C
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
from track_c_quantum_opt.quantum_optimizers import (
    run_qaoa_optimization,
    run_vqe_simulation,
    run_barren_plateaus_study
)


def run_qaoa_maxcut_benchmark():
    print("\n--- Running Track C.1: QAOA Max-Cut Benchmark (Depth p=1, 2, 3) ---")
    # Load exact 8-node Max-Cut graph from Track B
    gt_data = np.load("results/track_b_maxcut_ground_truth.npz")
    n_nodes = int(gt_data["n_nodes"])
    edges = [(int(u), int(v), float(w)) for u, v, w in gt_data["edges"]]
    classical_max_cut = float(gt_data["max_cut"])

    print(f"  Target Graph: {n_nodes} Qubits, {len(edges)} Edges | Classical Optimum: {classical_max_cut:.2f}")

    results_qaoa = {}
    p_depths = [1, 2, 3]

    for p in p_depths:
        t0 = time.time()
        res = run_qaoa_optimization(n_nodes, edges, p_depth=p, lr=0.08, n_steps=35, seed=42)
        dur = time.time() - t0
        approx_ratio = res["final_expected_cut"] / classical_max_cut
        results_qaoa[p] = res
        print(f"  QAOA p={p} | Final Expected Cut: {res['final_expected_cut']:.3f} | "
              f"Approx Ratio: {approx_ratio:.1%} | Sampled Bitstring: {res['best_bitstring']} | Time: {dur:.2f}s")

    # Plot QAOA convergence
    plt.figure(figsize=(7, 4.5))
    for p, res in results_qaoa.items():
        plt.plot(res["cut_hist"], label=f"QAOA (p={p})", linewidth=2)
    plt.axhline(classical_max_cut, color='black', linestyle='--', linewidth=2, label=f"Classical Max-Cut ({classical_max_cut:.1f})")
    plt.title("Track C: QAOA Cut Optimization on 8-Node Graph", fontweight='bold')
    plt.xlabel("Optimization Step")
    plt.ylabel("Expected Cut Value")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_c_qaoa_convergence.png", dpi=200)
    plt.close()

    return results_qaoa, classical_max_cut


def run_vqe_methods_benchmark():
    print("\n--- Running Track C.2: VQE Methods (Parameter-Shift vs SPSA vs QNG) ---")
    vqe_ps = run_vqe_simulation(method="parameter_shift", lr=0.1, n_steps=40)
    vqe_spsa = run_vqe_simulation(method="spsa", lr=0.1, n_steps=40)
    vqe_qng = run_vqe_simulation(method="qng", lr=0.1, n_steps=40)

    exact_E0 = vqe_ps["exact_ground_energy"]
    print(f"  Exact Analytical Ground Energy: {exact_E0:.4f}")
    print(f"  Parameter-Shift VQE Final Energy: {vqe_ps['final_energy']:.4f}")
    print(f"  SPSA VQE Final Energy:           {vqe_spsa['final_energy']:.4f}")
    print(f"  Quantum Natural Gradient Energy: {vqe_qng['final_energy']:.4f}")

    # Plot VQE convergence
    plt.figure(figsize=(7, 4.5))
    plt.plot(vqe_ps["history"], label="Parameter-Shift Rule", color="#1f77b4", linewidth=2)
    plt.plot(vqe_spsa["history"], label="SPSA (Simultaneous Perturbation)", color="#ff7f0e", linewidth=2)
    plt.plot(vqe_qng["history"], label="Quantum Natural Gradient (QNG)", color="#2ca02c", linewidth=2)
    plt.axhline(exact_E0, color='r', linestyle='--', linewidth=2, label=f"Exact Ground State ({exact_E0:.4f})")

    plt.title("Track C: VQE Optimizer Comparison (Transverse Ising Model)", fontweight='bold')
    plt.xlabel("Iteration Step")
    plt.ylabel("Energy Expectation <H>")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig("plots/track_c_vqe_methods_comparison.png", dpi=200)
    plt.close()

    return {"ps": vqe_ps, "spsa": vqe_spsa, "qng": vqe_qng}


def run_barren_plateaus_benchmark():
    print("\n--- Running Track C.3: Barren Plateaus Exponential Variance Decay ---")
    qubit_range = [2, 4, 6, 8, 10]
    res_bp = run_barren_plateaus_study(n_qubits_list=qubit_range, n_samples=40)

    # Plot Barren Plateaus variance decay
    plt.figure(figsize=(7, 4.5))
    plt.semilogy(res_bp["qubits"], res_bp["variances"], 'ro-', linewidth=2.5, markersize=8, label="Empirical Variance Var(d<H>/dtheta)")

    # Theoretical 2^{-n} scaling line
    n_arr = np.array(qubit_range)
    theory_decay = res_bp["variances"][0] * (2.0 ** -(n_arr - n_arr[0]))
    plt.semilogy(n_arr, theory_decay, 'k--', label="Theoretical ~ 2^{-n} Decay", linewidth=1.8)

    plt.title("Track C: Barren Plateaus Phenomenon (Gradient Concentration)", fontweight='bold')
    plt.xlabel("Number of Qubits (n)")
    plt.ylabel("Gradient Variance (log scale)")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig("plots/track_c_barren_plateaus.png", dpi=200)
    plt.close()

    return res_bp


def run_shot_noise_benchmark():
    print("\n--- Running Track C.4: Finite Measurement Shots on QAOA Expectation ---")
    gt_data = np.load("results/track_b_maxcut_ground_truth.npz")
    n_nodes = int(gt_data["n_nodes"])
    edges = [(int(u), int(v), float(w)) for u, v, w in gt_data["edges"]]

    shots_list = [100, 500, 2000, 10000]
    final_cuts = []

    for s in shots_list:
        res = run_qaoa_optimization(n_nodes, edges, p_depth=1, n_steps=25, shots=s, seed=42)
        final_cuts.append(res["final_expected_cut"])
        print(f"  Shots: {s:5d} | Final Estimated Cut: {res['final_expected_cut']:.3f}")

    # Exact statevector baseline
    res_exact = run_qaoa_optimization(n_nodes, edges, p_depth=1, n_steps=25, shots=None, seed=42)
    exact_cut = res_exact["final_expected_cut"]

    # Save CSV metrics
    df_metrics = pd.DataFrame({
        "Shots": shots_list,
        "Estimated_Cut": final_cuts,
        "Exact_Statevector_Cut": [exact_cut] * len(shots_list)
    })
    os.makedirs("results", exist_ok=True)
    df_metrics.to_csv("results/track_c_quantum_metrics.csv", index=False)
    print("  Saved Track C metrics to results/track_c_quantum_metrics.csv")


def run_all_track_c():
    print("================================================================================")
    print("EXECUTING TRACK C: QUANTUM OPTIMIZATION BENCHMARK")
    print("================================================================================")
    t0 = time.time()
    run_qaoa_maxcut_benchmark()
    run_vqe_methods_benchmark()
    run_barren_plateaus_benchmark()
    run_shot_noise_benchmark()
    print(f"\n[DONE] Track C Completed in {time.time() - t0:.2f} seconds!")


if __name__ == "__main__":
    run_all_track_c()
