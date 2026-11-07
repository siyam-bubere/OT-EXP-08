"""
track_i_quantum_algorithms/run_track_i.py - Benchmarks & Visualizations for Track I
Optimization Techniques Lab - Assignment 8 (Chapter 6 Quantum Algorithms & QML)
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
from track_i_quantum_algorithms.quantum_algorithms_qml import (
    run_deutsch_jozsa,
    run_bernstein_vazirani,
    run_grover_search,
    run_quantum_phase_estimation,
    run_shor_order_finding,
    run_quantum_annealing,
    run_qml_vs_classical_comparison
)


def main():
    print("================================================================================")
    print("EXECUTING TRACK I: QUANTUM ALGORITHMS & QML ADVANTAGE BENCHMARK")
    print("================================================================================")

    records = []

    # 1. Deutsch-Jozsa
    dj_res = run_deutsch_jozsa(n_qubits=3, is_balanced=True)
    print(f"\n--- 1. Deutsch-Jozsa Algorithm ---")
    print(f"  Target Function: Balanced | Detected: {dj_res['detected_type']} | Success? {dj_res['success']}")
    print(f"  Complexity: 1 Quantum Query vs {dj_res['classical_worst_case_queries']} Classical Queries")
    records.append({
        "Algorithm": "Deutsch-Jozsa",
        "Problem": "Balanced vs Constant Oracle",
        "Quantum_Queries": dj_res["quantum_queries"],
        "Classical_Queries": dj_res["classical_worst_case_queries"],
        "Speedup_Class": "Exponential (O(1) vs O(2^(n-1)))",
        "Empirical_Success": dj_res["success"]
    })

    # 2. Bernstein-Vazirani
    bv_res = run_bernstein_vazirani(hidden_string="101")
    print(f"\n--- 2. Bernstein-Vazirani Algorithm ---")
    print(f"  Hidden String: {bv_res['hidden_string']} | Recovered: {bv_res['recovered_string']} (Prob: {bv_res['success_probability']:.1%})")
    records.append({
        "Algorithm": "Bernstein-Vazirani",
        "Problem": "Hidden Bitstring Recovery",
        "Quantum_Queries": bv_res["quantum_queries"],
        "Classical_Queries": bv_res["classical_queries"],
        "Speedup_Class": "Polynomial (1 query vs n)",
        "Empirical_Success": bv_res["success"]
    })

    # 3. Grover Search
    grover_res = run_grover_search(n_qubits=3, target_idx=5)
    print(f"\n--- 3. Grover Quantum Search Algorithm ---")
    print(f"  Database N={grover_res['database_size_N']} | Target: {grover_res['target_index']} | Found: {grover_res['found_index']}")
    print(f"  Iterations: {grover_res['optimal_iterations']} | Target Probability: {grover_res['final_success_prob']:.1%}")
    records.append({
        "Algorithm": "Grover Search",
        "Problem": "Unsorted Database Search",
        "Quantum_Queries": grover_res["optimal_iterations"],
        "Classical_Queries": grover_res["classical_query_complexity"],
        "Speedup_Class": "Quadratic (O(sqrt(N)) vs O(N))",
        "Empirical_Success": grover_res["success"]
    })

    # 4. QPE
    qpe_res = run_quantum_phase_estimation(true_phase=0.625, n_counting_qubits=3)
    print(f"\n--- 4. Quantum Phase Estimation (QPE) ---")
    print(f"  True Phase: {qpe_res['true_phase']} | Estimated: {qpe_res['estimated_phase']} | Success? {qpe_res['success']}")
    records.append({
        "Algorithm": "QPE",
        "Problem": "Unitary Eigenphase Estimation",
        "Quantum_Queries": qpe_res["counting_qubits"],
        "Classical_Queries": 2**qpe_res["counting_qubits"],
        "Speedup_Class": "Exponential Precision",
        "Empirical_Success": qpe_res["success"]
    })

    # 5. Shor's Algorithm
    shor_res = run_shor_order_finding(N=15, a=7)
    print(f"\n--- 5. Shor's Factoring Algorithm ---")
    print(f"  Factoring N={shor_res['composite_N']} (Base a={shor_res['coprime_base_a']}) -> Period r={shor_res['quantum_period_r']}")
    print(f"  Factors: {shor_res['factor_1']} and {shor_res['factor_2']} (Product check: {shor_res['product_check']})")
    records.append({
        "Algorithm": "Shor's Factoring",
        "Problem": "Integer Factorization",
        "Quantum_Queries": 4,
        "Classical_Queries": 15,
        "Speedup_Class": "Superpolynomial / Exponential",
        "Empirical_Success": shor_res["success"]
    })

    # 6. Quantum Annealing
    anneal_res = run_quantum_annealing(n_spins=6, n_steps=30)
    print(f"\n--- 6. Quantum Annealing (Transverse-Field Ising) ---")
    print(f"  Spins N={anneal_res['n_spins']} | Ground State Fidelity: {anneal_res['final_ground_state_fidelity']:.1%}")
    records.append({
        "Algorithm": "Quantum Annealing",
        "Problem": "Transverse-Field Max-Cut",
        "Quantum_Queries": anneal_res["annealing_steps"],
        "Classical_Queries": 2**6,
        "Speedup_Class": "Heuristic Tunneling Advantage",
        "Empirical_Success": anneal_res["success"]
    })

    # 7. QML vs Classical MLP
    print(f"\n--- 7. Quantum Machine Learning (VQC) vs Classical MLP ---")
    qml_res = run_qml_vs_classical_comparison()
    print(f"  VQC (Quantum):     Params = {qml_res['vqc_params']:2d} | Accuracy = {qml_res['vqc_accuracy']:.1%}")
    print(f"  MLP (Classical):   Params = {qml_res['mlp_params']:2d} | Accuracy = {qml_res['mlp_accuracy']:.1%}")
    records.append({
        "Algorithm": "VQC (Quantum ML)",
        "Problem": "Non-Linear Classification",
        "Quantum_Queries": len(qml_res["vqc_losses"]),
        "Classical_Queries": len(qml_res["mlp_losses"]),
        "Speedup_Class": "Sample / Parameter Efficiency",
        "Empirical_Success": qml_res["vqc_accuracy"] >= 0.70
    })

    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(records)
    df_metrics.to_csv("results/track_i_quantum_metrics.csv", index=False)
    print("\n  Saved: results/track_i_quantum_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Grover Amplitude Amplification
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(range(1, len(grover_res["prob_trajectory"]) + 1), [p * 100.0 for p in grover_res["prob_trajectory"]], 'mo-', linewidth=2.5, markersize=8)
    plt.axhline(100.0, color='g', linestyle=':', label="Target State |101> Ideal (100%)")
    plt.xlabel("Grover Diffusion Iteration")
    plt.ylabel("Marked State Probability (%)")
    plt.title(f"Track I: Grover Amplitude Amplification (N=8, Target=5)", fontweight='bold')
    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_i_grover_amplification.png", dpi=200)
    plt.close()

    # Plot 2: QML vs Classical MLP Loss Trajectory
    plt.figure(figsize=(7, 4.5))
    plt.plot(qml_res["vqc_losses"], 'b-o', linewidth=2.2, label=f"Variational Quantum Classifier ({qml_res['vqc_params']} params, Acc: {qml_res['vqc_accuracy']:.1%})")
    plt.plot(qml_res["mlp_losses"], 'r--', linewidth=2.2, label=f"Classical 2-Layer MLP ({qml_res['mlp_params']} params, Acc: {qml_res['mlp_accuracy']:.1%})")
    plt.xlabel("Training Epoch")
    plt.ylabel("Empirical Mean Squared Error")
    plt.title("Track I: Quantum VQC vs Classical MLP on Non-Linear Benchmark", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_i_qml_vs_classical.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_i_grover_amplification.png, track_i_qml_vs_classical.png")
    print("\n[DONE] Track I Execution Complete!\n")


if __name__ == "__main__":
    main()
