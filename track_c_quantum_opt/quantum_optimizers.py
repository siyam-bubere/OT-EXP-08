"""
track_c_quantum_opt/quantum_optimizers.py
Complete Chapter 6 Quantum Optimization Suite from Scratch
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Algorithms Covered:
1. QAOA (Quantum Approximate Optimization Algorithm) for Track B Max-Cut
2. VQE (Variational Quantum Eigensolver) with Parameter-Shift Rule & SPSA
3. Quantum Natural Gradient (QNG) using Fubini-Study Metric Tensor
4. Barren Plateaus Study across Qubit Scaling (N=2 to 10)
5. Finite Measurement Shots & Hardware Noise Channels (Depolarizing / Damping)
"""

import sys, os
import numpy as np
from typing import List, Tuple, Dict, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quantum_sim import (
    QuantumCircuit,
    PauliHamiltonian,
    parameter_shift_gradient,
    spsa_gradient,
    compute_fubini_study_metric,
    quantum_natural_gradient,
    apply_depolarizing_channel,
    apply_amplitude_damping_channel
)


def build_qaoa_maxcut_circuit(n_qubits: int, edges: List[Tuple[int, int, float]], p_depth: int) -> Tuple[QuantumCircuit, PauliHamiltonian]:
    """
    Builds the QAOA circuit and Cost Hamiltonian for Max-Cut.
    Cost Hamiltonian H_C = sum_{(u,v) in E} w_uv/2 * (I - Z_u Z_v).
    To minimize energy <=> maximize cut, H_opt = -H_C = sum w_uv/2 * (Z_u Z_v - I).
    Mixer Hamiltonian H_B = sum_i X_i.
    """
    H_cost = PauliHamiltonian(n_qubits)
    # Add Z_u Z_v terms (ignoring constant identity offset for gradients)
    for u, v, w in edges:
        p_str = ['I'] * n_qubits
        p_str[u] = 'Z'
        p_str[v] = 'Z'
        # Coupling in energy: + 0.5 * w * Z_u Z_v
        H_cost.add_term(0.5 * w, "".join(p_str))

    qc = QuantumCircuit(n_qubits)
    # 1. Initial State: |+>^n via Hadamard on all qubits
    for i in range(n_qubits):
        qc.h(i)

    # 2. Alternating p layers of Cost and Mixer unitaries
    param_idx = 0
    for layer in range(p_depth):
        gamma_idx = param_idx
        beta_idx = param_idx + 1
        param_idx += 2

        # Cost Unitary: U_C(gamma) = exp(-i gamma H_C)
        # For each edge (u, v): CNOT(u, v) -> Rz(2 * gamma * w) -> CNOT(u, v)
        for u, v, w in edges:
            qc.cnot(u, v)
            # Parametric Rz
            qc.rz(v, gamma_idx)
            qc.cnot(u, v)

        # Mixer Unitary: U_B(beta) = exp(-i beta H_B) = prod Rx(2 * beta)
        for i in range(n_qubits):
            qc.rx(i, beta_idx)

    return qc, H_cost


def run_qaoa_optimization(n_qubits: int, edges: List[Tuple[int, int, float]], p_depth: int = 2,
                          lr: float = 0.05, n_steps: int = 40, shots: Optional[int] = None, seed: int = 42) -> Dict:
    """Optimizes QAOA variational parameters (gamma, beta) using Parameter-Shift Gradient."""
    qc, H_cost = build_qaoa_maxcut_circuit(n_qubits, edges, p_depth)
    rng = np.random.default_rng(seed)
    # 2 * p parameters: [gamma_1, beta_1, ..., gamma_p, beta_p]
    params = rng.uniform(0.1, np.pi / 2, size=2 * p_depth)

    energy_hist = []
    cut_hist = []

    # Total weight sum
    total_w = sum(w for _, _, w in edges)

    for step in range(n_steps):
        # Calculate expected cut value: Cut = 0.5 * (total_w - sum w_uv <Z_u Z_v>)
        exp_H = H_cost.expectation_from_circuit(qc, params, shots=shots, seed=seed + step)
        expected_cut = 0.5 * total_w - exp_H
        energy_hist.append(exp_H)
        cut_hist.append(expected_cut)

        # Parameter shift gradient on cost
        grad = parameter_shift_gradient(qc, H_cost, params)
        # Gradient descent to minimize H_cost <=> maximize Cut
        params -= lr * grad

    # Final measurement sample
    final_counts = qc.sample_measurements(params, shots=1000, seed=seed)
    best_bitstring = max(final_counts, key=final_counts.get)
    best_partition = np.array([int(b) for b in best_bitstring])

    return {
        "p_depth": p_depth,
        "final_params": params,
        "energy_hist": energy_hist,
        "cut_hist": cut_hist,
        "best_bitstring": best_bitstring,
        "best_partition": best_partition,
        "measurement_counts": final_counts,
        "final_expected_cut": cut_hist[-1]
    }


def run_vqe_simulation(method: str = "parameter_shift", lr: float = 0.1, n_steps: int = 40, seed: int = 42) -> Dict:
    """
    Variational Quantum Eigensolver (VQE) on a 2-qubit transverse Ising Hamiltonian:
    H = -1.0 (Z0 Z1) - 0.5 (X0 + X1).
    Analytical minimum eigenvalue = -np.sqrt(1 + 0.5^2) - 0.5 approx -1.6180.
    """
    n_qubits = 2
    H = PauliHamiltonian(n_qubits)
    H.add_term(-1.0, "ZZ")
    H.add_term(-0.5, "XI")
    H.add_term(-0.5, "IX")

    exact_matrix = H.get_matrix()
    exact_ground_energy = float(np.min(np.linalg.eigvalsh(exact_matrix)))

    # Ansatz: Ry(theta0) Ry(theta1) -> CNOT(0, 1) -> Ry(theta2) Ry(theta3)
    qc = QuantumCircuit(n_qubits)
    qc.ry(0, 0).ry(1, 1)
    qc.cnot(0, 1)
    qc.ry(0, 2).ry(1, 3)

    rng = np.random.default_rng(seed)
    params = rng.uniform(0.1, np.pi, size=4)
    loss_history = []

    for step in range(n_steps):
        current_energy = H.expectation_from_circuit(qc, params)
        loss_history.append(current_energy)

        if method == "parameter_shift":
            grad = parameter_shift_gradient(qc, H, params)
            params -= lr * grad
        elif method == "spsa":
            c_k = 0.1 / (step + 1)**0.101
            a_k = lr / (step + 1 + 5)**0.602
            grad = spsa_gradient(qc, H, params, c_k=c_k, seed=seed + step)
            params -= a_k * grad
        elif method == "qng":
            nat_grad = quantum_natural_gradient(qc, H, params)
            params -= lr * nat_grad

    return {
        "method": method,
        "exact_ground_energy": exact_ground_energy,
        "final_energy": loss_history[-1],
        "history": loss_history
    }


def run_barren_plateaus_study(n_qubits_list: List[int] = [2, 4, 6, 8, 10], n_samples: int = 50) -> Dict:
    """
    Demonstrates McClean et al. (2018) Barren Plateaus phenomenon:
    Variance of gradients decays exponentially as 2^{-n} with qubit count n.
    Circuit: Random alternating layers of Ry rotations and entangling CNOT rings.
    """
    variances = []

    for n in n_qubits_list:
        H = PauliHamiltonian(n)
        # Measure Z0 Z1 observable
        p_str = ['I'] * n
        p_str[0] = 'Z'
        p_str[1] = 'Z'
        H.add_term(1.0, "".join(p_str))

        # Build random ansatz of depth 4
        qc = QuantumCircuit(n)
        param_idx = 0
        for layer in range(4):
            for q in range(n):
                qc.ry(q, param_idx)
                param_idx += 1
            for q in range(n - 1):
                qc.cnot(q, q + 1)
            qc.cnot(n - 1, 0)

        grads_sample = []
        rng = np.random.default_rng(n * 10 + 42)

        for s in range(n_samples):
            # Uniform Haar-random angles
            random_params = rng.uniform(0, 2 * np.pi, size=param_idx)
            # Parameter-shift gradient specifically for the first parameter theta_0
            p_plus = np.copy(random_params)
            p_minus = np.copy(random_params)
            p_plus[0] += np.pi / 2
            p_minus[0] -= np.pi / 2
            e_plus = H.expectation_from_circuit(qc, p_plus)
            e_minus = H.expectation_from_circuit(qc, p_minus)
            grad_0 = 0.5 * (e_plus - e_minus)
            grads_sample.append(grad_0)

        var_g = float(np.var(grads_sample))
        variances.append(var_g)
        print(f"  Barren Plateau Qubits n={n:2d} | Grad Variance Var(d<H>/dtheta): {var_g:.6e}")

    return {"qubits": n_qubits_list, "variances": variances}
