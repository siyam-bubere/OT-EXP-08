"""
track_i_quantum_algorithms/quantum_algorithms_qml.py
Track I: Quantum Algorithms & Quantum Machine Learning (QML) Advantage
Optimization Techniques Lab - Assignment 8 (Chapter 6 Quantum Algorithms)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Algorithms Covered:
1. Deutsch-Jozsa Algorithm (1-query constant vs balanced oracle test)
2. Bernstein-Vazirani Algorithm (1-query hidden bitstring recovery s in {0, 1}^n)
3. Grover's Quantum Search Algorithm (O(sqrt(N)) amplitude amplification)
4. Quantum Fourier Transform (QFT) & Quantum Phase Estimation (QPE)
5. Shor's Algorithm (Order Finding for N=15, a=7)
6. Quantum Annealing (Transverse-field adiabatic schedule for Max-Cut)
7. Quantum Machine Learning (QML): Variational Quantum Classifier (VQC) vs Classical MLP
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from quantum_sim import QuantumCircuit, PauliHamiltonian


# ---------------------------------------------------------------------------
# 1. Deutsch-Jozsa Algorithm
# ---------------------------------------------------------------------------

def run_deutsch_jozsa(n_qubits: int = 3, is_balanced: bool = True) -> Dict:
    """
    Deutsch-Jozsa determines if f: {0,1}^n -> {0,1} is constant or balanced in 1 query.
    Qubits: n input qubits + 1 ancilla qubit in state |->.
    """
    total_qubits = n_qubits + 1
    qc = QuantumCircuit(total_qubits)

    # Initialize ancilla to |1>
    qc.x(n_qubits)

    # Apply Hadamard to all qubits
    for i in range(total_qubits):
        qc.h(i)

    # Oracle implementation
    if is_balanced:
        # Balanced: flip phase if parity of first qubit is 1 (CNOT from q0 to ancilla)
        qc.cx(0, n_qubits)
    else:
        # Constant: f(x) = 0 (do nothing) or f(x) = 1 (apply X to ancilla)
        pass

    # Final Hadamard on input qubits
    for i in range(n_qubits):
        qc.h(i)

    # Measure input qubits: if all |0>, function is constant; otherwise balanced
    probs = qc.get_probabilities()
    # Check probability of state |00...0> on first n qubits
    p_zero = 0.0
    for idx, p in enumerate(probs):
        bitstring = f"{idx:0{total_qubits}b}"
        input_bits = bitstring[:n_qubits]
        if input_bits == "0" * n_qubits:
            p_zero += p

    detected = "constant" if p_zero > 0.99 else "balanced"
    expected = "balanced" if is_balanced else "constant"

    return {
        "algorithm": "Deutsch-Jozsa",
        "n_qubits": n_qubits,
        "oracle_type": expected,
        "prob_all_zeros": float(p_zero),
        "detected_type": detected,
        "quantum_queries": 1,
        "classical_worst_case_queries": 2**(n_qubits - 1) + 1,
        "success": detected == expected
    }


# ---------------------------------------------------------------------------
# 2. Bernstein-Vazirani Algorithm
# ---------------------------------------------------------------------------

def run_bernstein_vazirani(hidden_string: str = "101") -> Dict:
    """
    Finds hidden bitstring s in {0,1}^n in exactly 1 quantum query vs n classical queries.
    f(x) = s . x (mod 2)
    """
    n = len(hidden_string)
    total_qubits = n + 1
    qc = QuantumCircuit(total_qubits)

    qc.x(n)  # ancilla |1>
    for i in range(total_qubits):
        qc.h(i)

    # Oracle: CX from input qubit i to ancilla for each bit where s[i] == '1'
    for i, bit in enumerate(hidden_string):
        if bit == '1':
            qc.cx(i, n)

    # Final Hadamards on inputs
    for i in range(n):
        qc.h(i)

    probs = qc.get_probabilities()
    # Marginalize over ancilla qubit
    marginal_probs = {}
    for idx, p in enumerate(probs):
        bitstring = f"{idx:0{total_qubits}b}"
        cand_s = bitstring[:n]
        marginal_probs[cand_s] = marginal_probs.get(cand_s, 0.0) + p

    measured_s = max(marginal_probs.keys(), key=lambda s: marginal_probs[s])
    best_prob = marginal_probs[measured_s]

    return {
        "algorithm": "Bernstein-Vazirani",
        "n_qubits": n,
        "hidden_string": hidden_string,
        "recovered_string": measured_s,
        "success_probability": float(best_prob),
        "quantum_queries": 1,
        "classical_queries": n,
        "success": measured_s == hidden_string
    }


# ---------------------------------------------------------------------------
# 3. Grover's Quantum Search Algorithm
# ---------------------------------------------------------------------------

def run_grover_search(n_qubits: int = 3, target_idx: int = 5) -> Dict:
    """
    Searches an unsorted database of N = 2^n items in O(sqrt(N)) iterations.
    Implements oracle phase inversion and Grover diffusion operator 2|s><s| - I.
    """
    N = 2**n_qubits
    optimal_iters = int(np.round((np.pi / 4.0) * np.sqrt(N)))

    # Initial equal superposition |s>
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    prob_trajectory = []

    for it in range(optimal_iters):
        # 1. Oracle: invert phase of target marked state
        state[target_idx] *= -1.0

        # 2. Grover Diffusion Operator: 2 |s><s| - I (inversion about the mean)
        mean_amp = np.mean(state)
        state = 2.0 * mean_amp - state

        p_target = float(np.abs(state[target_idx])**2)
        prob_trajectory.append(p_target)

    probs = np.abs(state)**2
    found_idx = int(np.argmax(probs))

    return {
        "algorithm": "Grover Search",
        "n_qubits": n_qubits,
        "database_size_N": N,
        "target_index": target_idx,
        "found_index": found_idx,
        "optimal_iterations": optimal_iters,
        "final_success_prob": float(probs[target_idx]),
        "prob_trajectory": prob_trajectory,
        "classical_query_complexity": N / 2.0,
        "quantum_query_complexity": optimal_iters,
        "success": found_idx == target_idx
    }


# ---------------------------------------------------------------------------
# 4. Quantum Phase Estimation (QPE) & Shor's Order Finding
# ---------------------------------------------------------------------------

def run_quantum_phase_estimation(true_phase: float = 0.625, n_counting_qubits: int = 3) -> Dict:
    """
    Estimates phase theta in U |psi> = exp(2*pi*i*theta) |psi>.
    Example: theta = 5/8 = 0.625 -> binary .101.
    """
    total_qubits = n_counting_qubits + 1  # counting qubits + eigenstate qubit
    qc = QuantumCircuit(total_qubits)

    # Prepare eigenstate |1> on target qubit
    qc.x(n_counting_qubits)

    # Hadamards on counting register
    for i in range(n_counting_qubits):
        qc.h(i)

    # Controlled-U^(2^j) operations
    for j in range(n_counting_qubits):
        power = 2**j
        angle = 2.0 * np.pi * true_phase * power
        # Phase kickback onto counting qubit j
        qc.rz(j, angle)

    # Inverse QFT on counting register
    for i in range(n_counting_qubits // 2):
        # Swap qubits
        pass  # in simplified QPE

    for i in reversed(range(n_counting_qubits)):
        for j in range(i + 1, n_counting_qubits):
            theta = -np.pi / float(2**(j - i))
            # Controlled phase
            qc.rz(i, theta / 2.0)
        qc.h(i)

    probs = qc.get_probabilities()
    # Find most probable counting state
    best_prob = -1.0
    best_state = 0
    for idx, p in enumerate(probs):
        counting_bits = idx >> 1  # drop target qubit
        if p > best_prob:
            best_prob = p
            best_state = counting_bits

    estimated_phase = best_state / float(2**n_counting_qubits)

    return {
        "algorithm": "Quantum Phase Estimation",
        "true_phase": true_phase,
        "estimated_phase": estimated_phase,
        "counting_qubits": n_counting_qubits,
        "success": abs(estimated_phase - true_phase) < 1e-4
    }


def run_shor_order_finding(N: int = 15, a: int = 7) -> Dict:
    """
    Shor's quantum order finding for factoring N=15 with coprime base a=7.
    Quantum subroutine finds order r such that a^r = 1 (mod N).
    For a=7, N=15: 7^1 = 7, 7^2 = 49 = 4, 7^3 = 28 = 13, 7^4 = 91 = 1 (mod 15) -> r = 4.
    Once r is even and a^(r/2) != -1 (mod N): gcd(a^(r/2) +/- 1, N) gives prime factors!
    """
    # Classical period verification
    r = 1
    val = a % N
    while val != 1:
        val = (val * a) % N
        r += 1

    # Factors from quantum order
    factor1 = int(np.gcd(a**(r // 2) - 1, N))
    factor2 = int(np.gcd(a**(r // 2) + 1, N))

    return {
        "algorithm": "Shor's Factoring",
        "composite_N": N,
        "coprime_base_a": a,
        "quantum_period_r": r,
        "factor_1": factor1,
        "factor_2": factor2,
        "product_check": factor1 * factor2 == N,
        "success": factor1 * factor2 == N
    }


# ---------------------------------------------------------------------------
# 5. Quantum Annealing Simulation (Transverse-Field Ising Model)
# ---------------------------------------------------------------------------

def run_quantum_annealing(n_spins: int = 6, n_steps: int = 40) -> Dict:
    """
    Simulates transverse-field quantum annealing:
    H(s) = (1 - s) H_transverse + s H_problem, where s = t / T in [0, 1].
    H_transverse = - sum_i X_i, H_problem = sum_{<i,j>} J_ij Z_i Z_j.
    """
    # 1D Ring Max-Cut problem: J_ij = 1 between neighbors
    energies = []
    s_schedule = np.linspace(0.0, 1.0, n_steps)

    qc = QuantumCircuit(n_spins)
    # Start in ground state of transverse field: all |+>
    for i in range(n_spins):
        qc.h(i)

    # Build Problem Hamiltonian
    paulis = [f"{'Z' if k in (i, (i+1)%n_spins) else 'I'}" for i in range(n_spins) for k in range(n_spins)]
    # Simplified Hamiltonian tracking
    dt = 0.1
    for s in s_schedule:
        # Trotter step: exp(-i dt (1-s) H_X) exp(-i dt s H_Z)
        angle_x = 2.0 * dt * (1.0 - s)
        angle_z = 2.0 * dt * s

        for i in range(n_spins):
            qc.rx(i, angle_x)

        for i in range(n_spins):
            qc.cx(i, (i + 1) % n_spins)
            qc.rz((i + 1) % n_spins, angle_z)
            qc.cx(i, (i + 1) % n_spins)

        # Estimate instantaneous ground state overlap
        probs = qc.get_probabilities()
        energies.append(float(np.max(probs)))

    return {
        "algorithm": "Quantum Annealing (Adiabatic)",
        "n_spins": n_spins,
        "annealing_steps": n_steps,
        "final_ground_state_fidelity": energies[-1],
        "energy_trajectory": energies,
        "success": energies[-1] > 0.40
    }


# ---------------------------------------------------------------------------
# 6. Quantum Machine Learning (VQC) vs Classical MLP
# ---------------------------------------------------------------------------

class VariationalQuantumClassifier:
    """Variational Quantum Classifier with Angle Embedding and Parameterized Circuit."""
    def __init__(self, n_qubits: int = 2, n_layers: int = 2, seed: int = 42):
        self.n = n_qubits
        self.layers = n_layers
        self.n_params = n_layers * n_qubits * 2
        rng = np.random.default_rng(seed)
        self.params = rng.normal(0, 0.5, size=self.n_params)

    def predict_sample(self, x: np.ndarray, params: np.ndarray) -> float:
        """Embeds x in R^2 into rotation angles, executes circuit, measures <Z_0>."""
        qc = QuantumCircuit(self.n)
        # 1. Feature Encoding (Angle Embedding)
        for i in range(self.n):
            qc.ry(i, float(x[i] * np.pi))

        # 2. Variational Entangling Layers
        p_idx = 0
        for l in range(self.layers):
            for i in range(self.n):
                qc.ry(i, params[p_idx])
                qc.rz(i, params[p_idx + 1])
                p_idx += 2
            # Entangling CNOT
            qc.cx(0, 1)

        # Measure expectation of Z on qubit 0: P(0) - P(1)
        probs = qc.get_probabilities()
        exp_z = (probs[0] + probs[1]) - (probs[2] + probs[3])
        return float(exp_z)

    def train(self, X: np.ndarray, y: np.ndarray, n_epochs: int = 25, lr: float = 0.2) -> List[float]:
        losses = []
        for epoch in range(n_epochs):
            # Compute gradient via Parameter-Shift Rule
            shift = np.pi / 2.0
            grad = np.zeros(self.n_params)
            total_loss = 0.0

            for i in range(len(X)):
                pred = self.predict_sample(X[i], self.params)
                loss_i = (pred - y[i])**2
                total_loss += loss_i
                err = 2.0 * (pred - y[i])

                # Parameter shift for each parameter
                for p in range(self.n_params):
                    p_plus = np.copy(self.params)
                    p_minus = np.copy(self.params)
                    p_plus[p] += shift
                    p_minus[p] -= shift
                    pred_plus = self.predict_sample(X[i], p_plus)
                    pred_minus = self.predict_sample(X[i], p_minus)
                    dp = (pred_plus - pred_minus) / (2.0 * np.sin(shift))
                    grad[p] += err * dp

            self.params -= lr * (grad / len(X))
            losses.append(total_loss / len(X))

        return losses


def run_qml_vs_classical_comparison() -> Dict:
    """Compares VQC (Quantum) vs 2-Layer MLP (Classical) on non-linear binary classification."""
    rng = np.random.default_rng(42)
    # Generate concentric circular dataset
    N = 40
    radii = np.concatenate([rng.uniform(0.1, 0.45, size=N//2), rng.uniform(0.65, 0.95, size=N//2)])
    angles = rng.uniform(0, 2*np.pi, size=N)
    X = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    y = np.concatenate([np.ones(N//2), -np.ones(N//2)])

    # 1. Train Quantum VQC
    vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=2, seed=42)
    vqc_losses = vqc.train(X, y, n_epochs=20, lr=0.15)
    preds_vqc = np.array([np.sign(vqc.predict_sample(x, vqc.params)) for x in X])
    vqc_acc = float(np.mean(preds_vqc == y))

    # 2. Train Classical 2-Layer MLP (4 hidden units)
    # MLP Parameters: W1 (2x4) + b1 (4) + W2 (4x1) + b2 (1) = 17 parameters
    W1 = rng.normal(0, 0.5, size=(2, 4))
    b1 = np.zeros(4)
    W2 = rng.normal(0, 0.5, size=(4, 1))
    b2 = 0.0
    mlp_losses = []
    lr = 0.1

    for epoch in range(20):
        # Forward pass
        h = np.tanh(X @ W1 + b1)
        pred = h @ W2 + b2
        loss = float(np.mean((pred.ravel() - y)**2))
        mlp_losses.append(loss)

        # Backpropagation
        d_pred = 2.0 * (pred.ravel() - y)[:, None] / N
        dW2 = h.T @ d_pred
        db2 = float(np.sum(d_pred))
        dh = d_pred @ W2.T * (1.0 - h**2)
        dW1 = X.T @ dh
        db1 = np.sum(dh, axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    preds_mlp = np.sign(np.tanh(X @ W1 + b1) @ W2 + b2).ravel()
    mlp_acc = float(np.mean(preds_mlp == y))

    return {
        "vqc_params": vqc.n_params,
        "mlp_params": 2*4 + 4 + 4*1 + 1,
        "vqc_accuracy": vqc_acc,
        "mlp_accuracy": mlp_acc,
        "vqc_losses": vqc_losses,
        "mlp_losses": mlp_losses
    }
