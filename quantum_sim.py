"""
quantum_sim.py - Pure NumPy Quantum Circuit & Statevector Simulator
Optimization Techniques Lab - Assignment 8 (Chapter 6 Coverage)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Features:
1. Exact N-qubit statevector evolution (|psi> in C^{2^N}).
2. Comprehensive gate library: H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ, SWAP, Toffoli.
3. Parametric quantum circuits U(theta) with parameter-shift rule.
4. Pauli Hamiltonian expectation evaluation <psi|H|psi>.
5. SPSA (Simultaneous Perturbation Stochastic Approximation).
6. Quantum Natural Gradient (QNG) via Fubini-Study metric tensor.
7. Finite-shot projective measurement sampling.
8. Density matrix & Kraus noise channels (Depolarizing, Amplitude Damping).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable


# Standard Single-Qubit Matrices
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X_GATE = np.array([[0, 1], [1, 0]], dtype=complex)
Y_GATE = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_GATE = np.array([[1, 0], [0, -1]], dtype=complex)
H_GATE = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [-1j * np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def ry(theta: float) -> np.ndarray:
    return np.array([
        [np.cos(theta / 2), -np.sin(theta / 2)],
        [np.sin(theta / 2), np.cos(theta / 2)]
    ], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


class QuantumCircuit:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 1 << n_qubits
        self.gates = []  # List of (gate_name, target_qubits, param_index or value)
        self.n_params = 0

    def h(self, target: int):
        self.gates.append(("H", (target,), None))
        return self

    def x(self, target: int):
        self.gates.append(("X", (target,), None))
        return self

    def y(self, target: int):
        self.gates.append(("Y", (target,), None))
        return self

    def z(self, target: int):
        self.gates.append(("Z", (target,), None))
        return self

    def s(self, target: int):
        self.gates.append(("S", (target,), None))
        return self

    def t(self, target: int):
        self.gates.append(("T", (target,), None))
        return self

    def rx(self, target: int, param_idx_or_val):
        if isinstance(param_idx_or_val, int):
            self.n_params = max(self.n_params, param_idx_or_val + 1)
        self.gates.append(("Rx", (target,), param_idx_or_val))
        return self

    def ry(self, target: int, param_idx_or_val):
        if isinstance(param_idx_or_val, int):
            self.n_params = max(self.n_params, param_idx_or_val + 1)
        self.gates.append(("Ry", (target,), param_idx_or_val))
        return self

    def rz(self, target: int, param_idx_or_val):
        if isinstance(param_idx_or_val, int):
            self.n_params = max(self.n_params, param_idx_or_val + 1)
        self.gates.append(("Rz", (target,), param_idx_or_val))
        return self

    def cnot(self, control: int, target: int):
        self.gates.append(("CNOT", (control, target), None))
        return self

    def cx(self, control: int, target: int):
        return self.cnot(control, target)

    def cz(self, control: int, target: int):
        self.gates.append(("CZ", (control, target), None))
        return self

    def swap(self, q1: int, q2: int):
        self.gates.append(("SWAP", (q1, q2), None))
        return self

    def toffoli(self, c1: int, c2: int, target: int):
        self.gates.append(("TOFFOLI", (c1, c2, target), None))
        return self

    def _apply_single_qubit_gate(self, state: np.ndarray, target: int, matrix: np.ndarray) -> np.ndarray:
        # Reshape to (2, 2, ..., 2) of length n_qubits
        shape = [2] * self.n_qubits
        state_tensor = state.reshape(shape)
        # Apply matrix along target axis
        state_tensor = np.tensordot(matrix, state_tensor, axes=(1, target))
        # tensordot places the new target axis at position 0. Move back to 'target'
        state_tensor = np.moveaxis(state_tensor, 0, target)
        return state_tensor.reshape(-1)

    def _apply_two_qubit_gate(self, state: np.ndarray, q1: int, q2: int, matrix_4x4: np.ndarray) -> np.ndarray:
        shape = [2] * self.n_qubits
        state_tensor = state.reshape(shape)
        matrix_tensor = matrix_4x4.reshape((2, 2, 2, 2))
        state_tensor = np.tensordot(matrix_tensor, state_tensor, axes=([2, 3], [q1, q2]))
        # Roll back
        state_tensor = np.moveaxis(state_tensor, [0, 1], [q1, q2])
        return state_tensor.reshape(-1)

    def simulate(self, params: Optional[np.ndarray] = None, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        if initial_state is None:
            state = np.zeros(self.dim, dtype=complex)
            state[0] = 1.0  # |00...0>
        else:
            state = np.copy(initial_state).astype(complex)

        for gate, qubits, param in self.gates:
            angle = 0.0
            if param is not None:
                angle = params[param] if isinstance(param, int) else float(param)

            if gate == "H":
                state = self._apply_single_qubit_gate(state, qubits[0], H_GATE)
            elif gate == "X":
                state = self._apply_single_qubit_gate(state, qubits[0], X_GATE)
            elif gate == "Y":
                state = self._apply_single_qubit_gate(state, qubits[0], Y_GATE)
            elif gate == "Z":
                state = self._apply_single_qubit_gate(state, qubits[0], Z_GATE)
            elif gate == "S":
                state = self._apply_single_qubit_gate(state, qubits[0], S_GATE)
            elif gate == "T":
                state = self._apply_single_qubit_gate(state, qubits[0], T_GATE)
            elif gate == "Rx":
                state = self._apply_single_qubit_gate(state, qubits[0], rx(angle))
            elif gate == "Ry":
                state = self._apply_single_qubit_gate(state, qubits[0], ry(angle))
            elif gate == "Rz":
                state = self._apply_single_qubit_gate(state, qubits[0], rz(angle))
            elif gate == "CNOT":
                c, t = qubits
                cnot_mat = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]
                ], dtype=complex)
                state = self._apply_two_qubit_gate(state, c, t, cnot_mat)
            elif gate == "CZ":
                c, t = qubits
                cz_mat = np.diag([1, 1, 1, -1]).astype(complex)
                state = self._apply_two_qubit_gate(state, c, t, cz_mat)
            elif gate == "SWAP":
                q1, q2 = qubits
                swap_mat = np.array([
                    [1, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1]
                ], dtype=complex)
                state = self._apply_two_qubit_gate(state, q1, q2, swap_mat)
            elif gate == "TOFFOLI":
                c1, c2, t = qubits
                shape = [2] * self.n_qubits
                state_tensor = state.reshape(shape)
                idx_0 = [slice(None)] * self.n_qubits
                idx_1 = [slice(None)] * self.n_qubits
                idx_0[c1], idx_0[c2], idx_0[t] = 1, 1, 0
                idx_1[c1], idx_1[c2], idx_1[t] = 1, 1, 1
                tmp = np.copy(state_tensor[tuple(idx_0)])
                state_tensor[tuple(idx_0)] = state_tensor[tuple(idx_1)]
                state_tensor[tuple(idx_1)] = tmp
                state = state_tensor.reshape(-1)

        return state

    def get_probabilities(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        state = self.simulate(params)
        return np.abs(state) ** 2

    def sample_measurements(self, params: Optional[np.ndarray] = None, shots: int = 1000, seed: int = 42) -> Dict[str, int]:
        probs = self.get_probabilities(params)
        rng = np.random.default_rng(seed)
        samples = rng.choice(self.dim, size=shots, p=probs)
        counts = {}
        for s in samples:
            bitstr = format(s, f"0{self.n_qubits}b")
            counts[bitstr] = counts.get(bitstr, 0) + 1
        return counts


class PauliHamiltonian:
    """Represents a Hamiltonian as a linear combination of Pauli strings: H = sum c_k P_k."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.terms: List[Tuple[float, str]] = []  # (coeff, "XYZI...")
        self._cached_matrix: Optional[np.ndarray] = None

    def add_term(self, coeff: float, pauli_str: str):
        assert len(pauli_str) == self.n_qubits, f"Length must match {self.n_qubits}"
        self.terms.append((float(coeff), pauli_str.upper()))
        self._cached_matrix = None

    def get_matrix(self) -> np.ndarray:
        if self._cached_matrix is not None:
            return self._cached_matrix

        dim = 1 << self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        p_map = {'I': I2, 'X': X_GATE, 'Y': Y_GATE, 'Z': Z_GATE}

        for coeff, p_str in self.terms:
            mat = p_map[p_str[0]]
            for p in p_str[1:]:
                mat = np.kron(mat, p_map[p])
            H += coeff * mat

        self._cached_matrix = H
        return H

    def expectation(self, state: np.ndarray) -> float:
        """Computes exact expectation value <psi|H|psi>."""
        H_mat = self.get_matrix()
        val = np.vdot(state, H_mat @ state).real
        return float(val)

    def expectation_from_circuit(self, circuit: QuantumCircuit, params: np.ndarray, shots: Optional[int] = None, seed: int = 42) -> float:
        state = circuit.simulate(params)
        exact_val = self.expectation(state)
        if shots is None:
            return exact_val
        else:
            rng = np.random.default_rng(seed)
            noise_std = max(1e-4, np.sqrt(max(0.01, 1.0 - min(1.0, abs(exact_val)**2))) / np.sqrt(shots))
            return float(exact_val + rng.normal(0, noise_std))


def parameter_shift_gradient(circuit: QuantumCircuit, H: PauliHamiltonian, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
    """Exact analytic gradient computation using the Parameter-Shift Rule."""
    grad = np.zeros_like(params, dtype=float)
    for i in range(len(params)):
        p_plus = np.copy(params)
        p_minus = np.copy(params)
        p_plus[i] += shift
        p_minus[i] -= shift

        e_plus = H.expectation_from_circuit(circuit, p_plus)
        e_minus = H.expectation_from_circuit(circuit, p_minus)
        grad[i] = 0.5 * (e_plus - e_minus)
    return grad


def spsa_gradient(circuit: QuantumCircuit, H: PauliHamiltonian, params: np.ndarray, c_k: float = 0.1, seed: Optional[int] = None) -> np.ndarray:
    """Simultaneous Perturbation Stochastic Approximation (SPSA) gradient estimator."""
    rng = np.random.default_rng(seed)
    delta = rng.choice([-1.0, 1.0], size=len(params))
    p_plus = params + c_k * delta
    p_minus = params - c_k * delta

    e_plus = H.expectation_from_circuit(circuit, p_plus)
    e_minus = H.expectation_from_circuit(circuit, p_minus)

    grad = (e_plus - e_minus) / (2.0 * c_k * delta)
    return grad


def compute_fubini_study_metric(circuit: QuantumCircuit, params: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """
    Computes the real part of the Fubini-Study metric tensor (Quantum Fisher Information matrix):
    g_ij = Re[ <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> ]
    """
    n_params = len(params)
    psi = circuit.simulate(params)

    d_psi = []
    for i in range(n_params):
        p_plus = np.copy(params)
        p_minus = np.copy(params)
        p_plus[i] += eps
        p_minus[i] -= eps
        psi_p = circuit.simulate(p_plus)
        psi_m = circuit.simulate(p_minus)
        d_psi.append((psi_p - psi_m) / (2.0 * eps))

    g = np.zeros((n_params, n_params), dtype=float)
    for i in range(n_params):
        for j in range(i, n_params):
            term1 = np.vdot(d_psi[i], d_psi[j])
            term2 = np.vdot(d_psi[i], psi) * np.vdot(psi, d_psi[j])
            val = (term1 - term2).real
            g[i, j] = val
            g[j, i] = val

    return g + 1e-4 * np.eye(n_params)


def quantum_natural_gradient(circuit: QuantumCircuit, H: PauliHamiltonian, params: np.ndarray) -> np.ndarray:
    """Quantum Natural Gradient step: g_inv @ grad."""
    grad = parameter_shift_gradient(circuit, H, params)
    F = compute_fubini_study_metric(circuit, params)
    nat_grad = np.linalg.solve(F, grad)
    return nat_grad


def apply_depolarizing_channel(rho: np.ndarray, p_error: float, n_qubits: int) -> np.ndarray:
    dim = 1 << n_qubits
    return (1.0 - p_error) * rho + (p_error / dim) * np.eye(dim, dtype=complex)


def apply_amplitude_damping_channel(rho: np.ndarray, gamma: float) -> np.ndarray:
    E0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex)
    E1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex)
    return E0 @ rho @ E0.conj().T + E1 @ rho @ E1.conj().T
