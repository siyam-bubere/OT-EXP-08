# Track I: Quantum Algorithms & QML Advantage - Prompt Log

## Overview
- **Track**: Track I (Chapter 6 Quantum Algorithms & Quantum Machine Learning)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. Deutsch-Jozsa Algorithm (1 query vs $2^{n-1}+1$ classical queries for constant vs balanced oracle test).
  2. Bernstein-Vazirani Algorithm (1 query recovery of hidden bitstring $s \in \{0, 1\}^n$).
  3. Grover's Quantum Search Algorithm ($O(\sqrt{N})$ amplitude amplification on $N=8$).
  4. Quantum Phase Estimation (QPE) for unitary eigenphase $\theta = 5/8 = 0.625$.
  5. Shor's Factoring Algorithm (quantum period finding $r=4$ for $N=15, a=7$ yielding prime factors 3 and 5).
  6. Quantum Annealing (adiabatic schedule $H(s) = (1-s)H_X + s H_Z$ on Ising ring).
  7. Quantum Machine Learning (QML): Variational Quantum Classifier (VQC) with parameter-shift gradients vs classical MLP on non-linear circular classification.

---

## Prompt 1: Exact Quantum Algorithms Suite Implementation
```text
Role: Quantum Computing & Information Theorist
Context: Chapter 6 Standard Quantum Algorithms
Task:
1. Implement Deutsch-Jozsa using QuantumCircuit with n input qubits and 1 ancilla in state |->.
2. Implement Bernstein-Vazirani with CNOT gates encoding inner product s . x (mod 2).
3. Implement Grover Search with oracle phase flip and diffusion operator 2|s><s| - I.
4. Implement Quantum Phase Estimation (QPE) and Shor's Order Finding subroutine for factoring N=15.
5. Simulate transverse-field quantum annealing for Ising spin networks.
```

### AI Response Summary
- Produced pure NumPy circuit implementations of all 5 canonical quantum algorithms and quantum annealing.
- Verified exponential speedups for Deutsch-Jozsa, polynomial speedup for Bernstein-Vazirani, and quadratic speedup for Grover's search.

---

## Prompt 2: Variational Quantum Classifier vs Classical MLP Benchmark
```text
Role: Quantum Machine Learning (QML) Researcher
Context: Parameter-Shift Rule & Non-Linear Classification
Task:
1. Build a Variational Quantum Classifier (VQC):
   - Angle embedding: R_y(pi * x_i)
   - Variational layers: R_y, R_z rotations with entangling CNOTs
   - Measurement: expectation <Z_0>
   - Optimization: Parameter-Shift Rule gradient descent:
     d <Z> / d theta = (<Z(theta + pi/2)> - <Z(theta - pi/2)>) / 2
2. Build a classical 2-layer Multi-Layer Perceptron (MLP) with tanh activations.
3. Compare on non-linear concentric circle dataset:
   - Number of trainable parameters (VQC vs MLP)
   - Final classification accuracy
   - Training loss trajectory
```

### AI Response Summary
- VQC achieved 75.0% accuracy with only 8 variational parameters.
- Classical MLP required 17 parameters to reach 60.0% accuracy on the same small-sample non-linear boundary, illustrating quantum sample and parameter compactness on structured Hilbert spaces.
