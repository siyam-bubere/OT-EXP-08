# Vibe-Coding Prompt Log: Track C — Quantum Optimization
**Assignment:** Optimization Techniques Lab — Assignment 8  
**Student:** Siyam Bubere (Reg. Id: 241070012) | VJTI Mumbai  
**Track:** Track C (Quantum Optimization)  
**Tools Used:** Antigravity AI, Claude 3.7 Sonnet  
**Date:** November 2026  

---

## 1. Initial Prompt (Problem Formulation & Implementation)
```text
Task: Implement the Chapter 6 quantum optimization methods:
1. QAOA (Quantum Approximate Optimization Algorithm) on the 8-node Max-Cut graph from Track B:
   - Alternating cost U_C(gamma) and mixer U_B(beta) layers for depth p=1, 2, 3.
   - Compare with classical ground truth (Max Cut = 14.00).
2. Variational Quantum Eigensolver (VQE) for a transverse Ising model:
   - Compare Parameter-Shift Rule, SPSA, and Quantum Natural Gradient (QNG).
3. Barren Plateaus Study:
   - Measure gradient variance Var[d<H>/dtheta] across qubit count n in [2, 4, 6, 8, 10].
   - Demonstrate exponential decay Var ~ O(2^{-n}).
4. Hardware effects:
   - Finite measurement shots (100, 500, 2000, 10000).
   - Depolarizing and amplitude damping noise channels.
```

## 2. AI Response Summary & Code Generation
- Built `quantum_optimizers.py` with QAOA, VQE, and Barren Plateaus simulation routines.
- Evaluated Parameter-Shift Rule $\frac{\partial \langle H \rangle}{\partial \theta_j} = \frac{1}{2}[\langle H \rangle_{\theta_j + \pi/2} - \langle H \rangle_{\theta_j - \pi/2}]$.
- Implemented Quantum Natural Gradient by computing the Fubini-Study metric tensor $g_{ij} = \text{Re}[\langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle]$.

## 3. Human Critical Review & AI Failure Rectification (Documented Bug 2)
- **AI Tool Inefficiency Detected:**
  - In `PauliHamiltonian`, `self.get_matrix()` recalculated $2^n \times 2^n$ Kronecker matrix products on every single expectation call. For a 35-step QAOA optimization with 6 parameters and 12 Hamiltonian terms, this triggered over 35,000 tensor products, freezing execution.
  - Furthermore, in the Barren Plateau study, `parameter_shift_gradient` computed shifts for all 40 parameters instead of just $\theta_0$.
  - **Correction:** Cached the generated matrix inside `_cached_matrix` so it is constructed exactly once. Optimized the Barren Plateau script to evaluate the parameter-shift specifically for $\theta_0$.
  - **Result:** Execution time dropped from >3 minutes to **11.05 seconds** (a ~20x speedup).

## 4. Benchmark Results & Traceability
- **QAOA Approximation Ratio:** Achieved 63.1% of classical maximum cut at $p=3$.
- **VQE Ground State Energy:** QNG reached the exact analytical ground energy of **-1.4142**, outperforming parameter-shift (-1.3956) and SPSA (0.2382).
- **Barren Plateaus:** Verified exponential concentration:
  - $n=2$: $\text{Var} = 1.49 \times 10^{-1}$
  - $n=4$: $\text{Var} = 5.56 \times 10^{-2}$
  - $n=6$: $\text{Var} = 1.20 \times 10^{-2}$
  - $n=8$: $\text{Var} = 4.33 \times 10^{-3}$
  - $n=10$: $\text{Var} = 7.62 \times 10^{-4}$
- **Artifacts Generated:**
  - `track_c_quantum_opt/quantum_optimizers.py`
  - `track_c_quantum_opt/run_track_c.py`
  - `results/track_c_quantum_metrics.csv`
  - `plots/track_c_qaoa_convergence.png`
  - `plots/track_c_vqe_methods_comparison.png`
  - `plots/track_c_barren_plateaus.png`
