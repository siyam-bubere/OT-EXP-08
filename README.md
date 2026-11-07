# Optimization Techniques: Lab Experiment 8
## OPTIMIZE ANYTHING: The 14-Day Vibe-Coding Challenge
**Veermata Jijabai Technological Institute (VJTI), Mumbai-19**

- **Student Name**: Siyam Bubere
- **Registration ID**: 241070012
- **Course**: Optimization Techniques
- **Assignment**: Lab Experiment 8 (10 Compulsory Tracks A–J)

---

## 1. Executive Summary & Syllabus Mapping
This repository contains the complete implementation, experimental benchmarking, and mathematical verification for **all 10 compulsory tracks (Tracks A through J)** covering the entire **Chapter 6 syllabus** of Optimization Techniques:

| Track | Module Title | Optimization Formulation & Key Algorithms | Primary Result / Finding |
| :--- | :--- | :--- | :--- |
| **Track A** | Continuous Optimization | GD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton-Raphson, Gauss-Newton, Pearlmutter HVP, MLP Backpropagation | Adam reached $1.8 \times 10^{-7}$ loss; Newton quadratic convergence; HVP accurately captured curvature along valleys |
| **Track B** | Discrete Optimization | Greedy, Dynamic Programming, Branch-and-Bound, ILP (via SciPy `milp`), QUBO Max-Cut & Knapsack | Exact Max-Cut $C^* = 14.00$ on 8-node Ising graph; Knapsack exact DP value = 275 |
| **Track C** | Quantum Optimization | QAOA ($p \in \{1, 2, 3\}$), VQE (Parameter-Shift, SPSA, QNG), Barren Plateaus Study ($n=2..10$), Measurement Noise | QNG achieved exact ground state $E_0 = -1.4142$ in 14 steps; Barren Plateau gradient variance decayed from $1.49 \times 10^{-1}$ to $7.62 \times 10^{-4}$ |
| **Track D** | Submodular, Parallel & Distributed | Sensor Coverage $(1 - 1/e)$ Bound, Work-Stealing Amdahl/Gustafson, Non-IID FedAvg ($E \in \{1, 5, 20\}$), MGDA vs GradNorm | Greedy reached 100% OPT (exceeding 63.2% bound); GradNorm dynamically balanced conflicting tasks (Loss: 0.0062 vs 0.0782) |
| **Track E** | Main Case Study: Smart Microgrid | 24-hr Dynamic Dispatch with Solar PV, 250 kWh BESS, Corporate EV Fleet Logistics, ToU Tariffs, Epigraph Peak Shaving | **51.4% total cost reduction** ($1,183.92 -> $575.12); **93.6% peak demand reduction** (80.9 kW -> 5.2 kW) |
| **Track F** | Compiler Optimization | 4 Passes (Constant Folding/Prop, CSE, LICM, DCE) + Chaitin-Briggs Graph Coloring Register Allocation | **23.1% instruction reduction** (26 -> 20 inst); 100% semantic equivalence verified; spill analysis on $K=3, 4$ |
| **Track G** | xv6 OS Scheduling | 4-Queue Multi-Level Feedback Queue (MLFQ) with Anti-Starvation Aging vs Round-Robin | **85.7% response time reduction** on interactive tasks (1.8t -> 0.2t); **15.8% turnaround improvement** on mixed jobs |
| **Track H** | DBMS Optimization | Selinger DP Join-Order Enumeration, 0/1 Knapsack Automated Index Selection, SQLite `EXPLAIN` Verification | **99.37% intermediate join cost reduction**; **2.96x query speedup** (66.2% latency reduction) within 2.5 MB storage |
| **Track I** | Quantum Algorithms & QML | Deutsch-Jozsa, Bernstein-Vazirani, Grover, QPE, Shor's Factoring ($N=15$), Quantum Annealing, VQC vs MLP | 100% exact recovery across oracles; Grover found marked state in 2 queries (94.5% prob); VQC outperformed classical MLP |
| **Track J** | Software Engineering TSMP | 0/1 ILP Set Cover Test-Suite Minimization & Additional Coverage Prioritization (APFD) on 1,200+ LOC codebase | **71.7% CI runtime reduction** (2,226 ms -> 630 ms) preserving **100.0% statement coverage**; APFD reached 0.926 |

---

## 2. Mandatory Vibe-Coding Disclosure

### 2.1 AI Tools Register
- **Primary AI Assistant**: Antigravity (Google DeepMind Advanced Agentic Coding, Gemini 2.5 Pro reasoning architecture).
- **Secondary Tools**: GitHub Copilot CLI, Cursor IDE (syntax and prompt refinement).
- **Total Development Duration**: 14-day equivalent sprint executed with agentic pair programming.
- **Verification Strategy**: 100% independent mathematical verification, unit testing, assertion checks, and empirical benchmarking. Zero unverified code incorporated.

### 2.2 Documented AI Failure Cases & Corrections

#### Failure Case 1: Track B — SciPy `milp` Bounds Interface Compatibility
- **Initial Generated Code**: The AI generated `bounds=[(0, 1) for _ in range(n)]` for `scipy.optimize.milp`.
- **Failure**: SciPy's modern `milp` interface rejected the standard list-of-tuples format, raising `ValueError: bounds must be convertible into an instance of scipy.optimize.Bounds`.
- **Correction**: Replaced with `from scipy.optimize import Bounds; bounds = Bounds(lb=0.0, ub=1.0)`. Documented in `prompts/track_b/prompt_log.md`.

#### Failure Case 2: Track C — Exponential Kronecker Product Explosion in Hamiltonian Expectation
- **Initial Generated Code**: The AI recalculated the $2^n \times 2^n$ matrix tensor product via Kronecker products on every expectation value evaluation in parameter-shift loops.
- **Failure**: For $n=8$ qubits and multi-angle QAOA, circuit expectation calls exceeded 3 minutes, causing benchmark timeouts.
- **Correction**: Implemented lazy matrix caching (`self._cached_matrix`) inside `PauliHamiltonian`, generating the matrix once and performing $O(2^n)$ matrix-vector multiplications. Runtime dropped from >180s to 11.05s. Documented in `prompts/track_c/prompt_log.md`.

#### Failure Case 3: Track D — Python GIL Serialization in Parallel Work-Stealing Kernel
- **Initial Generated Code**: The AI drafted a pure Python scalar loop executing `int(500 + 400 * np.sin(val))` transcendental functions per item over 80,000 floats inside `ThreadPoolExecutor`.
- **Failure**: Because scalar Python bytecode cannot release the Global Interpreter Lock (GIL), worker threads contended on the GIL, resulting in severe CPU thrashing (>60s runtime) and an anomalous speedup curve.
- **Correction**: Refactored the numerical kernel into vectorized NumPy chunk operations (`np.sin`, `np.cos`, `np.sqrt`), which release the GIL in compiled C-extensions, producing realistic multi-core speedup curves in 0.11s. Documented in `prompts/track_d/prompt_log.md`.

#### Failure Case 4: Track F — Basic Block Boundary Leakage in Constant Propagation
- **Initial Generated Code**: In the compiler optimization pass, `constants` was maintained globally without clearing at control flow labels (`LABEL`, `JUMP`, `IF_LE`).
- **Failure**: Constant propagation carried the pre-loop initialization `i = 1` into the loop body, folding `step_val = 1 * loop_const` and corrupting loop semantics (`accum = 214` instead of `522`).
- **Correction**: Enforced strict basic block scoping by resetting active constants (`constants.clear()`) at any branch or label delimiter. Verified 100% semantic equivalence. Documented in `prompts/track_f/prompt_log.md`.

---

## 3. Directory Structure
```
optimization_techniques_assignment8/
├── quantum_sim.py                  # Pure NumPy N-qubit quantum circuit & statevector simulator
├── track_a_continuous/             # Continuous Optimization (GD, Adam, Newton, MLP, HVP)
│   ├── continuous_optimizers.py
│   └── run_track_a.py
├── track_b_discrete/               # Discrete Optimization (DP, B&B, ILP, QUBO Max-Cut)
│   ├── discrete_solvers.py
│   └── run_track_b.py
├── track_c_quantum_opt/            # Quantum Optimization (QAOA, VQE, Barren Plateaus)
│   ├── quantum_optimizers.py
│   └── run_track_c.py
├── track_d_submodular_parallel/    # Submodular Coverage, Amdahl's Law, FedAvg, MGDA/GradNorm
│   ├── parallel_distributed.py
│   └── run_track_d.py
├── track_e_case_study/             # Real-Process Case Study: Microgrid Dynamic Dispatch
│   ├── microgrid_dispatch.py
│   └── run_track_e.py
├── track_f_compiler/               # Mini-Compiler 4 Passes & Chaitin Register Allocation
│   ├── mini_compiler.py
│   └── run_track_f.py
├── track_g_xv6/                    # xv6 OS Multi-Level Feedback Queue Scheduler Simulation & Patch
│   ├── xv6_scheduler_sim.py
│   ├── xv6_mlfq_priority_scheduler.patch
│   └── run_track_g.py
├── track_h_dbms/                   # DBMS Selinger DP Join Enumeration & Knapsack Index Selection
│   ├── dbms_optimizer.py
│   └── run_track_h.py
├── track_i_quantum_algorithms/     # Deutsch-Jozsa, BV, Grover, QPE, Shor, Annealing, VQC vs MLP
│   ├── quantum_algorithms_qml.py
│   └── run_track_i.py
├── track_j_swe_opt/                # Test-Suite Minimization & APFD Prioritization on 1,200 LOC
│   ├── test_suite_optimizer.py
│   └── run_track_j.py
├── prompts/                        # Mandatory Prompt Logs for All 10 Tracks (Tracks A-J)
│   ├── track_a/prompt_log.md
│   ├── track_b/prompt_log.md
│   ├── ...
│   └── track_j/prompt_log.md
├── results/                        # Experimental CSV Metrics for All 10 Tracks
├── plots/                          # 26 High-Resolution PNG Plots for Technical Report
└── README.md                       # This Comprehensive Project Manifest
```

---

## 4. Execution & Reproduction Guide

### Prerequisites
- Python 3.10+
- NumPy, SciPy, Matplotlib, Pandas, ReportLab
```bash
pip install numpy scipy matplotlib pandas reportlab
```

### Running All Benchmarks
To run the entire suite and regenerate all metrics and plots:
```bash
python track_a_continuous/run_track_a.py
python track_b_discrete/run_track_b.py
python track_c_quantum_opt/run_track_c.py
python track_d_submodular_parallel/run_track_d.py
python track_e_case_study/run_track_e.py
python track_f_compiler/run_track_f.py
python track_g_xv6/run_track_g.py
python track_h_dbms/run_track_h.py
python track_i_quantum_algorithms/run_track_i.py
python track_j_swe_opt/run_track_j.py
```

### Generating the Final Technical Report PDF
```bash
python generate_pdf_report_lab8.py
```
This generates `Optimization_Techniques_Lab8_Report.pdf` with the dedicated VJTI cover page and complete analysis of Section 11 questions.
