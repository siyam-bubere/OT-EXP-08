# Vibe-Coding Prompt Log: Track A — Continuous Optimization
**Assignment:** Optimization Techniques Lab — Assignment 8  
**Student:** Siyam Bubere (Reg. Id: 241070012) | VJTI Mumbai  
**Track:** Track A (Continuous Optimization)  
**Tools Used:** Antigravity AI, Claude 3.7 Sonnet  
**Date:** November 2026  

---

## 1. Initial Prompt (Problem Formulation & Implementation)
```text
Task: Implement from scratch in pure Python with NumPy all required continuous optimization algorithms taught in Chapter 6:
1. Gradient Descent (GD), SGD, Mini-batch SGD.
2. Momentum, Nesterov Accelerated Gradient (NAG), AdaGrad, RMSProp, Adam.
3. Second-order / curvature methods: Newton-Raphson, Gauss-Newton for non-linear least squares.
4. From-scratch 2-layer MLP backpropagation engine decoupled from the optimizer.
5. Curvature approximations: Empirical Fisher, Diagonal Hessian, Pearlmutter Hessian-Vector Products (HVP).
6. Landscapes: Ill-conditioned strongly convex quadratic bowl (kappa=20) and a non-convex saddle function (x^2 - y^2 + 0.25*y^4) with local minima.
7. Conduct step-size sensitivity and noise robustness experiments.
```

## 2. AI Response Summary & Code Generation
- Generated `continuous_optimizers.py` with standalone functional implementations of GD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, and Gauss-Newton.
- Provided analytical Jacobians and Hessians for the benchmark functions.
- Formulated the exact finite-difference HVP $H v \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x - \epsilon v)}{2\epsilon}$.
- Built the 2-layer MLP with forward pass, MSE loss, and analytical backward pass through ReLU activations.

## 3. Human Critical Review & Verification
- **Audit Findings:**
  1. Newton's method on ill-conditioned or saddle landscapes can fail if the Hessian is not positive definite or is singular. Introduced Levenberg-Marquardt regularization ($H + \lambda I$).
  2. Nesterov momentum originally computed the gradient at the current position instead of the lookahead point $x - \beta v$. Corrected to properly evaluate $\nabla f(x - \beta v)$.
  3. Verified HVP against analytical Hessian: exact $H v = [12.0, 0.8]^T$, Pearlmutter approximation yielded $[12.0, 0.8]^T$ with an absolute error of $2.80 \times 10^{-11}$.

## 4. Benchmark Results & Traceability
- **Convex Bowl:** Newton converged in 1 step; Adam converged in 35 steps; standard GD suffered from ravine oscillations.
- **Saddle Point:** Momentum and Adam successfully escaped the saddle point at $(0,0)$, whereas standard GD stagnated with tiny gradient steps.
- **Backpropagation:** Adam achieved final MSE loss of 0.00887, outperforming standard SGD (0.08591).
- **Artifacts Generated:**
  - `track_a_continuous/continuous_optimizers.py`
  - `track_a_continuous/run_track_a.py`
  - `results/track_a_continuous_metrics.csv`
  - `plots/track_a_convex_trajectories.png`
  - `plots/track_a_nonconvex_saddle_trajectories.png`
  - `plots/track_a_mlp_training.png`
  - `plots/track_a_noise_sensitivity.png`
