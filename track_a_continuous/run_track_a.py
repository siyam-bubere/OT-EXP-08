"""
track_a_continuous/run_track_a.py - Experimental Benchmarking Suite for Track A
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
from track_a_continuous.continuous_optimizers import (
    ObjectiveLandscapes,
    run_gradient_descent,
    run_momentum,
    run_nesterov,
    run_adagrad,
    run_rmsprop,
    run_adam,
    run_newton,
    run_gauss_newton,
    compute_hvp,
    compute_empirical_fisher,
    MultiLayerPerceptron
)


def run_convex_quadratic_experiments():
    print("\n--- Running Track A.1: Convex Ill-Conditioned Quadratic Bowl ---")
    x0 = np.array([-4.0, 4.0])
    fn = ObjectiveLandscapes.ill_conditioned_quadratic
    g_fn = ObjectiveLandscapes.grad_quadratic
    h_fn = ObjectiveLandscapes.hess_quadratic

    results = {}
    results["GD"] = run_gradient_descent(x0, g_fn, fn, lr=0.045, n_steps=80)
    results["Momentum"] = run_momentum(x0, g_fn, fn, lr=0.045, beta=0.85, n_steps=80)
    results["NAG"] = run_nesterov(x0, g_fn, fn, lr=0.045, beta=0.85, n_steps=80)
    results["AdaGrad"] = run_adagrad(x0, g_fn, fn, lr=1.2, n_steps=80)
    results["RMSProp"] = run_rmsprop(x0, g_fn, fn, lr=0.25, beta=0.9, n_steps=80)
    results["Adam"] = run_adam(x0, g_fn, fn, lr=0.35, n_steps=80)
    results["Newton"] = run_newton(x0, g_fn, h_fn, fn, lr=1.0, n_steps=10)

    # Plot 2D Contour & Trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    X_grid = np.linspace(-5, 5, 200)
    Y_grid = np.linspace(-5, 5, 200)
    X, Y = np.meshgrid(X_grid, Y_grid)
    Z = 0.5 * (20.0 * X**2 + Y**2)

    levels = np.logspace(-1, 3, 25)
    ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)

    colors = {
        "GD": "#1f77b4", "Momentum": "#ff7f0e", "NAG": "#2ca02c",
        "AdaGrad": "#d62728", "RMSProp": "#9467bd", "Adam": "#8c564b", "Newton": "#e377c2"
    }

    for name, res in results.items():
        tr = res["traj"]
        ax1.plot(tr[:, 0], tr[:, 1], 'o-', label=name, color=colors[name], markersize=3, alpha=0.85, linewidth=1.5)

    ax1.plot(0, 0, 'r*', markersize=14, label="Global Min (0,0)")
    ax1.set_title("Optimizer Trajectories on Ill-Conditioned Bowl (kappa=20)", fontweight='bold')
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle=':')

    # Loss convergence curves
    for name, res in results.items():
        ax2.semilogy(res["losses"], label=name, color=colors[name], linewidth=2)

    ax2.set_title("Convergence Rate (Loss vs Iterations)", fontweight='bold')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss (log scale)")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/track_a_convex_trajectories.png", dpi=200)
    plt.close()

    return results


def run_nonconvex_saddle_experiments():
    print("\n--- Running Track A.2: Non-Convex Landscape with Saddle Point ---")
    # Start very close to the saddle point at (0, 0.05) with x1=2.0
    x0 = np.array([2.0, 0.05])
    fn = ObjectiveLandscapes.non_convex_saddle
    g_fn = ObjectiveLandscapes.grad_saddle
    h_fn = ObjectiveLandscapes.hess_saddle

    results = {}
    results["GD"] = run_gradient_descent(x0, g_fn, fn, lr=0.1, n_steps=80)
    results["Momentum"] = run_momentum(x0, g_fn, fn, lr=0.1, beta=0.9, n_steps=80)
    results["NAG"] = run_nesterov(x0, g_fn, fn, lr=0.1, beta=0.9, n_steps=80)
    results["AdaGrad"] = run_adagrad(x0, g_fn, fn, lr=0.5, n_steps=80)
    results["RMSProp"] = run_rmsprop(x0, g_fn, fn, lr=0.1, n_steps=80)
    results["Adam"] = run_adam(x0, g_fn, fn, lr=0.15, n_steps=80)

    # Plot 2D Contour & Trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    X_grid = np.linspace(-3, 3, 200)
    Y_grid = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(X_grid, Y_grid)
    Z = X**2 - Y**2 + 0.25 * Y**4

    ax1.contour(X, Y, Z, levels=30, cmap='coolwarm', alpha=0.7)

    colors = {
        "GD": "#1f77b4", "Momentum": "#ff7f0e", "NAG": "#2ca02c",
        "AdaGrad": "#d62728", "RMSProp": "#9467bd", "Adam": "#8c564b"
    }

    for name, res in results.items():
        tr = res["traj"]
        ax1.plot(tr[:, 0], tr[:, 1], 'o-', label=name, color=colors[name], markersize=3, alpha=0.85, linewidth=1.5)

    ax1.plot(0, 0, 'kx', markersize=12, label="Saddle Point (0,0)")
    ax1.plot(0, np.sqrt(2), 'g*', markersize=12, label="Global Min (0, +sqrt(2))")
    ax1.plot(0, -np.sqrt(2), 'g*', markersize=12, label="Global Min (0, -sqrt(2))")
    ax1.set_title("Non-Convex Saddle Landscape: Escaping Saddle Point", fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(True, linestyle=':')

    # Loss convergence curves
    for name, res in results.items():
        ax2.plot(res["losses"], label=name, color=colors[name], linewidth=2)

    ax2.set_title("Objective Function Value vs Iteration", fontweight='bold')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    plt.savefig("plots/track_a_nonconvex_saddle_trajectories.png", dpi=200)
    plt.close()

    return results


def run_gauss_newton_experiment():
    print("\n--- Running Track A.3: Gauss-Newton Non-Linear Curve Fitting ---")
    # Synthetic non-linear model: y = a * exp(b * t)
    # Ground truth: a = 2.5, b = -1.2
    rng = np.random.default_rng(42)
    t_data = np.linspace(0, 2, 40)
    y_true = 2.5 * np.exp(-1.2 * t_data)
    y_noisy = y_true + rng.normal(0, 0.05, size=len(t_data))

    def residual(theta):
        a, b = theta
        return (a * np.exp(b * t_data)) - y_noisy

    def jacobian(theta):
        a, b = theta
        dr_da = np.exp(b * t_data)
        dr_db = a * t_data * np.exp(b * t_data)
        return np.column_stack([dr_da, dr_db])

    theta0 = np.array([1.0, -0.2])
    res_gn = run_gauss_newton(residual, jacobian, theta0, lr=1.0, n_steps=15)
    theta_opt = res_gn["x_final"]
    print(f"  Gauss-Newton Initial: a={theta0[0]:.2f}, b={theta0[1]:.2f} | Loss: {res_gn['losses'][0]:.4f}")
    print(f"  Gauss-Newton Final:   a={theta_opt[0]:.4f}, b={theta_opt[1]:.4f} | Loss: {res_gn['losses'][-1]:.4f}")

    return res_gn


def run_backprop_mlp_experiment():
    print("\n--- Running Track A.4: Backpropagation 2-Layer Neural Network ---")
    # Non-linear regression: y = sin(2 * pi * x) + 0.3 * cos(4 * pi * x)
    rng = np.random.default_rng(42)
    N = 200
    X = np.sort(rng.uniform(-1, 1, size=(N, 1)), axis=0)
    y = np.sin(np.pi * X) + 0.3 * np.cos(3 * np.pi * X) + rng.normal(0, 0.05, size=(N, 1))

    # Compare SGD vs Momentum vs Adam on MLP training
    optimizers = ["SGD", "Momentum", "Adam"]
    mlp_losses = {}

    for opt_name in optimizers:
        mlp = MultiLayerPerceptron(d_in=1, d_hidden=32, d_out=1, seed=42)
        losses = []

        if opt_name == "SGD":
            lr = 0.05
            for ep in range(300):
                y_pred = mlp.forward(X)
                l = mlp.compute_loss(y_pred, y)
                losses.append(l)
                grads = mlp.backward(y)
                mlp.W1 -= lr * grads["dW1"]
                mlp.b1 -= lr * grads["db1"]
                mlp.W2 -= lr * grads["dW2"]
                mlp.b2 -= lr * grads["db2"]

        elif opt_name == "Momentum":
            lr = 0.03
            beta = 0.9
            v_W1, v_b1 = np.zeros_like(mlp.W1), np.zeros_like(mlp.b1)
            v_W2, v_b2 = np.zeros_like(mlp.W2), np.zeros_like(mlp.b2)
            for ep in range(300):
                y_pred = mlp.forward(X)
                l = mlp.compute_loss(y_pred, y)
                losses.append(l)
                grads = mlp.backward(y)
                v_W1 = beta * v_W1 + lr * grads["dW1"]
                v_b1 = beta * v_b1 + lr * grads["db1"]
                v_W2 = beta * v_W2 + lr * grads["dW2"]
                v_b2 = beta * v_b2 + lr * grads["db2"]
                mlp.W1 -= v_W1
                mlp.b1 -= v_b1
                mlp.W2 -= v_W2
                mlp.b2 -= v_b2

        elif opt_name == "Adam":
            lr = 0.02
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            mW1, vW1 = np.zeros_like(mlp.W1), np.zeros_like(mlp.W1)
            mb1, vb1 = np.zeros_like(mlp.b1), np.zeros_like(mlp.b1)
            mW2, vW2 = np.zeros_like(mlp.W2), np.zeros_like(mlp.W2)
            mb2, vb2 = np.zeros_like(mlp.b2), np.zeros_like(mlp.b2)

            for ep in range(1, 301):
                y_pred = mlp.forward(X)
                l = mlp.compute_loss(y_pred, y)
                losses.append(l)
                grads = mlp.backward(y)

                # W1
                mW1 = beta1 * mW1 + (1 - beta1) * grads["dW1"]
                vW1 = beta2 * vW1 + (1 - beta2) * (grads["dW1"]**2)
                m_hat = mW1 / (1 - beta1**ep)
                v_hat = vW1 / (1 - beta2**ep)
                mlp.W1 -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # b1
                mb1 = beta1 * mb1 + (1 - beta1) * grads["db1"]
                vb1 = beta2 * vb1 + (1 - beta2) * (grads["db1"]**2)
                m_hat = mb1 / (1 - beta1**ep)
                v_hat = vb1 / (1 - beta2**ep)
                mlp.b1 -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # W2
                mW2 = beta1 * mW2 + (1 - beta1) * grads["dW2"]
                vW2 = beta2 * vW2 + (1 - beta2) * (grads["dW2"]**2)
                m_hat = mW2 / (1 - beta1**ep)
                v_hat = vW2 / (1 - beta2**ep)
                mlp.W2 -= lr * m_hat / (np.sqrt(v_hat) + eps)

                # b2
                mb2 = beta1 * mb2 + (1 - beta1) * grads["db2"]
                vb2 = beta2 * vb2 + (1 - beta2) * (grads["db2"]**2)
                m_hat = mb2 / (1 - beta1**ep)
                v_hat = vb2 / (1 - beta2**ep)
                mlp.b2 -= lr * m_hat / (np.sqrt(v_hat) + eps)

        mlp_losses[opt_name] = losses
        print(f"  MLP Trained with {opt_name:8s} | Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.5f}")

    # Plot MLP Training Curves & Fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    for opt_name, l_hist in mlp_losses.items():
        ax1.plot(l_hist, label=opt_name, linewidth=2)
    ax1.set_title("MLP Training Loss Curves: Backpropagation Engine", fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle=':')

    # Fitted model output with Adam
    y_fit = mlp.forward(X)
    ax2.scatter(X, y, color='gray', alpha=0.5, label="Noisy Data")
    ax2.plot(X, y_fit, 'r-', linewidth=2.5, label="Adam MLP Fit")
    ax2.set_title("Non-Linear Function Approximation", fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("y")
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle=':')

    plt.tight_layout()
    plt.savefig("plots/track_a_mlp_training.png", dpi=200)
    plt.close()

    return mlp_losses


def run_curvature_and_noise_sensitivity():
    print("\n--- Running Track A.5: Curvature & Noise Sensitivity Analysis ---")
    x_test = np.array([1.5, -0.8])
    v_test = np.array([0.6, 0.8])

    # 1. Hessian-Vector Product Verification on Beale / Quadratic
    exact_H = ObjectiveLandscapes.hess_quadratic(x_test)
    exact_hvp = exact_H @ v_test
    approx_hvp = compute_hvp(ObjectiveLandscapes.grad_quadratic, x_test, v_test)
    hvp_err = float(np.linalg.norm(exact_hvp - approx_hvp))
    print(f"  Exact HVP: {exact_hvp} | Approx HVP (Pearlmutter): {approx_hvp} | Abs Error: {hvp_err:.2e}")

    # 2. Noise Sensitivity across SGD, Momentum, Adam on Quadratic
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    final_losses = {"GD": [], "Momentum": [], "Adam": []}
    fn = ObjectiveLandscapes.ill_conditioned_quadratic
    g_fn = ObjectiveLandscapes.grad_quadratic
    x0 = np.array([-3.0, 3.0])

    for noise in noise_levels:
        res_gd = run_gradient_descent(x0, g_fn, fn, lr=0.045, n_steps=100, noise_std=noise)
        res_mom = run_momentum(x0, g_fn, fn, lr=0.045, beta=0.85, n_steps=100, noise_std=noise)
        res_adm = run_adam(x0, g_fn, fn, lr=0.35, n_steps=100, noise_std=noise)

        final_losses["GD"].append(res_gd["losses"][-1])
        final_losses["Momentum"].append(res_mom["losses"][-1])
        final_losses["Adam"].append(res_adm["losses"][-1])

    # Plot Noise Sensitivity
    plt.figure(figsize=(7, 4.5))
    for name, f_vals in final_losses.items():
        plt.plot(noise_levels, f_vals, 'o-', label=name, linewidth=2, markersize=7)
    plt.title("Robustness to Gradient Noise (sigma)", fontweight='bold')
    plt.xlabel("Gradient Noise Standard Deviation (sigma)")
    plt.ylabel("Final Loss after 100 Steps")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig("plots/track_a_noise_sensitivity.png", dpi=200)
    plt.close()

    # Save summary CSV
    df = pd.DataFrame({
        "Noise_Std": noise_levels,
        "GD_Final_Loss": final_losses["GD"],
        "Momentum_Final_Loss": final_losses["Momentum"],
        "Adam_Final_Loss": final_losses["Adam"]
    })
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/track_a_continuous_metrics.csv", index=False)
    print("  Saved Track A metrics to results/track_a_continuous_metrics.csv")


def run_all_track_a():
    print("================================================================================")
    print("EXECUTING TRACK A: CONTINUOUS OPTIMIZATION BENCHMARK")
    print("================================================================================")
    t0 = time.time()
    run_convex_quadratic_experiments()
    run_nonconvex_saddle_experiments()
    run_gauss_newton_experiment()
    run_backprop_mlp_experiment()
    run_curvature_and_noise_sensitivity()
    print(f"\n[DONE] Track A Completed in {time.time() - t0:.2f} seconds!")


if __name__ == "__main__":
    run_all_track_a()
