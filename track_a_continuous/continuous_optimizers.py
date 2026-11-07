"""
track_a_continuous/continuous_optimizers.py
Complete Chapter 6 Continuous Optimization Implementations from Scratch
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Algorithms Covered:
1. Gradient Descent (GD) with constant learning rate
2. Stochastic Gradient Descent (SGD)
3. Mini-batch SGD
4. Classical Momentum
5. Nesterov Accelerated Gradient (NAG)
6. AdaGrad
7. RMSProp
8. Adam (Adaptive Moment Estimation)
9. Newton-Raphson (Exact 2nd-order with regularized Hessian inversion)
10. Gauss-Newton (Non-linear least squares)
11. Backpropagation 2-Layer Neural Network
12. Curvature approximations: Empirical Fisher, Diagonal Hessian, Hessian-Vector Products (HVP)
"""

import numpy as np
from typing import Callable, Tuple, Dict, List, Optional


class ObjectiveLandscapes:
    """Benchmark test functions for convex and non-convex continuous optimization."""

    @staticmethod
    def ill_conditioned_quadratic(x: np.ndarray, A: Optional[np.ndarray] = None) -> float:
        """Strongly convex quadratic bowl with condition number kappa = 20."""
        if A is None:
            A = np.array([[20.0, 0.0], [0.0, 1.0]])
        return 0.5 * float(x.T @ A @ x)

    @staticmethod
    def grad_quadratic(x: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if A is None:
            A = np.array([[20.0, 0.0], [0.0, 1.0]])
        return A @ x

    @staticmethod
    def hess_quadratic(x: np.ndarray, A: Optional[np.ndarray] = None) -> np.ndarray:
        if A is None:
            A = np.array([[20.0, 0.0], [0.0, 1.0]])
        return A

    @staticmethod
    def non_convex_saddle(x: np.ndarray) -> float:
        """Non-convex landscape with a strict saddle point at (0, 0) and two symmetric minima at (0, +-sqrt(2))."""
        # f(x, y) = x^2 - y^2 + 0.25 * y^4
        return float(x[0]**2 - x[1]**2 + 0.25 * x[1]**4)

    @staticmethod
    def grad_saddle(x: np.ndarray) -> np.ndarray:
        # df/dx = 2x, df/dy = -2y + y^3
        return np.array([2.0 * x[0], -2.0 * x[1] + x[1]**3])

    @staticmethod
    def hess_saddle(x: np.ndarray) -> np.ndarray:
        return np.array([
            [2.0, 0.0],
            [0.0, -2.0 + 3.0 * x[1]**2]
        ])

    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Non-convex banana valley: f(x, y) = (1 - x)^2 + 100(y - x^2)^2."""
        return float((1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2)

    @staticmethod
    def grad_rosenbrock(x: np.ndarray) -> np.ndarray:
        df_dx = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0]**2)
        df_dy = 200.0 * (x[1] - x[0]**2)
        return np.array([df_dx, df_dy])

    @staticmethod
    def hess_rosenbrock(x: np.ndarray) -> np.ndarray:
        d2f_dx2 = 2.0 - 400.0 * x[1] + 1200.0 * x[0]**2
        d2f_dxdy = -400.0 * x[0]
        d2f_dy2 = 200.0
        return np.array([[d2f_dx2, d2f_dxdy], [d2f_dxdy, d2f_dy2]])


# ---------------------------------------------------------------------------
# Continuous Optimizers (Iterative Step & Full Trajectory Runners)
# ---------------------------------------------------------------------------

def run_gradient_descent(x0: np.ndarray, grad_fn: Callable, fn: Callable,
                         lr: float = 0.05, n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        g = grad_fn(x)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        x -= lr * g
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "Gradient Descent (GD)", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_momentum(x0: np.ndarray, grad_fn: Callable, fn: Callable,
                 lr: float = 0.05, beta: float = 0.9, n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    v = np.zeros_like(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        g = grad_fn(x)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        v = beta * v + lr * g
        x -= v
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "Momentum", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_nesterov(x0: np.ndarray, grad_fn: Callable, fn: Callable,
                 lr: float = 0.05, beta: float = 0.9, n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    v = np.zeros_like(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        # Look-ahead point
        x_ahead = x - beta * v
        g = grad_fn(x_ahead)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        v = beta * v + lr * g
        x -= v
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "NAG (Nesterov)", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_adagrad(x0: np.ndarray, grad_fn: Callable, fn: Callable,
                lr: float = 0.5, eps: float = 1e-8, n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    G = np.zeros_like(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        g = grad_fn(x)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        G += g**2
        x -= (lr / (np.sqrt(G) + eps)) * g
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "AdaGrad", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_rmsprop(x0: np.ndarray, grad_fn: Callable, fn: Callable,
                lr: float = 0.05, beta: float = 0.99, eps: float = 1e-8, n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    s = np.zeros_like(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        g = grad_fn(x)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        s = beta * s + (1.0 - beta) * (g**2)
        x -= (lr / (np.sqrt(s) + eps)) * g
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "RMSProp", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_adam(x0: np.ndarray, grad_fn: Callable, fn: Callable,
             lr: float = 0.05, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
             n_steps: int = 100, noise_std: float = 0.0, seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    x = np.copy(x0).astype(float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        if noise_std > 0:
            g += rng.normal(0, noise_std, size=g.shape)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g**2)
        # Bias correction
        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        x -= (lr / (np.sqrt(v_hat) + eps)) * m_hat
        traj.append(np.copy(x))
        losses.append(fn(x))

    return {"name": "Adam", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


def run_newton(x0: np.ndarray, grad_fn: Callable, hess_fn: Callable, fn: Callable,
               lr: float = 1.0, reg: float = 1e-4, n_steps: int = 50) -> Dict:
    """Exact Newton-Raphson with Levenberg-Marquardt regularization."""
    x = np.copy(x0).astype(float)
    dim = len(x)
    traj = [np.copy(x)]
    losses = [fn(x)]

    for _ in range(n_steps):
        g = grad_fn(x)
        H = hess_fn(x) + reg * np.eye(dim)
        try:
            step_dir = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step_dir = g
        x -= lr * step_dir
        traj.append(np.copy(x))
        losses.append(fn(x))
        if np.linalg.norm(g) < 1e-7:
            break

    return {"name": "Newton-Raphson", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


# ---------------------------------------------------------------------------
# Gauss-Newton on Non-Linear Least Squares
# ---------------------------------------------------------------------------
def run_gauss_newton(residual_fn: Callable, jacobian_fn: Callable, x0: np.ndarray,
                     lr: float = 1.0, reg: float = 1e-4, n_steps: int = 30) -> Dict:
    """
    Minimizes S(x) = 0.5 * || r(x) ||^2.
    Gauss-Newton approximation to Hessian: H_GN = J^T J.
    Step: (J^T J + reg * I) delta = -J^T r.
    """
    x = np.copy(x0).astype(float)
    traj = [np.copy(x)]
    losses = []

    for _ in range(n_steps):
        r = residual_fn(x)
        J = jacobian_fn(x)
        loss = 0.5 * float(np.sum(r**2))
        losses.append(loss)

        g = J.T @ r
        H_gn = J.T @ J + reg * np.eye(len(x))
        delta = np.linalg.solve(H_gn, -g)
        x += lr * delta
        traj.append(np.copy(x))

    losses.append(0.5 * float(np.sum(residual_fn(x)**2)))
    return {"name": "Gauss-Newton", "x_final": x, "traj": np.array(traj), "losses": np.array(losses)}


# ---------------------------------------------------------------------------
# Curvature Approximations (Fisher, Diagonal, Hessian-Vector Products)
# ---------------------------------------------------------------------------
def compute_hvp(grad_fn: Callable, x: np.ndarray, v: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Computes Hessian-Vector Product H @ v via finite differences of gradients."""
    g_plus = grad_fn(x + eps * v)
    g_minus = grad_fn(x - eps * v)
    return (g_plus - g_minus) / (2.0 * eps)


def compute_empirical_fisher(sample_grads: np.ndarray) -> np.ndarray:
    """
    Computes Empirical Fisher Information Matrix:
    F = 1/N sum_{i=1}^N (g_i g_i^T).
    sample_grads has shape (N_samples, n_params).
    """
    N, d = sample_grads.shape
    return (sample_grads.T @ sample_grads) / float(N)


# ---------------------------------------------------------------------------
# From-Scratch Backpropagation Engine (2-Layer Neural Network)
# ---------------------------------------------------------------------------
class MultiLayerPerceptron:
    """
    From-scratch 2-Layer Neural Network for non-linear regression/classification.
    Architecture: Input -> Linear(d_in, d_hidden) -> ReLU -> Linear(d_hidden, d_out) -> Loss.
    Decoupled backpropagation gradient engine.
    """
    def __init__(self, d_in: int, d_hidden: int, d_out: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # He initialization
        self.W1 = rng.normal(0, np.sqrt(2.0 / d_in), size=(d_in, d_hidden))
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / d_hidden), size=(d_hidden, d_out))
        self.b2 = np.zeros(d_out)

        # Cache for backprop
        self.cache = {}

    def forward(self, X: np.ndarray) -> np.ndarray:
        # X: (N, d_in)
        Z1 = X @ self.W1 + self.b1  # (N, d_hidden)
        A1 = np.maximum(0, Z1)      # ReLU
        Z2 = A1 @ self.W2 + self.b2 # (N, d_out)
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
        return Z2

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Mean Squared Error
        return float(np.mean((y_pred - y_true)**2))

    def backward(self, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        """Exact analytical backpropagation step (gradient computation)."""
        X = self.cache["X"]
        A1 = self.cache["A1"]
        Z1 = self.cache["Z1"]
        Z2 = self.cache["Z2"]
        N = len(X)

        # dLoss / dZ2
        dZ2 = 2.0 * (Z2 - y_true) / N  # (N, d_out)

        # Gradients for Layer 2
        dW2 = A1.T @ dZ2               # (d_hidden, d_out)
        db2 = np.sum(dZ2, axis=0)       # (d_out,)

        # Backprop through ReLU
        dA1 = dZ2 @ self.W2.T          # (N, d_hidden)
        dZ1 = dA1 * (Z1 > 0).astype(float) # (N, d_hidden)

        # Gradients for Layer 1
        dW1 = X.T @ dZ1                # (d_in, d_hidden)
        db1 = np.sum(dZ1, axis=0)       # (d_hidden,)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def get_params_vector(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(), self.W2.ravel(), self.b2.ravel()])

    def set_params_vector(self, p: np.ndarray):
        s1 = self.W1.size
        s2 = s1 + self.b1.size
        s3 = s2 + self.W2.size
        self.W1 = p[:s1].reshape(self.W1.shape)
        self.b1 = p[s1:s2].reshape(self.b1.shape)
        self.W2 = p[s2:s3].reshape(self.W2.shape)
        self.b2 = p[s3:].reshape(self.b2.shape)
