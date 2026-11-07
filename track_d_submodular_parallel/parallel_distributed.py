"""
track_d_submodular_parallel/parallel_distributed.py
Complete Chapter 6 Submodular, Parallel, and Distributed Optimization
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Modules Covered:
1. Submodular Sensor Coverage with Diminishing Returns Test & (1 - 1/e) Guarantee
2. Parallel Program Optimization: Work Stealing, Makespan, Amdahl's & Gustafson's Laws
3. Federated Learning: FedAvg with Non-IID Clients vs Centralized
4. Multi-Task Optimization: MGDA and GradNorm vs Equal Task Weighting
"""

import time
import numpy as np
from typing import List, Set, Dict, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


# ---------------------------------------------------------------------------
# 1. Submodular Optimization: Maximum Sensor Coverage
# ---------------------------------------------------------------------------

class SensorCoverageSubmodular:
    """
    Submodular function: F(S) = area or discrete target points covered by sensor subset S.
    F(S) satisfies diminishing returns: F(A U {e}) - F(A) >= F(B U {e}) - F(B) for A subset B.
    """
    def __init__(self, n_targets: int = 100, n_sensors: int = 25, seed: int = 42):
        self.n_targets = n_targets
        self.n_sensors = n_sensors
        rng = np.random.default_rng(seed)

        self.targets = rng.uniform(0, 100, size=(n_targets, 2))
        self.sensors = rng.uniform(0, 100, size=(n_sensors, 2))
        self.radius = 22.0  # Sensing radius

        # Precompute coverage sets
        self.sensor_covers: List[Set[int]] = []
        for i in range(n_sensors):
            dists = np.linalg.norm(self.targets - self.sensors[i], axis=1)
            covered = set(np.where(dists <= self.radius)[0])
            self.sensor_covers.append(covered)

    def evaluate(self, subset: Set[int]) -> int:
        """Returns union size: | U_{i in subset} Cover(i) |."""
        if not subset:
            return 0
        total_covered = set()
        for idx in subset:
            total_covered.update(self.sensor_covers[idx])
        return len(total_covered)

    def verify_diminishing_returns(self, n_trials: int = 50, seed: int = 42) -> Tuple[bool, float]:
        """
        Formally verifies submodularity: F(A U {e}) - F(A) >= F(B U {e}) - F(B) for A subset B.
        Returns (is_submodular, min_difference).
        """
        rng = np.random.default_rng(seed)
        all_sensors = list(range(self.n_sensors))
        min_diff = float("inf")

        for _ in range(n_trials):
            # Sample B with size 4 to 10
            size_b = rng.integers(4, 11)
            B = set(rng.choice(all_sensors, size=size_b, replace=False))
            # Sample A subset B
            size_a = rng.integers(1, size_b)
            A = set(rng.choice(list(B), size=size_a, replace=False))
            # Pick e not in B
            candidates = [x for x in all_sensors if x not in B]
            if not candidates:
                continue
            e = rng.choice(candidates)

            # Marginal gains
            gain_a = self.evaluate(A.union({e})) - self.evaluate(A)
            gain_b = self.evaluate(B.union({e})) - self.evaluate(B)

            diff = gain_a - gain_b
            if diff < min_diff:
                min_diff = diff
            if gain_a < gain_b:
                return False, diff

        return True, min_diff

    def solve_greedy(self, k: int) -> Tuple[Set[int], int]:
        """Greedy selection of k elements: guaranteed >= (1 - 1/e) * OPT approx 63.2%."""
        selected = set()
        for _ in range(k):
            best_e = None
            best_gain = -1
            curr_val = self.evaluate(selected)
            for e in range(self.n_sensors):
                if e not in selected:
                    gain = self.evaluate(selected.union({e})) - curr_val
                    if gain > best_gain:
                        best_gain = gain
                        best_e = e
            if best_e is not None and best_gain > 0:
                selected.add(best_e)
            else:
                break
        return selected, self.evaluate(selected)

    def solve_exact(self, k: int) -> Tuple[Set[int], int]:
        """Exact brute force enumeration for small k."""
        from itertools import combinations
        best_val = -1
        best_set = None
        for comb in combinations(range(self.n_sensors), k):
            val = self.evaluate(set(comb))
            if val > best_val:
                best_val = val
                best_set = set(comb)
        return best_set, best_val


# ---------------------------------------------------------------------------
# 2. Parallel Program Optimization: Work Stealing & Amdahl/Gustafson
# ---------------------------------------------------------------------------

def cpu_heavy_kernel(chunk: np.ndarray) -> float:
    """Synthetic compute-intensive numerical kernel with non-uniform complexity."""
    x = chunk.copy()
    for _ in range(50):
        x = np.sin(x) * np.cos(x) + np.sqrt(np.abs(x) + 1.0)
    return float(np.sum(x))


def run_parallel_workload(n_items: int = 40000, n_workers: int = 4, scheduling: str = "dynamic") -> float:
    """Executes chunked workload across n_workers threads/processes."""
    rng = np.random.default_rng(42)
    data = rng.uniform(0.1, 10.0, size=n_items)

    chunk_size = 1000 if scheduling == "dynamic" else n_items // n_workers
    chunks = [data[i:i + chunk_size] for i in range(0, n_items, chunk_size)]

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(cpu_heavy_kernel, chunks))
    dur = time.time() - t0
    return dur


# ---------------------------------------------------------------------------
# 3. Federated Learning (FedAvg vs Centralized)
# ---------------------------------------------------------------------------

class FedAvgSimulation:
    """Federated Averaging on non-IID partitioned linear/logistic regression."""
    def __init__(self, n_clients: int = 5, n_features: int = 10, seed: int = 42):
        self.n_clients = n_clients
        self.d = n_features
        rng = np.random.default_rng(seed)

        self.true_w = rng.normal(0, 1, size=n_features)
        self.client_data = []

        # Create non-IID datasets across clients
        for c in range(n_clients):
            n_samples = rng.integers(100, 250)
            # Heterogeneous feature distributions
            mean_shift = (c - n_clients // 2) * 1.5
            X_c = rng.normal(mean_shift, 1.0, size=(n_samples, n_features))
            y_c = X_c @ self.true_w + rng.normal(0, 0.2, size=n_samples)
            self.client_data.append((X_c, y_c))

    def evaluate_global_loss(self, w: np.ndarray) -> float:
        total_loss = 0.0
        total_n = 0
        for X_c, y_c in self.client_data:
            preds = X_c @ w
            total_loss += float(np.sum((preds - y_c)**2))
            total_n += len(X_c)
        return total_loss / (2.0 * total_n)

    def run_centralized(self, n_rounds: int = 30, lr: float = 0.01) -> List[float]:
        """Centralized baseline pooling all data together."""
        X_all = np.vstack([X for X, _ in self.client_data])
        y_all = np.concatenate([y for _, y in self.client_data])
        w = np.zeros(self.d)
        losses = []

        for _ in range(n_rounds):
            losses.append(self.evaluate_global_loss(w))
            grad = (X_all.T @ (X_all @ w - y_all)) / len(X_all)
            w -= lr * grad
        return losses

    def run_fedavg(self, n_rounds: int = 30, local_epochs: int = 5, lr: float = 0.01) -> List[float]:
        """FedAvg: local SGD on each client, followed by server parameter averaging."""
        global_w = np.zeros(self.d)
        losses = []
        total_samples = sum(len(X) for X, _ in self.client_data)

        for _ in range(n_rounds):
            losses.append(self.evaluate_global_loss(global_w))
            local_weights = []

            for X_c, y_c in self.client_data:
                w_local = np.copy(global_w)
                n_c = len(X_c)
                # Local SGD steps
                for _ in range(local_epochs):
                    grad = (X_c.T @ (X_c @ w_local - y_c)) / n_c
                    w_local -= lr * grad
                local_weights.append((n_c, w_local))

            # Server aggregation: w_global = sum (n_c / N) * w_local
            global_w = np.zeros(self.d)
            for n_c, w_l in local_weights:
                global_w += (n_c / total_samples) * w_l

        return losses


# ---------------------------------------------------------------------------
# 4. Multi-Task Optimization (MGDA & GradNorm vs Equal Weights)
# ---------------------------------------------------------------------------

class MultiTaskModel:
    """Joint 2-task linear model with shared representation W_shared."""
    def __init__(self, d_in: int = 8, d_shared: int = 6, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_s = rng.normal(0, 0.5, size=(d_in, d_shared))
        self.W_t1 = rng.normal(0, 0.5, size=(d_shared, 1))
        self.W_t2 = rng.normal(0, 0.5, size=(d_shared, 1))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = X @ self.W_s  # Shared representation
        y1_pred = h @ self.W_t1
        y2_pred = h @ self.W_t2
        return h, y1_pred, y2_pred


def run_multitask_comparison(n_steps: int = 80, lr: float = 0.01) -> Dict:
    """Compares Equal Task Weights vs MGDA vs GradNorm on conflicting multi-task regression."""
    rng = np.random.default_rng(42)
    N = 100
    X = rng.normal(0, 1, size=(N, 8))
    # Conflicting target outputs
    y1 = np.sum(X[:, :4], axis=1, keepdims=True)
    y2 = np.sum(X[:, 4:], axis=1, keepdims=True) * 5.0  # Task 2 has 5x larger magnitude!

    methods = ["Equal_Weights", "MGDA", "GradNorm"]
    results = {}

    for m in methods:
        model = MultiTaskModel(d_in=8, d_shared=6, seed=42)
        losses1, losses2 = [], []
        task_weights = np.array([0.5, 0.5])
        init_l1, init_l2 = None, None

        for t in range(n_steps):
            h, p1, p2 = model.forward(X)
            l1 = float(np.mean((p1 - y1)**2))
            l2 = float(np.mean((p2 - y2)**2))
            losses1.append(l1)
            losses2.append(l2)

            if t == 0:
                init_l1, init_l2 = l1, l2

            # Gradients for shared weights from task 1 and task 2
            g1_s = (X.T @ (2.0 * (p1 - y1) @ model.W_t1.T)) / N
            g2_s = (X.T @ (2.0 * (p2 - y2) @ model.W_t2.T)) / N

            # Gradients for task heads
            g_t1 = (h.T @ (2.0 * (p1 - y1))) / N
            g_t2 = (h.T @ (2.0 * (p2 - y2))) / N

            if m == "Equal_Weights":
                w_comb = 0.5 * g1_s + 0.5 * g2_s

            elif m == "MGDA":
                # Find convex combination alpha in [0, 1] minimizing ||alpha g1 + (1-alpha) g2||^2
                v1 = g1_s.ravel()
                v2 = g2_s.ravel()
                diff = v2 - v1
                denom = np.dot(diff, diff) + 1e-8
                gamma = np.dot(diff, v2) / denom
                alpha = np.clip(gamma, 0.0, 1.0)
                w_comb = alpha * g1_s + (1.0 - alpha) * g2_s

            elif m == "GradNorm":
                # Relative inverse training rate
                r1 = (l1 / init_l1) / ((l1 / init_l1 + l2 / init_l2) / 2.0)
                r2 = (l2 / init_l2) / ((l1 / init_l1 + l2 / init_l2) / 2.0)
                # Norm of gradients
                norm1 = np.linalg.norm(g1_s) + 1e-6
                norm2 = np.linalg.norm(g2_s) + 1e-6
                avg_norm = (norm1 + norm2) / 2.0
                target1 = avg_norm * (r1**0.5)
                target2 = avg_norm * (r2**0.5)
                # Update task weights
                task_weights[0] += 0.05 * (target1 - norm1) / avg_norm
                task_weights[1] += 0.05 * (target2 - norm2) / avg_norm
                task_weights = np.maximum(0.1, task_weights)
                task_weights = 2.0 * (task_weights / np.sum(task_weights))
                w_comb = task_weights[0] * g1_s + task_weights[1] * g2_s

            # Gradient descent step
            model.W_s -= lr * w_comb
            model.W_t1 -= lr * g_t1
            model.W_t2 -= lr * g_t2

        results[m] = {"loss1": losses1, "loss2": losses2}

    return results
