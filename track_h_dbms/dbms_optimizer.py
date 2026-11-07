"""
track_h_dbms/dbms_optimizer.py
Track H: DBMS Query & Storage Optimization
Optimization Techniques Lab - Assignment 8 (Chapter 6 DBMS Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Components:
1. Dynamic Programming Join-Order Enumerator (Selinger Algorithm):
   - Computes optimal left-deep & bushy join order across N-way relational joins
   - Uses relational catalog cardinalities & join predicate selectivity
2. Automated Index Selection via 0/1 Knapsack / ILP Optimization:
   - Formulates physical index selection under storage budget B (MB)
   - Evaluates query workload benefit matrix W_{q, i}
3. SQLite Schema & Synthetic TPC-H Benchmark Engine:
   - Tables: Customers, Orders, LineItem, Products (15,000+ rows)
   - Real EXPLAIN QUERY PLAN analysis & wall-clock execution time benchmarking
"""

import sqlite3
import numpy as np
import time
from typing import List, Dict, Tuple, Set, Optional


# ---------------------------------------------------------------------------
# 1. Selinger Dynamic Programming Join-Order Optimizer
# ---------------------------------------------------------------------------

class TableCatalog:
    def __init__(self, name: str, cardinality: int):
        self.name = name
        self.cardinality = cardinality


class JoinOptimizerDP:
    """
    Selinger DP Join Enumeration algorithm:
    Solves optimal join tree for relations {R1, R2, ..., Rn}.
    Subproblem: BestPlan(S) for subset S subset of {R1..Rn}.
    Recurrence: BestPlan(S) = min_{S1 subset S, S2 = S \ S1} [ Cost(S1) + Cost(S2) + JoinCost(S1, S2) ]
    """
    def __init__(self, tables: Dict[str, TableCatalog], selectivities: Dict[Tuple[str, str], float]):
        self.tables = tables
        self.selectivities = selectivities

    def get_selectivity(self, t1: str, t2: str) -> float:
        pair = (min(t1, t2), max(t1, t2))
        return self.selectivities.get(pair, 1.0)  # default cross-product if no predicate

    def solve(self, relation_names: List[str]) -> Tuple[Dict, List[str]]:
        memo: Dict[frozenset, Tuple[float, int, str]] = {}  # subset -> (min_cost, resulting_cardinality, plan_str)

        # Base cases: single tables
        for name in relation_names:
            card = self.tables[name].cardinality
            memo[frozenset([name])] = (0.0, card, name)

        n = len(relation_names)

        # Iterative subset size from 2 to n
        for size in range(2, n + 1):
            from itertools import combinations
            for subset in combinations(relation_names, size):
                subset_set = frozenset(subset)
                best_cost = float('inf')
                best_card = 0
                best_plan = ""

                # Enumerate bipartitions (S1, S2)
                subset_list = list(subset)
                for k in range(1, size):
                    for sub1 in combinations(subset_list, k):
                        s1 = frozenset(sub1)
                        s2 = subset_set - s1
                        if s1 not in memo or s2 not in memo:
                            continue

                        cost1, card1, plan1 = memo[s1]
                        cost2, card2, plan2 = memo[s2]

                        # Calculate joined selectivity
                        sel = 1.0
                        for t1 in s1:
                            for t2 in s2:
                                sel *= self.get_selectivity(t1, t2)

                        out_card = int(max(1, card1 * card2 * sel))
                        # Cost model: Block Nested Loop / Hash Join: Cost = Cost1 + Cost2 + JoinProcessingCost
                        join_cost = cost1 + cost2 + (card1 * 1.0 + card1 * card2 * 0.05)

                        if join_cost < best_cost:
                            best_cost = join_cost
                            best_card = out_card
                            best_plan = f"({plan1} JOIN {plan2})"

                memo[subset_set] = (best_cost, best_card, best_plan)

        all_rel = frozenset(relation_names)
        return memo[all_rel]


# ---------------------------------------------------------------------------
# 2. Automated Index Selection (0/1 Knapsack / ILP)
# ---------------------------------------------------------------------------

class CandidateIndex:
    def __init__(self, index_name: str, table: str, column: str, storage_mb: float, latency_gain_ms: float):
        self.index_name = index_name
        self.table = table
        self.column = column
        self.storage_mb = storage_mb
        self.latency_gain_ms = latency_gain_ms


def solve_index_selection_knapsack(candidates: List[CandidateIndex], budget_mb: float) -> Tuple[List[CandidateIndex], float, float]:
    """
    Solves 0/1 Knapsack problem for physical index selection under storage budget.
    Maximizes total query latency reduction (benefit).
    """
    scale = 10  # 0.1 MB discretization
    int_budget = int(budget_mb * scale)
    n = len(candidates)

    dp = [[0.0 for _ in range(int_budget + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        item = candidates[i - 1]
        w = int(item.storage_mb * scale)
        v = item.latency_gain_ms
        for b in range(int_budget + 1):
            if w <= b:
                dp[i][b] = max(dp[i - 1][b], dp[i - 1][b - w] + v)
            else:
                dp[i][b] = dp[i - 1][b]

    # Backtrack chosen items
    chosen = []
    b = int_budget
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            item = candidates[i - 1]
            chosen.append(item)
            b -= int(item.storage_mb * scale)

    total_storage = sum(item.storage_mb for item in chosen)
    total_gain = sum(item.latency_gain_ms for item in chosen)
    return chosen, total_storage, total_gain


# ---------------------------------------------------------------------------
# 3. SQLite Database Engine & Workload Benchmarking
# ---------------------------------------------------------------------------

def create_sqlite_benchmark_db(db_path: str = "benchmark.db", seed: int = 42):
    """Generates synthetic relational database with 15,000+ records."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS lineitem")
    cur.execute("DROP TABLE IF EXISTS orders")
    cur.execute("DROP TABLE IF EXISTS customers")
    cur.execute("DROP TABLE IF EXISTS products")

    cur.execute("""
        CREATE TABLE customers (
            cust_id INTEGER PRIMARY KEY,
            name TEXT,
            nation_id INTEGER,
            acctbal REAL
        )
    """)

    cur.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            cust_id INTEGER,
            order_date TEXT,
            total_price REAL,
            status TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE products (
            prod_id INTEGER PRIMARY KEY,
            name TEXT,
            category TEXT,
            price REAL
        )
    """)

    cur.execute("""
        CREATE TABLE lineitem (
            line_id INTEGER PRIMARY KEY,
            order_id INTEGER,
            prod_id INTEGER,
            quantity INTEGER,
            discount REAL
        )
    """)

    # Populate Customers (2,000)
    custs = [(i, f"Customer_{i}", int(rng.integers(1, 25)), float(rng.uniform(100, 50000))) for i in range(1, 2001)]
    cur.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", custs)

    # Populate Products (500)
    prods = [(i, f"Product_{i}", f"Cat_{i % 10}", float(rng.uniform(10, 1500))) for i in range(1, 501)]
    cur.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", prods)

    # Populate Orders (5,000)
    orders = [(i, int(rng.integers(1, 2001)), f"2026-0{int(rng.integers(1, 9))}-15", float(rng.uniform(50, 8000)), rng.choice(['O', 'F', 'P'])) for i in range(1, 5001)]
    cur.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    # Populate LineItems (12,000)
    lines = [(i, int(rng.integers(1, 5001)), int(rng.integers(1, 501)), int(rng.integers(1, 50)), float(rng.uniform(0.0, 0.25))) for i in range(1, 12001)]
    cur.executemany("INSERT INTO lineitem VALUES (?, ?, ?, ?, ?)", lines)

    conn.commit()
    conn.close()


def benchmark_query_workload(db_path: str, indexes_to_create: List[str] = None) -> Tuple[float, List[str]]:
    """Runs a multi-table analytics workload and returns (total_latency_ms, explain_plans)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if indexes_to_create:
        for idx_sql in indexes_to_create:
            cur.execute(idx_sql)
        conn.commit()

    queries = [
        # Q1: Customer Order aggregation
        """
        SELECT c.cust_id, c.name, COUNT(o.order_id), SUM(o.total_price)
        FROM customers c JOIN orders o ON c.cust_id = o.cust_id
        WHERE o.status = 'F' AND c.nation_id = 5
        GROUP BY c.cust_id, c.name
        """,
        # Q2: Order Item lookup join
        """
        SELECT l.order_id, p.name, l.quantity, p.price * l.quantity
        FROM lineitem l JOIN products p ON l.prod_id = p.prod_id
        WHERE l.quantity > 35 AND p.category = 'Cat_3'
        """,
        # Q3: 3-way join analytics
        """
        SELECT c.name, o.order_date, SUM(l.quantity)
        FROM customers c
        JOIN orders o ON c.cust_id = o.cust_id
        JOIN lineitem l ON o.order_id = l.order_id
        WHERE o.total_price > 5000
        GROUP BY c.cust_id, o.order_id
        LIMIT 100
        """
    ]

    plans = []
    for q in queries:
        cur.execute(f"EXPLAIN QUERY PLAN {q}")
        plan_rows = cur.fetchall()
        plans.append(" -> ".join([r[3] for r in plan_rows]))

    # Benchmark runtime (5 runs)
    runtimes = []
    for _ in range(5):
        t0 = time.time()
        for q in queries:
            cur.execute(q)
            _ = cur.fetchall()
        runtimes.append((time.time() - t0) * 1000.0)

    conn.close()
    return float(np.mean(runtimes)), plans
