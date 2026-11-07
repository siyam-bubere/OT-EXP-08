# Track H: DBMS Optimization - Prompt Log

## Overview
- **Track**: Track H (Chapter 6 Database Management System Optimization)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. Selinger Dynamic Programming Join-Order Enumeration over relational catalog with cardinality and selectivity estimations.
  2. Automated Physical Index Selection formulated as a 0/1 Knapsack optimization problem under storage capacity constraints ($B \le 2.5$ MB).
  3. SQLite database generation (15,000+ records) and empirical workload benchmarking comparing full table scans vs index-assisted execution with `EXPLAIN QUERY PLAN` verification.

---

## Prompt 1: Selinger DP Join-Order Enumerator
```text
Role: Database Engine Architect & Query Optimizer Specialist
Context: System R / Selinger Dynamic Programming Join Enumeration
Task:
1. Model relational tables: Customers (2,000 rows), Orders (5,000 rows), LineItem (12,000 rows), Products (500 rows).
2. Define predicate selectivities:
   - FK(Customers, Orders) = 1/2000
   - FK(LineItem, Orders) = 1/5000
   - FK(LineItem, Products) = 1/500
3. Implement the bottom-up Selinger DP algorithm:
   - Base case: cost(R_i) = 0, card(R_i) = |R_i|
   - Recurrence: for each subset S of size 2..N, find min-cost bipartition (S1, S2)
4. Compare DP optimal join tree against naive cross-product order.
```

### AI Response Summary
- Produced `JoinOptimizerDP` evaluating left-deep and bushy join permutations.
- Found optimal plan: `(Products JOIN ((Customers JOIN Orders) JOIN LineItem))` with cost 3,807,500 units, achieving a 99.37% cost reduction compared to naive unoptimized cross-product order (600,050,000 units).

---

## Prompt 2: 0/1 Knapsack Automated Index Selection & SQLite Benchmarks
```text
Role: Storage Engine & DBA Optimizer
Context: Physical Index Selection under Storage Budget B
Task:
1. Define candidate secondary B-tree indexes with storage costs (MB) and expected workload latency gains (ms):
   - idx_orders_cust (0.8 MB, 18 ms)
   - idx_orders_status (0.5 MB, 8 ms)
   - idx_lineitem_order (1.4 MB, 32 ms)
   - idx_lineitem_prod (1.2 MB, 24 ms)
   - idx_cust_nation (0.3 MB, 7 ms)
   - idx_prod_cat (0.2 MB, 5 ms)
2. Solve 0/1 Knapsack optimization for budget B = 2.5 MB.
3. Instantiate a real SQLite database with 15,000+ records and benchmark multi-table analytics queries.
4. Verify EXPLAIN QUERY PLAN changes and measure physical wall-clock speedup.
```

### AI Response Summary
- Implemented `solve_index_selection_knapsack`: optimal solution packed `idx_cust_nation` (0.3 MB), `idx_orders_cust` (0.8 MB), and `idx_lineitem_order` (1.4 MB), utilizing exactly 2.5 MB (100% budget efficiency) for 57.0 ms expected gain.
- In actual SQLite benchmarks, query latency dropped from 18.14 ms to 6.12 ms (2.96x speedup, 66.2% latency reduction).
- Query plan shifted from unindexed `SCAN` with temporary sorting B-trees to indexed pointer lookups (`SEARCH ... USING INDEX`).
