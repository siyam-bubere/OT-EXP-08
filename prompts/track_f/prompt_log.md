# Track F: Compiler Optimization & Register Allocation - Prompt Log

## Overview
- **Track**: Track F (Chapter 6 Compiler Optimization & Register Allocation)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. Three-Address Code (TAC) Intermediate Representation with basic block structures and virtual machine interpreter.
  2. Four optimization passes: Constant Folding/Propagation, Common Subexpression Elimination (CSE), Loop-Invariant Code Motion (LICM), and Dead Code Elimination (DCE).
  3. Register Allocation via Chaitin-Briggs graph coloring with Kempe simplification heuristic and register spill handling ($K=3, 4$).
  4. Exact semantic equivalence verification between unoptimized and optimized AST/TAC bytecode.

---

## Prompt 1: IR Representation & Optimization Passes Architecture
```text
Role: Senior Compiler Engineer & Static Analysis Specialist
Context: Chapter 6 Compiler Intermediate Representation & Optimization Passes
Task:
1. Define a Three-Address Code (TAC) data structure representing program statements:
   (ASSIGN, ADD, SUB, MUL, DIV, PRINT, LABEL, JUMP, IF_LE).
2. Implement four fundamental optimization passes:
   - Constant Folding & Constant Propagation: Evaluate compile-time constant arithmetic.
   - Common Subexpression Elimination (CSE): Identify duplicate subexpressions within basic blocks.
   - Loop-Invariant Code Motion (LICM): Hoist loop-invariant calculations outside the loop header.
   - Dead Code Elimination (DCE): Remove unused variable definitions.
3. Build a TAC virtual machine / interpreter to verify exact execution outputs before and after optimizations.
```

### AI Response Summary
- Created `Instruction`, `TACProgram`, and `TACInterpreter`.
- Provided pass functions for constant folding, CSE, LICM, and DCE.
- Constructed a test numerical kernel exhibiting constants, CSE, dead code, and a loop with invariants.

---

## Prompt 2: Debugging Semantic Equivalence Failure (AI Failure Case 4)
```text
Role: Compiler Verifier & Bug Triage
Bug Report:
When executing run_track_f.py, the interpreter produced:
  Unoptimized Output: accum = 522
  Optimized Output:   accum = 214
  AssertionError: Semantic Equivalence Violation!
Investigate the cause of this miscompilation. Identify which optimization pass corrupted the program semantics and provide a robust fix.
```

### AI Diagnosis & Root Cause Analysis (AI Failure Case 4)
- **Root Cause**: In the initial AI-generated `pass_constant_folding_and_propagation`, the constant propagation dictionary `constants` was maintained globally across instruction iterations without clearing at basic block boundaries (`LABEL`, `JUMP`, `IF_LE`).
- **Mechanism of Failure**: Prior to the loop, the variable `i` was initialized to `1` (`constants['i'] = 1`). When the loop header `loop_head:` was traversed, the compiler retained `i = 1` as a global invariant. As a result, inside the loop body, `step_val = i * loop_const` was falsely folded to `1 * 22 = 22`, and `i = i + 1` was folded to `2`. When LICM subsequently inspected the loop, it assumed these values were constants and hoisted them outside the loop, corrupting the loop back-edge and terminating the loop prematurely after a single iteration (`accum = 214` instead of `522`).
- **Fix Applied**: Updated `pass_constant_folding_and_propagation` to strictly reset the active constant table (`constants.clear()`) at any basic block delimiter (`LABEL`, `JUMP`, `IF_LE`), enforcing local basic block scoping for constant propagation. After the fix, the optimized program produced `accum = 522`, achieving 100% semantic equivalence.

---

## Prompt 3: Register Allocation via Graph Coloring
```text
Role: Code Generation & Register Allocation Specialist
Context: Chaitin-Briggs Graph Coloring with Kempe Heuristic
Task:
1. Implement backward dataflow live variable analysis to compute variable liveness at every instruction.
2. Construct the undirected Interference Graph: add an edge between two variables if they are simultaneously live.
3. Implement Chaitin-Briggs K-register coloring:
   - Simplify stage: push nodes with degree < K onto the stack.
   - Spill stage: if all nodes have degree >= K, select highest-degree node to spill to stack memory.
   - Select stage: pop and assign physical registers R0..R(K-1).
4. Evaluate allocation on K=3 and K=4 registers.
```

### AI Response Summary
- Generated `RegisterAllocator` with complete liveness analysis and interference graph construction.
- For $K=3$, Chaitin-Briggs identified degree $\ge 3$ bottlenecks and spilled 2 variables (`limit`, `accum`).
- For $K=4$, interference bottlenecks were resolved with only 1 spill (`accum`), verifying that chromatic number bounds dictate register pressure in loop-heavy workloads.
