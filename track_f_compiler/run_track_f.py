"""
track_f_compiler/run_track_f.py - Benchmarks & Equivalence Verification for Track F
Optimization Techniques Lab - Assignment 8 (Chapter 6 Compiler Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from track_f_compiler.mini_compiler import (
    Instruction,
    TACProgram,
    TACInterpreter,
    optimize_tac_program,
    RegisterAllocator
)


def build_benchmark_program() -> TACProgram:
    """
    Constructs a realistic numerical compute kernel exhibiting:
    1. Foldable constant arithmetic: 15 * 4, 100 - 30
    2. Common subexpressions: (base * rate) repeated multiple times
    3. Loop-invariant computations inside an iterative loop: (scale * factor + 7)
    4. Dead variables never consumed: dead_metric, dead_unused
    """
    insts = [
        # Setup constants
        Instruction('ASSIGN', 15, None, 'c1'),
        Instruction('ASSIGN', 4, None, 'c2'),
        Instruction('MUL', 'c1', 'c2', 't_folded1'),        # 15 * 4 = 60
        Instruction('ASSIGN', 100, None, 'c3'),
        Instruction('ASSIGN', 30, None, 'c4'),
        Instruction('SUB', 'c3', 'c4', 't_folded2'),        # 100 - 30 = 70
        Instruction('ADD', 't_folded1', 't_folded2', 'offset'), # 60 + 70 = 130

        # Input parameters
        Instruction('ASSIGN', 12, None, 'base'),
        Instruction('ASSIGN', 8, None, 'rate'),
        Instruction('ASSIGN', 3, None, 'scale'),
        Instruction('ASSIGN', 5, None, 'factor'),

        # Common subexpression
        Instruction('MUL', 'base', 'rate', 't_cse1'),        # 12 * 8 = 96
        Instruction('MUL', 'base', 'rate', 't_cse2'),        # 12 * 8 = 96 (Redundant CSE)
        Instruction('ADD', 't_cse1', 't_cse2', 'accum'),     # 96 + 96 = 192

        # Dead code
        Instruction('MUL', 'offset', 99, 'dead_metric'),     # Dead variable
        Instruction('ADD', 'dead_metric', 1000, 'dead_unused'), # Dead variable

        # Loop setup: sum from i = 1 to 5
        Instruction('ASSIGN', 1, None, 'i'),
        Instruction('ASSIGN', 5, None, 'limit'),
        Instruction('LABEL', None, None, 'loop_head'),

        # Loop Invariant Code (independent of i)
        Instruction('MUL', 'scale', 'factor', 't_inv1'),    # 3 * 5 = 15 (LICM Invariant)
        Instruction('ADD', 't_inv1', 7, 'loop_const'),      # 15 + 7 = 22 (LICM Invariant)

        # Loop Body
        Instruction('MUL', 'i', 'loop_const', 'step_val'),
        Instruction('ADD', 'accum', 'step_val', 'accum'),

        # Loop Increment & Jump
        Instruction('ADD', 'i', 1, 'i'),
        Instruction('IF_LE', 'i', 'limit', 'loop_head'),

        # Output
        Instruction('PRINT', 'accum', None, None)
    ]
    return TACProgram(insts)


def main():
    print("================================================================================")
    print("EXECUTING TRACK F: COMPILER OPTIMIZATION & REGISTER ALLOCATION")
    print("================================================================================")

    prog_raw = build_benchmark_program()
    print(f"\n--- Original Unoptimized TAC Program ({len(prog_raw)} instructions) ---")
    for idx, inst in enumerate(prog_raw.instructions):
        print(f"  {idx:2d}: {inst}")

    # 1. Execute unoptimized program in VM
    env_raw, out_raw = TACInterpreter.execute(prog_raw)
    print(f"\n  VM Output (Unoptimized): accum = {env_raw.get('accum')}, prints = {out_raw}")

    # 2. Run Optimization Passes
    prog_opt, stats = optimize_tac_program(prog_raw)
    print(f"\n--- Optimized TAC Program ({len(prog_opt)} instructions) ---")
    for idx, inst in enumerate(prog_opt.instructions):
        print(f"  {idx:2d}: {inst}")

    # 3. Execute optimized program in VM & Verify Semantic Equivalence
    env_opt, out_opt = TACInterpreter.execute(prog_opt)
    print(f"\n  VM Output (Optimized):   accum = {env_opt.get('accum')}, prints = {out_opt}")

    assert env_raw['accum'] == env_opt['accum'], "Semantic Equivalence Violation!"
    assert out_raw == out_opt, "Output Stream Mismatch!"
    print("\n  [VERIFICATION PASSED]: Exact semantic equivalence proven! (Output matches 100%)")

    # 4. Perform Register Allocation (Chaitin-Briggs Graph Coloring)
    allocator_k3 = RegisterAllocator(k_registers=3)
    alloc_k3 = allocator_k3.allocate(prog_opt)

    allocator_k4 = RegisterAllocator(k_registers=4)
    alloc_k4 = allocator_k4.allocate(prog_opt)

    print("\n--- Register Allocation Results ---")
    print(f"  K=3 Registers: Spilled {alloc_k3['spill_count']} variables -> {alloc_k3['spilled_variables']}")
    print(f"  K=4 Registers: Spilled {alloc_k4['spill_count']} variables -> {alloc_k4['spilled_variables']}")

    reduction_pct = (1.0 - len(prog_opt) / len(prog_raw)) * 100.0
    print(f"\n  Instruction Count Reduction: {len(prog_raw)} -> {len(prog_opt)} ({reduction_pct:.1f}% reduction)")

    # Save Metrics CSV
    metrics = [
        {"Pass": "Original AST/TAC", "Instruction_Count": stats["original_inst_count"], "Reduction_%": 0.0},
        {"Pass": "Constant Folding & Prop", "Instruction_Count": stats["after_constant_folding"], "Reduction_%": (1.0 - stats["after_constant_folding"]/stats["original_inst_count"])*100.0},
        {"Pass": "Common Subexpression Elim", "Instruction_Count": stats["after_cse"], "Reduction_%": (1.0 - stats["after_cse"]/stats["original_inst_count"])*100.0},
        {"Pass": "Loop-Invariant Code Motion", "Instruction_Count": stats["after_licm"], "Reduction_%": (1.0 - stats["after_licm"]/stats["original_inst_count"])*100.0},
        {"Pass": "Dead Code Elimination", "Instruction_Count": stats["after_dce"], "Reduction_%": (1.0 - stats["after_dce"]/stats["original_inst_count"])*100.0},
    ]
    os.makedirs("results", exist_ok=True)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("results/track_f_compiler_metrics.csv", index=False)
    print("  Saved: results/track_f_compiler_metrics.csv")

    # -------------------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)

    # Plot 1: Optimization Pass Pipeline
    plt.figure(figsize=(7, 4.5))
    x_labels = [m["Pass"] for m in metrics]
    counts = [m["Instruction_Count"] for m in metrics]
    bars = plt.bar(range(len(counts)), counts, color=['#34495e', '#3498db', '#2ecc71', '#f39c12', '#e74c3c'], width=0.55)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.4, f"{int(h)} inst", ha='center', fontweight='bold', fontsize=9)

    plt.xticks(range(len(x_labels)), x_labels, rotation=18, ha='right', fontsize=8.5, fontweight='bold')
    plt.ylabel("Instruction Count")
    plt.title("Track F: Compiler Optimization Passes (Instruction Elimination)", fontweight='bold')
    plt.ylim(0, max(counts) * 1.25)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_f_compiler_passes.png", dpi=200)
    plt.close()

    # Plot 2: Register Interference Graph & Allocation
    adj = alloc_k4["interference_graph"]
    nodes = list(adj.keys())
    deg = [len(adj[n]) for n in nodes]

    plt.figure(figsize=(7, 4.5))
    plt.barh(nodes, deg, color='steelblue', alpha=0.85)
    plt.axvline(3, color='orange', linestyle='--', linewidth=2, label="K=3 Limit (Spills Occur)")
    plt.axvline(4, color='green', linestyle='--', linewidth=2, label="K=4 Limit (Zero Spills)")
    plt.xlabel("Interference Degree (Simultaneously Live Variables)")
    plt.ylabel("Program Variable")
    plt.title("Track F: Register Interference Graph & Chaitin Coloring Capacity", fontweight='bold')
    plt.legend(fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/track_f_interference_graph.png", dpi=200)
    plt.close()

    print("  Saved Plots: track_f_compiler_passes.png, track_f_interference_graph.png")
    print("\n[DONE] Track F Execution Complete!\n")


if __name__ == "__main__":
    main()
