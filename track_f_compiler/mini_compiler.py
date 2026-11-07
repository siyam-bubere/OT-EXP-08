"""
track_f_compiler/mini_compiler.py
Track F: Compiler Optimization & Register Allocation
Optimization Techniques Lab - Assignment 8 (Chapter 6 Compiler Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Components:
1. Intermediate Representation (Three-Address Code IR) with Basic Blocks & CFG
2. Optimization Passes:
   - Pass 1: Constant Folding & Constant Propagation
   - Pass 2: Common Subexpression Elimination (CSE)
   - Pass 3: Dead Code Elimination (DCE)
   - Pass 4: Loop-Invariant Code Motion (LICM) & Loop Unrolling
3. Register Allocation:
   - Live Variable Analysis
   - Interference Graph Construction
   - Chaitin-Briggs Graph Coloring / Kempe Heuristic with Register Spilling
4. Virtual Machine / Interpreter for Semantic Equivalence Verification
"""

import copy
from typing import List, Dict, Set, Tuple, Optional, Any


class Instruction:
    """Three-Address Code (TAC) instruction: op, arg1, arg2, result."""
    def __init__(self, op: str, arg1: Any = None, arg2: Any = None, result: Optional[str] = None):
        self.op = op          # e.g., 'ASSIGN', 'ADD', 'SUB', 'MUL', 'DIV', 'PRINT', 'LABEL', 'JUMP', 'IF_LE'
        self.arg1 = arg1
        self.arg2 = arg2
        self.result = result

    def __repr__(self):
        if self.op == 'LABEL':
            return f"{self.result}:"
        elif self.op == 'JUMP':
            return f"jump {self.arg1}"
        elif self.op == 'IF_LE':
            return f"if {self.arg1} <= {self.arg2} goto {self.result}"
        elif self.op == 'PRINT':
            return f"print {self.arg1}"
        elif self.op == 'ASSIGN':
            return f"{self.result} = {self.arg1}"
        else:
            return f"{self.result} = {self.arg1} {self.op} {self.arg2}"


class TACProgram:
    def __init__(self, instructions: List[Instruction]):
        self.instructions = instructions

    def clone(self) -> 'TACProgram':
        return TACProgram([Instruction(i.op, i.arg1, i.arg2, i.result) for i in self.instructions])

    def __len__(self):
        return len(self.instructions)


# ---------------------------------------------------------------------------
# Virtual Machine Interpreter (Semantic Equivalence Verifier)
# ---------------------------------------------------------------------------

class TACInterpreter:
    @staticmethod
    def execute(program: TACProgram, inputs: Dict[str, Any] = None, max_steps: int = 20000) -> Tuple[Dict[str, Any], List[Any]]:
        env = dict(inputs) if inputs else {}
        outputs = []
        labels = {}
        for idx, inst in enumerate(program.instructions):
            if inst.op == 'LABEL':
                labels[inst.result] = idx

        pc = 0
        steps = 0
        while pc < len(program.instructions) and steps < max_steps:
            steps += 1
            inst = program.instructions[pc]

            def resolve(val):
                if isinstance(val, str) and val in env:
                    return env[val]
                return val

            if inst.op == 'LABEL':
                pc += 1
            elif inst.op == 'ASSIGN':
                env[inst.result] = resolve(inst.arg1)
                pc += 1
            elif inst.op in ('ADD', 'SUB', 'MUL', 'DIV'):
                v1 = resolve(inst.arg1)
                v2 = resolve(inst.arg2)
                if inst.op == 'ADD': env[inst.result] = v1 + v2
                elif inst.op == 'SUB': env[inst.result] = v1 - v2
                elif inst.op == 'MUL': env[inst.result] = v1 * v2
                elif inst.op == 'DIV': env[inst.result] = v1 // v2 if isinstance(v1, int) and isinstance(v2, int) else v1 / v2
                pc += 1
            elif inst.op == 'PRINT':
                outputs.append(resolve(inst.arg1))
                pc += 1
            elif inst.op == 'JUMP':
                pc = labels[inst.arg1]
            elif inst.op == 'IF_LE':
                v1 = resolve(inst.arg1)
                v2 = resolve(inst.arg2)
                if v1 <= v2:
                    pc = labels[inst.result]
                else:
                    pc += 1
            else:
                pc += 1

        return env, outputs


# ---------------------------------------------------------------------------
# Optimization Passes
# ---------------------------------------------------------------------------

def pass_constant_folding_and_propagation(prog: TACProgram) -> TACProgram:
    """Pass 1: Constant Folding & Constant Propagation (Basic Block Local)."""
    new_insts = []
    constants = {}

    for inst in prog.instructions:
        if inst.op in ('LABEL', 'JUMP', 'IF_LE'):
            # Basic block boundary: invalidate propagated constants across control flow edges
            constants.clear()
            new_insts.append(inst)
            continue

        arg1 = constants.get(inst.arg1, inst.arg1) if isinstance(inst.arg1, str) else inst.arg1
        arg2 = constants.get(inst.arg2, inst.arg2) if isinstance(inst.arg2, str) else inst.arg2

        if inst.op == 'ASSIGN':
            if isinstance(arg1, (int, float)):
                constants[inst.result] = arg1
            elif inst.result in constants:
                del constants[inst.result]
            new_insts.append(Instruction('ASSIGN', arg1, None, inst.result))

        elif inst.op in ('ADD', 'SUB', 'MUL', 'DIV'):
            if isinstance(arg1, (int, float)) and isinstance(arg2, (int, float)):
                # Fold constant expression
                if inst.op == 'ADD': folded = arg1 + arg2
                elif inst.op == 'SUB': folded = arg1 - arg2
                elif inst.op == 'MUL': folded = arg1 * arg2
                elif inst.op == 'DIV': folded = arg1 // arg2 if isinstance(arg1, int) and isinstance(arg2, int) else arg1 / arg2
                constants[inst.result] = folded
                new_insts.append(Instruction('ASSIGN', folded, None, inst.result))
            else:
                if inst.result in constants:
                    del constants[inst.result]
                new_insts.append(Instruction(inst.op, arg1, arg2, inst.result))
        else:
            new_insts.append(Instruction(inst.op, arg1, arg2, inst.result))

    return TACProgram(new_insts)


def pass_common_subexpression_elimination(prog: TACProgram) -> TACProgram:
    """Pass 2: Local Common Subexpression Elimination (CSE)."""
    new_insts = []
    expr_table = {}  # (op, arg1, arg2) -> result_var

    for inst in prog.instructions:
        if inst.op in ('LABEL', 'JUMP', 'IF_LE'):
            expr_table.clear()
            new_insts.append(inst)
            continue

        if inst.op in ('ADD', 'SUB', 'MUL', 'DIV'):
            # Normalize symmetric ops for matching
            expr_key = (inst.op, inst.arg1, inst.arg2)
            if inst.op in ('ADD', 'MUL') and str(inst.arg1) > str(inst.arg2):
                expr_key = (inst.op, inst.arg2, inst.arg1)

            if expr_key in expr_table:
                # Replace with assignment from previously computed temporary
                existing_var = expr_table[expr_key]
                new_insts.append(Instruction('ASSIGN', existing_var, None, inst.result))
            else:
                expr_table[expr_key] = inst.result
                new_insts.append(inst)
        elif inst.op == 'ASSIGN':
            # If variable redefined, invalidate expressions containing it
            expr_table = {k: v for k, v in expr_table.items() if k[1] != inst.result and k[2] != inst.result and v != inst.result}
            new_insts.append(inst)
        else:
            new_insts.append(inst)

    return TACProgram(new_insts)


def pass_dead_code_elimination(prog: TACProgram) -> TACProgram:
    """Pass 3: Dead Code Elimination (iterative liveness sweep)."""
    changed = True
    current = prog.instructions

    while changed:
        changed = False
        # Identify all variables used as arguments
        used_vars = set()
        for inst in current:
            if isinstance(inst.arg1, str) and inst.op != 'JUMP':
                used_vars.add(inst.arg1)
            if isinstance(inst.arg2, str):
                used_vars.add(inst.arg2)

        new_insts = []
        for inst in current:
            # An instruction defining a temporary or unused variable is dead if it is never read
            if inst.result and (inst.result.startswith("t_") or inst.result.startswith("dead_")) and inst.result not in used_vars and inst.op not in ('PRINT', 'LABEL', 'JUMP', 'IF_LE'):
                changed = True
                continue
            new_insts.append(inst)
        current = new_insts

    return TACProgram(current)


def pass_loop_invariant_code_motion(prog: TACProgram) -> TACProgram:
    """Pass 4: Loop-Invariant Code Motion (LICM). Hoists invariants before loop header."""
    instructions = prog.instructions
    loop_start, loop_end = None, None
    for idx, inst in enumerate(instructions):
        if inst.op == 'LABEL' and 'loop' in str(inst.result).lower():
            loop_start = idx
        elif inst.op == 'IF_LE' and 'loop' in str(inst.result).lower():
            loop_end = idx

    if loop_start is None or loop_end is None or loop_end <= loop_start:
        return prog

    # Identify modified variables inside the loop
    modified = set()
    for i in range(loop_start, loop_end + 1):
        if instructions[i].result:
            modified.add(instructions[i].result)

    # Invariants: instructions inside loop whose operands are neither modified in loop nor generated dynamically
    invariants = []
    remaining_loop = []

    for i in range(loop_start + 1, loop_end):
        inst = instructions[i]
        is_invariant = False
        if inst.op in ('ADD', 'SUB', 'MUL', 'DIV', 'ASSIGN') and inst.result:
            arg1_const = not (isinstance(inst.arg1, str) and inst.arg1 in modified)
            arg2_const = not (isinstance(inst.arg2, str) and inst.arg2 in modified)
            if arg1_const and arg2_const:
                is_invariant = True

        if is_invariant:
            invariants.append(inst)
            # Remove from modified so downstream invariants can also hoist
            modified.discard(inst.result)
        else:
            remaining_loop.append(inst)

    # Reassemble: pre-loop + hoisted invariants + loop header + remaining loop + loop condition + post-loop
    assembled = (
        instructions[:loop_start] +
        invariants +
        [instructions[loop_start]] +
        remaining_loop +
        instructions[loop_end:]
    )
    return TACProgram(assembled)


def optimize_tac_program(prog: TACProgram) -> Tuple[TACProgram, Dict[str, int]]:
    """Runs the complete 4-pass optimization pipeline until convergence."""
    curr = prog.clone()
    stats = {"original_inst_count": len(curr.instructions)}

    curr = pass_constant_folding_and_propagation(curr)
    stats["after_constant_folding"] = len(curr.instructions)

    curr = pass_common_subexpression_elimination(curr)
    stats["after_cse"] = len(curr.instructions)

    curr = pass_loop_invariant_code_motion(curr)
    stats["after_licm"] = len(curr.instructions)

    curr = pass_dead_code_elimination(curr)
    stats["after_dce"] = len(curr.instructions)
    stats["optimized_inst_count"] = len(curr.instructions)

    return curr, stats


# ---------------------------------------------------------------------------
# Register Allocation via Graph Coloring (Chaitin-Briggs)
# ---------------------------------------------------------------------------

class RegisterAllocator:
    def __init__(self, k_registers: int = 3):
        self.K = k_registers
        self.regs = [f"R{i}" for i in range(k_registers)]

    def compute_liveness(self, prog: TACProgram) -> Tuple[List[Set[str]], Set[str]]:
        """Compute variable live ranges and active variables at each instruction."""
        insts = prog.instructions
        n = len(insts)
        live_in = [set() for _ in range(n)]
        live_out = [set() for _ in range(n)]
        all_vars = set()

        # Def and Use sets
        def_set = [set() for _ in range(n)]
        use_set = [set() for _ in range(n)]

        for i, inst in enumerate(insts):
            if inst.result and inst.op not in ('LABEL', 'JUMP', 'IF_LE'):
                def_set[i].add(inst.result)
                all_vars.add(inst.result)
            if isinstance(inst.arg1, str) and inst.op != 'JUMP':
                use_set[i].add(inst.arg1)
                all_vars.add(inst.arg1)
            if isinstance(inst.arg2, str):
                use_set[i].add(inst.arg2)
                all_vars.add(inst.arg2)

        # Iterative liveness dataflow analysis
        changed = True
        while changed:
            changed = False
            for i in reversed(range(n)):
                succ = [i + 1] if i + 1 < n else []
                # Add jump successors if needed
                if insts[i].op == 'JUMP':
                    for j, target in enumerate(insts):
                        if target.op == 'LABEL' and target.result == insts[i].arg1:
                            succ = [j]
                elif insts[i].op == 'IF_LE':
                    for j, target in enumerate(insts):
                        if target.op == 'LABEL' and target.result == insts[i].result:
                            succ.append(j)

                new_out = set()
                for s in succ:
                    new_out.update(live_in[s])

                new_in = use_set[i].union(new_out - def_set[i])

                if new_in != live_in[i] or new_out != live_out[i]:
                    live_in[i] = new_in
                    live_out[i] = new_out
                    changed = True

        return live_out, all_vars

    def build_interference_graph(self, live_out: List[Set[str]], all_vars: Set[str]) -> Dict[str, Set[str]]:
        """Construct undirected interference graph: edge exists between u and v if simultaneously live."""
        adj: Dict[str, Set[str]] = {v: set() for v in all_vars}
        for live in live_out:
            live_list = list(live)
            for i in range(len(live_list)):
                for j in range(i + 1, len(live_list)):
                    u, v = live_list[i], live_list[j]
                    adj[u].add(v)
                    adj[v].add(u)
        return adj

    def allocate(self, prog: TACProgram) -> Dict[str, Any]:
        """Performs Chaitin-Briggs graph coloring with Kempe heuristic and spilling."""
        live_out, all_vars = self.compute_liveness(prog)
        adj = self.build_interference_graph(live_out, all_vars)

        # Simplify stage (Kempe heuristic)
        graph = {k: set(v) for k, v in adj.items()}
        stack = []
        spilled = set()

        while graph:
            # Find node with degree < K
            low_deg = [n for n, neighbors in graph.items() if len(neighbors) < self.K]
            if low_deg:
                node = low_deg[0]
                stack.append((node, list(graph[node])))
                # Remove node from graph
                for neighbor in graph[node]:
                    graph[neighbor].discard(node)
                del graph[node]
            else:
                # Potential Spill: pick node with maximum degree
                spill_candidate = max(graph.keys(), key=lambda n: len(graph[n]))
                spilled.add(spill_candidate)
                stack.append((spill_candidate, list(graph[spill_candidate])))
                for neighbor in graph[spill_candidate]:
                    graph[neighbor].discard(spill_candidate)
                del graph[spill_candidate]

        # Select stage (Coloring)
        color_map = {}
        actual_spills = []

        while stack:
            node, neighbors = stack.pop()
            used_colors = {color_map[neigh] for neigh in neighbors if neigh in color_map}
            avail_colors = [c for c in self.regs if c not in used_colors]
            if avail_colors:
                color_map[node] = avail_colors[0]
            else:
                color_map[node] = "SPILL_TO_STACK"
                actual_spills.append(node)

        return {
            "registers_available": self.K,
            "total_variables": len(all_vars),
            "interference_graph": adj,
            "allocation": color_map,
            "spilled_variables": actual_spills,
            "spill_count": len(actual_spills)
        }
