"""
track_g_xv6/xv6_scheduler_sim.py
Track G: xv6 Operating-System Optimization (Process Scheduling)
Optimization Techniques Lab - Assignment 8 (Chapter 6 OS Optimization)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Simulates and benchmarks the MIT xv6 OS kernel process scheduler:
1. Baseline: Standard xv6 Round-Robin (RR) scheduler (flat scan over proc[NPROC])
2. Optimized: Multi-Level Feedback Queue (MLFQ) Priority Scheduler with Dynamic Aging
   - 4 Priority Queues (Q0 = highest, Q3 = lowest)
   - Dynamic time-slice quantum: Q0=1 tick, Q1=2 ticks, Q2=4 ticks, Q3=8 ticks
   - Anti-starvation Aging: processes waiting > 16 ticks promoted by 1 level
   - Interactive boost: processes yielding before quantum exhaustion stay in high-priority queue

Evaluated on 3 workloads:
- Workload 1: CPU-Bound processes (long computational bursts)
- Workload 2: I/O-Bound processes (short interactive bursts with I/O yields)
- Workload 3: Mixed Workload (heterogeneous mix of interactive and compute jobs)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy


class Process:
    """Simulates xv6 struct proc."""
    def __init__(self, pid: int, name: str, arrival_time: int, bursts: List[Tuple[str, int]], priority: int = 0):
        self.pid = pid
        self.name = name
        self.arrival_time = arrival_time
        # bursts is a list of ('CPU', duration) or ('IO', duration)
        self.remaining_bursts = list(bursts)
        self.initial_priority = priority
        self.priority = priority  # 0 (highest) to 3 (lowest)
        self.quantum_left = 1
        self.wait_time = 0
        self.starvation_wait = 0

        # Performance metrics
        self.first_run_time: Optional[int] = None
        self.completion_time: Optional[int] = None
        self.total_cpu_time = sum(dur for kind, dur in bursts if kind == 'CPU')
        self.cpu_time_received = 0
        self.io_wait_remaining = 0

    def is_finished(self) -> bool:
        return len(self.remaining_bursts) == 0 and self.io_wait_remaining == 0


def create_workload_1_cpu_bound() -> List[Process]:
    """Workload 1: 8 CPU-bound numerical processes arriving simultaneously."""
    procs = []
    for i in range(8):
        bursts = [('CPU', 25 + (i % 3) * 10)]
        procs.append(Process(pid=i + 1, name=f"cpu_job_{i+1}", arrival_time=0, bursts=bursts, priority=2))
    return procs


def create_workload_2_io_bound() -> List[Process]:
    """Workload 2: 8 Interactive/IO-bound shell and editor processes with short bursts."""
    procs = []
    for i in range(8):
        bursts = []
        for _ in range(5):
            bursts.append(('CPU', 2))
            bursts.append(('IO', 3))
        procs.append(Process(pid=i + 1, name=f"interactive_{i+1}", arrival_time=i * 2, bursts=bursts, priority=0))
    return procs


def create_workload_3_mixed() -> List[Process]:
    """Workload 3: Mixed realistic workload with 4 interactive and 4 heavy compute tasks."""
    procs = []
    # 4 interactive
    for i in range(4):
        bursts = [('CPU', 2), ('IO', 2), ('CPU', 2), ('IO', 3), ('CPU', 1)]
        procs.append(Process(pid=i + 1, name=f"gui_proc_{i+1}", arrival_time=i * 2, bursts=bursts, priority=0))
    # 4 CPU-bound
    for i in range(4, 8):
        bursts = [('CPU', 35 + (i - 4) * 8)]
        procs.append(Process(pid=i + 1, name=f"batch_proc_{i+1}", arrival_time=i, bursts=bursts, priority=3))
    return procs


def simulate_round_robin(processes: List[Process], quantum: int = 1) -> Dict:
    """Standard MIT xv6 Round-Robin scheduler: flat scan over proc table."""
    procs = [copy.deepcopy(p) for p in processes]
    current_time = 0
    ready_queue: List[Process] = []
    io_queue: List[Process] = []
    completed: List[Process] = []

    active_proc: Optional[Process] = None
    curr_quantum_used = 0

    while len(completed) < len(procs):
        # 1. Admit arriving processes
        for p in procs:
            if p.arrival_time == current_time:
                ready_queue.append(p)

        # 2. Advance I/O waiting processes
        new_io_queue = []
        for p in io_queue:
            p.io_wait_remaining -= 1
            if p.io_wait_remaining <= 0:
                if len(p.remaining_bursts) == 0:
                    p.completion_time = current_time + 1
                    completed.append(p)
                else:
                    ready_queue.append(p)
            else:
                new_io_queue.append(p)
        io_queue = new_io_queue

        # 3. Handle active process quantum or burst completion
        if active_proc:
            active_proc.cpu_time_received += 1
            curr_quantum_used += 1

            kind, dur = active_proc.remaining_bursts[0]
            if dur - 1 <= 0:
                # Finished this CPU burst
                active_proc.remaining_bursts.pop(0)
                curr_quantum_used = 0
                if len(active_proc.remaining_bursts) > 0 and active_proc.remaining_bursts[0][0] == 'IO':
                    # Enter I/O wait
                    active_proc.io_wait_remaining = active_proc.remaining_bursts.pop(0)[1]
                    io_queue.append(active_proc)
                else:
                    if active_proc.is_finished():
                        active_proc.completion_time = current_time + 1
                        completed.append(active_proc)
                    else:
                        ready_queue.append(active_proc)
                active_proc = None
            elif curr_quantum_used >= quantum:
                # Preempt by Round Robin quantum
                active_proc.remaining_bursts[0] = (kind, dur - 1)
                curr_quantum_used = 0
                ready_queue.append(active_proc)
                active_proc = None
            else:
                active_proc.remaining_bursts[0] = (kind, dur - 1)

        # 4. Schedule next process if CPU idle
        if active_proc is None and ready_queue:
            active_proc = ready_queue.pop(0)
            if active_proc.first_run_time is None:
                active_proc.first_run_time = current_time

        # Update wait times for processes in ready queue
        for p in ready_queue:
            p.wait_time += 1

        current_time += 1

    # Calculate metrics
    turnarounds = [p.completion_time - p.arrival_time for p in completed]
    response_times = [p.first_run_time - p.arrival_time for p in completed]
    wait_times = [p.wait_time for p in completed]

    return {
        "scheduler": "Standard xv6 Round-Robin",
        "avg_turnaround_ticks": float(np.mean(turnarounds)),
        "avg_response_ticks": float(np.mean(response_times)),
        "avg_wait_ticks": float(np.mean(wait_times)),
        "max_wait_ticks": float(np.max(wait_times)),
        "total_ticks": current_time
    }


def simulate_mlfq_priority(processes: List[Process], aging_threshold: int = 16) -> Dict:
    """
    Optimized xv6 MLFQ Scheduler with Priority Aging:
    - 4 priority levels (0 highest, 3 lowest)
    - Time quanta: Q0=1, Q1=2, Q2=4, Q3=8
    - Anti-starvation Aging: wait > 16 ticks promotes priority by 1 level
    - Voluntary yield on I/O preserves high priority
    """
    procs = [copy.deepcopy(p) for p in processes]
    current_time = 0
    queues: Dict[int, List[Process]] = {0: [], 1: [], 2: [], 3: []}
    io_queue: List[Process] = []
    completed: List[Process] = []

    quantum_map = {0: 1, 1: 2, 2: 4, 3: 8}

    active_proc: Optional[Process] = None
    curr_quantum_used = 0

    while len(completed) < len(procs):
        # 1. Admit arriving processes
        for p in procs:
            if p.arrival_time == current_time:
                queues[p.priority].append(p)

        # 2. Advance I/O waiting processes
        new_io_queue = []
        for p in io_queue:
            p.io_wait_remaining -= 1
            if p.io_wait_remaining <= 0:
                if len(p.remaining_bursts) == 0:
                    p.completion_time = current_time + 1
                    completed.append(p)
                else:
                    queues[p.priority].append(p)
            else:
                new_io_queue.append(p)
        io_queue = new_io_queue

        # 3. Anti-Starvation Aging check across all ready queues
        for q_level in range(1, 4):
            retained = []
            for p in queues[q_level]:
                if p.starvation_wait >= aging_threshold:
                    p.priority = max(0, p.priority - 1)
                    p.starvation_wait = 0
                    queues[p.priority].append(p)
                else:
                    retained.append(p)
            queues[q_level] = retained

        # 4. Handle active process quantum or burst completion
        if active_proc:
            active_proc.cpu_time_received += 1
            curr_quantum_used += 1

            kind, dur = active_proc.remaining_bursts[0]
            if dur - 1 <= 0:
                # Finished this CPU burst (Voluntary yield / completion)
                active_proc.remaining_bursts.pop(0)
                curr_quantum_used = 0
                if len(active_proc.remaining_bursts) > 0 and active_proc.remaining_bursts[0][0] == 'IO':
                    active_proc.io_wait_remaining = active_proc.remaining_bursts.pop(0)[1]
                    io_queue.append(active_proc)
                else:
                    if active_proc.is_finished():
                        active_proc.completion_time = current_time + 1
                        completed.append(active_proc)
                    else:
                        queues[active_proc.priority].append(active_proc)
                active_proc = None
            elif curr_quantum_used >= quantum_map[active_proc.priority]:
                # Quantum exhausted: demote priority level
                active_proc.remaining_bursts[0] = (kind, dur - 1)
                active_proc.priority = min(3, active_proc.priority + 1)
                curr_quantum_used = 0
                queues[active_proc.priority].append(active_proc)
                active_proc = None
            else:
                active_proc.remaining_bursts[0] = (kind, dur - 1)

        # 5. Schedule next process from highest priority non-empty queue
        if active_proc is None:
            for q_level in range(4):
                if queues[q_level]:
                    active_proc = queues[q_level].pop(0)
                    active_proc.starvation_wait = 0
                    if active_proc.first_run_time is None:
                        active_proc.first_run_time = current_time
                    break

        # Update wait times
        for q_level in range(4):
            for p in queues[q_level]:
                p.wait_time += 1
                p.starvation_wait += 1

        current_time += 1

    turnarounds = [p.completion_time - p.arrival_time for p in completed]
    response_times = [p.first_run_time - p.arrival_time for p in completed]
    wait_times = [p.wait_time for p in completed]

    return {
        "scheduler": "Optimized xv6 MLFQ Priority + Aging",
        "avg_turnaround_ticks": float(np.mean(turnarounds)),
        "avg_response_ticks": float(np.mean(response_times)),
        "avg_wait_ticks": float(np.mean(wait_times)),
        "max_wait_ticks": float(np.max(wait_times)),
        "total_ticks": current_time
    }
