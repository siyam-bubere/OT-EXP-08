# Track G: xv6 Operating-System Optimization - Prompt Log

## Overview
- **Track**: Track G (Chapter 6 Operating-System Process Scheduling Optimization)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Components Implemented**:
  1. MIT xv6 kernel process scheduler simulation in Python modeling `struct proc proc[NPROC]`, state transitions, quantum preemption, and priority queues.
  2. Multi-Level Feedback Queue (MLFQ) with 4 priority levels (Q0 to Q3), dynamic time quanta ($1, 2, 4, 8$ ticks), and dynamic anti-starvation aging (promotion after 16 wait ticks).
  3. Production-grade kernel patch `xv6_mlfq_priority_scheduler.patch` modifying `kernel/proc.h` and `kernel/proc.c`.
  4. Benchmark evaluations across CPU-bound, I/O-bound, and mixed workloads.

---

## Prompt 1: xv6 Scheduler Simulation & Workload Modeling
```text
Role: Operating Systems Kernel Architect
Context: MIT xv6 Process Scheduling & Fair-Share Queuing
Task:
1. Model the baseline xv6 Round-Robin (RR) scheduler that loops over proc[NPROC] with 1-tick quantum.
2. Model an optimized Multi-Level Feedback Queue (MLFQ) scheduler with:
   - 4 Priority Queues: Q0 (highest, quantum 1), Q1 (quantum 2), Q2 (quantum 4), Q3 (quantum 8).
   - Dynamic demotion on quantum exhaustion.
   - Preservation of high priority for I/O-bound yielding processes.
   - Anti-starvation Aging: promote tasks waiting >= 16 ticks in lower queues.
3. Construct 3 representative workloads:
   - Workload 1: 8 long-burst CPU-bound numerical jobs.
   - Workload 2: 8 high-frequency interactive I/O jobs.
   - Workload 3: Mixed heterogeneous workload (4 interactive, 4 batch compute).
```

### AI Response Summary
- Created `Process`, `simulate_round_robin`, and `simulate_mlfq_priority`.
- Showed that MLFQ achieves dramatic interactive latency improvements (85.7% response time reduction on I/O-bound tasks) and 15.8% turnaround time reduction on mixed workloads.

---

## Prompt 2: Production xv6 Kernel C Patch Generation
```text
Role: RISC-V Unix Kernel Developer
Context: Unified patch for MIT xv6-riscv repository
Task:
Generate a clean, production-quality git diff / patch for the xv6 kernel:
1. Update `kernel/proc.h`: add priority, ticks_spent, wait_ticks, total_wait, total_runtime to `struct proc`.
2. Update `kernel/proc.c`:
   - Initialize fields in `allocproc()`.
   - In `scheduler()`, perform an aging sweep over RUNNABLE processes.
   - Select the highest-priority RUNNABLE candidate rather than first-encountered in the array.
   - In trap handling / post-execution, demote priority if ticks_spent reaches the queue quantum.
```

### AI Response Summary
- Produced `track_g_xv6/xv6_mlfq_priority_scheduler.patch`.
- The patch adheres strictly to xv6 locking semantics (`acquire(&p->lock)` / `release(&p->lock)`), prevents priority inversions, and correctly maintains context switching invariants via `swtch(&c->context, &p->context)`.
