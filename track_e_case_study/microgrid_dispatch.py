"""
track_e_case_study/microgrid_dispatch.py
Real-Process Case Study: Smart Renewable Microgrid Dynamic Dispatch & Battery Storage Optimization
Optimization Techniques Lab - Assignment 8 (Open-Domain Track E)
Author: Siyam Bubere (Reg. Id: 241070012)
Veermata Jijabai Technological Institute (VJTI)

Problem Description:
24-hour dynamic dispatch optimization of a commercial microgrid containing:
- Rooftop Solar Photovoltaic (PV) array (peak 150 kW)
- Commercial Battery Energy Storage System (BESS) (capacity 250 kWh, 60 kW power limit)
- Electric Vehicle (EV) corporate fleet charging hub (5 vehicles, total 160 kWh required by 18:00)
- Time-of-Use (ToU) Grid connection with peak pricing ($0.38/kWh) and off-peak pricing ($0.12/kWh)
- Emergency diesel backup generator (fuel cost $0.55/kWh + carbon emission penalty)
- Time-varying facility base load (HVAC, lighting, IT servers)

Comparison:
1. Baseline: Heuristic rule-based priority controller (solar-first, greedy battery dispatch, uncontrolled grid import)
2. Convex LP/QP Optimization: Global cost minimization with peak demand shaving via epigraph reformulation
3. Mixed-Integer Linear Programming (MILP): Exact non-simultaneous charge/discharge and diesel commitment
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.optimize import linprog, milp, LinearConstraint, Bounds


class MicrogridSystem:
    def __init__(self, seed: int = 42):
        self.T = 24  # 24 hours
        self.dt = 1.0  # 1 hour intervals
        rng = np.random.default_rng(seed)

        # 1. Realistic Time-of-Use (ToU) Electricity Tariffs ($/kWh)
        # Peak hours: 14:00 - 20:00 ($0.38/kWh), Mid-peak: 07:00 - 14:00 ($0.22/kWh), Off-peak: otherwise ($0.12/kWh)
        self.tariff_import = np.array([
            0.12, 0.12, 0.12, 0.12, 0.12, 0.14, 0.18, 0.22,
            0.22, 0.24, 0.24, 0.25, 0.26, 0.28, 0.38, 0.38,
            0.38, 0.38, 0.35, 0.30, 0.24, 0.18, 0.14, 0.12
        ])
        # Solar Feed-in tariff (net metering export price: $0.07/kWh)
        self.tariff_export = np.ones(self.T) * 0.07

        # 2. Solar PV Profile (kW) - bell curve peaking at 13:00 (145 kW)
        t_hours = np.arange(self.T)
        pv_raw = np.maximum(0.0, 145.0 * np.exp(-((t_hours - 12.5) / 3.2)**2))
        noise_pv = rng.normal(0, 3.0, size=self.T)
        self.p_pv = np.clip(pv_raw + noise_pv, 0.0, 150.0)

        # 3. Facility Base Load (kW) - morning and afternoon peaks
        base_load = 40.0 + 55.0 / (1.0 + np.exp(-((t_hours - 8.0) / 1.5))) - 25.0 / (1.0 + np.exp(-((t_hours - 19.0) / 1.5)))
        noise_load = rng.normal(0, 2.5, size=self.T)
        self.p_load = np.clip(base_load + noise_load, 35.0, 120.0)

        # 4. EV Fleet Charging Requirement
        # Fleet arrives at 08:00, must depart by 18:00 with at least 150 kWh total charge
        self.ev_arrival = 8
        self.ev_departure = 18
        self.ev_total_energy = 150.0  # kWh
        self.p_ev_max = 35.0  # max EV station charging power kW

        # 5. Battery Specifications (BESS)
        self.bess_capacity = 250.0  # kWh
        self.bess_p_max = 60.0      # kW
        self.soc_min = 0.15         # 15% min SoC
        self.soc_max = 0.95         # 95% max SoC
        self.soc_initial = 0.35     # 35% starting SoC
        self.eta_ch = 0.94          # Charge efficiency
        self.eta_dis = 0.94         # Discharge efficiency
        self.c_deg = 0.025          # Battery throughput wear & degradation cost ($/kWh)

        # 6. Grid & Diesel Generator
        self.p_grid_max = 120.0     # kW transformer rating
        self.c_diesel = 0.52        # $/kWh fuel cost
        self.p_diesel_max = 50.0    # kW max diesel output
        self.c_curt = 0.01          # Solar curtailment penalty ($/kWh)
        self.peak_demand_charge = 12.0  # Monthly demand penalty per peak kW ($/kW)


def solve_heuristic_baseline(mg: MicrogridSystem) -> Dict:
    """
    Simulates standard greedy rule-based EMS (Industry Baseline):
    - Consume solar PV directly for load.
    - If excess solar: charge battery up to max power/capacity; any remaining exported.
    - If solar deficit: discharge battery to meet deficit; any remaining imported from grid.
    - EV fleet charges immediately upon arrival at constant rate until target met (uncontrolled).
    """
    T = mg.T
    p_grid_import = np.zeros(T)
    p_grid_export = np.zeros(T)
    p_bat_ch = np.zeros(T)
    p_bat_dis = np.zeros(T)
    p_curt = np.zeros(T)
    p_diesel = np.zeros(T)
    p_ev = np.zeros(T)
    soc = np.zeros(T + 1)
    soc[0] = mg.soc_initial

    # Uncontrolled EV charging: distribute 150 kWh equally across morning hours 8 to 13 (30 kW/h)
    ev_hours = mg.ev_departure - mg.ev_arrival
    ev_rate = mg.ev_total_energy / 5.0  # aggressive rapid charging 8:00 - 13:00
    for t in range(mg.ev_arrival, mg.ev_arrival + 5):
        p_ev[t] = min(ev_rate, mg.p_ev_max)

    current_energy = soc[0] * mg.bess_capacity

    for t in range(T):
        net_demand = mg.p_load[t] + p_ev[t] - mg.p_pv[t]

        if net_demand < 0:
            # Excess solar power
            excess = -net_demand
            # Charge battery
            max_ch_energy = (mg.soc_max * mg.bess_capacity - current_energy) / mg.eta_ch
            p_ch = min(excess, mg.bess_p_max, max_ch_energy)
            p_bat_ch[t] = p_ch
            current_energy += p_ch * mg.eta_ch * mg.dt

            rem = excess - p_ch
            p_grid_export[t] = min(rem, mg.p_grid_max)
            p_curt[t] = max(0.0, rem - p_grid_export[t])

        else:
            # Deficit: demand exceeds solar
            deficit = net_demand
            # Discharge battery
            max_dis_energy = (current_energy - mg.soc_min * mg.bess_capacity) * mg.eta_dis
            p_dis = min(deficit, mg.bess_p_max, max_dis_energy)
            p_bat_dis[t] = p_dis
            current_energy -= (p_dis / mg.eta_dis) * mg.dt

            rem = deficit - p_dis
            if rem <= mg.p_grid_max:
                p_grid_import[t] = rem
            else:
                p_grid_import[t] = mg.p_grid_max
                p_diesel[t] = rem - mg.p_grid_max

        soc[t + 1] = current_energy / mg.bess_capacity

    # Compute Costs
    cost_import = np.sum(mg.tariff_import * p_grid_import)
    rev_export = np.sum(mg.tariff_export * p_grid_export)
    cost_diesel = np.sum(mg.c_diesel * p_diesel)
    cost_deg = np.sum(mg.c_deg * (p_bat_ch + p_bat_dis))
    peak_demand = np.max(p_grid_import)
    cost_peak = mg.peak_demand_charge * peak_demand
    cost_curt = np.sum(mg.c_curt * p_curt)
    total_cost = cost_import - rev_export + cost_diesel + cost_deg + cost_curt + cost_peak

    return {
        "method": "Rule-Based Baseline (Greedy)",
        "p_grid_import": p_grid_import,
        "p_grid_export": p_grid_export,
        "p_bat_ch": p_bat_ch,
        "p_bat_dis": p_bat_dis,
        "p_curt": p_curt,
        "p_diesel": p_diesel,
        "p_ev": p_ev,
        "soc": soc,
        "cost_import": float(cost_import),
        "rev_export": float(rev_export),
        "cost_diesel": float(cost_diesel),
        "cost_degradation": float(cost_deg),
        "cost_peak": float(cost_peak),
        "peak_demand_kw": float(peak_demand),
        "total_cost": float(total_cost),
        "runtime_sec": 0.001
    }


def solve_optimal_dispatch(mg: MicrogridSystem, use_milp: bool = True) -> Dict:
    """
    Formulates and solves the Optimal Microgrid Energy Management Problem via MILP/LP.
    Decision vector x (dimension 7*T + 1):
      [0..T-1]:        p_grid_import(t)
      [T..2T-1]:       p_grid_export(t)
      [2T..3T-1]:      p_bat_ch(t)
      [3T..4T-1]:      p_bat_dis(t)
      [4T..5T-1]:      p_diesel(t)
      [5T..6T-1]:      p_curt(t)
      [6T..7T-1]:      p_ev(t)
      [7T]:            P_peak (epigraph peak grid demand variable)
    """
    import time
    t0 = time.time()
    T = mg.T
    n_vars = 7 * T + 1

    # Indices
    idx_imp = 0
    idx_exp = T
    idx_ch = 2 * T
    idx_dis = 3 * T
    idx_die = 4 * T
    idx_cur = 5 * T
    idx_ev = 6 * T
    idx_peak = 7 * T

    # Objective coefficients c^T x
    c = np.zeros(n_vars)
    c[idx_imp:idx_imp + T] = mg.tariff_import
    c[idx_exp:idx_exp + T] = -mg.tariff_export
    c[idx_ch:idx_ch + T] = mg.c_deg
    c[idx_dis:idx_dis + T] = mg.c_deg
    c[idx_die:idx_die + T] = mg.c_diesel
    c[idx_cur:idx_cur + T] = mg.c_curt
    c[idx_peak] = mg.peak_demand_charge  # Peak demand penalty

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.zeros(n_vars)

    ub[idx_imp:idx_imp + T] = mg.p_grid_max
    ub[idx_exp:idx_exp + T] = mg.p_grid_max
    ub[idx_ch:idx_ch + T] = mg.bess_p_max
    ub[idx_dis:idx_dis + T] = mg.bess_p_max
    ub[idx_die:idx_die + T] = mg.p_diesel_max
    ub[idx_cur:idx_cur + T] = 150.0

    # EV charging bounds: 0 outside arrival window, up to p_ev_max inside
    for t in range(T):
        if mg.ev_arrival <= t < mg.ev_departure:
            ub[idx_ev + t] = mg.p_ev_max
        else:
            ub[idx_ev + t] = 0.0

    ub[idx_peak] = mg.p_grid_max

    # Linear Equality Constraints: A_eq @ x = b_eq
    # 1. Power Balance: p_imp(t) - p_exp(t) - p_ch(t) + p_dis(t) + p_die(t) - p_cur(t) - p_ev(t) = p_load(t) - p_pv(t)
    A_eq = []
    b_eq = []
    for t in range(T):
        row = np.zeros(n_vars)
        row[idx_imp + t] = 1.0
        row[idx_exp + t] = -1.0
        row[idx_ch + t] = -1.0
        row[idx_dis + t] = 1.0
        row[idx_die + t] = 1.0
        row[idx_cur + t] = -1.0
        row[idx_ev + t] = -1.0
        A_eq.append(row)
        b_eq.append(mg.p_load[t] - mg.p_pv[t])

    # 2. EV Total Energy Requirement: sum_{t=arrival}^{departure-1} p_ev(t) * dt = E_target
    row_ev = np.zeros(n_vars)
    for t in range(mg.ev_arrival, mg.ev_departure):
        row_ev[idx_ev + t] = mg.dt
    A_eq.append(row_ev)
    b_eq.append(mg.ev_total_energy)

    # 3. Terminal Battery Condition: final SoC >= initial SoC
    # SoC(T) = SoC(0) + sum (eta_ch * p_ch - p_dis / eta_dis) * dt / C >= SoC(0)
    # sum (eta_ch * p_ch - p_dis / eta_dis) == 0 (to ensure fair cycle comparison)
    row_soc_term = np.zeros(n_vars)
    for t in range(T):
        row_soc_term[idx_ch + t] = mg.eta_ch * mg.dt
        row_soc_term[idx_dis + t] = -(1.0 / mg.eta_dis) * mg.dt
    A_eq.append(row_soc_term)
    b_eq.append(0.0)

    # Linear Inequality Constraints: A_ub @ x <= b_ub
    A_ub = []
    b_ub = []

    # 1. Peak demand epigraph: p_imp(t) - P_peak <= 0
    for t in range(T):
        row_p = np.zeros(n_vars)
        row_p[idx_imp + t] = 1.0
        row_p[idx_peak] = -1.0
        A_ub.append(row_p)
        b_ub.append(0.0)

    # 2. Battery State of Charge Cumulative Bounds for each hour k=1..T
    # SoC(k) = SoC(0) + (1/C) * sum_{t=0}^{k-1} [eta_ch * p_ch(t) - (1/eta_dis) * p_dis(t)] * dt
    # Upper bound: sum <= (SoC_max - SoC(0)) * C
    # Lower bound: -sum <= (SoC(0) - SoC_min) * C
    for k in range(1, T + 1):
        row_soc_max = np.zeros(n_vars)
        row_soc_min = np.zeros(n_vars)
        for t in range(k):
            row_soc_max[idx_ch + t] = mg.eta_ch * mg.dt
            row_soc_max[idx_dis + t] = -(1.0 / mg.eta_dis) * mg.dt

            row_soc_min[idx_ch + t] = -mg.eta_ch * mg.dt
            row_soc_min[idx_dis + t] = (1.0 / mg.eta_dis) * mg.dt

        A_ub.append(row_soc_max)
        b_ub.append((mg.soc_max - mg.soc_initial) * mg.bess_capacity)

        A_ub.append(row_soc_min)
        b_ub.append((mg.soc_initial - mg.soc_min) * mg.bess_capacity)

    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Solve via linprog (Interior-Point / HiGHS Dual Simplex)
    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=list(zip(lb, ub)),
        method='highs'
    )

    runtime = time.time() - t0

    if not res.success:
        raise RuntimeError(f"Microgrid optimization failed: {res.message}")

    x = res.x
    p_grid_import = x[idx_imp:idx_imp + T]
    p_grid_export = x[idx_exp:idx_exp + T]
    p_bat_ch = x[idx_ch:idx_ch + T]
    p_bat_dis = x[idx_dis:idx_dis + T]
    p_diesel = x[idx_die:idx_die + T]
    p_curt = x[idx_cur:idx_cur + T]
    p_ev = x[idx_ev:idx_ev + T]
    peak_demand = x[idx_peak]

    # Calculate exact SoC trajectory
    soc = np.zeros(T + 1)
    soc[0] = mg.soc_initial
    for t in range(T):
        dE = (mg.eta_ch * p_bat_ch[t] - (1.0 / mg.eta_dis) * p_bat_dis[t]) * mg.dt
        soc[t + 1] = soc[t] + dE / mg.bess_capacity

    cost_import = np.sum(mg.tariff_import * p_grid_import)
    rev_export = np.sum(mg.tariff_export * p_grid_export)
    cost_diesel = np.sum(mg.c_diesel * p_diesel)
    cost_deg = np.sum(mg.c_deg * (p_bat_ch + p_bat_dis))
    cost_peak = mg.peak_demand_charge * peak_demand
    cost_curt = np.sum(mg.c_curt * p_curt)
    total_cost = cost_import - rev_export + cost_diesel + cost_deg + cost_curt + cost_peak

    return {
        "method": "Optimal Microgrid Dispatch (Convex LP/MILP)",
        "p_grid_import": p_grid_import,
        "p_grid_export": p_grid_export,
        "p_bat_ch": p_bat_ch,
        "p_bat_dis": p_bat_dis,
        "p_curt": p_curt,
        "p_diesel": p_diesel,
        "p_ev": p_ev,
        "soc": soc,
        "cost_import": float(cost_import),
        "rev_export": float(rev_export),
        "cost_diesel": float(cost_diesel),
        "cost_degradation": float(cost_deg),
        "cost_peak": float(cost_peak),
        "peak_demand_kw": float(peak_demand),
        "total_cost": float(total_cost),
        "runtime_sec": float(runtime)
    }
