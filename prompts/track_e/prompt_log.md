# Track E: Main Case Study (Microgrid Optimization) - Prompt Log

## Overview
- **Track**: Track E (Open-Domain Real-Process Case Study)
- **Author**: Siyam Bubere (Reg. Id: 241070012), VJTI Mumbai
- **Course**: Optimization Techniques (Lab Experiment 8)
- **Problem Selected**: 24-Hour Smart Renewable Microgrid Dynamic Dispatch with Solar PV, Stationary BESS, Corporate EV Fleet Charging Logistics, and Peak Demand Epigraph Shaving.

---

## Prompt 1: Microgrid Process Modeling & Baseline Formulation
```text
Role: Power Systems Operations & Optimization Engineer
Context: Chapter 6 Open-Domain Process Optimization
Task:
1. Define a 24-hour smart microgrid containing:
   - 150 kW Solar PV generation curve
   - 250 kWh Battery Energy Storage System (BESS, 60 kW power rating, 15%-95% SoC bounds)
   - 150 kWh Corporate EV fleet charging demand (arrival 08:00, departure 18:00)
   - Dynamic Time-of-Use (ToU) electricity import tariff ($0.12 - $0.38 / kWh) and net-metering export ($0.07 / kWh)
   - Emergency diesel generator ($0.52 / kWh) and transformer peak demand charge ($12 / peak kW).
2. Model the standard industrial baseline: Heuristic greedy priority controller (solar-first, greedy battery charging/discharging, uncontrolled morning EV charging).
3. Compute total daily operational cost, import costs, degradation wear, and transformer peak demand.
```

### AI Response Summary
- Created `MicrogridSystem` with realistic solar irradiance curves, diurnal building load profiles, and Time-of-Use tariff tiers.
- Implemented `solve_heuristic_baseline` simulating uncontrolled EV fleet charging upon morning arrival, resulting in a severe transformer peak load of 80.9 kW and a daily operating cost of $1,183.92 (dominated by a $970.23 peak demand penalty).

---

## Prompt 2: Global Convex Formulation & Epigraph Peak Shaving
```text
Role: Mathematical Programming Specialist
Context: Epigraph Reformulation & Linear Programming
Task:
1. Formulate the microgrid dispatch as a Linear Program (LP/MILP) over decision vector x in R^(7T + 1):
   x = [p_grid_import, p_grid_export, p_bat_ch, p_bat_dis, p_diesel, p_curt, p_ev, P_peak]
2. Define objective:
   min J = sum_t [c_imp(t)*p_imp(t) - c_exp(t)*p_exp(t) + c_deg*(p_ch(t) + p_dis(t)) + c_die*p_die(t)] + lambda_peak * P_peak
3. Enforce:
   - Power balance equality: p_imp - p_exp - p_ch + p_dis + p_die - p_cur - p_ev = p_load - p_pv
   - Epigraph peak inequality: p_imp(t) - P_peak <= 0 for all t
   - Battery SoC cumulative inequalities ensuring SoC stays within [0.15, 0.95] for every intermediate hour
   - Fleet total energy constraint: sum_{t=8}^{17} p_ev(t) * dt = 150 kWh
   - Terminal battery neutral condition: SoC(24) >= SoC(0)
4. Solve via HiGHS simplex/interior-point solver.
```

### AI Response Summary
- Formulated exact linear equality and inequality matrices for 24 time steps.
- The optimizer shifted EV charging from morning peak hours to peak solar production (11:00 - 14:00) and stored surplus solar energy in the BESS to discharge during evening peak tariff hours (17:00 - 20:00).
- Peak grid demand dropped from 80.9 kW down to 5.2 kW (93.6% reduction).
- Total daily operating cost dropped from $1,183.92 to $575.12 (proven 51.4% cost savings).
