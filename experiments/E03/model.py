"""
CPU-allocation-based DC model (Block-job).

This is the rebuilt unified formulation where:
- x[j,s,t] ∈ Z_+ is the integer number of CPUs allocated to job j at site s and time t
- c[j,s,t] >= 0 is delivered compute service
- No legacy binary execution or auxiliary action variables
- Instantaneous cross-site reallocation is allowed (unless enable_reallocation=False)
- Service termination/incomplete service is represented directly through delivered-work sum

Key concepts:
- Original window: [start_hour_rel, end_hour_rel) - original deadline
- Slack window: extension after original deadline (SPOT_EXTRA_HOURS)
- Late delivery: index set T_delay_j = { t | end_orig <= t < end_orig + Delta_j } (TeX [T_j^end, T_j^end+Delta_j));
  Delta_j from `job_slack_delta_hours` or inferred from `job_active` vs original deadline
- CPU reallocation: spatial price awareness (can move CPUs across sites)
- Service termination: jobs may be partially or fully terminated (incomplete service), subject to optional aggregate minimum completion fraction
- Unfinished work penalty: sum_j phi_j * (W_j - sum c); `unfinished_work_penalty` is scalar (uniform phi_j) or dict[job_id]
- Minimum service completion (termination only): sum_{j,s,t} c >= min_completion_fraction * sum_j W[j]
- Site ramping (only if ramping_charge > 0): auxiliary g_{s,t}, constraints g >= |ΔP| - δ; objective either γ·Σ g² (quadratic, MIQP) or γ·Σ g (linear, MILP). If γ=0, g and ramping constraints are omitted.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from typing import Literal

import pyomo.environ as pyo

RampingPenaltyMode = Literal["quadratic", "linear"]


def _print_to_console(msg: str) -> None:
    """Bypass contextlib.redirect_stdout (e.g. E01 run_once) so progress is visible in the terminal."""
    try:
        out = getattr(sys, "__stdout__", None) or sys.stdout
        print(msg, file=out, flush=True)
    except Exception:
        pass


def solve_data_center_block_jobs_CPU(
    data,
    jobs,          # from workload.block_jobs_loader.load_block_jobs()
    job_active,    # dict[(job_id, t)] -> 0/1, t is relative hour index
    cpu_per_task: float = 1.0,
    tee: bool = False,
    solver_name: str = "gurobi",
    reallocation_penalty: float = 0.0,  # rho: penalty coefficient for reallocation
    late_delivery_penalty: float = 0.0,  # eta: penalty coefficient for late delivery
    ramping_charge: float = 0.0,  # gamma: site-level ramping penalty weight on g (see ramping_penalty_mode)
    # δ_s (MW): free ramp band per site; g >= |ΔP| - δ. Scalar or dict[site] (keys must match data['dc']['sites']).
    ramping_tolerance_MW: float | Mapping = 0.0,
    unfinished_work_penalty: float | Mapping[int, float] = 0.0,
    # phi_j: unfinished-work penalty ($/undelivered vCPU-h). Scalar applies to all jobs; dict is per job_id.
    job_slack_delta_hours: dict[int, int] | None = None,
    # Delta_j in TeX: hours t with end_orig <= t < end_orig + Delta_j incur late-delivery penalty.
    # If None, infer from job_active vs job_end_orig (matches load_block_jobs slack extension).
    enable_reallocation: bool = True,  # If False, disable CPU reallocation (fixed site assignment)
    allow_termination: bool = False,  # If False, enforce strict completion (sum c == W); If True, allow incomplete service (sum c <= W)
    min_completion_fraction: float | None = None,
    # When allow_termination=True: require sum_{j,s,t} c >= frac * sum_j W[j] (e.g. 0.95 = 95% total service).
    pi_HP=None,  # Unit price for HP jobs: dict[s][t] -> $/vCPU-hour, or None if not provided
    pi_SP=None,  # Unit price for SP jobs: dict[s][t] -> $/vCPU-hour, or None if not provided
    load_balance_only: bool = False,  # If True, minimize load imbalance instead of maximizing net revenue
    solver_options: Mapping[str, float | int | str] | None = None,
    # Gurobi: e.g. MIPGap, TimeLimit, Threads, NodefileStart (see compare_flexibility_contributions_CPU.GUROBI_OPTIONS)
    print_model_stats_before_solve: bool = False,
    # Optional tag for print_model_stats_before_solve (e.g. "initial_assignment" vs main solve).
    model_stats_label: str | None = None,
    ramping_penalty_mode: RampingPenaltyMode = "quadratic",
):
    t_start = time.perf_counter()
    if print_model_stats_before_solve:
        try:
            _ns = len(list(data["dc"]["sites"]))
            _nt = len(list(data["time"]["periods"]))
            _nj = len(jobs)
            tag = f"[{model_stats_label}] " if model_stats_label else ""
            _print_to_console(
                f"[Model]{tag}building Pyomo model (sites={_ns}, horizon_hours={_nt}, jobs={_nj})..."
            )
        except Exception:
            pass

    # --------------------------------------------------------------
    # 1. Sets
    # --------------------------------------------------------------
    S = list(data["dc"]["sites"])
    T_list = list(data["time"]["periods"])
    J = list(jobs.keys())

    def _jt(j):
        return str(getattr(jobs[j], "job_type", "HP")).lower()

    J_HP = [j for j in J if _jt(j) == "hp"]
    J_SP = [j for j in J if _jt(j) == "spot"]

    # --------------------------------------------------------------
    # 2. Parameters from data_14bus
    # --------------------------------------------------------------
    bidding_price = data["dc"]["day_ahead_prices"]    # {s: [price_t]}
    P_busy = data["dc"]["capacity_MW"]                # MW
    P_idle = data["dc"]["Pfixed_MW"]                  # MW
    pue_dict = data["dc"].get("PUE", {s: 1.0 for s in S})

    if "N_cpu" not in data["dc"]:
        raise KeyError("data['dc']['N_cpu'] is required for capacity limits.")
    N_cpu_dict = data["dc"]["N_cpu"]

    # Unit prices for HP and SP jobs ($/vCPU-hour). If not provided, use 0.
    if pi_HP is None:
        pi_HP = {s: {t: 0.0 for t in T_list} for s in S}
    if pi_SP is None:
        pi_SP = {s: {t: 0.0 for t in T_list} for s in S}

    # Service rate bounds: kappa_low[s] and kappa_high[s]
    # For now, use fixed bounds. Can be made configurable later.
    # Default: kappa_low = 0.5, kappa_high = 1.0 (meaning each CPU can deliver 0.5-1.0 vCPU-hours per hour)
    kappa_low_dict = {s: 0.5 for s in S}  # Lower per-CPU service rate
    kappa_high_dict = {s: 1.0 for s in S}  # Upper per-CPU service rate

    # --------------------------------------------------------------
    # 3. Job parameters: W[j], X_bar[j], a[j,t], T_late[j]
    # --------------------------------------------------------------
    # W[j]: total required compute work for job j (vCPU-hours)
    # X_bar[j]: per-job max CPU allocation per time slot
    # a[j,t]: admissible execution indicator
    # T_late[j]: late-delivery index set (times after original deadline)
    
    job_work = {}  # W[j]
    job_max_cpu = {}  # X_bar[j]
    job_start_orig = {}  # Original start time
    job_end_orig = {}  # Original end time
    
    for j in J:
        # Calculate original runtime and requested CPU
        start_hour_rel = int(getattr(jobs[j], "start_hour_rel", 0))
        end_hour_rel = int(getattr(jobs[j], "end_hour_rel", 0))
        runtime = max(0, end_hour_rel - start_hour_rel)
        
        # Requested CPU amount
        requested_cpu = float(jobs[j].cpu_total / cpu_per_task)
        
        # W[j] = requested_cpu * original_runtime
        job_work[j] = requested_cpu * runtime
        
        # X_bar[j] = requested_cpu
        job_max_cpu[j] = requested_cpu
        
        job_start_orig[j] = start_hour_rel
        job_end_orig[j] = end_hour_rel

    # --------------------------------------------------------------
    # 4. Pyomo model & sets
    # --------------------------------------------------------------
    m = pyo.ConcreteModel("DC_BlockJobs_CPU")

    # Ramping block only when gamma>0; gamma==0 omits g and ramping constraints (pure MILP, no ramp slack).
    include_ramping = abs(float(ramping_charge)) > 0.0
    _rpm = str(ramping_penalty_mode).lower()
    effective_ramp_mode: RampingPenaltyMode = _rpm if _rpm in ("quadratic", "linear") else "quadratic"
    ramp_linear = include_ramping and effective_ramp_mode == "linear"

    m.S = pyo.Set(initialize=S)
    m.T = pyo.Set(initialize=T_list, ordered=True)
    m.J = pyo.Set(initialize=J)
    m.J_hp = pyo.Set(initialize=J_HP)
    m.J_spot = pyo.Set(initialize=J_SP)

    # --------------------------------------------------------------
    # 5. Data center parameters
    # --------------------------------------------------------------
    m.Pidle = pyo.Param(m.S, initialize=P_idle)       # idle IT power (MW)
    m.Pbusy = pyo.Param(m.S, initialize=P_busy)       # full-load IT power (MW)
    m.Ncpu = pyo.Param(m.S, initialize=N_cpu_dict)    # total vCPUs per site
    m.pue = pyo.Param(m.S, initialize=pue_dict)
    m.kappa_low = pyo.Param(m.S, initialize=kappa_low_dict, within=pyo.NonNegativeReals)
    m.kappa_high = pyo.Param(m.S, initialize=kappa_high_dict, within=pyo.NonNegativeReals)

    if include_ramping:
        if isinstance(ramping_tolerance_MW, Mapping):
            delta_ramp_init = {s: float(ramping_tolerance_MW.get(s, 0.0)) for s in S}
        else:
            _d0 = float(ramping_tolerance_MW)
            delta_ramp_init = {s: _d0 for s in S}
        m.delta_ramp_MW = pyo.Param(m.S, initialize=delta_ramp_init, within=pyo.NonNegativeReals)

    # --------------------------------------------------------------
    # 6. Job parameters
    # --------------------------------------------------------------
    m.W = pyo.Param(m.J, initialize=job_work, within=pyo.NonNegativeReals)  # Total required work (vCPU-hours)
    m.X_bar = pyo.Param(m.J, initialize=job_max_cpu, within=pyo.NonNegativeReals)  # Max CPU per time slot
    m.job_start_orig = pyo.Param(m.J, initialize=job_start_orig, within=pyo.NonNegativeIntegers, default=0)
    m.job_end_orig = pyo.Param(m.J, initialize=job_end_orig, within=pyo.NonNegativeIntegers, default=0)

    def _infer_slack_delta_hours() -> dict[int, int]:
        """Match TeX: slack admissible hours are [end_orig, end_orig + Delta); infer Delta from job_active."""
        delta: dict[int, int] = {}
        for j in J:
            end_o = int(job_end_orig.get(j, len(T_list)))
            active_ts = [t for t in T_list if float(job_active.get((j, t), 0.0)) > 0.5]
            if not active_ts or max(active_ts) < end_o:
                delta[j] = 0
            else:
                delta[j] = int(max(active_ts) - end_o + 1)
        return delta

    inferred_delta = _infer_slack_delta_hours()
    if job_slack_delta_hours is None:
        slack_delta_by_job = inferred_delta
    else:
        slack_delta_by_job = {
            j: int(job_slack_delta_hours[j]) if j in job_slack_delta_hours else inferred_delta[j] for j in J
        }

    if isinstance(unfinished_work_penalty, Mapping):
        phi_by_job = {j: float(unfinished_work_penalty.get(j, 0.0)) for j in J}
    else:
        phi0 = float(unfinished_work_penalty)
        phi_by_job = {j: phi0 for j in J}

    m.phi_unfinished = pyo.Param(m.J, initialize=phi_by_job, within=pyo.NonNegativeReals, default=0.0)
    m.slack_delta_hours = pyo.Param(m.J, initialize=slack_delta_by_job, within=pyo.NonNegativeIntegers, default=0)

    def _job_active_init(_m, j, t):
        return float(job_active.get((j, t), 0.0))

    m.job_active = pyo.Param(m.J, m.T, initialize=_job_active_init, within=pyo.NonNegativeReals)

    # Prices as Params (time-indexed)
    m.pi_HP = pyo.Param(m.S, m.T, initialize=lambda _m, s, t: float(pi_HP[s][t]), within=pyo.NonNegativeReals)
    m.pi_SP = pyo.Param(m.S, m.T, initialize=lambda _m, s, t: float(pi_SP[s][t]), within=pyo.NonNegativeReals)

    # Service price: pi_svc[s,t] (combines HP and SP prices based on job type)
    def _pi_svc_init(_m, j, s, t):
        if j in _m.J_hp:
            return float(pi_HP[s][t])
        else:
            return float(pi_SP[s][t])

    m.pi_svc = pyo.Param(m.J, m.S, m.T, initialize=_pi_svc_init, within=pyo.NonNegativeReals)

    # --------------------------------------------------------------
    # 7. Decision variables
    # --------------------------------------------------------------
    # x[j,s,t] ∈ Z_+: integer number of CPUs allocated to job j at site s and time t
    m.x = pyo.Var(m.J, m.S, m.T, domain=pyo.NonNegativeIntegers, initialize=0)

    # c[j,s,t] >= 0: delivered compute service (vCPU-hours per hour, so vCPU)
    m.c = pyo.Var(m.J, m.S, m.T, domain=pyo.NonNegativeReals, initialize=0)

    # r[j,s,t] >= 0: reallocation magnitude (for t > first(T))
    m.r = pyo.Var(m.J, m.S, m.T, domain=pyo.NonNegativeReals, initialize=0)

    # g[s,t] >= 0: site-level excess ramp (only if ramping_charge > 0; MIQP term gamma * g^2)
    if include_ramping:
        m.g = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals, initialize=0)

    # --------------------------------------------------------------
    # 8. Derived expressions
    # --------------------------------------------------------------
    # JobLoad / L[s,t]: aggregate delivered workload at site s and time t
    def _job_load_rule(_m, s, t):
        return sum(_m.c[j, s, t] for j in _m.J)

    m.JobLoad = pyo.Expression(m.S, m.T, rule=_job_load_rule)

    # Maximum effective service capacity: L_bar[s] = kappa_high[s] * N_cpu[s]
    def _L_bar_rule(_m, s):
        return _m.kappa_high[s] * _m.Ncpu[s]

    m.L_bar = pyo.Expression(m.S, rule=_L_bar_rule)

    # Power: P[s,t] = PUE[s] * P_idle[s] + PUE[s] * (P_busy[s] - P_idle[s]) * L[s,t] / L_bar[s]
    # Note: L_bar[s] is always > 0 (kappa_high * N_cpu), so we can safely divide
    def _power_rule(_m, s, t):
        utilization = _m.JobLoad[s, t] / _m.L_bar[s]
        dynamic_power = (_m.Pbusy[s] - _m.Pidle[s]) * utilization
        return _m.pue[s] * (_m.Pidle[s] + dynamic_power)

    m.Power = pyo.Expression(m.S, m.T, rule=_power_rule)

    # --------------------------------------------------------------
    # 9. Constraints
    # --------------------------------------------------------------
    # 9.1 Availability: x[j,s,t] <= X_bar[j] * a[j,t]
    def _avail_rule(_m, j, s, t):
        return _m.x[j, s, t] <= _m.X_bar[j] * _m.job_active[j, t]

    m.Avail = pyo.Constraint(m.J, m.S, m.T, rule=_avail_rule)

    # 9.2 Service coupling: kappa_low[s] * x[j,s,t] <= c[j,s,t] <= kappa_high[s] * x[j,s,t]
    def _service_lower_rule(_m, j, s, t):
        return _m.kappa_low[s] * _m.x[j, s, t] <= _m.c[j, s, t]

    m.ServiceLower = pyo.Constraint(m.J, m.S, m.T, rule=_service_lower_rule)

    def _service_upper_rule(_m, j, s, t):
        return _m.c[j, s, t] <= _m.kappa_high[s] * _m.x[j, s, t]

    m.ServiceUpper = pyo.Constraint(m.J, m.S, m.T, rule=_service_upper_rule)

    # 9.3 Delivered-work constraint: sum_{s,t} c[j,s,t] * Delta_t <= W[j] or == W[j]
    # Note: Delta_t = 1 hour, so we sum c[j,s,t] directly
    # If allow_termination=False: strict completion (sum c == W[j])
    # If allow_termination=True: allow incomplete service (sum c <= W[j])
    def _delivered_work_rule(_m, j):
        total_delivered = sum(sum(_m.c[j, s, t] for s in _m.S) for t in _m.T)
        if allow_termination:
            return total_delivered <= _m.W[j]
        else:
            return total_delivered == _m.W[j]

    m.DeliveredWork = pyo.Constraint(m.J, rule=_delivered_work_rule)

    total_required_work = float(sum(job_work[j] for j in J))
    if allow_termination and min_completion_fraction is not None:
        frac = float(min_completion_fraction)
        if 0.0 < frac < 1.0 and total_required_work > 0.0:

            def _min_completion_rule(_m):
                delivered = sum(
                    sum(_m.c[j, s, t] for s in _m.S for t in _m.T)
                    for j in _m.J
                )
                return delivered >= frac * total_required_work

            m.MinServiceCompletion = pyo.Constraint(rule=_min_completion_rule)

    # 9.4 Site CPU capacity: sum_j x[j,s,t] <= N_cpu[s]
    def _capacity_rule(_m, s, t):
        return sum(_m.x[j, s, t] for j in _m.J) <= _m.Ncpu[s]

    m.Capacity = pyo.Constraint(m.S, m.T, rule=_capacity_rule)

    # 9.4a No reallocation constraint: if enabled, force fixed site assignment
    # Once a job starts at a site, it must continue at that site
    if not enable_reallocation:
        # Use binary variable to track which site each job is assigned to
        m.z_site = pyo.Var(m.J, m.S, domain=pyo.Binary, initialize=0)
        
        # Each job can only be assigned to one site
        def _site_selection_rule(_m, j):
            return sum(_m.z_site[j, s] for s in _m.S) == 1
        
        m.SiteSelection = pyo.Constraint(m.J, rule=_site_selection_rule)
        
        # Link x[j,s,t] to z_site[j,s]: if z_site[j,s] = 0, then x[j,s,t] = 0
        # Use big-M: x[j,s,t] <= X_bar[j] * z_site[j,s]
        def _no_reallocation_rule(_m, j, s, t):
            return _m.x[j, s, t] <= _m.X_bar[j] * _m.z_site[j, s]
        
        m.NoReallocation = pyo.Constraint(m.J, m.S, m.T, rule=_no_reallocation_rule)

    # 9.5 Reallocation magnitude: for t > first(T)
    # r[j,s,t] >= x[j,s,t] - x[j,s,t-1]
    # r[j,s,t] >= x[j,s,t-1] - x[j,s,t]
    def _reallocation_rule(_m, j, s, t):
        if t == _m.T.first():
            return pyo.Constraint.Skip
        t_prev = _m.T.prev(t)
        return _m.r[j, s, t] >= _m.x[j, s, t] - _m.x[j, s, t_prev]

    m.ReallocationPos = pyo.Constraint(m.J, m.S, m.T, rule=_reallocation_rule)

    def _reallocation_neg_rule(_m, j, s, t):
        if t == _m.T.first():
            return pyo.Constraint.Skip
        t_prev = _m.T.prev(t)
        return _m.r[j, s, t] >= _m.x[j, s, t_prev] - _m.x[j, s, t]

    m.ReallocationNeg = pyo.Constraint(m.J, m.S, m.T, rule=_reallocation_neg_rule)

    if include_ramping:
        # 9.6 Site-level excess ramp: for t > first(T); δ_s free band (MW), g >= |ΔP| - δ
        def _ramping_pos_rule(_m, s, t):
            if t == _m.T.first():
                return pyo.Constraint.Skip
            t_prev = _m.T.prev(t)
            return _m.g[s, t] >= _m.Power[s, t] - _m.Power[s, t_prev] - _m.delta_ramp_MW[s]

        m.RampingPos = pyo.Constraint(m.S, m.T, rule=_ramping_pos_rule)

        def _ramping_neg_rule(_m, s, t):
            if t == _m.T.first():
                return pyo.Constraint.Skip
            t_prev = _m.T.prev(t)
            return _m.g[s, t] >= _m.Power[s, t_prev] - _m.Power[s, t] - _m.delta_ramp_MW[s]

        m.RampingNeg = pyo.Constraint(m.S, m.T, rule=_ramping_neg_rule)

    # --------------------------------------------------------------
    # 10. Objective
    # --------------------------------------------------------------
    # Original window end for energy cost calculation
    if len(J_SP) > 0:
        original_window_end = max(job_end_orig[j] for j in J_SP)
    else:
        original_window_end = len(T_list)

    # T_delay_j = { t : end_orig <= t < end_orig + Delta_j } (TeX \mathcal{T}^{\mathrm{delay}}_j)
    def _is_late_delivery(j, t):
        end_o = int(job_end_orig.get(j, len(T_list)))
        d = int(slack_delta_by_job.get(j, 0))
        return end_o <= t < end_o + d

    def _net_revenue_rule(_m):
        # 1) Revenue = sum_{j,s,t} pi_svc[s,t] * c[j,s,t] * Delta_t
        # Note: Delta_t = 1 hour, so we sum c[j,s,t] directly
        revenue = 0.0
        for t in _m.T:
            for s in _m.S:
                for j in _m.J:
                    revenue += _m.pi_svc[j, s, t] * _m.c[j, s, t]

        # 2) Electricity cost
        # Dynamic cost: sum_{s,t in full horizon} pi_ele[s,t] * PUE[s] * (P_busy[s]-P_idle[s]) * L[s,t]/L_bar[s] * Delta_t
        # Idle cost: sum_{s,t in original horizon only} pi_ele[s,t] * PUE[s] * P_idle[s] * Delta_t
        energy_cost = 0.0
        for t in _m.T:
            for s in _m.S:
                price = float(bidding_price[s][t])  # $/MWh
                # Dynamic power component
                # Note: L_bar[s] is always > 0, so we can safely divide
                utilization = _m.JobLoad[s, t] / _m.L_bar[s]
                dynamic_power = _m.pue[s] * (_m.Pbusy[s] - _m.Pidle[s]) * utilization
                energy_cost += price * dynamic_power  # MW * $/MWh = $ (Delta_t = 1 hour)
                
                # Idle power component (only in original window)
                if t < original_window_end:
                    idle_power = _m.pue[s] * _m.Pidle[s]
                    energy_cost += price * idle_power

        # 3) Reallocation penalty: rho * sum r[j,s,t]
        # Always include in objective, even if coefficient is 0.0
        realloc_penalty = 0.0
        for t in _m.T:
            if t != _m.T.first():
                for s in _m.S:
                    for j in _m.J:
                        realloc_penalty += reallocation_penalty * _m.r[j, s, t]

        # 4) Late-delivery penalty: eta * sum_{j,s,t in T_late[j]} c[j,s,t] * Delta_t
        # Always include in objective, even if coefficient is 0.0
        late_penalty = 0.0
        for t in _m.T:
            for s in _m.S:
                for j in _m.J:
                    if _is_late_delivery(j, t):
                        late_penalty += late_delivery_penalty * _m.c[j, s, t]

        # 5) Ramping penalty on excess ramp g (only if g exists; see RampingPos/Neg): γ·Σ g² or γ·Σ g
        ramp_charge = 0.0
        if include_ramping:
            for t in _m.T:
                if t != _m.T.first():
                    for s in _m.S:
                        if ramp_linear:
                            ramp_charge += ramping_charge * _m.g[s, t]
                        else:
                            ramp_charge += ramping_charge * (_m.g[s, t] * _m.g[s, t])

        # 6) Unfinished-work penalty: sum_j phi_j * (W[j] - sum_{s,t} c[j,s,t] * Delta_t)
        unfinished_penalty = 0.0
        for j in _m.J:
            total_delivered = sum(sum(_m.c[j, s, t] for s in _m.S) for t in _m.T)
            unfinished_work = _m.W[j] - total_delivered
            unfinished_penalty += _m.phi_unfinished[j] * unfinished_work

        return revenue - energy_cost - realloc_penalty - late_penalty - ramp_charge - unfinished_penalty

    def _load_balance_rule(_m):
        # Minimize the sum of squared differences between each site's load and the average load
        load_imbalance = 0.0
        for t in _m.T:
            total_load = sum(_m.JobLoad[s, t] for s in _m.S)
            avg_load = total_load / len(_m.S) if len(_m.S) > 0 else 0.0
            for s in _m.S:
                diff = _m.JobLoad[s, t] - avg_load
                load_imbalance += diff * diff
        return load_imbalance

    if load_balance_only:
        m.LoadBalance = pyo.Objective(rule=_load_balance_rule, sense=pyo.minimize)
    else:
        m.NetRevenue = pyo.Objective(rule=_net_revenue_rule, sense=pyo.maximize)

    # --------------------------------------------------------------
    # 11. Model stats (before solve — available as soon as the Pyomo model is built)
    # --------------------------------------------------------------
    var_count = 0
    binary_var_count = 0
    integer_var_count = 0
    continuous_var_count = 0
    for var_obj in m.component_objects(pyo.Var, active=True):
        for var in var_obj.values():
            var_count += 1
            if var.domain == pyo.Binary:
                binary_var_count += 1
            elif var.domain == pyo.Integers or var.domain == pyo.NonNegativeIntegers:
                integer_var_count += 1
            else:
                continuous_var_count += 1

    constraint_count = 0
    for constr_obj in m.component_objects(pyo.Constraint, active=True):
        constraint_count += len(list(constr_obj.values()))

    expression_count = 0
    for expr_obj in m.component_objects(pyo.Expression, active=True):
        expression_count += len(list(expr_obj.values()))

    t_build = time.perf_counter()
    if print_model_stats_before_solve:
        tag = f"[{model_stats_label}] " if model_stats_label else ""
        _print_to_console(
            f"[Model]{tag}var_count={var_count}, constraint_count={constraint_count}, "
            f"expr_count={expression_count} "
            f"(binary={binary_var_count}, integer={integer_var_count}, continuous={continuous_var_count}) "
            f"- starting solver"
        )

    # --------------------------------------------------------------
    # 12. Solve
    # --------------------------------------------------------------
    name = str(solver_name).lower()
    # gurobi_direct builds the Gurobi model in-process and skips the LP-file
    # write/read path (that path OOM'd on 4-site future models).
    if name in ("gurobi", "gurobi_direct"):
        solver = pyo.SolverFactory("gurobi_direct")
        merged_opts = dict(solver_options) if solver_options else {}
        if include_ramping and not ramp_linear:
            merged_opts.setdefault("NonConvex", 2)
        for _k, _v in merged_opts.items():
            solver.options[str(_k)] = _v
    else:
        solver = pyo.SolverFactory(solver_name)

    res = solver.solve(m, tee=tee)
    t_solve = time.perf_counter()

    # --------------------------------------------------------------
    # 13. Collect results
    # --------------------------------------------------------------
    out = {
        "model": m,
        "result": res,
        "x": {},
        "c": {},
        "realloc": {},
        "ramp": {},
        "build_seconds": t_build - t_start,
        "solve_seconds": t_solve - t_build,
        "total_seconds": t_solve - t_start,
        "var_count": var_count,
        "constraint_count": constraint_count,
        "expression_count": expression_count,
        "binary_var_count": binary_var_count,
        "integer_var_count": integer_var_count,
        "continuous_var_count": continuous_var_count,
        "ramping_penalty_mode": effective_ramp_mode,
    }

    # Extract x[j,s,t]
    for j in J:
        for s in S:
            for t in T_list:
                out["x"][(j, s, t)] = float(pyo.value(m.x[j, s, t]))

    # Extract c[j,s,t]
    for j in J:
        for s in S:
            for t in T_list:
                out["c"][(j, s, t)] = float(pyo.value(m.c[j, s, t]))

    # Extract r[j,s,t] (reallocation)
    for j in J:
        for s in S:
            for t in T_list:
                if t != T_list[0]:
                    out["realloc"][(j, s, t)] = float(pyo.value(m.r[j, s, t]))
                else:
                    out["realloc"][(j, s, t)] = 0.0

    # Extract g[s,t] (ramping); omitted when gamma=0
    if include_ramping:
        for s in S:
            for t in T_list:
                if t != T_list[0]:
                    out["ramp"][(s, t)] = float(pyo.value(m.g[s, t]))
                else:
                    out["ramp"][(s, t)] = 0.0
    else:
        for s in S:
            for t in T_list:
                out["ramp"][(s, t)] = 0.0

    # Calculate summary statistics
    total_realloc = sum(out["realloc"].values())
    total_ramp = sum(out["ramp"].values())
    
    # Calculate delivered work per job and termination statistics
    TOLERANCE = 1e-5
    job_delivered_work = {}  # {j: delivered_work}
    total_delivered_work = 0.0
    total_unfinished_work = 0.0
    fully_served_count = 0
    partially_served_count = 0
    zero_served_count = 0
    
    for j in J:
        delivered_work_j = sum(out["c"].get((j, s, t), 0.0) for s in S for t in T_list)
        job_delivered_work[j] = delivered_work_j
        total_delivered_work += delivered_work_j
        
        unfinished_work_j = job_work[j] - delivered_work_j
        total_unfinished_work += unfinished_work_j
        
        # Classify job termination status
        if delivered_work_j >= job_work[j] - TOLERANCE:
            fully_served_count += 1
        elif delivered_work_j <= TOLERANCE:
            zero_served_count += 1
        else:
            partially_served_count += 1
    
    # Late-delivered work: same index set as eta penalty (end_orig <= t < end_orig + Delta_j)
    total_late_work = 0.0
    for j in J:
        end_o = int(job_end_orig.get(j, len(T_list)))
        d = int(slack_delta_by_job.get(j, 0))
        for s in S:
            for t in T_list:
                if end_o <= t < end_o + d:
                    total_late_work += out["c"].get((j, s, t), 0.0)

    # Power by site
    power_by_site = {}
    for s in S:
        for t in T_list:
            power_by_site[(s, t)] = float(pyo.value(m.Power[s, t]))

    out["total_realloc"] = total_realloc
    out["total_ramp"] = total_ramp
    out["total_required_work"] = total_required_work
    out["total_delivered_work"] = total_delivered_work
    out["total_unfinished_work"] = total_unfinished_work
    out["min_completion_fraction"] = (
        float(min_completion_fraction) if allow_termination and min_completion_fraction is not None else 1.0
    )
    out["service_completion_fraction"] = (
        total_delivered_work / total_required_work if total_required_work > 0.0 else 1.0
    )
    out["total_late_work"] = total_late_work
    out["fully_served_count"] = fully_served_count
    out["partially_served_count"] = partially_served_count
    out["zero_served_count"] = zero_served_count
    out["job_delivered_work"] = job_delivered_work
    out["power_by_site"] = power_by_site

    # Realized penalty costs ($) at the optimal primal — same structure as _net_revenue_rule (not part of energy_cost)
    t_first = T_list[0]
    penalty_cost_realloc = 0.0
    for t in T_list:
        if t == t_first:
            continue
        for s in S:
            for j in J:
                penalty_cost_realloc += reallocation_penalty * float(pyo.value(m.r[j, s, t]))

    penalty_cost_late_delivery = 0.0
    for t in T_list:
        for s in S:
            for j in J:
                if _is_late_delivery(j, t):
                    penalty_cost_late_delivery += late_delivery_penalty * float(pyo.value(m.c[j, s, t]))

    penalty_cost_ramping = 0.0
    if include_ramping:
        for t in T_list:
            if t == t_first:
                continue
            for s in S:
                gv = float(pyo.value(m.g[s, t]))
                if ramp_linear:
                    penalty_cost_ramping += ramping_charge * gv
                else:
                    penalty_cost_ramping += ramping_charge * (gv * gv)

    penalty_cost_unfinished = 0.0
    for j in J:
        delivered_j = sum(float(pyo.value(m.c[j, s, t])) for s in S for t in T_list)
        unfinished_j = float(job_work[j]) - delivered_j
        penalty_cost_unfinished += phi_by_job[j] * unfinished_j

    penalty_cost_total = (
        penalty_cost_realloc + penalty_cost_late_delivery + penalty_cost_ramping + penalty_cost_unfinished
    )
    out["penalty_cost_realloc"] = penalty_cost_realloc
    out["penalty_cost_late_delivery"] = penalty_cost_late_delivery
    out["penalty_cost_ramping"] = penalty_cost_ramping
    out["penalty_cost_unfinished"] = penalty_cost_unfinished
    out["penalty_cost_total"] = penalty_cost_total

    return out
