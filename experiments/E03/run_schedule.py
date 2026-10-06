"""
Texas 7-site E01 flexibility contribution study (CPU model).

Uses experiments prepared data only:
  - build_operator_system_data()  -> dc capacity / N_cpu / nodal LMP CSVs
  - operator trace job CSVs under experiments/data/jobs/by_trace/

Does NOT import legacy data.data_14bus or IEEE14 assets.
"""

from __future__ import annotations

import copy
import csv
import gc
import io
import random
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from experiments.lib.paths import package_root

REPO_ROOT = package_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.lib.texas_config import (  # noqa: E402
    get_min_service_completion_fraction,
    get_operator_job_csv,
    get_operator_sites,
    get_trace_window,
)
from experiments.lib.solve.system_data import build_operator_system_data  # noqa: E402
from experiments.E03.model import solve_data_center_block_jobs_CPU  # noqa: E402
from experiments.lib.job_sampling import parse_job_id_mod  # noqa: E402
from experiments.lib.solve.calibration import (  # noqa: E402
    apply_job_workload_scale,
    calibrate_trace_for_operator,
)
from experiments.lib.solve.solver_report import _extract_solver_info, _is_feasible, _safe_value  # noqa: E402
from experiments.lib.solve.job_loading import load_eval_block_jobs
from experiments.lib.solve.fcfs import (  # noqa: E402
    simulate_capacity_proportional_fcfs,
    write_fcfs_diagnostics,
)
from experiments.lib.solve.bar_figure import make_categorical_bar_figure  # noqa: E402
import pyomo.environ as pyo  # noqa: E402


@dataclass(frozen=True)
class TexasE01Config:
    operator_id: str
    season: str  # winter | summer
    results_root: Path
    days: int = 7
    spot_extra_hours: int = 24
    job_sample_rate: str = "10%"
    job_types: tuple[str, ...] = ("HP", "Spot")
    convert_all_hp_to_spot: bool = True
    min_task_duration_hours: int | None = None
    max_task_duration_hours: int | None = None
    scenario_names: tuple[str, ...] | None = None
    make_plots: bool = True
    gurobi_options: dict | None = None


_CFG: TexasE01Config | None = None


# ============================================================================
# CONFIGURATION PARAMETERS - Modify all parameters here
# ============================================================================

# --- Run parameters (defaults; overridden per run via TexasE01Config) ---
JOB_TYPES = ["HP", "Spot"]
JOB_SAMPLE_RATE = "10%"
DAYS = 7
START_HOUR = 0
SPOT_EXTRA_HOURS = 24
MIN_SUBMIT_TIME_SEC = -1
MAX_SUBMIT_TIME_SEC: float | None = None
HP_TO_SPOT_SUBMIT_TIME_SEC = 0
CONVERT_ALL_HP_TO_SPOT = True
MIN_TASK_DURATION_HOURS: int | None = None
MAX_TASK_DURATION_HOURS: int | None = None

# --- Model Configuration ---
SOLVER_NAME = "gurobi"
# Gurobi only: passed to solve_data_center_block_jobs_CPU to limit RAM / runtime on large MILPs.
# Set to None to disable (strict optimality, higher memory use).
# MIPGap: stop when |bestbound-incumbent|/|incumbent| <= this (e.g. 0.005 = 0.5%).
# TimeLimit: seconds per solve (returns best incumbent if hit).
# Threads: fewer threads lowers peak memory from concurrent LP relaxations.
# NodefileStart: spill B&B nodes to disk when queue exceeds this many GB (slower, less RAM).
#
# Ramp tolerance in MW. A scenario uses this unless it sets ramping_tolerance_MW.
# The penalty is ramping_charge times the sum of squared excess ramp. Hour 0 is unconstrained.
DEFAULT_RAMPING_TOLERANCE_MW = 0.0

GUROBI_OPTIONS: dict | None = {
    "MIPGap": 0.002,
    "TimeLimit": 28800,  # 8 hours per scenario solve
    "Threads": 16
}
REALLOCATION_PENALTY = 0.0  # rho: default 0.0
LATE_DELIVERY_PENALTY = 0.0  # eta: default 0.0
RAMPING_CHARGE = 0.0  # gamma: default 0.0
UNFINISHED_WORK_PENALTY = 0.0  # phi: default 0.0 (penalty for incomplete service/termination)

# --- Price Discount Configuration ---
NUM_DISCOUNTED_JOBS = 30  # Number of jobs to randomly select for price discount
PRICE_DISCOUNT_FACTOR = 0.1  # Price discount factor (0.1 means 10% of original price)
RANDOM_SEED = 42  # Fixed random seed for reproducibility
REMOVE_DISCOUNTED_JOBS = False  # If True, remove discounted jobs before optimization; If False, include them with discount price

# --- Output Configuration ---
LOG_DIR = "experiments/E01/E02_flexibility_contributions/results"
JOB_CSV_PATH = ""
MAKE_PLOTS = True

# --- Console Progress / Heartbeat ---
SHOW_PROGRESS = True
HEARTBEAT_SECONDS = 60  # print "still running" every N seconds during a long solve
# If you want to see Gurobi output in console (very verbose), set this to True:
SHOW_SOLVER_OUTPUT = False

# Ramping penalty on excess ramp g: "quadratic" = γ·Σ g² (MIQP when γ>0); "linear" = γ·Σ g (MILP when γ>0).
RAMPING_PENALTY_MODE = "quadratic"  # "quadratic" | "linear"

# ============================================================================


def _require_cfg() -> TexasE01Config:
    if _CFG is None:
        raise RuntimeError("TexasE01Config not set; call run_texas_contribution_study() first")
    return _CFG


def _apply_run_config(cfg: TexasE01Config) -> None:
    global _CFG, LOG_DIR, JOB_CSV_PATH, JOB_SAMPLE_RATE, DAYS, START_HOUR
    global SPOT_EXTRA_HOURS, MIN_SUBMIT_TIME_SEC, MAX_SUBMIT_TIME_SEC
    global CONVERT_ALL_HP_TO_SPOT, MIN_TASK_DURATION_HOURS, MAX_TASK_DURATION_HOURS, JOB_TYPES
    global GUROBI_OPTIONS, MAKE_PLOTS

    _CFG = cfg
    MAKE_PLOTS = cfg.make_plots
    if cfg.gurobi_options is not None:
        GUROBI_OPTIONS = dict(cfg.gurobi_options)
    trace = get_trace_window(cfg.operator_id)
    LOG_DIR = str(cfg.results_root)
    JOB_CSV_PATH = str(get_operator_job_csv(cfg.operator_id))
    JOB_SAMPLE_RATE = cfg.job_sample_rate
    DAYS = cfg.days
    SPOT_EXTRA_HOURS = cfg.spot_extra_hours
    START_HOUR = trace["day_start"] * 24
    MIN_SUBMIT_TIME_SEC = trace["min_submit_time_sec"]
    MAX_SUBMIT_TIME_SEC = trace["max_submit_time_sec"]
    CONVERT_ALL_HP_TO_SPOT = cfg.convert_all_hp_to_spot
    MIN_TASK_DURATION_HOURS = cfg.min_task_duration_hours
    MAX_TASK_DURATION_HOURS = cfg.max_task_duration_hours
    JOB_TYPES = list(cfg.job_types)


def _active_scenarios(cfg: TexasE01Config) -> list[Scenario]:
    if not cfg.scenario_names:
        return list(SCENARIOS)
    by_name = {sc.name: sc for sc in SCENARIOS}
    missing = [name for name in cfg.scenario_names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown scenario name(s): {missing}; expected subset of {[s.name for s in SCENARIOS]}")
    return [by_name[name] for name in cfg.scenario_names]


def _load_system_data_and_jobs(
    days: int,
    spot_extra_hours: int,
    job_id_mod: int | None,
) -> tuple[dict, dict, dict, int, int]:
    cfg = _require_cfg()
    h_original = 24 * days
    system_data = build_operator_system_data(
        cfg.operator_id, cfg.season, days=days, spot_extra_hours=spot_extra_hours
    )
    h = len(system_data["time"]["periods"])

    jobs, job_active, _trace = load_eval_block_jobs(
        cfg.operator_id,
        horizon_hours=h,
        spot_extra_hours=spot_extra_hours,
        job_id_mod=job_id_mod,
        job_type_filter=list(JOB_TYPES) if JOB_TYPES else None,
        original_window_end=h_original,
        hp_to_spot_submit_time_sec=HP_TO_SPOT_SUBMIT_TIME_SEC,
    )
    cal = calibrate_trace_for_operator(
        cfg.operator_id,
        jobs,
        job_active,
        h_original,
        job_sample_mod=job_id_mod,
        n_cpu_by_site=system_data["dc"]["N_cpu"],
    )
    system_data["dc"]["N_cpu"] = cal.n_cpu_by_site
    apply_job_workload_scale(jobs, cal.job_scale)
    return system_data, jobs, job_active, h, h_original


@dataclass(frozen=True)
class Scenario:
    name: str
    use_original_window: bool  # If True, restrict to original window only (no slack window, no late delivery)
    enable_reallocation: bool  # If False, disable CPU reallocation (fixed site assignment)
    reallocation_penalty: float
    late_delivery_penalty: float  # Penalty for work delivered after original deadline (in slack window)
    ramping_charge: float
    unfinished_work_penalty: float  # Penalty for incomplete service/termination (phi)
    ramping_tolerance_MW: float = DEFAULT_RAMPING_TOLERANCE_MW  # δ_s: free |ΔP| band (MW) before γ·g²


SCENARIOS: list[Scenario] = [
    # Ordered by flexibility level (lowest to highest):
    # 1. baseline: capacity-proportional FCFS (price-agnostic; strict no backfill)
    Scenario("baseline", use_original_window=True,
             enable_reallocation=False, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 2. initial_assignment: original window only, fixed site assignment (no CPU reallocation)
    #    Flexibility: price-aware in time (within original window), but fixed in space
    Scenario("initial_assignment", use_original_window=True,
             enable_reallocation=False, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 3. termination: original window only, fixed site assignment, allow service termination
    #    Flexibility: time-aware (within original window) + service termination, but fixed in space
    Scenario("termination", use_original_window=True,
             enable_reallocation=False, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 4. slack: slack window allowed (late delivery), fixed site assignment (no CPU reallocation)
    #    Flexibility: price-aware in time (extended to slack window), but fixed in space
    Scenario("slack", use_original_window=False,
             enable_reallocation=False, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 5. reallocation: original window only, allow CPU reallocation
    #    Flexibility: price-aware in both time (within original window) and space (CPU reallocation)
    Scenario("reallocation", use_original_window=True,
             enable_reallocation=True, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 6. reallocation_termination: original window only, allow CPU reallocation, allow service termination
    #    Flexibility: time-aware (within original window) + space-aware (CPU reallocation) + service termination
    Scenario("reallocation_termination", use_original_window=True,
             enable_reallocation=True, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 7. reallocation_slack: slack window allowed (late delivery), allow CPU reallocation
    #    Flexibility: price-aware in time (extended to slack window) and space (CPU reallocation)
    Scenario("reallocation_slack", use_original_window=False,
             enable_reallocation=True, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 8. slack_termination: slack window allowed, fixed site assignment, allow service termination
    #    Flexibility: time-aware (extended to slack window) + service termination, but fixed in space
    Scenario("slack_termination", use_original_window=False,
             enable_reallocation=False, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
    # 9. reallocation_slack_termination: slack window allowed, allow CPU reallocation, allow service termination
    #    Flexibility: maximum flexibility (time + space + service termination)
    Scenario("reallocation_slack_termination", use_original_window=False,
             enable_reallocation=True, reallocation_penalty=0.0, late_delivery_penalty=0.0, ramping_charge=0.0, unfinished_work_penalty=0.0),
]


def _effective_late_penalty_coeff(sc: Scenario) -> float:
    """Eta only applies when slack / late-delivery window is allowed (matches run_once)."""
    return sc.late_delivery_penalty if not sc.use_original_window else 0.0


def _penalty_activation_and_billed_line(sc: Scenario, res: dict) -> str:
    """Nonzero coeff => term is in objective; billed_usd is realized $ at the solution."""
    rho_c = sc.reallocation_penalty
    eta_c = _effective_late_penalty_coeff(sc)
    gam_c = sc.ramping_charge
    phi_c = sc.unfinished_work_penalty
    pr = float(res.get("penalty_cost_realloc", 0.0))
    pl = float(res.get("penalty_cost_late_delivery", 0.0))
    pga = float(res.get("penalty_cost_ramping", 0.0))
    pu = float(res.get("penalty_cost_unfinished", 0.0))
    pt = float(res.get("penalty_cost_total", 0.0))
    eps = 1e-4

    def _onoff(coeff: float, billed: float) -> str:
        if coeff <= 0.0:
            return "off"
        if billed > eps:
            return f"on billed={billed:.2f}"
        return "on billed=0"

    return (
        "[Penalties] "
        f"rho({_onoff(rho_c, pr)}) | "
        f"eta({_onoff(eta_c, pl)}) | "
        f"gamma({_onoff(gam_c, pga)}) | "
        f"phi({_onoff(phi_c, pu)}) | "
        f"total_usd={pt:.2f}"
    )


def _get_termination_stats(res: dict, jobs: dict) -> dict:
    """Get termination statistics from optimization results."""
    job_delivered_work = res.get("job_delivered_work", {})
    job_work = {}
    for j, job in jobs.items():
        start_hour_rel = int(getattr(job, "start_hour_rel", 0))
        end_hour_rel = int(getattr(job, "end_hour_rel", 0))
        runtime = max(0, end_hour_rel - start_hour_rel)
        requested_cpu = float(job.cpu_total / 1.0)  # cpu_per_task = 1.0
        job_work[j] = requested_cpu * runtime
    
    TOLERANCE = 1e-5
    fully_served = []
    partially_served = []
    zero_served = []
    
    for j in jobs.keys():
        delivered = job_delivered_work.get(j, 0.0)
        required = job_work.get(j, 0.0)
        
        if delivered >= required - TOLERANCE:
            fully_served.append(j)
        elif delivered <= TOLERANCE:
            zero_served.append(j)
        else:
            partially_served.append(j)
    
    return {
        "fully_served_count": len(fully_served),
        "partially_served_count": len(partially_served),
        "zero_served_count": len(zero_served),
        "fully_served_jobs": sorted(fully_served),
        "partially_served_jobs": sorted(partially_served),
        "zero_served_jobs": sorted(zero_served),
    }


def filter_jobs_by_duration(jobs, min_duration_hours=None, max_duration_hours=None):
    """
    Filter jobs to only include those with duration >= min_duration_hours and < max_duration_hours.
    
    Args:
        jobs: Dictionary of {job_id: BlockJob}
        min_duration_hours: Minimum duration in hours (None means no lower limit)
        max_duration_hours: Maximum duration in hours (None means no upper limit)
    
    Returns:
        dict: Filtered jobs dictionary
    """
    filtered_jobs = {}
    for job_id, job in jobs.items():
        duration = job.end_hour_rel - job.start_hour_rel
        if min_duration_hours is None or duration >= min_duration_hours:
            if max_duration_hours is None or duration < max_duration_hours:
                filtered_jobs[job_id] = job
    return filtered_jobs


def calculate_baseline_energy_cost(data, initial_assignment_power_by_site, H):
    """
    Calculate baseline energy cost by evenly distributing total power across all sites.
    Includes both dynamic power cost and idle power cost (in original window only).
    
    Args:
        data: System data dictionary
        initial_assignment_power_by_site: {(s, t): power_MW} from initial_assignment scenario (in-memory data from current run)
        H: Number of time periods
    
    Returns:
        tuple: (total_energy_cost, calc_time)
    """
    t_start = time.perf_counter()
    
    S = list(data["dc"]["sites"])
    num_sites = len(S)
    T_list = list(range(H))
    
    # Get LMP prices
    bidding_price = data["dc"]["day_ahead_prices"]  # {s: [price_t]}
    
    # Get original window end (assuming 7 days = 168 hours for this calculation)
    # This should match the original window used in the model
    original_window_end = 24 * 7  # 7 days = 168 hours
    if H < original_window_end:
        original_window_end = H
    
    # Get power parameters
    P_idle = data["dc"]["Pfixed_MW"]  # {s: idle_power_MW}
    pue_dict = data["dc"].get("PUE", {s: 1.0 for s in S})
    
    # Calculate total power at each time
    total_power_by_time = {}  # {t: total_power_MW}
    for t in T_list:
        total_power = 0.0
        for s in S:
            total_power += initial_assignment_power_by_site.get((s, t), 0.0)
        total_power_by_time[t] = total_power
    
    # Calculate energy cost: each site gets 1/num_sites of total power, multiply by LMP
    # For baseline, we evenly distribute the total power across all sites
    # Each site has its own idle power, and the remaining is dynamic power
    total_energy_cost = 0.0
    for t in T_list:
        total_power_t = total_power_by_time[t]
        power_per_site = total_power_t / num_sites if num_sites > 0 else 0.0
        
        for s in S:
            lmp = bidding_price[s][t] if t < len(bidding_price[s]) else bidding_price[s][-1]
            pue = pue_dict.get(s, 1.0)
            P_idle_s = P_idle.get(s, 0.0)
            site_idle_power = pue * P_idle_s  # MW
            
            # For baseline: evenly distribute total power
            # Site total power = power_per_site
            # Site dynamic power = max(0, power_per_site - site_idle_power)
            site_dynamic_power = max(0.0, power_per_site - site_idle_power)
            
            # Dynamic power cost
            site_dynamic_cost = site_dynamic_power * lmp  # MW * $/MWh = $
            total_energy_cost += site_dynamic_cost
            
            # Idle power cost (only in original window, same as other scenarios)
            if t < original_window_end:
                site_idle_cost = site_idle_power * lmp  # MW * $/MWh = $
                total_energy_cost += site_idle_cost
    
    t_end = time.perf_counter()
    
    return total_energy_cost, t_end - t_start


def _calculate_power_by_site(res: dict, system_data: dict) -> tuple[dict, dict]:
    """Calculate power consumption for each site and time period.
    
    Returns:
        tuple: (power_by_site, total_power)
        - power_by_site: {(s, t): power_MW} where power_MW is the power at site s and time t
        - total_power: {t: total_power_MW} where total_power_MW is the sum of power across all sites at time t
    """
    # Prefer schedule-derived power (FCFS / exported MILP); never invent uniform splits.
    if "power_by_site" in res and res["power_by_site"]:
        power_by_site = res["power_by_site"]
        total_power = {}
        for (s, t), power in power_by_site.items():
            total_power[t] = total_power.get(t, 0.0) + power
        return power_by_site, total_power

    m = res.get("model")
    if m is None:
        return {}, {}
    
    # Otherwise calculate from model
    S = list(m.S)
    T_list = list(m.T)
    
    power_by_site = {}
    total_power = {}
    
    for t in T_list:
        total_power_t = 0.0
        for s in S:
            try:
                power_s_t = float(pyo.value(m.Power[s, t]))
            except Exception:
                power_s_t = 0.0
            power_by_site[(s, t)] = power_s_t
            total_power_t += power_s_t
        total_power[t] = total_power_t
    
    return power_by_site, total_power


def calculate_job_revenue_and_energy_cost_CPU(res, system_data, jobs):
    """
    Calculate revenue and energy cost for each job (CPU model version).
    
    Args:
        res: Result dictionary from solve_data_center_block_jobs_CPU
        system_data: System data dictionary
        jobs: Dictionary of jobs {job_id: BlockJob}
    
    Returns:
        tuple: (job_stats_dict, total_idle_cost)
        - job_stats_dict: {job_id: {"revenue": float, "dynamic_cost": float, "net_revenue": float}}
        - total_idle_cost: float, total idle power cost in original window
    """
    m = res["model"]
    S = list(m.S)
    T_list = list(m.T)
    J = list(m.J)
    
    # Get original window end time
    if len(J) > 0:
        job_end_orig = {}
        for j in J:
            try:
                job_end_orig[j] = int(pyo.value(m.job_end_orig[j]))
            except:
                job_end_orig[j] = int(m.job_end_orig[j])
        original_window_end = max(job_end_orig[j] for j in J) if job_end_orig else len(T_list)
    else:
        original_window_end = len(T_list)
    
    # Get price data
    bidding_price = system_data["dc"]["day_ahead_prices"]  # {s: [price_t]}
    P_idle = system_data["dc"]["Pfixed_MW"]
    P_busy = system_data["dc"]["capacity_MW"]
    N_cpu = system_data["dc"]["N_cpu"]
    pue_dict = system_data["dc"].get("PUE", {s: 1.0 for s in S})
    
    # Initialize results
    job_stats = {}
    for j in J:
        job_stats[j] = {
            "revenue": 0.0,
            "dynamic_cost": 0.0,
            "net_revenue": 0.0,
        }
    
    # Track total idle power cost separately
    total_idle_cost = 0.0
    
    # 1. Calculate revenue for each job: sum_{s,t} pi_svc[j,s,t] * c[j,s,t]
    for j in J:
        for t in T_list:
            for s in S:
                try:
                    c_val = float(pyo.value(m.c[j, s, t]))
                except:
                    c_val = 0.0
                
                if c_val > 0:
                    # Get service price
                    try:
                        price = float(pyo.value(m.pi_svc[j, s, t]))
                    except:
                        price = 0.0
                    
                    # Apply job-level price discount if available
                    discount_factor = 1.0
                    if hasattr(m, 'pi_job_discount'):
                        try:
                            discount_factor = float(pyo.value(m.pi_job_discount[j]))
                        except:
                            discount_factor = 1.0
                    
                    job_stats[j]["revenue"] += discount_factor * price * c_val
    
    # 2. Calculate energy cost for each job
    # Allocate based on delivered compute service c[j,s,t]
    job_c_usage = {}  # {(j, s, t): c_value}
    total_c_usage = {}  # {(s, t): total_c}
    
    for s in S:
        for t in T_list:
            total_c_usage[(s, t)] = 0.0
            for j in J:
                try:
                    c_val = float(pyo.value(m.c[j, s, t]))
                except:
                    c_val = 0.0
                job_c_usage[(j, s, t)] = c_val
                total_c_usage[(s, t)] += c_val
    
    # Allocate energy cost to each job based on c[j,s,t] proportion
    for s in S:
        pue = pue_dict.get(s, 1.0)
        P_idle_s = P_idle.get(s, 0.0)
        P_busy_s = P_busy.get(s, 0.0)
        N_cpu_s = N_cpu.get(s, 1.0)
        kappa_high_s = float(pyo.value(m.kappa_high[s])) if hasattr(m, 'kappa_high') else 1.0
        L_bar_s = kappa_high_s * N_cpu_s
        
        for t in T_list:
            price_mwh = float(bidding_price[s][t])  # $/MWh
            
            # Total dynamic power at this site and time
            total_load = total_c_usage.get((s, t), 0.0)
            if total_load > 0 and L_bar_s > 0:
                utilization = total_load / L_bar_s
                total_dynamic_power = pue * (P_busy_s - P_idle_s) * utilization  # MW
            else:
                total_dynamic_power = 0.0
            
            # Allocate dynamic power cost to each job based on c[j,s,t] proportion
            for j in J:
                c_usage = job_c_usage.get((j, s, t), 0.0)
                if total_load > 0 and total_dynamic_power > 0:
                    job_dynamic_power = total_dynamic_power * (c_usage / total_load)
                    job_stats[j]["dynamic_cost"] += price_mwh * job_dynamic_power  # $/MWh * MW = $
            
            # Calculate idle power cost separately (only in original window)
            if t < original_window_end:
                total_idle_power = pue * P_idle_s  # MW
                idle_cost = price_mwh * total_idle_power  # $/MWh * MW = $
                total_idle_cost += idle_cost
    
    # 3. Calculate net revenue (revenue - dynamic_cost only, idle cost is separate)
    for j in J:
        job_stats[j]["net_revenue"] = job_stats[j]["revenue"] - job_stats[j]["dynamic_cost"]
    
    return job_stats, total_idle_cost


def _prepare_prices(system_data: dict, H: int, job_id_mod: int | None, jobs: dict | None = None, load_balance_only: bool = False) -> tuple[dict, dict]:
    """Prepare pi_HP / pi_SP as {site: {t: price}} scaled by sampling rate.
    
    Args:
        system_data: System data dictionary
        H: Number of time periods
        job_id_mod: Job sampling rate modifier
        jobs: Dictionary of jobs {job_id: BlockJob} (optional, not used but kept for compatibility)
        load_balance_only: If True, use uniform prices across all sites (for load balancing only, no price optimization)
    
    Returns:
        tuple: (pi_HP, pi_SP)
        - pi_HP: {site: {t: price}} - same for all HP jobs
        - pi_SP: {site: {t: price}} - same for all SP jobs
    """
    price_scale = float(job_id_mod) if job_id_mod is not None and job_id_mod > 1 else 1.0
    T_list = list(range(H))
    pi_HP = {}
    pi_SP = {}
    
    if load_balance_only:
        # For load balancing only: use average prices across all sites
        sites = system_data["dc"]["sites"]
        
        # Calculate average HP price
        hp_prices = [float(system_data["dc"].get("HP_prices", {}).get(s, 0.0)) for s in sites]
        avg_hp = sum(hp_prices) / len(hp_prices) if hp_prices else 0.0
        avg_hp *= price_scale
        
        # Calculate average SP price
        sp_prices = [float(system_data["dc"].get("SP_prices", {}).get(s, 0.0)) for s in sites]
        avg_sp = sum(sp_prices) / len(sp_prices) if sp_prices else 0.0
        avg_sp *= price_scale
        
        # Use same price for all sites
        for s in sites:
            pi_HP[s] = {t: avg_hp for t in T_list}
            pi_SP[s] = {t: avg_sp for t in T_list}
    else:
        # Original behavior: site-specific prices
        for s in system_data["dc"]["sites"]:
            hp = float(system_data["dc"].get("HP_prices", {}).get(s, 0.0)) * price_scale
            pi_HP[s] = {t: hp for t in T_list}
        
        for s in system_data["dc"]["sites"]:
            sp = float(system_data["dc"].get("SP_prices", {}).get(s, 0.0)) * price_scale
            pi_SP[s] = {t: sp for t in T_list}
    
    return pi_HP, pi_SP


def _print_progress_to_console(msg: str) -> None:
    """Visible even when run_once runs under redirect_stdout (e.g. contribution study)."""
    try:
        print(msg, file=getattr(sys, "__stdout__", None) or sys.stdout, flush=True)
    except Exception:
        pass


def run_once(days: int, spot_extra_hours: int, scenario: Scenario, job_id_mod: int | None):
    system_data, jobs, job_active, H, H_original = _load_system_data_and_jobs(
        days, spot_extra_hours, job_id_mod
    )
    if SHOW_PROGRESS:
        cfg = _require_cfg()
        _print_progress_to_console(
            f"[Model] operator={cfg.operator_id!r} season={cfg.season!r} "
            f"scenario={scenario.name!r}: MILP (H={H} hours, sites={system_data['dc']['sites']})..."
        )

    # Convert all HP jobs to Spot jobs if enabled
    if CONVERT_ALL_HP_TO_SPOT:
        converted_count = 0
        for j, job in jobs.items():
            if str(getattr(job, "job_type", "HP")).lower() == "hp":
                job.job_type = "Spot"
                converted_count += 1
        if SHOW_PROGRESS and converted_count > 0:
            print(f"  Converted {converted_count} HP jobs to Spot jobs", flush=True)

    # Filter jobs by duration if specified
    jobs_before_filter = len(jobs)
    if MIN_TASK_DURATION_HOURS is not None or MAX_TASK_DURATION_HOURS is not None:
        jobs = filter_jobs_by_duration(jobs, min_duration_hours=MIN_TASK_DURATION_HOURS, max_duration_hours=MAX_TASK_DURATION_HOURS)
        jobs_after_filter = len(jobs)
        if SHOW_PROGRESS:
            if MIN_TASK_DURATION_HOURS is None:
                filter_desc = f"duration < {MAX_TASK_DURATION_HOURS} hours"
            elif MAX_TASK_DURATION_HOURS is None:
                filter_desc = f"duration >= {MIN_TASK_DURATION_HOURS} hours"
            else:
                filter_desc = f"{MIN_TASK_DURATION_HOURS} <= duration < {MAX_TASK_DURATION_HOURS} hours"
            print(f"  Filtered jobs: {jobs_before_filter} -> {jobs_after_filter} ({filter_desc})", flush=True)
        
        # Update job_active to only include filtered jobs
        job_active = {
            (j, t): val for (j, t), val in job_active.items()
            if j in jobs
        }

    # Prepare prices
    load_balance_only = (scenario.name == "baseline")
    pi_HP = None
    pi_SP = None
    
    # Identify discounted jobs
    discounted_job_ids_list = []
    if NUM_DISCOUNTED_JOBS > 0 and PRICE_DISCOUNT_FACTOR < 1.0:
        random.seed(RANDOM_SEED)
        job_ids_list = list(jobs.keys())
        if len(job_ids_list) > 0:
            num_to_discount = min(NUM_DISCOUNTED_JOBS, len(job_ids_list))
            discounted_job_ids_list = random.sample(job_ids_list, num_to_discount)
            if SHOW_PROGRESS:
                print(f"[Discount] Identified {len(discounted_job_ids_list)} discounted jobs: {discounted_job_ids_list[:5]}...")
    
    # Filter discounted jobs based on REMOVE_DISCOUNTED_JOBS configuration
    jobs_before_discount_filter = len(jobs)
    if REMOVE_DISCOUNTED_JOBS and discounted_job_ids_list:
        jobs = {j: job for j, job in jobs.items() if j not in discounted_job_ids_list}
        job_active = {
            (j, t): val for (j, t), val in job_active.items()
            if j not in discounted_job_ids_list
        }
        if SHOW_PROGRESS:
            print(f"[Filter] REMOVE_DISCOUNTED_JOBS=True: Excluding {len(discounted_job_ids_list)} discounted jobs from optimization", flush=True)
            print(f"[Filter] Jobs after excluding discounted: {jobs_before_discount_filter} -> {len(jobs)}", flush=True)
    elif discounted_job_ids_list:
        if SHOW_PROGRESS:
            print(f"[Filter] REMOVE_DISCOUNTED_JOBS=False: Keeping {len(discounted_job_ids_list)} discounted jobs in optimization with {PRICE_DISCOUNT_FACTOR}x price", flush=True)
    
    # Prepare pi_job_discount
    if REMOVE_DISCOUNTED_JOBS:
        pi_job_discount = {job_id: 1.0 for job_id in jobs.keys()}
    else:
        pi_job_discount = {}
        for job_id in jobs.keys():
            if job_id in discounted_job_ids_list:
                pi_job_discount[job_id] = PRICE_DISCOUNT_FACTOR
            else:
                pi_job_discount[job_id] = 1.0

    _gurobi_opts = (
        GUROBI_OPTIONS if str(SOLVER_NAME).lower() == "gurobi" and GUROBI_OPTIONS else None
    )

    # For baseline: capacity-proportional FCFS (price-agnostic; site-specific power).
    if scenario.name == "baseline":
        t0 = time.perf_counter()
        res, fcfs_diag = simulate_capacity_proportional_fcfs(
            system_data=system_data,
            jobs=jobs,
            H=H,
            operator_id=str(getattr(_require_cfg(), "operator_id", "")),
            season=str(getattr(_require_cfg(), "season", "")),
            cpu_per_task=1.0,
            kappa_high=1.0,
        )
        res["total_seconds"] = float(time.perf_counter() - t0)
        res["_discounted_job_ids"] = discounted_job_ids_list
        res["_fcfs_diagnostics"] = fcfs_diag.as_dict()
        # Prepare uniform service prices only for revenue bookkeeping (not used in dispatch).
        pi_HP, pi_SP = _prepare_prices(system_data, H, job_id_mod, jobs, load_balance_only=False)
        if SHOW_PROGRESS:
            d = fcfs_diag.as_dict()
            print(
                f"[FCFS] completion={d['service_completion_fraction']:.4f} "
                f"window_infeas={d['n_window_infeasible']} unassigned={d['n_unassigned']} "
                f"energy_cost=${d['energy_cost_usd']:.2f}",
                flush=True,
            )
    else:
        # For non-baseline scenarios, prepare prices here
        pi_HP, pi_SP = _prepare_prices(system_data, H, job_id_mod, jobs, load_balance_only=load_balance_only)
        
        # Modify job_active based on scenario
        modified_job_active = job_active.copy()
        if scenario.use_original_window:
            # Restrict to original window only (no slack window, no late delivery)
            for j, job in jobs.items():
                start_orig = int(getattr(job, "start_hour_rel", 0))
                end_orig = int(getattr(job, "end_hour_rel", 0))
                for t in range(H):
                    if not (start_orig <= t < end_orig):
                        modified_job_active[(j, t)] = 0.0
        
        # Late delivery penalty: applies to work delivered after original deadline
        # If use_original_window=False, slack window is allowed, so late_delivery_penalty applies
        # If use_original_window=True, no slack window, so late_delivery_penalty is irrelevant (set to 0)
        effective_late_penalty = scenario.late_delivery_penalty if not scenario.use_original_window else 0.0
        
        # Determine if termination is allowed based on scenario name
        allow_termination = "termination" in scenario.name.lower()
        min_completion_fraction = (
            get_min_service_completion_fraction() if allow_termination else None
        )

        res = solve_data_center_block_jobs_CPU(
            data=system_data,
            jobs=jobs,
            job_active=modified_job_active,
            cpu_per_task=1.0,
            solver_name=SOLVER_NAME,
            tee=bool(SHOW_SOLVER_OUTPUT),
            reallocation_penalty=scenario.reallocation_penalty,
            late_delivery_penalty=effective_late_penalty,
            ramping_charge=scenario.ramping_charge,
            ramping_tolerance_MW=scenario.ramping_tolerance_MW,
            unfinished_work_penalty=scenario.unfinished_work_penalty,
            enable_reallocation=scenario.enable_reallocation,
            allow_termination=allow_termination,  # Allow termination only if scenario name contains "termination"
            min_completion_fraction=min_completion_fraction,
            pi_HP=pi_HP,
            pi_SP=pi_SP,
            pi_job_discount=pi_job_discount,
            load_balance_only=load_balance_only,
            solver_options=_gurobi_opts,
            print_model_stats_before_solve=SHOW_PROGRESS,
            ramping_penalty_mode=RAMPING_PENALTY_MODE,
        )

    res["jobs"] = jobs
    res["system_data"] = system_data
    if NUM_DISCOUNTED_JOBS > 0 and PRICE_DISCOUNT_FACTOR < 1.0:
        res["_discounted_job_ids"] = discounted_job_ids_list

    # compute job stats
    if scenario.name == "baseline":
        baseline_energy_cost = float(res.pop("_baseline_energy_cost", 0.0))
        S = list(system_data["dc"]["sites"])
        T_list = list(range(H))
        bidding_price = system_data["dc"]["day_ahead_prices"]
        P_idle = system_data["dc"]["Pfixed_MW"]
        P_busy = system_data["dc"]["capacity_MW"]
        N_cpu = system_data["dc"]["N_cpu"]
        pue_dict = system_data["dc"].get("PUE", {s: 1.0 for s in S})
        original_window_end = min(24 * DAYS, H)

        # Service revenue from schedule c and site service prices (bookkeeping only).
        hp_price = system_data["dc"].get("HP_prices", {})
        sp_price = system_data["dc"].get("SP_prices", {})
        c_map = res.get("c", {})
        job_stats = {j: {"revenue": 0.0, "dynamic_cost": 0.0, "net_revenue": 0.0} for j in jobs}
        job_c_usage = {}
        total_c_usage = {(s, t): 0.0 for s in S for t in T_list}
        for (j, s, t), cv in c_map.items():
            if j not in job_stats:
                continue
            cv = float(cv)
            job_c_usage[(j, s, t)] = cv
            total_c_usage[(s, t)] = total_c_usage.get((s, t), 0.0) + cv
            jt = str(getattr(jobs[j], "job_type", "Spot")).lower()
            unit = float(hp_price.get(s, 0.0) if jt == "hp" else sp_price.get(s, 0.0))
            discount = float(pi_job_discount.get(j, 1.0))
            job_stats[j]["revenue"] += discount * unit * cv

        baseline_idle_cost = 0.0
        for s in S:
            pue = float(pue_dict.get(s, 1.0))
            P_idle_s = float(P_idle.get(s, 0.0))
            P_busy_s = float(P_busy.get(s, 0.0))
            N_cpu_s = float(N_cpu.get(s, 1.0))
            L_bar = 1.0 * N_cpu_s
            for t in T_list:
                price_mwh = float(bidding_price[s][t] if t < len(bidding_price[s]) else bidding_price[s][-1])
                total_load = total_c_usage.get((s, t), 0.0)
                if total_load > 0 and L_bar > 0:
                    util = total_load / L_bar
                    total_dyn = pue * (P_busy_s - P_idle_s) * util
                else:
                    total_dyn = 0.0
                for j in jobs:
                    cu = job_c_usage.get((j, s, t), 0.0)
                    if total_load > 0 and total_dyn > 0:
                        job_stats[j]["dynamic_cost"] += price_mwh * total_dyn * (cu / total_load)
                if t < original_window_end:
                    baseline_idle_cost += price_mwh * (pue * P_idle_s)

        for j in jobs:
            job_stats[j]["net_revenue"] = job_stats[j]["revenue"] - job_stats[j]["dynamic_cost"]

        total_revenue = sum(float(v["revenue"]) for v in job_stats.values())
        total_dynamic = sum(float(v["dynamic_cost"]) for v in job_stats.values())
        # Prefer schedule-integrated energy (idle + dynamic) from FCFS power trajectory.
        if abs(baseline_energy_cost) < 1e-9:
            baseline_energy_cost = total_dynamic + baseline_idle_cost
        baseline_net_revenue = total_revenue - baseline_energy_cost

        # Site ramp from actual FCFS power (not uniform split).
        power_by_site = res.get("power_by_site", {})
        baseline_total_ramp = 0.0
        for s in S:
            for t in T_list:
                if t == 0:
                    continue
                baseline_total_ramp += abs(
                    float(power_by_site.get((s, t), 0.0)) - float(power_by_site.get((s, t - 1), 0.0))
                )
        res["total_ramp"] = baseline_total_ramp
        res["job_stats"] = job_stats
        res["total_idle_cost"] = baseline_idle_cost
        res["_baseline_total_energy_cost"] = baseline_energy_cost
        res["_baseline_net_revenue"] = baseline_net_revenue
    else:
        job_stats, total_idle_cost = calculate_job_revenue_and_energy_cost_CPU(res, system_data, jobs)
        res["job_stats"] = job_stats
        res["total_idle_cost"] = total_idle_cost

    # job counts
    hp_count = sum(1 for j, jb in jobs.items() if str(getattr(jb, "job_type", "HP")).lower() == "hp")
    spot_count = sum(1 for j, jb in jobs.items() if str(getattr(jb, "job_type", "HP")).lower() == "spot")
    total_jobs = len(jobs)
    res["job_counts"] = {"HP": hp_count, "Spot": spot_count, "Total": total_jobs}

    return res


def _run_with_heartbeat(fn, label: str):
    """
    Run a long task while printing periodic heartbeat messages to console.
    Returns: (result, elapsed_seconds)
    """
    start = time.perf_counter()
    stop_evt = threading.Event()

    def _beat():
        last_print = 0.0
        while not stop_evt.is_set():
            now = time.perf_counter()
            if last_print == 0.0 or (now - last_print) >= float(HEARTBEAT_SECONDS):
                elapsed = now - start
                print(f"[Heartbeat] {label} running... elapsed={elapsed/60.0:.1f} min", flush=True)
                last_print = now
            stop_evt.wait(1.0)

    t = threading.Thread(target=_beat, daemon=True)
    if SHOW_PROGRESS and HEARTBEAT_SECONDS is not None and HEARTBEAT_SECONDS > 0:
        t.start()

    try:
        res = fn()
        elapsed = time.perf_counter() - start
        return res, elapsed
    finally:
        stop_evt.set()


def run_texas_contribution_study(cfg: TexasE01Config) -> Path:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    _apply_run_config(cfg)
    job_id_mod_int, pct_str = parse_job_id_mod(JOB_SAMPLE_RATE)
    season = cfg.season
    if season not in ("winter", "summer"):
        raise ValueError(f"season must be winter or summer, got {season!r}")

    trace = get_trace_window(cfg.operator_id)
    sites_meta = get_operator_sites(cfg.operator_id)
    # Create season subdirectory for this run (winter or summer)
    log_dir_path = Path(LOG_DIR)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    run_dir = log_dir_path / season
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp for file naming (but use season for directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    log_path = run_dir / f"flexibility_contribution_CPU_{season}_{timestamp}.log"
    csv_path = run_dir / f"flexibility_contribution_CPU_{season}_{timestamp}.csv"

    rows: list[dict] = []

    # Load jobs once to get total count
    system_data_sample = build_operator_system_data(
        cfg.operator_id, season, days=DAYS, spot_extra_hours=SPOT_EXTRA_HOURS
    )
    H_total_sample = len(system_data_sample["time"]["periods"])
    H_original_sample = 24 * DAYS
    H_sample = min(H_original_sample + int(SPOT_EXTRA_HOURS), H_total_sample)
    jobs_sample, _, _ = load_eval_block_jobs(
        cfg.operator_id,
        horizon_hours=H_sample,
        spot_extra_hours=SPOT_EXTRA_HOURS,
        job_id_mod=job_id_mod_int,
        job_type_filter=list(JOB_TYPES) if JOB_TYPES else None,
        original_window_end=H_original_sample,
        hp_to_spot_submit_time_sec=HP_TO_SPOT_SUBMIT_TIME_SEC,
    )
    if CONVERT_ALL_HP_TO_SPOT:
        for j, job in jobs_sample.items():
            if str(getattr(job, "job_type", "HP")).lower() == "hp":
                job.job_type = "Spot"
    
    if MIN_TASK_DURATION_HOURS is not None or MAX_TASK_DURATION_HOURS is not None:
        jobs_sample = filter_jobs_by_duration(jobs_sample, min_duration_hours=MIN_TASK_DURATION_HOURS, max_duration_hours=MAX_TASK_DURATION_HOURS)
    
    total_jobs_all = len(jobs_sample)
    hp_count_all = sum(1 for j, jb in jobs_sample.items() if str(getattr(jb, "job_type", "HP")).lower() == "hp")
    spot_count_all = sum(1 for j, jb in jobs_sample.items() if str(getattr(jb, "job_type", "HP")).lower() == "spot")
    
    del jobs_sample

    if SHOW_PROGRESS:
        print("=" * 90)
        print(f"[Texas E01 Flex CPU] operator={cfg.operator_id} season={season}")
        site_names = [str(s["name"]).replace("\u2733", "*") for s in sites_meta]
        print(f"  sites ({len(sites_meta)}): {site_names}")
        print(
            f"  trace segment: day {trace['segment_day_start']}-{trace['segment_day_end']} "
            f"(jobs: first {trace['job_window_days']}d -> day {trace['day_start']}-{trace['day_end']}) "
            f"START_HOUR={START_HOUR}"
        )
        print(f"- DAYS={DAYS}, SPOT_EXTRA_HOURS={SPOT_EXTRA_HOURS}")
        print(f"- JOB_SAMPLE_RATE={JOB_SAMPLE_RATE}, JOB_CSV_PATH={JOB_CSV_PATH}, JOB_TYPES={JOB_TYPES}")
        if MIN_TASK_DURATION_HOURS is not None or MAX_TASK_DURATION_HOURS is not None:
            if MIN_TASK_DURATION_HOURS is None:
                print(f"- Task Duration Filter: duration < {MAX_TASK_DURATION_HOURS} hours")
            elif MAX_TASK_DURATION_HOURS is None:
                print(f"- Task Duration Filter: duration >= {MIN_TASK_DURATION_HOURS} hours")
            else:
                print(f"- Task Duration Filter: {MIN_TASK_DURATION_HOURS} <= duration < {MAX_TASK_DURATION_HOURS} hours")
        print(f"- Total Jobs: {total_jobs_all} (HP: {hp_count_all}, Spot: {spot_count_all})")
        active = _active_scenarios(cfg)
        print(f"- Scenarios={len(active)}: {[s.name for s in active]}")
        if cfg.scenario_names:
            print(f"- SMOKE / partial run: scenario_names={cfg.scenario_names}")
        print(f"- LMP: nodal CSVs under experiments/E01/data/lmp/")
        print(f"- Output dir: {LOG_DIR}")
        print("=" * 90, flush=True)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=== Texas E01 Flexibility Contribution Study (CPU Model) ===\n")
        f.write(f"Operator: {cfg.operator_id}\n")
        f.write(f"Season: {season}\n")
        f.write(
            f"Trace segment: days {trace['segment_day_start']}-{trace['segment_day_end']}, "
            f"job window: days {trace['day_start']}-{trace['day_end']} "
            f"(first {trace['job_window_days']}d), START_HOUR={START_HOUR}\n"
        )
        f.write(f"Timestamp: {timestamp}\n")
        f.write("Parameters:\n")
        f.write(f"  DAYS = {DAYS}\n")
        f.write(f"  JOB_SAMPLE_RATE = {JOB_SAMPLE_RATE}\n")
        f.write(f"  JOB_CSV_PATH = {JOB_CSV_PATH}\n")
        f.write(f"  JOB_TYPES = {JOB_TYPES}\n")
        f.write(f"  MIN_SUBMIT_TIME_SEC = {MIN_SUBMIT_TIME_SEC}\n")
        f.write(f"  HP_TO_SPOT_SUBMIT_TIME_SEC = {HP_TO_SPOT_SUBMIT_TIME_SEC}\n")
        f.write(f"  CONVERT_ALL_HP_TO_SPOT = {CONVERT_ALL_HP_TO_SPOT}\n")
        f.write(f"  SPOT_EXTRA_HOURS = {SPOT_EXTRA_HOURS}\n")
        f.write(f"  MIN_SERVICE_COMPLETION_FRACTION = {get_min_service_completion_fraction():.0%} (termination scenarios)\n")
        f.write(f"  LMP = nodal CSV (experiments/E01/data/lmp/)\n")
        if MIN_TASK_DURATION_HOURS is not None:
            f.write(f"  MIN_TASK_DURATION_HOURS = {MIN_TASK_DURATION_HOURS}\n")
        if MAX_TASK_DURATION_HOURS is not None:
            f.write(f"  MAX_TASK_DURATION_HOURS = {MAX_TASK_DURATION_HOURS}\n")
        if NUM_DISCOUNTED_JOBS > 0 and PRICE_DISCOUNT_FACTOR < 1.0:
            f.write(f"  NUM_DISCOUNTED_JOBS = {NUM_DISCOUNTED_JOBS}\n")
            f.write(f"  PRICE_DISCOUNT_FACTOR = {PRICE_DISCOUNT_FACTOR}\n")
            f.write(f"  RANDOM_SEED = {RANDOM_SEED}\n")
            f.write(f"  REMOVE_DISCOUNTED_JOBS = {REMOVE_DISCOUNTED_JOBS}\n")
        f.write("\n")
        f.write(f"Total Jobs: {total_jobs_all} (HP: {hp_count_all}, Spot: {spot_count_all})\n")
        if MIN_TASK_DURATION_HOURS is not None or MAX_TASK_DURATION_HOURS is not None:
            f.write("Task Duration Filter:\n")
            if MIN_TASK_DURATION_HOURS is None:
                f.write(f"  duration < {MAX_TASK_DURATION_HOURS} hours\n")
            elif MAX_TASK_DURATION_HOURS is None:
                f.write(f"  duration >= {MIN_TASK_DURATION_HOURS} hours\n")
            else:
                f.write(f"  {MIN_TASK_DURATION_HOURS} <= duration < {MAX_TASK_DURATION_HOURS} hours\n")
            f.write("\n")
        f.write("Scenarios (ordered by flexibility level, lowest to highest):\n")
        f.write("  1. baseline = capacity-proportional FCFS (price-agnostic; strict no backfill)\n")
        f.write("  2. initial_assignment = original window only, fixed site assignment (no CPU reallocation)\n")
        f.write("     Flexibility: time-aware (within original window), space-fixed\n")
        f.write("  3. termination = original window only, fixed site assignment, allow service termination\n")
        f.write("     Flexibility: time-aware (within original window) + service termination, space-fixed\n")
        f.write("  4. slack = slack window allowed (late delivery), fixed site assignment (no CPU reallocation)\n")
        f.write("     Flexibility: time-aware (extended to slack window), space-fixed\n")
        f.write("  5. reallocation = original window only, allow CPU reallocation\n")
        f.write("     Flexibility: time-aware (within original window) + space-aware (CPU reallocation)\n")
        f.write("  6. reallocation_termination = original window only, allow CPU reallocation, allow service termination\n")
        f.write("     Flexibility: time-aware (within original window) + space-aware (CPU reallocation) + service termination\n")
        f.write("  7. reallocation_slack = slack window allowed (late delivery), allow CPU reallocation\n")
        f.write("     Flexibility: time-aware (extended to slack window) + space-aware (CPU reallocation)\n")
        f.write("  8. slack_termination = slack window allowed, fixed site assignment, allow service termination\n")
        f.write("     Flexibility: time-aware (extended to slack window) + service termination, space-fixed\n")
        f.write("  9. reallocation_slack_termination = slack window allowed, allow CPU reallocation, allow service termination\n")
        f.write("     Flexibility: maximum (time + space + service termination)\n")
        f.write("\n")
        
        # Log data center capacity and power information
        system_data_sample = build_operator_system_data(
            cfg.operator_id, season, days=DAYS, spot_extra_hours=SPOT_EXTRA_HOURS
        )
        if job_id_mod_int is not None and job_id_mod_int > 1:
            scale = 1.0 / float(job_id_mod_int)
            system_data_sample["dc"]["N_cpu"] = {
                s: system_data_sample["dc"]["N_cpu"][s] * scale for s in system_data_sample["dc"]["N_cpu"]
            }
        
        f.write("Data Center Configuration:\n")
        sites = sorted(system_data_sample["dc"]["sites"])
        capacity_MW = system_data_sample["dc"]["capacity_MW"]
        Pfixed_MW = system_data_sample["dc"]["Pfixed_MW"]
        N_cpu = system_data_sample["dc"]["N_cpu"]
        PUE = system_data_sample["dc"].get("PUE", {s: 1.0 for s in sites})
        
        f.write(f"  {'Site':<6} {'Capacity_MW':<15} {'Idle_Power_MW':<18} {'N_CPU':<12} {'PUE':<8}\n")
        f.write(f"  {'-'*6} {'-'*15} {'-'*18} {'-'*12} {'-'*8}\n")
        for s in sites:
            f.write(f"  {s:<6} {capacity_MW.get(s, 0.0):<15.2f} {Pfixed_MW.get(s, 0.0):<18.2f} "
                   f"{N_cpu.get(s, 0.0):<12.0f} {PUE.get(s, 1.0):<8.2f}\n")
        f.write("\n")

        f.write(
            f"{'Scenario':<20} {'Status':<10} {'NetRevenue':<14} {'EnergyCost':<14} "
            f"{'Pen$rho':<12} {'Pen$eta':<12} {'Pen$gamma':<12} {'Pen$phi':<12} {'Pen$sum':<12} "
            f"{'DynCost':<12} {'IdleCost':<12} {'F/P/Z':<20} {'Realloc':<12} {'Ramp':<12} "
            f"{'HP':<4} {'Spot':<5} {'Total':<6} {'Build(s)':<9} {'Solve(s)':<9} "
            f"{'Vars':<10} {'Constr':<10}\n"
        )
        f.write(
            "EnergyCost = electricity only (dynamic+idle). Pen$* = realized penalty USD at optimum "
            "(not included in EnergyCost). NetRevenue = objective = revenue - EnergyCost - Pen$sum.\n"
        )
        f.write("-" * 220 + "\n")

        active_scenarios = _active_scenarios(cfg)
        # run all scenarios
        for idx, sc in enumerate(active_scenarios, start=1):
            captured = io.StringIO()
            if sc.name == "baseline":
                label = f"[{idx}/{len(active_scenarios)}] scenario={sc.name} (capacity-proportional FCFS)"
            else:
                label = f"[{idx}/{len(active_scenarios)}] scenario={sc.name} (slack={int(not sc.use_original_window)}, realloc={int(sc.enable_reallocation)}, termi={sc.unfinished_work_penalty})"

            if SHOW_PROGRESS:
                print(f"{label} START", flush=True)

            try:
                def _task():
                    with redirect_stdout(captured), redirect_stderr(captured):
                        return run_once(DAYS, SPOT_EXTRA_HOURS, sc, job_id_mod_int)

                res, elapsed_wall = _run_with_heartbeat(_task, label=label)
            except Exception as e:
                print(f"{label} ERROR: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                cap = captured.getvalue()
                if cap:
                    print("[Captured output (stdout/stderr)]:", flush=True)
                    print(cap, flush=True)
                f.write(f"\n{label} ERROR: {e}\n")
                f.write(traceback.format_exc() + "\n")
                if cap:
                    f.write("\n[Captured output (stdout/stderr)]:\n")
                    f.write(cap + "\n")
                f.flush()
                raise

            model = res.get("model")
            result = res["result"]
            system_data = res["system_data"]
            if sc.name == "baseline" and res.get("_baseline_name"):
                is_feasible = True
                solver_name, solver_status, termination_condition = (
                    "fcfs",
                    "ok",
                    "optimal",
                )
            else:
                is_feasible = _is_feasible(result)
                solver_name, solver_status, termination_condition = _extract_solver_info(result)
            
            # totals from job_stats
            job_stats = res.get("job_stats", {})
            total_idle_cost = float(res.get("total_idle_cost", 0.0))
            total_revenue = sum(float(v["revenue"]) for v in job_stats.values())
            total_dynamic_cost = sum(float(v["dynamic_cost"]) for v in job_stats.values())
            total_energy_cost = total_dynamic_cost + total_idle_cost
            
            # For baseline, use pre-calculated values
            if sc.name == "baseline":
                total_energy_cost = float(res.get("_baseline_total_energy_cost", total_energy_cost))
                net_rev = float(res.get("_baseline_net_revenue", total_revenue - total_energy_cost))
            else:
                if model is not None:
                    net_rev = _safe_value(model.NetRevenue, default=None)
                else:
                    net_rev = None
                if net_rev is None:
                    net_rev = total_revenue - total_energy_cost

            status = "Optimal" if (is_feasible and net_rev is not None) else ("Infeasible" if not is_feasible else "Unknown")
            net_rev_str = f"{net_rev:.6f}" if net_rev is not None else "N/A"
            
            f.write(f"  [Solver Status] Solver={solver_name}, Status={solver_status}, Termination={termination_condition}, Feasible={is_feasible}\n")

            termination_stats = _get_termination_stats(res, res.get("jobs", {}))
            fully_served_count = termination_stats["fully_served_count"]
            partially_served_count = termination_stats["partially_served_count"]
            zero_served_count = termination_stats["zero_served_count"]
            total_unfinished_work = float(res.get("total_unfinished_work", 0.0))
            total_delivered_work = float(res.get("total_delivered_work", 0.0))
            total_realloc = float(res.get("total_realloc", 0.0))
            total_ramp = float(res.get("total_ramp", 0.0))
            total_late_work = float(res.get("total_late_work", 0.0))
            hp_count = res.get("job_counts", {}).get("HP", 0)
            spot_count = res.get("job_counts", {}).get("Spot", 0)
            total_jobs = res.get("job_counts", {}).get("Total", 0)

            build_s = float(res.get("build_seconds", 0.0))
            solve_s = float(res.get("solve_seconds", 0.0))
            var_count_disp = int(res.get("var_count", 0) or 0)
            constraint_count_disp = int(res.get("constraint_count", 0) or 0)

            p_cr = float(res.get("penalty_cost_realloc", 0.0))
            p_cl = float(res.get("penalty_cost_late_delivery", 0.0))
            p_cg = float(res.get("penalty_cost_ramping", 0.0))
            p_cu = float(res.get("penalty_cost_unfinished", 0.0))
            p_ct = float(res.get("penalty_cost_total", 0.0))
            
            # Calculate power consumption by site (schedule-derived; no uniform split)
            power_by_site = {}
            total_power = {}
            if is_feasible:
                try:
                    power_by_site, total_power = _calculate_power_by_site(res, system_data)
                    if sc.name == "baseline" and res.get("total_ramp"):
                        total_ramp = float(res["total_ramp"])
                except Exception as e:
                    if SHOW_PROGRESS:
                        print(f"  Warning: Failed to calculate power consumption: {e}", flush=True)

            # Persist FCFS diagnostics next to power CSV
            if sc.name == "baseline" and res.get("_fcfs_diagnostics"):
                diag_path = run_dir / f"fcfs_diagnostics_{season}_{timestamp}.json"
                write_fcfs_diagnostics(diag_path, res["_fcfs_diagnostics"])
                if SHOW_PROGRESS:
                    print(f"  Saved FCFS diagnostics: {diag_path.name}", flush=True)
                d = res["_fcfs_diagnostics"]
                f.write(
                    f"  [FCFS] completion={d.get('service_completion_fraction', 0):.6f} "
                    f"window_infeas={d.get('n_window_infeasible', 0)} "
                    f"unassigned={d.get('n_unassigned', 0)} "
                    f"meets_full_work={d.get('meets_full_work_target', False)} "
                    f"meets_95pct={d.get('meets_95pct_work_floor', False)}\n"
                )

            f.write(
                f"{sc.name:<20} {status:<10} {net_rev_str:<14} {total_energy_cost:<14.2f} "
                f"{p_cr:<12.2f} {p_cl:<12.2f} {p_cg:<12.2f} {p_cu:<12.2f} {p_ct:<12.2f} "
                f"{total_dynamic_cost:<12.2f} {total_idle_cost:<12.2f} {fully_served_count:<6d}/{partially_served_count:<6d}/{zero_served_count:<6d} "
                f"{total_realloc:<12.2f} {total_ramp:<12.2f} {hp_count:<4d} {spot_count:<5d} "
                f"{total_jobs:<6d} {build_s:<9.2f} {solve_s:<9.2f} {var_count_disp:<10d} {constraint_count_disp:<10d}\n"
            )
            f.write(f"  {_penalty_activation_and_billed_line(sc, res)}\n")
            
            # Save power consumption to CSV file
            if power_by_site and total_power:
                power_filename = f"power_consumption_{sc.name}_{season}_{timestamp}.csv"
                power_path = run_dir / power_filename
                
                S_consistent = sorted(system_data["dc"]["sites"])
                if model is not None:
                    T_list = sorted(list(model.T))
                else:
                    T_list = sorted(list(total_power.keys()))
                
                with open(power_path, "w", encoding="utf-8", newline="") as pf:
                    writer = csv.writer(pf)
                    header = ["Time_Period"] + [f"Site{s}_Power_MW" for s in S_consistent] + ["Total_Power_MW"]
                    writer.writerow(header)
                    
                    for t in T_list:
                        row = [t]
                        for s in S_consistent:
                            power_s_t = power_by_site.get((s, t), 0.0)
                            row.append(f"{power_s_t:.6f}")
                        row.append(f"{total_power.get(t, 0.0):.6f}")
                        writer.writerow(row)
                
                if SHOW_PROGRESS:
                    print(f"  Saved power consumption: {power_filename}", flush=True)

            # Output flexibility-specific metrics only when the scenario has the corresponding flexibility
            # 1. Reallocation: output when enable_reallocation=True
            if sc.enable_reallocation:
                f.write(f"  Reallocation: {total_realloc:.2f}\n")
            
            # 2. Late delivery: output when use_original_window=False (slack window allowed)
            if not sc.use_original_window:
                f.write(f"  Delayed work: {total_late_work:.2f}\n")
            
            if "termination" in sc.name.lower():
                completion_pct = float(res.get("service_completion_fraction", 0.0)) * 100.0
                f.write(
                    f"  Delivered work: {total_delivered_work:.2f}, Unfinished work: {total_unfinished_work:.2f}, "
                    f"Completion: {completion_pct:.1f}% "
                    f"(min {float(res.get('min_completion_fraction', 1.0)):.0%})\n"
                )

            if SHOW_PROGRESS:
                cost_str = f"{total_energy_cost:.2f}"
                net_str = f"{net_rev:.2f}" if net_rev is not None else "N/A"
                print(
                    f"{label} DONE status={status}, NetRevenue={net_str}, EnergyCost={cost_str}, "
                    f"PenSum={p_ct:.2f}, "
                    f"var_count={var_count_disp}, constraint_count={constraint_count_disp}, "
                    f"elapsed_wall={elapsed_wall/60.0:.1f} min (build={build_s:.1f}s, solve={solve_s:.1f}s)",
                    flush=True,
                )
                print(f"  {_penalty_activation_and_billed_line(sc, res)}", flush=True)
                print(
                    f"  [Solver] {solver_name}, Status={solver_status}, Termination={termination_condition}, Feasible={is_feasible}",
                    flush=True,
                )

            rows.append(
                {
                    "scenario": sc.name,
                    "use_original_window": sc.use_original_window,
                    "enable_reallocation": sc.enable_reallocation,
                    "reallocation_penalty": sc.reallocation_penalty,
                    "late_delivery_penalty": sc.late_delivery_penalty,
                    "ramping_charge": sc.ramping_charge,
                    "ramping_penalty_mode": str(res.get("ramping_penalty_mode", RAMPING_PENALTY_MODE)),
                    "ramping_tolerance_MW": sc.ramping_tolerance_MW,
                    "unfinished_work_penalty": sc.unfinished_work_penalty,
                    "status": status,
                    "net_revenue": float(net_rev) if net_rev is not None else None,
                    "total_revenue": float(total_revenue),
                    "dynamic_cost": float(total_dynamic_cost),
                    "idle_cost": float(total_idle_cost),
                    "energy_cost": float(total_energy_cost),
                    "penalty_cost_realloc": p_cr,
                    "penalty_cost_late_delivery": p_cl,
                    "penalty_cost_ramping": p_cg,
                    "penalty_cost_unfinished": p_cu,
                    "penalty_cost_total": p_ct,
                    "fully_served_count": int(fully_served_count),
                    "partially_served_count": int(partially_served_count),
                    "zero_served_count": int(zero_served_count),
                    "total_delivered_work": float(total_delivered_work),
                    "total_unfinished_work": float(total_unfinished_work),
                    "total_required_work": float(res.get("total_required_work", 0.0)),
                    "service_completion_fraction": float(res.get("service_completion_fraction", 1.0)),
                    "min_completion_fraction": float(res.get("min_completion_fraction", 1.0)),
                    "total_realloc": float(total_realloc),
                    "total_ramp": float(total_ramp),
                    "hp_jobs": int(hp_count),
                    "spot_jobs": int(spot_count),
                    "total_jobs": int(total_jobs),
                    "var_count": int(res.get("var_count", 0)),
                    "constraint_count": int(res.get("constraint_count", 0)),
                    "build_seconds": build_s,
                    "solve_seconds": solve_s,
                }
            )

            f.flush()

            # Clean up large objects
            del model, result
            if "model" in res:
                del res["model"]
            if "result" in res:
                del res["result"]
            if "system_data" in res:
                del res["system_data"]
            if "jobs" in res:
                del res["jobs"]
            gc.collect()

        # contribution section (reference row: baseline if present, else first scenario)
        f.write("\n" + "=" * 130 + "\n")
        baseline = next((r for r in rows if r["scenario"] == "baseline"), None)
        if baseline is None:
            baseline = rows[0]
            f.write("CONTRIBUTIONS (Δ vs first scenario in run)\n")
        else:
            f.write("CONTRIBUTIONS (Δ vs baseline)\n")
        f.write("=" * 130 + "\n")

        base_cost = float(baseline["energy_cost"])
        base_net = float(baseline["net_revenue"]) if baseline["net_revenue"] is not None else None

        f.write(f"{'Scenario':<20} {'ΔEnergyCost':<14} {'ΔEnergyCost (%)':<18} {'ΔNetRevenue':<14}\n")
        f.write("-" * 70 + "\n")
        for r in rows:
            dcost = float(r["energy_cost"]) - base_cost
            dcost_pct = (dcost / base_cost * 100) if base_cost > 0 else 0.0
            if base_net is not None and r["net_revenue"] is not None:
                dnet = float(r["net_revenue"]) - base_net
                dnet_str = f"{dnet:<14.2f}"
            else:
                dnet_str = f"{'N/A':<14}"
            f.write(f"{r['scenario']:<20} {dcost:<14.2f} {dcost_pct:<18.3f} {dnet_str}\n")

    # write CSV (penalty columns mirror Scenario / solve_data_center_block_jobs_CPU)
    fieldnames = list(rows[0].keys()) + [
        "delta_energy_cost_vs_baseline",
        "delta_energy_cost_pct_vs_baseline",
        "delta_net_revenue_vs_baseline",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            dcost = float(r["energy_cost"]) - float(base_cost)
            dcost_pct = (dcost / float(base_cost) * 100) if base_cost > 0 else 0.0
            r2["delta_energy_cost_vs_baseline"] = dcost
            r2["delta_energy_cost_pct_vs_baseline"] = dcost_pct
            if base_net is not None and r["net_revenue"] is not None:
                r2["delta_net_revenue_vs_baseline"] = float(r["net_revenue"]) - float(base_net)
            else:
                r2["delta_net_revenue_vs_baseline"] = None
            w.writerow(r2)

    # Plots
    if MAKE_PLOTS:
        order = [sc.name for sc in SCENARIOS]
        make_categorical_bar_figure(
            file_path=str(csv_path),
            file_type="csv",
            x_column="scenario",
            y_column="delta_energy_cost_vs_baseline",
            order=order,
            x_label=r'$\mathrm{Scenario}$',
            y_label=r'$\mathrm{\Delta\ Energy\ Cost}$',
            rotate_xticks=0,
            show_values=False,
            save_pdf=False,
            output_file=str(run_dir / f"delta_energy_cost_CPU_{season}_{timestamp}.png"),
        )

        make_categorical_bar_figure(
            file_path=str(csv_path),
            file_type="csv",
            x_column="scenario",
            y_column="delta_net_revenue_vs_baseline",
            order=order,
            x_label=r'$\mathrm{Scenario}$',
            y_label=r'$\mathrm{\Delta\ Net\ Revenue}$',
            rotate_xticks=0,
            show_values=False,
            save_pdf=False,
            output_file=str(run_dir / f"delta_net_revenue_CPU_{season}_{timestamp}.png"),
        )

    print(f"[Done] All results saved to: {run_dir}")
    print(f"[Done] Log file: {log_path.name}")
    print(f"[Done] CSV file: {csv_path.name}")
    if MAKE_PLOTS:
        print(f"[Done] Plots saved to: {run_dir / f'delta_energy_cost_CPU_{season}_{timestamp}.png'}")
        print(f"[Done] Plots saved to: {run_dir / f'delta_net_revenue_CPU_{season}_{timestamp}.png'}")
    return run_dir


if __name__ == "__main__":
    import argparse

    from experiments.lib.solve.runner import run_operator_study

    parser = argparse.ArgumentParser(
        description="Job-level schedules. Evaluation is 168 h; the solve window adds 24 h."
    )
    parser.add_argument("--operator", required=True)
    parser.add_argument("--season", choices=("winter", "summer"), required=True)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["baseline", "reallocation_slack_termination"],
    )
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Original short check. It is not the manuscript solve.",
    )
    args = parser.parse_args()
    run_operator_study(
        args.operator,
        args.season,
        scenario_names=tuple(args.scenarios),
        results_root=args.results,
        smoke=args.smoke,
        make_plots=False,
    )
