"""Capacity-proportional FCFS baseline (price-agnostic, within-operator).

Strict no-backfill dispatch:
  - Jobs ordered by (release/start_hour_rel, job_id).
  - Only the global queue head may start; if no site has free capacity for its
    nominal X_bar, wait (do not skip to a later job).
  - Among feasible sites, assign to the lowest current used_cpu / N_cpu
    (capacity-proportional); ties broken by site id.
  - Once started: fixed site, fixed X_bar allocation each hour until work is
    delivered or the original window ends. No migration, no price response.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class FCFSDiagnostics:
    operator_id: str
    season: str
    n_jobs: int
    total_required_work: float = 0.0
    total_delivered_work: float = 0.0
    service_completion_fraction: float = 0.0
    n_capacity_violations: int = 0
    n_window_infeasible: int = 0
    n_unassigned: int = 0
    capacity_violations: list[dict] = field(default_factory=list)
    window_infeasible: list[dict] = field(default_factory=list)
    unassigned: list[dict] = field(default_factory=list)
    site_peak_util: dict[str, float] = field(default_factory=dict)
    site_r_max: float = 0.0
    energy_cost_usd: float = 0.0
    total_energy_mwh: float = 0.0
    label: str = "capacity-proportional FCFS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_name": self.label,
            "operator_id": self.operator_id,
            "season": self.season,
            "n_jobs": self.n_jobs,
            "total_required_work": self.total_required_work,
            "total_delivered_work": self.total_delivered_work,
            "service_completion_fraction": self.service_completion_fraction,
            "n_capacity_violations": self.n_capacity_violations,
            "n_window_infeasible": self.n_window_infeasible,
            "n_unassigned": self.n_unassigned,
            "capacity_violations": self.capacity_violations[:50],
            "window_infeasible": self.window_infeasible[:50],
            "unassigned": self.unassigned[:50],
            "site_peak_util": self.site_peak_util,
            "R_max_site": self.site_r_max,
            "energy_cost_usd": self.energy_cost_usd,
            "total_energy_mwh": self.total_energy_mwh,
            "meets_full_work_target": self.service_completion_fraction >= 1.0 - 1e-9,
            "meets_95pct_work_floor": self.service_completion_fraction >= 0.95 - 1e-9,
        }


def _job_work(job, cpu_per_task: float = 1.0) -> tuple[float, float, int, int]:
    start = int(getattr(job, "start_hour_rel", 0))
    end = int(getattr(job, "end_hour_rel", 0))
    runtime = max(0, end - start)
    x_bar = float(job.cpu_total) / float(cpu_per_task)
    w = x_bar * float(runtime)
    return w, x_bar, start, end


def simulate_capacity_proportional_fcfs(
    *,
    system_data: dict,
    jobs: dict,
    H: int,
    operator_id: str = "",
    season: str = "",
    cpu_per_task: float = 1.0,
    kappa_high: float = 1.0,
) -> tuple[dict[str, Any], FCFSDiagnostics]:
    """Return (result_dict compatible with E02 export path, diagnostics)."""
    S = sorted(int(s) for s in system_data["dc"]["sites"])
    N_cpu = {int(s): float(system_data["dc"]["N_cpu"][s]) for s in S}
    P_busy = {int(s): float(system_data["dc"]["capacity_MW"][s]) for s in S}
    P_idle = {int(s): float(system_data["dc"]["Pfixed_MW"][s]) for s in S}
    pue = {
        int(s): float(system_data["dc"].get("PUE", {}).get(s, 1.0)) for s in S
    }
    prices = {
        int(s): [float(v) for v in system_data["dc"]["day_ahead_prices"][s][:H]]
        for s in S
    }
    for s in S:
        if len(prices[s]) < H:
            pad = prices[s][-1] if prices[s] else 0.0
            prices[s] = prices[s] + [pad] * (H - len(prices[s]))

    ordered = sorted(
        ((int(jid), job) for jid, job in jobs.items()),
        key=lambda kv: (
            int(getattr(kv[1], "start_hour_rel", 0)),
            kv[0],
        ),
    )

    diag = FCFSDiagnostics(
        operator_id=operator_id,
        season=season,
        n_jobs=len(ordered),
        total_required_work=0.0,
        total_delivered_work=0.0,
    )

    job_meta: dict[int, dict] = {}
    for jid, job in ordered:
        w, x_bar, start, end = _job_work(job, cpu_per_task)
        diag.total_required_work += w
        job_meta[jid] = {
            "W": w,
            "X_bar": x_bar,
            "start": start,
            "end": end,
            "job_type": str(getattr(job, "job_type", "Spot")),
        }
        # Instantaneous demand exceeds every site → permanently unassigned.
        if not any(N_cpu[s] + 1e-12 >= x_bar for s in S):
            diag.n_unassigned += 1
            diag.unassigned.append(
                {
                    "job_id": jid,
                    "X_bar": x_bar,
                    "reason": "X_bar exceeds all site N_cpu",
                }
            )

    waiting = [jid for jid, _ in ordered if jid not in {u["job_id"] for u in diag.unassigned}]
    # running: jid -> (site, remaining_work)
    running: dict[int, tuple[int, float]] = {}
    site_of: dict[int, int | None] = {u["job_id"]: None for u in diag.unassigned}
    x: dict[tuple[int, int, int], float] = {}
    c: dict[tuple[int, int, int], float] = {}
    wait_idx = 0

    for t in range(H):
        used = {s: 0.0 for s in S}

        # 1) Continue already-started jobs (fixed rate, fixed site).
        finished: list[int] = []
        for jid, (site, rem) in list(running.items()):
            meta = job_meta[jid]
            if not (meta["start"] <= t < meta["end"]):
                # Window ended while unfinished → leave remaining as infeasible.
                continue
            need = rem / max(kappa_high, 1e-12)
            alloc = min(meta["X_bar"], need, N_cpu[site] - used[site])
            if alloc < 0:
                alloc = 0.0
            if alloc > 1e-12:
                x[(jid, site, t)] = alloc
                delivered = alloc * kappa_high
                c[(jid, site, t)] = delivered
                rem -= delivered
                used[site] += alloc
            running[jid] = (site, rem)
            if rem <= 1e-12:
                finished.append(jid)
        for jid in finished:
            del running[jid]

        # 2) Strict FCFS: only start the current head if some site can host it.
        while wait_idx < len(waiting):
            jid = waiting[wait_idx]
            meta = job_meta[jid]
            if t < meta["start"]:
                break  # head not released yet
            if t >= meta["end"]:
                # Missed entire window without starting
                site_of[jid] = None
                diag.n_window_infeasible += 1
                diag.window_infeasible.append(
                    {
                        "job_id": jid,
                        "remaining_work": meta["W"],
                        "W": meta["W"],
                        "window": [meta["start"], meta["end"]],
                        "reason": "never_started",
                    }
                )
                wait_idx += 1
                continue

            x_bar = meta["X_bar"]
            candidates = [s for s in S if (N_cpu[s] - used[s]) + 1e-12 >= x_bar]
            if not candidates:
                break  # head blocked → no backfill

            def _score(s: int) -> tuple[float, int]:
                return ((used[s] + x_bar) / max(N_cpu[s], 1e-12), s)

            site = min(candidates, key=_score)
            alloc = min(x_bar, meta["W"] / max(kappa_high, 1e-12))
            x[(jid, site, t)] = alloc
            delivered = alloc * kappa_high
            c[(jid, site, t)] = delivered
            used[site] += alloc
            rem = meta["W"] - delivered
            site_of[jid] = site
            if rem > 1e-12:
                running[jid] = (site, rem)
            wait_idx += 1

        for s in S:
            if used[s] > N_cpu[s] + 1e-6:
                diag.n_capacity_violations += 1
                diag.capacity_violations.append(
                    {"site": s, "t": t, "used": used[s], "N_cpu": N_cpu[s]}
                )

    # Jobs still running or never finished after horizon
    for jid, (site, rem) in running.items():
        if rem > 1e-6:
            diag.n_window_infeasible += 1
            diag.window_infeasible.append(
                {
                    "job_id": jid,
                    "site": site,
                    "remaining_work": rem,
                    "W": job_meta[jid]["W"],
                    "window": [job_meta[jid]["start"], job_meta[jid]["end"]],
                    "reason": "incomplete_at_horizon",
                }
            )
    while wait_idx < len(waiting):
        jid = waiting[wait_idx]
        meta = job_meta[jid]
        site_of[jid] = None
        diag.n_window_infeasible += 1
        diag.window_infeasible.append(
            {
                "job_id": jid,
                "remaining_work": meta["W"],
                "W": meta["W"],
                "window": [meta["start"], meta["end"]],
                "reason": "still_waiting",
            }
        )
        wait_idx += 1

    job_delivered = {j: 0.0 for j in job_meta}
    for (jid, _s, _t), cv in c.items():
        job_delivered[jid] += float(cv)
    # Cap at W
    for jid in job_delivered:
        job_delivered[jid] = min(job_delivered[jid], job_meta[jid]["W"])

    diag.total_delivered_work = float(sum(job_delivered.values()))
    diag.service_completion_fraction = float(
        diag.total_delivered_work / diag.total_required_work
        if diag.total_required_work > 0
        else 1.0
    )

    L = {(s, t): 0.0 for s in S for t in range(H)}
    for (jid, s, t), cv in c.items():
        L[(s, t)] = L.get((s, t), 0.0) + float(cv)

    power_by_site: dict[tuple[int, int], float] = {}
    energy_cost = 0.0
    energy_mwh = 0.0
    for s in S:
        l_bar = kappa_high * N_cpu[s]
        peak_u = 0.0
        series = []
        for t in range(H):
            util = (L[(s, t)] / l_bar) if l_bar > 0 else 0.0
            peak_u = max(peak_u, util)
            if util > 1.0 + 1e-6:
                diag.n_capacity_violations += 1
                diag.capacity_violations.append(
                    {"site": s, "t": t, "util": util, "type": "L_over_Lbar"}
                )
            p = pue[s] * (P_idle[s] + (P_busy[s] - P_idle[s]) * util)
            power_by_site[(s, t)] = p
            series.append(p)
            energy_cost += prices[s][t] * p
            energy_mwh += p
        diag.site_peak_util[str(s)] = peak_u
        rmax = 0.0
        for t in range(1, H):
            rmax = max(rmax, abs(series[t] - series[t - 1]))
        diag.site_r_max = max(diag.site_r_max, rmax)

    diag.energy_cost_usd = float(energy_cost)
    diag.total_energy_mwh = float(energy_mwh)

    class _Sol:
        termination_condition = "optimal"

    class _Res:
        solver = _Sol()

    out = {
        "model": None,
        "result": _Res(),
        "x": x,
        "c": c,
        "discard": {},
        "realloc": {},
        "ramp": {},
        "build_seconds": 0.0,
        "solve_seconds": 0.0,
        "total_seconds": 0.0,
        "var_count": 0,
        "constraint_count": 0,
        "expression_count": 0,
        "binary_var_count": 0,
        "integer_var_count": 0,
        "continuous_var_count": 0,
        "job_delivered_work": job_delivered,
        "total_delivered_work": diag.total_delivered_work,
        "total_unfinished_work": float(
            diag.total_required_work - diag.total_delivered_work
        ),
        "total_realloc": 0.0,
        "total_ramp": 0.0,
        "total_late_work": 0.0,
        "penalty_cost_realloc": 0.0,
        "penalty_cost_late_delivery": 0.0,
        "penalty_cost_ramping": 0.0,
        "penalty_cost_unfinished": 0.0,
        "penalty_cost_total": 0.0,
        "power_by_site": power_by_site,
        "site_assignment": {str(k): v for k, v in site_of.items()},
        "_baseline_energy_cost": diag.energy_cost_usd,
        "_fcfs_diagnostics": diag.as_dict(),
        "_baseline_name": "capacity-proportional FCFS",
    }
    return out, diag


def write_fcfs_diagnostics(path: Path, diag: FCFSDiagnostics | dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = diag.as_dict() if isinstance(diag, FCFSDiagnostics) else diag
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
