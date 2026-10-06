"""Aggregate (continuous) DC schedulers — AGG_CONT / AGG_ENERGY / AGG_POOL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pyomo.environ as pyo
from pyomo.opt import SolverFactory

from experiments.lib.solve.aggregate.power_io import Profile

EnergyMode = Literal["none", "site", "operator"]


@dataclass
class AggregateSpec:
    sites: list[int]
    hours: int
    prices: dict[int, list[float]]  # $/MWh per site per hour
    p_min: dict[int, float]
    p_max: dict[int, float]
    energy_mode: EnergyMode = "none"
    energy_target_mwh: dict[int, float] | None = None  # per-site (site mode)
    operator_sites: dict[str, list[int]] | None = None
    operator_energy_mwh: dict[str, float] | None = None  # operator mode


def solve_aggregate(spec: AggregateSpec, *, solver: str = "gurobi") -> tuple[Profile, dict]:
    S = spec.sites
    T = spec.hours
    m = pyo.ConcreteModel(name="aggregate_dc")
    m.S = pyo.Set(initialize=S)
    m.T = pyo.RangeSet(0, T - 1)

    def _bounds(_m, s, _t):
        return (spec.p_min[s], spec.p_max[s])

    m.P = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals, bounds=_bounds)

    if spec.energy_mode == "site" and spec.energy_target_mwh is not None:

        def energy_site(_m, s):
            return sum(m.P[s, t] for t in m.T) == spec.energy_target_mwh[s]

        m.energy_site = pyo.Constraint(m.S, rule=energy_site)

    if spec.energy_mode == "operator" and spec.operator_sites and spec.operator_energy_mwh:

        def energy_op(_m, op):
            sites = spec.operator_sites[op]
            return sum(m.P[s, t] for s in sites for t in m.T) == spec.operator_energy_mwh[op]

        m.energy_op = pyo.Constraint(list(spec.operator_sites.keys()), rule=energy_op)

    def obj_rule(_m):
        return sum(spec.prices[s][t] * m.P[s, t] for s in m.S for t in m.T)

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    opt = SolverFactory(solver)
    res = opt.solve(m, tee=False)
    status = str(res.solver.termination_condition)
    if status not in {"optimal", "feasible"}:
        raise RuntimeError(f"aggregate {spec.energy_mode} solver status: {status}")

    profile: Profile = {s: {} for s in S}
    for s in S:
        for t in range(T):
            # A zero price leaves P free inside its bounds. The lower bound
            # is still optimal, and it is the value used when the solver
            # returns no number for that hour.
            val = m.P[s, t].value
            if val is None:
                val = spec.p_min[s]
                m.P[s, t].set_value(val)
            profile[s][t] = float(val)

    model_name = {
        "none": "AGG_CONT",
        "site": "AGG_ENERGY",
        "operator": "AGG_POOL",
    }[spec.energy_mode]

    meta = {
        "solver_status": status,
        "objective_usd": float(
            sum(spec.prices[s][t] * profile[s][t] for s in S for t in range(T))
        ),
        "model": model_name,
        "energy_mode": spec.energy_mode,
        "hourly_ramp_limit": False,
    }
    return profile, meta
