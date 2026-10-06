"""Run E01 flexibility study for one operator × season."""

from __future__ import annotations

from pathlib import Path

from experiments.E03.run_schedule import (
    TexasE01Config,
    run_texas_contribution_study,
)
from experiments.lib.texas_config import (
    SEASONS,
    TEXAS_ROOT,
    get_operator_sites,
    get_trace_job_sample_rate,
    operator_ids,
)

FULL_RUN_GUROBI_OPTIONS = {
    "MIPGap": 0.002,
    "TimeLimit": 28800,
    "Threads": 16,
}

# Oracle ~40k+ jobs @ 100%: Gurobi LP reader OOM; config default is 5%.
ORACLE_GUROBI_OPTIONS = {
    "MIPGap": 0.002,
    "TimeLimit": 28800,
    "Threads": 1,
    "NodefileStart": 4.0,
    "Method": 1,
}

# Google 3-site (incl. 1 GW Goodnight): larger assignment LP than 2-site case.
GOOGLE_GUROBI_OPTIONS = {
    "MIPGap": 0.005,
    "TimeLimit": 28800,
    "Threads": 12,
    "NodefileStart": 2.0,
    "Method": 1,
}

# texas_UNUSED_FUTURE is 4 sites/operator. Dual simplex + few threads + early nodefile
# keeps peak RAM down; gurobi_direct (set in the solver) skips the LP reader.
FUTURE_GUROBI_OPTIONS = {
    "MIPGap": 0.005,
    "TimeLimit": 28800,
    "Threads": 4,
    "NodefileStart": 0.5,
    "Method": 1,
    "Presolve": 1,
}

# Meta (~19k jobs) and Oracle (~12k jobs) × 4 sites are the largest LPs.
FUTURE_LARGE_GUROBI_OPTIONS = {
    "MIPGap": 0.005,
    "TimeLimit": 28800,
    "Threads": 2,
    "NodefileStart": 0.25,
    "Method": 1,
    "Presolve": 1,
}

# Nationwide 48-site SI: many sites × multi-ISO LMP; keep peak RAM very low.
NATIONWIDE_GUROBI_OPTIONS = {
    "MIPGap": 0.01,
    "TimeLimit": 28800,
    "Threads": 2,
    "NodefileStart": 0.25,
    "Method": 1,
    "Presolve": 1,
}

NATIONWIDE_LARGE_GUROBI_OPTIONS = {
    "MIPGap": 0.01,
    "TimeLimit": 28800,
    "Threads": 1,
    "NodefileStart": 0.1,
    "Method": 1,
    "Presolve": 1,
}

_FUTURE_LARGE_OPERATORS = frozenset({"meta", "oracle"})
_NATIONWIDE_LARGE_OPERATORS = frozenset({"meta", "google", "oracle", "amazon"})


def _is_future_study() -> bool:
    return False


def _is_nationwide_study() -> bool:
    from experiments.lib.texas_config import TEXAS7_CONFIG_PATH

    return TEXAS7_CONFIG_PATH.name == "nationwide_operators.json"


def run_operator_study(
    operator_id: str,
    season: str,
    *,
    smoke: bool = False,
    job_sample_rate: str | None = None,
    scenario_names: tuple[str, ...] | None = None,
    results_root: Path | None = None,
    make_plots: bool | None = None,
) -> Path:
    known = operator_ids()
    if operator_id not in known:
        raise ValueError(f"operator must be one of {known}, got {operator_id!r}")
    if season not in SEASONS:
        raise ValueError(f"season must be one of {SEASONS}, got {season!r}")

    if results_root is None:
        results_root = TEXAS_ROOT / "results" / operator_id
    if smoke:
        effective_rate = "25%"
    elif job_sample_rate is not None:
        effective_rate = job_sample_rate
    else:
        effective_rate = get_trace_job_sample_rate(operator_id)
    cfg_kwargs: dict = {
        "operator_id": operator_id,
        "season": season,
        "results_root": results_root,
        "job_sample_rate": effective_rate,
    }
    if smoke:
        cfg_kwargs.update(
            {
                "scenario_names": ("initial_assignment",),
                "make_plots": False,
                "gurobi_options": {"MIPGap": 0.05, "TimeLimit": 900, "Threads": 8},
            }
        )
    else:
        n_sites = len(get_operator_sites(operator_id))
        if _is_nationwide_study():
            if operator_id in _NATIONWIDE_LARGE_OPERATORS or n_sites >= 10:
                opts = NATIONWIDE_LARGE_GUROBI_OPTIONS
            else:
                opts = NATIONWIDE_GUROBI_OPTIONS
        elif _is_future_study() or n_sites >= 4:
            if operator_id in _FUTURE_LARGE_OPERATORS:
                opts = FUTURE_LARGE_GUROBI_OPTIONS
            else:
                opts = FUTURE_GUROBI_OPTIONS
        elif operator_id == "oracle":
            opts = ORACLE_GUROBI_OPTIONS
        elif operator_id == "google":
            opts = GOOGLE_GUROBI_OPTIONS
        else:
            opts = FULL_RUN_GUROBI_OPTIONS
        cfg_kwargs["gurobi_options"] = dict(opts)
        if scenario_names is not None:
            cfg_kwargs["scenario_names"] = tuple(scenario_names)
        if make_plots is not None:
            cfg_kwargs["make_plots"] = make_plots
    cfg = TexasE01Config(**cfg_kwargs)
    return run_texas_contribution_study(cfg)
