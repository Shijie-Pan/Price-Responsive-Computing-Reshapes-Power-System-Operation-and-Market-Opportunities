from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from experiments.lib.job_pool import (
    TRACE_JOB_WINDOW_DAYS,
    TRACE_JOB_WINDOW_OFFSET_DAYS,
    trace_window_from_operator_spec,
)
from experiments.lib.job_sampling import parse_job_id_mod

from experiments.lib.paths import package_root

REPO_ROOT = package_root()
TEXAS_ROOT = Path(os.environ.get("TEXAS_STUDY_ROOT", REPO_ROOT / "experiments" / "E03"))
TEXAS7_CONFIG_PATH = Path(
    os.environ.get("TEXAS_OPERATORS_JSON", TEXAS_ROOT / "config" / "texas7_operators.json")
)


def study_e02_root() -> Path:
    """Texas transmission results (manuscript Fig. 4)."""
    return REPO_ROOT / "experiments" / "E04"


def study_e02ex_root() -> Path:
    """Alias for ``study_e02_root``."""
    return study_e02_root()


def study_e03_root() -> Path:
    """Aggregate-model comparison (manuscript Fig. 8)."""
    return REPO_ROOT / "experiments" / "E08"

SEASONS = ("winter", "summer")
SEC_PER_DAY = 86400


@lru_cache(maxsize=8)
def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_texas7_config() -> dict:
    return _load_config(str(TEXAS7_CONFIG_PATH.resolve()))


def operator_ids() -> tuple[str, ...]:
    return tuple(load_texas7_config()["operators"].keys())


# Back-compat: resolved from the default (or env-selected) operator JSON at import.
OPERATOR_IDS = operator_ids()


def get_min_service_completion_fraction() -> float:
    """Minimum aggregate delivered-work fraction when E01 termination scenarios are enabled."""
    cfg = load_texas7_config()
    return float(cfg.get("min_service_completion_fraction", 1.0))


def get_operator_spec(operator_id: str) -> dict:
    cfg = load_texas7_config()
    known = operator_ids()
    if operator_id not in known:
        raise ValueError(f"Unknown operator {operator_id!r}; expected one of {known}")
    return cfg["operators"][operator_id]


def get_operator_job_csv(operator_id: str) -> Path:
    op = get_operator_spec(operator_id)
    if "job_csv" in op:
        p = REPO_ROOT / op["job_csv"]
        if p.exists():
            return p
    seg = op.get("trace_segment")
    if seg:
        return REPO_ROOT / "experiments/data/jobs/by_trace" / seg / "job_info.csv"
    for p in (
        REPO_ROOT / "experiments" / "data" / "sources" / "job_info_df_A10.csv",
        REPO_ROOT / "data" / "job_info_df_A10.csv",
    ):
        if p.exists():
            return p
    return REPO_ROOT / "experiments" / "data" / "sources" / "job_info_df_A10.csv"


def get_site_lmp_csv(site_id: int, season: str, *, allow_override: bool = True) -> Path | None:
    if allow_override:
        override = os.environ.get("TEXAS_LMP_OVERRIDE_DIR")
        if override:
            root = Path(override)
            if not root.is_absolute():
                root = REPO_ROOT / root
            candidates = (
                root / f"site_{int(site_id)}" / f"{season}.csv",
                root / f"site_{int(site_id)}.csv",
            )
            for cand in candidates:
                if cand.is_file():
                    return cand
            raise FileNotFoundError(
                f"TEXAS_LMP_OVERRIDE_DIR={root} has no LMP CSV for site {site_id} season {season}"
            )
    cfg = load_texas7_config()
    key = f"lmp_{season}_csv"
    for site in cfg["sites"]:
        if int(site["site_id"]) == int(site_id) and key in site:
            return REPO_ROOT / site[key]
    return REPO_ROOT / "experiments" / "E03" / "data" / "lmp" / f"site_{int(site_id)}" / f"{season}.csv"


def get_operator_sites(operator_id: str) -> list[dict]:
    cfg = load_texas7_config()
    op = get_operator_spec(operator_id)
    site_ids = set(op["site_ids"])
    return [s for s in cfg["sites"] if s["site_id"] in site_ids]


def get_operator_total_mw(operator_id: str) -> float:
    return sum(float(s["projected_total_mw"]) for s in get_operator_sites(operator_id))


def target_util() -> float:
    cfg = load_texas7_config()
    return float(cfg.get("trace_target_util", 0.6))


def get_trace_min_submit_time_sec() -> float:
    """Drop jobs with ``submit_time`` before ``trace_min_submit_day`` (default: keep all)."""
    cfg = load_texas7_config()
    day = cfg.get("trace_min_submit_day")
    if day is None:
        return -1.0
    day = int(day)
    if day <= 0:
        return -1.0
    return day * SEC_PER_DAY - 1.0


def get_trace_max_eval_carryover_long_jobs(operator_id: str | None = None) -> int | None:
    """Max pre-eval long jobs to keep; ``None`` = no cap (keep all)."""
    cfg = load_texas7_config()

    def _parse(val) -> int | None:
        if val is None:
            return None
        val = int(val)
        return val if val > 0 else None

    if operator_id:
        op = cfg.get("operators", {}).get(operator_id, {})
        if "trace_max_eval_carryover_long_jobs" in op:
            return _parse(op["trace_max_eval_carryover_long_jobs"])
    return _parse(cfg.get("trace_max_eval_carryover_long_jobs"))


def get_trace_carryover_long_min_hours() -> int:
    cfg = load_texas7_config()
    return int(cfg.get("trace_carryover_long_min_hours", 24))


def get_trace_carryover_cap_policy(operator_id: str) -> str:
    """How to rank pre-eval long jobs when capping carryover (default: lightest)."""
    cfg = load_texas7_config()
    default = str(cfg.get("trace_carryover_cap_policy", "lightest"))
    op = cfg.get("operators", {}).get(operator_id, {})
    return str(op.get("trace_carryover_cap_policy", default))


def get_trace_watts_per_cpu_target() -> float | None:
    """Target dynamic facility W per fleet CPU unit (load ≡ N_cpu basis). ``None`` = off."""
    cfg = load_texas7_config()
    val = cfg.get("trace_watts_per_cpu_target")
    if val is None:
        return None
    val = float(val)
    return val if val > 0 else None


def get_trace_job_sample_rate(operator_id: str | None = None) -> str:
    """Job sampling rate string, e.g. ``10%`` (per-operator override, else global)."""
    cfg = load_texas7_config()
    default = str(cfg.get("trace_job_sample_rate", "100%"))
    if operator_id:
        op = cfg.get("operators", {}).get(operator_id, {})
        if "trace_job_sample_rate" in op:
            return str(op["trace_job_sample_rate"])
    return default


def get_trace_job_sample_mod(operator_id: str | None = None) -> int | None:
    """``job_id_mod`` for trace/E01 loading; ``None`` means 100%% (no sampling).

    Rates below 25%% are clamped to 25%% (``mod`` ≤ 4).
    """
    mod, _ = parse_job_id_mod(get_trace_job_sample_rate(operator_id))
    if mod is not None and mod > 4:
        return 4
    return mod if mod is not None and mod > 1 else None


def get_trace_window(operator_id: str) -> dict:
    op = get_operator_spec(operator_id)
    cfg = load_texas7_config()
    default_job_days = int(cfg.get("trace_job_window_days", TRACE_JOB_WINDOW_DAYS))
    default_offset = int(cfg.get("trace_job_window_offset_days", TRACE_JOB_WINDOW_OFFSET_DAYS))
    trace = trace_window_from_operator_spec(
        op,
        default_job_window_days=default_job_days,
        default_job_window_offset_days=default_offset,
    )
    trace["min_submit_time_sec"] = get_trace_min_submit_time_sec()
    return trace
