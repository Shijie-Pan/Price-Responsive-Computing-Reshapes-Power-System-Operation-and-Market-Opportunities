"""E03 model identifiers and narrative labels."""

from __future__ import annotations

FULL_JOB_SCENARIO = "reallocation_slack_termination"
FCFS_SCENARIO = "baseline"

# Private-side comparison models (scheduling layer)
SCHEDULING_MODELS: tuple[str, ...] = (
    "FULL_JOB",
    "AGG_CONT",
    "AGG_ENERGY",
    "AGG_POOL",
    "FCFS",
    "FIXED_DC",
)

AGGREGATE_MODELS: tuple[str, ...] = ("AGG_CONT", "AGG_ENERGY", "AGG_POOL")

# Grid LOPF subset (skip FCFS / FIXED_DC for clarity)
GRID_MODELS: tuple[str, ...] = ("NO_DC", "FULL_JOB", "AGG_ENERGY", "AGG_CONT", "AGG_POOL")

MODEL_LABELS: dict[str, str] = {
    "FULL_JOB": "Job-level (Full flexibility)",
    "AGG_CONT": "AGG Continuous",
    "AGG_ENERGY": "AGG Site",
    "AGG_POOL": "AGG Pool",
    "FCFS": "FCFS (no price response)",
    "FIXED_DC": "Fixed mean load",
    "NO_DC": "Grid baseline (no DC)",
}

MODEL_NOTES: dict[str, str] = {
    "FULL_JOB": (
        "Integer x[j,s,t]: tasks have site affinity, windows, CPU granularity; "
        "power P[s,t]=f(sum x) is piecewise and not freely shiftable."
    ),
    "AGG_CONT": (
        "Continuous P[s,t] in [P_min,P_max] with no hourly ramp limit; optimizes LMP cost "
        "without job identity — assumes idle–busy load can move smoothly in time and space."
    ),
    "AGG_ENERGY": (
        "AGG_CONT plus per-site total MWh matched to FULL_JOB — same workload energy "
        "but still continuous/deferrable; ignores task size and scheduling resolution."
    ),
    "AGG_POOL": (
        "AGG_CONT plus per-operator pooled MWh (sites within operator can trade energy) — "
        "even more optimistic cross-site flexibility than job-level migration allows."
    ),
    "FCFS": "E01 baseline portfolio; no reallocation/slack/termination.",
    "FIXED_DC": "168 h flat load at site mean — no temporal flexibility.",
}
