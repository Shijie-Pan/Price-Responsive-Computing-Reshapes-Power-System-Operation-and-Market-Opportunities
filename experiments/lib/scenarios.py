"""E01 scenario names shared by aggregation, E02 merge, and plotting."""

from __future__ import annotations

E01_SCENARIOS: tuple[str, ...] = (
    "baseline",
    "initial_assignment",
    "termination",
    "slack",
    "reallocation",
    "reallocation_termination",
    "reallocation_slack",
    "slack_termination",
    "reallocation_slack_termination",
)

FCFS_SCENARIO = "baseline"
E02_REF_SCENARIO = "no_dc"
FULL_FLEX_SCENARIO = "reallocation_slack_termination"
# No-DC is always solved as the E02 reference. These two DC cases are the first plots.
E02EX_PRIORITY_SCENARIOS: tuple[str, ...] = (FCFS_SCENARIO, FULL_FLEX_SCENARIO)
