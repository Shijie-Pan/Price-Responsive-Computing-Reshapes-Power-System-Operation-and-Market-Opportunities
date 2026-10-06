"""Prepared grid shell for Texas flexibility scheduling.

Texas scheduling only needs ``time.periods`` (trimmed per run) plus a ``dc.*``
block filled per operator. Topology fields are empty placeholders for shared
helpers; the CPU scheduling MILP does not solve OPF on this shell.
"""

from __future__ import annotations

import json
from pathlib import Path

PREPARED_DIR = Path(__file__).resolve().parent / "prepared"
MANIFEST_PATH = PREPARED_DIR / "grid_template.json"

# Enough for 14-day calibration windows (+ spot buffer used elsewhere).
_N_PERIODS = 24 * 21


def build_grid_template() -> dict:
    """Return a deep-copyable system_data shell with empty DC block."""
    return {
        "case_type": "texas7_e01_grid_shell",
        "time": {"periods": list(range(_N_PERIODS))},
        "buses": {},
        "branches": {},
        "generators": {},
        "load_profile": {},
        "dc": {
            "sites": [],
            "capacity_MW": {},
            "Pfixed_MW": {},
            "N_cpu": {},
            "PUE": {},
            "day_ahead_prices": {},
            "HP_prices": {},
            "SP_prices": {},
        },
    }


def write_grid_template_manifest() -> Path:
    sd = build_grid_template()
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "Texas flexibility grid shell (DC/LMP filled per operator at run time)",
        "source_grid": "minimal_shell",
        "case_type": sd.get("case_type"),
        "n_time_periods": len(sd["time"]["periods"]),
        "n_buses": len(sd.get("buses", {})),
        "n_branches": len(sd.get("branches", {})),
        "n_generators": len(sd.get("generators", {})),
        "dc_sites_in_template": sd["dc"]["sites"],
        "lmp_policy": "per-site nodal CSV under experiments/E01/data/lmp/",
        "operator_capacity_source": "experiments/E01/config/texas7_operators.json",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return MANIFEST_PATH


# Module-level template used by build_operator_system_data and import shims.
data = build_grid_template()
