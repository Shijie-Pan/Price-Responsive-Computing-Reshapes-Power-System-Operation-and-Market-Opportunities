"""Job sampling helpers for experiments (no legacy utils/data_14bus imports)."""

from __future__ import annotations


def parse_job_id_mod(job_id_mod: str | int | None) -> tuple[int | None, str]:
    """Parse JOB_SAMPLE_RATE like '25%' -> (4, '25pct')."""
    mod_to_pct = {1: "100pct", 2: "50pct", 4: "25pct", 5: "20pct", 10: "10pct", 20: "5pct", 100: "1pct"}

    if job_id_mod is None:
        return None, "all"

    if isinstance(job_id_mod, str):
        if not job_id_mod.endswith("%"):
            raise ValueError(f"JOB_ID_MOD string must end with '%', got: {job_id_mod}")
        pct_val = float(job_id_mod.rstrip("%"))
        pct_to_mod = {100: 1, 50: 2, 25: 4, 20: 5, 10: 10, 5: 20, 1: 100}
        mod_value = pct_to_mod.get(int(pct_val))
        if mod_value is None:
            raise ValueError(
                f"Unsupported percentage: {pct_val}% "
                f"(allowed: 25%, 50%, 100%; legacy: 1%, 5%, 10%, 20%)"
            )
    else:
        mod_value = int(job_id_mod)
        allowed = {1, 2, 4, 5, 10, 20, 100}
        if mod_value not in allowed:
            raise ValueError(
                f"JOB_ID_MOD must be one of {sorted(allowed)} or percentage string, got: {mod_value}"
            )

    return mod_value, mod_to_pct.get(mod_value, "all")
