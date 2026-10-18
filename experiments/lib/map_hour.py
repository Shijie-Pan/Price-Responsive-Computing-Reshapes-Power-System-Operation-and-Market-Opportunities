"""Pick the eval-week hour with the largest line-loading change for E02 maps."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.lib.scenarios import E01_SCENARIOS, E02_REF_SCENARIO

DELTA_PP = 5.0
_VOLTAGE_KEYS = (
    "voltage_violation_count",
    "max_voltage_deviation_pu",
    "mean_voltage_deviation_pu",
    "n_buses_checked",
    "violating_bus_ids",
    "pf_converged",
    "notes",
    "max_voltage_deviation",
    "pf_backend_voltage",
    "snapshot_idx",
    "snapshot",
    "voltage_limits_pu",
    "voltage_violation_count_total",
    "max_voltage_deviation_pu_window",
    "pf_converged_hours",
    "pf_failed_hours",
)


def snapshot_label(n, idx: int, season: str | None = None) -> str:
    from experiments.lib.network_season import shift_snapshot_label_to_season

    if not len(n.snapshots):
        return ""
    snap = n.snapshots[int(idx)]
    ts = snap[-1] if isinstance(snap, tuple) else snap
    label = pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")
    return shift_snapshot_label_to_season(label, season)


def snapshot_labels(n, season: str | None = None) -> list[str]:
    return [snapshot_label(n, i, season) for i in range(len(n.snapshots))]


def _delta_count(a: pd.Series, b: pd.Series, threshold: float) -> int:
    left, right = a.align(b, join="inner")
    if left.empty:
        return 0
    return int((left - right).abs().ge(threshold).sum())


def select_hour_max_change_vs_ref(
    frames: dict[str, pd.DataFrame],
    *,
    ref_key: str = E02_REF_SCENARIO,
    threshold: float = DELTA_PP,
) -> tuple[int, int, list[int]]:
    """Hour maximizing |Δloading|≥threshold branches vs the reference, summed over other scenarios."""
    if ref_key not in frames:
        first = next(iter(frames.values()))
        return 0, 0, [0] * len(first)
    ref = frames[ref_key]
    others = {k: v for k, v in frames.items() if k != ref_key}
    n_hours = len(ref)
    scores = [0] * n_hours
    best_t, best = 0, -1
    for t in range(n_hours):
        score = 0
        ref_row = ref.iloc[t]
        for df in others.values():
            score += _delta_count(ref_row, df.iloc[t], threshold)
        scores[t] = score
        if score > best:
            best, best_t = score, t
    return best_t, max(best, 0), scores


def select_hour_max_study_delta(
    frames_a: dict[str, pd.DataFrame],
    frames_f: dict[str, pd.DataFrame],
    *,
    keys: list[str] | None = None,
    threshold: float = DELTA_PP,
) -> tuple[int, int, list[int]]:
    """Hour maximizing |Δloading|≥threshold branches between actual and future (same scenarios)."""
    keys = list(keys or [E02_REF_SCENARIO, *E01_SCENARIOS])
    shared = [k for k in keys if k in frames_a and k in frames_f]
    n_hours = min(len(frames_a[shared[0]]), len(frames_f[shared[0]])) if shared else 0
    scores = [0] * n_hours
    best_t, best = 0, -1
    for t in range(n_hours):
        score = 0
        for key in shared:
            score += _delta_count(frames_a[key].iloc[t], frames_f[key].iloc[t], threshold)
        scores[t] = score
        if score > best:
            best, best_t = score, t
    return best_t, max(best, 0), scores


def hourly_dir(results_dir: Path, season: str) -> Path:
    return results_dir / f"hourly_{season}"


def save_hourly_frames(
    frames: dict[str, pd.DataFrame],
    *,
    results_dir: Path,
    season: str,
    labels: list[str],
    map_idx: int,
    map_label: str,
    selection: str,
    score: int,
    scores: list[int],
) -> Path:
    out_dir = hourly_dir(results_dir, season)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in frames.items():
        df.to_pickle(out_dir / f"{key}.pkl")
    scan = pd.DataFrame(
        {
            "hour": list(range(len(scores))),
            "snapshot": labels[: len(scores)],
            "n_delta_branches": scores,
        }
    )
    scan.to_csv(out_dir / "hour_scan.csv", index=False)
    meta = {
        "season": season,
        "n_hours": len(labels),
        "snapshots": labels,
        "map_snapshot_idx": int(map_idx),
        "map_snapshot_label": map_label,
        "map_snapshot_selection": selection,
        "map_impact_score": int(score),
        "delta_threshold_pp": DELTA_PP,
        "scenarios": list(frames),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def save_hourly_scenario(
    df: pd.DataFrame, *, results_dir: Path, season: str, scenario: str
) -> Path:
    """Write one scenario pickle immediately so a killed LOPF can resume."""
    out_dir = hourly_dir(results_dir, season)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scenario}.pkl"
    df.to_pickle(path)
    return path


def load_hourly_frames(results_dir: Path, season: str) -> tuple[dict[str, pd.DataFrame], dict]:
    out_dir = hourly_dir(results_dir, season)
    if not out_dir.exists():
        return {}, {}
    meta: dict = {}
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(out_dir.glob("*.pkl")):
        frames[path.stem] = pd.read_pickle(path)
    return frames, meta


def strip_voltage_fields(metrics: dict) -> None:
    for key in _VOLTAGE_KEYS:
        metrics.pop(key, None)


def portfolios_from_hourly(
    frames: dict[str, pd.DataFrame],
    *,
    hour: int,
    label: str,
    static_n,
) -> dict[str, dict]:
    from experiments.lib.pypsa.geo_network_plot import edges_from_loading_series

    out: dict[str, dict] = {}
    for key, df in frames.items():
        if hour >= len(df):
            continue
        row = df.iloc[int(hour)]
        edges = edges_from_loading_series(static_n, row)
        loadings = edges["loading_pct"] if len(edges) else None
        peak = float(loadings.max()) if loadings is not None and len(loadings) else 0.0
        n_ov = int((loadings >= 99.9 - 1e-9).sum()) if loadings is not None else 0
        n_near = int((loadings >= 90.0).sum()) if loadings is not None else 0
        out[key] = {
            "map_snapshot_idx": int(hour),
            "map_snapshot_label": label,
            "peak_loading_pct": round(peak, 4),
            "n_overloaded": n_ov,
            "n_lines_at_or_above_90pct": n_near,
            "edges": edges.to_dict(orient="records"),
            "pf_converged": None,
        }
    return out
