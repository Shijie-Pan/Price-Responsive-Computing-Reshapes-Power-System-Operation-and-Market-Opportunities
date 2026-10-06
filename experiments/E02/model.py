"""Directional bias (D) and gross magnitude (G) vs a reference price series.

Convention
----------
Δp_i(t) = p_i(t) − p_ref(t)

  Δp > 0  → location i is more expensive than the reference
  Δp < 0  → reference is more expensive than location i

A₊ = Σ max(Δp, 0) Δt
A₋ = Σ max(−Δp, 0) Δt
G  = A₊ + A₋                 (gross magnitude; primary map metric)
D  = (A₊ − A₋) / G           ∈ [-1, 1]   (0 if G=0)
M  = G / (N Δt)              = mean |Δp| (diagnostic only)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpreadMetrics:
    n_valid_hours: int
    A_plus: float
    A_minus: float
    G: float
    directional_bias_D: float
    directional_balance: float
    mean_abs_spread_M: float
    zero_spread: bool

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["A_plus_plus_A_minus"] = d.pop("G")
        return d


def pair_valid(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise drop NaN/Inf hours; return aligned float arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def compute_spread_metrics(
    prices: np.ndarray,
    texas_ref: np.ndarray,
    *,
    dt_h: float = 1.0,
) -> SpreadMetrics:
    """Compute D and M for one location vs Texas (Δp = p − p_Texas)."""
    p, pref = pair_valid(prices, texas_ref)
    n = int(len(p))
    if n == 0:
        return SpreadMetrics(
            n_valid_hours=0,
            A_plus=0.0,
            A_minus=0.0,
            G=0.0,
            directional_bias_D=0.0,
            directional_balance=1.0,
            mean_abs_spread_M=0.0,
            zero_spread=True,
        )
    delta = p - pref
    a_plus = float(np.sum(np.maximum(delta, 0.0)) * dt_h)
    a_minus = float(np.sum(np.maximum(-delta, 0.0)) * dt_h)
    g = a_plus + a_minus
    zero = g <= 0.0 or not np.isfinite(g)
    if zero:
        d = 0.0
    else:
        d = float((a_plus - a_minus) / g)
        d = float(np.clip(d, -1.0, 1.0))
    m = float(g / (n * dt_h)) if n > 0 else 0.0
    return SpreadMetrics(
        n_valid_hours=n,
        A_plus=a_plus,
        A_minus=a_minus,
        G=g,
        directional_bias_D=d,
        directional_balance=float(1.0 - abs(d)),
        mean_abs_spread_M=m,
        zero_spread=bool(zero),
    )


def validate_spread_row(row: dict[str, Any] | pd.Series, *, tol: float = 1e-6) -> list[str]:
    """Return list of failed checks (empty = OK)."""
    errs: list[str] = []
    ap = float(row["A_plus"])
    am = float(row["A_minus"])
    g = float(row.get("A_plus_plus_A_minus", row.get("G", ap + am)))
    d = float(row["directional_bias_D"])
    m = float(row["mean_abs_spread_M"])
    if ap < -tol:
        errs.append(f"A+={ap}<0")
    if am < -tol:
        errs.append(f"A-={am}<0")
    if d < -1.0 - tol or d > 1.0 + tol:
        errs.append(f"D={d} out of [-1,1]")
    if m < -tol:
        errs.append(f"M={m}<0")
    if abs(g - (ap + am)) > tol:
        errs.append(f"G≠A++A- ({g} vs {ap + am})")
    if g > tol:
        ap_exp = g * (1.0 + d) / 2.0
        am_exp = g * (1.0 - d) / 2.0
        if abs(ap - ap_exp) > max(tol, 1e-6 * g):
            errs.append(f"A+ identity fail: {ap} vs {ap_exp}")
        if abs(am - am_exp) > max(tol, 1e-6 * g):
            errs.append(f"A- identity fail: {am} vs {am_exp}")
    return errs


def validate_dataframe(df: pd.DataFrame, *, tol: float = 1e-6) -> pd.DataFrame:
    """Return rows that fail validation (empty if all OK)."""
    bad = []
    for i, row in df.iterrows():
        errs = validate_spread_row(row, tol=tol)
        if errs:
            bad.append({"index": i, "errors": "; ".join(errs), **{k: row.get(k) for k in ("bus", "location", "state", "scenario") if k in row}})
    return pd.DataFrame(bad)
