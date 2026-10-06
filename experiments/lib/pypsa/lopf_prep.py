#!/usr/bin/env python3
"""Prepare a PyPSA network for linear OPF and optionally solve one window.

``prepare_for_lopf`` is the load-shedding setup used by the Texas and
continental solves. The command-line entry writes only the path given by
``--out`` and does not replace the deposited result files.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


DEFAULT_NETWORK = ROOT / "experiments/data/networks/texas/elec_s1250.nc"


def _peak_transmission_loading(n) -> float:
    if len(n.lines):
        try:
            return float(n.lines_t.p0.abs().max().max() / n.lines.s_nom.max() * 100.0)
        except Exception:
            return 0.0
    if len(n.links):
        try:
            cap = n.links.p_nom.max()
            if cap <= 0:
                return 0.0
            return float(n.links_t.p0.abs().max().max() / cap * 100.0)
        except Exception:
            return 0.0
    return 0.0


def prepare_for_lopf(n, *, clip_p_max_pu: float = 0.01, load_shedding: bool = True) -> None:
    """Clip renewable availability and add load shedding at every bus."""
    df = n.generators_t.p_max_pu
    n.generators_t.p_max_pu = df.where(df > clip_p_max_pu, other=0.0)
    if hasattr(n, "generators_t") and "p_min_pu" in n.generators_t:
        df = n.generators_t.p_min_pu
        n.generators_t.p_min_pu = df.where(df > clip_p_max_pu, other=0.0)

    if not load_shedding:
        return

    if "load" not in n.carriers.index:
        n.add("Carrier", "load", color="#dd2e23", nice_name="Load shedding")
    buses_i = n.buses.query("carrier == 'AC'").index
    to_add = [b for b in buses_i if f"{b} load" not in n.generators.index]
    if len(to_add):
        # PyPSA >=1.2 dropped Network.madd and ignores ``suffix=``. Adding a bare
        # name-list with suffix collapses into one broken generator; add per bus.
        for b in to_add:
            n.add(
                "Generator",
                f"{b} load",
                bus=b,
                carrier="load",
                sign=1e-3,
                marginal_cost=1e2,
                p_nom=1e9,
            )


def run_lopf(n, *, solver_name: str = "gurobi") -> dict:
    t0 = time.time()
    prepare_for_lopf(n)
    n.optimize(solver_name=solver_name)
    status = str(getattr(n, "status", getattr(n, "model", None) and "ok") or "unknown")
    try:
        term = str(n.model.status) if hasattr(n, "model") and n.model is not None else status
    except Exception:
        term = status
    obj = getattr(n, "objective", None)
    obj = float(obj) if obj is not None else None
    return {
        "solver_status": term,
        "objective_usd": obj,
        "lopf_ok": obj is not None,
        "runtime_seconds": time.time() - t0,
        "peak_line_loading_pct": _peak_transmission_loading(n),
        "peak_transmission_loading_pct": _peak_transmission_loading(n),
        "total_generation_mw": float(n.generators_t.p.sum().sum()) if len(n.generators) else 0.0,
        "total_load_mw": float(n.loads_t.p.sum().sum()) if len(n.loads) else 0.0,
        "pf_backend": "pypsa_lopf",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network", default=str(DEFAULT_NETWORK))
    p.add_argument("--with-dc", default=None, help="network_with_dc.nc from inject_dc_load.py")
    p.add_argument("--baseline-network", default=None, help="FIXED_DC or FCFS baseline for externality")
    p.add_argument("--portfolio", default="FULL_JOB")
    p.add_argument("--out", default="validation/lopf_metrics.json")
    p.add_argument("--solver", default="gurobi")
    args = p.parse_args()

    import pypsa

    n = pypsa.Network(args.with_dc or args.network)
    metrics = run_lopf(n, solver_name=args.solver)
    metrics["portfolio"] = args.portfolio
    metrics["network"] = str(args.with_dc or args.network)

    if args.baseline_network:
        nb = pypsa.Network(args.baseline_network)
        base = run_lopf(nb, solver_name=args.solver)
        dc_cost = metrics.get("objective_usd")
        base_cost = base.get("objective_usd")
        if dc_cost is not None and base_cost is not None:
            metrics["grid_production_cost"] = dc_cost
            metrics["grid_cost_change"] = dc_cost - base_cost
            metrics["baseline_objective_usd"] = base_cost

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
