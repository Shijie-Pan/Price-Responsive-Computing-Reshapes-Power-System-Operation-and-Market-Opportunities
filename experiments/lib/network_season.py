"""Seasonal native-grid window for Texas s1250 LOPF.

``elec_s1250.nc`` only stores 168 h (PyPSA labels them 2030-01-01..07). Those hours
are the first week of the 2019 profiles (January). July 1 00:00 is hour 4344 in
``profile_solar.nc`` / ``profile_onwind.nc``. When the network has a full year,
summer LOPF slices from that index; otherwise we overlay the July week onto the
168 h extract via GIS ``sub_id`` (January correlation is 1.0).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 2019 is non-leap: day-of-year 182 (July 1) * 24 = 4344
JULY_START_HOUR = 4344
SEASON_START_HOUR = {"winter": 0, "summer": JULY_START_HOUR}
PROFILE_FILES = {"solar": "profile_solar.nc", "onwind": "profile_onwind.nc"}


def shift_snapshot_label_to_season(label: str | None, season: str | None) -> str:
    """Map the 168 h extract's January clock onto July for summer figures."""
    text = str(label or "")
    if season == "summer" and "2030-01-" in text:
        return text.replace("2030-01-", "2030-07-", 1)
    return text


def season_eval_start(season: str, n_snapshots: int, eval_length: int = 168) -> int:
    """Hour index to slice if the network actually contains that week."""
    start = int(SEASON_START_HOUR.get(season, 0))
    if n_snapshots >= start + eval_length:
        return start
    return 0


def apply_july_renewable_profiles(n, resources_dir: Path, *, n_hours: int = 168) -> int:
    """Overwrite solar/onwind ``p_max_pu`` with the July week from profile_*.nc."""
    import netCDF4 as ncdf

    resources_dir = Path(resources_dir)
    gis_path = resources_dir / "bus_gis.csv"
    if not gis_path.exists():
        print(
            f"[season] skip July renewable overlay: missing {gis_path}",
            flush=True,
        )
        return 0
    gis = pd.read_csv(gis_path)
    gx = gis["x"].to_numpy(float)
    gy = gis["y"].to_numpy(float)
    gsub = gis["sub_id"].to_numpy(float)
    bx = n.buses["x"].to_numpy(float)
    by = n.buses["y"].to_numpy(float)
    nn = ((gx[None, :] - bx[:, None]) ** 2 + (gy[None, :] - by[:, None]) ** 2).argmin(axis=1)
    bus2sub = {str(bus): float(gsub[j]) for bus, j in zip(n.buses.index.astype(str), nn)}

    pu = n.generators_t.p_max_pu
    n_hours = min(n_hours, len(n.snapshots), len(pu))
    updated = 0
    sl = slice(JULY_START_HOUR, JULY_START_HOUR + n_hours)

    for carrier, fname in PROFILE_FILES.items():
        path = resources_dir / fname
        if not path.exists():
            print(f"[season] missing {path}, skip {carrier}", flush=True)
            continue
        gens = [
            g
            for g in n.generators.index
            if str(n.generators.at[g, "carrier"]) == carrier and g in pu.columns
        ]
        if not gens:
            continue
        ds = ncdf.Dataset(path)
        pbus = np.array(ds.variables["bus"][:], dtype=float)
        prof = np.array(ds.variables["profile"][sl, :], dtype=float)
        ds.close()
        if prof.shape[0] < n_hours:
            raise RuntimeError(
                f"{fname} July slice has {prof.shape[0]} h, need {n_hours} "
                f"(JULY_START_HOUR={JULY_START_HOUR})"
            )
        sub2col = {float(s): i for i, s in enumerate(pbus)}
        for gen in gens:
            sub = bus2sub.get(str(n.generators.at[gen, "bus"]))
            col = sub2col.get(sub)
            if col is None:
                continue
            idx = pu.index[:n_hours]
            pu.loc[idx, gen] = prof[:n_hours, col]
            updated += 1
        print(
            f"[season] overlay July {carrier} p_max_pu on {len(gens)} gens "
            f"from {fname} hour {JULY_START_HOUR}",
            flush=True,
        )
    return updated


def prepare_seasonal_network(
    n,
    *,
    season: str,
    network_path: Path,
    eval_start: int,
    eval_length: int,
):
    """Slice the native week, overlay July renewables if the nc is a 168 h January extract."""
    n_snap = len(n.snapshots)
    length = min(eval_length, n_snap)
    abs_start = season_eval_start(season, n_snap, eval_length=length)
    if abs_start > 0:
        start = abs_start + eval_start
        end = min(start + length, n_snap)
        n.set_snapshots(n.snapshots[start:end])
        print(
            f"[season] {season}: sliced native snapshots [{start}:{end}] "
            f"(season_0={abs_start})",
            flush=True,
        )
        return n
    end = min(eval_start + length, n_snap)
    n.set_snapshots(n.snapshots[eval_start:end])
    if season == "summer":
        n_upd = apply_july_renewable_profiles(
            n, Path(network_path).parent, n_hours=len(n.snapshots)
        )
        print(
            f"[season] summer: {n_upd} renewable gens <- July week "
            f"(network has {n_snap} h, not {JULY_START_HOUR}+; demand stays extract)",
            flush=True,
        )
    return n
