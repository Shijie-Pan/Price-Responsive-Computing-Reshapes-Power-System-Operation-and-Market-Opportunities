from __future__ import annotations

import copy

from experiments.lib.solve.grid_template import data as grid_template
from experiments.lib.texas_config import get_operator_sites, load_texas7_config
from experiments.lib.solve.site_lmp import load_site_lmp

_REFERENCE_PEAK_MW = 325.0
_REFERENCE_N_CPU = 80000.0
_PFIXED_RATIO = 0.2


def build_operator_system_data(
    operator_id: str,
    season: str,
    days: int,
    spot_extra_hours: int,
) -> dict:
    cfg = load_texas7_config()
    ref_peak = float(cfg.get("reference_peak_mw_per_site", _REFERENCE_PEAK_MW))
    ref_n_cpu = float(cfg.get("reference_n_cpu", _REFERENCE_N_CPU))

    sites_meta = get_operator_sites(operator_id)
    site_ids = [s["site_id"] for s in sites_meta]

    sd = copy.deepcopy(grid_template)
    h_original = 24 * days
    h_total = min(h_original + int(spot_extra_hours), len(sd["time"]["periods"]))
    sd["time"]["periods"] = list(range(h_total))

    capacity_mw: dict[int, float] = {}
    pfixed_mw: dict[int, float] = {}
    n_cpu: dict[int, float] = {}
    pue: dict[int, float] = {}
    hp_prices: dict[int, float] = {}
    sp_prices: dict[int, float] = {}
    lmp_zones: dict[int, str] = {}

    for site in sites_meta:
        sid = int(site["site_id"])
        projected = float(site["projected_total_mw"])
        scale = projected / ref_peak
        capacity_mw[sid] = projected
        pfixed_mw[sid] = projected * _PFIXED_RATIO
        n_cpu[sid] = ref_n_cpu * scale
        pue[sid] = float(site.get("pue", 1.3))
        hp_prices[sid] = 0.484
        sp_prices[sid] = 0.243
        lmp_zones[sid] = str(site["lmp_zone"])

    sd["dc"] = {
        "sites": site_ids,
        "capacity_MW": capacity_mw,
        "Pfixed_MW": pfixed_mw,
        "N_cpu": n_cpu,
        "PUE": pue,
        "day_ahead_prices": {},
        "HP_prices": hp_prices,
        "SP_prices": sp_prices,
    }

    load_site_lmp(sd, lmp_zones, hours=h_total, season=season)
    return sd
