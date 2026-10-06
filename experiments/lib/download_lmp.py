"""Read cached hourly day-ahead prices and write site price tables."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from experiments.lib.lmp_io import (
    HOURS_DEFAULT,
    SEASONS,
    season_dates as _season_dates,
    season_hours,
    write_hourly_lmp_csv,
)


def season_dates(season: str) -> tuple[date, date]:
    return _season_dates(season)


REGIONAL_ISO_PROXY: dict[tuple[str, str], tuple[str, str]] = {
    ("BPA", "MID-C"): ("CAISO", "NP15"),
    ("SERC", "SOUTHERN_HUB"): ("MISO", "LOUISIANA.HUB"),
    ("WECC", "WYoming_HUB"): ("CAISO", "SP15"),
    ("WECC", "UTAH_HUB"): ("CAISO", "SP15"),
    ("WECC", "FOUR_CORNERS"): ("CAISO", "SP15"),
    ("TVA", "TVA_HUB"): ("MISO", "LOUISIANA.HUB"),
}


def resolve_iso_hub(iso: str, hub: str) -> tuple[str, str, str | None]:
    """Return (iso, hub, proxy_note) for regions stored under a neighbouring hub."""
    key = (iso.upper(), hub)
    if key in REGIONAL_ISO_PROXY:
        mapped_iso, mapped_hub = REGIONAL_ISO_PROXY[key]
        return mapped_iso, mapped_hub, f"{iso}/{hub} -> {mapped_iso}/{mapped_hub}"
    return iso.upper(), hub, None


def _cache_path(
    cache_dir: Path,
    iso: str,
    hub: str,
    season: str,
    *,
    nodal: bool = False,
    cache_key: str | None = None,
) -> Path:
    if cache_key:
        return cache_dir / "USA" / cache_key / f"{season}_{SEASONS[season]['label']}.json"
    safe_hub = hub.replace(" ", "_").replace("/", "-")
    if nodal:
        return cache_dir / iso / "nodal" / safe_hub / f"{season}_{SEASONS[season]['label']}.json"
    return cache_dir / iso / safe_hub / f"{season}_{SEASONS[season]['label']}.json"


def _load_cache(path: Path, *, hours: int = HOURS_DEFAULT) -> list[float] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    prices = [float(x) for x in payload.get("prices", [])]
    if len(prices) >= hours:
        return prices[:hours]
    return None


def _save_cache(path: Path, prices: list[float], meta: dict, *, hours: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = hours if hours is not None else int(meta.get("hours") or len(prices) or HOURS_DEFAULT)
    payload = {**meta, "prices": prices[:n], "hours": n}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def download_prices(
    iso: str,
    hub: str,
    season: str,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    node_type: str | None = None,
    cache_key: str | None = None,
) -> tuple[list[float], dict]:
    """Return a cached price series. ``force`` is ignored; this package does not fetch prices."""
    del force
    if season not in SEASONS:
        raise ValueError(f"season must be one of {tuple(SEASONS)}, got {season!r}")

    nodal = iso.upper() == "ERCOT" and node_type is not None
    cache_dir = cache_dir or Path("experiments/data/lmp_cache")
    hours = season_hours(season)
    cache_path = _cache_path(
        cache_dir, iso, hub, season, nodal=nodal, cache_key=cache_key
    )
    cached = _load_cache(cache_path, hours=hours)
    if cached is None:
        raise FileNotFoundError(f"Cached prices not found: {cache_path.as_posix()}")
    meta = json.loads(cache_path.read_text(encoding="utf-8"))
    meta.pop("prices", None)
    return cached, meta


def materialize_site_lmp(
    out_dir: Path,
    site_id: int,
    *,
    iso: str,
    hub: str,
    location_label: str,
    season: str,
    cache_dir: Path | None = None,
    force: bool = False,
    repo_root: Path | None = None,
    node_type: str | None = None,
    price_column: str = "spp",
    cache_key: str | None = None,
) -> dict:
    prices, meta = download_prices(
        iso,
        hub,
        season,
        cache_dir=cache_dir,
        force=force,
        node_type=node_type,
        cache_key=cache_key,
    )
    out_path = out_dir / f"site_{site_id}" / f"{season}.csv"
    write_hourly_lmp_csv(
        prices,
        out_path,
        location=location_label,
        start=SEASONS[season]["start_utc"],
        price_column=price_column,
    )
    rel_path = out_path
    if repo_root is not None:
        try:
            rel_path = out_path.relative_to(repo_root)
        except ValueError:
            rel_path = out_path
    return {
        "path": str(rel_path).replace("\\", "/"),
        "hours": len(prices),
        **meta,
    }


def materialize_site_lmp_from_prices(
    out_dir: Path,
    site_id: int,
    prices: list[float],
    meta: dict,
    *,
    location_label: str,
    season: str,
    repo_root: Path | None = None,
    price_column: str = "spp",
    bus_cache_key: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Write a site price CSV and, when requested, a bus-keyed cache copy."""
    out_path = out_dir / f"site_{site_id}" / f"{season}.csv"
    write_hourly_lmp_csv(
        prices,
        out_path,
        location=location_label,
        start=SEASONS[season]["start_utc"],
        price_column=price_column,
    )
    if bus_cache_key and cache_dir is not None:
        bus_path = _cache_path(cache_dir, meta.get("iso", ""), meta.get("hub", ""), season, cache_key=bus_cache_key)
        if not bus_path.exists():
            _save_cache(bus_path, prices, {**meta, "lmp_cache_key": bus_cache_key})
    rel_path = out_path
    if repo_root is not None:
        try:
            rel_path = out_path.relative_to(repo_root)
        except ValueError:
            rel_path = out_path
    out_meta = {**meta, "path": str(rel_path).replace("\\", "/"), "hours": len(prices)}
    if bus_cache_key:
        out_meta["lmp_cache_key"] = bus_cache_key
    return out_meta


def materialize_site_seasons(
    out_dir: Path,
    site_id: int,
    *,
    iso: str,
    hub: str,
    location_label: str,
    cache_dir: Path | None = None,
    force: bool = False,
    repo_root: Path | None = None,
    node_type: str | None = None,
    price_column: str = "spp",
    cache_key: str | None = None,
) -> dict:
    out: dict = {"site_id": site_id, "seasons": {}}
    for season in ("winter", "summer"):
        out["seasons"][season] = materialize_site_lmp(
            out_dir,
            site_id,
            iso=iso,
            hub=hub,
            location_label=location_label,
            season=season,
            cache_dir=cache_dir,
            force=force,
            repo_root=repo_root,
            node_type=node_type,
            price_column=price_column,
            cache_key=cache_key,
        )
    return out
