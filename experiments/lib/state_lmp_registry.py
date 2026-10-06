"""US state LMP registry and cache coverage checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from experiments.lib.paths import package_root

ROOT = package_root()
REGISTRY_PATH = ROOT / "experiments" / "E02" / "config" / "state_lmp_registry.yaml"
DEFAULT_CACHE = ROOT / "experiments" / "data" / "lmp_cache"


def load_registry(path: Path | None = None) -> dict[str, Any]:
    p = path or REGISTRY_PATH
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _hub_cache_path(cache_dir: Path, iso: str, hub: str, season: str) -> Path:
    from experiments.lib.lmp_io import SEASONS

    safe = hub.replace(" ", "_").replace("/", "-")
    return cache_dir / iso / safe / f"{season}_{SEASONS[season]['label']}.json"


def _state_cache_path(cache_dir: Path, state: str, season: str) -> Path:
    from experiments.lib.lmp_io import SEASONS

    return cache_dir / "STATES" / state / f"{season}_{SEASONS[season]['label']}.json"


def hub_cached(cache_dir: Path, iso: str, hub: str, season: str) -> bool:
    from experiments.lib.lmp_io import season_hours

    p = _hub_cache_path(cache_dir, iso, hub, season)
    if not p.exists():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return len(payload.get("prices", [])) >= season_hours(season)
    except Exception:
        return False


def state_cached(cache_dir: Path, state: str, season: str) -> bool:
    from experiments.lib.lmp_io import season_hours

    p = _state_cache_path(cache_dir, state, season)
    if not p.exists():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return len(payload.get("prices", [])) >= season_hours(season)
    except Exception:
        return False


def effective_coverage(
    state: str,
    info: dict,
    *,
    cache_dir: Path,
    season: str = "winter",
) -> str:
    """Return: direct | proxy | cached | missing."""
    declared = str(info.get("coverage", "missing"))
    iso = info.get("iso")
    hub = info.get("hub")
    if not iso or not hub:
        return "missing"
    if state_cached(cache_dir, state, season):
        return "direct"
    if hub_cached(cache_dir, str(iso), str(hub), season):
        return "direct"
    return "missing"


def coverage_table(*, cache_dir: Path | None = None, season: str = "winter") -> list[dict[str, Any]]:
    reg = load_registry()
    cache_dir = cache_dir or DEFAULT_CACHE
    rows = []
    for state, info in sorted(reg.get("states", {}).items()):
        eff = effective_coverage(state, info, cache_dir=cache_dir, season=season)
        rows.append(
            {
                "state": state,
                "iso": info.get("iso"),
                "hub": info.get("hub"),
                "timezone": info.get("timezone"),
                "declared_coverage": info.get("coverage"),
                "effective_coverage": eff,
                "note": info.get("note", ""),
            }
        )
    return rows


def states_needing_download(*, cache_dir: Path | None = None, seasons: tuple[str, ...] = ("winter", "summer")) -> list[dict]:
    """States where hub cache is absent for at least one season."""
    reg = load_registry()
    cache_dir = cache_dir or DEFAULT_CACHE
    out = []
    for state, info in reg.get("states", {}).items():
        iso, hub = info.get("iso"), info.get("hub")
        if not iso or not hub:
            continue
        missing_seasons = [
            s for s in seasons if not hub_cached(cache_dir, str(iso), str(hub), s) and not state_cached(cache_dir, state, s)
        ]
        if missing_seasons:
            out.append(
                {
                    "state": state,
                    "iso": iso,
                    "hub": hub,
                    "missing_seasons": missing_seasons,
                    "coverage": info.get("coverage"),
                }
            )
    return out
