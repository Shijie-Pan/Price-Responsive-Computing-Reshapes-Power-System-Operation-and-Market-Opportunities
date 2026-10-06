"""Write per-site day-ahead price tables from a local cache.

The cache is not in this deposit. Build it as described in
``experiments/E03/README.md``, then run this script. It reads
``experiments/data/lmp_cache`` and writes
``experiments/E03/data/lmp/site_<id>/{winter,summer}.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (
            candidate / "experiments" / "lib"
        ).is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")


ROOT = _package_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.lib.download_lmp import materialize_site_lmp  # noqa: E402
from experiments.lib.texas_config import load_texas7_config  # noqa: E402

CACHE = ROOT / "experiments" / "data" / "lmp_cache"
OUT = ROOT / "experiments" / "E03" / "data" / "lmp"


def main() -> None:
    if not CACHE.is_dir():
        raise SystemExit(
            "Missing experiments/data/lmp_cache. See experiments/E03/README.md."
        )
    for site in load_texas7_config()["sites"]:
        site_id = int(site["site_id"])
        for season in ("winter", "summer"):
            entry = materialize_site_lmp(
                OUT,
                site_id,
                iso=str(site["lmp_iso"]),
                hub=str(site["lmp_node"]),
                location_label=str(site["lmp_node"]),
                season=season,
                cache_dir=CACHE,
                repo_root=ROOT,
                node_type=str(site["lmp_node_type"]),
                price_column="lmp",
            )
            print(entry["path"])


if __name__ == "__main__":
    main()
