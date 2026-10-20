"""Supplementary Fig. 7: expanded data-center capacity."""

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

from experiments.E01.plot_demand_maps import build_summary_board  # noqa: E402


def main() -> None:
    print(build_summary_board("summer", only="full_flex"))


if __name__ == "__main__":
    main()
