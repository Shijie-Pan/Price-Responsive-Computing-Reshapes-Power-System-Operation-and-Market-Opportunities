"""Deposit root, independent of how deep a module sits."""

from __future__ import annotations

from pathlib import Path


def package_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "experiments" / "data" / "networks").is_dir() and (candidate / "experiments" / "lib").is_dir():
            return candidate
    raise RuntimeError("Cannot find the deposit root")
