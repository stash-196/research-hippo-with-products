"""Helpers for saving freq-SSM development artifacts to disk."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def init_dev_output(save_artifacts: bool = True) -> Path | None:
    """Create an output directory for a development run.

    Artifacts are written to ``outputs/freq_dev/latest/`` so collaborators
    can inspect the most recent run at a stable path.
    """
    if not save_artifacts:
        return None

    base = repo_root() / "outputs" / "freq_dev"
    out = base / "latest"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    return out


def check_dir(out: Path | None, check_name: str) -> Path | None:
    if out is None:
        return None
    path = out / check_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(out: Path | None, check_name: str, filename: str) -> Path | None:
    check = check_dir(out, check_name)
    if check is None:
        return None
    return check / filename


def save_metrics(path: Path | None, metrics: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)


def save_run_config(out: Path | None, config: dict[str, Any]) -> None:
    save_metrics(out / "run_config.json" if out is not None else None, config)


def archive_latest_output(out: Path | None) -> Path | None:
    """Copy ``latest/`` into ``runs/<timestamp>/`` after a run completes."""
    if out is None:
        return None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive = repo_root() / "outputs" / "freq_dev" / "runs" / run_id
    shutil.copytree(out, archive)
    runs_dir = archive.parent
    (runs_dir / "last_run_id.txt").write_text(run_id + "\n", encoding="utf-8")
    return archive