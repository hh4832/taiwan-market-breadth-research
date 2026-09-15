from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPOSITORY = "hh4832/taiwan-market-breadth-research"
OFFICIAL_DRIVE_OUTPUT_ROOT = Path("/content/drive/MyDrive/Quant_Research/taiwan-market-breadth-research")


def project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if root.name != "taiwan-market-breadth-research-v7" and not (root / ".git").exists():
        raise RuntimeError(f"Cannot resolve repository root from {__file__}")
    return root


def git_value(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=project_root(), check=True, capture_output=True, text=True).stdout.strip()


@dataclass(frozen=True)
class RunContext:
    run_id: str
    timestamp: str
    version_slug: str
    git_commit: str
    git_branch: str
    local_run_dir: Path


def create_run_context(version_slug: str, output_root: Path | None = None, now: datetime | None = None) -> RunContext:
    moment = now or datetime.now(ZoneInfo("Asia/Taipei"))
    stamp = moment.astimezone(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    commit = git_value("rev-parse", "HEAD")
    branch = git_value("branch", "--show-current") or "detached"
    run_id = f"{stamp}_{version_slug}_{commit[:12]}"
    base = (output_root or project_root() / "output").resolve()
    local = base / version_slug / run_id
    if local.exists():
        raise FileExistsError(f"Run directory already exists; refusing overwrite: {local}")
    local.mkdir(parents=True, exist_ok=False)
    return RunContext(run_id, moment.isoformat(), version_slug, commit, branch, local)


def validate_drive_root(drive_output_root: Path, *, require_official: bool = False) -> Path:
    root = drive_output_root.expanduser().resolve()
    if require_official and str(drive_output_root) != str(OFFICIAL_DRIVE_OUTPUT_ROOT):
        raise AssertionError(f"Unexpected DRIVE_OUTPUT_ROOT: {drive_output_root}")
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"DRIVE_OUTPUT_ROOT does not exist: {root}")
    return root


def archive_run(local_run_dir: Path, drive_output_root: Path, run_id: str, *, require_official_root: bool = False) -> Path:
    root = validate_drive_root(drive_output_root, require_official=require_official_root)
    archive = root / run_id
    if archive.exists():
        raise FileExistsError(f"Drive archive already exists; refusing overwrite: {archive}")
    if archive.resolve() == local_run_dir.resolve():
        raise ValueError("Local output and Drive archive must be different directories")
    shutil.copytree(local_run_dir, archive)
    if archive.parent.resolve() != root:
        raise AssertionError("DRIVE_RUN_DIR parent differs from DRIVE_OUTPUT_ROOT")
    validate_archive(local_run_dir, archive)
    return archive


def validate_archive(local: Path, archive: Path) -> None:
    required = ["market_breadth_summary.xlsx", "daily_dataset.parquet", "run_info.txt", "validation_summary.md", "plots"]
    for name in required:
        source, target = local / name, archive / name
        if not source.exists() or not target.exists():
            raise AssertionError(f"Archive missing required item: {name}")
        if source.is_file() and (source.stat().st_size == 0 or target.stat().st_size != source.stat().st_size):
            raise AssertionError(f"Archive size mismatch: {name}")
    local_files = sorted(p.relative_to(local) for p in local.rglob("*") if p.is_file())
    archive_files = sorted(p.relative_to(archive) for p in archive.rglob("*") if p.is_file())
    if local_files != archive_files:
        raise AssertionError("Drive archive file manifest differs from local output")
