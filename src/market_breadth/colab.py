from __future__ import annotations

import subprocess
from pathlib import Path


EXPECTED_REPOSITORY_FRAGMENT = "hh4832/taiwan-market-breadth-research"


def validate_existing_clone(repo_dir: Path) -> str:
    """Return the safe remote or refuse removal of an unexpected path."""
    if not repo_dir.exists():
        return ""
    if not (repo_dir / ".git").is_dir():
        raise RuntimeError(f"Refusing to remove non-git path: {repo_dir}")
    remote = subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "get-url", "origin"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if EXPECTED_REPOSITORY_FRAGMENT not in remote:
        raise RuntimeError(f"Refusing to remove unexpected repository: {remote}")
    return remote
