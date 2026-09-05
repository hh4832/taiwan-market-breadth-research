from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import PREDICTOR_SPECS, TARGET_METADATA, V6Config
from .summaries import build_signal_definitions


def git_info() -> tuple[str, str]:
    def run(args: list[str], default: str) -> str:
        try:
            return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip() or default
        except Exception:
            return default
    return run(["git", "rev-parse", "HEAD"], "uncommitted"), run(["git", "branch", "--show-current"], "unknown")


def build_metadata(config: V6Config, dataset: pd.DataFrame, breadth_metadata: dict, selected_keys: dict) -> pd.DataFrame:
    commit, branch = git_info()
    now = datetime.now(ZoneInfo("Asia/Taipei"))
    items = {
        "study_version": "v6 Breadth Dynamics & Extreme Breadth Study",
        "analysis_type": "signal discovery; not a composite trading strategy",
        "run_timestamp_asia_taipei": now.isoformat(),
        "git_commit": commit,
        "git_branch": branch,
        "start_date_config": config.start_date,
        "end_date_config": config.end_date or "latest",
        "actual_start_date": dataset.index.min(),
        "actual_end_date": dataset.index.max(),
        "rolling_pr": f"trailing {config.pr_window}, includes t, min_history={config.min_history}",
        "rolling_z": f"trailing {config.z_window}, includes t, min_history={config.min_history}",
        "regime": f"Close[t] vs MA{config.ma_window}[t]",
        "multiple_testing_family": "predictor_family × target × regime × signal_method; hypotheses corrected separately",
        **{f"breadth_{k}": v for k, v in breadth_metadata.items()},
        **{f"finlab_{k}": v for k, v in selected_keys.items()},
    }
    return pd.DataFrame(items.items(), columns=["item", "value"])


def export_results(
    dataset: pd.DataFrame,
    results: pd.DataFrame,
    monotonicity: pd.DataFrame,
    yearly: pd.DataFrame,
    metadata: pd.DataFrame,
    validations: dict[str, pd.DataFrame],
    config: V6Config,
) -> dict[str, Path]:
    out = config.output_dir
    plots = out / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(exist_ok=True)
    paths = {
        "summary": out / "market_breadth_summary.xlsx",
        "dataset": out / "daily_dataset.parquet",
        "run_info": out / "run_info.txt",
        "plots": plots,
    }
    dataset.to_parquet(paths["dataset"])
    pearson = dataset[list(PREDICTOR_SPECS)].corr("pearson")
    spearman = dataset[list(PREDICTOR_SPECS)].corr("spearman")
    sheets = {
        "all_results_v6": results,
        "mean_vs_zero_results": results,
        "group_vs_non_group": results,
        "group_vs_unconditional": results,
        "regime_comparison": results.loc[results["market_regime"].isin(["BULL", "BEAR"])],
        "predictor_pearson": pearson.reset_index(),
        "predictor_spearman": spearman.reset_index(),
        "monotonicity_results": monotonicity,
        "signal_definitions": build_signal_definitions(config),
        "yearly_results": yearly,
        "metadata": metadata,
        "target_definitions": pd.DataFrame([{"target": k, **v} for k, v in TARGET_METADATA.items()]),
    }
    for name, frame in validations.items():
        sheets[("val_" + name)[:31]] = frame
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
    paths["run_info"].write_text("\n".join(f"{row.item}: {row.value}" for row in metadata.itertuples()), encoding="utf-8")
    return paths


def archive_to_drive(paths: dict[str, Path], repo_name: str, drive_root: Path = Path("/content/drive/MyDrive/Quant_Research")) -> Path:
    commit, _ = git_info()
    stamp = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    archive = drive_root / repo_name / f"{stamp}_{commit[:12]}"
    archive.mkdir(parents=True, exist_ok=False)
    for key in ("summary", "dataset", "run_info"):
        shutil.copy2(paths[key], archive / paths[key].name)
    if paths["plots"].exists():
        shutil.copytree(paths["plots"], archive / "plots")
    return archive
