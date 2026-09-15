from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import V10Config, V10_RAW_PREDICTORS, V9_TARGET_METADATA
from .run_context import REPOSITORY, RunContext
from .v10 import PR_BINS, validation_summary_markdown


def build_v10_metadata(
    config: V10Config, context: RunContext, dataset: pd.DataFrame,
    selected: dict, breadth_meta: dict, drive_output_root: Path | None,
) -> pd.DataFrame:
    drive_run = drive_output_root / context.run_id if drive_output_root is not None else None
    values = {
        "study_version": config.study_version, "version_slug": config.version_slug,
        "run_id": context.run_id, "timestamp": context.timestamp,
        "git_commit": context.git_commit, "branch": context.git_branch, "repository": REPOSITORY,
        "python_version": sys.version.replace("\n", " "), "python_executable": sys.executable,
        "python_baseline": "3.11", "python_runtime_deviation": sys.version_info[:2] != (3, 11),
        "sample_start": dataset.index.min(), "sample_end": dataset.index.max(),
        "sample_rule": "post_2015_10pct_limit_regime",
        "price_source_open": selected.get("adj_open"), "price_source_close": selected.get("adj_close"),
        "outcome_price_adjusted": selected.get("adj_open") == "etl:adj_open" and selected.get("adj_close") == "etl:adj_close",
        "signal_availability": "after t close", "formal_entry": "Open[t+1]",
        "predictors": ",".join(V10_RAW_PREDICTORS),
        "mean_windows": ",".join(map(str, config.signal_mean_windows)), "transforms": "level,delta",
        "PR_windows": ",".join(map(str, config.pr_windows)),
        "PR_bins": ",".join(name for name, _, _ in PR_BINS),
        "PR_reference": "strict preceding window [t-window,t-1]; current t excluded",
        "targets": ",".join(V9_TARGET_METADATA),
        "HAC_lags": ",".join(f"{k}:{v['hac_lag']}" for k, v in V9_TARGET_METADATA.items()),
        "multiple_testing_family_definition": "predictor_family×target×PR_window×mean_window×transform×overlap_policy; ordered bins together",
        "win_rate_method": "bin-vs-non-bin LPM HAC primary; 2x2 odds ratio and binomial vs 50% supplementary",
        "dedup_method": "exact signal-date mask hash × target × overlap policy",
        "local_run_dir": str(context.local_run_dir),
        "drive_output_root": str(drive_output_root) if drive_output_root else "not configured",
        "drive_run_dir": str(drive_run) if drive_run else "not configured",
        "limit_rule": "corporate-action-aware previous valid raw close, tick rounding, post-2015 10%",
        "big_move_rule": "strict return > +0.05 / < -0.05; valid-stock denominator",
        **{f"breadth_{k}": v for k, v in breadth_meta.items()},
        **{f"finlab_{k}": v for k, v in selected.items()},
    }
    return pd.DataFrame(values.items(), columns=["item", "value"])


def export_v10(
    dataset: pd.DataFrame, results: pd.DataFrame, shape: pd.DataFrame,
    yearly: pd.DataFrame, definitions: pd.DataFrame, metadata: pd.DataFrame,
    validations: dict[str, pd.DataFrame], context: RunContext,
) -> dict[str, Path]:
    out = context.local_run_dir
    canonical = results.loc[~results.is_duplicate_hypothesis]
    non_overlap = results.loc[results.overlap_policy.eq("non_overlapping_events")]
    metrics = ["N", "mean_difference", "bin_minus_non_bin_win_rate", "Cohen_d"]
    participation = canonical.groupby(["predictor_family", "direction", "target"], as_index=False)[metrics].mean()
    level_delta = canonical.groupby(["transform", "predictor_family", "target"], as_index=False)[metrics].mean()
    horizons = canonical.groupby(["predictor", "target"], as_index=False)[metrics].mean()
    paths = {
        "summary": out / "market_breadth_summary.xlsx", "dataset": out / "daily_dataset.parquet",
        "run_info": out / "run_info.txt", "validation_summary": out / "validation_summary.md",
        "plots": out / "plots", "all_results": out / "all_results_v10.parquet",
        "win_rate": out / "win_rate_results_v10.parquet", "pr_bins": out / "pr_bin_results_v10.parquet",
        "yearly": out / "yearly_results_v10.parquet", "non_overlap": out / "non_overlapping_results_v10.parquet",
        "deduplicated": out / "deduplicated_hypotheses_v10.parquet",
        "definitions": out / "signal_definitions.csv", "registry": out / "hypothesis_registry.csv",
    }
    paths["plots"].mkdir(exist_ok=False)
    dataset.to_parquet(paths["dataset"])
    results.to_parquet(paths["all_results"]); results.to_parquet(paths["win_rate"]); results.to_parquet(paths["pr_bins"])
    yearly.to_parquet(paths["yearly"]); non_overlap.to_parquet(paths["non_overlap"]); canonical.to_parquet(paths["deduplicated"])
    definitions.to_csv(paths["definitions"], index=False); results.to_csv(paths["registry"], index=False)
    target_defs = pd.DataFrame([{"target": key, **value} for key, value in V9_TARGET_METADATA.items()])
    sheets = {
        "all_results_v10": results, "mean_return_results": results, "win_rate_results": results,
        "pr_bin_results": results, "pr_bin_shape": shape,
        "participation_vs_intensity": participation, "level_vs_delta": level_delta,
        "horizon_results": horizons, "yearly_results": yearly, "non_overlapping": non_overlap,
        "deduplicated_hypotheses": canonical, "signal_definitions": definitions,
        "target_definitions": target_defs, "metadata": metadata,
        "validation_summary": pd.concat(validations.values(), ignore_index=True),
    }
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            sheet = writer.sheets[name[:31]]; sheet.freeze_panes = "A2"; sheet.auto_filter.ref = sheet.dimensions
    paths["run_info"].write_text("\n".join(f"{r.item}: {r.value}" for r in metadata.itertuples()), encoding="utf-8")
    paths["validation_summary"].write_text(validation_summary_markdown(results, shape, yearly), encoding="utf-8")
    return paths
