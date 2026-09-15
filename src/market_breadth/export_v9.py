from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import V9Config, V9_TARGET_METADATA
from .run_context import REPOSITORY, RunContext
from .v9 import validation_summary_markdown


def build_v9_metadata(
    config: V9Config, context: RunContext, dataset: pd.DataFrame, selected: dict,
    breadth_meta: dict, drive_output_root: Path | None = None,
) -> pd.DataFrame:
    sys = __import__("sys")
    drive_run_dir = drive_output_root / context.run_id if drive_output_root is not None else None
    values = {
        "study_version": config.study_version, "version_slug": config.version_slug,
        "run_id": context.run_id, "run_timestamp_asia_taipei": context.timestamp,
        "git_commit": context.git_commit, "git_branch": context.git_branch, "repository": REPOSITORY,
        "python_version": sys.version.replace("\n", " "), "python_executable": sys.executable,
        "python_baseline": "3.11", "python_runtime_deviation": sys.version_info[:2] != (3, 11),
        "actual_start_date": dataset.index.min(), "actual_end_date": dataset.index.max(),
        "sample_start_date": dataset.index.min(), "sample_end_date": dataset.index.max(),
        "sample_rule": "post_2015_10pct_limit_regime",
        "price_source_open": selected.get("adj_open"), "price_source_close": selected.get("adj_close"),
        "outcome_price_adjusted": selected.get("adj_open") == "etl:adj_open" and selected.get("adj_close") == "etl:adj_close",
        "signal_availability": "after t close", "formal_entry": "Open[t+1]",
        "pr_windows": ",".join(map(str, config.pr_windows)),
        "signal_mean_windows": ",".join(map(str, config.signal_mean_windows)),
        "pr_thresholds": "5,20,40,60,80,95", "pr_reference": "historical distribution t-1 and earlier; excludes t",
        "predictor_families": "LIMIT,EXTREME_5PCT,DELTA", "targets": ",".join(V9_TARGET_METADATA),
        "HAC_lags": ",".join(f"{k}:{v['hac_lag']}" for k, v in V9_TARGET_METADATA.items()),
        "multiple_testing_definition": "mean-return and win-rate separate; global plus predictor_family×target×PR_window×mean_window×raw/delta×overlap family",
        "win_rate_statistical_methods": "signal-vs-non-signal LPM HAC primary; binomial vs 50% and 2x2 odds ratio supplementary",
        "local_run_dir": str(context.local_run_dir),
        "drive_output_root": str(drive_output_root) if drive_output_root is not None else "not configured",
        "drive_run_dir": str(drive_run_dir) if drive_run_dir is not None else "not configured",
        "limit_up/down_detection_rule": "corporate-action-aware reference price + Taiwan tick rounding + 10% rule",
        "limit_status_is_approximation": False, "big_move_threshold": config.big_move_threshold,
        "big_move_operator": "strict > +0.05 / < -0.05", "big_move_denominator": "valid_stock_count; flat included; missing excluded",
        "cache_version": "v9_post2015_pr_excludes_t_adjusted_outcomes",
        **{f"breadth_{k}": v for k, v in breadth_meta.items()}, **{f"finlab_{k}": v for k, v in selected.items()},
    }
    return pd.DataFrame(values.items(), columns=["item", "value"])


def export_v9(
    dataset: pd.DataFrame, results: pd.DataFrame, bins: pd.DataFrame, yearly: pd.DataFrame,
    definitions: pd.DataFrame, metadata: pd.DataFrame, validations: dict[str, pd.DataFrame],
    context: RunContext,
) -> dict[str, Path]:
    out = context.local_run_dir
    paths = {
        "summary": out / "market_breadth_summary.xlsx", "dataset": out / "daily_dataset.parquet",
        "run_info": out / "run_info.txt", "validation_summary": out / "validation_summary.md",
        "plots": out / "plots", "win_rate": out / "win_rate_results.parquet",
        "yearly": out / "yearly_results.parquet", "registry": out / "hypothesis_registry.csv",
        "definitions": out / "signal_definitions.csv",
        "all_results": out / "all_results_v9.parquet",
        "deduplicated": out / "deduplicated_hypotheses.parquet",
    }
    paths["plots"].mkdir(exist_ok=False)
    dataset.to_parquet(paths["dataset"])
    results.to_parquet(paths["win_rate"])
    results.to_parquet(paths["all_results"])
    results.loc[~results.is_duplicate_hypothesis].to_parquet(paths["deduplicated"])
    yearly.to_parquet(paths["yearly"])
    results.to_csv(paths["registry"], index=False)
    definitions.to_csv(paths["definitions"], index=False)
    canonical = results.loc[~results.is_duplicate_hypothesis]
    sheets = {
        "all_results_v9": results, "mean_return_results": results,
        "win_rate_results": results, "signal_vs_non_signal": results,
        "pr_bin_results": bins, "yearly_results": yearly,
        "non_overlapping_results": results.loc[results.overlap_policy.eq("non_overlapping_events")],
        "deduplicated_hypotheses": canonical, "signal_definitions": definitions,
        "target_definitions": pd.DataFrame([{"target": k, **v} for k, v in V9_TARGET_METADATA.items()]),
        "metadata": metadata,
    }
    sheets.update({("val_" + k)[:31]: v for k, v in validations.items()})
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            writer.sheets[name[:31]].freeze_panes = "A2"
            writer.sheets[name[:31]].auto_filter.ref = writer.sheets[name[:31]].dimensions
    paths["run_info"].write_text("\n".join(f"{r.item}: {r.value}" for r in metadata.itertuples()), encoding="utf-8")
    paths["validation_summary"].write_text(validation_summary_markdown(results, yearly), encoding="utf-8")
    return paths
