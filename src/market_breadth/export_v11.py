from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .config import V11Config
from .run_context import REPOSITORY, RunContext


def build_v11_metadata(config: V11Config, context: RunContext, dataset: pd.DataFrame,
                       selected: dict, breadth_meta: dict, drive_root: Path | None) -> pd.DataFrame:
    drive_run = drive_root / context.run_id if drive_root else None
    candidates = {
        "candidate_A_definition": "big_up_ratio mean5 level PR126 bin60-80; O1-C3/C5",
        "candidate_B_definition": "big_up_ratio mean5 level PR252 bin60-80; O1-C3",
        "candidate_C_definition": "up_ratio mean1 level PR60/126/252 bin95-100; O1-C1 win rate",
    }
    values = {"study_version": config.study_version, "version_slug": config.version_slug, "run_id": context.run_id,
              "git_commit": context.git_commit, "branch": context.git_branch, "repository": REPOSITORY,
              "timestamp": context.timestamp, "python_version": sys.version.replace("\n", " "), "python_baseline": "3.11",
              "module_A_prior_return": True, "module_B_pr_mapping": True, "module_C_regime_comparison": True,
              **candidates, "prior_return_windows": "1,3,5,10", "PR_windows": "60,126,252",
              "regime_change_date": config.regime_change_date, "pre_limit_rate": config.old_limit_rate,
              "post_limit_rate": config.current_limit_rate, "price_source": "adjusted 0050 etl:adj_open/etl:adj_close",
              "price_source_open": selected.get("adj_open"), "price_source_close": selected.get("adj_close"),
              "signal_timing": "after t close", "formal_entry": "O1", "sample_start": dataset.index.min(),
              "sample_end": dataset.index.max(), "local_run_dir": str(context.local_run_dir),
              "drive_output_root": str(drive_root) if drive_root else "not configured",
              "drive_run_dir": str(drive_run) if drive_run else "not configured",
              **{f"breadth_{k}": v for k, v in breadth_meta.items()}, **{f"finlab_{k}": v for k, v in selected.items()}}
    return pd.DataFrame(values.items(), columns=["item", "value"])


def _table(frame: pd.DataFrame, columns: list[str], limit: int = 40) -> str:
    frame = frame[[c for c in columns if c in frame]].head(limit)
    if frame.empty: return "(no rows)"
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def mechanism_summary(regressions: pd.DataFrame, threshold: pd.DataFrame, signal_raw: pd.DataFrame,
                      regime: pd.DataFrame, interactions: pd.DataFrame, candidates: pd.DataFrame) -> str:
    adjusted = regressions.loc[regressions.model.ne("breadth_only")]
    up95 = threshold.loc[threshold.predictor.eq("up_ratio__mean_1d")]
    big = threshold.loc[threshold.predictor.eq("big_up_ratio__mean_5d")]
    broad = interactions.loc[interactions.predictor.str.startswith("down_ratio__")]
    lines = ["# v11 Breadth Mechanism & Regime Reconciliation", "",
             "## Candidate decisions", "", _table(candidates, list(candidates.columns)), "",
             "## 1. big_up_ratio 5D PR60–80 after prior-return controls", "",
             _table(adjusted.loc[adjusted.candidate_group.isin(["A","B"])], ["candidate","target","model","N","breadth_beta","breadth_HAC_p","attenuation_ratio","mechanism_label"]), "",
             "## 2. up_ratio 1D PR95+ exhaustion after prior-rally controls", "",
             _table(adjusted.loc[adjusted.candidate_group.eq("C")], ["candidate","model","N","breadth_beta","breadth_HAC_p","attenuation_ratio","mechanism_label"]), "",
             "## 3. up_ratio PR95 raw mapping", "", _table(up95, ["pr_window","N","mean","median","P25","P75","min","max"]), "",
             "## 4. big_up_ratio 5D PR60/80 raw mapping", "", _table(big, ["pr_window","quantile","N","median","P25","P75","min","max"]), "",
             "## Signal-date actual raw values", "", _table(signal_raw, list(signal_raw.columns)), "",
             "## 5–7. Broad-decline pre/post regime and interaction", "",
             f"Down-ratio interaction raw/family-FDR significant rows: {int(broad.interaction_HAC_p.lt(.05).sum())}/{int(broad.interaction_FDR_family.lt(.05).sum())}.", "",
             _table(broad, ["predictor","mean_window","pr_window","pr_bin","target","endpoint","N","beta_signal_pre","beta_interaction","interaction_HAC_p","interaction_FDR_family"]), "",
             "A structural break is not inferred from pre-significant/post-nonsignificant results alone; interaction, effect size and pre-2015 sample size govern the classification.", "",
             "## Limitations", "",
             "This is targeted mechanism validation, not a new alpha grid. Results remain exposed to survivorship bias, event clustering, regime concentration, costs/slippage and lack of untouched OOS/walk-forward evidence."]
    return "\n".join(lines) + "\n"


def export_v11(dataset: pd.DataFrame, prior: pd.DataFrame, regressions: pd.DataFrame,
               threshold: pd.DataFrame, signal_raw: pd.DataFrame, threshold_yearly: pd.DataFrame,
               regime: pd.DataFrame, interactions: pd.DataFrame, regime_yearly: pd.DataFrame,
               candidates: pd.DataFrame, metadata: pd.DataFrame, validations: dict[str, pd.DataFrame],
               context: RunContext) -> dict[str, Path]:
    out = context.local_run_dir
    paths = {"summary": out/"market_breadth_summary.xlsx", "dataset": out/"daily_dataset.parquet",
             "prior": out/"prior_return_analysis.parquet", "regressions": out/"prior_return_regressions.parquet",
             "threshold": out/"pr_raw_threshold_mapping.parquet", "signal_raw": out/"pr_signal_raw_values.parquet",
             "threshold_yearly": out/"pr_threshold_yearly.parquet", "regime": out/"regime_comparison.parquet",
             "interactions": out/"regime_interaction_results.parquet", "regime_yearly": out/"regime_yearly_results.parquet",
             "mechanism_summary": out/"mechanism_summary.md", "validation_summary": out/"validation_summary.md",
             "run_info": out/"run_info.txt", "plots": out/"plots"}
    paths["plots"].mkdir(exist_ok=False)
    for frame, key in ((dataset,"dataset"),(prior,"prior"),(regressions,"regressions"),(threshold,"threshold"),(signal_raw,"signal_raw"),
                       (threshold_yearly,"threshold_yearly"),(regime,"regime"),(interactions,"interactions"),(regime_yearly,"regime_yearly")):
        frame.to_parquet(paths[key])
    validation = pd.concat(validations.values(), ignore_index=True)
    sheets = {"prior_return_descriptive": prior, "prior_return_regression": regressions,
              "pr_threshold_mapping": threshold, "pr_threshold_yearly": threshold_yearly,
              "signal_raw_ratio_mapping": signal_raw,
              "regime_broad_decline": regime.loc[regime.predictor.str.startswith("down_ratio__")],
              "regime_broad_rally": regime.loc[regime.predictor.str.startswith("up_ratio__")],
              "regime_big": regime.loc[regime.predictor.str.startswith(("big_up_ratio__","big_down_ratio__"))],
              "regime_limit": regime.loc[regime.predictor.str.startswith(("limit_up_ratio__","limit_down_ratio__"))],
              "regime_interactions": interactions, "candidate_summary": candidates,
              "validation_summary": validation, "metadata": metadata}
    with pd.ExcelWriter(paths["summary"], engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False); sheet=writer.sheets[name[:31]]; sheet.freeze_panes="A2"; sheet.auto_filter.ref=sheet.dimensions
    paths["run_info"].write_text("\n".join(f"{r.item}: {r.value}" for r in metadata.itertuples()), encoding="utf-8")
    paths["mechanism_summary"].write_text(mechanism_summary(regressions, threshold, signal_raw, regime, interactions, candidates), encoding="utf-8")
    paths["validation_summary"].write_text("# v11 Validation Summary\n\n" + _table(validation, list(validation.columns)) + "\n", encoding="utf-8")
    return paths
