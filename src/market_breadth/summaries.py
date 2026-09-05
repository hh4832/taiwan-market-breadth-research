from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import PREDICTOR_SPECS, TARGET_METADATA, V6Config


def build_signal_definitions(config: V6Config | None = None) -> pd.DataFrame:
    cfg = config or V6Config()
    formulas = {
        "up_ratio": "up_count / (up_count + down_count)",
        "down_ratio": "down_count / (up_count + down_count)",
        "up_ratio_3d_mean": "mean(up_ratio[t-2:t])",
        "down_ratio_3d_mean": "mean(down_ratio[t-2:t])",
        "delta_up_ratio_1d": "up_ratio[t] - up_ratio[t-1]",
        "delta_down_ratio_1d": "down_ratio[t] - down_ratio[t-1]",
        "accel_up_ratio_1d": "delta_up[t] - delta_up[t-1]",
        "accel_down_ratio_1d": "delta_down[t] - delta_down[t-1]",
        "big_up_ratio": "count(return >= 5%) / valid_stock_count",
        "big_down_ratio": "count(return <= -5%) / valid_stock_count",
        "limit_up_ratio": "limit_up_count / valid_stock_count",
        "limit_down_ratio": "limit_down_count / valid_stock_count",
    }
    rows = []
    for name, (family, direction) in PREDICTOR_SPECS.items():
        rows.append({
            "name": name, "family": family, "direction": direction,
            "formula": formulas[name],
            "window": 3 if family == "ROLLING_3D" else (2 if family == "ACCELERATION" else 1),
            "PR_definition": f"rank of x[t] within trailing {cfg.pr_window}, including t; min history {cfg.min_history}",
            "Z_definition": f"(x[t]-rolling mean)/rolling sample sd over {cfg.z_window}, including t",
            "signal_availability": "after signal-day close",
        })
    return pd.DataFrame(rows)


def build_monotonicity_results(dataset: pd.DataFrame, config: V6Config | None = None) -> pd.DataFrame:
    cfg = config or V6Config()
    rows = []
    for predictor, (family, _) in PREDICTOR_SPECS.items():
        pr_col = f"{predictor}__PR_{cfg.pr_window}"
        for target in TARGET_METADATA:
            for regime in ("ALL", "BULL", "BEAR"):
                mask = pd.Series(True, index=dataset.index) if regime == "ALL" else dataset["market_regime"].eq(regime)
                frame = dataset.loc[mask, [pr_col, target]].dropna().copy()
                if len(frame) < 10:
                    continue
                frame["quintile"] = pd.cut(frame[pr_col], [0, .2, .4, .6, .8, 1], labels=[1, 2, 3, 4, 5], include_lowest=True)
                means = frame.groupby("quintile", observed=True)[target].mean().reindex([1, 2, 3, 4, 5])
                if means.isna().any():
                    continue
                diffs = np.diff(means.to_numpy())
                rho = stats.spearmanr(np.arange(1, 6), means.to_numpy()).statistic
                violations = int(min(np.count_nonzero(diffs < 0), np.count_nonzero(diffs > 0)))
                if abs(rho) >= 0.9 and violations <= 1:
                    shape = "monotonic"
                elif (means.iloc[0] > means.iloc[1:4].max()) or (means.iloc[4] > means.iloc[1:4].max()):
                    shape = "tail_effect"
                elif means.iloc[[0, 4]].mean() > means.iloc[[1, 2, 3]].mean():
                    shape = "U_shape"
                elif means.iloc[[0, 4]].mean() < means.iloc[[1, 2, 3]].mean():
                    shape = "inverted_U_shape"
                else:
                    shape = "unstable"
                ci_above = 0
                ci_below = 0
                for q in range(1, 6):
                    q_values = frame.loc[frame["quintile"] == q, target].to_numpy(float)
                    if len(q_values) > 1:
                        se = q_values.std(ddof=1) / np.sqrt(len(q_values))
                        ci_low, ci_high = q_values.mean() - 1.96 * se, q_values.mean() + 1.96 * se
                        ci_above += int(ci_low > 0)
                        ci_below += int(ci_high < 0)
                rows.append({
                    "predictor": predictor, "predictor_family": family, "target": target,
                    "market_regime": regime, "Q5_minus_Q1_mean_ret": means.iloc[4] - means.iloc[0],
                    "spearman_group_vs_mean": rho, "monotonic_violation_count": violations,
                    "positive_group_count": int((means > 0).sum()), "negative_group_count": int((means < 0).sum()),
                    "CI_above_zero_count": ci_above, "CI_below_zero_count": ci_below,
                    "shape": shape, **{f"Q{i}_mean": means.iloc[i - 1] for i in range(1, 6)},
                })
    return pd.DataFrame(rows)


def build_yearly_results(dataset: pd.DataFrame, results: pd.DataFrame, config: V6Config | None = None) -> pd.DataFrame:
    cfg = config or V6Config()
    rows = []
    key_cols = ["predictor", "signal_method", "group", "market_regime", "target"]
    focus_results = results.loc[results["group_type"].eq("tradable_tail")]
    for key in focus_results[key_cols].drop_duplicates().itertuples(index=False, name=None):
        predictor, method, group, regime, target = key
        signal_col = f"{predictor}__{method}_{cfg.pr_window if method == 'PR' else cfg.z_window}"
        threshold = float(focus_results.loc[(focus_results[key_cols] == pd.Series(key, index=key_cols)).all(axis=1), "group_threshold"].iloc[0])
        operator = "le" if "LE" in group else "ge"
        mask = dataset[signal_col].le(threshold) if operator == "le" else dataset[signal_col].ge(threshold)
        if regime != "ALL":
            mask &= dataset["market_regime"].eq(regime)
        frame = dataset.loc[mask, [target]].dropna().copy()
        frame["year"] = frame.index.year
        for year, values in frame.groupby("year")[target]:
            rows.append({**dict(zip(key_cols, key)), "year": int(year), "N": len(values), "mean": values.mean(), "median": values.median(), "win_rate": values.gt(0).mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    group_keys = key_cols
    contribution = out["mean"] * out["N"]
    out["return_sum_proxy"] = contribution
    summary = out.groupby(group_keys, dropna=False).agg(
        positive_year_count=("mean", lambda x: int((x > 0).sum())),
        negative_year_count=("mean", lambda x: int((x < 0).sum())),
        best_year_contribution=("return_sum_proxy", "max"),
        worst_year_contribution=("return_sum_proxy", "min"),
        total_abs_year_contribution=("return_sum_proxy", lambda x: float(np.abs(x).sum())),
    ).reset_index()
    summary["year_concentration_warning"] = summary["best_year_contribution"].abs().div(summary["total_abs_year_contribution"].replace(0, np.nan)).gt(0.5)
    return out.merge(summary, on=group_keys, how="left")
