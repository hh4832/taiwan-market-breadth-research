from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import V10_RAW_PREDICTORS, V11Config, V9_TARGET_METADATA
from .statistics import _bh_bonferroni
from .v10 import _one_bin_result, pr_bin_mask


CANDIDATES = (
    {"candidate": "A", "raw_predictor": "big_up_ratio", "predictor": "big_up_ratio__mean_5d", "mean_window": 5,
     "pr_window": 126, "pr_bin": "BIN_60_80", "lower": .60, "upper": .80,
     "targets": ("ret_o1_c3", "ret_o1_c5"), "controls": (3, 5, 10), "endpoint": "return"},
    {"candidate": "B", "raw_predictor": "big_up_ratio", "predictor": "big_up_ratio__mean_5d", "mean_window": 5,
     "pr_window": 252, "pr_bin": "BIN_60_80", "lower": .60, "upper": .80,
     "targets": ("ret_o1_c3",), "controls": (3, 5, 10), "endpoint": "return"},
    {"candidate": "C60", "candidate_group": "C", "raw_predictor": "up_ratio", "predictor": "up_ratio__mean_1d", "mean_window": 1,
     "pr_window": 60, "pr_bin": "BIN_95_100", "lower": .95, "upper": 1.,
     "targets": ("ret_o1_c1",), "controls": (1, 3, 5), "endpoint": "win"},
    {"candidate": "C126", "candidate_group": "C", "raw_predictor": "up_ratio", "predictor": "up_ratio__mean_1d", "mean_window": 1,
     "pr_window": 126, "pr_bin": "BIN_95_100", "lower": .95, "upper": 1.,
     "targets": ("ret_o1_c1",), "controls": (1, 3, 5), "endpoint": "win"},
    {"candidate": "C252", "candidate_group": "C", "raw_predictor": "up_ratio", "predictor": "up_ratio__mean_1d", "mean_window": 1,
     "pr_window": 252, "pr_bin": "BIN_95_100", "lower": .95, "upper": 1.,
     "targets": ("ret_o1_c1",), "controls": (1, 3, 5), "endpoint": "win"},
)

HIGH_BINS = (("BIN_60_80", .60, .80), ("BIN_80_95", .80, .95), ("BIN_95_100", .95, 1.0))


def add_prior_returns(dataset: pd.DataFrame, adjusted_close: pd.Series, windows: Iterable[int] = (1, 3, 5, 10)) -> pd.DataFrame:
    """Prior returns use adjusted Close[t] and only prices at t or earlier."""
    out = dataset.copy()
    close = pd.to_numeric(adjusted_close, errors="coerce").reindex(out.index)
    for window in windows:
        out[f"prior_ret_{window}d"] = close.div(close.shift(window)).sub(1)
    return out


def candidate_signal(dataset: pd.DataFrame, candidate: dict) -> pd.Series:
    pr = dataset[f"{candidate['predictor']}__PR_{candidate['pr_window']}"]
    return pr_bin_mask(pr, float(candidate["lower"]), float(candidate["upper"])) & pr.notna()


def build_prior_return_descriptive(dataset: pd.DataFrame, config: V11Config) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prior_cols = [f"prior_ret_{w}d" for w in config.prior_return_windows]
    for candidate in CANDIDATES:
        signal = candidate_signal(dataset, candidate)
        valid_pr = dataset[f"{candidate['predictor']}__PR_{candidate['pr_window']}"].notna()
        for group, mask in (("signal", signal), ("non_signal", valid_pr & ~signal)):
            for column in prior_cols:
                values = dataset.loc[mask & dataset[column].notna(), column]
                if values.empty:
                    continue
                rows.append({"candidate": candidate["candidate"], "candidate_group": candidate.get("candidate_group", candidate["candidate"]),
                             "pr_window": candidate["pr_window"], "group": group, "prior_window": int(column.split("_")[-1][:-1]),
                             "N": len(values), "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
                             "q05": values.quantile(.05), "q25": values.quantile(.25), "q50": values.quantile(.50),
                             "q75": values.quantile(.75), "q95": values.quantile(.95),
                             "percentile_mean": values.rank(pct=True).mean()})
    return pd.DataFrame(rows)


def ols_hac(y: np.ndarray, design: np.ndarray, names: list[str], lag: int) -> pd.DataFrame:
    """OLS with Bartlett/Newey-West HAC covariance and explicit coefficient extraction."""
    y = np.asarray(y, float); x = np.asarray(design, float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y, x = y[valid], x[valid]
    if len(y) <= x.shape[1] + lag or np.linalg.matrix_rank(x) < x.shape[1]:
        return pd.DataFrame(columns=["term", "beta", "HAC_se", "HAC_t", "HAC_p", "ci_lower", "ci_upper", "N"])
    xtx_inv = np.linalg.pinv(x.T @ x); beta = xtx_inv @ x.T @ y; residual = y - x @ beta
    score = x * residual[:, None]
    meat = score.T @ score
    for k in range(1, min(lag, len(y) - 1) + 1):
        weight = 1 - k / (lag + 1)
        gamma = score[k:].T @ score[:-k]
        meat += weight * (gamma + gamma.T)
    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(covariance), 0)); t_values = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * stats.norm.sf(np.abs(t_values))
    return pd.DataFrame({"term": names, "beta": beta, "HAC_se": se, "HAC_t": t_values, "HAC_p": p_values,
                         "ci_lower": beta - 1.96 * se, "ci_upper": beta + 1.96 * se, "N": len(y)})


def run_prior_return_regressions(dataset: pd.DataFrame, candidates: Iterable[dict] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in (candidates or CANDIDATES):
        pr_col = f"{candidate['predictor']}__PR_{candidate['pr_window']}"
        signal = candidate_signal(dataset, candidate).astype(float).where(dataset[pr_col].notna())
        for target in candidate["targets"]:
            lag = int(V9_TARGET_METADATA[target]["hac_lag"])
            y_base = dataset[target].gt(0).astype(float) if candidate["endpoint"] == "win" else dataset[target]
            unadjusted_beta = np.nan
            models = [("breadth_only", None)] + [(f"plus_prior_ret_{w}d", w) for w in candidate["controls"]]
            for model, control_window in models:
                columns = [y_base.rename("y"), signal.rename("breadth")]
                names = ["const", "breadth_bin"]
                if control_window is not None:
                    columns.append(dataset[f"prior_ret_{control_window}d"]); names.append(f"prior_ret_{control_window}d")
                frame = pd.concat(columns, axis=1).dropna()
                design = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1:].to_numpy(float)])
                fit = ols_hac(frame.iloc[:, 0].to_numpy(float), design, names, lag)
                if fit.empty:
                    continue
                breadth = fit.loc[fit.term.eq("breadth_bin")].iloc[0]
                if control_window is None:
                    unadjusted_beta = float(breadth.beta)
                attenuation = 1 - float(breadth.beta) / unadjusted_beta if control_window is not None and np.isfinite(unadjusted_beta) and unadjusted_beta != 0 else 0.0
                mechanism = "BREADTH_ONLY_BASELINE" if control_window is None else ("PRIOR_RETURN_MEDIATED" if abs(attenuation) >= .5 else "INCREMENTAL_BREADTH_SUPPORT")
                rows.append({"candidate": candidate["candidate"], "candidate_group": candidate.get("candidate_group", candidate["candidate"]),
                             "target": target, "endpoint": candidate["endpoint"], "pr_window": candidate["pr_window"],
                             "model": model, "control_window": control_window, "N": int(breadth.N),
                             "breadth_beta": breadth.beta, "breadth_HAC_se": breadth.HAC_se, "breadth_HAC_t": breadth.HAC_t,
                             "breadth_HAC_p": breadth.HAC_p, "breadth_ci_lower": breadth.ci_lower,
                             "breadth_ci_upper": breadth.ci_upper, "unadjusted_beta": unadjusted_beta,
                             "attenuation_ratio": attenuation, "mechanism_label": mechanism})
    return pd.DataFrame(rows)


def historical_raw_threshold(series: pd.Series, window: int, quantile: float) -> pd.Series:
    """Empirical order-statistic threshold from exactly [t-window, ..., t-1]."""
    def threshold(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) != window:
            return np.nan
        rank = max(1, int(math.ceil(quantile * len(values))))
        return float(np.sort(values)[rank - 1])
    return pd.to_numeric(series, errors="coerce").shift(1).rolling(window, min_periods=window).apply(threshold, raw=True)


def build_pr_threshold_mapping(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = [("up_ratio__mean_1d", w, .95) for w in (60, 126, 252)]
    definitions += [("big_up_ratio__mean_5d", w, q) for w in (126, 252) for q in (.60, .80)]
    daily_parts: list[pd.DataFrame] = []
    for predictor, window, quantile in definitions:
        threshold = historical_raw_threshold(dataset[predictor], window, quantile)
        daily_parts.append(pd.DataFrame({"date": dataset.index, "predictor": predictor, "pr_window": window,
                                         "quantile": quantile, "raw_threshold": threshold.to_numpy()}))
    daily = pd.concat(daily_parts, ignore_index=True).dropna(subset=["raw_threshold"])
    summary = daily.groupby(["predictor", "pr_window", "quantile"], as_index=False).raw_threshold.agg(
        N="count", mean="mean", median="median", std="std", min="min", max="max",
        P10=lambda x: x.quantile(.10), P25=lambda x: x.quantile(.25), P50=lambda x: x.quantile(.50),
        P75=lambda x: x.quantile(.75), P90=lambda x: x.quantile(.90))
    daily["year"] = pd.to_datetime(daily.date).dt.year
    yearly = daily.groupby(["predictor", "pr_window", "quantile", "year"], as_index=False).raw_threshold.agg(
        median_threshold="median", P25=lambda x: x.quantile(.25), P75=lambda x: x.quantile(.75))
    yearly["IQR"] = yearly.P75 - yearly.P25
    signal_rows: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        values = dataset.loc[candidate_signal(dataset, candidate), candidate["predictor"]].dropna()
        signal_rows.append({"candidate": candidate["candidate"], "candidate_group": candidate.get("candidate_group", candidate["candidate"]),
                            "predictor": candidate["predictor"], "pr_window": candidate["pr_window"], "pr_bin": candidate["pr_bin"],
                            "N": len(values), "mean": values.mean(), "median": values.median(), "P25": values.quantile(.25),
                            "P75": values.quantile(.75), "min": values.min(), "max": values.max()})
    return summary, pd.DataFrame(signal_rows), yearly, daily


def _apply_module_c_fdr(frame: pd.DataFrame, p_col: str, prefix: str, family_cols: list[str]) -> pd.DataFrame:
    out = frame.copy(); fdr, bonf = _bh_bonferroni(out[p_col]); out[f"{prefix}_FDR_global"] = fdr; out[f"{prefix}_Bonferroni_global"] = bonf
    out[f"{prefix}_FDR_family"] = np.nan; out[f"{prefix}_Bonferroni_family"] = np.nan
    for _, idx in out.groupby(family_cols, dropna=False).groups.items():
        fdr, bonf = _bh_bonferroni(out.loc[idx, p_col]); out.loc[idx, f"{prefix}_FDR_family"] = fdr; out.loc[idx, f"{prefix}_Bonferroni_family"] = bonf
    return out


def run_regime_comparison(
    dataset: pd.DataFrame, config: V11Config, *,
    predictors: Iterable[str] | None = None, mean_windows: Iterable[int] | None = None,
    pr_windows: Iterable[int] | None = None, bins: Iterable[tuple[str, float, float]] | None = None,
    targets: Iterable[str] | None = None, show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []; interactions: list[dict[str, object]] = []; yearly_rows: list[dict[str, object]] = []
    change = pd.Timestamp(config.regime_change_date); post = pd.Series(dataset.index >= change, index=dataset.index)
    raw_names = list(predictors or V10_RAW_PREDICTORS)
    for position, raw in enumerate(raw_names, start=1):
        if show_progress:
            print(f"Regime predictor [{position}/{len(raw_names)}]: {raw}", flush=True)
        family, direction = V10_RAW_PREDICTORS[raw]
        for mean_window in (mean_windows or config.signal_mean_windows):
            predictor = f"{raw}__mean_{mean_window}d"
            for pr_window in (pr_windows or config.pr_windows):
                pr_col = f"{predictor}__PR_{pr_window}"
                for bin_name, lower, upper in (bins or HIGH_BINS):
                    full_signal = pr_bin_mask(dataset[pr_col], lower, upper)
                    for target in (targets or ("ret_o1_c1", "ret_o1_c3", "ret_o1_c5")):
                        lag = int(V9_TARGET_METADATA[target]["hac_lag"])
                        for regime_name, regime_mask in (("PRE_2015_7PCT", ~post), ("POST_2015_10PCT", post)):
                            base = dataset.loc[regime_mask]
                            row = _one_bin_result(base, predictor, pr_col, target, lower, upper, bin_name, family, direction,
                                                  mean_window, "level", pr_window, "all_events", config.annual_rf, lag)
                            if row is not None:
                                row.update({"regime": regime_name, "sample_size_flag": row["sample_size_flag"]}); rows.append(row)
                        signal_for_model = full_signal.astype(float).where(dataset[pr_col].notna()).rename("signal")
                        valid = pd.concat([dataset[target], signal_for_model, post.rename("post")], axis=1).dropna()
                        signal = valid.signal.astype(float).to_numpy(); post_values = valid.post.astype(float).to_numpy()
                        design = np.column_stack([np.ones(len(valid)), signal, post_values, signal * post_values])
                        for endpoint, y in (("mean_return", valid[target].to_numpy(float)), ("win_rate", valid[target].gt(0).to_numpy(float))):
                            fit = ols_hac(y, design, ["const", "signal", "post2015", "signal_x_post2015"], lag)
                            if fit.empty:
                                continue
                            b1 = fit.loc[fit.term.eq("signal")].iloc[0]; b3 = fit.loc[fit.term.eq("signal_x_post2015")].iloc[0]
                            interactions.append({"predictor": predictor, "predictor_family": family, "mean_window": mean_window,
                                                 "pr_window": pr_window, "pr_bin": bin_name, "target": target, "endpoint": endpoint,
                                                 "hac_lag": lag, "N": int(b3.N), "beta_signal_pre": b1.beta,
                                                 "beta_interaction": b3.beta, "interaction_HAC_se": b3.HAC_se,
                                                 "interaction_HAC_p": b3.HAC_p, "interaction_ci_lower": b3.ci_lower,
                                                 "interaction_ci_upper": b3.ci_upper})
                        annual_base = dataset[[target, pr_col]].dropna()
                        annual_regime = np.where(annual_base.index < change, "PRE_2015_7PCT", "POST_2015_10PCT")
                        for (year, annual_regime_name), part in annual_base.groupby([annual_base.index.year, annual_regime]):
                            mask = pr_bin_mask(part[pr_col], lower, upper); group, other = part.loc[mask, target], part.loc[~mask, target]
                            if len(group) and len(other):
                                yearly_rows.append({"predictor": predictor, "pr_window": pr_window, "pr_bin": bin_name,
                                                    "target": target, "year": int(year), "regime": annual_regime_name,
                                                    "N": len(group), "mean_ret": group.mean(), "mean_difference": group.mean()-other.mean(),
                                                    "win_rate": group.gt(0).mean(), "win_rate_difference": group.gt(0).mean()-other.gt(0).mean()})
    comparison = pd.DataFrame(rows)
    comparison = _apply_module_c_fdr(comparison, "mean_difference_HAC_p", "mean_return", ["predictor", "target", "pr_window", "mean_window", "regime"])
    comparison = _apply_module_c_fdr(comparison, "win_difference_HAC_p", "win_rate", ["predictor", "target", "pr_window", "mean_window", "regime"])
    wide = comparison.pivot_table(index=["predictor", "predictor_family", "mean_window", "pr_window", "pr_bin", "target"], columns="regime", values=["mean_difference", "bin_minus_non_bin_win_rate", "N"], aggfunc="first")
    if not wide.empty:
        wide.columns = ["_".join(map(str, c)) for c in wide.columns]; wide = wide.reset_index()
        for metric in ("mean_difference", "bin_minus_non_bin_win_rate"):
            wide[f"{metric}_regime_difference_pre_minus_post"] = wide.get(f"{metric}_PRE_2015_7PCT") - wide.get(f"{metric}_POST_2015_10PCT")
    interaction = pd.DataFrame(interactions)
    interaction = _apply_module_c_fdr(interaction, "interaction_HAC_p", "interaction", ["predictor", "target", "pr_window", "mean_window", "endpoint"])
    return comparison.merge(wide, on=["predictor", "predictor_family", "mean_window", "pr_window", "pr_bin", "target"], how="left"), interaction, pd.DataFrame(yearly_rows)


def classify_mechanism(regressions: pd.DataFrame, interaction: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in ("A", "B", "C"):
        part = regressions.loc[regressions.candidate_group.eq(group) & regressions.model.ne("breadth_only")]
        retained = part.breadth_HAC_p.lt(.05) & part.attenuation_ratio.abs().lt(.5)
        mediated = part.attenuation_ratio.abs().ge(.5)
        decision = "保留" if retained.any() else ("修改後再測" if len(part) and not mediated.all() else "淘汰")
        rows.append({"hypothesis": f"Candidate {group}", "decision": decision,
                     "adjusted_significant_models": int(retained.sum()), "mediated_models": int(mediated.sum())})
    broad = interaction.loc[interaction.predictor.str.startswith("down_ratio__")]
    pre = comparison.loc[comparison.predictor.str.startswith("down_ratio__") & comparison.regime.eq("PRE_2015_7PCT")]
    insufficient = pre.N.lt(20).mean() > .5 if len(pre) else True
    interaction_support = broad.interaction_FDR_family.lt(.05).any() if len(broad) else False
    unique_effects = pre.drop_duplicates(["predictor", "mean_window", "pr_window", "pr_bin", "target"])
    pre_mean = unique_effects.get("mean_difference_PRE_2015_7PCT", pd.Series(dtype=float))
    post_mean = unique_effects.get("mean_difference_POST_2015_10PCT", pd.Series(dtype=float))
    direction_diff = bool((np.sign(pre_mean) != np.sign(post_mean)).mean() >= .5) if len(pre_mean) else False
    pre_support = bool((pre.mean_return_FDR_family.lt(.05) | pre.win_rate_FDR_family.lt(.05)).any()) if len(pre) else False
    if insufficient:
        classification, decision = "INSUFFICIENT_PRE2015_POWER", "修改後再測"
    elif interaction_support and pre_support:
        classification, decision = "STRUCTURAL_BREAK_SUPPORTED", "保留"
    elif direction_diff:
        classification, decision = "POSSIBLE_REGIME_DIFFERENCE", "修改後再測"
    elif pre_support:
        classification, decision = "NO_REGIME_DIFFERENCE", "修改後再測"
    else:
        classification, decision = "OLD_RESULT_NOT_REPRODUCED", "淘汰"
    rows.append({"hypothesis": "Broad-decline mean-reversion", "decision": decision,
                 "regime_classification": classification, "interaction_family_support": int(interaction_support)})
    return pd.DataFrame(rows)
