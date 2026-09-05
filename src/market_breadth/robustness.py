from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import PREDICTOR_SPECS, PR_GROUPS, TARGET_METADATA, V7_CANDIDATES, V7Config, Z_GROUPS
from .statistics import _bh_bonferroni, _cohen_d, _describe, _hac_mean_test, _mask_for_group, _weighted_difference_test


GROUP_PRIORITY = {
    "PR_LE_20": 0, "PR_GE_80": 0, "PR_LE_5": 0, "PR_GE_95": 0,
    "Z_LE_-2": 0, "Z_LE_-1": 0, "Z_GE_1": 0, "Z_GE_2": 0,
    "Q1": 1, "Q5": 1, "LOW_5": 2, "HIGH_5": 2,
}
PREDICTOR_CANONICAL = {
    "up_ratio": "up_ratio", "down_ratio": "up_ratio",
    "up_ratio_3d_mean": "up_ratio_3d_mean", "down_ratio_3d_mean": "up_ratio_3d_mean",
    "delta_up_ratio_1d": "delta_down_ratio_1d", "delta_down_ratio_1d": "delta_down_ratio_1d",
    "accel_up_ratio_1d": "accel_up_ratio_1d", "accel_down_ratio_1d": "accel_up_ratio_1d",
}


def _stable_mask_hash(index: pd.Index, *parts: object) -> str:
    dates = pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()
    payload = "|".join([*(str(p) for p in parts), *dates])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_signal_mask(dataset: pd.DataFrame, row: pd.Series, config: V7Config) -> pd.Series:
    signal = dataset[str(row["signal_column"])]
    groups = PR_GROUPS if row["signal_method"] == "PR" else Z_GROUPS
    operator, threshold = groups[str(row["group"])]
    mask = _mask_for_group(signal, operator, threshold)
    if row["market_regime"] != "ALL":
        mask &= dataset["market_regime"].eq(row["market_regime"])
    return mask & dataset[str(row["target"])].notna()


def attach_hypothesis_identity(
    dataset: pd.DataFrame, results: pd.DataFrame, config: V7Config | None = None,
) -> pd.DataFrame:
    cfg = config or V7Config()
    out = results.copy()
    hashes = []
    for _, row in out.iterrows():
        mask = _row_signal_mask(dataset, row, cfg)
        hashes.append(_stable_mask_hash(dataset.index[mask], row["market_regime"], row["target"], row["signal_method"]))
    out["signal_mask_hash"] = hashes
    out["canonical_predictor"] = out["predictor"].map(PREDICTOR_CANONICAL).fillna(out["predictor"])
    out["_group_priority"] = out["group"].map(GROUP_PRIORITY).fillna(1).astype(int)
    out["_predictor_priority"] = (out["predictor"] != out["canonical_predictor"]).astype(int)
    out["_row_order"] = np.arange(len(out))
    group_cols = ["signal_mask_hash", "market_regime", "target", "signal_method"]
    ordered = out.sort_values(group_cols + ["_group_priority", "_predictor_priority", "_row_order"])
    canonical_index = ordered.groupby(group_cols, sort=False, dropna=False).head(1).index
    canonical_lookup = out.loc[canonical_index, ["predictor", "group"]].to_dict("index")
    canonical_map = ordered.loc[canonical_index, group_cols].copy()
    canonical_map["_canonical_index"] = canonical_index
    out = out.merge(canonical_map, on=group_cols, how="left", validate="many_to_one")
    out["is_duplicate_hypothesis"] = out.index.to_numpy() != out["_canonical_index"].to_numpy()
    out["duplicate_group_size"] = out.groupby(group_cols, dropna=False)["predictor"].transform("size")
    out["canonical_hypothesis_id"] = out.apply(
        lambda r: hashlib.sha256(
            f"{r.signal_mask_hash}|{r.market_regime}|{r.target}|{r.signal_method}".encode("utf-8")
        ).hexdigest()[:20], axis=1,
    )
    reasons = []
    for _, row in out.iterrows():
        if not row["is_duplicate_hypothesis"]:
            reasons.append("")
            continue
        canonical = canonical_lookup[int(row["_canonical_index"])]
        if row["predictor"] != canonical["predictor"]:
            reasons.append("mirror_or_identical_predictor_mask")
        elif row["group"] != canonical["group"]:
            reasons.append("group_alias_or_identical_threshold")
        else:
            reasons.append("identical_signal_mask")
    out["duplicate_reason"] = reasons
    return out.drop(columns=["_group_priority", "_predictor_priority", "_row_order", "_canonical_index"])


def add_deduplicated_corrections(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    tests = {
        "group_vs_non_group": "HAC_p_value",
        "group_vs_unconditional": "unconditional_HAC_p",
        "mean_vs_zero": "HAC_mean_vs_zero_p",
        "binomial_win_rate": "binomial_p_value",
    }
    canonical = ~out["is_duplicate_hypothesis"]
    family_keys = ["predictor_family", "target", "market_regime", "signal_method"]
    for test_name, p_col in tests.items():
        global_fdr = pd.Series(np.nan, index=out.index, dtype=float)
        global_bonf = pd.Series(np.nan, index=out.index, dtype=float)
        fdr, bonf = _bh_bonferroni(out.loc[canonical, p_col])
        global_fdr.loc[canonical], global_bonf.loc[canonical] = fdr, bonf
        fam_fdr = pd.Series(np.nan, index=out.index, dtype=float)
        fam_bonf = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.loc[canonical].groupby(family_keys, dropna=False).groups.items():
            fdr, bonf = _bh_bonferroni(out.loc[idx, p_col])
            fam_fdr.loc[idx], fam_bonf.loc[idx] = fdr, bonf
        lookup_cols = ["canonical_hypothesis_id"]
        adjustments = pd.DataFrame({
            "canonical_hypothesis_id": out.loc[canonical, "canonical_hypothesis_id"],
            f"{test_name}_FDR_global_v7": global_fdr.loc[canonical],
            f"{test_name}_Bonferroni_global_v7": global_bonf.loc[canonical],
            f"{test_name}_FDR_family_v7": fam_fdr.loc[canonical],
            f"{test_name}_Bonferroni_family_v7": fam_bonf.loc[canonical],
        }).drop_duplicates(lookup_cols)
        out = out.merge(adjustments, on=lookup_cols, how="left", validate="many_to_one")
    return out


def build_fdr_comparison(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tests = {
        "group_vs_non_group": "HAC_p_value",
        "group_vs_unconditional": "unconditional_HAC_p",
        "mean_vs_zero": "HAC_mean_vs_zero_p",
        "binomial_win_rate": "binomial_p_value",
    }
    canonical = ~results["is_duplicate_hypothesis"]
    for name, p_col in tests.items():
        old = f"{name}_FDR_global"
        new = f"{name}_FDR_global_v7"
        rows.append({
            "comparison_type": name,
            "p_value_column": p_col,
            "original_hypothesis_count": int(results[p_col].notna().sum()),
            "deduplicated_hypothesis_count": int((canonical & results[p_col].notna()).sum()),
            "duplicates_removed": int((~canonical & results[p_col].notna()).sum()),
            "v6_global_fdr_significant_05": int(results[old].lt(.05).sum()) if old in results else np.nan,
            "v7_global_fdr_significant_05_unique": int((canonical & results[new].lt(.05)).sum()),
            "minimum_v6_global_fdr": results[old].min() if old in results else np.nan,
            "minimum_v7_global_fdr": results.loc[canonical, new].min(),
        })
    return pd.DataFrame(rows)


def _hac_regression_slope(y: np.ndarray, x: np.ndarray, lag: int) -> tuple[float, float, float]:
    valid = np.isfinite(y) & np.isfinite(x)
    y, x = np.asarray(y)[valid], np.asarray(x)[valid]
    if len(y) < max(5, lag + 3) or np.unique(x).size < 2:
        return np.nan, np.nan, np.nan
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    meat = np.zeros((2, 2), dtype=float)
    for t in range(len(y)):
        meat += residual[t] ** 2 * np.outer(design[t], design[t])
    for k in range(1, min(lag, len(y) - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        for t in range(k, len(y)):
            cross = residual[t] * residual[t-k] * np.outer(design[t], design[t-k])
            meat += weight * (cross + cross.T)
    inv = np.linalg.pinv(design.T @ design)
    variance = inv @ meat @ inv
    se = np.sqrt(max(variance[1, 1], 0.0))
    if se == 0:
        return float(beta[1]), np.nan, np.nan
    t_value = float(beta[1] / se)
    return float(beta[1]), t_value, float(2 * stats.t.sf(abs(t_value), df=len(y)-2))


def build_quintile_trend_results(dataset: pd.DataFrame, config: V7Config | None = None) -> pd.DataFrame:
    cfg = config or V7Config()
    rows = []
    for predictor, (family, _) in PREDICTOR_SPECS.items():
        pr_col = f"{predictor}__PR_{cfg.pr_window}"
        for target, meta in TARGET_METADATA.items():
            for regime in ("ALL", "BULL", "BEAR"):
                regime_mask = pd.Series(True, index=dataset.index) if regime == "ALL" else dataset["market_regime"].eq(regime)
                frame = dataset.loc[regime_mask, [pr_col, target]].dropna().copy()
                frame["quintile_rank"] = pd.cut(
                    frame[pr_col], [0, .2, .4, .6, .8, 1], labels=[1, 2, 3, 4, 5], include_lowest=True,
                ).astype(float)
                if len(frame) < 10 or frame["quintile_rank"].nunique() < 5:
                    continue
                grouped = frame.groupby("quintile_rank", observed=True)[target]
                means = grouped.mean().reindex(range(1, 6))
                medians = grouped.median().reindex(range(1, 6))
                wins = grouped.apply(lambda s: s.gt(0).mean()).reindex(range(1, 6))
                counts = grouped.size().reindex(range(1, 6))
                diffs = np.diff(means.to_numpy())
                rho = float(stats.spearmanr(range(1, 6), means.to_numpy()).statistic)
                violations = int(min(np.count_nonzero(diffs < 0), np.count_nonzero(diffs > 0)))
                slope, slope_t, slope_p = _hac_regression_slope(
                    frame[target].to_numpy(float), frame["quintile_rank"].to_numpy(float), int(meta["hac_lag"]),
                )
                endpoints = frame[frame["quintile_rank"].isin([1., 5.])]
                endpoint_group = endpoints["quintile_rank"].eq(5).to_numpy()
                qdiff, qt, qp = _weighted_difference_test(endpoints[target].to_numpy(float), endpoint_group, int(meta["hac_lag"]), "non_group")
                rows.append({
                    "predictor": predictor, "predictor_family": family, "canonical_predictor": PREDICTOR_CANONICAL.get(predictor, predictor),
                    "target": target, "market_regime": regime, "hac_lag": int(meta["hac_lag"]),
                    "trend_beta": slope, "trend_HAC_t": slope_t, "trend_HAC_p": slope_p,
                    "Q5_minus_Q1": qdiff, "Q5_vs_Q1_HAC_t": qt, "Q5_vs_Q1_HAC_p": qp,
                    "spearman_group_vs_mean": rho, "monotonic_violation_count": violations,
                    "trend_direction": "positive" if slope > 0 else "negative",
                    "trend_shape": "monotonic" if abs(rho) >= .9 and violations <= 1 else "non_monotonic",
                    "isolated_middle_bucket_warning": bool(predictor == "big_down_ratio" and regime == "BEAR" and np.argmax(np.abs(means - means.mean())) in (1, 2, 3)),
                    **{f"Q{i}_N": int(counts.loc[i]) for i in range(1, 6)},
                    **{f"Q{i}_mean": float(means.loc[i]) for i in range(1, 6)},
                    **{f"Q{i}_median": float(medians.loc[i]) for i in range(1, 6)},
                    **{f"Q{i}_win_rate": float(wins.loc[i]) for i in range(1, 6)},
                })
    out = pd.DataFrame(rows)
    out["trend_hypothesis_id"] = out.apply(
        lambda r: f"{r.canonical_predictor}|{r.target}|{r.market_regime}", axis=1,
    )
    out["is_duplicate_hypothesis"] = out.duplicated("trend_hypothesis_id")
    canonical = ~out["is_duplicate_hypothesis"]
    fdr, bonf = _bh_bonferroni(out.loc[canonical, "trend_HAC_p"])
    out.loc[canonical, "trend_FDR_global"] = fdr
    out.loc[canonical, "trend_Bonferroni_global"] = bonf
    family_fdr = pd.Series(np.nan, index=out.index)
    family_bonf = pd.Series(np.nan, index=out.index)
    for _, idx in out.loc[canonical].groupby(["predictor_family", "target", "market_regime"], dropna=False).groups.items():
        ff, bb = _bh_bonferroni(out.loc[idx, "trend_HAC_p"])
        family_fdr.loc[idx], family_bonf.loc[idx] = ff, bb
    out.loc[canonical, "trend_FDR_family"] = family_fdr.loc[canonical]
    out.loc[canonical, "trend_Bonferroni_family"] = family_bonf.loc[canonical]
    for col in ["trend_FDR_global", "trend_Bonferroni_global", "trend_FDR_family", "trend_Bonferroni_family"]:
        lookup = out.loc[canonical].set_index("trend_hypothesis_id")[col]
        out[col] = out["trend_hypothesis_id"].map(lookup)
    return out


def _candidate_rows(results: pd.DataFrame, candidates: Iterable[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    keys = ["predictor", "signal_method", "group", "market_regime", "target"]
    wanted = pd.MultiIndex.from_tuples(candidates, names=keys)
    actual = pd.MultiIndex.from_frame(results[keys])
    return results.loc[actual.isin(wanted)].drop_duplicates(keys)


def build_yearly_stability(
    dataset: pd.DataFrame, results: pd.DataFrame, config: V7Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = config or V7Config()
    rows, loyo_rows, detail_rows = [], [], []
    keys = ["predictor", "signal_method", "group", "market_regime", "target"]
    focus = results.loc[(~results["is_duplicate_hypothesis"]) & results["group_type"].eq("tradable_tail")].drop_duplicates(keys)
    candidate_set = set(V7_CANDIDATES)
    for _, spec in focus.iterrows():
        group_mask = _row_signal_mask(dataset, spec, cfg)
        regime_mask = pd.Series(True, index=dataset.index) if spec.market_regime == "ALL" else dataset["market_regime"].eq(spec.market_regime)
        eligible = regime_mask & dataset[spec.target].notna()
        frame = pd.DataFrame({"return": dataset[spec.target], "group": group_mask}).loc[eligible]
        full_diff, _, _ = _weighted_difference_test(frame["return"].to_numpy(), frame["group"].to_numpy(), int(spec.hac_lag), "non_group")
        yearly_effects = []
        for year, part in frame.groupby(frame.index.year):
            group_values = part.loc[part.group, "return"]
            other_values = part.loc[~part.group, "return"]
            diff = group_values.mean() - other_values.mean() if len(group_values) and len(other_values) else np.nan
            yearly_effects.append((int(year), len(group_values), group_values.mean(), group_values.median(), group_values.gt(0).mean(), diff))
            detail_rows.append({
                **{k: spec[k] for k in keys}, "is_prespecified_candidate": tuple(spec[k] for k in keys) in candidate_set,
                "year": int(year), "N": len(group_values), "mean_return": group_values.mean(),
                "median_return": group_values.median(), "win_rate": group_values.gt(0).mean(),
                "group_vs_non_group_diff": diff,
            })
        contributions = np.array([n * mean for _, n, mean, _, _, _ in yearly_effects], dtype=float)
        denom = np.nansum(np.abs(contributions))
        max_share = float(np.nanmax(np.abs(contributions)) / denom) if denom else np.nan
        positive = sum(diff > 0 for *_, diff in yearly_effects if np.isfinite(diff))
        negative = sum(diff < 0 for *_, diff in yearly_effects if np.isfinite(diff))
        expected_positive = full_diff >= 0
        consistent = sum((diff >= 0) == expected_positive for *_, diff in yearly_effects if np.isfinite(diff))
        valid_years = positive + negative
        effect_values = np.array([x[5] for x in yearly_effects], dtype=float)
        finite_effects = np.isfinite(effect_values)
        best_year = yearly_effects[int(np.nanargmax(effect_values))][0] if finite_effects.any() else np.nan
        worst_year = yearly_effects[int(np.nanargmin(effect_values))][0] if finite_effects.any() else np.nan
        rows.append({
            **{k: spec[k] for k in keys}, "is_prespecified_candidate": tuple(spec[k] for k in keys) in candidate_set,
            "full_sample_diff": full_diff,
            "positive_year_count": positive, "negative_year_count": negative,
            "direction_consistency_rate": consistent / valid_years if valid_years else np.nan,
            "max_single_year_contribution_share": max_share,
            "best_year": best_year, "worst_year": worst_year,
            "concentration_warning": bool(max_share > .5),
            "total_signal_N": int(frame.group.sum()),
        })
        for year in sorted(frame.index.year.unique()):
            kept = frame.loc[frame.index.year != year]
            diff, tval, pval = _weighted_difference_test(kept["return"].to_numpy(), kept["group"].to_numpy(), int(spec.hac_lag), "non_group")
            loyo_rows.append({
                **{k: spec[k] for k in keys}, "excluded_year": int(year),
                "N_group": int(kept.group.sum()), "mean_return": kept.loc[kept.group, "return"].mean(),
                "group_vs_non_group_diff": diff, "HAC_t": tval, "HAC_p": pval,
                "effect_direction": "positive" if diff >= 0 else "negative",
                "direction_matches_full_sample": bool((diff >= 0) == (full_diff >= 0)),
            })
    return pd.DataFrame(rows), pd.DataFrame(loyo_rows), pd.DataFrame(detail_rows)


PULLBACK_TARGETS = {
    "ret_o1_c1": (0, "direct_open_entry"), "ret_o1_c2": (1, "direct_open_entry"),
    "ret_o1_c3": (2, "direct_open_entry"), "ret_o1_c5": (4, "direct_open_entry"),
    "ret_o2_c2": (0, "tradable_open_entry"), "ret_o2_c3": (1, "tradable_open_entry"),
    "ret_o2_c5": (3, "tradable_open_entry"),
    "ret_c1_c2": (0, "diagnostic_close_entry"), "ret_c1_c3": (1, "diagnostic_close_entry"),
    "ret_c1_c5": (3, "diagnostic_close_entry"),
}

DIRECT_COMPARATOR = {
    "ret_o2_c2": "ret_o1_c2", "ret_o2_c3": "ret_o1_c3", "ret_o2_c5": "ret_o1_c5",
    "ret_c1_c2": "ret_o1_c2", "ret_c1_c3": "ret_o1_c3", "ret_c1_c5": "ret_o1_c5",
}


def _non_overlapping_mask(mask: pd.Series, hold_days: int) -> pd.Series:
    selected = np.flatnonzero(mask.to_numpy(bool))
    keep = np.zeros(len(mask), dtype=bool)
    next_allowed = 0
    for pos in selected:
        if pos >= next_allowed:
            keep[pos] = True
            next_allowed = pos + hold_days + 1
    return pd.Series(keep, index=mask.index)


def build_limit_up_pullback_validation(
    dataset: pd.DataFrame, config: V7Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = config or V7Config()
    signal_col = f"limit_up_ratio__PR_{cfg.pr_window}"
    rows = []
    yearly_rows = []
    for regime in ("ALL", "BULL", "BEAR"):
        regime_mask = pd.Series(True, index=dataset.index) if regime == "ALL" else dataset["market_regime"].eq(regime)
        base_signal = dataset[signal_col].ge(.8) & regime_mask
        pullback = dataset["ret_o1_c1"]
        conditions = [("direct", np.nan, base_signal)]
        for threshold in cfg.pullback_thresholds:
            conditions.append((f"pullback_le_{threshold:g}", threshold, base_signal & pullback.le(threshold)))
        for condition, threshold, event_mask in conditions:
            target_names = ["ret_o1_c1", "ret_o1_c2", "ret_o1_c3", "ret_o1_c5"] if condition == "direct" else [
                "ret_o2_c2", "ret_o2_c3", "ret_o2_c5", "ret_c1_c2", "ret_c1_c3", "ret_c1_c5",
            ]
            for target in target_names:
                lag, execution = PULLBACK_TARGETS[target]
                hold_days = lag + 1
                eligible = regime_mask & dataset[target].notna()
                raw_mask = event_mask & eligible
                for overlap_policy, selected_mask in (
                    ("raw", raw_mask), ("non_overlapping", _non_overlapping_mask(raw_mask, hold_days)),
                ):
                    y_all = dataset.loc[eligible, target].to_numpy(float)
                    group = selected_mask.loc[eligible].to_numpy(bool)
                    y_group, y_other = y_all[group], y_all[~group]
                    if len(y_group) == 0:
                        continue
                    desc = _describe(y_group, cfg.annual_rf)
                    diff, tval, pval = _weighted_difference_test(y_all, group, lag, "non_group")
                    udiff, ut, up = _weighted_difference_test(y_all, group, lag, "unconditional")
                    zt, zp, lo, hi = _hac_mean_test(y_group, lag)
                    comparator = DIRECT_COMPARATOR.get(target)
                    comparable_diff = comparable_t = comparable_p = np.nan
                    direct_same_events_mean = np.nan
                    if comparator is not None:
                        paired = dataset.loc[selected_mask, [target, comparator]].dropna()
                        if not paired.empty:
                            direct_same_events_mean = float(paired[comparator].mean())
                            comparable_diff = float((paired[target] - paired[comparator]).mean())
                            comparable_t, comparable_p, _, _ = _hac_mean_test(
                                (paired[target] - paired[comparator]).to_numpy(float), max(lag, PULLBACK_TARGETS[comparator][0]),
                            )
                    rows.append({
                        "predictor": "limit_up_ratio", "predictor_family": "LIMIT_PULLBACK", "direction": "up",
                        "signal_method": "PR", "group": "PR_GE_80", "group_type": "tradable_tail",
                        "market_regime": regime, "target": target, "hac_lag": lag,
                        "entry_condition": condition, "pullback_threshold": threshold,
                        "execution_status": execution, "overlap_policy": overlap_policy,
                        "raw_event_count": int(raw_mask.sum()), "non_overlapping_event_count": int(selected_mask.sum()),
                        "signal_count": int(base_signal.sum()), "pullback_event_count": int(event_mask.sum()),
                        "pullback_opportunity_rate": float(event_mask.sum() / base_signal.sum()) if base_signal.sum() else np.nan,
                        **desc, "non_group_mean_ret": float(np.mean(y_other)) if len(y_other) else np.nan,
                        "mean_ret_minus_non_group": diff, "HAC_t": tval, "HAC_p_value": pval,
                        "Cohen_d": _cohen_d(y_group, y_other), "unconditional_mean_ret": float(np.mean(y_all)),
                        "mean_ret_minus_unconditional": udiff, "unconditional_HAC_t": ut, "unconditional_HAC_p": up,
                        "Cohen_d_vs_unconditional": float((np.mean(y_group)-np.mean(y_all))/np.std(y_all, ddof=1)) if np.std(y_all, ddof=1)>0 else np.nan,
                        "HAC_mean_vs_zero_t": zt, "HAC_mean_vs_zero_p": zp,
                        "mean_ret_ci_lower": lo, "mean_ret_ci_upper": hi,
                        "probability_loss_gt_1pct": float(np.mean(y_group < -.01)),
                        "direct_comparator_target": comparator,
                        "same_events_direct_o1_mean": direct_same_events_mean,
                        "mean_minus_direct_same_events": comparable_diff,
                        "wait_vs_direct_HAC_t": comparable_t,
                        "wait_vs_direct_HAC_p": comparable_p,
                    })
                    if overlap_policy == "raw":
                        frame = dataset.loc[selected_mask, [target]].dropna()
                        for year, values in frame.groupby(frame.index.year)[target]:
                            yearly_rows.append({
                                "market_regime": regime, "entry_condition": condition,
                                "pullback_threshold": threshold, "execution_status": execution,
                                "target": target, "year": int(year), "N": len(values),
                                "mean_return": values.mean(), "median_return": values.median(),
                                "win_rate": values.gt(0).mean(),
                            })
    out = pd.DataFrame(rows)
    # Existing correction function expects these four raw p-value columns and research-family keys.
    from .statistics import add_multiple_testing_corrections
    out = add_multiple_testing_corrections(out)
    yearly = pd.DataFrame(yearly_rows)
    if not yearly.empty:
        keys = ["market_regime", "entry_condition", "pullback_threshold", "execution_status", "target"]
        summary = yearly.groupby(keys, dropna=False).agg(
            positive_year_count=("mean_return", lambda x: int((x > 0).sum())),
            negative_year_count=("mean_return", lambda x: int((x < 0).sum())),
            best_year_contribution=("mean_return", "max"), worst_year_contribution=("mean_return", "min"),
        ).reset_index()
        yearly = yearly.merge(summary, on=keys, how="left")
    return out, yearly
