from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

from .config import V10Config, V10_RAW_PREDICTORS, V9_TARGET_METADATA
from .statistics import _bh_bonferroni, _cohen_d, _describe, _hac_mean_test, _weighted_difference_test
from .v9 import _event_subset_mask, _hac_lpm, _odds_ratio, historical_percentile_rank


PR_BINS = (
    ("BIN_00_05", 0.00, 0.05),
    ("BIN_05_20", 0.05, 0.20),
    ("BIN_20_40", 0.20, 0.40),
    ("BIN_40_60", 0.40, 0.60),
    ("BIN_60_80", 0.60, 0.80),
    ("BIN_80_95", 0.80, 0.95),
    ("BIN_95_100", 0.95, 1.00),
)
PR_BIN_ORDER = {name: i for i, (name, _, _) in enumerate(PR_BINS)}


def build_v10_features(
    breadth: pd.DataFrame, config: V10Config,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str, int, str]]]:
    """Build raw -> trailing mean -> LEVEL/DELTA features only."""
    out = breadth.copy()
    specs: dict[str, tuple[str, str, int, str]] = {}
    for raw_name, (family, direction) in V10_RAW_PREDICTORS.items():
        raw = pd.to_numeric(out[raw_name], errors="coerce")
        for window in config.signal_mean_windows:
            level = f"{raw_name}__mean_{window}d"
            out[level] = raw.rolling(window, min_periods=window).mean()
            specs[level] = (family, direction, window, "level")
            delta = f"{level}__delta"
            out[delta] = out[level].diff()
            specs[delta] = (family, direction, window, "delta")
    return out, specs


def add_v10_pr_features(
    dataset: pd.DataFrame,
    specs: dict[str, tuple[str, str, int, str]],
    config: V10Config,
) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}
    for predictor in specs:
        for window in config.pr_windows:
            pr_col = f"{predictor}__PR_{window}"
            pr = historical_percentile_rank(dataset[predictor], window)
            additions[pr_col] = pr
            for name, lower, upper in PR_BINS:
                additions[f"{pr_col}__{name}"] = pr_bin_mask(pr, lower, upper)
    return pd.concat([dataset, pd.DataFrame(additions, index=dataset.index)], axis=1)


def pr_bin_mask(pr: pd.Series, lower: float, upper: float) -> pd.Series:
    return pr.ge(lower) & (pr.le(upper) if upper == 1.0 else pr.lt(upper))


def _sample_flag(n: int) -> str:
    if n < 20:
        return "VERY_SMALL_N"
    if n < 50:
        return "SMALL_N"
    if n < 100:
        return "LOW_POWER"
    return "ADEQUATE"


def _one_bin_result(
    base: pd.DataFrame, predictor: str, pr_col: str, target: str,
    lower: float, upper: float, bin_name: str, family: str, direction: str,
    mean_window: int, transform: str, pr_window: int, overlap_policy: str,
    annual_rf: float, lag: int,
) -> dict[str, object] | None:
    valid = base[[predictor, pr_col, target]].dropna()
    if valid.empty:
        return None
    g = pr_bin_mask(valid[pr_col], lower, upper).to_numpy(bool)
    y = valid[target].to_numpy(float)
    if g.sum() < 2 or (~g).sum() < 2:
        return None
    y_bin, y_other = y[g], y[~g]
    desc = _describe(y_bin, annual_rf)
    mean_t0, mean_p0, mean_lo, mean_hi = _hac_mean_test(y_bin, lag)
    mean_diff, mean_t, mean_p = _weighted_difference_test(y, g, lag, "non_group")
    mean_se = abs(mean_diff / mean_t) if np.isfinite(mean_t) and mean_t != 0 else np.nan
    mean_zero_se = abs(float(y_bin.mean()) / mean_t0) if np.isfinite(mean_t0) and mean_t0 != 0 else np.nan
    wins, other_wins = y_bin > 0, y_other > 0
    win_diff, win_se, win_t, win_p, win_half = _hac_lpm(y > 0, g, lag)
    odds_ratio, log_odds, or_lo, or_hi, logistic_p = _odds_ratio(wins, other_wins)
    positive, negative = y_bin[y_bin > 0], y_bin[y_bin < 0]
    avg_win = float(positive.mean()) if len(positive) else np.nan
    avg_loss = float(negative.mean()) if len(negative) else np.nan
    payoff = avg_win / abs(avg_loss) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss else np.nan
    n = int(len(y_bin))
    win_rate = float(wins.mean())
    z = 1.96
    wilson_center = (win_rate + z * z / (2 * n)) / (1 + z * z / n)
    wilson_half = z * np.sqrt(win_rate * (1 - win_rate) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {
        "predictor": predictor, "predictor_family": family, "direction": direction,
        "mean_window": mean_window, "transform": transform, "pr_window": pr_window,
        "signal_column": pr_col, "pr_bin": bin_name, "pr_bin_order": PR_BIN_ORDER[bin_name],
        "pr_lower": lower, "pr_upper": upper, "target": target, "hac_lag": lag,
        "overlap_policy": overlap_policy, "N": n, "sample_size_flag": _sample_flag(n), **desc,
        "bin_mean": float(y_bin.mean()), "non_bin_mean": float(y_other.mean()),
        "unconditional_mean": float(y.mean()), "bin_minus_unconditional_mean": float(y_bin.mean() - y.mean()),
        "mean_difference": mean_diff, "mean_difference_HAC_se": mean_se,
        "mean_difference_HAC_t": mean_t, "mean_difference_HAC_p": mean_p,
        "mean_difference_ci_lower": mean_diff - 1.96 * mean_se if np.isfinite(mean_se) else np.nan,
        "mean_difference_ci_upper": mean_diff + 1.96 * mean_se if np.isfinite(mean_se) else np.nan,
        "HAC_mean_vs_zero_se": mean_zero_se, "HAC_mean_vs_zero_t": mean_t0,
        "HAC_mean_vs_zero_p": mean_p0, "mean_ret_ci_lower": mean_lo,
        "mean_ret_ci_upper": mean_hi, "Cohen_d": _cohen_d(y_bin, y_other),
        "avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff,
        "expectancy": float(y_bin.mean()),
        "bin_win_rate": win_rate, "bin_win_rate_ci_lower": wilson_center - wilson_half,
        "bin_win_rate_ci_upper": wilson_center + wilson_half,
        "non_bin_win_rate": float(other_wins.mean()),
        "unconditional_win_rate": float((y > 0).mean()),
        "bin_minus_non_bin_win_rate": win_diff,
        "bin_minus_unconditional_win_rate": float(wins.mean() - (y > 0).mean()),
        "bin_minus_50pct": float(wins.mean() - .5),
        "successes": int(wins.sum()), "failures": int((~wins).sum()),
        "binomial_vs_50_p": float(stats.binomtest(int(wins.sum()), len(wins), .5).pvalue),
        "win_difference_HAC_se": win_se, "win_difference_HAC_t": win_t,
        "win_difference_HAC_p": win_p,
        "win_difference_ci_lower": win_diff - win_half if np.isfinite(win_half) else np.nan,
        "win_difference_ci_upper": win_diff + win_half if np.isfinite(win_half) else np.nan,
        "odds_ratio_supplementary": odds_ratio,
        "log_odds_coefficient_supplementary": log_odds,
        "odds_ratio_ci_lower_supplementary": or_lo,
        "odds_ratio_ci_upper_supplementary": or_hi,
        "iid_logistic_p_supplementary": logistic_p,
    }


def run_v10_bin_study(
    dataset: pd.DataFrame,
    specs: dict[str, tuple[str, str, int, str]],
    config: V10Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor, (family, direction, mean_window, transform) in specs.items():
        for pr_window in config.pr_windows:
            pr_col = f"{predictor}__PR_{pr_window}"
            for bin_name, lower, upper in PR_BINS:
                for target, meta in V9_TARGET_METADATA.items():
                    policies = ("all_events",) if int(meta["horizon"]) == 1 else ("all_events", "non_overlapping_events")
                    for policy in policies:
                        base = dataset
                        if policy == "non_overlapping_events":
                            base = dataset.loc[_event_subset_mask(dataset.index, int(meta["horizon"]))]
                        row = _one_bin_result(
                            base, predictor, pr_col, target, lower, upper, bin_name,
                            family, direction, mean_window, transform, pr_window, policy,
                            config.annual_rf, int(meta["hac_lag"]),
                        )
                        if row is not None:
                            rows.append(row)
    return pd.DataFrame(rows)


def attach_identity_and_corrections(results: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    hashes: list[str] = []
    for row in out.itertuples():
        valid = dataset[[row.signal_column, row.target]].dropna()
        if row.overlap_policy == "non_overlapping_events":
            horizon = int(V9_TARGET_METADATA[row.target]["horizon"])
            valid = valid.loc[_event_subset_mask(valid.index, horizon)]
        mask = pr_bin_mask(valid[row.signal_column], row.pr_lower, row.pr_upper)
        payload = "|".join(valid.index[mask].strftime("%Y-%m-%d"))
        hashes.append(hashlib.sha256(payload.encode()).hexdigest())
    out["signal_mask_hash"] = hashes
    identity = ["signal_mask_hash", "target", "overlap_policy"]
    out["canonical_hypothesis_id"] = out.apply(
        lambda r: hashlib.sha256("|".join(str(r[c]) for c in identity).encode()).hexdigest()[:20], axis=1
    )
    out["is_duplicate_hypothesis"] = out.duplicated(identity)
    out["duplicate_group_size"] = out.groupby(identity, dropna=False)["predictor"].transform("size")
    out["duplicate_reason"] = np.where(out.is_duplicate_hypothesis, "identical_signal_date_mask", "")
    canonical = ~out.is_duplicate_hypothesis
    family_cols = ["predictor_family", "target", "pr_window", "mean_window", "transform", "overlap_policy"]
    for endpoint, p_col in (("mean_return", "mean_difference_HAC_p"), ("win_rate", "win_difference_HAC_p")):
        fdr, bonf = _bh_bonferroni(out.loc[canonical, p_col])
        out.loc[canonical, f"{endpoint}_FDR_global"] = fdr
        out.loc[canonical, f"{endpoint}_Bonferroni_global"] = bonf
        for _, idx in out.loc[canonical].groupby(family_cols, dropna=False).groups.items():
            fdr, bonf = _bh_bonferroni(out.loc[idx, p_col])
            out.loc[idx, f"{endpoint}_FDR_family"] = fdr
            out.loc[idx, f"{endpoint}_Bonferroni_family"] = bonf
        lookup = out.loc[canonical].drop_duplicates("canonical_hypothesis_id").set_index("canonical_hypothesis_id")
        for suffix in ("FDR_global", "Bonferroni_global", "FDR_family", "Bonferroni_family"):
            column = f"{endpoint}_{suffix}"
            out[column] = out.canonical_hypothesis_id.map(lookup[column])
    out["eligible_for_retention"] = ~out.sample_size_flag.eq("VERY_SMALL_N")
    return out


def build_shape_analysis(results: pd.DataFrame) -> pd.DataFrame:
    primary = results.loc[results.overlap_policy.eq("all_events")].copy()
    keys = ["predictor", "predictor_family", "direction", "mean_window", "transform", "pr_window", "target"]
    rows: list[dict[str, object]] = []
    for values, frame in primary.groupby(keys, dropna=False):
        ordered = frame.sort_values("pr_bin_order")
        if len(ordered) != len(PR_BINS):
            continue
        order = ordered.pr_bin_order.to_numpy(float)
        rho_mean, p_mean = stats.spearmanr(order, ordered.mean_ret)
        rho_win, p_win = stats.spearmanr(order, ordered.bin_win_rate)
        indexed = ordered.set_index("pr_bin")
        contrasts = {
            "contrast_95_100_vs_00_05_mean": indexed.loc["BIN_95_100", "mean_ret"] - indexed.loc["BIN_00_05", "mean_ret"],
            "contrast_80_95_vs_05_20_mean": indexed.loc["BIN_80_95", "mean_ret"] - indexed.loc["BIN_05_20", "mean_ret"],
            "contrast_60_80_vs_20_40_mean": indexed.loc["BIN_60_80", "mean_ret"] - indexed.loc["BIN_20_40", "mean_ret"],
            "contrast_95_100_vs_00_05_win": indexed.loc["BIN_95_100", "bin_win_rate"] - indexed.loc["BIN_00_05", "bin_win_rate"],
            "contrast_80_95_vs_05_20_win": indexed.loc["BIN_80_95", "bin_win_rate"] - indexed.loc["BIN_05_20", "bin_win_rate"],
            "contrast_60_80_vs_20_40_win": indexed.loc["BIN_60_80", "bin_win_rate"] - indexed.loc["BIN_20_40", "bin_win_rate"],
        }
        middle = indexed.loc[["BIN_20_40", "BIN_40_60", "BIN_60_80"], "mean_ret"].mean()
        tails = indexed.loc[["BIN_00_05", "BIN_95_100"], "mean_ret"].mean()
        if abs(rho_mean) >= .7:
            shape = "monotonic"
        elif tails > middle and indexed.loc["BIN_00_05", "mean_ret"] > middle and indexed.loc["BIN_95_100", "mean_ret"] > middle:
            shape = "U-shape"
        elif tails < middle and indexed.loc["BIN_00_05", "mean_ret"] < middle and indexed.loc["BIN_95_100", "mean_ret"] < middle:
            shape = "inverted-U"
        elif abs(contrasts["contrast_95_100_vs_00_05_mean"]) > ordered.mean_ret.std(ddof=1):
            shape = "tail-only/threshold"
        else:
            shape = "noisy"
        rows.append({**dict(zip(keys, values)), "spearman_mean_rho": rho_mean, "spearman_mean_p": p_mean,
                     "spearman_win_rho": rho_win, "spearman_win_p": p_win, "shape_classification": shape, **contrasts})
    return pd.DataFrame(rows)


def build_yearly_results(results: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    focus = results.loc[(~results.is_duplicate_hypothesis) & results.overlap_policy.eq("all_events")]
    for row in focus.itertuples():
        base = dataset[[row.signal_column, row.target]].dropna()
        mask = pr_bin_mask(base[row.signal_column], row.pr_lower, row.pr_upper)
        for year, part in base.groupby(base.index.year):
            group = part.loc[mask.reindex(part.index).fillna(False)]
            other = part.loc[~mask.reindex(part.index).fillna(False)]
            if group.empty or other.empty:
                continue
            win_diff = group[row.target].gt(0).mean() - other[row.target].gt(0).mean()
            rows.append({"canonical_hypothesis_id": row.canonical_hypothesis_id, "predictor": row.predictor,
                         "pr_bin": row.pr_bin, "target": row.target, "year": int(year), "N": len(group),
                         "mean_ret": group[row.target].mean(), "median_ret": group[row.target].median(),
                         "win_rate": group[row.target].gt(0).mean(), "bin_minus_non_bin_win_rate": win_diff,
                         "direction": np.sign(win_diff), "return_contribution": len(group) * group[row.target].mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    key = "canonical_hypothesis_id"
    out["positive_years"] = out.groupby(key).mean_ret.transform(lambda x: int(x.gt(0).sum()))
    out["negative_years"] = out.groupby(key).mean_ret.transform(lambda x: int(x.lt(0).sum()))
    out["direction_consistency"] = out.groupby(key).bin_minus_non_bin_win_rate.transform(lambda x: max(x.gt(0).mean(), x.lt(0).mean()))
    out["best_year"] = out.groupby(key).mean_ret.transform(lambda x: out.loc[x.idxmax(), "year"])
    out["worst_year"] = out.groupby(key).mean_ret.transform(lambda x: out.loc[x.idxmin(), "year"])
    out["best_year_contribution"] = out.groupby(key).return_contribution.transform("max")
    out["worst_year_contribution"] = out.groupby(key).return_contribution.transform("min")
    out["largest_abs_year_contribution_share"] = out.groupby(key).return_contribution.transform(
        lambda x: float(x.abs().max() / x.abs().sum()) if x.abs().sum() else np.nan
    )
    out["regime_year_2020"] = out.year.eq(2020)
    out["regime_year_2022"] = out.year.eq(2022)
    out["regime_year_2024_2026"] = out.year.between(2024, 2026)
    return out


def _markdown_rows(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> list[str]:
    shown = frame.loc[:, [c for c in columns if c in frame]].head(limit)
    if shown.empty:
        return ["(no rows)"]
    headers = list(shown.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown.itertuples(index=False, name=None):
        values = [f"{v:.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def validation_summary_markdown(results: pd.DataFrame, shape: pd.DataFrame, yearly: pd.DataFrame | None = None) -> str:
    canonical = results.loc[~results.is_duplicate_hypothesis]
    count = lambda col: int(canonical[col].lt(.05).sum())
    high_bins = ["BIN_60_80", "BIN_80_95", "BIN_95_100"]
    high_down = canonical.loc[canonical.predictor.str.startswith("down_ratio__") & canonical["transform"].eq("level") & canonical.pr_bin.isin(high_bins) & canonical.target.isin(["ret_o1_c1", "ret_o1_c3", "ret_o1_c5"]) & canonical.overlap_policy.eq("all_events")]
    high_up = canonical.loc[canonical.predictor.str.startswith("up_ratio__") & canonical["transform"].eq("level") & canonical.pr_bin.isin(high_bins) & canonical.target.eq("ret_o1_c1") & canonical.overlap_policy.eq("all_events")]
    all_events = canonical.loc[canonical.overlap_policy.eq("all_events")]
    family_summary = all_events.groupby(["predictor_family", "direction"], as_index=False).agg(
        mean_effect=("mean_difference", "mean"), win_effect=("bin_minus_non_bin_win_rate", "mean"),
        mean_family_sig=("mean_return_FDR_family", lambda x: int(x.lt(.05).sum())),
        win_family_sig=("win_rate_FDR_family", lambda x: int(x.lt(.05).sum())),
    )
    transform_summary = all_events.groupby("transform", as_index=False).agg(
        mean_abs_effect=("mean_difference", lambda x: float(x.abs().mean())),
        win_abs_effect=("bin_minus_non_bin_win_rate", lambda x: float(x.abs().mean())),
        family_sig=("mean_return_FDR_family", lambda x: int(x.lt(.05).sum())),
    )
    mean_window_summary = all_events.groupby("mean_window", as_index=False).agg(
        mean_abs_effect=("mean_difference", lambda x: float(x.abs().mean())),
        win_abs_effect=("bin_minus_non_bin_win_rate", lambda x: float(x.abs().mean())),
        family_sig=("mean_return_FDR_family", lambda x: int(x.lt(.05).sum())),
    )
    pr_summary = all_events.groupby("pr_window", as_index=False).agg(
        mean_abs_effect=("mean_difference", lambda x: float(x.abs().mean())),
        win_abs_effect=("bin_minus_non_bin_win_rate", lambda x: float(x.abs().mean())),
        family_sig=("mean_return_FDR_family", lambda x: int(x.lt(.05).sum())),
    )
    labels = []
    for raw in V10_RAW_PREDICTORS:
        for transform in ("level", "delta"):
            part = canonical.loc[canonical.predictor.str.startswith(raw + "__") & canonical["transform"].eq(transform) & canonical.eligible_for_retention]
            global_support = part.mean_return_FDR_global.lt(.05) | part.win_rate_FDR_global.lt(.05)
            family_support = part.mean_return_FDR_family.lt(.05) | part.win_rate_FDR_family.lt(.05)
            if global_support.any() and part.loc[global_support, "overlap_policy"].eq("non_overlapping_events").any():
                decision = "保留"
            elif family_support.any():
                decision = "修改後再測"
            else:
                decision = "淘汰"
            labels.append({"family": f"{raw} {transform}", "decision": decision,
                           "global_support": int(global_support.sum()), "family_support": int(family_support.sum())})
    decisions = pd.DataFrame(labels)
    freeze = canonical.loc[canonical.eligible_for_retention & (canonical.mean_return_FDR_global.lt(.05) | canonical.win_rate_FDR_global.lt(.05))]
    c1_down = high_down.loc[high_down.target.eq("ret_o1_c1")]
    overheat_support = high_up.bin_minus_non_bin_win_rate.lt(0)
    lines = [
        "# v10 Validation Summary", "",
        f"1. Canonical hypotheses: {len(canonical)}",
        f"2. Mean return raw / family FDR / global FDR: {count('mean_difference_HAC_p')} / {count('mean_return_FDR_family')} / {count('mean_return_FDR_global')}",
        f"3. Win rate raw / family FDR / global FDR: {count('win_difference_HAC_p')} / {count('win_rate_FDR_family')} / {count('win_rate_FDR_global')}",
        f"4. down_ratio high-PR C1/C3/C5 rows: {len(high_down)}; family-FDR mean/win: {int(high_down.mean_return_FDR_family.lt(.05).sum())}/{int(high_down.win_rate_FDR_family.lt(.05).sum())}",
        f"5. up_ratio high-PR O1→C1 rows: {len(high_up)}; family-FDR mean/win: {int(high_up.mean_return_FDR_family.lt(.05).sum())}/{int(high_up.win_rate_FDR_family.lt(.05).sum())}",
        "6. Participation vs intensity: see participation_vs_intensity sheet.",
        "7. Level vs delta: see level_vs_delta sheet.",
        "8. 1/3/5 mean robustness: see all_results_v10 and plots.",
        "9. PR60/126/252 robustness: see pr_bin_shape and plots.",
        f"10. Non-overlap rows: {int(canonical.overlap_policy.eq('non_overlapping_events').sum())}.",
        f"11. Small-N warnings (VERY_SMALL/SMALL/LOW_POWER): {(canonical.sample_size_flag == 'VERY_SMALL_N').sum()}/{(canonical.sample_size_flag == 'SMALL_N').sum()}/{(canonical.sample_size_flag == 'LOW_POWER').sum()}.",
        "", "## Pre-specified broad-decline rows", "",
        *_markdown_rows(high_down, ["predictor", "mean_window", "pr_window", "pr_bin", "target", "N", "mean_difference", "mean_return_FDR_family", "bin_minus_non_bin_win_rate", "win_rate_FDR_family"]),
        "", "## Pre-specified broad-rally rows", "",
        *_markdown_rows(high_up, ["predictor", "mean_window", "pr_window", "pr_bin", "N", "mean_difference", "mean_return_FDR_family", "bin_minus_non_bin_win_rate", "win_rate_FDR_family"]),
        "", "## Participation vs intensity", "", *_markdown_rows(family_summary, list(family_summary.columns)),
        "", "## Level vs delta", "", *_markdown_rows(transform_summary, list(transform_summary.columns)),
        "", "## 1D / 3D / 5D", "", *_markdown_rows(mean_window_summary, list(mean_window_summary.columns)),
        "", "## PR60 / PR126 / PR252", "", *_markdown_rows(pr_summary, list(pr_summary.columns)),
        "", "## Rule-based family disposition", "",
        "`保留` requires eligible global-FDR support also present in a non-overlapping row; `修改後再測` requires eligible family-FDR support; otherwise `淘汰`.", "",
        *_markdown_rows(decisions, list(decisions.columns)),
        "", "## Direct answers", "",
        f"- A. Broad-decline mean reversion family-FDR support rows: {int((high_down.mean_return_FDR_family.lt(.05) | high_down.win_rate_FDR_family.lt(.05)).sum())}.",
        f"- B. O1→C1 high-down positive win-rate rows / family-FDR rows: {int(c1_down.bin_minus_non_bin_win_rate.gt(0).sum())} / {int(c1_down.win_rate_FDR_family.lt(.05).sum())}.",
        f"- C. Broad-rally O1→C1 negative win-rate rows / family-FDR rows: {int(overheat_support.sum())} / {int((overheat_support & high_up.win_rate_FDR_family.lt(.05)).sum())}; this is avoid-chasing evidence, not automatically a short signal.",
        f"- D. Largest average absolute mean effect family: {family_summary.loc[family_summary.mean_effect.abs().idxmax(), 'predictor_family'] if len(family_summary) else 'NA'}.",
        f"- E. Larger average absolute mean effect transform: {transform_summary.loc[transform_summary.mean_abs_effect.idxmax(), 'transform'] if len(transform_summary) else 'NA'}.",
        f"- F. Largest average absolute mean effect smoothing window: {int(mean_window_summary.loc[mean_window_summary.mean_abs_effect.idxmax(), 'mean_window']) if len(mean_window_summary) else 'NA'}D.",
        f"- G. Global-FDR, sufficient-N candidates to consider freezing: {len(freeze)} (must still review yearly/non-overlap direction and untouched OOS).",
        "", "## Interpretation limits", "",
        "No result is described as proven alpha or a tradable edge without global FDR, yearly stability, non-overlap, sufficient N, and untouched OOS/walk-forward support.",
        "Review data snooping, PR/window/horizon mining, overlap, survivorship bias, market drift, event clustering, regime concentration, costs and slippage.",
    ]
    return "\n".join(lines) + "\n"
