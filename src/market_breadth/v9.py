from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import V9Config, V9_RAW_PREDICTORS, V9_TARGET_METADATA
from .statistics import _bh_bonferroni, _cohen_d, _describe, _hac_mean_test, _weighted_difference_test


PR_BINS = ((0, .05), (.05, .20), (.20, .40), (.40, .60), (.60, .80), (.80, .95), (.95, 1.0))


def build_v9_features(breadth: pd.DataFrame, config: V9Config) -> tuple[pd.DataFrame, dict[str, tuple[str, str, int, str]]]:
    """Build only the pre-specified raw/delta features; no acceleration."""
    out = breadth.copy()
    specs: dict[str, tuple[str, str, int, str]] = {}
    for raw_name, (family, direction) in V9_RAW_PREDICTORS.items():
        raw = pd.to_numeric(out[raw_name], errors="coerce")
        for mean_window in config.signal_mean_windows:
            mean_name = f"{raw_name}__mean_{mean_window}d"
            out[mean_name] = raw if mean_window == 1 else raw.rolling(mean_window, min_periods=mean_window).mean()
            specs[mean_name] = (family, direction, mean_window, "raw")
            delta_name = f"{mean_name}__delta"
            out[delta_name] = out[mean_name].diff()
            specs[delta_name] = ("DELTA", direction, mean_window, "delta")
    return out, specs


def historical_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Rank x[t] against exactly the preceding window observations (t excluded)."""
    def rank_current(values: np.ndarray) -> float:
        current, history = values[-1], values[:-1]
        if not np.isfinite(current) or not np.isfinite(history).all():
            return np.nan
        return float(np.count_nonzero(history <= current) / len(history))

    return pd.to_numeric(series, errors="coerce").rolling(
        window + 1, min_periods=window + 1
    ).apply(rank_current, raw=True)


def add_v9_pr_features(
    dataset: pd.DataFrame, specs: dict[str, tuple[str, str, int, str]], config: V9Config,
) -> pd.DataFrame:
    out = dataset.copy()
    for predictor in specs:
        for pr_window in config.pr_windows:
            out[f"{predictor}__PR_{pr_window}"] = historical_percentile_rank(out[predictor], pr_window)
    return out


def _thresholds(config: V9Config) -> Iterable[tuple[str, str, float]]:
    for value in config.pr_thresholds[:3]:
        yield f"PR_LE_{int(value * 100)}", "le", value
    for value in config.pr_thresholds[3:]:
        yield f"PR_GE_{int(value * 100)}", "ge", value


def _hac_lpm(y: np.ndarray, signal: np.ndarray, lag: int) -> tuple[float, float, float, float, float]:
    y = np.asarray(y, float)
    signal = np.asarray(signal, bool)
    diff, t_value, p_value = _weighted_difference_test(y, signal, lag, "non_group")
    if not np.isfinite(t_value) or t_value == 0:
        return diff, np.nan, t_value, p_value, np.nan
    se = abs(diff / t_value)
    critical = float(stats.norm.ppf(.975))
    return diff, se, t_value, p_value, critical * se


def _odds_ratio(wins_signal: np.ndarray, wins_other: np.ndarray) -> tuple[float, float, float, float, float]:
    # Haldane correction keeps the supplementary 2x2 estimate finite.
    a = float(wins_signal.sum()) + .5
    b = float(len(wins_signal) - wins_signal.sum()) + .5
    c = float(wins_other.sum()) + .5
    d = float(len(wins_other) - wins_other.sum()) + .5
    log_or = np.log(a * d / (b * c))
    se = np.sqrt(1/a + 1/b + 1/c + 1/d)
    p = float(2 * stats.norm.sf(abs(log_or / se)))
    return float(np.exp(log_or)), float(log_or), float(np.exp(log_or - 1.96*se)), float(np.exp(log_or + 1.96*se)), p


def _event_subset_mask(index: pd.Index, horizon: int) -> pd.Series:
    """Deterministic calendar-row thinning independent of signal membership."""
    mask = np.zeros(len(index), dtype=bool)
    mask[::max(1, horizon)] = True
    return pd.Series(mask, index=index)


def _one_result(
    base: pd.DataFrame, predictor: str, signal_col: str, signal: pd.Series,
    target: str, lag: int, family: str, direction: str, mean_window: int,
    transform: str, pr_window: int, group: str, threshold: float,
    overlap_policy: str, annual_rf: float,
) -> dict[str, object] | None:
    valid = base[[predictor, signal_col, target]].dropna()
    if valid.empty:
        return None
    g = signal.reindex(valid.index).fillna(False).to_numpy(bool)
    y = valid[target].to_numpy(float)
    if g.sum() < 2 or (~g).sum() < 2:
        return None
    yg, yo = y[g], y[~g]
    desc = _describe(yg, annual_rf)
    t0, p0, lo0, hi0 = _hac_mean_test(yg, lag)
    mean_diff, mean_t, mean_p = _weighted_difference_test(y, g, lag, "non_group")
    wins, other_wins = yg > 0, yo > 0
    positive_returns = yg[yg > 0]
    negative_returns = yg[yg < 0]
    avg_win = float(positive_returns.mean()) if len(positive_returns) else np.nan
    avg_loss = float(negative_returns.mean()) if len(negative_returns) else np.nan
    payoff_ratio = avg_win / abs(avg_loss) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss else np.nan
    win_diff, win_se, win_t, win_p, win_half = _hac_lpm(y > 0, g, lag)
    odds_ratio, log_odds, or_low, or_high, logistic_p = _odds_ratio(wins, other_wins)
    return {
        "predictor": predictor, "predictor_family": family, "direction": direction,
        "mean_window": mean_window, "transform": transform, "pr_window": pr_window,
        "signal_column": signal_col, "group": group, "threshold": threshold,
        "target": target, "hac_lag": lag, "overlap_policy": overlap_policy,
        "N": int(len(yg)), **desc,
        "avg_win": avg_win, "avg_loss": avg_loss, "payoff_ratio": payoff_ratio,
        "expectancy": float(yg.mean()),
        "HAC_mean_vs_zero_se": abs(float(yg.mean()) / t0) if np.isfinite(t0) and t0 != 0 else np.nan,
        "HAC_mean_vs_zero_t": t0, "HAC_mean_vs_zero_p": p0,
        "mean_ret_ci_lower": lo0, "mean_ret_ci_upper": hi0,
        "non_signal_N": int(len(yo)), "non_signal_mean_ret": float(yo.mean()),
        "mean_ret_minus_non_signal": mean_diff, "mean_difference_HAC_t": mean_t,
        "mean_difference_HAC_se": abs(mean_diff / mean_t) if np.isfinite(mean_t) and mean_t != 0 else np.nan,
        "mean_difference_HAC_p": mean_p, "Cohen_d": _cohen_d(yg, yo),
        "signal_win_rate": float(wins.mean()), "non_signal_win_rate": float(other_wins.mean()),
        "unconditional_win_rate": float((y > 0).mean()),
        "signal_minus_non_signal_win_rate": win_diff,
        "signal_minus_unconditional_win_rate": float(wins.mean() - (y > 0).mean()),
        "signal_minus_50pct": float(wins.mean() - .5),
        "successes": int(wins.sum()), "failures": int(len(wins) - wins.sum()),
        "binomial_vs_50_p": float(stats.binomtest(int(wins.sum()), len(wins), .5).pvalue),
        "win_difference_HAC_se": win_se, "win_difference_HAC_t": win_t,
        "win_difference_HAC_p": win_p,
        "win_difference_ci_lower": win_diff - win_half if np.isfinite(win_half) else np.nan,
        "win_difference_ci_upper": win_diff + win_half if np.isfinite(win_half) else np.nan,
        "odds_ratio_supplementary": odds_ratio, "log_odds_coefficient_supplementary": log_odds,
        "odds_ratio_ci_lower_supplementary": or_low, "odds_ratio_ci_upper_supplementary": or_high,
        "iid_logistic_p_supplementary": logistic_p,
    }


def run_v9_threshold_study(
    dataset: pd.DataFrame, specs: dict[str, tuple[str, str, int, str]], config: V9Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for predictor, (family, direction, mean_window, transform) in specs.items():
        for pr_window in config.pr_windows:
            signal_col = f"{predictor}__PR_{pr_window}"
            for group, op, threshold in _thresholds(config):
                signal = dataset[signal_col].le(threshold) if op == "le" else dataset[signal_col].ge(threshold)
                for target, meta in V9_TARGET_METADATA.items():
                    for overlap_policy in ("all_events", "non_overlapping_events"):
                        base = dataset
                        if overlap_policy == "non_overlapping_events" and meta["horizon"] > 1:
                            base = dataset.loc[_event_subset_mask(dataset.index, int(meta["horizon"]))]
                        row = _one_result(
                            base, predictor, signal_col, signal, target, int(meta["hac_lag"]), family,
                            direction, mean_window, transform, pr_window, group, threshold,
                            overlap_policy, config.annual_rf,
                        )
                        if row:
                            rows.append(row)
    return pd.DataFrame(rows)


def attach_identity_and_corrections(results: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    hashes = []
    for row in out.itertuples():
        valid = dataset[[row.signal_column, row.target]].dropna()
        if row.overlap_policy == "non_overlapping_events":
            horizon = int(V9_TARGET_METADATA[row.target]["horizon"])
            valid = valid.loc[_event_subset_mask(valid.index, horizon)]
        mask = valid[row.signal_column].le(row.threshold) if "_LE_" in row.group else valid[row.signal_column].ge(row.threshold)
        payload = "|".join(valid.index[mask].strftime("%Y-%m-%d"))
        hashes.append(hashlib.sha256(payload.encode()).hexdigest())
    out["signal_mask_hash"] = hashes
    identity_cols = ["signal_mask_hash", "target", "overlap_policy"]
    out["canonical_hypothesis_id"] = out.apply(
        lambda r: hashlib.sha256("|".join(str(r[c]) for c in identity_cols).encode()).hexdigest()[:20], axis=1
    )
    out["is_duplicate_hypothesis"] = out.duplicated(identity_cols)
    out["duplicate_group_size"] = out.groupby(identity_cols, dropna=False)["predictor"].transform("size")
    out["duplicate_reason"] = np.where(out["is_duplicate_hypothesis"], "identical_signal_date_mask", "")
    canonical = ~out["is_duplicate_hypothesis"]
    family_cols = ["predictor_family", "target", "pr_window", "mean_window", "transform", "overlap_policy"]
    for name, p_col in {
        "mean_return": "mean_difference_HAC_p",
        "win_rate": "win_difference_HAC_p",
    }.items():
        fdr, bonf = _bh_bonferroni(out.loc[canonical, p_col])
        out.loc[canonical, f"{name}_FDR_global"] = fdr
        out.loc[canonical, f"{name}_Bonferroni_global"] = bonf
        for _, idx in out.loc[canonical].groupby(family_cols, dropna=False).groups.items():
            fdr, bonf = _bh_bonferroni(out.loc[idx, p_col])
            out.loc[idx, f"{name}_FDR_family"] = fdr
            out.loc[idx, f"{name}_Bonferroni_family"] = bonf
        lookup = out.loc[canonical].drop_duplicates("canonical_hypothesis_id").set_index("canonical_hypothesis_id")
        for suffix in ("FDR_global", "Bonferroni_global", "FDR_family", "Bonferroni_family"):
            col = f"{name}_{suffix}"
            out[col] = out["canonical_hypothesis_id"].map(lookup[col])
    return out


def build_pr_bins(dataset: pd.DataFrame, specs: dict[str, tuple[str, str, int, str]], config: V9Config) -> pd.DataFrame:
    rows = []
    for predictor, (family, direction, mean_window, transform) in specs.items():
        for pr_window in config.pr_windows:
            pr_col = f"{predictor}__PR_{pr_window}"
            for target in V9_TARGET_METADATA:
                base = dataset[[pr_col, target]].dropna()
                for lower, upper in PR_BINS:
                    mask = base[pr_col].ge(lower) & (base[pr_col].le(upper) if upper == 1 else base[pr_col].lt(upper))
                    y = base.loc[mask, target]
                    if y.empty:
                        continue
                    se = y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.nan
                    rows.append({
                        "predictor": predictor, "predictor_family": family, "direction": direction,
                        "mean_window": mean_window, "transform": transform, "pr_window": pr_window,
                        "target": target, "pr_bin": f"{int(lower*100)}-{int(upper*100)}",
                        "N": len(y), "mean_ret": y.mean(), "median_ret": y.median(),
                        "win_rate": y.gt(0).mean(), "mean_ci_lower": y.mean()-1.96*se,
                        "mean_ci_upper": y.mean()+1.96*se,
                    })
    return pd.DataFrame(rows)


def build_v9_yearly(results: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus = results.loc[(~results.is_duplicate_hypothesis) & results.overlap_policy.eq("all_events")]
    for row in focus.itertuples():
        base = dataset[[row.signal_column, row.target]].dropna()
        signal = base[row.signal_column].le(row.threshold) if "_LE_" in row.group else base[row.signal_column].ge(row.threshold)
        unconditional = base[row.target].gt(0).mean()
        for year, part in base.loc[signal].groupby(base.loc[signal].index.year):
            y = part[row.target]
            rows.append({
                "canonical_hypothesis_id": row.canonical_hypothesis_id, "predictor": row.predictor,
                "group": row.group, "target": row.target, "year": int(year), "N": len(y),
                "mean_ret": y.mean(), "median_ret": y.median(), "win_rate": y.gt(0).mean(),
                "unconditional_win_rate_same_sample": unconditional,
                "win_rate_minus_unconditional": y.gt(0).mean()-unconditional,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    key = "canonical_hypothesis_id"
    out["positive_return_years"] = out.groupby(key)["mean_ret"].transform(lambda x: int(x.gt(0).sum()))
    out["negative_return_years"] = out.groupby(key)["mean_ret"].transform(lambda x: int(x.lt(0).sum()))
    out["win_rate_above_unconditional_years"] = out.groupby(key)["win_rate_minus_unconditional"].transform(lambda x: int(x.gt(0).sum()))
    out["win_rate_below_unconditional_years"] = out.groupby(key)["win_rate_minus_unconditional"].transform(lambda x: int(x.lt(0).sum()))
    out["direction_consistency"] = out.groupby(key)["win_rate_minus_unconditional"].transform(
        lambda x: max(x.gt(0).mean(), x.lt(0).mean())
    )
    out["return_contribution"] = out["N"] * out["mean_ret"]
    out["best_year"] = out.groupby(key)["mean_ret"].transform(lambda x: out.loc[x.idxmax(), "year"])
    out["worst_year"] = out.groupby(key)["mean_ret"].transform(lambda x: out.loc[x.idxmin(), "year"])
    out["best_year_abs_contribution_share"] = out.groupby(key)["return_contribution"].transform(
        lambda x: abs(x).max() / abs(x).sum() if abs(x).sum() else np.nan
    )
    return out


def validation_summary_markdown(results: pd.DataFrame, yearly: pd.DataFrame) -> str:
    canonical = results.loc[~results.is_duplicate_hypothesis]
    mean_raw = canonical.mean_difference_HAC_p.lt(.05)
    win_raw = canonical.win_difference_HAC_p.lt(.05)
    mean_family = canonical.mean_return_FDR_family.lt(.05)
    win_family = canonical.win_rate_FDR_family.lt(.05)
    mean_global = canonical.mean_return_FDR_global.lt(.05)
    win_global = canonical.win_rate_FDR_global.lt(.05)
    both = mean_family & win_family
    lines = [
        "# v9 Validation Summary", "",
        f"- Canonical hypotheses: {len(canonical)}",
        f"- Mean return raw / family FDR / global FDR: {mean_raw.sum()} / {mean_family.sum()} / {mean_global.sum()}",
        f"- Win rate raw / family FDR / global FDR: {win_raw.sum()} / {win_family.sum()} / {win_global.sum()}",
        f"- Both mean and win-rate family FDR significant: {both.sum()}",
        f"- Win-rate-only family FDR significant: {(win_family & ~mean_family).sum()}", "",
        "## Interpretation guardrails", "",
        "Raw p<0.05 alone is not evidence of a validated strategy. Review global/family FDR, yearly stability, non-overlapping results, payoff distribution, costs, slippage, event clustering and the absence of untouched OOS data.",
    ]
    return "\n".join(lines) + "\n"
