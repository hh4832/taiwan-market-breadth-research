from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from .config import PREDICTOR_SPECS, PR_GROUPS, TARGET_METADATA, V6Config, Z_GROUPS


def _hac_mean_test(values: np.ndarray, lag: int) -> tuple[float, float, float, float]:
    """Intercept-only Newey-West test with Bartlett weights."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < max(3, lag + 2):
        return np.nan, np.nan, np.nan, np.nan
    mean = float(x.mean())
    residual = x - mean
    long_run = float(residual @ residual / n)
    for k in range(1, min(lag, n - 1) + 1):
        gamma = float(residual[k:] @ residual[:-k] / n)
        long_run += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    variance_mean = max(long_run / n, 0.0)
    if variance_mean == 0:
        return np.nan, np.nan, mean, mean
    se = np.sqrt(variance_mean)
    t_value = mean / se
    p_value = float(2 * stats.t.sf(abs(t_value), df=n - 1))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    return float(t_value), p_value, mean - critical * se, mean + critical * se


def _weighted_difference_test(y: np.ndarray, group: np.ndarray, lag: int, baseline: str):
    """HAC test for group mean minus non-group or unconditional mean."""
    y = np.asarray(y, dtype=float)
    g = np.asarray(group, dtype=bool)
    valid = np.isfinite(y)
    y, g = y[valid], g[valid]
    n, ng = len(y), int(g.sum())
    nn = n - ng
    if n < 3 or ng < 2 or (baseline == "non_group" and nn < 2):
        return np.nan, np.nan, np.nan
    if baseline == "non_group":
        weights = np.where(g, n / ng, -n / nn)
        estimate = y[g].mean() - y[~g].mean()
    elif baseline == "unconditional":
        weights = np.where(g, n / ng - 1.0, -1.0)
        estimate = y[g].mean() - y.mean()
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    transformed = weights * y
    t_value, p_value, _, _ = _hac_mean_test(transformed, lag)
    return float(estimate), t_value, p_value


def _cohen_d(group: np.ndarray, other: np.ndarray) -> float:
    a = np.asarray(group, dtype=float)
    b = np.asarray(other, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return np.nan
    return float((a.mean() - b.mean()) / np.sqrt(pooled_var))


def _describe(y: np.ndarray, annual_rf: float) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return {"observation_count": 0}
    std = y.std(ddof=1) if n > 1 else np.nan
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    sharpe = np.sqrt(252) * (y.mean() - daily_rf) / std if std and np.isfinite(std) else np.nan
    wins = int(np.count_nonzero(y > 0))
    losses = n - wins
    binom_p = float(stats.binomtest(wins, n, 0.5).pvalue)
    stable_moments = n > 3 and np.isfinite(std) and std > np.finfo(float).eps * max(1.0, abs(y.mean()))
    return {
        "observation_count": n,
        "mean_ret": float(y.mean()),
        "median_ret": float(np.median(y)),
        "std_ret": float(std),
        "win_rate": wins / n,
        "win_rate_minus_50pct": wins / n - 0.5,
        "binomial_successes": wins,
        "binomial_failures": losses,
        "binomial_p_value": binom_p,
        "annualized_sharpe": float(sharpe),
        "skewness": float(stats.skew(y, bias=False)) if stable_moments else np.nan,
        "kurtosis": float(stats.kurtosis(y, fisher=True, bias=False)) if stable_moments else np.nan,
        "min_ret": float(y.min()),
        "max_ret": float(y.max()),
        "q05_ret": float(np.quantile(y, 0.05)),
        "q25_ret": float(np.quantile(y, 0.25)),
        "q75_ret": float(np.quantile(y, 0.75)),
        "q95_ret": float(np.quantile(y, 0.95)),
        "prob_ret_lt_minus_1pct": float(np.mean(y < -0.01)),
        "prob_ret_lt_minus_2pct": float(np.mean(y < -0.02)),
        "prob_ret_gt_plus_1pct": float(np.mean(y > 0.01)),
        "prob_ret_gt_plus_2pct": float(np.mean(y > 0.02)),
    }


def _mask_for_group(signal: pd.Series, operator: str, threshold) -> pd.Series:
    if operator == "le":
        return signal.le(threshold)
    if operator == "ge":
        return signal.ge(threshold)
    if operator == "between":
        lower, upper = threshold
        # Avoid overlapping quintiles while retaining PR==0 in Q1 and PR==1 in Q5.
        return signal.ge(lower) & (signal.le(upper) if upper == 1 else signal.lt(upper))
    raise ValueError(f"unknown operator: {operator}")


def _bh_bonferroni(p_values: pd.Series) -> tuple[pd.Series, pd.Series]:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    bonf = pd.Series(np.nan, index=p.index, dtype=float)
    fdr = pd.Series(np.nan, index=p.index, dtype=float)
    if not valid.any():
        return fdr, bonf
    vals = p.loc[valid].clip(0, 1).to_numpy()
    m = len(vals)
    bonf.loc[valid] = np.minimum(vals * m, 1.0)
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(m)
    restored[order] = np.minimum(adjusted, 1.0)
    fdr.loc[valid] = restored
    return fdr, bonf


def add_multiple_testing_corrections(results: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    out = results.copy()
    families = {
        "group_vs_non_group": "HAC_p_value",
        "group_vs_unconditional": "unconditional_HAC_p",
        "mean_vs_zero": "HAC_mean_vs_zero_p",
        "binomial_win_rate": "binomial_p_value",
    }
    for family, p_col in families.items():
        global_fdr, global_bonf = _bh_bonferroni(out[p_col])
        out[f"{family}_FDR_global{suffix}"] = global_fdr
        out[f"{family}_Bonferroni_global{suffix}"] = global_bonf
        family_fdr = pd.Series(np.nan, index=out.index, dtype=float)
        family_bonf = pd.Series(np.nan, index=out.index, dtype=float)
        # A research family holds predictor family, target, regime, and signal method fixed.
        keys = ["predictor_family", "target", "market_regime", "signal_method"]
        for _, idx in out.groupby(keys, dropna=False).groups.items():
            fdr, bonf = _bh_bonferroni(out.loc[idx, p_col])
            family_fdr.loc[idx], family_bonf.loc[idx] = fdr, bonf
        out[f"{family}_FDR_family{suffix}"] = family_fdr
        out[f"{family}_Bonferroni_family{suffix}"] = family_bonf
    return out


def run_signal_study(
    dataset: pd.DataFrame,
    *,
    predictors: Iterable[str] | None = None,
    config: V6Config | None = None,
) -> pd.DataFrame:
    cfg = config or V6Config()
    predictor_names = list(predictors or PREDICTOR_SPECS)
    rows: list[dict[str, object]] = []
    regimes = {
        "ALL": pd.Series(True, index=dataset.index),
        "BULL": dataset["market_regime"].eq("BULL"),
        "BEAR": dataset["market_regime"].eq("BEAR"),
    }
    methods = {
        "PR": (f"__PR_{cfg.pr_window}", PR_GROUPS),
        "Z": (f"__Z_{cfg.z_window}", Z_GROUPS),
    }
    for predictor in predictor_names:
        family, direction = PREDICTOR_SPECS[predictor]
        for signal_method, (suffix, groups) in methods.items():
            signal_col = predictor + suffix
            if signal_col not in dataset:
                raise ValueError(f"缺少 normalized signal: {signal_col}")
            for target, target_meta in TARGET_METADATA.items():
                if target not in dataset:
                    raise ValueError(f"缺少 target: {target}")
                lag = int(target_meta["hac_lag"])
                for regime_name, regime_mask in regimes.items():
                    base = dataset.loc[regime_mask, [predictor, signal_col, target]].dropna()
                    if base.empty:
                        continue
                    y_all = base[target].to_numpy(float)
                    for group_name, (operator, threshold) in groups.items():
                        group_mask = _mask_for_group(base[signal_col], operator, threshold).to_numpy(bool)
                        y_group, y_other = y_all[group_mask], y_all[~group_mask]
                        desc = _describe(y_group, cfg.annual_rf)
                        if desc.get("observation_count", 0) == 0:
                            continue
                        t0, p0, lo0, hi0 = _hac_mean_test(y_group, lag)
                        diff_non, t_non, p_non = _weighted_difference_test(y_all, group_mask, lag, "non_group")
                        diff_unc, t_unc, p_unc = _weighted_difference_test(y_all, group_mask, lag, "unconditional")
                        rows.append({
                            "predictor": predictor,
                            "predictor_family": family,
                            "direction": direction,
                            "signal_method": signal_method,
                            "signal_column": signal_col,
                            "group_method": f"rolling_{signal_method.lower()}_{cfg.pr_window if signal_method == 'PR' else cfg.z_window}",
                            "group_threshold": str(threshold) if isinstance(threshold, tuple) else threshold,
                            "group_type": "quintile" if group_name.startswith("Q") else ("descriptive_tail" if group_name in {"LOW_5", "HIGH_5"} else "tradable_tail"),
                            "group": group_name,
                            "market_regime": regime_name,
                            "target": target,
                            "hac_lag": lag,
                            "raw_value_mean": float(base.loc[group_mask, predictor].mean()),
                            **desc,
                            "non_group_mean_ret": float(np.mean(y_other)) if len(y_other) else np.nan,
                            "mean_ret_minus_non_group": diff_non,
                            "HAC_t": t_non,
                            "HAC_p_value": p_non,
                            "Cohen_d": _cohen_d(y_group, y_other),
                            "unconditional_mean_ret": float(np.mean(y_all)),
                            "mean_ret_minus_unconditional": diff_unc,
                            "unconditional_HAC_t": t_unc,
                            "unconditional_HAC_p": p_unc,
                            "Cohen_d_vs_unconditional": float((np.mean(y_group) - np.mean(y_all)) / np.std(y_all, ddof=1)) if len(y_all) > 1 and np.std(y_all, ddof=1) > 0 else np.nan,
                            "HAC_mean_vs_zero_t": t0,
                            "HAC_mean_vs_zero_p": p0,
                            "mean_ret_ci_lower": lo0,
                            "mean_ret_ci_upper": hi0,
                        })
    if not rows:
        return pd.DataFrame()
    return add_multiple_testing_corrections(pd.DataFrame(rows))
