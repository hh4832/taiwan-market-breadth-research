from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V10Config, V10_RAW_PREDICTORS, V9_TARGET_METADATA
from .v10 import PR_BINS, historical_percentile_rank


def validate_v10(
    dataset: pd.DataFrame, results: pd.DataFrame, config: V10Config,
    selected_keys: dict[str, str | None] | None = None,
) -> dict[str, pd.DataFrame]:
    checks: list[dict[str, object]] = []
    if not dataset.index.is_monotonic_increasing or dataset.index.has_duplicates:
        raise AssertionError("v10 dates must be ascending and unique")
    if dataset.index.min() < pd.Timestamp(config.start_date):
        raise AssertionError("primary sample contains pre-2015-06-01 rows")
    if np.isinf(dataset.select_dtypes(include=[np.number]).to_numpy()).any():
        raise AssertionError("dataset contains inf")
    if selected_keys is not None and (selected_keys.get("adj_open"), selected_keys.get("adj_close")) != ("etl:adj_open", "etl:adj_close"):
        raise AssertionError("0050 outcomes are not adjusted")
    identity = dataset[["up_ratio", "down_ratio"]].dropna().sum(axis=1)
    if not np.allclose(identity, 1.0, atol=1e-10):
        raise AssertionError("up_ratio + down_ratio identity failed")
    checks.append({"check": "post-2015, ascending, unique, adjusted, participation identity", "passed": True})

    for raw in V10_RAW_PREDICTORS:
        source = pd.to_numeric(dataset[raw], errors="coerce")
        for window in config.signal_mean_windows:
            level = f"{raw}__mean_{window}d"; delta = f"{level}__delta"
            expected = source.rolling(window, min_periods=window).mean()
            if not np.allclose(dataset[level], expected, equal_nan=True):
                raise AssertionError(f"trailing mean identity failed: {level}")
            if not np.allclose(dataset[delta], expected.diff(), equal_nan=True):
                raise AssertionError(f"delta identity failed: {delta}")
    checks.append({"check": "1/3/5 trailing means and level-to-delta identities", "passed": True})

    sample = pd.Series(range(1, 8), dtype=float)
    if historical_percentile_rank(sample, 3).iloc[3] != 1.0:
        raise AssertionError("PR reference includes t or future")
    pr_columns = [c for c in dataset if "__PR_" in c and "__BIN_" not in c]
    if not pr_columns or {int(c.rsplit("_", 1)[-1]) for c in pr_columns} != set(config.pr_windows):
        raise AssertionError("PR60/126/252 columns incomplete")
    for pr_col in pr_columns:
        bin_cols = [f"{pr_col}__{name}" for name, _, _ in PR_BINS]
        if not set(bin_cols).issubset(dataset.columns):
            raise AssertionError(f"missing bins for {pr_col}")
        valid = dataset[pr_col].notna()
        membership = dataset.loc[valid, bin_cols].sum(axis=1)
        if not membership.eq(1).all():
            raise AssertionError(f"PR bins are not exclusive/exhaustive: {pr_col}")
    checks.append({"check": "strict historical PR and 7 exclusive/exhaustive bins", "passed": True})

    for target, meta in V9_TARGET_METADATA.items():
        if target not in dataset or not results.loc[results.target.eq(target), "hac_lag"].eq(meta["hac_lag"]).all():
            raise AssertionError(f"target/HAC mapping failed: {target}")
    if not (results.successes + results.failures).eq(results.N).all():
        raise AssertionError("successes + failures != N")
    if not np.allclose(results.bin_win_rate, results.successes / results.N):
        raise AssertionError("win-rate identity failed")
    keys = ["predictor", "mean_window", "transform", "pr_window", "pr_bin", "target", "overlap_policy"]
    if results.duplicated(keys).any():
        raise AssertionError("duplicate result keys")
    canonical = results.loc[~results.is_duplicate_hypothesis]
    if canonical.canonical_hypothesis_id.duplicated().any():
        raise AssertionError("duplicate canonical IDs")
    probability = [c for c in results if c.endswith("_p") or "FDR" in c or "Bonferroni" in c]
    for column in probability:
        values = pd.to_numeric(results[column], errors="coerce").dropna()
        if not values.between(0, 1).all():
            raise AssertionError(f"{column} outside [0,1]")
    if np.isinf(results.select_dtypes(include=[np.number]).to_numpy()).any():
        raise AssertionError("results contain inf")
    mean_ci = results[["mean_difference_ci_lower", "mean_difference", "mean_difference_ci_upper"]].dropna()
    if not (mean_ci.mean_difference_ci_lower.le(mean_ci.mean_difference) & mean_ci.mean_difference.le(mean_ci.mean_difference_ci_upper)).all():
        raise AssertionError("mean difference CI failed")
    win_ci = results[["win_difference_ci_lower", "bin_minus_non_bin_win_rate", "win_difference_ci_upper"]].dropna()
    if not (win_ci.win_difference_ci_lower.le(win_ci.bin_minus_non_bin_win_rate) & win_ci.bin_minus_non_bin_win_rate.le(win_ci.win_difference_ci_upper)).all():
        raise AssertionError("win difference CI failed")
    checks.append({"check": "statistics identities, keys, HAC, p/q, CI and finite values", "passed": True})
    return {
        "data": pd.DataFrame(checks[:2]),
        "pr_bins": pd.DataFrame(checks[2:3]),
        "statistics": pd.DataFrame(checks[3:]),
    }
