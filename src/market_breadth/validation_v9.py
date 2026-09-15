from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V9Config, V9_TARGET_METADATA
from .v9 import historical_percentile_rank


def validate_v9(dataset: pd.DataFrame, results: pd.DataFrame, config: V9Config) -> dict[str, pd.DataFrame]:
    if not dataset.index.is_monotonic_increasing or dataset.index.has_duplicates:
        raise AssertionError("v9 index must be ascending and unique")
    if dataset.index.min() < pd.Timestamp(config.start_date):
        raise AssertionError("Primary sample contains pre-2015-06-01 rows")
    if np.isinf(dataset.select_dtypes(include=[np.number]).to_numpy()).any():
        raise AssertionError("dataset contains inf")
    for target, meta in V9_TARGET_METADATA.items():
        if target not in dataset:
            raise AssertionError(f"missing target {target}")
        if not results.loc[results.target.eq(target), "hac_lag"].eq(meta["hac_lag"]).all():
            raise AssertionError(f"incorrect HAC lag for {target}")
    sample = pd.Series(range(1, 8), dtype=float)
    pr = historical_percentile_rank(sample, 3)
    if pr.iloc[3] != 1.0:
        raise AssertionError("PR does not exclude current t")
    if not (results.successes + results.failures).eq(results.N).all():
        raise AssertionError("win-rate counts do not sum to N")
    if not np.allclose(results.signal_win_rate, results.successes / results.N):
        raise AssertionError("win-rate identity failed")
    if results.duplicated(["predictor", "mean_window", "transform", "pr_window", "group", "target", "overlap_policy"]).any():
        raise AssertionError("duplicate raw result key")
    probability = [c for c in results if c.endswith("_p") or "FDR" in c or "Bonferroni" in c]
    for col in probability:
        values = pd.to_numeric(results[col], errors="coerce").dropna()
        if not values.between(0, 1).all():
            raise AssertionError(f"{col} outside [0,1]")
    if np.isinf(results.select_dtypes(include=[np.number]).to_numpy()).any():
        raise AssertionError("results contain inf")
    return {
        "data": pd.DataFrame([{"check": "post_2015_unique_ascending_no_inf", "passed": True}]),
        "pr": pd.DataFrame([{"check": "PR reference excludes t", "passed": True}]),
        "statistics": pd.DataFrame([{"check": "win counts, p/q, HAC lags and keys", "passed": True, "rows": len(results)}]),
    }
