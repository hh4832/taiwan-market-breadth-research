from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PREDICTOR_SPECS, TARGET_METADATA, V6Config


def _assert_probability_columns(results: pd.DataFrame) -> None:
    probability_tokens = ("_p", "_FDR_", "_Bonferroni_", "prob_ret_")
    columns = [c for c in results if any(token in c for token in probability_tokens)]
    columns += [c for c in ("win_rate",) if c in results]
    for column in columns:
        values = pd.to_numeric(results[column], errors="coerce").dropna()
        if not values.between(0, 1).all():
            raise AssertionError(f"{column} 有值不在 [0, 1]")


def validate_v6(
    close: pd.DataFrame,
    breadth: pd.DataFrame,
    dataset: pd.DataFrame,
    results: pd.DataFrame | None = None,
    *,
    config: V6Config | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or V6Config()
    if not close.index.is_monotonic_increasing or not dataset.index.is_monotonic_increasing:
        raise AssertionError("日期未遞增排序")
    if close.index.has_duplicates or dataset.index.has_duplicates:
        raise AssertionError("資料存在 duplicate date key")
    numeric = dataset.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy()).any():
        raise AssertionError("dataset 存在 inf")

    directional = breadth["up_count"] + breadth["down_count"]
    identity_mask = directional.gt(0)
    identity_error = (breadth.loc[identity_mask, "up_ratio"] + breadth.loc[identity_mask, "down_ratio"] - 1).abs()
    if not identity_error.le(1e-12).all():
        raise AssertionError("up_ratio + down_ratio != 1")

    checks = {
        "up_ratio_3d_mean": breadth["up_ratio"].rolling(3, min_periods=3).mean(),
        "down_ratio_3d_mean": breadth["down_ratio"].rolling(3, min_periods=3).mean(),
        "delta_up_ratio_1d": breadth["up_ratio"].diff(),
        "delta_down_ratio_1d": breadth["down_ratio"].diff(),
        "accel_up_ratio_1d": breadth["up_ratio"].diff().diff(),
        "accel_down_ratio_1d": breadth["down_ratio"].diff().diff(),
    }
    transform_rows = []
    for name, expected in checks.items():
        actual = breadth[name]
        equal = np.allclose(actual.to_numpy(float), expected.to_numpy(float), equal_nan=True)
        transform_rows.append({"check": name, "passed": bool(equal)})
        if not equal:
            raise AssertionError(f"{name} validation failed")

    if "market_regime" in dataset:
        expected_bull = dataset["close_0050"].gt(dataset["ma60_0050"])
        actual_bull = dataset["market_regime"].eq("BULL")
        available = dataset[["close_0050", "ma60_0050"]].notna().all(axis=1)
        if not expected_bull.loc[available].equals(actual_bull.loc[available]):
            raise AssertionError("MA60 regime validation failed")

    normalized_rows = []
    for predictor in PREDICTOR_SPECS:
        pr_col = f"{predictor}__PR_{cfg.pr_window}"
        z_col = f"{predictor}__Z_{cfg.z_window}"
        for col in (pr_col, z_col):
            if col not in dataset:
                raise AssertionError(f"缺少 {col}")
        pr = dataset[pr_col].dropna()
        if not pr.between(0, 1).all():
            raise AssertionError(f"{pr_col} 不在 [0,1]")
        normalized_rows.append({"predictor": predictor, "pr_count": len(pr), "z_count": int(dataset[z_col].notna().sum())})

    output = {
        "identity": pd.DataFrame([{"check": "up_ratio_plus_down_ratio", "passed": True, "max_abs_error": identity_error.max()}]),
        "transforms": pd.DataFrame(transform_rows),
        "normalization": pd.DataFrame(normalized_rows),
    }
    if results is not None and not results.empty:
        ci_available = results[["mean_ret_ci_lower", "mean_ret", "mean_ret_ci_upper"]].notna().all(axis=1)
        if not (results.loc[ci_available, "mean_ret_ci_lower"].le(results.loc[ci_available, "mean_ret"]) & results.loc[ci_available, "mean_ret"].le(results.loc[ci_available, "mean_ret_ci_upper"])).all():
            raise AssertionError("HAC CI 未包含 mean_ret")
        if not np.allclose(results["win_rate_minus_50pct"], results["win_rate"] - 0.5):
            raise AssertionError("win_rate_minus_50pct 錯誤")
        if not (results["binomial_successes"] + results["binomial_failures"] == results["observation_count"]).all():
            raise AssertionError("binomial counts 與 observation_count 不一致")
        key = ["predictor", "predictor_family", "signal_method", "target", "group_type", "group", "market_regime"]
        if results.duplicated(key).any():
            raise AssertionError("result key 重複")
        _assert_probability_columns(results)
        if not results.apply(lambda row: row["hac_lag"] == TARGET_METADATA[row["target"]]["hac_lag"], axis=1).all():
            raise AssertionError("HAC lag 與 target metadata 不一致")
        output["results"] = pd.DataFrame([{"check": "result_integrity", "passed": True, "rows": len(results)}])
    return output
