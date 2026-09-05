from __future__ import annotations

import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import PREDICTOR_SPECS, V6Config


def normalize_datetime_index(obj: pd.DataFrame | pd.Series, name: str):
    out = obj.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        parsed_index = pd.to_datetime(out.index, format="mixed", errors="coerce")
        index_ratio = float(parsed_index.notna().mean()) if len(parsed_index) else 0.0

        # Some FinLab/runtime combinations can expose date × symbol data with
        # the two axes reversed. Only transpose on strong, inspectable evidence.
        if isinstance(out, pd.DataFrame) and index_ratio < 0.8:
            parsed_columns = pd.to_datetime(out.columns, format="mixed", errors="coerce")
            column_ratio = float(parsed_columns.notna().mean()) if len(parsed_columns) else 0.0
            if column_ratio >= 0.8:
                out = out.T
                out.index = parsed_columns
            else:
                raise ValueError(
                    f"{name} index 無法轉為 DatetimeIndex；"
                    f"index_type={type(obj.index).__name__}, "
                    f"index_sample={list(obj.index[:5])}, "
                    f"index_date_ratio={index_ratio:.3f}, "
                    f"column_sample={list(obj.columns[:5])}, "
                    f"column_date_ratio={column_ratio:.3f}"
                )
        elif index_ratio >= 0.8:
            if index_ratio < 1.0:
                invalid = list(out.index[pd.isna(parsed_index)][:5])
                raise ValueError(
                    f"{name} index 含無法解析的日期；invalid_sample={invalid}, "
                    f"valid_ratio={index_ratio:.3f}"
                )
            out.index = parsed_index
        else:
            raise ValueError(
                f"{name} index 無法轉為 DatetimeIndex；"
                f"index_type={type(obj.index).__name__}, "
                f"index_sample={list(obj.index[:5])}, "
                f"index_date_ratio={index_ratio:.3f}"
            )
    if out.index.has_duplicates:
        raise ValueError(f"{name} index 有重複日期")
    return out.sort_index()


def safe_div(numerator, denominator):
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def _aligned_optional(frame: pd.DataFrame | None, close: pd.DataFrame, name: str):
    if frame is None:
        return None
    out = normalize_datetime_index(frame, name).reindex(index=close.index, columns=close.columns)
    return out.apply(pd.to_numeric, errors="coerce")


def get_tick_size(reference_price: pd.DataFrame) -> pd.DataFrame:
    """Taiwan equity tick sizes, matching the referenced production pipeline."""
    values = reference_price.to_numpy(dtype=float)
    ticks = np.select(
        [values < 10, values < 50, values < 100, values < 500, values < 1000],
        [0.01, 0.05, 0.1, 0.5, 1.0],
        default=5.0,
    )
    ticks[~np.isfinite(values) | (values <= 0)] = np.nan
    return pd.DataFrame(ticks, index=reference_price.index, columns=reference_price.columns)


def calculate_limit_prices(
    reference_price: pd.DataFrame,
    *,
    config: V6Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate legal price limits from the reference price and tick size.

    Taiwan widened the daily limit from 7% to 10% on 2015-06-01. The rounding
    rule follows complete-pullback-fubon-pipeline: floor for limit-up and ceil
    for limit-down, both to the tick determined by the reference price.
    """
    cfg = config or V6Config()
    reference = normalize_datetime_index(reference_price, "reference_price").apply(pd.to_numeric, errors="coerce")
    tick = get_tick_size(reference)
    rates = pd.Series(
        np.where(reference.index < pd.Timestamp(cfg.limit_change_date), cfg.old_limit_rate, cfg.current_limit_rate),
        index=reference.index,
    )
    up_multiplier = 1.0 + rates
    down_multiplier = 1.0 - rates
    limit_up = reference.mul(up_multiplier, axis=0).div(tick).apply(np.floor).mul(tick).round(2)
    limit_down = reference.mul(down_multiplier, axis=0).div(tick).apply(np.ceil).mul(tick).round(2)
    return limit_up, limit_down


def build_market_breadth(
    close: pd.DataFrame,
    *,
    reference_price: pd.DataFrame | None = None,
    config: V6Config | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Vectorized breadth features using only close[t] and close[t-1]."""
    cfg = config or V6Config()
    close = normalize_datetime_index(close, "close").apply(pd.to_numeric, errors="coerce")
    previous = close.shift(1)
    valid = close.notna() & previous.notna() & previous.ne(0)
    returns = close.div(previous).sub(1).where(valid)

    up = returns.gt(0) & valid
    down = returns.lt(0) & valid
    flat = returns.eq(0) & valid
    directional = up.sum(axis=1) + down.sum(axis=1)

    out = pd.DataFrame(index=close.index)
    out["up_count"] = up.sum(axis=1).astype("int64")
    out["down_count"] = down.sum(axis=1).astype("int64")
    out["flat_count"] = flat.sum(axis=1).astype("int64")
    out["valid_stock_count"] = valid.sum(axis=1).astype("int64")
    out["up_ratio"] = safe_div(out["up_count"], directional)
    out["down_ratio"] = safe_div(out["down_count"], directional)

    out["up_ratio_3d_mean"] = out["up_ratio"].rolling(3, min_periods=3).mean()
    out["down_ratio_3d_mean"] = out["down_ratio"].rolling(3, min_periods=3).mean()
    out["delta_up_ratio_1d"] = out["up_ratio"].diff()
    out["delta_down_ratio_1d"] = out["down_ratio"].diff()
    out["accel_up_ratio_1d"] = out["delta_up_ratio_1d"].diff()
    out["accel_down_ratio_1d"] = out["delta_down_ratio_1d"].diff()

    threshold = cfg.big_move_threshold
    out["big_up_count"] = returns.ge(threshold).sum(axis=1).astype("int64")
    out["big_down_count"] = returns.le(-threshold).sum(axis=1).astype("int64")
    out["big_up_ratio"] = safe_div(out["big_up_count"], out["valid_stock_count"])
    out["big_down_ratio"] = safe_div(out["big_down_count"], out["valid_stock_count"])

    if reference_price is None:
        # Same baseline as the referenced pipeline: previous valid raw close.
        reference = close.ffill().shift(1)
        reference_method = "previous_valid_raw_close"
    else:
        reference = _aligned_optional(reference_price, close, "reference_price")
        reference_method = "previous_valid_raw_close_with_corporate_action_overrides"
    limit_up, limit_down = calculate_limit_prices(reference, config=cfg)
    tol = cfg.limit_tolerance
    limit_up_mask = valid & limit_up.notna() & close.ge(limit_up - tol)
    limit_down_mask = valid & limit_down.notna() & close.le(limit_down + tol)

    out["limit_up_count"] = limit_up_mask.sum(axis=1).astype("int64")
    out["limit_down_count"] = limit_down_mask.sum(axis=1).astype("int64")
    out["limit_up_ratio"] = safe_div(out["limit_up_count"], out["valid_stock_count"])
    out["limit_down_ratio"] = safe_div(out["limit_down_count"], out["valid_stock_count"])

    metadata = {
        "limit_status_is_approximation": False,
        "limit_detection_method": "reference_price_tick_rounding",
        "limit_reference_method": reference_method,
        "limit_change_date": cfg.limit_change_date,
        "old_limit_rate": cfg.old_limit_rate,
        "current_limit_rate": cfg.current_limit_rate,
        "big_move_threshold": threshold,
        "signal_availability": "t close after market close",
    }
    return out.replace([np.inf, -np.inf], np.nan), metadata


def rolling_percentile_rank(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile of x[t] within [t-window+1, t], never future/full-sample rank."""
    def last_rank(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[np.isfinite(values)]
        if not np.isfinite(current) or len(valid) == 0:
            return np.nan
        return float(np.count_nonzero(valid <= current) / len(valid))

    return series.rolling(window, min_periods=min_periods).apply(last_rank, raw=True)


def add_rolling_normalization(
    dataset: pd.DataFrame,
    predictors: Iterable[str] | None = None,
    *,
    config: V6Config | None = None,
) -> pd.DataFrame:
    cfg = config or V6Config()
    out = dataset.copy()
    names = list(predictors or PREDICTOR_SPECS)
    missing = [name for name in names if name not in out]
    if missing:
        raise ValueError(f"rolling normalization 缺少 predictors: {missing}")
    for name in names:
        s = pd.to_numeric(out[name], errors="coerce")
        out[f"{name}__PR_{cfg.pr_window}"] = rolling_percentile_rank(
            s, cfg.pr_window, cfg.min_history
        )
        rolling = s.rolling(cfg.z_window, min_periods=cfg.min_history)
        mean = rolling.mean()
        std = rolling.std(ddof=1).replace(0, np.nan)
        out[f"{name}__Z_{cfg.z_window}"] = (s - mean) / std
    return out


def add_market_regime(
    dataset: pd.DataFrame,
    close_0050: pd.Series,
    *,
    config: V6Config | None = None,
) -> pd.DataFrame:
    cfg = config or V6Config()
    out = dataset.copy()
    close = normalize_datetime_index(close_0050, "close_0050").reindex(out.index)
    out["close_0050"] = pd.to_numeric(close, errors="coerce")
    out["ma60_0050"] = out["close_0050"].rolling(cfg.ma_window, min_periods=cfg.ma_window).mean()
    out["market_regime"] = np.select(
        [out["close_0050"].gt(out["ma60_0050"]), out["close_0050"].lt(out["ma60_0050"])],
        ["BULL", "BEAR"],
        default="NEUTRAL",
    )
    out.loc[out[["close_0050", "ma60_0050"]].isna().any(axis=1), "market_regime"] = pd.NA
    return out


def add_forward_returns(dataset: pd.DataFrame, open_0050: pd.Series, close_0050: pd.Series) -> pd.DataFrame:
    out = dataset.copy()
    o = normalize_datetime_index(open_0050, "open_0050").reindex(out.index).astype(float)
    c = normalize_datetime_index(close_0050, "close_0050").reindex(out.index).astype(float)
    out["open_0050"] = o
    out["close_0050"] = c
    out["ret_c0_o1"] = o.shift(-1).div(c).sub(1)
    out["ret_c0_c1"] = c.shift(-1).div(c).sub(1)
    out["ret_o1_c1"] = c.shift(-1).div(o.shift(-1)).sub(1)
    out["ret_o1_o2"] = o.shift(-2).div(o.shift(-1)).sub(1)
    out["ret_o1_c2"] = c.shift(-2).div(o.shift(-1)).sub(1)
    out["ret_o1_c3"] = c.shift(-3).div(o.shift(-1)).sub(1)
    return out.replace([np.inf, -np.inf], np.nan)
