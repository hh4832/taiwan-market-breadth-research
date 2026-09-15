from __future__ import annotations

import numpy as np
import pandas as pd

from .config import V11Config, V9_TARGET_METADATA
from .core import calculate_limit_prices
from .v11 import historical_raw_threshold


def validate_v11(
    dataset: pd.DataFrame, prior_desc: pd.DataFrame, regressions: pd.DataFrame,
    mapping_daily: pd.DataFrame, regime: pd.DataFrame, interactions: pd.DataFrame,
    config: V11Config, selected: dict[str, str | None] | None = None,
) -> dict[str, pd.DataFrame]:
    rows: dict[str, list[dict[str, object]]] = {"prior_return": [], "pr_mapping": [], "regime": [], "interaction": []}
    close = dataset.close_0050
    for window in config.prior_return_windows:
        expected = close.div(close.shift(window)).sub(1)
        if not np.allclose(dataset[f"prior_ret_{window}d"], expected, equal_nan=True):
            raise AssertionError(f"prior_ret_{window}d identity/leakage failure")
    if selected is not None and (selected.get("adj_open"), selected.get("adj_close")) != ("etl:adj_open", "etl:adj_close"):
        raise AssertionError("v11 outcomes/prior returns require adjusted 0050 prices")
    rows["prior_return"].append({"check": "prior returns use adjusted C[t]/C[t-k]-1 only", "passed": True})

    for item in mapping_daily.itertuples():
        predictor = item.predictor; window = int(item.pr_window); quantile = float(item.quantile)
        current = dataset.at[pd.Timestamp(item.date), predictor]
        assigned = dataset.at[pd.Timestamp(item.date), f"{predictor}__PR_{window}"] >= quantile
        if bool(current >= item.raw_threshold) != bool(assigned):
            raise AssertionError(f"raw threshold/PR assignment mismatch: {predictor}/{window}/{quantile}/{item.date}")
    sample = pd.Series([1., 2., 3., 100.])
    if historical_raw_threshold(sample, 3, .95).iloc[-1] != 3.:
        raise AssertionError("threshold mapping did not exclude current t")
    rows["pr_mapping"].append({"check": "PR95/60/80 raw mapping uses t-1 history and matches assignment", "passed": True})

    change = pd.Timestamp(config.regime_change_date)
    pre = dataset.index < change; post = dataset.index >= change
    if (pre & post).any() or not pre.any() or not post.any() or dataset.index[pre].max() > pd.Timestamp("2015-05-31") or dataset.index[post].min() < change:
        raise AssertionError("regime split boundary/overlap failure")
    refs = pd.DataFrame({"x": [100., 100.]}, index=pd.to_datetime(["2015-05-29", "2015-06-01"]))
    upper, lower = calculate_limit_prices(refs, config=config)
    if not np.isclose(upper.iloc[0, 0], 107.) or not np.isclose(lower.iloc[0, 0], 93.) or not np.isclose(upper.iloc[1, 0], 110.) or not np.isclose(lower.iloc[1, 0], 90.):
        raise AssertionError("7%/10% historical limit rule failure")
    if set(regime.regime.dropna()) - {"PRE_2015_7PCT", "POST_2015_10PCT"}:
        raise AssertionError("unexpected regime label")
    rows["regime"].append({"check": "pre/post boundary and historical 7%/10% rules", "passed": True})

    for target, meta in V9_TARGET_METADATA.items():
        part = interactions.loc[interactions.target.eq(target)]
        if len(part) and not part.hac_lag.eq(meta["hac_lag"]).all():
            raise AssertionError(f"interaction HAC lag mismatch: {target}")
    if not {"beta_signal_pre", "beta_interaction", "interaction_HAC_p"}.issubset(interactions.columns):
        raise AssertionError("interaction coefficient extraction incomplete")
    for frame in (regressions, regime, interactions):
        numeric = frame.select_dtypes(include=[np.number])
        if np.isinf(numeric.to_numpy()).any():
            raise AssertionError("infinite statistical output")
    rows["interaction"].append({"check": "post dummy, signal×post coefficient and HAC lag outputs", "passed": True})
    return {key: pd.DataFrame(value) for key, value in rows.items()}
