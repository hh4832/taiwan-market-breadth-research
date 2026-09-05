import unittest
import warnings

import numpy as np
import pandas as pd

from market_breadth.config import PREDICTOR_SPECS, TARGET_METADATA, V6Config
from market_breadth.core import (
    add_forward_returns,
    add_market_regime,
    add_rolling_normalization,
    build_market_breadth,
    calculate_limit_prices,
    rolling_percentile_rank,
)
from market_breadth.statistics import run_signal_study
from market_breadth.validation import validate_v6


class V6CoreTest(unittest.TestCase):
    def setUp(self):
        self.index = pd.bdate_range("2020-01-01", periods=320)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0003, 0.025, size=(320, 30))
        prices = 50 * np.cumprod(1 + returns, axis=0)
        self.close = pd.DataFrame(prices, index=self.index, columns=[f"{1000+i}" for i in range(30)])
        self.cfg = V6Config(pr_window=60, z_window=60, min_history=20, ma_window=20)

    def test_breadth_identity_and_transforms(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            breadth, metadata = build_market_breadth(self.close, config=self.cfg)
        valid = (breadth.up_count + breadth.down_count) > 0
        np.testing.assert_allclose((breadth.loc[valid, "up_ratio"] + breadth.loc[valid, "down_ratio"]).to_numpy(), 1)
        pd.testing.assert_series_equal(breadth["delta_down_ratio_1d"], breadth["down_ratio"].diff(), check_names=False)
        self.assertFalse(metadata["limit_status_is_approximation"])
        self.assertEqual(set(PREDICTOR_SPECS) - set(breadth.columns), set())

    def test_official_limit_price_path(self):
        reference = self.close.ffill().shift(1)
        limit_up, limit_down = calculate_limit_prices(reference, config=self.cfg)
        close = self.close.copy()
        close.iloc[100, 0] = limit_up.iloc[100, 0]
        close.iloc[101, 1] = limit_down.iloc[101, 1]
        breadth, metadata = build_market_breadth(close, reference_price=reference, config=self.cfg)
        self.assertFalse(metadata["limit_status_is_approximation"])
        self.assertEqual(breadth.iloc[100].limit_up_count, 1)
        self.assertEqual(breadth.iloc[101].limit_down_count, 1)

    def test_historical_limit_rate_and_tick_rounding(self):
        idx = pd.to_datetime(["2015-05-29", "2015-06-01"])
        reference = pd.DataFrame({"2330": [100.0, 100.0]}, index=idx)
        limit_up, limit_down = calculate_limit_prices(reference)
        self.assertEqual(limit_up.loc[idx[0], "2330"], 107.0)
        self.assertEqual(limit_down.loc[idx[0], "2330"], 93.0)
        self.assertEqual(limit_up.loc[idx[1], "2330"], 110.0)
        self.assertEqual(limit_down.loc[idx[1], "2330"], 90.0)

    def test_rolling_pr_has_no_future_leakage(self):
        s = pd.Series(np.arange(100, dtype=float), index=self.index[:100])
        before = rolling_percentile_rank(s, 20, 5)
        changed = s.copy(); changed.iloc[-1] = -999
        after = rolling_percentile_rank(changed, 20, 5)
        pd.testing.assert_series_equal(before.iloc[:-1], after.iloc[:-1])
        self.assertNotEqual(before.iloc[-1], after.iloc[-1])

    def test_targets_regime_statistics_and_validation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            breadth, _ = build_market_breadth(self.close, config=self.cfg)
        target_returns = np.linspace(-.01, .012, len(self.index))
        close_0050 = pd.Series(100 * np.cumprod(1 + target_returns), index=self.index)
        open_0050 = close_0050.shift(1).fillna(close_0050.iloc[0]) * 1.0005
        dataset = add_forward_returns(breadth, open_0050, close_0050)
        dataset = add_market_regime(dataset, close_0050, config=self.cfg)
        dataset = add_rolling_normalization(dataset, config=self.cfg)
        results = run_signal_study(dataset, config=self.cfg)
        self.assertFalse(results.empty)
        self.assertEqual(set(results.target), set(TARGET_METADATA))
        self.assertEqual(set(results.signal_method), {"PR", "Z"})
        validations = validate_v6(self.close, breadth, dataset, results, config=self.cfg)
        self.assertIn("results", validations)


if __name__ == "__main__":
    unittest.main()
