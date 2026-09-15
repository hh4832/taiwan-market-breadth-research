from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_breadth.config import V10Config, V10_RAW_PREDICTORS, V9_TARGET_METADATA
from market_breadth.validation_v10 import validate_v10
from market_breadth.v10 import (PR_BINS, add_v10_pr_features,
                                attach_identity_and_corrections,
                                build_shape_analysis, build_v10_features,
                                historical_percentile_rank, run_v10_bin_study)


class V10BinStudyTests(unittest.TestCase):
    def setUp(self):
        self.config = V10Config()
        self.index = pd.date_range("2015-06-01", periods=420)
        x = np.arange(len(self.index), dtype=float)
        up = .5 + .45 * np.sin(x / 11)
        self.breadth = pd.DataFrame({
            "up_ratio": up, "down_ratio": 1 - up,
            "big_up_ratio": .08 + .06 * (np.sin(x / 7) + 1) / 2,
            "big_down_ratio": .08 + .06 * (np.cos(x / 7) + 1) / 2,
            "limit_up_ratio": .02 + .02 * (np.sin(x / 9) + 1) / 2,
            "limit_down_ratio": .02 + .02 * (np.cos(x / 9) + 1) / 2,
        }, index=self.index)

    def _dataset(self):
        data, specs = build_v10_features(self.breadth, self.config)
        data = add_v10_pr_features(data, specs, self.config)
        returns = .012 * np.sin(np.arange(len(data)) / 5)
        targets = pd.DataFrame({target: returns for target in V9_TARGET_METADATA}, index=data.index)
        return pd.concat([data, targets], axis=1), specs

    def test_predictors_means_and_delta(self):
        data, specs = build_v10_features(self.breadth, self.config)
        self.assertEqual(set(V10_RAW_PREDICTORS), set(self.breadth.columns))
        self.assertEqual(len(specs), 6 * 3 * 2)
        expected = self.breadth.up_ratio.rolling(3, min_periods=3).mean()
        pd.testing.assert_series_equal(data["up_ratio__mean_3d"], expected, check_names=False)
        pd.testing.assert_series_equal(data["up_ratio__mean_3d__delta"], expected.diff(), check_names=False)

    def test_strict_pr_and_exclusive_exhaustive_bins(self):
        self.assertEqual(historical_percentile_rank(pd.Series([1., 2., 3., 100.]), 3).iloc[-1], 1.0)
        data, _ = self._dataset()
        for window in self.config.pr_windows:
            pr = f"up_ratio__mean_1d__PR_{window}"
            bins = [f"{pr}__{name}" for name, _, _ in PR_BINS]
            valid = data[pr].notna()
            self.assertTrue(data.loc[valid, bins].sum(axis=1).eq(1).all())

    def test_primary_statistics_non_overlap_dedup_and_shape(self):
        data, specs = self._dataset()
        subset = {
            "up_ratio__mean_1d": specs["up_ratio__mean_1d"],
            "down_ratio__mean_1d": specs["down_ratio__mean_1d"],
        }
        results = attach_identity_and_corrections(run_v10_bin_study(data, subset, self.config), data)
        self.assertTrue((results.successes + results.failures).eq(results.N).all())
        self.assertTrue(np.allclose(results.bin_win_rate, results.successes / results.N))
        self.assertEqual(set(results.loc[results.target.eq("ret_o1_c1"), "overlap_policy"]), {"all_events"})
        self.assertIn("non_overlapping_events", set(results.loc[results.target.eq("ret_o1_c20"), "overlap_policy"]))
        self.assertTrue(results.is_duplicate_hypothesis.any())
        self.assertFalse(build_shape_analysis(results).empty)
        self.assertTrue(set(results.sample_size_flag).issubset({"VERY_SMALL_N", "SMALL_N", "LOW_POWER", "ADEQUATE"}))

    def test_validation_checks_adjusted_source_and_hac(self):
        data, specs = self._dataset()
        subset = {"down_ratio__mean_1d": specs["down_ratio__mean_1d"]}
        results = attach_identity_and_corrections(run_v10_bin_study(data, subset, self.config), data)
        checks = validate_v10(data, results, self.config, {"adj_open": "etl:adj_open", "adj_close": "etl:adj_close"})
        self.assertTrue(all(frame.passed.all() for frame in checks.values()))
        with self.assertRaises(AssertionError):
            validate_v10(data, results, self.config, {"adj_open": "price:開盤價", "adj_close": "etl:adj_close"})


if __name__ == "__main__":
    unittest.main()
