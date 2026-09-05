import unittest

import numpy as np
import pandas as pd

from market_breadth.config import V7Config
from market_breadth.core import add_forward_returns, add_market_regime, add_rolling_normalization, build_market_breadth
from market_breadth.robustness import (
    _non_overlapping_mask,
    _stable_mask_hash,
    add_deduplicated_corrections,
    attach_hypothesis_identity,
    build_limit_up_pullback_validation,
    build_quintile_trend_results,
    build_yearly_stability,
)
from market_breadth.statistics import run_signal_study


class V7RobustnessTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = pd.bdate_range("2018-01-01", periods=720)
        rng = np.random.default_rng(7)
        stock_returns = rng.normal(.0002, .025, (len(cls.index), 40))
        close = pd.DataFrame(
            50 * np.cumprod(1 + stock_returns, axis=0), index=cls.index,
            columns=[str(1000 + i) for i in range(40)],
        )
        cls.cfg = V7Config(pr_window=60, z_window=60, min_history=20, ma_window=20)
        breadth, _ = build_market_breadth(close, config=cls.cfg)
        target_returns = rng.normal(.0003, .009, len(cls.index))
        close_0050 = pd.Series(100 * np.cumprod(1 + target_returns), index=cls.index)
        open_0050 = close_0050.shift(1).fillna(close_0050.iloc[0]) * (1 + rng.normal(0, .002, len(cls.index)))
        data = add_forward_returns(breadth, open_0050, close_0050)
        data = add_market_regime(data, close_0050, config=cls.cfg)
        cls.dataset = add_rolling_normalization(data, config=cls.cfg)
        cls.v6_results = run_signal_study(cls.dataset, config=cls.cfg)
        identified = attach_hypothesis_identity(cls.dataset, cls.v6_results, cls.cfg)
        cls.results = add_deduplicated_corrections(identified)

    def test_mask_hash_is_stable_and_context_sensitive(self):
        idx = self.index[:5]
        self.assertEqual(_stable_mask_hash(idx, "ALL"), _stable_mask_hash(idx, "ALL"))
        self.assertNotEqual(_stable_mask_hash(idx, "ALL"), _stable_mask_hash(idx, "BULL"))

    def test_aliases_and_mirrors_are_deduplicated_but_retained(self):
        subset = self.results[
            (self.results.predictor == "up_ratio")
            & (self.results.signal_method == "PR")
            & (self.results.market_regime == "ALL")
            & (self.results.target == "ret_o1_c1")
            & (self.results.group.isin(["HIGH_5", "PR_GE_95"]))
        ]
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset.canonical_hypothesis_id.nunique(), 1)
        self.assertEqual(int((~subset.is_duplicate_hypothesis).sum()), 1)
        self.assertEqual(subset.loc[~subset.is_duplicate_hypothesis, "group"].iloc[0], "PR_GE_95")
        self.assertTrue(subset["group_vs_non_group_FDR_global_v7"].notna().all())

    def test_target_or_regime_are_not_merged(self):
        subset = self.results[
            (self.results.predictor == "limit_up_ratio")
            & (self.results.signal_method == "PR")
            & (self.results.group == "PR_GE_80")
        ]
        ids = subset.groupby(["market_regime", "target"]).canonical_hypothesis_id.first()
        self.assertEqual(ids.nunique(), len(ids))

    def test_quintile_rank_and_hac_lags(self):
        trend = build_quintile_trend_results(self.dataset, self.cfg)
        self.assertFalse(trend.empty)
        for i in range(1, 6):
            self.assertTrue((trend[f"Q{i}_N"] > 0).all())
        expected = {"ret_c0_o1": 0, "ret_c0_c1": 0, "ret_o1_c1": 0, "ret_o1_o2": 0, "ret_o1_c2": 1, "ret_o1_c3": 2}
        self.assertTrue(trend.apply(lambda r: r.hac_lag == expected[r.target], axis=1).all())

    def test_yearly_and_leave_one_year_out_reconcile(self):
        summary, loyo, detail = build_yearly_stability(self.dataset, self.results, self.cfg)
        self.assertFalse(summary.empty)
        self.assertFalse(loyo.empty)
        self.assertFalse(detail.empty)
        totals = detail.groupby(["predictor", "signal_method", "group", "market_regime", "target"])["N"].sum()
        expected = summary.set_index(["predictor", "signal_method", "group", "market_regime", "target"])["total_signal_N"]
        pd.testing.assert_series_equal(totals.sort_index(), expected.sort_index(), check_names=False)
        self.assertTrue((loyo.N_group < loyo.groupby(["predictor", "signal_method", "group", "market_regime", "target"])["N_group"].transform("max") + 1000).all())
        self.assertTrue(loyo.direction_matches_full_sample.isin([True, False]).all())

    def test_pullback_has_no_lookahead_and_o2_uses_open_t2(self):
        validation, yearly = build_limit_up_pullback_validation(self.dataset, self.cfg)
        self.assertFalse(validation.empty)
        tradable = validation[validation.execution_status == "tradable_open_entry"]
        self.assertTrue(tradable.target.str.startswith("ret_o2_").all())
        self.assertTrue(tradable.direct_comparator_target.str.startswith("ret_o1_").all())
        self.assertTrue(tradable["same_events_direct_o1_mean"].notna().any())
        row = self.dataset.dropna(subset=["ret_o2_c3"]).iloc[100]
        pos = self.dataset.index.get_loc(row.name)
        expected = self.dataset.close_0050.iloc[pos + 3] / self.dataset.open_0050.iloc[pos + 2] - 1
        self.assertAlmostEqual(row.ret_o2_c3, expected)
        # Changing t+2 cannot alter whether t+1 is a pullback.
        original = self.dataset.ret_o1_c1.iloc[100]
        changed = self.dataset.copy()
        changed.iloc[102, changed.columns.get_loc("close_0050")] *= 2
        self.assertEqual(original, changed.ret_o1_c1.iloc[100])

    def test_non_overlapping_events(self):
        mask = pd.Series([True, True, True, False, True, True], index=pd.RangeIndex(6))
        selected = _non_overlapping_mask(mask, 2)
        self.assertEqual(selected[selected].index.tolist(), [0, 4])


if __name__ == "__main__":
    unittest.main()
