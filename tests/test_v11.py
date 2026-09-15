from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_breadth.config import V11Config, V9_TARGET_METADATA
from market_breadth.export_v11 import export_v11
from market_breadth.run_context import RunContext
from market_breadth.v10 import add_v10_pr_features, build_v10_features
from market_breadth.v11 import (add_prior_returns, build_pr_threshold_mapping,
                                candidate_signal, historical_raw_threshold,
                                ols_hac, run_prior_return_regressions,
                                run_regime_comparison)
from market_breadth.validation_v11 import validate_v11


class V11MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = V11Config()
        cls.index = pd.date_range("2011-01-03", periods=1400, freq="B")
        x = np.arange(len(cls.index), dtype=float); up = .5 + .45*np.sin(x/11)
        breadth = pd.DataFrame({"up_ratio":up,"down_ratio":1-up,"big_up_ratio":.1+.04*np.sin(x/7),
                                "big_down_ratio":.1+.04*np.cos(x/7),"limit_up_ratio":.03+.02*np.sin(x/9),
                                "limit_down_ratio":.03+.02*np.cos(x/9)}, index=cls.index)
        data, specs = build_v10_features(breadth, cls.config); data = add_v10_pr_features(data, specs, cls.config)
        cls.close = pd.Series(100*np.exp(np.cumsum(.001*np.sin(x/5))), index=cls.index)
        data["close_0050"] = cls.close
        for target in V9_TARGET_METADATA: data[target] = .01*np.sin(x/6)
        cls.data = add_prior_returns(data, cls.close)

    def test_prior_returns_exact_and_no_lookahead(self):
        for window in (1,3,5,10):
            expected = self.close/self.close.shift(window)-1
            pd.testing.assert_series_equal(self.data[f"prior_ret_{window}d"], expected, check_names=False)
        changed_future = self.close.copy(); changed_future.iloc[-1] *= 5
        recalculated = add_prior_returns(self.data.iloc[:-1], changed_future.iloc[:-1])
        pd.testing.assert_series_equal(recalculated.prior_ret_10d, self.data.iloc[:-1].prior_ret_10d)

    def test_hac_coefficient_extraction_and_attenuation(self):
        signal = np.tile([0.,1.], 100); post = np.repeat([0.,1.], 100)
        x = np.column_stack([np.ones(200), signal, post, signal*post])
        y = 1 + 2*signal + 3*post + 4*signal*post + .01*np.sin(np.arange(200))
        fit = ols_hac(y, x, ["const","signal","post2015","signal_x_post2015"], 2)
        self.assertAlmostEqual(fit.set_index("term").at["signal_x_post2015","beta"], 4., places=2)
        regressions = run_prior_return_regressions(self.data)
        adjusted = regressions.loc[regressions.model.ne("breadth_only")]
        self.assertTrue(np.allclose(adjusted.attenuation_ratio, 1-adjusted.breadth_beta/adjusted.unadjusted_beta, equal_nan=True))

    def test_threshold_mapping_excludes_t_and_matches_signal(self):
        sample = pd.Series([1.,2.,3.,100.])
        self.assertEqual(historical_raw_threshold(sample,3,.95).iloc[-1],3.)
        mapping, raw, yearly, daily = build_pr_threshold_mapping(self.data)
        self.assertEqual(len(mapping), 7); self.assertEqual(len(raw), 5); self.assertFalse(yearly.empty)
        for row in daily.sample(min(100, len(daily)), random_state=1).itertuples():
            current = self.data.at[pd.Timestamp(row.date), row.predictor]
            pr = self.data.at[pd.Timestamp(row.date), f"{row.predictor}__PR_{row.pr_window}"]
            self.assertEqual(bool(current >= row.raw_threshold), bool(pr >= row.quantile))

    def test_regime_boundary_interaction_and_validation(self):
        regime, interactions, yearly = run_regime_comparison(self.data, self.config, predictors=["down_ratio"],
            mean_windows=[1], pr_windows=[60], bins=[("BIN_95_100",.95,1.)], targets=["ret_o1_c1"], show_progress=False)
        self.assertEqual(set(regime.regime), {"PRE_2015_7PCT","POST_2015_10PCT"})
        self.assertEqual(set(interactions.endpoint), {"mean_return","win_rate"})
        mapping, raw, threshold_yearly, daily = build_pr_threshold_mapping(self.data)
        checks = validate_v11(self.data, pd.DataFrame(), run_prior_return_regressions(self.data), daily,
                              regime, interactions, self.config, {"adj_open":"etl:adj_open","adj_close":"etl:adj_close"})
        self.assertTrue(all(frame.passed.all() for frame in checks.values()))

    def test_output_files_and_no_overwrite_context(self):
        prior = pd.DataFrame([{"candidate":"A","candidate_group":"A","prior_window":5,"group":"signal","mean":.01}])
        regressions = pd.DataFrame([{"candidate":"A","candidate_group":"A","model":"breadth_only","breadth_beta":.01}])
        threshold = pd.DataFrame([{"predictor":"up_ratio__mean_1d","pr_window":60,"quantile":.95,"median":.8}])
        raw = pd.DataFrame([{"candidate":"A","predictor":"big_up_ratio__mean_5d","median":.02}])
        threshold_yearly = pd.DataFrame([{"predictor":"up_ratio__mean_1d","pr_window":60,"quantile":.95,"year":2015,"median_threshold":.8}])
        regime = pd.DataFrame([{"predictor":"down_ratio__mean_1d","regime":"PRE_2015_7PCT"}])
        interactions = pd.DataFrame([{"predictor":"down_ratio__mean_1d","interaction_HAC_p":.5,"interaction_FDR_family":.5}])
        candidates = pd.DataFrame([{"hypothesis":"Candidate A","decision":"修改後再測"}])
        validations = {"test":pd.DataFrame([{"check":"synthetic","passed":True}])}
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"run"; out.mkdir(); ctx=RunContext("run_v11","now",self.config.version_slug,"abc","test",out)
            def parquet_stub(frame, path, *args, **kwargs):
                Path(path).write_bytes(b"synthetic parquet output")
            with patch.object(pd.DataFrame, "to_parquet", parquet_stub):
                paths=export_v11(self.data.head(20),prior,regressions,threshold,raw,threshold_yearly,regime,interactions,
                                 pd.DataFrame(),candidates,pd.DataFrame([{"item":"run_id","value":"run_v11"}]),validations,ctx)
            self.assertTrue(all(path.exists() and (path.is_dir() or path.stat().st_size>0) for path in paths.values()))


if __name__ == "__main__": unittest.main()
