from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_breadth.config import V9Config, V9_TARGET_METADATA
from market_breadth.colab import validate_existing_clone
from market_breadth.core import add_forward_returns, build_market_breadth
from market_breadth.run_context import OFFICIAL_DRIVE_OUTPUT_ROOT, archive_run, create_run_context, validate_drive_root
from market_breadth.v9 import (
    add_v9_pr_features, attach_identity_and_corrections, build_v9_features,
    historical_percentile_rank, run_v9_threshold_study,
)


class V9FeatureTests(unittest.TestCase):
    def test_post_2015_default_and_hac_lags(self):
        cfg = V9Config()
        self.assertEqual(cfg.start_date, "2015-06-01")
        self.assertEqual([V9_TARGET_METADATA[x]["hac_lag"] for x in V9_TARGET_METADATA], [0, 2, 4, 9, 19])

    def test_pr_excludes_t_for_all_windows(self):
        values = pd.Series(np.arange(300, dtype=float))
        for window in (60, 126, 252):
            pr = historical_percentile_rank(values, window)
            self.assertTrue(pr.iloc[:window].isna().all())
            self.assertEqual(pr.iloc[window], 1.0)
        changed_current = values.copy(); changed_current.iloc[252] = -1
        self.assertEqual(historical_percentile_rank(changed_current, 252).iloc[252], 0.0)

    def test_averaging_and_delta(self):
        idx = pd.date_range("2015-06-01", periods=6)
        breadth = pd.DataFrame({
            "limit_up_ratio": range(6), "limit_down_ratio": range(6),
            "big_up_ratio": range(6), "big_down_ratio": range(6),
        }, index=idx, dtype=float)
        out, specs = build_v9_features(breadth, V9Config())
        self.assertEqual(out["limit_up_ratio__mean_1d"].iloc[3], 3)
        self.assertEqual(out["limit_up_ratio__mean_3d"].iloc[2], 1)
        self.assertEqual(out["limit_up_ratio__mean_5d"].iloc[4], 2)
        self.assertEqual(out["limit_up_ratio__mean_3d__delta"].iloc[3], 1)
        self.assertNotIn("acceleration", " ".join(specs))

    def test_forward_outcomes_adjusted_formula(self):
        idx = pd.date_range("2020-01-01", periods=25)
        o = pd.Series(np.arange(100, 125), index=idx, dtype=float)
        c = o + 1
        out = add_forward_returns(pd.DataFrame(index=idx), o, c)
        for horizon in (1, 3, 5, 10, 20):
            self.assertAlmostEqual(out.loc[idx[0], f"ret_o1_c{horizon}"], c.iloc[horizon]/o.iloc[1]-1)

    def test_strict_extreme_and_valid_denominator(self):
        idx = pd.date_range("2015-06-01", periods=2)
        close = pd.DataFrame([[100, 100, 100, np.nan], [105, 105.1, 94.9, 50]], index=idx, columns=list("ABCD"))
        breadth, _ = build_market_breadth(close, config=V9Config())
        self.assertEqual(breadth.big_up_count.iloc[1], 1)
        self.assertEqual(breadth.big_down_count.iloc[1], 1)
        self.assertEqual(breadth.valid_stock_count.iloc[1], 3)

    def test_threshold_study_win_rate_and_dedup(self):
        idx = pd.date_range("2020-01-01", periods=320)
        wave = np.sin(np.arange(len(idx)) / 7) + np.arange(len(idx)) / 1000
        raw = pd.DataFrame({k: wave for k in ("limit_up_ratio", "limit_down_ratio", "big_up_ratio", "big_down_ratio")}, index=idx)
        data, specs = build_v9_features(raw, V9Config())
        data = add_v9_pr_features(data, specs, V9Config())
        for target in V9_TARGET_METADATA:
            data[target] = np.where(np.arange(len(idx)) % 3, .01, -.01)
        results = run_v9_threshold_study(data, {"limit_up_ratio__mean_1d": specs["limit_up_ratio__mean_1d"]}, V9Config())
        results = attach_identity_and_corrections(results, data)
        self.assertTrue(np.allclose(results.signal_win_rate, results.successes/results.N))
        self.assertTrue(results.canonical_hypothesis_id.notna().all())
        self.assertTrue(results.filter(regex="FDR").apply(lambda s: s.dropna().between(0, 1).all()).all())
        self.assertTrue((results.successes + results.failures).eq(results.N).all())
        self.assertTrue(results[["avg_win", "avg_loss", "payoff_ratio", "expectancy"]].notna().any().all())

    def test_non_overlapping_rows_are_thinned(self):
        idx = pd.date_range("2020-01-01", periods=400)
        wave = np.sin(np.arange(len(idx)) / 5)
        raw = pd.DataFrame({k: wave for k in ("limit_up_ratio", "limit_down_ratio", "big_up_ratio", "big_down_ratio")}, index=idx)
        data, specs = build_v9_features(raw, V9Config())
        data = add_v9_pr_features(data, specs, V9Config())
        for target in V9_TARGET_METADATA: data[target] = np.where(np.arange(len(idx)) % 2, .01, -.01)
        results = run_v9_threshold_study(data, {"limit_up_ratio__mean_1d": specs["limit_up_ratio__mean_1d"]}, V9Config())
        all_n = results.query("target == 'ret_o1_c20' and overlap_policy == 'all_events'").N.max()
        non_n = results.query("target == 'ret_o1_c20' and overlap_policy == 'non_overlapping_events'").N.max()
        self.assertLess(non_n, all_n)


class V9ArchiveTests(unittest.TestCase):
    def test_official_drive_root_constant(self):
        self.assertEqual(str(OFFICIAL_DRIVE_OUTPUT_ROOT), "/content/drive/MyDrive/Quant_Research/taiwan-market-breadth-research")

    def test_colab_clone_path_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            unexpected = Path(tmp) / "repo"; unexpected.mkdir()
            with self.assertRaises(RuntimeError): validate_existing_clone(unexpected)

    def test_versioned_run_id_and_no_overwrite_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("market_breadth.run_context.project_root", return_value=root), \
                 patch("market_breadth.run_context.git_value", side_effect=["abcdef1234567890", "feature/v9"]):
                ctx = create_run_context("v9_test", root/"out", datetime(2026, 9, 15, 9, 15, 30, tzinfo=ZoneInfo("Asia/Taipei")))
            self.assertEqual(ctx.run_id, "20260915_091530_v9_test_abcdef123456")
            for name in ("market_breadth_summary.xlsx", "daily_dataset.parquet", "run_info.txt", "validation_summary.md"):
                (ctx.local_run_dir/name).write_bytes(b"x")
            (ctx.local_run_dir/"plots").mkdir(); (ctx.local_run_dir/"plots"/"p.png").write_bytes(b"x")
            drive = root/"drive"; drive.mkdir()
            archive = archive_run(ctx.local_run_dir, drive, ctx.run_id)
            self.assertTrue(archive.exists())
            with self.assertRaises(FileExistsError):
                archive_run(ctx.local_run_dir, drive, ctx.run_id)


if __name__ == "__main__":
    unittest.main()
