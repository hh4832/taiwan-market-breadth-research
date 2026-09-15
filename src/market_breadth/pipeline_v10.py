from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import V10Config, V10_RAW_PREDICTORS
from .core import add_forward_returns, build_market_breadth
from .data import FINLAB_KEYS, build_reference_price_matrix, filter_common_stocks, limit_date, load_first_available
from .export_v10 import build_v10_metadata, export_v10
from .pipeline import _symbol
from .plots_v10 import make_v10_plots
from .run_context import RunContext, create_run_context
from .v10 import (PR_BINS, add_v10_pr_features, attach_identity_and_corrections,
                  build_shape_analysis, build_v10_features, build_yearly_results,
                  run_v10_bin_study)
from .validation_v10 import validate_v10


def _cache_path(symbols: list[str], config: V10Config) -> Path:
    payload = json.dumps({"version": config.version_slug, "sample_start": config.start_date,
                          "universe": hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest(),
                          "predictors": V10_RAW_PREDICTORS, "limit_logic": "corp_action_tick_10pct_v1"}, sort_keys=True)
    return config.cache_dir / f"breadth_{config.version_slug}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}.parquet"


def build_signal_definitions(config: V10Config) -> pd.DataFrame:
    rows = []
    for raw, (family, direction) in V10_RAW_PREDICTORS.items():
        for mean_window in config.signal_mean_windows:
            for transform in ("level", "delta"):
                rows.append({"raw_predictor": raw, "predictor_family": family, "direction": direction,
                             "mean_window": mean_window, "transform": transform,
                             "formula": f"rolling({mean_window}, min_periods={mean_window}).mean({raw})" + (" - lag1" if transform == "delta" else ""),
                             "PR_windows": str(config.pr_windows), "PR_reference": "t-1 and earlier only",
                             "PR_bins": ",".join(name for name, _, _ in PR_BINS), "signal_availability": "after t close"})
    return pd.DataFrame(rows)


def run_v10(
    config: V10Config | None = None, context: RunContext | None = None,
    drive_output_root: Path | None = None,
) -> dict[str, object]:
    cfg = config or V10Config(); ctx = context or create_run_context(cfg.version_slug, cfg.output_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, str | None] = {}
    stock_close, selected["stock_close"] = load_first_available("stock_close", FINLAB_KEYS["stock_close"], cfg.cache_dir, cfg.refresh)
    metadata_raw, selected["metadata"] = load_first_available("metadata", FINLAB_KEYS["metadata"], cfg.cache_dir, cfg.refresh, normalize_index=False)
    stock_close = limit_date(stock_close, cfg)
    common_close, selected_metadata, metadata_audit = filter_common_stocks(stock_close, metadata_raw)
    reference, event_sources = build_reference_price_matrix(common_close, cfg.cache_dir, refresh=cfg.refresh)
    selected["event_reference_sources"] = str(event_sources)
    cache = _cache_path(common_close.columns.astype(str).tolist(), cfg)
    required = set(V10_RAW_PREDICTORS) | {"valid_stock_count", "up_count", "down_count", "flat_count"}
    breadth_meta = {"cache_reused": False}
    if cache.exists() and not cfg.refresh:
        breadth = pd.read_parquet(cache)
        if not required.issubset(breadth.columns) or breadth.index.min() < pd.Timestamp(cfg.start_date):
            breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg); breadth.to_parquet(cache)
        else:
            breadth_meta.update({"cache_reused": True, "limit_status_is_approximation": False})
    else:
        breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg)
        cache.parent.mkdir(parents=True, exist_ok=True); breadth.to_parquet(cache)
    adj_open, selected["adj_open"] = load_first_available("adj_open", FINLAB_KEYS["adj_open"], cfg.cache_dir, cfg.refresh)
    adj_close, selected["adj_close"] = load_first_available("adj_close", FINLAB_KEYS["adj_close"], cfg.cache_dir, cfg.refresh)
    if (selected["adj_open"], selected["adj_close"]) != ("etl:adj_open", "etl:adj_close"):
        raise AssertionError("v10 requires etl:adj_open and etl:adj_close")
    open_0050 = limit_date(_symbol(adj_open, cfg.target_symbol, "open_0050"), cfg)
    close_0050 = limit_date(_symbol(adj_close, cfg.target_symbol, "close_0050"), cfg)
    dataset = add_forward_returns(breadth, open_0050, close_0050).loc[cfg.start_date:cfg.end_date]
    dataset, specs = build_v10_features(dataset, cfg); dataset = add_v10_pr_features(dataset, specs, cfg)
    results = attach_identity_and_corrections(run_v10_bin_study(dataset, specs, cfg), dataset)
    shape = build_shape_analysis(results); yearly = build_yearly_results(results, dataset)
    validations = validate_v10(dataset, results, cfg, selected)
    metadata = build_v10_metadata(cfg, ctx, dataset, selected, breadth_meta, drive_output_root)
    definitions = build_signal_definitions(cfg)
    paths = export_v10(dataset, results, shape, yearly, definitions, metadata, validations, ctx)
    plot_manifest = make_v10_plots(results, yearly, paths["plots"])
    plot_manifest.to_csv(paths["plots"] / "manifest.csv", index=False)
    return {"dataset": dataset, "results": results, "pr_bin_shape": shape, "yearly": yearly,
            "validations": validations, "metadata": metadata, "output_paths": paths,
            "run_context": ctx, "plot_manifest": plot_manifest,
            "selected_metadata": selected_metadata, "metadata_audit": metadata_audit}
