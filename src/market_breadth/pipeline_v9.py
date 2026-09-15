from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import V9Config, V9_RAW_PREDICTORS
from .core import add_forward_returns, build_market_breadth
from .data import FINLAB_KEYS, build_reference_price_matrix, filter_common_stocks, limit_date, load_first_available
from .export_v9 import build_v9_metadata, export_v9
from .pipeline import _symbol
from .plots_v9 import make_v9_plots
from .run_context import RunContext, create_run_context
from .v9 import add_v9_pr_features, attach_identity_and_corrections, build_pr_bins, build_v9_features, build_v9_yearly, run_v9_threshold_study
from .validation_v9 import validate_v9


def _cache_path(symbols: list[str], config: V9Config) -> Path:
    payload = json.dumps({
        "version": config.version_slug, "sample_start": config.start_date,
        "universe": hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest(),
        "predictors": V9_RAW_PREDICTORS, "mean_windows": config.signal_mean_windows,
        "pr_windows": config.pr_windows, "thresholds": config.pr_thresholds,
        "adjusted_price_logic": "etl_adj_open_close_v1", "limit_logic": "corp_action_tick_10pct_v1",
    }, sort_keys=True)
    return config.cache_dir / f"breadth_{config.version_slug}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}.parquet"


def build_signal_definitions(config: V9Config) -> pd.DataFrame:
    rows = []
    for predictor, (family, direction) in V9_RAW_PREDICTORS.items():
        for mean_window in config.signal_mean_windows:
            for transform in ("raw", "delta"):
                rows.append({
                    "raw_predictor": predictor, "predictor_family": family if transform == "raw" else "DELTA",
                    "direction": direction, "mean_window": mean_window, "transform": transform,
                    "formula": f"mean_{mean_window}d({predictor})" + (" - lag1" if transform == "delta" else ""),
                    "PR_windows": str(config.pr_windows), "PR_reference": "t-1 and earlier only",
                    "signal_availability": "after t close",
                })
    return pd.DataFrame(rows)


def run_v9(
    config: V9Config | None = None, context: RunContext | None = None,
    drive_output_root: Path | None = None,
) -> dict[str, object]:
    cfg = config or V9Config()
    ctx = context or create_run_context(cfg.version_slug, cfg.output_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, str | None] = {}
    stock_close, selected["stock_close"] = load_first_available("stock_close", FINLAB_KEYS["stock_close"], cfg.cache_dir, cfg.refresh)
    metadata_raw, selected["metadata"] = load_first_available("metadata", FINLAB_KEYS["metadata"], cfg.cache_dir, cfg.refresh, normalize_index=False)
    stock_close = limit_date(stock_close, cfg)
    common_close, selected_metadata, metadata_audit = filter_common_stocks(stock_close, metadata_raw)
    reference, event_sources = build_reference_price_matrix(common_close, cfg.cache_dir, refresh=cfg.refresh)
    selected["event_reference_sources"] = str(event_sources)
    cache = _cache_path(common_close.columns.astype(str).tolist(), cfg)
    required = set(V9_RAW_PREDICTORS) | {"valid_stock_count", "up_count", "down_count", "flat_count"}
    breadth_meta = {"cache_reused": False}
    if cache.exists() and not cfg.refresh:
        breadth = pd.read_parquet(cache)
        if not required.issubset(breadth.columns) or breadth.index.min() < pd.Timestamp(cfg.start_date):
            breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg)
            breadth.to_parquet(cache)
        else:
            breadth_meta.update({"cache_reused": True, "limit_status_is_approximation": False})
    else:
        breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg)
        cache.parent.mkdir(parents=True, exist_ok=True); breadth.to_parquet(cache)

    adj_open, selected["adj_open"] = load_first_available("adj_open", FINLAB_KEYS["adj_open"], cfg.cache_dir, cfg.refresh)
    adj_close, selected["adj_close"] = load_first_available("adj_close", FINLAB_KEYS["adj_close"], cfg.cache_dir, cfg.refresh)
    if selected["adj_open"] != "etl:adj_open" or selected["adj_close"] != "etl:adj_close":
        raise AssertionError("v9 requires FinLab etl:adj_open and etl:adj_close for all 0050 outcomes")
    open_0050 = limit_date(_symbol(adj_open, cfg.target_symbol, "open_0050"), cfg)
    close_0050 = limit_date(_symbol(adj_close, cfg.target_symbol, "close_0050"), cfg)
    dataset = add_forward_returns(breadth, open_0050, close_0050).loc[cfg.start_date:cfg.end_date]
    dataset, specs = build_v9_features(dataset, cfg)
    dataset = add_v9_pr_features(dataset, specs, cfg)
    results = run_v9_threshold_study(dataset, specs, cfg)
    results = attach_identity_and_corrections(results, dataset)
    bins = build_pr_bins(dataset, specs, cfg)
    yearly = build_v9_yearly(results, dataset)
    validations = validate_v9(dataset, results, cfg, selected)
    metadata = build_v9_metadata(cfg, ctx, dataset, selected, breadth_meta, drive_output_root)
    definitions = build_signal_definitions(cfg)
    paths = export_v9(dataset, results, bins, yearly, definitions, metadata, validations, ctx)
    plot_manifest = make_v9_plots(results, bins, yearly, paths["plots"])
    plot_manifest.to_csv(ctx.local_run_dir / "plots" / "manifest.csv", index=False)
    return {
        "dataset": dataset, "results": results, "pr_bins": bins, "yearly": yearly,
        "validations": validations, "metadata": metadata, "output_paths": paths,
        "run_context": ctx, "plot_manifest": plot_manifest,
        "selected_metadata": selected_metadata, "metadata_audit": metadata_audit,
    }
