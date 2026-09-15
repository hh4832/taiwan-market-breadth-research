from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import V10_RAW_PREDICTORS, V11Config
from .core import add_forward_returns, build_market_breadth
from .data import FINLAB_KEYS, build_reference_price_matrix, filter_common_stocks, limit_date, load_first_available
from .export_v11 import build_v11_metadata, export_v11
from .pipeline import _symbol
from .plots_v11 import make_v11_plots
from .run_context import RunContext, create_run_context
from .v10 import add_v10_pr_features, build_v10_features
from .v11 import (CANDIDATES, add_prior_returns, build_pr_threshold_mapping,
                  build_prior_return_descriptive, classify_mechanism,
                  run_prior_return_regressions, run_regime_comparison)
from .validation_v11 import validate_v11


def _stage(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _cache_path(symbols: list[str], config: V11Config) -> Path:
    payload = json.dumps({"version": config.version_slug, "start": config.start_date,
                          "universe": hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest(),
                          "limit_logic": "historical_7pct_10pct_corp_action_tick_v1"}, sort_keys=True)
    return config.cache_dir / f"breadth_{config.version_slug}_{hashlib.sha256(payload.encode()).hexdigest()[:16]}.parquet"


def run_v11(config: V11Config | None = None, context: RunContext | None = None,
            drive_output_root: Path | None = None) -> dict[str, object]:
    cfg = config or V11Config(); ctx = context or create_run_context(cfg.version_slug, cfg.output_dir)
    cfg.cache_dir.mkdir(parents=True, exist_ok=True); selected: dict[str, str | None] = {}
    _stage("Load data")
    stock_close, selected["stock_close"] = load_first_available("stock_close", FINLAB_KEYS["stock_close"], cfg.cache_dir, cfg.refresh)
    metadata_raw, selected["metadata"] = load_first_available("metadata", FINLAB_KEYS["metadata"], cfg.cache_dir, cfg.refresh, normalize_index=False)
    stock_close = limit_date(stock_close, cfg); common_close, selected_metadata, metadata_audit = filter_common_stocks(stock_close, metadata_raw)
    reference, event_sources = build_reference_price_matrix(common_close, cfg.cache_dir, refresh=cfg.refresh); selected["event_reference_sources"] = str(event_sources)
    cache = _cache_path(common_close.columns.astype(str).tolist(), cfg); required = set(V10_RAW_PREDICTORS) | {"valid_stock_count"}
    breadth_meta = {"cache_reused": False}
    if cache.exists() and not cfg.refresh:
        breadth = pd.read_parquet(cache)
        if not required.issubset(breadth.columns) or breadth.index.min() > pd.Timestamp(cfg.start_date):
            breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg); breadth.to_parquet(cache)
        else: breadth_meta.update({"cache_reused": True, "limit_status_is_approximation": False})
    else:
        breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference, config=cfg); cache.parent.mkdir(parents=True, exist_ok=True); breadth.to_parquet(cache)
    adj_open, selected["adj_open"] = load_first_available("adj_open", FINLAB_KEYS["adj_open"], cfg.cache_dir, cfg.refresh)
    adj_close, selected["adj_close"] = load_first_available("adj_close", FINLAB_KEYS["adj_close"], cfg.cache_dir, cfg.refresh)
    if (selected["adj_open"], selected["adj_close"]) != ("etl:adj_open", "etl:adj_close"): raise AssertionError("v11 requires adjusted 0050")
    open_0050 = limit_date(_symbol(adj_open, cfg.target_symbol, "open_0050"), cfg); close_0050 = limit_date(_symbol(adj_close, cfg.target_symbol, "close_0050"), cfg)

    _stage("Build v10 baseline features")
    dataset = add_forward_returns(breadth, open_0050, close_0050).loc[cfg.start_date:cfg.end_date]
    dataset, specs = build_v10_features(dataset, cfg); dataset = add_v10_pr_features(dataset, specs, cfg)
    dataset = add_prior_returns(dataset, close_0050, cfg.prior_return_windows)

    _stage("Prior return analysis")
    prior = build_prior_return_descriptive(dataset, cfg)
    _stage("Candidate A regression")
    regression_a = run_prior_return_regressions(dataset, [c for c in CANDIDATES if c["candidate"] == "A"])
    _stage("Candidate B regression")
    regression_b = run_prior_return_regressions(dataset, [c for c in CANDIDATES if c["candidate"] == "B"])
    _stage("Candidate C win-rate regression")
    regression_c = run_prior_return_regressions(dataset, [c for c in CANDIDATES if c.get("candidate_group") == "C"])
    regressions = pd.concat([regression_a, regression_b, regression_c], ignore_index=True)

    _stage("PR raw threshold mapping")
    threshold, signal_raw, threshold_yearly, threshold_daily = build_pr_threshold_mapping(dataset)
    _stage("Pre/post regime split")
    _stage("Interaction models")
    regime, interactions, regime_yearly = run_regime_comparison(dataset, cfg)
    candidates = classify_mechanism(regressions, interactions, regime)
    validations = validate_v11(dataset, prior, regressions, threshold_daily, regime, interactions, cfg, selected)
    metadata = build_v11_metadata(cfg, ctx, dataset, selected, breadth_meta, drive_output_root)

    _stage("Export")
    paths = export_v11(dataset, prior, regressions, threshold, signal_raw, threshold_yearly, regime,
                       interactions, regime_yearly, candidates, metadata, validations, ctx)
    plots = make_v11_plots(prior, regressions, threshold_yearly, regime, interactions, paths["plots"])
    plots.to_csv(paths["plots"] / "manifest.csv", index=False)
    return {"dataset": dataset, "prior_return": prior, "regressions": regressions,
            "threshold_mapping": threshold, "threshold_daily": threshold_daily,
            "signal_raw": signal_raw, "threshold_yearly": threshold_yearly,
            "regime": regime, "interactions": interactions, "regime_yearly": regime_yearly,
            "candidate_summary": candidates, "validations": validations, "metadata": metadata,
            "output_paths": paths, "plot_manifest": plots, "run_context": ctx,
            "selected_metadata": selected_metadata, "metadata_audit": metadata_audit}
