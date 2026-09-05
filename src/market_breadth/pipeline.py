from __future__ import annotations

import warnings

import pandas as pd

from .config import V6Config
from .core import add_forward_returns, add_market_regime, add_rolling_normalization, build_market_breadth
from .data import (
    FINLAB_KEYS,
    breadth_cache_path,
    build_reference_price_matrix,
    filter_common_stocks,
    limit_date,
    load_first_available,
    validate_breadth_cache,
    write_cache_metadata,
)
from .export import build_metadata, export_results
from .plots import make_core_plots
from .statistics import run_signal_study
from .summaries import build_monotonicity_results, build_yearly_results
from .validation import validate_v6


def _symbol(frame: pd.DataFrame, symbol: str, label: str) -> pd.Series:
    mapping = {str(c): c for c in frame.columns}
    if symbol not in mapping:
        raise ValueError(f"{label} 找不到 {symbol}; columns sample={list(frame.columns[:20])}")
    out = pd.to_numeric(frame[mapping[symbol]], errors="coerce")
    out.name = label
    return out


def run(config: V6Config | None = None) -> dict[str, object]:
    cfg = config or V6Config()
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    selected_keys: dict[str, str | None] = {}

    stock_close, selected_keys["stock_close"] = load_first_available(
        "stock_close", FINLAB_KEYS["stock_close"], cfg.cache_dir, cfg.refresh
    )
    metadata_raw, selected_keys["metadata"] = load_first_available(
        "metadata", FINLAB_KEYS["metadata"], cfg.cache_dir, cfg.refresh, normalize_index=False
    )
    stock_close = limit_date(stock_close, cfg)
    common_close, selected_metadata, metadata_audit = filter_common_stocks(stock_close, metadata_raw)
    symbols = common_close.columns.astype(str).tolist()

    reference_price, event_sources = build_reference_price_matrix(
        common_close, cfg.cache_dir, refresh=cfg.refresh
    )
    selected_keys["event_reference_sources"] = str(event_sources)

    cache = breadth_cache_path(symbols, cfg)
    if cache.exists() and not cfg.refresh:
        breadth = pd.read_parquet(cache)
        if not validate_breadth_cache(breadth):
            warnings.warn("v6 breadth cache 欄位不足，自動重建", UserWarning)
            breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference_price, config=cfg)
        else:
            breadth_meta = {"cache_reused": True, "limit_status_is_approximation": False, "limit_detection_method": "reference_price_tick_rounding"}
    else:
        breadth, breadth_meta = build_market_breadth(common_close, reference_price=reference_price, config=cfg)
    if not cache.exists() or cfg.refresh or not validate_breadth_cache(pd.read_parquet(cache)):
        breadth.to_parquet(cache)
        write_cache_metadata(cache, symbols, cfg)

    adj_open, selected_keys["adj_open"] = load_first_available("adj_open", FINLAB_KEYS["adj_open"], cfg.cache_dir, cfg.refresh, required=False)
    adj_close, selected_keys["adj_close"] = load_first_available("adj_close", FINLAB_KEYS["adj_close"], cfg.cache_dir, cfg.refresh, required=False)
    if adj_open is None or adj_close is None:
        adj_open, selected_keys["open"] = load_first_available("open", FINLAB_KEYS["open"], cfg.cache_dir, cfg.refresh)
        adj_close, selected_keys["close"] = load_first_available("close", FINLAB_KEYS["close"], cfg.cache_dir, cfg.refresh)
        warnings.warn("0050 使用未還原價格，除權息可能影響跨日報酬", UserWarning)
    open_0050 = limit_date(_symbol(adj_open, cfg.target_symbol, "open_0050"), cfg)
    close_0050 = limit_date(_symbol(adj_close, cfg.target_symbol, "close_0050"), cfg)

    dataset = add_forward_returns(breadth, open_0050, close_0050)
    dataset = add_market_regime(dataset, close_0050, config=cfg)
    dataset = add_rolling_normalization(dataset, config=cfg)
    results = run_signal_study(dataset, config=cfg)
    monotonicity = build_monotonicity_results(dataset, cfg)
    yearly = build_yearly_results(dataset, results, cfg)
    validations = validate_v6(common_close, breadth, dataset, results, config=cfg)
    metadata = build_metadata(cfg, dataset, breadth_meta, selected_keys)
    paths = export_results(dataset, results, monotonicity, yearly, metadata, validations, cfg)
    figure_manifest = make_core_plots(dataset, results, paths["plots"], cfg)
    return {
        "dataset": dataset, "results": results, "monotonicity": monotonicity,
        "yearly": yearly, "validations": validations, "metadata": metadata,
        "selected_metadata": selected_metadata, "metadata_audit": metadata_audit,
        "figure_manifest": figure_manifest, "output_paths": paths,
    }
