from __future__ import annotations

import warnings

import pandas as pd

from .config import V6Config, V7Config
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
from .robustness import (
    add_deduplicated_corrections,
    attach_hypothesis_identity,
    build_fdr_comparison,
    build_limit_up_pullback_validation,
    build_quintile_trend_results,
    build_yearly_stability,
)


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
    v7_outputs: dict[str, pd.DataFrame] = {}
    if isinstance(cfg, V7Config):
        results = attach_hypothesis_identity(dataset, results, cfg)
        results = add_deduplicated_corrections(results)
        fdr_comparison = build_fdr_comparison(results)
        trend = build_quintile_trend_results(dataset, cfg)
        yearly_stability, leave_one_year_out, yearly_stability_detail = build_yearly_stability(dataset, results, cfg)
        pullback, pullback_yearly = build_limit_up_pullback_validation(dataset, cfg)
        duplicate_map = results.loc[results["duplicate_group_size"].gt(1), [
            "canonical_hypothesis_id", "signal_mask_hash", "predictor", "canonical_predictor",
            "signal_method", "group", "group_type", "market_regime", "target",
            "is_duplicate_hypothesis", "duplicate_group_size", "duplicate_reason",
        ]]
        profile_focus = trend.loc[
            trend["predictor"].isin(["limit_up_ratio", "down_ratio", "delta_down_ratio_1d", "limit_down_ratio", "big_down_ratio"])
        ]
        validation_summary = _build_validation_summary(results, trend, yearly_stability, pullback)
        v7_outputs = {
            "deduplicated_hypotheses": results.loc[~results["is_duplicate_hypothesis"]],
            "fdr_comparison_v6_v7": fdr_comparison,
            "duplicate_hypothesis_map": duplicate_map,
            "yearly_stability_summary": yearly_stability,
            "yearly_stability_detail": yearly_stability_detail,
            "leave_one_year_out": leave_one_year_out,
            "quintile_trend_results": trend,
            "candidate_quintile_profiles": profile_focus,
            "limit_up_pullback_validation": pullback,
            "limit_up_pullback_yearly": pullback_yearly,
            "validation_summary": validation_summary,
        }
    validations = validate_v6(common_close, breadth, dataset, results, config=cfg)
    metadata = build_metadata(cfg, dataset, breadth_meta, selected_keys)
    paths = export_results(dataset, results, monotonicity, yearly, metadata, validations, cfg, v7_outputs=v7_outputs)
    figure_manifest = make_core_plots(dataset, results, paths["plots"], cfg)
    return {
        "dataset": dataset, "results": results, "monotonicity": monotonicity,
        "yearly": yearly, "validations": validations, "metadata": metadata,
        "selected_metadata": selected_metadata, "metadata_audit": metadata_audit,
        "figure_manifest": figure_manifest, "output_paths": paths, "v7_outputs": v7_outputs,
    }


def _build_validation_summary(
    results: pd.DataFrame, trend: pd.DataFrame, yearly: pd.DataFrame, pullback: pd.DataFrame,
) -> pd.DataFrame:
    global_cols = [c for c in results if c.endswith("FDR_global_v7")]
    family_cols = [c for c in results if c.endswith("FDR_family_v7")]
    canonical = ~results["is_duplicate_hypothesis"]
    global_count = int(results.loc[canonical, global_cols].lt(.05).any(axis=1).sum())
    family_count = int(results.loc[canonical, family_cols].lt(.05).any(axis=1).sum())
    stable = int((yearly["is_prespecified_candidate"] & yearly["direction_consistency_rate"].ge(.6)).sum()) if not yearly.empty else 0
    trend_count = int((~trend["is_duplicate_hypothesis"] & trend["trend_FDR_global"].lt(.05)).sum())
    tradable = pullback.loc[(pullback["execution_status"] == "tradable_open_entry") & (pullback["overlap_policy"] == "non_overlapping")]
    improved = int((tradable["mean_minus_direct_same_events"].gt(0) & tradable["wait_vs_direct_HAC_p"].lt(.05)).sum()) if not tradable.empty else 0
    rows = [
        ("去重後是否有任何 global FDR < 0.05？", f"共有 {global_count} 個 canonical signal 通過任一 v7 global FDR。"),
        ("哪些訊號通過 family FDR？", f"共有 {family_count} 個 canonical signal 通過任一 v7 family FDR；詳見 deduplicated_hypotheses。"),
        ("哪些訊號跨年度方向穩定？", f"預先指定候選中有 {stable} 個方向一致率至少 60%；詳見 yearly_stability_summary。"),
        ("哪些五分位呈現顯著單調趨勢？", f"共有 {trend_count} 個去重後趨勢通過 global FDR 5%。"),
        ("漲停後等待回檔是否優於直接 O1 追價？", f"可交易、非重疊回檔組合中有 {improved} 個同事件比較呈正差且 HAC p<0.05；仍需對照完整表格判讀。"),
        ("扣除重複與重疊事件後，候選是否仍存在？", "以 non_overlapping 列及 v7 校正欄位為準，不以 raw event 顯著性代替。"),
        ("最終保留、降級與淘汰哪些訊號？", "此檔只提供可重現統計證據，不自動宣稱訊號可交易；由研究者依機制、穩定性與成本決策。"),
    ]
    return pd.DataFrame(rows, columns=["question", "answer"])
