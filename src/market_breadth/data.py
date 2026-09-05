from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from .config import PREDICTOR_SPECS, V6Config
from .core import normalize_datetime_index


FINLAB_KEYS = {
    "stock_close": ["price:收盤價"],
    "metadata": ["company_basic_info", "security_categories", "stock_basic_info"],
    "adj_open": ["etl:adj_open", "price:還原開盤價", "price:調整開盤價"],
    "adj_close": ["etl:adj_close", "price:還原收盤價", "price:調整收盤價"],
    "open": ["price:開盤價"],
    "close": ["price:收盤價"],
}

EVENT_REFERENCE_KEYS = [
    "dividend_tse:除權息參考價",
    "dividend_otc:除權息參考價",
    "capital_reduction_tse:恢復買賣參考價",
    "capital_reduction_otc:減資恢復買賣開始日參考價格",
    "par_value_change_tse:恢復買賣參考價",
    "par_value_change_otc:恢復買賣開始日參考價",
]


def _cache_name(label: str, key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"{label}_{digest}.parquet"


def load_or_download(label: str, key: str, cache_dir: Path, refresh: bool = False, normalize_index: bool = True):
    path = cache_dir / _cache_name(label, key)
    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
    else:
        try:
            from finlab import data
        except ImportError as exc:
            raise ImportError("缺少 finlab；請依 requirements.txt 安裝並完成 FinLab 登入。") from exc
        frame = data.get(key)
        if frame is None or len(frame) == 0:
            raise ValueError(f"FinLab key {key!r} 回傳空資料")
        cache_dir.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path)
    return normalize_datetime_index(frame, label) if normalize_index else frame


def load_first_available(label: str, keys: list[str], cache_dir: Path, refresh: bool = False, required: bool = True, normalize_index: bool = True):
    errors = []
    for key in keys:
        try:
            return load_or_download(label, key, cache_dir, refresh, normalize_index), key
        except Exception as exc:
            errors.append(f"{key}: {type(exc).__name__}: {exc}")
    if required:
        raise RuntimeError(f"{label} 所有候選 key 均失敗:\n" + "\n".join(errors))
    return None, None


def limit_date(frame: pd.DataFrame | pd.Series, config: V6Config):
    return frame.loc[config.start_date:config.end_date]


def standardize_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    frame.columns = ["_".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in frame.columns]
    candidates = {
        "symbol": ["stock_id", "股票代號", "證券代號", "symbol", "code"],
        "security_name": ["公司簡稱", "公司名稱", "股票名稱", "證券名稱", "name"],
        "market": ["市場別", "市場", "上市櫃", "market", "exchange"],
        "security_type": ["有價證券別", "證券類別", "商品類型", "security_type", "type"],
    }
    mapping = {}
    normalized = {c.strip().casefold().replace(" ", ""): c for c in frame.columns}
    for target, names in candidates.items():
        found = next((normalized[n.casefold().replace(" ", "")] for n in names if n.casefold().replace(" ", "") in normalized), None)
        if found is None:
            found = next((c for c in frame.columns if any(n.casefold().replace(" ", "") in c.casefold().replace(" ", "") for n in names)), None)
        if found is not None:
            mapping[found] = target
    if "symbol" not in mapping.values():
        index_text = pd.Index(frame.index).astype(str)
        if len(index_text) and index_text.str.match(r"^[0-9A-Za-z]{4,10}$").mean() > 0.8:
            frame.insert(0, "__symbol_from_index__", index_text)
            mapping["__symbol_from_index__"] = "symbol"
    required = {"symbol", "security_name", "market"}
    if not required.issubset(mapping.values()):
        raise ValueError(f"metadata 欄位不足；辨識結果={mapping}, columns={list(frame.columns[:50])}")
    out = frame[list(mapping)].rename(columns=mapping).copy()
    for col in out:
        out[col] = out[col].astype(str).str.strip()
    return out.drop_duplicates("symbol", keep="last")


def filter_common_stocks(close: pd.DataFrame, metadata_raw: pd.DataFrame):
    metadata = standardize_metadata(metadata_raw)
    allowed_market = metadata["market"].str.casefold().isin({"sii", "otc", "上市", "上櫃"})
    if "security_type" in metadata:
        common_type = metadata["security_type"].str.contains(r"普通股|common", case=False, na=False)
    else:
        common_type = pd.Series(True, index=metadata.index)
    excluded_name = metadata["security_name"].str.contains(
        r"ETF|ETN|權證|特別股|存託憑證|TDR|受益證券|指數投資證券|債券|REIT", case=False, na=False
    )
    excluded_symbol = metadata["symbol"].str.contains(r"^\d{5,}$|^[A-Z]", regex=True, na=False)
    metadata["include_common_stock"] = allowed_market & common_type & ~excluded_name & ~excluded_symbol & metadata["symbol"].isin(close.columns.astype(str))
    symbols = metadata.loc[metadata["include_common_stock"], "symbol"].tolist()
    if not symbols:
        raise ValueError("普通股 universe 篩選後為 0")
    selected = close.loc[:, close.columns.astype(str).isin(symbols)].copy()
    selected.columns = selected.columns.astype(str)
    return selected, metadata.loc[metadata["include_common_stock"]].copy(), metadata


def breadth_cache_path(symbols: list[str], config: V6Config) -> Path:
    signature = hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest()[:12]
    return config.cache_dir / f"market_breadth_v6_limitref_{len(symbols)}_{signature}.parquet"


def validate_breadth_cache(frame: pd.DataFrame) -> bool:
    required = set(PREDICTOR_SPECS) | {"up_count", "down_count", "flat_count", "valid_stock_count"}
    return required.issubset(frame.columns)


def build_reference_price_matrix(
    close: pd.DataFrame,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Previous valid close with FinLab corporate-action reference overlays."""
    reference = close.ffill().shift(1)
    loaded: dict[str, str] = {}
    for index, key in enumerate(EVENT_REFERENCE_KEYS):
        label = f"event_reference_{index}"
        event, selected = load_first_available(
            label, [key], cache_dir, refresh, required=False
        )
        if event is None:
            loaded[key] = "unavailable"
            continue
        event = event.reindex(index=reference.index, columns=reference.columns)
        reference = reference.where(event.isna(), event)
        loaded[selected or key] = str(event.index.max()) if len(event.index) else "empty"
    return reference, loaded


def write_cache_metadata(path: Path, symbols: list[str], config: V6Config) -> None:
    metadata = {
        "version": "v6",
        "predictor_version": "breadth_dynamics_extremes_regime_v6_limitref",
        "date_range": [config.start_date, config.end_date],
        "universe_hash": hashlib.sha256("|".join(sorted(symbols)).encode()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
