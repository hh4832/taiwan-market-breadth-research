# Taiwan Market Breadth Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/taiwan-market-breadth-research/blob/main/%E5%B8%82%E5%A0%B4%E5%BB%A3%E5%BA%A6%E9%A0%90%E6%B8%AC0050%E5%A0%B1%E9%85%AC_v6_breadth_extremes_regime.ipynb)

以 FinLab 的臺灣上市、上櫃普通股日線資料，研究市場廣度的程度、持續性、變化速度、加速度與極端強度，是否與 0050 接下來 1–3 個交易日報酬有關。

目前版本：**v6 — Breadth Dynamics & Extreme Breadth Study**。

## 研究紀律

- signal date 為 `t`，所有 predictor、rolling PR/Z 與 MA60 regime 只使用 `t` 或以前資料。
- 正式訊號採 rolling PR/Z；full-sample quantile 不作為可交易訊號。
- 分開檢定 group vs non-group、group vs 同 regime unconditional mean、group mean vs zero。
- 多重比較依統計問題分開，再同時提供 global 與 research-family correction。
- 本版只做個別 signal discovery，不建立複合條件策略。
- 尚未納入交易成本、滑價及樣本外 walk-forward 前，研究結論只能是「修改後再測」。

## 專案結構

```text
src/market_breadth/       核心研究邏輯
tests/                    無 FinLab token 也能執行的合成資料測試
市場廣度預測0050報酬_v6_breadth_extremes_regime.ipynb
                          GitHub → Colab runner
市場廣度預測0050報酬_v5_mean_vs_zero.ipynb
                          原始 baseline，保留供追溯
```

## Colab 執行

1. 在 Colab Secrets 建立 `GITHUB_TOKEN`（private repo clone）及 FinLab 所需登入資訊。
2. 開啟 v6 notebook，依序 Run All。
3. 正式輸出會寫入 `output/market_breadth_0050_study_v6/`，並另存至：

```text
MyDrive/Quant_Research/taiwan-market-breadth-research/
YYYYMMDD_HHMMSS_<git_commit>/
```

每次 archive 包含 summary Excel、daily parquet、`run_info.txt` 與 plots，且不覆蓋舊結果。

## 本機測試

專案指定 Python 3.11。請先使用既有的 Python 3.11 環境安裝依賴，再執行：

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Limit-up / limit-down 判斷

依 `complete-pullback-fubon-pipeline` 的 production rule，以個股前一個有效收盤價為基準，依臺股價格級距取得 tick size，再以向下／向上取整計算漲停價與跌停價。歷史制度採 2015-06-01 前 ±7%、之後 ±10%。FinLab 公司行動參考價可取得時，會覆蓋原始前收基準，避免除權息、減資及面額變更造成誤判。
