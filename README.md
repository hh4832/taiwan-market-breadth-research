# Taiwan Market Breadth Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/taiwan-market-breadth-research/blob/feature/v11-breadth-mechanism-regime/%E5%B8%82%E5%A0%B4%E5%BB%A3%E5%BA%A6%E9%A0%90%E6%B8%AC0050%E5%A0%B1%E9%85%AC_v11_mechanism_regime.ipynb)

目前研究版本：**v11 Breadth Mechanism & Regime Reconciliation**。

## v11 研究設計

v11 不擴大 predictor grid，而是針對 v10 Candidate A/B/C 做三個預先指定模組：

- Prior-return mechanism：以 adjusted 0050 prior 1/3/5/10 日報酬，檢查 breadth effect 的增額資訊與 attenuation。
- PR→raw mapping：將 up-ratio PR95、big-up 5D PR60/80 映射回實際 breadth ratio及年度漂移。
- Regime reconciliation：使用正確歷史 7%/10% 漲跌停規則，比較 2015-06-01 前後效果並正式檢定 Signal×Post2015 interaction。

Colab runner：`市場廣度預測0050報酬_v11_mechanism_regime.ipynb`

```text
run_id = YYYYMMDD_HHMMSS_v11_breadth_mechanism_regime_<commit12>
local = <repo>/output/v11_breadth_mechanism_regime/<run_id>/
drive = /content/drive/MyDrive/Quant_Research/taiwan-market-breadth-research/<run_id>/
```

這是 pre-specified state/shape re-test：在 2015-06-01 臺股 ±10% 漲跌停制度開始後，比較 participation、extreme ±5% 與 limit breadth，驗證其 Level／Delta 所處互斥 PR bin 是否改變 0050 自次日開盤起未來 1、3、5、10、20 日的平均報酬及上漲機率。

## v10 研究設計

- 六個 raw predictors：`up_ratio`、`down_ratio`、`big_up_ratio`、`big_down_ratio`、`limit_up_ratio`、`limit_down_ratio`。
- 每個 predictor 建立 1/3/5 日 trailing mean，再分 Level 與一階 Delta；不測 acceleration。
- rolling PR 60/126/252 嚴格以 t-1 以前資料評估 x[t]。
- Primary inference 使用七個互斥且完備的 PR bins：<5、5–20、20–40、40–60、60–80、80–95、≥95。
- Primary comparison 是 bin vs non-bin；平均報酬與勝率分開做 family/global FDR 與 Bonferroni。
- 輸出 ordered-bin shape、Spearman、high-low contrasts、non-overlap、年度穩健性與 small-N guardrails。

## v9 研究設計

- 樣本：2015-06-01 至最新資料。
- 0050 outcomes：只准使用 FinLab `etl:adj_open`、`etl:adj_close`。
- Predictors：`limit_up_ratio`、`limit_down_ratio`、`big_up_ratio`、`big_down_ratio`。
- ±5% 定義：個股報酬嚴格 `> +5%` 或 `< -5%`，分母為 `valid_stock_count`；平盤納入有效分母，缺值排除。
- Signal averaging：1、3、5 日 trailing mean；其後建立一階 delta，不建立 acceleration。
- Normalization：rolling PR 60、126、252；x[t] 只和 t-1 以前的歷史分布比較。
- Thresholds：PR≤5/20/40、PR≥60/80/95；另輸出七個 descriptive PR bins。
- Outcomes：O1→C1/C3/C5/C10/C20，HAC lag 分別為 0/2/4/9/19。
- 平均報酬與勝率 inference 分開做 global FDR、research-family FDR 與 Bonferroni。
- 勝率主要檢定為 signal vs non-signal 的 LPM + HAC；vs 50% binomial 與 odds ratio 為 supplementary。
- exact signal-date mask hash 用於假說去重；overlapping 與 non-overlapping 結果分開輸出。

Research-family 固定為：

```text
predictor_family × target × PR_window × mean_window × raw/delta × overlap_policy
```

## Colab 執行

1. 確認既有 Drive output folder 位於 `/content/drive/MyDrive/Quant_Research/taiwan-market-breadth-research`。Notebook 只驗證此固定路徑，不會猜測或自動建立資料夾。
2. Private repo 才需要在 Colab Secrets 提供 `GITHUB_TOKEN`；token 不會寫入 remote 或 notebook output。
3. 提供 FinLab 登入狀態或 `FINLAB_API_TOKEN`。
4. 開啟 v10 notebook後 Run All。同一 Runtime 再 Run All 時會先切回 `/content`，驗證既有 repo remote 後才移除並 fresh clone，避免 deleted-cwd 與 clone collision。

每次執行只建立一次 Asia/Taipei timestamp，並共用：

```text
run_id = YYYYMMDD_HHMMSS_v10_post2015_breadth_pr_bins_<commit12>
local = <repo>/output/v10_post2015_breadth_pr_bins/<run_id>/
drive = /content/drive/MyDrive/Quant_Research/taiwan-market-breadth-research/<run_id>/
```

任何同名 local/Drive run folder已存在時都會直接報錯，不覆寫舊結果。研究先完整寫入 local、完成驗證，再複製到 Drive 並比較檔案 manifest。

## 每次輸出

- `market_breadth_summary.xlsx`
- `daily_dataset.parquet`
- `run_info.txt`
- `validation_summary.md`
- `all_results_v10.parquet`
- `win_rate_results_v10.parquet`
- `pr_bin_results_v10.parquet`
- `yearly_results_v10.parquet`
- `non_overlapping_results_v10.parquet`
- `deduplicated_hypotheses_v10.parquet`
- `hypothesis_registry.csv`
- `signal_definitions.csv`
- `plots/`

## 本機執行與測試

專案指定 Python 3.11：

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python run_research.py
```

完整 FinLab 研究需要有效登入；unit tests 不需要 FinLab token。

## 漲跌停判斷

沿用 production rule：以前一有效 raw close 為基準，套用公司行動參考價 override、臺灣 tick-size rounding 與 10% limit rule。v10 不以 `daily return >= 9.5%` 近似漲跌停。

## 解讀限制

Raw p<0.05 不代表策略有效。必須一起檢查 family/global FDR、年度與 non-overlapping 穩健性、long-run drift、survivorship bias、event clustering、COVID/2022/AI bull 集中、交易成本、滑價、threshold/window mining，以及尚無 untouched OOS 的限制。勝率增加但 payoff ratio 惡化時，不得稱為 profitable strategy。
