# Taiwan Market Breadth Research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hh4832/taiwan-market-breadth-research/blob/feature/v9-post2015-breadth-pr-winrate/%E5%B8%82%E5%A0%B4%E5%BB%A3%E5%BA%A6%E9%A0%90%E6%B8%AC0050%E5%A0%B1%E9%85%AC_v9_post2015_pr_winrate.ipynb)

目前研究版本：**v9 Post-2015 Breadth PR & Win-Rate Validation**。

這是 pre-specified robustness re-test：在 2015-06-01 臺股 ±10% 漲跌停制度開始後，重新驗證市場極端廣度與變化速度是否改變 0050 自次日開盤起未來 1、3、5、10、20 日的平均報酬及上漲機率。

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

1. 將指定的既有 Drive output folder 建立 MyDrive 捷徑，固定命名為 `taiwan-market-breadth-output`。Notebook 只驗證此捷徑，不會猜路徑或建立同名資料夾。
2. Private repo 才需要在 Colab Secrets 提供 `GITHUB_TOKEN`；token 不會寫入 remote 或 notebook output。
3. 提供 FinLab 登入狀態或 `FINLAB_API_TOKEN`。
4. 開啟 v9 notebook後 Run All。同一 Runtime 再 Run All 時會先切回 `/content`，驗證既有 repo remote 後才移除並 fresh clone，避免 deleted-cwd 與 clone collision。

每次執行只建立一次 Asia/Taipei timestamp，並共用：

```text
run_id = YYYYMMDD_HHMMSS_v9_post2015_breadth_pr_winrate_<commit12>
local = <repo>/output/v9_post2015_breadth_pr_winrate/<run_id>/
drive = <existing Drive output root>/<run_id>/
```

任何同名 local/Drive run folder已存在時都會直接報錯，不覆寫舊結果。研究先完整寫入 local、完成驗證，再複製到 Drive 並比較檔案 manifest。

## 每次輸出

- `market_breadth_summary.xlsx`
- `daily_dataset.parquet`
- `run_info.txt`
- `validation_summary.md`
- `win_rate_results.parquet`
- `yearly_results.parquet`
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

沿用 production rule：以前一有效 raw close 為基準，套用公司行動參考價 override、臺灣 tick-size rounding 與 10% limit rule。v9 不以 `daily return >= 9.5%` 近似漲跌停。

## 解讀限制

Raw p<0.05 不代表策略有效。必須一起檢查 family/global FDR、年度與 non-overlapping 穩健性、long-run drift、survivorship bias、event clustering、COVID/2022/AI bull 集中、交易成本、滑價、threshold/window mining，以及尚無 untouched OOS 的限制。勝率增加但 payoff ratio 惡化時，不得稱為 profitable strategy。
