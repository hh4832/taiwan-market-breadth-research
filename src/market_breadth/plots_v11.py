from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _plot(frame: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for y in ys:
        ax.plot(frame[x].astype(str), frame[y], marker="o", label=y)
    ax.set_title(title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_v11_plots(prior: pd.DataFrame, regressions: pd.DataFrame, threshold_yearly: pd.DataFrame,
                   regime: pd.DataFrame, interactions: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True); files = []
    def save(frame, x, ys, title, filename):
        if len(frame): _plot(frame, x, ys, title, output/filename); files.append(filename)
    ab = prior.loc[prior.candidate_group.isin(["A", "B"]) & prior.prior_window.eq(5)].pivot_table(index="candidate", columns="group", values="mean").reset_index()
    save(ab, "candidate", [c for c in ["signal", "non_signal"] if c in ab], "Candidate A/B prior 5D return", "01_candidate_ab_prior5d.png")
    c = prior.loc[prior.candidate_group.eq("C") & prior.prior_window.isin([1, 3])].copy(); c["series"] = c.group + "_" + c.prior_window.astype(str) + "d"
    c = c.pivot_table(index="candidate", columns="series", values="mean").reset_index()
    save(c, "candidate", [x for x in c.columns if x != "candidate"], "Candidate C prior rally", "02_candidate_c_prior.png")
    beta = regressions.groupby("model", as_index=False).breadth_beta.mean()
    save(beta, "model", ["breadth_beta"], "Breadth beta before/after prior-return controls", "03_beta_attenuation.png")
    up = threshold_yearly.loc[threshold_yearly.predictor.eq("up_ratio__mean_1d")].copy(); up["series"] = "PR" + up.pr_window.astype(str)
    up = up.pivot_table(index="year", columns="series", values="median_threshold").reset_index()
    save(up, "year", [x for x in up.columns if x != "year"], "up_ratio PR95 raw threshold", "04_up_ratio_pr95_threshold.png")
    big = threshold_yearly.loc[threshold_yearly.predictor.eq("big_up_ratio__mean_5d")].copy(); big["series"] = "PR" + big.pr_window.astype(str) + "_Q" + (big["quantile"]*100).astype(int).astype(str)
    big = big.pivot_table(index="year", columns="series", values="median_threshold").reset_index()
    save(big, "year", [x for x in big.columns if x != "year"], "big_up 5D PR60/80 threshold", "05_big_up_threshold.png")
    down = regime.loc[regime.predictor.str.startswith("down_ratio__")].groupby(["pr_bin", "regime"], as_index=False)[["mean_ret", "bin_win_rate"]].mean()
    down_mean = down.pivot(index="pr_bin", columns="regime", values="mean_ret").reset_index()
    save(down_mean, "pr_bin", [x for x in down_mean.columns if x != "pr_bin"], "Pre/post down_ratio mean return", "06_regime_down_mean.png")
    down_win = down.pivot(index="pr_bin", columns="regime", values="bin_win_rate").reset_index()
    save(down_win, "pr_bin", [x for x in down_win.columns if x != "pr_bin"], "Pre/post down_ratio win rate", "07_regime_down_win.png")
    inter = interactions.loc[interactions.predictor.str.startswith("down_ratio__")].groupby("pr_bin", as_index=False).beta_interaction.mean()
    save(inter, "pr_bin", ["beta_interaction"], "Signal×Post2015 interaction", "08_interaction.png")
    return pd.DataFrame({"plot": files})
