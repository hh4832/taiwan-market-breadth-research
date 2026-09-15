from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _line(frame: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for y in ys:
        ax.plot(frame[x].astype(str), frame[y], marker="o", label=y)
    ax.set_title(title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def make_v10_plots(results: pd.DataFrame, yearly: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    def save(frame: pd.DataFrame, x: str, ys: list[str], title: str, filename: str) -> None:
        if frame.empty:
            return
        _line(frame, x, ys, title, output / filename); files.append(filename)

    primary = results.loc[results.overlap_policy.eq("all_events")]
    base = primary.query("pr_window == 252 and mean_window == 1 and transform == 'level'")
    ordered = base.groupby(["pr_bin_order", "pr_bin"], as_index=False)[["mean_ret", "bin_win_rate"]].mean().sort_values("pr_bin_order")
    save(ordered, "pr_bin", ["mean_ret"], "PR bin vs mean return", "01_pr_bin_mean_return.png")
    save(ordered, "pr_bin", ["bin_win_rate"], "PR bin vs win rate", "02_pr_bin_win_rate.png")

    down = base.loc[base.predictor.str.startswith("down_ratio__") & base.target.isin(["ret_o1_c1", "ret_o1_c3", "ret_o1_c5"])]
    down = down.pivot_table(index=["pr_bin_order", "pr_bin"], columns="target", values="mean_ret", aggfunc="mean").reset_index().sort_values("pr_bin_order")
    save(down, "pr_bin", [c for c in ["ret_o1_c1", "ret_o1_c3", "ret_o1_c5"] if c in down], "down_ratio across C1/C3/C5", "03_down_ratio_c1_c3_c5.png")

    up = base.loc[base.predictor.str.startswith("up_ratio__") & base.target.eq("ret_o1_c1")]
    up = up.groupby(["pr_bin_order", "pr_bin"], as_index=False).bin_win_rate.mean().sort_values("pr_bin_order")
    save(up, "pr_bin", ["bin_win_rate"], "up_ratio O1→C1", "04_up_ratio_c1.png")

    family = primary.groupby("predictor_family", as_index=False)[["mean_difference", "bin_minus_non_bin_win_rate"]].mean()
    save(family, "predictor_family", ["mean_difference", "bin_minus_non_bin_win_rate"], "Participation vs intensity", "05_participation_intensity.png")
    transform = primary.groupby("transform", as_index=False)[["mean_difference", "bin_minus_non_bin_win_rate"]].mean()
    save(transform, "transform", ["mean_difference", "bin_minus_non_bin_win_rate"], "Level vs delta", "06_level_delta.png")
    means = primary.groupby("mean_window", as_index=False)[["mean_difference", "bin_minus_non_bin_win_rate"]].mean()
    save(means, "mean_window", ["mean_difference", "bin_minus_non_bin_win_rate"], "1D vs 3D vs 5D", "07_mean_windows.png")
    pr = primary.groupby("pr_window", as_index=False)[["mean_difference", "bin_minus_non_bin_win_rate"]].mean()
    save(pr, "pr_window", ["mean_difference", "bin_minus_non_bin_win_rate"], "PR-window robustness", "08_pr_windows.png")
    horizons = primary.groupby("target", as_index=False)[["mean_difference", "bin_minus_non_bin_win_rate"]].mean()
    save(horizons, "target", ["mean_difference", "bin_minus_non_bin_win_rate"], "Horizon decay", "09_horizon_decay.png")
    if not yearly.empty:
        annual = yearly.groupby("year", as_index=False)[["mean_ret", "win_rate"]].mean()
        save(annual, "year", ["mean_ret", "win_rate"], "Yearly stability", "10_yearly_stability.png")
    return pd.DataFrame({"plot": files})
