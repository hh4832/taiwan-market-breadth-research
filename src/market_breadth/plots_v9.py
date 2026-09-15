from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(frame: pd.DataFrame, x: str, ys: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for y in ys:
        ax.plot(frame[x].astype(str), frame[y], marker="o", label=y)
    ax.set_title(title); ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(path, dpi=150); plt.close(fig)


def make_v9_plots(results: pd.DataFrame, bins: pd.DataFrame, yearly: pd.DataFrame, output: Path) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    bin_focus = bins.query("pr_window == 252 and mean_window == 1 and transform == 'raw' and target == 'ret_o1_c5'")
    if not bin_focus.empty:
        agg = bin_focus.groupby("pr_bin", sort=False)[["mean_ret", "win_rate"]].mean().reset_index()
        for y, name in (("mean_ret", "pr_bins_vs_mean_return.png"), ("win_rate", "pr_bins_vs_win_rate.png")):
            _save(agg, "pr_bin", [y], y.replace("_", " ").title(), output/name); manifest.append(name)
    focus = results.query("overlap_policy == 'all_events' and transform == 'raw'").copy()
    if not focus.empty:
        for x, ys, name in (
            ("target", ["signal_win_rate", "non_signal_win_rate"], "signal_vs_non_signal_win_rate.png"),
            ("target", ["mean_ret", "mean_ret_minus_non_signal"], "horizon_decay.png"),
            ("pr_window", ["mean_ret"], "pr_window_robustness.png"),
            ("mean_window", ["mean_ret"], "averaging_robustness.png"),
        ):
            agg = focus.groupby(x)[ys].mean().reset_index()
            _save(agg, x, ys, name.replace("_", " ").title(), output/name); manifest.append(name)
    if not yearly.empty:
        agg = yearly.groupby("year")[["win_rate", "unconditional_win_rate_same_sample"]].mean().reset_index()
        name = "yearly_win_rate_stability.png"
        _save(agg, "year", ["win_rate", "unconditional_win_rate_same_sample"], "Yearly Win-rate Stability", output/name); manifest.append(name)
    return pd.DataFrame({"plot": manifest})
