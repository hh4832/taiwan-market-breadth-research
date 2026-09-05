from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import V6Config


def make_core_plots(dataset: pd.DataFrame, results: pd.DataFrame, output_dir: Path, config: V6Config | None = None) -> pd.DataFrame:
    cfg = config or V6Config()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    columns = ["down_ratio", "down_ratio_3d_mean", "big_down_ratio", "limit_down_ratio"]
    available = [c for c in columns if c in dataset]
    if available:
        ax = dataset[available].plot(figsize=(14, 7), alpha=.8, title="Breadth dynamics")
        ax.set_ylabel("ratio")
        path = output_dir / "breadth_dynamics.png"
        ax.figure.tight_layout(); ax.figure.savefig(path, dpi=160); plt.close(ax.figure); paths.append(path)

    focus = results.loc[
        (results["predictor"] == "down_ratio")
        & (results["signal_method"] == "PR")
        & (results["group"].isin(["PR_LE_20", "PR_GE_80"]))
        & (results["market_regime"].isin(["BULL", "BEAR"]))
    ]
    if not focus.empty:
        pivot = focus.pivot_table(index=["target", "group"], columns="market_regime", values="mean_ret")
        ax = pivot.plot.bar(figsize=(12, 6), title="down_ratio: Bull vs Bear")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_ylabel("mean return")
        path = output_dir / "down_ratio_bull_vs_bear.png"
        ax.figure.tight_layout(); ax.figure.savefig(path, dpi=160); plt.close(ax.figure); paths.append(path)

    marker_col = f"down_ratio__PR_{cfg.pr_window}"
    if marker_col in dataset and "close_0050" in dataset:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(dataset.index, dataset["close_0050"], label="0050 close", color="black")
        for mask, color, label in [
            (dataset[marker_col].ge(.95), "red", "PR >= 95"),
            (dataset[marker_col].le(.05), "green", "PR <= 5"),
        ]:
            ax.scatter(dataset.index[mask], dataset.loc[mask, "close_0050"], s=15, color=color, label=label)
        ax.legend(); ax.set_title("0050 with down_ratio rolling-PR signals")
        path = output_dir / "0050_down_ratio_pr_markers.png"
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(path)
    return pd.DataFrame({"figure_path": [str(p) for p in paths], "file_name": [p.name for p in paths]})

