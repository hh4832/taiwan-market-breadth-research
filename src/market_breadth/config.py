from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class V6Config:
    start_date: str = "2011-01-01"
    end_date: str | None = None
    target_symbol: str = "0050"
    pr_window: int = 252
    z_window: int = 252
    min_history: int = 60
    big_move_threshold: float = 0.05
    ma_window: int = 60
    annual_rf: float = 0.02
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("output/market_breadth_0050_study_v6")
    refresh: bool = False
    limit_tolerance: float = 1e-8
    limit_change_date: str = "2015-06-01"
    old_limit_rate: float = 0.07
    current_limit_rate: float = 0.10


PREDICTOR_SPECS = {
    "up_ratio": ("LEVEL", "up"),
    "down_ratio": ("LEVEL", "down"),
    "up_ratio_3d_mean": ("ROLLING_3D", "up"),
    "down_ratio_3d_mean": ("ROLLING_3D", "down"),
    "delta_up_ratio_1d": ("DELTA", "up"),
    "delta_down_ratio_1d": ("DELTA", "down"),
    "accel_up_ratio_1d": ("ACCELERATION", "up"),
    "accel_down_ratio_1d": ("ACCELERATION", "down"),
    "big_up_ratio": ("EXTREME_5PCT", "up"),
    "big_down_ratio": ("EXTREME_5PCT", "down"),
    "limit_up_ratio": ("LIMIT", "up"),
    "limit_down_ratio": ("LIMIT", "down"),
}

TARGET_METADATA = {
    "ret_c0_o1": {"formula": "Open[t+1] / Close[t] - 1", "hac_lag": 0},
    "ret_c0_c1": {"formula": "Close[t+1] / Close[t] - 1", "hac_lag": 0},
    "ret_o1_c1": {"formula": "Close[t+1] / Open[t+1] - 1", "hac_lag": 0},
    "ret_o1_o2": {"formula": "Open[t+2] / Open[t+1] - 1", "hac_lag": 0},
    "ret_o1_c2": {"formula": "Close[t+2] / Open[t+1] - 1", "hac_lag": 1},
    "ret_o1_c3": {"formula": "Close[t+3] / Open[t+1] - 1", "hac_lag": 2},
}

PR_GROUPS = {
    "LOW_5": ("le", 0.05),
    "Q1": ("between", (0.00, 0.20)),
    "Q2": ("between", (0.20, 0.40)),
    "Q3": ("between", (0.40, 0.60)),
    "Q4": ("between", (0.60, 0.80)),
    "Q5": ("between", (0.80, 1.00)),
    "HIGH_5": ("ge", 0.95),
    "PR_LE_5": ("le", 0.05),
    "PR_LE_20": ("le", 0.20),
    "PR_GE_80": ("ge", 0.80),
    "PR_GE_95": ("ge", 0.95),
}

Z_GROUPS = {
    "Z_LE_-2": ("le", -2.0),
    "Z_LE_-1": ("le", -1.0),
    "Z_GE_1": ("ge", 1.0),
    "Z_GE_2": ("ge", 2.0),
}
