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


@dataclass(frozen=True)
class V7Config(V6Config):
    """v7 keeps the v6 discovery grid and adds prespecified robustness tests."""

    output_dir: Path = Path("output/market_breadth_0050_study_v7")
    pullback_thresholds: tuple[float, ...] = (0.0, -0.005, -0.01)


@dataclass(frozen=True)
class V8Config(V7Config):
    """Prespecified validation of next-day cooling after high limit-up breadth."""

    output_dir: Path = Path("output/market_breadth_0050_study_v8_cooldown")
    cooldown_thresholds: tuple[float, ...] = (0.0, -0.005, -0.01)


@dataclass(frozen=True)
class V9Config(V6Config):
    """Pre-specified post-2015 breadth PR and win-rate validation."""

    start_date: str = "2015-06-01"
    study_version: str = "v9 Post-2015 Breadth PR & Win-Rate Validation"
    version_slug: str = "v9_post2015_breadth_pr_winrate"
    pr_windows: tuple[int, ...] = (60, 126, 252)
    signal_mean_windows: tuple[int, ...] = (1, 3, 5)
    pr_thresholds: tuple[float, ...] = (0.05, 0.20, 0.40, 0.60, 0.80, 0.95)
    output_dir: Path = Path("output")


@dataclass(frozen=True)
class V10Config(V9Config):
    """Pre-specified post-2015 state and mutually-exclusive PR-bin study."""

    study_version: str = "v10 Post-2015 Breadth State & PR-Bin Validation"
    version_slug: str = "v10_post2015_breadth_pr_bins"


V9_RAW_PREDICTORS = {
    "limit_up_ratio": ("LIMIT", "up"),
    "limit_down_ratio": ("LIMIT", "down"),
    "big_up_ratio": ("EXTREME_5PCT", "up"),
    "big_down_ratio": ("EXTREME_5PCT", "down"),
}

V10_RAW_PREDICTORS = {
    "up_ratio": ("PARTICIPATION", "up"),
    "down_ratio": ("PARTICIPATION", "down"),
    "big_up_ratio": ("EXTREME_5PCT", "up"),
    "big_down_ratio": ("EXTREME_5PCT", "down"),
    "limit_up_ratio": ("LIMIT", "up"),
    "limit_down_ratio": ("LIMIT", "down"),
}

V9_TARGET_METADATA = {
    "ret_o1_c1": {"formula": "Adjusted Close[t+1] / Adjusted Open[t+1] - 1", "hac_lag": 0, "horizon": 1},
    "ret_o1_c3": {"formula": "Adjusted Close[t+3] / Adjusted Open[t+1] - 1", "hac_lag": 2, "horizon": 3},
    "ret_o1_c5": {"formula": "Adjusted Close[t+5] / Adjusted Open[t+1] - 1", "hac_lag": 4, "horizon": 5},
    "ret_o1_c10": {"formula": "Adjusted Close[t+10] / Adjusted Open[t+1] - 1", "hac_lag": 9, "horizon": 10},
    "ret_o1_c20": {"formula": "Adjusted Close[t+20] / Adjusted Open[t+1] - 1", "hac_lag": 19, "horizon": 20},
}


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


V7_CANDIDATES = (
    ("limit_down_ratio", "PR", "PR_GE_80", "ALL", "ret_o1_c2"),
    ("delta_down_ratio_1d", "PR", "PR_GE_80", "BULL", "ret_o1_c3"),
    ("up_ratio", "PR", "PR_GE_95", "ALL", "ret_o1_c1"),
    ("limit_up_ratio", "PR", "PR_GE_80", "ALL", "ret_o1_c3"),
    ("limit_up_ratio", "PR", "PR_GE_80", "BULL", "ret_o1_c3"),
    ("accel_up_ratio_1d", "PR", "PR_GE_80", "BEAR", "ret_c0_o1"),
)

V7_VALIDATION_TARGET_METADATA = {
    "ret_o1_c5": {"formula": "Close[t+5] / Open[t+1] - 1", "hac_lag": 4},
    "ret_o2_c2": {"formula": "Close[t+2] / Open[t+2] - 1", "hac_lag": 0},
    "ret_o2_c3": {"formula": "Close[t+3] / Open[t+2] - 1", "hac_lag": 1},
    "ret_o2_c5": {"formula": "Close[t+5] / Open[t+2] - 1", "hac_lag": 3},
    "ret_c1_c2": {"formula": "Close[t+2] / Close[t+1] - 1", "hac_lag": 0},
    "ret_c1_c3": {"formula": "Close[t+3] / Close[t+1] - 1", "hac_lag": 1},
    "ret_c1_c5": {"formula": "Close[t+5] / Close[t+1] - 1", "hac_lag": 3},
    "ret_o2_c10": {"formula": "Close[t+10] / Open[t+2] - 1", "hac_lag": 8},
    "ret_c1_c10": {"formula": "Close[t+10] / Close[t+1] - 1", "hac_lag": 8},
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
