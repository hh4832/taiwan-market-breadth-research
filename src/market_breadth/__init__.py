"""Taiwan market breadth research v6."""

from .config import PREDICTOR_SPECS, TARGET_METADATA, V6Config
from .core import (
    add_forward_returns,
    add_market_regime,
    add_rolling_normalization,
    build_market_breadth,
)
from .statistics import run_signal_study

__all__ = [
    "PREDICTOR_SPECS",
    "TARGET_METADATA",
    "V6Config",
    "add_forward_returns",
    "add_market_regime",
    "add_rolling_normalization",
    "build_market_breadth",
    "run_signal_study",
]

