"""Make ``hrp_core`` importable without importing the Home Assistant package."""

from __future__ import annotations

import sys
from pathlib import Path

_INTEGRATION_DIR = (
    Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor"
)
if str(_INTEGRATION_DIR) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_DIR))
