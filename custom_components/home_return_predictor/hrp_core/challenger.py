"""Optional gradient-boosting challenger (in-house numpy GBM, see ``gbm.py``).

Training needs numpy (``HistGBM.fit``); serving does not: the fitted trees are
exported to a plain-Python dict and evaluated with ``HistGBM.predict_proba_exported``
(~0.1 ms per row, thread-free). The exported model is JSON-safe and small, so it is
persisted with the person's state (``to_dict``) and a promoted challenger keeps
serving after a restart without waiting for a retraining.

scikit-learn is no longer used: it publishes no musllinux wheels, so it cannot be
installed on Home Assistant OS (Alpine).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any

from .fusion import P_MAX, P_MIN, PRIOR_CLAMP, clamp, logit, sigmoid
from .gbm import HistGBM, numpy_available

CHALLENGER_FEATURES: tuple[str, ...] = (
    "z_prior",
    "ln_lr",
    "away_horizons",
    "dep_hour_sin",
    "dep_hour_cos",
    "hour_sin",
    "hour_cos",
    "rest_day",
    "log1p_distance_km",
    "speed_kmh",
    "has_gps",
    "z_bayes",
)
N_CHALLENGER_FEATURES = len(CHALLENGER_FEATURES)
MAX_AWAY_HORIZONS = 16.0
MAX_ITER = 80
MAX_DEPTH = 3
MIN_SAMPLES_LEAF = 20
LEARNING_RATE = 0.1
L2_REGULARIZATION = 1.0
MAX_BINS = 64
# The exported model is persisted only below this JSON size (80 depth-3 trees are
# far smaller); a larger one is kept in memory and retrained after a restart.
MAX_PERSISTED_BYTES = 150_000


def sklearn_available() -> bool:
    """Deprecated alias of ``numpy_available`` (kept for one release)."""
    return numpy_available()


def _hour_angle(hour: float) -> float:
    return 2.0 * math.pi * hour / 24.0


def build_challenger_features(
    *,
    prior: float,
    likelihood_ratio: float,
    minutes_away: float,
    horizon_minutes: float,
    departure_hour: float | None,
    local_hour: float,
    rest_day: bool,
    distance_m: float | None,
    speed_kmh: float | None,
    has_gps: bool,
    p_bayes: float,
) -> list[float]:
    """Feature vector of the challenger (see CHALLENGER_FEATURES)."""
    dep = _hour_angle(departure_hour) if departure_hour is not None else None
    now = _hour_angle(local_hour)
    return [
        logit(clamp(prior, PRIOR_CLAMP, 1 - PRIOR_CLAMP)),
        math.log(max(likelihood_ratio, 1e-9)),
        min(max(minutes_away, 0.0) / max(horizon_minutes, 1.0), MAX_AWAY_HORIZONS),
        math.sin(dep) if dep is not None else 0.0,
        math.cos(dep) if dep is not None else 0.0,
        math.sin(now),
        math.cos(now),
        1.0 if rest_day else 0.0,
        math.log1p(distance_m / 1000.0) if distance_m is not None else 0.0,
        float(speed_kmh) if speed_kmh is not None else 0.0,
        1.0 if has_gps else 0.0,
        logit(clamp(p_bayes, P_MIN, P_MAX)),
    ]


def bayes_from_challenger_features(row: Sequence[float]) -> float:
    """The kernel-Bayes probability stored in a challenger feature row."""
    return clamp(sigmoid(row[-1]), P_MIN, P_MAX)


class GBMChallenger:
    """Gradient-boosted trees on the prediction-log features.

    The fitted model is held as the plain-Python export of ``HistGBM`` (or as a
    constant log-odds when the training log has a single class).
    """

    def __init__(self) -> None:
        self._model: dict[str, Any] | None = None
        self._constant: float | None = None
        self.trained_on = 0

    @property
    def trained(self) -> bool:
        return self._model is not None or self._constant is not None

    def fit(
        self,
        rows: Sequence[Sequence[float]],
        labels: Sequence[int],
        weights: Sequence[float] | None = None,
    ) -> None:
        """Fit (numpy, single thread). Raises RuntimeError if numpy is missing."""
        if not numpy_available():
            raise RuntimeError("numpy is not available")
        y = [int(v) for v in labels]
        self.trained_on = len(y)
        if len(set(y)) < 2:
            # Degenerate log (all returns or none): a constant predictor.
            rate = sum(y) / len(y) if y else 0.5
            self._model, self._constant = None, logit(clamp(rate, P_MIN, P_MAX))
            return
        gbm = HistGBM(
            max_iter=MAX_ITER,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            l2_regularization=L2_REGULARIZATION,
            max_bins=MAX_BINS,
        )
        gbm.fit(rows, y, sample_weight=None if weights is None else list(weights))
        self._model, self._constant = gbm.export(), None

    def predict_proba(self, row: Sequence[float]) -> float:
        """P(return within horizon) for one feature row, clamped to [0.01, 0.99]."""
        if self._model is not None:
            return clamp(HistGBM.predict_proba_exported(self._model, row), P_MIN, P_MAX)
        if self._constant is not None:
            return clamp(sigmoid(self._constant), P_MIN, P_MAX)
        raise RuntimeError("challenger not trained")

    def predict_many(self, rows: Sequence[Sequence[float]]) -> list[float]:
        return [self.predict_proba(r) for r in rows]

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any] | None:
        """JSON-safe state, or None when untrained or too large to persist."""
        if not self.trained:
            return None
        data: dict[str, Any] = {"trained_on": self.trained_on}
        if self._model is not None:
            data["model"] = self._model
        else:
            data["constant"] = self._constant
        if len(json.dumps(data)) > MAX_PERSISTED_BYTES:
            return None
        return data

    @classmethod
    def from_dict(cls, data: Any) -> GBMChallenger:
        """Restore a persisted challenger; anything unusable yields an untrained one."""
        ch = cls()
        if not isinstance(data, dict):
            return ch
        try:
            if isinstance(data.get("model"), dict):
                if not HistGBM.validate_exported(data["model"], N_CHALLENGER_FEATURES):
                    return cls()
                ch._model = data["model"]
            elif data.get("constant") is not None:
                ch._constant = float(data["constant"])
            else:
                return cls()
            ch.trained_on = int(data.get("trained_on", 0))
            p = ch.predict_proba([0.0] * N_CHALLENGER_FEATURES)  # validate the layout
            if not math.isfinite(p):
                return cls()
        except Exception:  # corrupted / incompatible export: retrain instead
            return cls()
        return ch
