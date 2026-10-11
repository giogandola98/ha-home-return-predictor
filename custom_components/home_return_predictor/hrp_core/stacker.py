"""Learned fusion: prior-centred, L2-regularised online logistic regression.

The weights are regularised toward the pure Bayesian fusion (``w_prior = w_lr = 1``,
everything else 0), so with little data the stacker reproduces
``logit(P) = logit(prior) + ln(LR)`` exactly and only deviates when resolved
predictions support it (e.g. a GPS that is less informative than assumed).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from .episodes import minutes_between
from .fusion import P_MAX, P_MIN, PRIOR_CLAMP, clamp, logit, sigmoid
from .maturity import LogEntry
from .prior import DAY_MINUTES, recency_weight

FEATURE_NAMES: tuple[str, ...] = (
    "z_prior",
    "z_lr",
    "away_horizons",
    "log1p_distance_km",
    "has_gps",
    "hour_sin",
    "hour_cos",
)
N_FEATURES = len(FEATURE_NAMES)
# Parameter vector = [bias, *feature weights]; centre = Bayes fusion.
CENTRE: tuple[float, ...] = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
REG_PSEUDO_SAMPLES = 20.0
MAX_AWAY_HORIZONS = 8.0
MAX_TRAIN_ENTRIES = 1500
MIN_ENTRIES_TO_ACTIVATE = 300  # holdout of >= 60 entries (several absences)
HOLDOUT_FRACTION = 0.2
NEWTON_ITERATIONS = 8
ACTIVATION_MARGIN = 0.01  # learned must beat Bayes holdout Brier by 1 %


def build_features(
    prior: float,
    likelihood_ratio: float,
    minutes_away: float,
    horizon_minutes: float,
    distance_m: float | None,
    has_gps: bool,
    local_hour: float,
) -> list[float]:
    """Feature vector (see FEATURE_NAMES)."""
    angle = 2.0 * math.pi * local_hour / 24.0
    return [
        logit(clamp(prior, PRIOR_CLAMP, 1 - PRIOR_CLAMP)),
        math.log(max(likelihood_ratio, 1e-9)),
        min(max(minutes_away, 0.0) / max(horizon_minutes, 1.0), MAX_AWAY_HORIZONS),
        math.log1p(distance_m / 1000.0) if distance_m is not None else 0.0,
        1.0 if has_gps else 0.0,
        math.sin(angle),
        math.cos(angle),
    ]


def bayes_from_features(features: Sequence[float]) -> float:
    """The plain Bayesian fusion expressed on the feature vector."""
    return clamp(sigmoid(features[0] + features[1]), P_MIN, P_MAX)


def apply_weights(weights: Sequence[float], features: Sequence[float]) -> float:
    z = weights[0] + sum(w * x for w, x in zip(weights[1:], features, strict=True))
    return clamp(sigmoid(z), P_MIN, P_MAX)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting (small dense systems)."""
    n = len(vector)
    a = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        if abs(a[col][col]) < 1e-12:
            continue
        for r in range(col + 1, n):
            f = a[r][col] / a[col][col]
            if f:
                for c in range(col, n + 1):
                    a[r][c] -= f * a[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        if abs(a[r][r]) < 1e-12:
            continue
        x[r] = (a[r][n] - sum(a[r][c] * x[c] for c in range(r + 1, n))) / a[r][r]
    return x


def fit_weights(
    rows: Sequence[Sequence[float]],
    outcomes: Sequence[int],
    sample_weights: Sequence[float],
    reg: float = REG_PSEUDO_SAMPLES,
    start: Sequence[float] | None = None,
) -> list[float]:
    """Newton/IRLS for logistic regression with penalty reg/2 * ||w - CENTRE||^2."""
    k = N_FEATURES + 1
    w = list(start) if start is not None else list(CENTRE)
    data = [
        ([1.0, *r], float(o), sw) for r, o, sw in zip(rows, outcomes, sample_weights, strict=True)
    ]
    rng = range(k)
    for _ in range(NEWTON_ITERATIONS):
        grad = [reg * (w[i] - CENTRE[i]) for i in range(k)]
        hess = [[reg if i == j else 0.0 for j in range(k)] for i in range(k)]
        for x, y, sw in data:
            z = 0.0
            for i in rng:
                z += w[i] * x[i]
            p = sigmoid(z)
            g = sw * (p - y)
            h = sw * max(p * (1 - p), 1e-6)
            for i in rng:
                xi = x[i]
                if xi == 0.0:
                    continue
                grad[i] += g * xi
                hx = h * xi
                row = hess[i]
                for j in range(i, k):
                    row[j] += hx * x[j]
        for i in range(k):
            for j in range(i):
                hess[i][j] = hess[j][i]
        step = _solve(hess, grad)
        w = [wi - si for wi, si in zip(w, step, strict=True)]
        if max(abs(s) for s in step) < 1e-4:
            break
    return w


def _brier(
    entries: Sequence[LogEntry], probs: Sequence[float], now: datetime, half_life: float
) -> float:
    num = den = 0.0
    for e, p in zip(entries, probs, strict=True):
        sw = recency_weight(minutes_between(e.ts, now) / DAY_MINUTES, half_life)
        num += sw * (p - float(e.outcome or 0)) ** 2
        den += sw
    return num / den if den > 0 else 0.0


class Stacker:
    """Per-person learned fusion with holdout-gated activation."""

    def __init__(self) -> None:
        self.weights: list[float] = list(CENTRE)
        self.active = False
        self.trained_on = 0
        self.holdout_brier_learned: float | None = None
        self.holdout_brier_bayes: float | None = None

    def predict(self, features: Sequence[float]) -> float:
        return apply_weights(self.weights, features)

    def fit(self, entries: Sequence[LogEntry], now: datetime, half_life_days: float) -> None:
        """Refit on the last resolved entries and decide whether to activate."""
        usable = [e for e in entries if e.outcome is not None and len(e.features) == N_FEATURES][
            -MAX_TRAIN_ENTRIES:
        ]
        self.trained_on = len(usable)
        if not usable:
            self.weights, self.active = list(CENTRE), False
            return

        previous = list(self.weights)

        def weights_for(subset: Sequence[LogEntry]) -> list[float]:
            return fit_weights(
                [e.features for e in subset],
                [int(e.outcome or 0) for e in subset],
                [
                    recency_weight(minutes_between(e.ts, now) / DAY_MINUTES, half_life_days)
                    for e in subset
                ],
                start=previous,
            )

        self.active = False
        self.holdout_brier_learned = self.holdout_brier_bayes = None
        if len(usable) >= MIN_ENTRIES_TO_ACTIVATE:
            split = int(len(usable) * (1 - HOLDOUT_FRACTION))
            train, hold = usable[:split], usable[split:]
            w_train = weights_for(train)
            hl = half_life_days
            learned = _brier(hold, [apply_weights(w_train, e.features) for e in hold], now, hl)
            bayes = _brier(hold, [bayes_from_features(e.features) for e in hold], now, hl)
            self.holdout_brier_learned, self.holdout_brier_bayes = learned, bayes
            self.active = learned < bayes * (1.0 - ACTIVATION_MARGIN)
        self.weights = weights_for(usable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": list(self.weights),
            "active": self.active,
            "trained_on": self.trained_on,
            "hb_learned": self.holdout_brier_learned,
            "hb_bayes": self.holdout_brier_bayes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Stacker:
        st = cls()
        weights = data.get("weights")
        if isinstance(weights, list) and len(weights) == N_FEATURES + 1:
            st.weights = [float(w) for w in weights]
        st.active = bool(data.get("active", False))
        st.trained_on = int(data.get("trained_on", 0))
        st.holdout_brier_learned = data.get("hb_learned")
        st.holdout_brier_bayes = data.get("hb_bayes")
        return st
