"""Log-odds fusion of prior and GPS evidence, and ETA combination."""

from __future__ import annotations

import math

P_MIN = 0.01
P_MAX = 0.99
PRIOR_CLAMP = 0.005
MIN_POSITION_SPEED_KMH = 5.0
POSITION_ETA_MAX_HORIZONS = 3.0
MIN_HABIT_SD_MINUTES = 10.0
IQR_TO_SD = 1.35


def clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fuse_probability(prior: float, likelihood_ratio: float) -> float:
    """logit(P) = logit(prior) + ln(LR), final P clamped to [0.01, 0.99]."""
    z = logit(clamp(prior, PRIOR_CLAMP, 1 - PRIOR_CLAMP)) + math.log(max(likelihood_ratio, 1e-9))
    return clamp(sigmoid(z), P_MIN, P_MAX)


def position_eta_minutes(
    distance_m: float | None, speed_kmh: float | None, horizon_minutes: float
) -> float | None:
    """distance / speed when approaching faster than 5 km/h and within 3 horizons."""
    if distance_m is None or speed_kmh is None or speed_kmh <= MIN_POSITION_SPEED_KMH:
        return None
    minutes = (distance_m / 1000.0) / speed_kmh * 60.0
    if minutes > POSITION_ETA_MAX_HORIZONS * horizon_minutes:
        return None
    return minutes


def combine_eta(
    habit_minutes: float | None,
    habit_iqr_minutes: float | None,
    position_minutes: float | None,
) -> tuple[float | None, str | None]:
    """Inverse-variance combination; returns (minutes from now, source)."""
    if habit_minutes is None and position_minutes is None:
        return None, None
    if position_minutes is None:
        return habit_minutes, "habit"
    if habit_minutes is None:
        return position_minutes, "position"
    sd_h = max((habit_iqr_minutes or 0.0) / IQR_TO_SD, MIN_HABIT_SD_MINUTES)
    var_h = sd_h**2
    var_p = (0.25 * position_minutes + 3.0) ** 2
    w_h, w_p = 1.0 / var_h, 1.0 / var_p
    return (w_h * habit_minutes + w_p * position_minutes) / (w_h + w_p), "fused"
