"""Log-odds fusion of prior and GPS evidence, and ETA combination."""

from __future__ import annotations

import math

P_MIN = 0.01
P_MAX = 0.99
# Prior floor/ceiling before fusion: a sharp habit prior (e.g. 0.005 at an odd time)
# must still let strong, calibrated GPS evidence through (LR up to 50 -> P ~ 0.5).
PRIOR_CLAMP = 0.02
MIN_POSITION_SPEED_KMH = 5.0
POSITION_ETA_MAX_HORIZONS = 3.0
MIN_HABIT_SD_MINUTES = 10.0
IQR_TO_SD = 1.35
# Door-to-door upper bound for ground travel (motorway car, high-speed train incl. stops).
# Straight-line distance / this speed is a hard lower bound on the time needed to get home.
MAX_TRAVEL_SPEED_KMH = 200.0


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
    """logit(P) = logit(clamp(prior, 0.02, 0.98)) + ln(LR), final P clamped to [0.01, 0.99]."""
    z = logit(clamp(prior, PRIOR_CLAMP, 1 - PRIOR_CLAMP)) + math.log(max(likelihood_ratio, 1e-9))
    return clamp(sigmoid(z), P_MIN, P_MAX)


def min_travel_minutes(distance_m: float | None) -> float | None:
    """Lower bound on the time to get home: straight-line distance at MAX_TRAVEL_SPEED_KMH."""
    if distance_m is None:
        return None
    return max(distance_m, 0.0) / 1000.0 / MAX_TRAVEL_SPEED_KMH * 60.0


def apply_feasibility(
    probability: float,
    eta_minutes: float | None,
    distance_m: float | None,
    horizon_minutes: float,
    elapsed_minutes: float = 0.0,
) -> tuple[float, float | None, bool]:
    """Physical sanity check on top of any model.

    If the person cannot reach home within the horizon even at MAX_TRAVEL_SPEED_KMH, a return
    within the horizon is impossible: probability -> P_MIN. The ETA is never earlier than the
    minimum travel time. Unknown distance -> no change.
    ``elapsed_minutes``: age of the fix the distance comes from; the person may have been
    travelling since, so it is subtracted from the minimum travel time.
    Returns (probability, eta_minutes, feasible).
    """
    minimum = min_travel_minutes(distance_m)
    if minimum is None:
        return probability, eta_minutes, True
    minimum = max(minimum - max(elapsed_minutes, 0.0), 0.0)
    if eta_minutes is not None and eta_minutes < minimum:
        eta_minutes = minimum
    if minimum > horizon_minutes:
        return P_MIN, eta_minutes, False
    return probability, eta_minutes, True


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
