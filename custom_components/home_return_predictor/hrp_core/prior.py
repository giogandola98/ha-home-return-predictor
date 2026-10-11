"""Kernel-weighted survival prior P(return in (d, d+H] | departure, away d, day)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, tzinfo

from .base import AbsenceEpisode
from .episodes import to_local

TOD_SIGMA_MINUTES = 75.0
SAME_WEEKDAY_WEIGHT = 1.0
SAME_CLASS_WEIGHT = 0.5
OTHER_DAY_WEIGHT = 0.15
GLOBAL_PSEUDO_WEIGHT = 1.5
FALLBACK_HAZARD = 0.3
LONG_ABSENCE_MINUTES = 12 * 60.0
LONG_FALLBACK_HAZARD = 0.05
MIN_ETA_WEIGHT = 0.3
DAY_MINUTES = 1440.0


def minute_of_day(dt: datetime) -> float:
    """Minutes since local midnight."""
    return dt.hour * 60.0 + dt.minute + dt.second / 60.0


def circular_minutes_diff(a: float, b: float) -> float:
    """Shortest distance between two times of day, in minutes (wraps at 24 h)."""
    diff = abs(a - b) % DAY_MINUTES
    return min(diff, DAY_MINUTES - diff)


def tod_kernel(a: float, b: float, sigma: float = TOD_SIGMA_MINUTES) -> float:
    """Circular Gaussian similarity of two times of day."""
    diff = circular_minutes_diff(a, b)
    return math.exp(-0.5 * (diff / sigma) ** 2)


def is_rest_day(weekday: int, is_holiday: bool) -> bool:
    """Weekend or holiday class."""
    return is_holiday or weekday >= 5


def day_weight(weekday_a: int, holiday_a: bool, weekday_b: int, holiday_b: bool) -> float:
    """Same weekday 1.0, same weekday/rest class 0.5, otherwise 0.15."""
    if is_rest_day(weekday_a, holiday_a) != is_rest_day(weekday_b, holiday_b):
        return OTHER_DAY_WEIGHT
    return SAME_WEEKDAY_WEIGHT if weekday_a == weekday_b else SAME_CLASS_WEIGHT


def recency_weight(age_days: float, half_life_days: float) -> float:
    """Exponential decay 2^(-age / half_life); future ages count as 0 days."""
    return 2.0 ** (-max(age_days, 0.0) / max(half_life_days, 1e-6))


def weighted_quantile(pairs: Sequence[tuple[float, float]], q: float) -> float | None:
    """Weighted quantile of (value, weight) pairs (step interpolation)."""
    items = sorted((v, w) for v, w in pairs if w > 0)
    total = sum(w for _, w in items)
    if total <= 0:
        return None
    target = q * total
    acc = 0.0
    for value, weight in items:
        acc += weight
        if acc >= target:
            return value
    return items[-1][0]


@dataclass(frozen=True)
class PriorResult:
    """Prior estimate plus the habit return-duration distribution."""

    probability: float
    effective_samples: float
    kernel_weight: float
    eta_duration_minutes: float | None  # weighted median total absence duration
    eta_iqr_minutes: float | None


@dataclass(frozen=True)
class EpisodeKey:
    """Precomputed per-episode quantities used by the prior (cacheable)."""

    left_ts: float  # POSIX seconds
    duration: float  # minutes
    tod: float  # local minute of day of departure
    weekday: int
    is_holiday: bool


def episode_keys(episodes: Sequence[AbsenceEpisode], tz: tzinfo | None) -> list[EpisodeKey]:
    """Precompute EpisodeKey for each episode."""
    return [
        EpisodeKey(
            e.left_at.timestamp(),
            e.duration_minutes,
            minute_of_day(to_local(e.left_at, tz)),
            e.weekday,
            e.is_holiday,
        )
        for e in episodes
    ]


def _shrunk(hits: float, total: float, target: float, pseudo: float) -> float:
    return (hits + pseudo * target) / (total + pseudo)


def estimate_prior(
    episodes: Sequence[EpisodeKey],
    trips: Sequence[EpisodeKey],
    *,
    now: datetime,
    left_at: datetime | None,
    minutes_away: float,
    horizon_minutes: float,
    weekday: int,
    is_holiday: bool,
    half_life_days: float,
    tz: tzinfo | None,
) -> PriorResult:
    """Weighted survival estimate of a return within the horizon.

    ``episodes``/``trips`` come from :func:`episode_keys`.
    """
    d = max(minutes_away, 0.0)
    end = d + horizon_minutes
    now_ts = now.timestamp()
    decay = math.log(2.0) / (max(half_life_days, 1e-6) * 86400.0)

    if d > LONG_ABSENCE_MINUTES:
        # Long absence: unkerneled, recency-weighted estimate over all absences.
        hits = total = 0.0
        for ep in (*episodes, *trips):
            dur = ep.duration
            if dur <= d:
                continue
            r = math.exp(-decay * max(now_ts - ep.left_ts, 0.0))
            total += r
            if dur <= end:
                hits += r
        p = _shrunk(hits, total, LONG_FALLBACK_HAZARD, GLOBAL_PSEUDO_WEIGHT)
        return PriorResult(p, 0.0, 0.0, None, None)

    cur_tod = minute_of_day(to_local(left_at, tz)) if left_at is not None else None
    k_hits = k_total = k_sq = 0.0
    g_hits = g_total = 0.0
    durations: list[tuple[float, float]] = []
    for ep in episodes:
        dur = ep.duration
        if dur <= d:
            continue
        r = math.exp(-decay * max(now_ts - ep.left_ts, 0.0))
        if r < 1e-4:
            continue
        hit = dur <= end
        g_total += r
        if hit:
            g_hits += r
        w = r * day_weight(weekday, is_holiday, ep.weekday, ep.is_holiday)
        if cur_tod is not None:
            w *= tod_kernel(cur_tod, ep.tod)
        if w <= 1e-9:
            continue
        k_total += w
        k_sq += w * w
        if hit:
            k_hits += w
        durations.append((dur, w))

    global_hazard = _shrunk(g_hits, g_total, FALLBACK_HAZARD, GLOBAL_PSEUDO_WEIGHT)
    p = _shrunk(k_hits, k_total, global_hazard, GLOBAL_PSEUDO_WEIGHT)
    n_eff = (k_total * k_total / k_sq) if k_sq > 0 else 0.0

    median = iqr = None
    if left_at is not None and k_total >= MIN_ETA_WEIGHT:
        median = weighted_quantile(durations, 0.5)
        q1 = weighted_quantile(durations, 0.25)
        q3 = weighted_quantile(durations, 0.75)
        if q1 is not None and q3 is not None:
            iqr = q3 - q1
    return PriorResult(min(max(p, 0.0), 1.0), n_eff, k_total, median, iqr)
