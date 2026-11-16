"""Distribution of the time until a person (or the first of several) arrives home.

The return probability answers "home within the horizon?"; a heating decision needs
the whole distribution of the arrival time: preheating ``L`` minutes ahead is right
when ``P(arrival <= now + L) >= c``, i.e. when ``now >= q_c(arrival) - L``.

Per person (pure Python, no numpy):

* **habit** - up to ``LONG_ABSENCE_MINUTES`` away: the residual durations of the past
  absences longer than the current one, weighted by the kernel prior (recency x day x
  departure time, ``PriorResult.durations``) and smoothed; plus a small pseudo-weight
  of the long-absence model below, so the distribution never ends.
* **long absences** - beyond ``LONG_ABSENCE_MINUTES``: a daily return hazard (the share
  of past absences longer than the current one that ended within the next day, shrunk
  to a default going from "same day" to "a trip of days", ``target_day_hazard``),
  spread over the day by the habitual arrival times of day.
* **evidence** - with a GPS approach the probability mass the evidence adds over the
  habit prior (``p_bayes - prior``) moves onto the kinematic ETA; whatever the horizon,
  no arrival is possible before ``distance / MAX_TRAVEL_SPEED_KMH`` (truncation).

The house: the first arrival among the persons away, assumed independent
``F = 1 - prod(1 - F_i)``.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, tzinfo

from .episodes import to_local
from .prior import (
    DAY_MINUTES,
    GLOBAL_PSEUDO_WEIGHT,
    LONG_ABSENCE_MINUTES,
    EpisodeKey,
    minute_of_day,
)

# Grid of minutes from now: 5-min steps for a day, 30-min steps up to 14 days.
GRID: tuple[float, ...] = tuple(
    [5.0 * i for i in range(int(DAY_MINUTES / 5) + 1)]
    + [DAY_MINUTES + 30.0 * i for i in range(1, int(13 * DAY_MINUTES / 30) + 1)]
)
MAX_MINUTES = GRID[-1]
# P(back within the next day) when the history has no absence as long as the current
# one: SHORT_DAY_HAZARD up to LONG_ABSENCE_MINUTES (an outing ends the same day), down
# to LONG_DAY_HAZARD at TRIP_MINUTES (then it is a trip of days), linear in between.
SHORT_DAY_HAZARD = 0.9
LONG_DAY_HAZARD = 0.2
TRIP_MINUTES = 24 * 60.0  # away a whole day without such a habit: a trip
ARRIVAL_TOD_SIGMA_MINUTES = 75.0
ARRIVAL_TOD_FLOOR = 0.1  # share of the time-of-day density spread uniformly
SMOOTHING_MIN_MINUTES = 10.0
SMOOTHING_FRACTION = 0.1  # residual duration spread: max(10 min, 10 %)
KINEMATIC_SPREAD_FRACTION = 0.25
KINEMATIC_SPREAD_MIN_MINUTES = 5.0
QUANTILES: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
CDF_POINTS: tuple[int, ...] = (15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 1440)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass(frozen=True)
class ArrivalDistribution:
    """CDF of the minutes until the arrival, on ``GRID`` (``F(0) = 0``)."""

    values: tuple[float, ...]
    kinematic_mass: float = 0.0  # share moved onto the GPS ETA
    min_minutes: float = 0.0  # physical lower bound applied (0 = none)

    def cdf(self, minutes: float) -> float:
        """P(arrival <= now + minutes)."""
        if minutes <= 0:
            return 0.0
        if minutes >= MAX_MINUTES:
            return self.values[-1]
        i = bisect.bisect_right(GRID, minutes)
        x0, x1 = GRID[i - 1], GRID[i]
        y0, y1 = self.values[i - 1], self.values[i]
        return y0 + (y1 - y0) * (minutes - x0) / (x1 - x0)

    def quantile(self, q: float) -> float | None:
        """Minutes until the arrival with probability ``q``; None beyond the grid."""
        if q <= 0:
            return 0.0
        i = bisect.bisect_left(self.values, q)
        if i >= len(self.values):
            return None
        if i == 0:
            return GRID[0]
        x0, x1 = GRID[i - 1], GRID[i]
        y0, y1 = self.values[i - 1], self.values[i]
        return x1 if y1 <= y0 else x0 + (x1 - x0) * (q - y0) / (y1 - y0)

    def quantiles(self, qs: Iterable[float] = QUANTILES) -> dict[float, float | None]:
        return {q: self.quantile(q) for q in qs}

    @property
    def source(self) -> str:
        """``gps`` when the position changed the distribution, else ``habit``."""
        return "gps" if self.kinematic_mass > 0.01 or self.min_minutes > 0 else "habit"


def _monotone(values: Sequence[float]) -> tuple[float, ...]:
    out, top = [], 0.0
    for v in values:
        top = min(max(top, v), 1.0)
        out.append(top)
    out[0] = 0.0
    return tuple(out)


def _residual_cdf(pairs: Sequence[tuple[float, float]]) -> list[float]:
    """Weighted, smoothed CDF of residual durations (``(minutes, weight)``, minutes > 0)."""
    total = sum(w for _, w in pairs)
    acc = [0.0] * len(GRID)
    if total <= 0:
        return acc
    tail = [0.0] * (len(GRID) + 1)  # weight whose CDF is already 1 from index i on
    for r, w in pairs:
        s = max(SMOOTHING_MIN_MINUTES, SMOOTHING_FRACTION * r)
        base = _phi(-r / s)  # truncate at 0 (the arrival is in the future)
        norm = 1.0 - base
        lo = bisect.bisect_left(GRID, max(r - 5 * s, 0.0))
        hi = bisect.bisect_right(GRID, r + 5 * s)
        for i in range(lo, min(hi, len(GRID))):
            acc[i] += w * (_phi((GRID[i] - r) / s) - base) / norm
        tail[min(hi, len(GRID))] += w
    running = 0.0
    for i in range(len(GRID)):
        running += tail[i]
        acc[i] = (acc[i] + running) / total
    return acc


def arrival_tod_density(
    keys: Sequence[EpisodeKey], now: datetime, half_life_days: float
) -> list[float]:
    """Density per minute of the habitual arrival time of day, on 0..1439 (step 5)."""
    decay = math.log(2.0) / (max(half_life_days, 1e-6) * 86400.0)
    now_ts = now.timestamp()
    pts = [
        ((k.tod + k.duration) % DAY_MINUTES, math.exp(-decay * max(now_ts - k.left_ts, 0.0)))
        for k in keys
    ]
    total = sum(w for _, w in pts)
    sigma = ARRIVAL_TOD_SIGMA_MINUTES
    norm = 1.0 / (sigma * math.sqrt(2 * math.pi))
    dens = []
    for j in range(int(DAY_MINUTES / 5)):
        m = 5.0 * j
        k = 0.0
        for a, w in pts:
            diff = abs(m - a) % DAY_MINUTES
            diff = min(diff, DAY_MINUTES - diff)
            if diff < 4 * sigma:
                k += w * norm * math.exp(-0.5 * (diff / sigma) ** 2)
        habit = k / total if total > 0 else 1.0 / DAY_MINUTES
        dens.append((1 - ARRIVAL_TOD_FLOOR) * habit + ARRIVAL_TOD_FLOOR / DAY_MINUTES)
    return dens


def target_day_hazard(minutes_away: float) -> float:
    if minutes_away <= LONG_ABSENCE_MINUTES:
        return SHORT_DAY_HAZARD
    if minutes_away >= TRIP_MINUTES:
        return LONG_DAY_HAZARD
    x = (minutes_away - LONG_ABSENCE_MINUTES) / (TRIP_MINUTES - LONG_ABSENCE_MINUTES)
    return SHORT_DAY_HAZARD + x * (LONG_DAY_HAZARD - SHORT_DAY_HAZARD)


def day_hazard(
    keys: Sequence[EpisodeKey], now: datetime, minutes_away: float, half_life_days: float
) -> float:
    """P(back within the next day | away ``minutes_away``), shrunk to ``target_day_hazard``."""
    decay = math.log(2.0) / (max(half_life_days, 1e-6) * 86400.0)
    now_ts = now.timestamp()
    hits = total = 0.0
    for k in keys:
        if k.duration <= minutes_away:
            continue
        r = math.exp(-decay * max(now_ts - k.left_ts, 0.0))
        total += r
        if k.duration <= minutes_away + DAY_MINUTES:
            hits += r
    target = target_day_hazard(minutes_away)
    p = (hits + GLOBAL_PSEUDO_WEIGHT * target) / (total + GLOBAL_PSEUDO_WEIGHT)
    return min(max(p, 0.01), 0.99)


def _hazard_cdf(
    now: datetime, tz: tzinfo | None, density: Sequence[float], p_day: float
) -> list[float]:
    """CDF with cumulative hazard ``-ln(1 - p_day)`` per day, shaped by ``density``."""
    rate = -math.log(1.0 - p_day)
    start = minute_of_day(to_local(now, tz))
    out = [0.0]
    cum = 0.0
    for i in range(1, len(GRID)):
        a, b = GRID[i - 1], GRID[i]
        # midpoint rule on the time-of-day density (5-min resolution)
        n = max(1, int((b - a) / 5))
        acc = 0.0
        for k in range(n):
            m = (start + a + (k + 0.5) * (b - a) / n) % DAY_MINUTES
            acc += density[int(m // 5) % len(density)]
        cum += rate * acc * (b - a) / n
        out.append(1.0 - math.exp(-cum))
    return out


def habit_arrival(
    episodes: Sequence[EpisodeKey],
    trips: Sequence[EpisodeKey],
    durations: Sequence[tuple[float, float]],
    *,
    now: datetime,
    minutes_away: float,
    half_life_days: float,
    tz: tzinfo | None,
    density: Sequence[float] | None = None,
) -> ArrivalDistribution | None:
    """Arrival distribution from habits alone; None without any past absence.

    ``durations`` are the kernel-weighted total durations of the past absences longer
    than the current one (``PriorResult.durations``); ``density`` the precomputed
    ``arrival_tod_density`` (it changes slowly: callers may cache it).
    """
    pool = (*episodes, *trips)
    if not pool:
        return None
    d = max(minutes_away, 0.0)
    if density is None:
        density = arrival_tod_density(pool, now, half_life_days)
    long_model = _hazard_cdf(now, tz, density, day_hazard(pool, now, d, half_life_days))
    if d > LONG_ABSENCE_MINUTES:
        return ArrivalDistribution(_monotone(long_model))
    pairs = [(dur - d, w) for dur, w in durations if dur > d and w > 0]
    weight = sum(w for _, w in pairs)
    habit = _residual_cdf(pairs)
    share = weight / (weight + GLOBAL_PSEUDO_WEIGHT)
    mixed = [share * h + (1 - share) * g for h, g in zip(habit, long_model, strict=True)]
    return ArrivalDistribution(_monotone(mixed))


def with_evidence(
    dist: ArrivalDistribution,
    *,
    prior: float | None,
    p_bayes: float | None,
    eta_minutes: float | None,
    min_minutes: float | None,
) -> ArrivalDistribution:
    """Move the mass added by a GPS approach onto the kinematic ETA; truncate both parts
    at the physical minimum travel time ``min_minutes``."""
    m0 = max(min_minutes or 0.0, 0.0)
    values = _truncate(list(dist.values), m0)
    alpha = 0.0
    if eta_minutes is not None and prior is not None and p_bayes is not None and p_bayes > prior:
        alpha = min(max((p_bayes - prior) / max(1.0 - prior, 1e-9), 0.0), 1.0)
    if alpha > 0:
        eta = max(eta_minutes, m0, 1.0)
        s = max(KINEMATIC_SPREAD_MIN_MINUTES, KINEMATIC_SPREAD_FRACTION * eta)
        base = _phi((m0 - eta) / s)
        kinematic = [0.0 if x <= m0 else (_phi((x - eta) / s) - base) / (1 - base) for x in GRID]
        values = [(1 - alpha) * v + alpha * k for v, k in zip(values, kinematic, strict=True)]
    return replace(dist, values=_monotone(values), kinematic_mass=alpha, min_minutes=m0)


def _truncate(values: list[float], m0: float) -> list[float]:
    """Condition a CDF on the arrival being later than ``m0`` minutes."""
    if m0 <= 0:
        return values
    f0 = ArrivalDistribution(_monotone(values)).cdf(m0)
    if f0 >= 1.0 - 1e-9:
        return [0.0 if x <= m0 else 1.0 for x in GRID]
    return [0.0 if x <= m0 else (v - f0) / (1.0 - f0) for x, v in zip(GRID, values, strict=True)]


def first_arrival(dists: Iterable[ArrivalDistribution]) -> ArrivalDistribution | None:
    """First of several independent arrivals."""
    items = list(dists)
    if not items:
        return None
    values = [1.0 - math.prod(1.0 - d.values[i] for d in items) for i in range(len(GRID))]
    return ArrivalDistribution(
        _monotone(values),
        kinematic_mass=max(d.kinematic_mass for d in items),
        min_minutes=min(d.min_minutes for d in items),
    )


def at(now: datetime, minutes: float | None) -> datetime | None:
    return now + timedelta(minutes=minutes) if minutes is not None else None
