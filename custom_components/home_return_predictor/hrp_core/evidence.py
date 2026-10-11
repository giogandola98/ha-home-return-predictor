"""GPS evidence: distance, approach speed regression and self-calibrated LR."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .base import PositionFix
from .episodes import minutes_between

EARTH_RADIUS_M = 6_371_008.8
CLASS_NAMES: tuple[str, ...] = (
    "away_fast",
    "away",
    "still",
    "approach_slow",
    "approach",
    "approach_fast",
)
# Upper speed bound (km/h, positive = toward home) of each class, in order.
CLASS_UPPER_KMH: tuple[float, ...] = (-15.0, -3.0, 3.0, 15.0, 40.0, math.inf)
DEFAULT_LR: dict[str, float] = {
    "away_fast": 0.25,
    "away": 0.5,
    "still": 1.0,
    "approach_slow": 1.8,
    "approach": 4.0,
    "approach_fast": 7.0,
}
LR_MIN = 0.05
LR_MAX = 50.0
CALIBRATION_PRIOR_STRENGTH = 8.0
MIN_FIXES = 3
MIN_SPAN_MINUTES = 2.0
DISTANCE_MAX_AGE_FACTOR = 3.0  # distance valid if last fix within 3 x window


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def classify_speed(speed_kmh: float) -> str:
    """Map approach speed (km/h, >0 toward home) to a class name."""
    for name, upper in zip(CLASS_NAMES, CLASS_UPPER_KMH, strict=True):
        if speed_kmh < upper:
            return name
    return CLASS_NAMES[-1]


@dataclass(frozen=True)
class Evidence:
    """Summary of recent GPS fixes."""

    speed_kmh: float | None
    approach_class: str | None
    distance_m: float | None
    n_fixes: int


def compute_evidence(
    fixes: Sequence[PositionFix],
    now: datetime,
    window_minutes: float,
    max_accuracy_m: float,
) -> Evidence:
    """OLS slope of distance vs time over the window -> approach speed and class."""
    valid = [
        f
        for f in fixes
        if (f.accuracy_m is None or f.accuracy_m <= max_accuracy_m)
        and minutes_between(f.ts, now) >= -1e-9
    ]
    if not valid:
        return Evidence(None, None, None, 0)
    valid.sort(key=lambda f: f.ts)
    last = valid[-1]
    distance = (
        last.distance_m
        if minutes_between(last.ts, now) <= DISTANCE_MAX_AGE_FACTOR * window_minutes
        else None
    )
    recent = [f for f in valid if minutes_between(f.ts, now) <= window_minutes]
    if len(recent) < MIN_FIXES:
        return Evidence(None, None, distance, len(recent))
    xs = [minutes_between(recent[0].ts, f.ts) for f in recent]
    if xs[-1] - xs[0] < MIN_SPAN_MINUTES:
        return Evidence(None, None, distance, len(recent))
    ys = [f.distance_m for f in recent]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope_m_per_min = sxy / sxx
    speed = -slope_m_per_min * 60.0 / 1000.0
    return Evidence(speed, classify_speed(speed), distance, len(recent))


def _solve_scale(lrs: Sequence[float]) -> float:
    """Find k so per-class shares s_c = k*LR/(1+k*LR) reproduce LR after normalisation."""

    def gap(k: float) -> float:
        s = [k * lr / (1 + k * lr) for lr in lrs]
        return k * sum(1 - x for x in s) / sum(s) - 1.0

    lo, hi = 1e-3, 1e3
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if gap(mid) < 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


class LRCalibrator:
    """Per-class pseudo-counts for P(class | return) and P(class | no return)."""

    def __init__(self) -> None:
        self.returned: dict[str, float] = {}
        self.not_returned: dict[str, float] = {}
        self.reset()

    def reset(self) -> None:
        """Restore pseudo-counts that encode DEFAULT_LR exactly."""
        lrs = [DEFAULT_LR[c] for c in CLASS_NAMES]
        k = _solve_scale(lrs)
        for name, lr in zip(CLASS_NAMES, lrs, strict=True):
            share = k * lr / (1 + k * lr)
            self.returned[name] = CALIBRATION_PRIOR_STRENGTH * share
            self.not_returned[name] = CALIBRATION_PRIOR_STRENGTH * (1 - share)

    def likelihood_ratio(self, approach_class: str | None) -> float:
        """LR = P(c | R) / P(c | not R), clamped; 1.0 when no class."""
        if approach_class is None or approach_class not in self.returned:
            return 1.0
        p_r = self.returned[approach_class] / sum(self.returned.values())
        p_n = self.not_returned[approach_class] / sum(self.not_returned.values())
        return min(max(p_r / p_n, LR_MIN), LR_MAX)

    def update(self, approach_class: str, returned: bool) -> None:
        """Add one observation of ``approach_class`` with its outcome."""
        if approach_class not in self.returned:
            return
        target = self.returned if returned else self.not_returned
        target[approach_class] += 1.0

    def has_observations(self) -> bool:
        """True once any count differs from the default pseudo-counts."""
        fresh = LRCalibrator()
        return any(
            abs(self.returned[c] - fresh.returned[c]) > 1e-9
            or abs(self.not_returned[c] - fresh.not_returned[c]) > 1e-9
            for c in fresh.returned
        )

    def to_dict(self) -> dict[str, Any]:
        return {"returned": dict(self.returned), "not_returned": dict(self.not_returned)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LRCalibrator:
        cal = cls()
        for name in CLASS_NAMES:
            cal.returned[name] = float(data.get("returned", {}).get(name, cal.returned[name]))
            cal.not_returned[name] = float(
                data.get("not_returned", {}).get(name, cal.not_returned[name])
            )
        return cal


APPROACH_CLASSES: frozenset[str] = frozenset({"approach_slow", "approach", "approach_fast"})
SEGMENT_RESET_MINUTES = 10.0


class CalibrationSegments:
    """Decides when an approach class yields a GPS calibration observation.

    Approach classes count once per *approach segment*: a segment ends when the
    person has been in a non-approach class (still / away) for >= 10 minutes, or a
    full horizon after the class was counted. A drive-by followed by the real
    return in the same absence therefore yields one negative and one positive
    observation. Non-approach classes (still / away / away_fast) keep counting
    once per absence, so long stays do not flood the counts with negatives.
    """

    def __init__(self) -> None:
        self.absence_key: str | None = None
        self.seen: dict[str, datetime] = {}
        self.calm_since: datetime | None = None

    def observe(
        self, ts: datetime, approach_class: str | None, absence_key: str | None, horizon: float
    ) -> bool:
        """Register one observation; True if it should feed the calibrator."""
        if absence_key != self.absence_key:
            self.absence_key, self.seen, self.calm_since = absence_key, {}, None
        if approach_class is None:
            return False
        if (
            self.calm_since is not None
            and minutes_between(self.calm_since, ts) >= SEGMENT_RESET_MINUTES
        ):
            self.seen = {c: t for c, t in self.seen.items() if c not in APPROACH_CLASSES}
        if approach_class in APPROACH_CLASSES:
            self.calm_since = None
        elif self.calm_since is None:
            self.calm_since = ts
        self.seen = {
            c: t
            for c, t in self.seen.items()
            if c not in APPROACH_CLASSES or minutes_between(t, ts) < horizon
        }
        if approach_class in self.seen:
            return False
        self.seen[approach_class] = ts
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.absence_key,
            "seen": {c: t.isoformat() for c, t in self.seen.items()},
            "calm": self.calm_since.isoformat() if self.calm_since else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationSegments:
        seg = cls()
        seg.absence_key = data.get("key")
        seg.seen = {c: datetime.fromisoformat(t) for c, t in (data.get("seen") or {}).items()}
        calm = data.get("calm")
        seg.calm_since = datetime.fromisoformat(calm) if calm else None
        return seg
