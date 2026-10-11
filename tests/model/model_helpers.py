"""Shared helpers for the pure model tests (imported after conftest sets sys.path)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from hrp_core import ModelConfig, ModelInput, PositionFix
from hrp_core.synthetic import DEFAULT_TZ

TZ = DEFAULT_TZ
CONFIG = ModelConfig(timezone=TZ)


def local(day: date, minutes: float) -> datetime:
    """Local datetime ``minutes`` after midnight of ``day``."""
    return datetime(day.year, day.month, day.day, tzinfo=TZ) + timedelta(minutes=minutes)


def next_weekday(day: date, weekday: int) -> date:
    while day.weekday() != weekday:
        day += timedelta(days=1)
    return day


def approach_fixes(
    now: datetime,
    distance_now_m: float,
    speed_kmh: float,
    minutes: float = 10.0,
    every: float = 1.5,
) -> list[PositionFix]:
    """Clean fixes of a straight approach (negative speed = moving away)."""
    fixes = []
    for i in range(int(minutes / every), -1, -1):
        t = i * every
        dist = distance_now_m + speed_kmh * 1000.0 / 60.0 * t
        fixes.append(PositionFix(now - timedelta(minutes=t), max(dist, 0.0), 15.0))
    return fixes


def away_input(
    day: date, left_minutes: float, now_minutes: float, fixes: list[PositionFix] | None = None
) -> ModelInput:
    return ModelInput(
        now=local(day, now_minutes),
        is_home=False,
        left_at=local(day, left_minutes),
        fixes=fixes or (),
    )
