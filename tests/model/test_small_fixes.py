"""Departure evidence after a GPS silence, weighted regression, LR decay, departure holiday."""

from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta

import pytest

from hrp_core import KernelBayesReturnModel, ModelConfig, ModelInput, PositionFix
from hrp_core.base import PresenceTransition
from hrp_core.evidence import (
    APPROACH_CLASSES,
    LRCalibrator,
    compute_evidence,
)
from hrp_core.synthetic import fixed_schedule_person, iphone_fixes, workplace_zone

NOW = datetime(2026, 3, 2, 17, 0, tzinfo=UTC)


def fix(minutes_ago: float, distance: float, acc: float = 15.0) -> PositionFix:
    return PositionFix(NOW - timedelta(minutes=minutes_ago), distance, acc)


def test_first_fix_after_a_silence_closer_to_home_is_a_departure() -> None:
    parked = fix(180, 12_000)
    ev = compute_evidence([parked, fix(1, 10_500)], NOW, 10, 250)
    assert ev.departure and ev.approach_class == "approach" and ev.speed_kmh is None
    # two fixes after the silence: their speed is used
    ev = compute_evidence([parked, fix(6, 11_000), fix(1, 7_000)], NOW, 10, 250)
    assert ev.departure and ev.approach_class == "approach_fast" and ev.speed_kmh > 40


def test_no_departure_without_silence_or_displacement() -> None:
    # moved less than 500 m: GPS noise or a walk
    assert compute_evidence([fix(180, 12_000), fix(1, 11_700)], NOW, 10, 250).approach_class is None
    # moving away
    assert compute_evidence([fix(180, 12_000), fix(1, 14_000)], NOW, 10, 250).approach_class is None
    # no silence: the previous fix is recent
    ev = compute_evidence([fix(15, 12_000), fix(1, 10_000)], NOW, 10, 250)
    assert not ev.departure
    # the new fix is too old
    assert compute_evidence([fix(180, 12_000), fix(12, 9_000)], NOW, 10, 250).approach_class is None
    # displacement within the accuracy of the two fixes
    ev = compute_evidence([fix(180, 12_000, 300), fix(1, 11_400, 300)], NOW, 10, 400)
    assert ev.approach_class is None


def _first_approach(tl) -> list[float]:
    out = []
    for ab, start in zip(tl.absences, tl.homeward, strict=False):
        if (ab.returned_at - start).total_seconds() < 12 * 60 or start <= ab.left_at:
            continue
        t = start
        while t < ab.returned_at:
            fixes = tl.recent_fixes(t, timedelta(minutes=30), ab.left_at)
            if compute_evidence(fixes, t, 10, 250).approach_class in APPROACH_CLASSES:
                out.append((t - start).total_seconds() / 60)
                break
            t += timedelta(minutes=1)
    return out


def test_iphone_drive_home_is_seen_within_eight_minutes() -> None:
    people = [
        iphone_fixes(workplace_zone(days=40, seed=4, style="android"), seed=5),
        iphone_fixes(fixed_schedule_person(days=40, seed=6), seed=7),
    ]
    lat = [x for tl in people for x in _first_approach(tl)]
    assert len(lat) > 20
    assert statistics.median(lat) <= 8


def test_regression_is_weighted_by_accuracy() -> None:
    # clean approach at 30 km/h plus one wild, low-accuracy fix
    fixes = [fix(m, 5_000 + 500 * m, 10) for m in (8, 6, 4, 2, 0.5)]
    fixes.insert(2, fix(5, 1_000, 240))
    ev = compute_evidence(fixes, NOW, 10, 250)
    assert ev.speed_kmh == pytest.approx(30, abs=4)


def test_lr_counts_decay_towards_the_defaults() -> None:
    cal = LRCalibrator()
    fresh = LRCalibrator()
    for _ in range(300):
        cal.update("still", returned=True)
    learnt = cal.returned["still"] - fresh.returned["still"]
    assert learnt < 300 * 0.8  # older observations weigh less
    for _ in range(600):
        cal.update("approach", returned=True)
    assert cal.returned["still"] - fresh.returned["still"] < learnt / 10
    assert cal.not_returned["away"] == pytest.approx(fresh.not_returned["away"])


def test_holiday_is_judged_on_the_departure_day() -> None:
    tz = UTC
    model = KernelBayesReturnModel(ModelConfig(timezone=tz))
    start = datetime(2026, 1, 5, 8, 0, tzinfo=tz)  # Monday
    transitions = []
    for i in range(56):
        day = start + timedelta(days=i)
        hours = 3 if day.weekday() >= 5 else 9  # weekends: short outings
        transitions += [
            PresenceTransition(day, False),
            PresenceTransition(day + timedelta(hours=hours), True),
        ]
    model.load_history(transitions)
    left = start + timedelta(days=57)  # a Tuesday that is a holiday
    model.observe_transition(PresenceTransition(left, False), is_holiday=True)
    now = left + timedelta(hours=2)

    def prior(holiday_now: bool) -> float:
        inp = ModelInput(now=now, is_home=False, left_at=left, is_holiday=holiday_now)
        return model.predict(inp).prior_probability

    # the calendar flag of "now" no longer matters: the departure day was a holiday
    assert prior(False) == pytest.approx(prior(True))
    # and it is treated like a rest day (back soon), unlike a plain working Tuesday
    plain = KernelBayesReturnModel(ModelConfig(timezone=tz))
    plain.load_history(transitions)
    plain.observe_transition(PresenceTransition(left, False), is_holiday=False)
    inp = ModelInput(now=now, is_home=False, left_at=left, is_holiday=True)
    assert prior(False) > plain.predict(inp).prior_probability
