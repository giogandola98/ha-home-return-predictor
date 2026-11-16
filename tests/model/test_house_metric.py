"""House-level metric on hand-built cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hrp_core.house_metric import (
    EmptyPeriod,
    ThermalParams,
    empty_periods,
    fixed_schedule,
    house_probability,
    never_setback,
    oracle,
    score_period,
    score_person,
    score_policy,
    step_value,
    threshold_policy,
)

T0 = datetime(2026, 3, 2, 8, 0, tzinfo=UTC)
PARAMS = ThermalParams()  # 0.55 K/h, tau 50 h, dT 12 K


def h(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


def test_thermal_model() -> None:
    assert PARAMS.cooling_k(0) == 0
    assert PARAMS.cooling_k(50) == pytest.approx(12 * (1 - 1 / 2.718281828), rel=1e-6)
    assert PARAMS.need_hours(10) == pytest.approx(PARAMS.cooling_k(10) / 0.55)


def test_full_deficit_without_preheating() -> None:
    period = EmptyPeriod(h(0), h(10))
    s = score_period(period, lambda t: False)
    assert s.deficit_k == pytest.approx(PARAMS.cooling_k(10))
    assert s.lead_minutes == 0 and s.early_hours == 0 and s.on_hours == 0


def test_oracle_has_no_deficit_and_no_waste() -> None:
    periods = [EmptyPeriod(h(0), h(10)), EmptyPeriod(h(30), h(34)), EmptyPeriod(h(50), h(130))]
    score = score_policy(periods, oracle(periods))
    for p in score.periods:
        assert p.deficit_k == pytest.approx(0.0, abs=0.02)  # one 2-min sampling step
        assert p.early_hours == 0
        assert p.lead_minutes == pytest.approx(p.need_hours * 60, abs=2.0)


def test_never_setback_wastes_everything_before_the_window() -> None:
    period = EmptyPeriod(h(0), h(10))
    s = score_period(period, never_setback)
    assert s.deficit_k == 0
    assert s.on_hours == pytest.approx(10)
    assert s.early_hours == pytest.approx(10 - s.need_hours, abs=2 / 60)


def test_only_the_last_uninterrupted_run_counts_as_lead() -> None:
    period = EmptyPeriod(h(0), h(10))
    # on 1-2 h (too early, interrupted), then on for the last 60 min
    s = score_period(period, lambda t: h(1) <= t < h(2) or t >= h(9))
    assert s.lead_minutes == pytest.approx(60)
    assert s.deficit_k == pytest.approx(PARAMS.cooling_k(10) - 0.55)
    assert s.early_hours == pytest.approx(1.0)  # the 1-2 h run is before the window
    assert s.on_hours == pytest.approx(2.0)


def test_fixed_schedule_wraps_midnight() -> None:
    tz = UTC
    evening = fixed_schedule(17 * 60, 23 * 60, tz)
    night = fixed_schedule(22 * 60, 6 * 60, tz)
    assert evening(T0.replace(hour=18)) and not evening(T0.replace(hour=23))
    assert night(T0.replace(hour=23)) and night(T0.replace(hour=5)) and not night(T0)


def test_empty_periods_from_two_persons() -> None:
    presence = {
        "a": [(h(0), True), (h(1), False), (h(9), True), (h(20), True)],
        "b": [(h(0), True), (h(2), False), (h(8), True), (h(8.5), False), (h(12), True)],
    }
    periods = empty_periods(presence)
    assert [(p.start, p.end, p.arrivals) for p in periods] == [
        (h(2), h(8), ("b",)),
        (h(8.5), h(9), ("a",)),
    ]
    assert empty_periods(presence, min_hours=1) == periods[:1]


def test_house_probability_counts_only_persons_away() -> None:
    probs = {"a": step_value([(h(0), 0.5)], 0.0), "b": step_value([(h(0), 0.5)], 0.0)}
    home = {"a": lambda t: False, "b": lambda t: t >= h(1)}
    p = house_probability(probs, home)
    assert p(h(0.5)) == pytest.approx(0.75)
    assert p(h(2)) == pytest.approx(0.5)
    assert threshold_policy(p, 0.7)(h(0.5)) and not threshold_policy(p, 0.7)(h(2))


def test_person_metrics() -> None:
    absences = [(h(0), h(10)), (h(24), h(30)), (h(48), h(48.1))]  # last one too short
    # absence 1: on at 4-5 h (false alarm) and for the last 30 min; absence 2: never on
    signal = lambda t: h(4) <= t < h(5) or h(9.5) <= t < h(10)  # noqa: E731
    score = score_person(absences, signal, days=2.0)
    assert len(score.absences) == 2
    first, second = score.absences
    assert first.sustained_lead_minutes == pytest.approx(30) and first.false_alarms == 1
    assert second.sustained_lead_minutes == 0 and second.false_alarms == 0
    assert score.missed == 1
    assert score.false_alarms_per_day == pytest.approx(0.5)
    assert score.median_lead_minutes == pytest.approx(15)
