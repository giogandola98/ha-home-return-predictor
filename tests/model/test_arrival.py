"""Arrival-time distribution per person and for the house."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from model_helpers import approach_fixes

from hrp_core import KernelBayesReturnModel, ModelConfig, ModelInput, PositionFix
from hrp_core.arrival import (
    GRID,
    ArrivalDistribution,
    first_arrival,
    with_evidence,
)
from hrp_core.base import PresenceTransition

TZ = UTC
D0 = datetime(2026, 9, 1, 8, 0, tzinfo=TZ)


def _model(days: int = 60, hours: float = 10.0) -> KernelBayesReturnModel:
    """Every day: out at 08:00, back after ``hours``."""
    model = KernelBayesReturnModel(ModelConfig(timezone=TZ))
    transitions = []
    for i in range(days):
        transitions += [
            PresenceTransition(D0 + timedelta(days=i), False),
            PresenceTransition(D0 + timedelta(days=i, hours=hours), True),
        ]
    model.load_history(transitions)
    return model


def _arrival(model, left: datetime, now: datetime, fixes=(), last_fix=None):
    inp = ModelInput(now=now, is_home=False, left_at=left, fixes=fixes, last_fix=last_fix)
    pred = model.predict(inp)
    return pred, model.arrival(inp, pred)


def test_ten_hour_habit_gives_an_arrival_long_before_the_horizon() -> None:
    model = _model()
    left = D0 + timedelta(days=61)
    pred, dist = _arrival(model, left, left + timedelta(hours=7))
    assert pred.probability < 0.25  # the 90-min probability says nothing yet
    q25 = dist.quantile(0.25)
    assert 150 <= q25 <= 210  # 2.5-3.5 h ahead
    assert dist.quantile(0.5) == pytest.approx(180, abs=20)
    assert dist.source == "habit"
    # always in the future and monotone
    assert dist.cdf(0) == 0 and all(
        a <= b for a, b in zip(dist.values, dist.values[1:], strict=False)
    )


def test_arrival_moves_with_the_time_away() -> None:
    model = _model()
    left = D0 + timedelta(days=61)
    early = _arrival(model, left, left + timedelta(hours=2))[1].quantile(0.5)
    late = _arrival(model, left, left + timedelta(hours=9))[1].quantile(0.5)
    assert early == pytest.approx(480, abs=30) and late == pytest.approx(60, abs=20)


def test_trip_keeps_a_far_arrival_until_gps_evidence() -> None:
    model = _model()
    left = D0 + timedelta(days=61)
    for hours in (13, 24, 48, 90):
        _, dist = _arrival(model, left, left + timedelta(hours=hours))
        assert dist is not None
        q25 = dist.quantile(0.25)
        assert q25 is not None and q25 > 12 * 60, hours
    # approaching on the motorway: the mass the GPS adds over the habit prior (the
    # 90-min probability habit x GPS) moves onto the kinematic ETA
    now = left + timedelta(hours=90)
    pred, near = _arrival(model, left, now, fixes=approach_fixes(now, 20_000, 80, every=1.0))
    _, still = _arrival(model, left, now)
    assert near.source == "gps" and near.quantile(0.1) < 30
    assert still.cdf(60) < 0.05
    assert near.cdf(60) == pytest.approx(pred.reason["probability_bayes"], abs=0.05)


def test_stationary_far_away_cannot_arrive_soon() -> None:
    model = _model()
    left = D0 + timedelta(days=61)
    now = left + timedelta(hours=9, minutes=30)  # habitually back within the hour
    fixes = [PositionFix(now - timedelta(minutes=m), 150_000.0, 20.0) for m in (25, 15, 5)]
    _, dist = _arrival(model, left, now, fixes=fixes)
    assert dist.cdf(45) == 0.0  # 150 km at 90 km/h: at least 100 min
    assert dist.min_minutes == pytest.approx(150 / 90 * 60, abs=1)
    assert dist.quantile(0.1) >= 100


def test_no_history_no_distribution() -> None:
    model = KernelBayesReturnModel(ModelConfig(timezone=TZ))
    now = D0 + timedelta(hours=1)
    assert _arrival(model, D0, now)[1] is None
    at_home = ModelInput(now=now, is_home=True, left_at=None)
    assert model.arrival(at_home, model.predict(at_home)) is None


def _step(at_minutes: float) -> ArrivalDistribution:
    return ArrivalDistribution(tuple(0.0 if x < at_minutes else 1.0 for x in GRID))


def test_first_arrival_of_the_house() -> None:
    a, b = _step(60), _step(300)
    house = first_arrival([a, b])
    assert house.quantile(0.5) <= 60
    assert first_arrival([]) is None
    # independent: P(first <= t) = 1 - (1 - Fa)(1 - Fb)
    half = ArrivalDistribution(tuple(0.0 if x < 100 else 0.5 for x in GRID))
    both = first_arrival([half, half])
    assert both.cdf(200) == pytest.approx(0.75)


def test_evidence_truncation_and_kinematic_mass() -> None:
    base = ArrivalDistribution(tuple(min(x / 600.0, 1.0) for x in GRID))  # uniform 0-10 h
    moved = with_evidence(base, prior=0.1, p_bayes=0.82, eta_minutes=20, min_minutes=None)
    assert moved.kinematic_mass == pytest.approx(0.8)
    assert moved.cdf(40) > 0.75
    cut = with_evidence(base, prior=0.1, p_bayes=0.1, eta_minutes=None, min_minutes=120)
    assert cut.cdf(120) == 0.0 and cut.quantile(0.5) == pytest.approx(360, abs=10)
