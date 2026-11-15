"""A return that is physically impossible within the horizon must not be predicted.

Regression: a person standing still ~1100 km from home at their usual return time got
P(return within 90 min) = 0.36 and an ETA a couple of hours away, because without movement
the GPS likelihood ratio is neutral and the distance itself was never used.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest
from model_helpers import CONFIG, away_input, local, next_weekday

from hrp_core import KernelBayesReturnModel, PositionFix
from hrp_core.fusion import (
    MAX_TRAVEL_SPEED_KMH,
    P_MIN,
    apply_feasibility,
    min_travel_minutes,
)
from hrp_core.synthetic import DEFAULT_START, fixed_schedule_person, replay_online

FAR_M = 1_102_970.0  # the real case: ~1103 km from home


def stationary_fixes(now: datetime, distance_m: float, minutes: float = 10.0) -> list[PositionFix]:
    """Fresh, accurate fixes of someone who is not moving."""
    return [
        PositionFix(now - timedelta(minutes=t), distance_m, 15.0)
        for t in (minutes, minutes * 0.66, minutes * 0.33, 0.0)
    ]


@pytest.fixture(scope="module")
def trained() -> KernelBayesReturnModel:
    tl = fixed_schedule_person(days=56, seed=7)
    model = KernelBayesReturnModel(CONFIG)
    for _ in replay_online(model, tl):
        pass
    return model


def test_min_travel_minutes() -> None:
    assert min_travel_minutes(None) is None
    assert min_travel_minutes(0.0) == 0.0
    assert min_travel_minutes(FAR_M) == pytest.approx(FAR_M / 1000 / MAX_TRAVEL_SPEED_KMH * 60)


def test_apply_feasibility_rules() -> None:
    # unknown distance or reachable within the horizon: unchanged
    assert apply_feasibility(0.6, 40.0, None, 90) == (0.6, 40.0, True)
    assert apply_feasibility(0.6, 40.0, 20_000.0, 90) == (0.6, 40.0, True)
    # unreachable within the horizon: minimum probability, ETA not before the minimum travel time
    p, eta, feasible = apply_feasibility(0.6, 40.0, FAR_M, 90)
    assert (p, feasible) == (P_MIN, False)
    assert eta == pytest.approx(min_travel_minutes(FAR_M))
    # an ETA already later than the minimum travel time is kept; no ETA stays None
    assert apply_feasibility(0.6, 900.0, FAR_M, 90)[1] == 900.0
    assert apply_feasibility(0.6, None, FAR_M, 90)[1] is None


def test_far_away_at_usual_return_time_is_not_returning(trained: KernelBayesReturnModel) -> None:
    day = next_weekday(DEFAULT_START + timedelta(days=57), 1)
    now = local(day, 17 * 60 + 15)  # the synthetic person's usual return time
    near = trained.predict(away_input(day, 465, 17 * 60 + 15, stationary_fixes(now, 3_000.0)))
    far = trained.predict(away_input(day, 465, 17 * 60 + 15, stationary_fixes(now, FAR_M)))

    assert near.probability > 0.5  # habit says: coming home now
    assert far.probability == P_MIN  # but 1100 km cannot be covered in 90 minutes
    assert far.eta is not None
    assert (far.eta - now).total_seconds() / 60 >= min_travel_minutes(FAR_M) - 1e-6
    assert far.reason["feasible"] is False
    assert far.reason["min_travel_minutes"] == pytest.approx(min_travel_minutes(FAR_M), abs=0.1)
    assert near.reason["feasible"] is True


def test_elapsed_time_since_fix_is_credited() -> None:
    minimum = min_travel_minutes(FAR_M)
    # fix 60 min old: still far beyond the horizon
    assert apply_feasibility(0.6, None, FAR_M, 90, elapsed_minutes=60)[0] == P_MIN
    # fix old enough that the person could be almost home by now: no cap
    p, _, feasible = apply_feasibility(0.6, None, FAR_M, 90, elapsed_minutes=minimum - 30)
    assert (p, feasible) == (0.6, True)


def test_stale_last_fix_still_caps_when_phone_is_quiet(trained: KernelBayesReturnModel) -> None:
    """A stationary phone may not report for >30 min: the fresh-fix window is empty."""
    day = next_weekday(DEFAULT_START + timedelta(days=57), 1)
    now = local(day, 17 * 60 + 15)
    base = away_input(day, 465, 17 * 60 + 15)  # no fresh fixes at all
    far_old = PositionFix(now - timedelta(minutes=60), FAR_M, 15.0)
    pred = trained.predict(replace(base, last_fix=far_old))
    assert pred.probability == P_MIN
    assert pred.reason["feasible"] is False
    expected = min_travel_minutes(FAR_M) - 60
    assert pred.reason["min_travel_minutes"] == pytest.approx(expected, abs=0.1)
    # without any position information the habit model is left untouched
    assert trained.predict(base).probability > 0.5
