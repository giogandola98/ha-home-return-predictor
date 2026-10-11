"""Kernel survival prior: circular kernel, fixed schedule, shifts, habit change."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from model_helpers import CONFIG, away_input, local, next_weekday

from hrp_core import KernelBayesReturnModel
from hrp_core.prior import (
    circular_minutes_diff,
    day_weight,
    episode_keys,
    estimate_prior,
    tod_kernel,
    weighted_quantile,
)
from hrp_core.synthetic import DEFAULT_START, fixed_schedule_person, rotating_shift_person


def test_circular_kernel_wraps_midnight() -> None:
    late, early = 23 * 60 + 50, 10
    assert circular_minutes_diff(late, early) == pytest.approx(20)
    assert tod_kernel(late, early) == pytest.approx(tod_kernel(600, 620))
    assert tod_kernel(late, early) > 0.95
    assert tod_kernel(late, 12 * 60) < 1e-6


def test_day_weight_classes() -> None:
    assert day_weight(0, False, 0, False) == 1.0
    assert day_weight(0, False, 3, False) == 0.5
    assert day_weight(5, False, 2, True) == 0.5  # Saturday vs holiday Wednesday
    assert day_weight(0, False, 6, False) == 0.15


def test_weighted_quantile() -> None:
    assert weighted_quantile([(1, 1), (2, 1), (10, 2)], 0.5) == 2
    assert weighted_quantile([], 0.5) is None


def test_no_data_falls_back() -> None:
    day = date(2026, 3, 2)
    p = estimate_prior(
        [],
        [],
        now=local(day, 600),
        left_at=local(day, 500),
        minutes_away=100,
        horizon_minutes=90,
        weekday=0,
        is_holiday=False,
        half_life_days=28,
        tz=CONFIG.timezone,
    )
    assert p.probability == pytest.approx(0.3)
    assert p.eta_duration_minutes is None and p.effective_samples == 0


@pytest.fixture(scope="module")
def fixed_model() -> KernelBayesReturnModel:
    tl = fixed_schedule_person(days=70, seed=3)
    model = KernelBayesReturnModel(CONFIG)
    model.load_history(tl.transitions, tl.fixes)
    return model


def test_fixed_schedule_prior_is_sharp(fixed_model: KernelBayesReturnModel) -> None:
    day = next_weekday(DEFAULT_START + timedelta(days=71), 0)
    morning = fixed_model.predict(away_input(day, 465, 11 * 60))
    evening = fixed_model.predict(away_input(day, 465, 17 * 60))
    assert morning.prior_probability is not None and morning.prior_probability < 0.05
    assert evening.prior_probability is not None and evening.prior_probability > 0.85
    assert evening.prior_effective_samples > 5
    assert evening.eta_habit is not None
    assert abs((evening.eta_habit - local(day, 17 * 60 + 45)).total_seconds()) < 20 * 60
    assert evening.reason["shift_label"] == "morning"
    assert evening.reason["left_at_local"] == "07:45"


def test_unknown_departure_uses_flat_time_weight(fixed_model: KernelBayesReturnModel) -> None:
    day = next_weekday(DEFAULT_START + timedelta(days=71), 0)
    inp = away_input(day, 465, 17 * 60)
    pred = fixed_model.predict(type(inp)(now=inp.now, is_home=False, left_at=None))
    assert 0.0 < pred.probability < 1.0 and pred.eta_habit is None


@pytest.fixture(scope="module")
def shift_model() -> KernelBayesReturnModel:
    tl = rotating_shift_person(days=90, seed=2)
    model = KernelBayesReturnModel(CONFIG)
    model.load_history(tl.transitions, tl.fixes)
    return model


@pytest.mark.parametrize("offset", [92, 95, 101])
def test_shift_learnt_from_departure_time(shift_model: KernelBayesReturnModel, offset: int) -> None:
    day = DEFAULT_START + timedelta(days=offset)
    # Left 05:30 -> morning shift, back ~14:30; left 13:30 -> afternoon shift, back ~22:30.
    morning_at_1340 = shift_model.predict(away_input(day, 330, 820)).probability
    morning_at_2140 = shift_model.predict(away_input(day, 330, 1300)).probability
    afternoon_at_1340 = shift_model.predict(away_input(day, 810, 820)).probability
    afternoon_at_2140 = shift_model.predict(away_input(day, 810, 1300)).probability
    assert morning_at_1340 > 0.7 and afternoon_at_2140 > 0.7
    assert morning_at_2140 < 0.2 and afternoon_at_1340 < 0.3
    pred = shift_model.predict(away_input(day, 330, 600))
    assert pred.eta_habit is not None
    assert abs((pred.eta_habit - local(day, 14 * 60 + 30)).total_seconds()) < 30 * 60


def test_habit_change_absorbed_by_decay() -> None:
    def build(days: int) -> tuple[KernelBayesReturnModel, date]:
        tl = fixed_schedule_person(days=days, seed=5, change_day=56)
        model = KernelBayesReturnModel(CONFIG)
        model.load_history(tl.transitions)
        return model, next_weekday(DEFAULT_START + timedelta(days=days), 0)

    before, day0 = build(56)
    assert before.predict(away_input(day0, 465, 735)).probability < 0.1
    after, day1 = build(56 + 28)
    p_new = after.predict(away_input(day1, 465, 735)).probability  # covers 13:00
    assert p_new > 0.45
    eta = after.predict(away_input(day1, 465, 510)).eta_habit
    assert eta is not None
    new_dt, old_dt = local(day1, 13 * 60), local(day1, 17 * 60 + 45)
    assert abs((eta - new_dt).total_seconds()) < abs((eta - old_dt).total_seconds())
    assert abs((eta - new_dt).total_seconds()) < 45 * 60


def test_episode_keys_roundtrip_values(fixed_model: KernelBayesReturnModel) -> None:
    keys = episode_keys(fixed_model.episodes()[:3], CONFIG.timezone)
    assert all(k.duration >= 10 for k in keys)
