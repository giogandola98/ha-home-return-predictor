"""GPS evidence, calibrated LR, fusion and ETA."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest
from model_helpers import CONFIG, approach_fixes, away_input, local, next_weekday

from hrp_core import (
    KernelBayesReturnModel,
    ModelInput,
    PositionFix,
    PresenceTransition,
    haversine_m,
)
from hrp_core.evidence import DEFAULT_LR, LRCalibrator, classify_speed, compute_evidence
from hrp_core.fusion import combine_eta, fuse_probability, position_eta_minutes
from hrp_core.synthetic import DEFAULT_START, fixed_schedule_person, replay_online

DAY = date(2026, 3, 2)


def test_haversine() -> None:
    assert haversine_m(45.0, 9.0, 45.0, 9.0) == 0
    assert haversine_m(45.0, 9.0, 46.0, 9.0) == pytest.approx(111_195, rel=1e-3)


def test_speed_regression_and_classes() -> None:
    now = local(DAY, 600)
    ev = compute_evidence(approach_fixes(now, 5000, 50), now, 10, 250)
    assert ev.speed_kmh == pytest.approx(50, abs=0.5)
    assert ev.approach_class == "approach_fast" and ev.distance_m == pytest.approx(5000)
    away = compute_evidence(approach_fixes(now, 5000, -30), now, 10, 250)
    assert away.approach_class == "away_fast"
    assert [classify_speed(s) for s in (-20, -5, 0, 10, 30, 50)] == [
        "away_fast",
        "away",
        "still",
        "approach_slow",
        "approach",
        "approach_fast",
    ]


def test_insufficient_or_inaccurate_fixes_are_neutral() -> None:
    now = local(DAY, 600)
    two = approach_fixes(now, 5000, 50, minutes=1.5)
    assert compute_evidence(two, now, 10, 250).approach_class is None
    bad = [PositionFix(f.ts, f.distance_m, 900.0) for f in approach_fixes(now, 5000, 50)]
    ev = compute_evidence(bad, now, 10, 250)
    assert ev.approach_class is None and ev.distance_m is None
    model = KernelBayesReturnModel(CONFIG)
    pred = model.predict(ModelInput(now=now, is_home=False, left_at=local(DAY, 500), fixes=bad))
    assert pred.likelihood_ratio == 1.0 and pred.approach_class is None


def test_calibrator_prior_encodes_default_lrs() -> None:
    cal = LRCalibrator()
    for name, lr in DEFAULT_LR.items():
        assert cal.likelihood_ratio(name) == pytest.approx(lr, rel=1e-6)
        assert cal.returned[name] + cal.not_returned[name] == pytest.approx(8.0)
    assert cal.likelihood_ratio(None) == 1.0
    for _ in range(500):
        cal.update("approach_fast", True)
        cal.update("away_fast", False)
    assert cal.likelihood_ratio("approach_fast") == 20.0  # clamped


def test_fusion_math() -> None:
    assert fuse_probability(0.3, 1.0) == pytest.approx(0.3)
    assert fuse_probability(0.5, 4.0) == pytest.approx(0.8)
    assert fuse_probability(0.999, 20) == 0.99 and fuse_probability(0.0, 0.05) == 0.01


def test_eta_combination() -> None:
    assert position_eta_minutes(10_000, 50, 90) == pytest.approx(12)
    assert position_eta_minutes(10_000, 4, 90) is None
    assert position_eta_minutes(400_000, 50, 90) is None  # beyond 3 horizons
    assert combine_eta(None, None, None) == (None, None)
    assert combine_eta(30, 20, None) == (30, "habit")
    assert combine_eta(None, None, 12) == (12, "position")
    eta, src = combine_eta(40, 20, 12)
    assert src == "fused" and 12 < eta < 40


@pytest.fixture(scope="module")
def trained() -> KernelBayesReturnModel:
    tl = fixed_schedule_person(days=56, seed=7)
    model = KernelBayesReturnModel(CONFIG)
    for _ in replay_online(model, tl):
        pass
    return model


def test_approach_moves_flat_prior_more_than_sharp(trained: KernelBayesReturnModel) -> None:
    day = next_weekday(DEFAULT_START + timedelta(days=57), 1)
    fresh = KernelBayesReturnModel(CONFIG)  # no data: flat prior 0.3
    now = local(day, 17 * 60 + 15)
    fixes = approach_fixes(now, 8000, 50)
    flat0 = fresh.predict(away_input(day, 465, 17 * 60 + 15)).probability
    flat1 = fresh.predict(away_input(day, 465, 17 * 60 + 15, fixes)).probability
    sharp0 = trained.predict(away_input(day, 465, 17 * 60 + 15)).probability
    sharp1 = trained.predict(away_input(day, 465, 17 * 60 + 15, fixes)).probability
    assert sharp0 > 0.85
    assert flat1 - flat0 > 0.3
    assert 0 <= sharp1 - sharp0 < 0.1
    assert flat1 - flat0 > 3 * (sharp1 - sharp0)


def test_position_and_fused_eta(trained: KernelBayesReturnModel) -> None:
    day = next_weekday(DEFAULT_START + timedelta(days=57), 1)
    now = local(day, 17 * 60)
    fresh = KernelBayesReturnModel(CONFIG)
    pred = fresh.predict(away_input(day, 465, 17 * 60, approach_fixes(now, 10_000, 50)))
    assert pred.eta_position is not None and pred.eta_habit is None
    assert (pred.eta_position - now).total_seconds() / 60 == pytest.approx(12, abs=1)
    assert pred.reason["eta_source"] == "position" and pred.eta == pred.eta_position

    pred = trained.predict(away_input(day, 465, 17 * 60, approach_fixes(now, 20_000, 50)))
    assert pred.eta_habit is not None and pred.eta_position is not None and pred.eta
    lo, hi = sorted([pred.eta_habit, pred.eta_position])
    assert lo < pred.eta < hi and pred.reason["eta_source"] == "fused"
    assert pred.reason["distance_km"] == pytest.approx(20, abs=0.1)
    assert pred.reason["approach_speed_kmh"] == pytest.approx(50, abs=1)


def test_drive_bys_lower_approach_lr(trained: KernelBayesReturnModel) -> None:
    model = KernelBayesReturnModel.from_dict(trained.to_dict(), CONFIG)
    start = next_weekday(DEFAULT_START + timedelta(days=57), 0)
    lrs = [model.calibrator.likelihood_ratio("approach_fast")]
    for i in range(10):
        day = start + timedelta(days=i)
        model.observe_transition(PresenceTransition(local(day, 465), False))
        now = local(day, 11 * 60)
        inp = away_input(day, 465, 11 * 60, approach_fixes(now, 400, 50))
        pred = model.predict(inp)
        assert pred.approach_class == "approach_fast"
        assert pred.probability < 0.6  # low prior: no strong false positive
        model.log_prediction(inp, pred)
        model.resolve(local(day, 13 * 60), False)  # horizon passed, still away
        model.observe_transition(PresenceTransition(local(day, 17 * 60 + 45), True))
        lrs.append(model.calibrator.likelihood_ratio("approach_fast"))
    assert all(b < a for a, b in zip(lrs, lrs[1:], strict=False))
    assert lrs[-1] < 0.6 * lrs[0]


def test_drive_by_in_simulation_stays_below_threshold() -> None:
    tl = fixed_schedule_person(days=84, seed=11, drive_by_rate=0.3)
    model = KernelBayesReturnModel(CONFIG)
    drive_bys = [a.drive_by_at for a in tl.absences if a.drive_by_at is not None]
    warm = tl.absences[0].left_at + timedelta(days=28)
    worst = 0.0
    for rec in replay_online(model, tl):
        if rec.ts < warm:
            continue
        for db in drive_bys:
            if db - timedelta(minutes=25) <= rec.ts <= db + timedelta(minutes=3):
                worst = max(worst, rec.prediction.probability)
    assert len(drive_bys) > 10
    assert worst < 0.6


def test_logit_helpers() -> None:
    assert math.isclose(fuse_probability(0.2, math.e), 1 / (1 + 4 / math.e))
