"""Model lifecycle: purity, persistence, logging, at-home behaviour."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from model_helpers import CONFIG, approach_fixes, away_input, local, next_weekday

from hrp_core import (
    KernelBayesReturnModel,
    ModelInput,
    PresenceTransition,
    ReturnModel,
    create_model,
)
from hrp_core.synthetic import (
    DEFAULT_START,
    fixed_schedule_person,
    replay_online,
    rotating_shift_person,
)


@pytest.fixture(scope="module")
def online_model() -> KernelBayesReturnModel:
    tl = rotating_shift_person(days=45, seed=4)
    model = KernelBayesReturnModel(CONFIG)
    records = list(replay_online(model, tl))
    assert records
    # Leave the model mid-absence so pending entries and state are non-trivial.
    last = tl.absences[-1].returned_at + timedelta(hours=2)
    model.observe_transition(PresenceTransition(last, False))
    inp = ModelInput(
        now=last + timedelta(minutes=20),
        is_home=False,
        left_at=last,
        fixes=approach_fixes(last + timedelta(minutes=20), 3000, -40),
    )
    model.log_prediction(inp, model.predict(inp))
    return model


def test_factory_and_contract() -> None:
    model = create_model(CONFIG)
    assert isinstance(model, ReturnModel) and isinstance(model, KernelBayesReturnModel)


def _inputs() -> list[ModelInput]:
    day = next_weekday(DEFAULT_START + timedelta(days=50), 2)
    now = local(day, 14 * 60)
    return [
        away_input(day, 330, 820),
        away_input(day, 810, 1300),
        away_input(day, 465, 14 * 60, approach_fixes(now, 9000, 45)),
        ModelInput(now=now, is_home=False, left_at=None),
    ]


def test_json_round_trip_gives_identical_predictions(online_model: KernelBayesReturnModel) -> None:
    payload = json.dumps(online_model.to_dict())
    restored = KernelBayesReturnModel.from_dict(json.loads(payload), CONFIG)
    assert json.dumps(restored.to_dict()) == payload
    for inp in _inputs():
        assert restored.predict(inp) == online_model.predict(inp)
    now = local(next_weekday(DEFAULT_START + timedelta(days=50), 2), 0)
    assert restored.maturity(now) == online_model.maturity(now)


def test_from_empty_dict() -> None:
    model = KernelBayesReturnModel.from_dict({}, CONFIG)
    assert model.episodes() == []


def test_predict_does_not_mutate(online_model: KernelBayesReturnModel) -> None:
    before = json.dumps(online_model.to_dict())
    for inp in _inputs():
        online_model.predict(inp)
    assert json.dumps(online_model.to_dict()) == before


def test_at_home_prediction() -> None:
    model = KernelBayesReturnModel(CONFIG)
    pred = model.predict(ModelInput(now=local(DEFAULT_START, 600), is_home=True, left_at=None))
    assert pred.probability == 1.0 and pred.prior_probability is None and pred.eta is None
    assert pred.is_home
    assert (
        model.log_prediction(
            ModelInput(now=local(DEFAULT_START, 600), is_home=True, left_at=None), pred
        )
        is False
    )


def test_log_rate_limit_and_resolution() -> None:
    model = KernelBayesReturnModel(CONFIG)
    day = DEFAULT_START
    model.observe_transition(PresenceTransition(local(day, 480), False))
    logged = []
    for minute in (490, 495, 499, 500, 505, 512, 520):
        inp = away_input(day, 480, minute)
        logged.append(model.log_prediction(inp, model.predict(inp)))
    assert logged == [True, False, False, True, False, True, False]
    assert model.resolve(local(day, 560), False) == 0  # horizons not elapsed yet
    assert model.resolve(local(day, 585), False) == 1  # 490 + 90 elapsed -> no return
    model.observe_transition(PresenceTransition(local(day, 595), True))
    report = model.maturity(local(day, 600))
    assert report.resolved_predictions == 3 and report.episodes == 1
    assert report.maturity == 0.0  # fewer than 30 resolved predictions


def test_reason_keys(online_model: KernelBayesReturnModel) -> None:
    pred = online_model.predict(_inputs()[2])
    for key in (
        "left_at_local",
        "shift_label",
        "minutes_away",
        "prior",
        "approach_class",
        "approach_speed_kmh",
        "distance_km",
        "lr",
        "eta_source",
        "fusion",
    ):
        assert key in pred.reason
    json.dumps(pred.reason)


def test_reset_and_reload() -> None:
    tl = fixed_schedule_person(days=20, seed=9)
    model = KernelBayesReturnModel(CONFIG)
    model.load_history(tl.transitions, tl.fixes)
    assert len(model.episodes()) > 15
    model.reset()
    fresh = KernelBayesReturnModel(CONFIG).to_dict()
    # Only the generation counter (stale challenger fits) survives a reset.
    assert model.episodes() == [] and model.to_dict() == {**fresh, "generation": 1}
