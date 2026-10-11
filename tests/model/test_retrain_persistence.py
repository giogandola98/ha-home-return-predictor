"""Section 6 model fixes: merge on retrain (M1), served = evaluated fusion weights (M2),
cheap observe_transition (M3), compact persistence (M4), split challenger training (M5)."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from model_helpers import CONFIG, approach_fixes, away_input, local

from hrp_core import KernelBayesReturnModel, ModelInput, PresenceTransition
from hrp_core import bayes_model as bayes_module
from hrp_core.bayes_model import STACKER_REFIT_EVERY
from hrp_core.maturity import LogEntry, decode_entries, encode_entries
from hrp_core.selection import (
    KERNEL_BAYES,
    LEARNED_FUSION,
    chronological_split,
    recency_weights,
    weighted_brier,
)
from hrp_core.stacker import Stacker, apply_weights
from hrp_core.synthetic import (
    DEFAULT_START,
    fixed_schedule_person,
    replay_online,
    rotating_shift_person,
)

MAX_STATE_BYTES = 400_000


@pytest.fixture(scope="module")
def long_model() -> tuple[KernelBayesReturnModel, object]:
    """120 simulated days (the log is at its 3000-entry cap)."""
    tl = fixed_schedule_person(days=120, seed=1)
    model = KernelBayesReturnModel(CONFIG)
    for _ in replay_online(model, tl, step_minutes=10):
        pass
    return model, tl


def _legacy(state: dict) -> dict:  # type: ignore[type-arg]
    """The same state in the pre-M4 format (log entries as a list of dicts)."""
    old = json.loads(json.dumps(state))
    for key in ("pending", "resolved"):
        old["log"][key] = [e.to_dict() for e in decode_entries(old["log"][key])]
    old["calib_pending"] = [e.to_dict() for e in decode_entries(old["calib_pending"])]
    return old


def _probe_inputs() -> list[ModelInput]:
    day = DEFAULT_START + timedelta(days=121)
    now = local(day, 14 * 60)
    return [
        away_input(day, 330, 820),
        away_input(day, 465, 14 * 60, approach_fixes(now, 9000, 45)),
        ModelInput(now=now, is_home=False, left_at=None),
    ]


# --------------------------------------------------------------------------- M1
def test_retrain_with_recent_history_keeps_learning(long_model) -> None:  # type: ignore[no-untyped-def]
    model, tl = long_model
    model = KernelBayesReturnModel.from_dict(json.loads(json.dumps(model.to_dict())), CONFIG)
    end = tl.absences[-1].returned_at
    before_eps = model.episodes()
    before_maturity = model.maturity(end)
    before_cal = model.calibrator.to_dict()
    before_log = len(model._log.resolved)
    assert before_maturity.maturity > 0 and len(before_eps) > 100

    cutoff = end - timedelta(days=10)
    recent = [t for t in tl.transitions if t.ts >= cutoff]
    fixes = [f for f in tl.fixes if f.ts >= cutoff]
    model.load_history(recent, fixes)
    assert model.episodes() == before_eps  # recorder duplicates merged, old ones kept
    assert model.maturity(end) == before_maturity
    assert model.calibrator.to_dict() == before_cal  # not re-warmed
    assert len(model._log.resolved) == before_log

    # A recorder departure a few minutes off replaces the stored episode (no double).
    last = tl.absences[-1]
    shifted = [
        PresenceTransition(last.left_at + timedelta(minutes=4), False),
        PresenceTransition(last.returned_at, True),
    ]
    model.load_history(shifted)
    eps = model.episodes()
    assert len(eps) == len(before_eps)
    assert eps[-1].left_at == last.left_at + timedelta(minutes=4)

    # replace=True: the old full rebuild from the given (short) history only.
    model.load_history(recent, fixes, replace=True)
    assert len(model.episodes()) < 15
    assert model.maturity(end).maturity < before_maturity.maturity


def test_merge_warms_calibration_only_when_empty() -> None:
    tl = fixed_schedule_person(days=20, seed=3)
    model = KernelBayesReturnModel(CONFIG)
    assert not model.calibrator.has_observations()
    model.load_history(tl.transitions, tl.fixes)
    assert model.calibrator.has_observations()
    warmed = model.calibrator.to_dict()
    model.calibrator.update("approach", True)
    bumped = model.calibrator.to_dict()
    model.load_history(tl.transitions, tl.fixes)
    assert model.calibrator.to_dict() == bumped  # kept, not re-warmed
    model.load_history(tl.transitions, tl.fixes, replace=True)
    assert model.calibrator.to_dict() == warmed


# --------------------------------------------------------------------------- M2
def test_stacker_serves_the_evaluated_weights() -> None:
    tl = rotating_shift_person(days=40, seed=6, drive_by_rate=0.6)
    model = KernelBayesReturnModel(CONFIG)
    for _ in replay_online(model, tl):
        pass
    now = tl.absences[-1].returned_at
    st = Stacker()
    st.fit(model._log.resolved, now, CONFIG.half_life_days)
    assert st.holdout_brier_learned is not None
    usable = [e for e in model._log.resolved if e.outcome is not None and e.features]
    _, hold = chronological_split(usable[-1500:])
    w = recency_weights([e.ts for e in hold], now, CONFIG.half_life_days)
    served = weighted_brier(
        [apply_weights(st.weights, e.features) for e in hold],
        [int(e.outcome or 0) for e in hold],
        w,
    )
    assert served == pytest.approx(st.holdout_brier_learned, rel=1e-9)


def test_challenger_installs_the_evaluated_fusion_weights() -> None:
    pytest.importorskip("numpy")
    tl = rotating_shift_person(days=35, seed=5)
    model = KernelBayesReturnModel(CONFIG)
    model.enable_challenger(True)
    for _ in replay_online(model, tl):
        pass
    now = tl.absences[-1].returned_at
    summary = model.train_challenger(now)
    assert summary["trained"]
    rows = model._challenger_rows()
    _, hold = chronological_split(rows)
    w = recency_weights([e.ts for e in hold], now, CONFIG.half_life_days)
    served = weighted_brier(
        [apply_weights(model.stacker.weights, e.features) for e in hold],
        [int(e.outcome or 0) for e in hold],
        w,
    )
    assert served == pytest.approx(summary["brier"][LEARNED_FUSION], rel=1e-9)

    # Learned fusion champion: predict uses exactly those weights.
    model.selector.champion = LEARNED_FUSION
    weights = list(model.stacker.weights)
    inp = away_input(tl.absences[-1].returned_at.date() + timedelta(days=1), 400, 600)
    pred = model.predict(inp)
    assert pred.reason["active_model"] == LEARNED_FUSION
    assert pred.prior_probability is not None
    features = model._features(
        inp.now, pred.prior_probability, pred.likelihood_ratio, 200.0, pred.distance_m
    )
    assert pred.probability == apply_weights(weights, features)

    # A stacker refit due in challenger mode must not overwrite the evaluated weights.
    model._since_refit = STACKER_REFIT_EVERY
    model.resolve(now + timedelta(minutes=1), True)
    assert model.stacker.weights == weights


# --------------------------------------------------------------------------- M3
def test_observe_transition_defers_refit_to_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    model = KernelBayesReturnModel(CONFIG)
    model.enable_challenger(True)
    day = DEFAULT_START
    model.observe_transition(PresenceTransition(local(day, 480), False))
    for minute in range(490, 700, 10):
        inp = away_input(day, 480, minute)
        model.log_prediction(inp, model.predict(inp))
    model._since_refit = STACKER_REFIT_EVERY

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("heavy work inside observe_transition")

    monkeypatch.setattr(Stacker, "fit", boom)
    monkeypatch.setattr(bayes_module, "numpy_available", boom)
    ep = model.observe_transition(PresenceTransition(local(day, 705), True))
    assert ep is not None and model._since_refit > STACKER_REFIT_EVERY

    calls: list[int] = []
    monkeypatch.setattr(Stacker, "fit", lambda self, *a, **k: calls.append(1))
    monkeypatch.setattr(bayes_module, "numpy_available", lambda: False)
    model.resolve(local(day, 706), True)
    assert calls == [1] and model._since_refit == 0
    assert model.selector.champion == KERNEL_BAYES


# --------------------------------------------------------------------------- M4
def test_compact_state_size_and_round_trip(long_model) -> None:  # type: ignore[no-untyped-def]
    model, _ = long_model
    state = model.to_dict()
    payload = json.dumps(state)
    assert len(model._log.resolved) == 3000
    assert len(payload) < MAX_STATE_BYTES
    assert isinstance(state["log"]["resolved"]["ts"][0], int)
    restored = KernelBayesReturnModel.from_dict(json.loads(payload), CONFIG)
    assert json.dumps(restored.to_dict()) == payload
    assert restored._log.resolved == model._log.resolved
    for inp in _probe_inputs():
        assert restored.predict(inp) == model.predict(inp)


def test_legacy_dict_rows_are_still_read(long_model) -> None:  # type: ignore[no-untyped-def]
    model, tl = long_model
    old = _legacy(model.to_dict())
    assert isinstance(old["log"]["resolved"], list) and "ts" in old["log"]["resolved"][0]
    restored = KernelBayesReturnModel.from_dict(json.loads(json.dumps(old)), CONFIG)
    assert restored._log.resolved == model._log.resolved
    assert restored._calib_pending == model._calib_pending
    assert restored.to_dict() == model.to_dict()
    end = tl.absences[-1].returned_at
    assert restored.maturity(end) == model.maturity(end)
    for inp in _probe_inputs():
        assert restored.predict(inp) == model.predict(inp)


def test_encode_keeps_unusual_rows_verbatim() -> None:
    now = local(DEFAULT_START, 600)
    entries = [
        LogEntry(now, 0.5, None, None, now + timedelta(minutes=90)),
        LogEntry(now, 0.25, 0.3, "away", now + timedelta(minutes=60), outcome=0, x=[1.5] * 12),
    ]
    out = decode_entries(json.loads(json.dumps(encode_entries(entries))))
    assert out == entries


# --------------------------------------------------------------------------- M5
def test_split_training_matches_wrapper_and_fit_touches_no_model() -> None:
    pytest.importorskip("numpy")
    tl = rotating_shift_person(days=35, seed=5)
    model = KernelBayesReturnModel(CONFIG)
    model.enable_challenger(True)
    for _ in replay_online(model, tl):
        pass
    now = tl.absences[-1].returned_at
    twin = KernelBayesReturnModel.from_dict(json.loads(json.dumps(model.to_dict())), CONFIG)
    twin.enable_challenger(True)

    snapshot = model.challenger_snapshot()
    before = json.dumps(model.to_dict())
    result = KernelBayesReturnModel.fit_challenger(snapshot, now)
    assert json.dumps(model.to_dict()) == before
    summary = model.apply_challenger(result, now)
    expected = twin.train_challenger(now)
    assert summary["trained"] and summary["reason"] == "ok"
    for key in ("trained", "reason", "rows", "brier", "active_model"):
        assert summary[key] == expected[key]
    assert model.stacker.weights == twin.stacker.weights
    assert not model.challenger_due(now + timedelta(hours=1))
    inp = away_input(now.date() + timedelta(days=1), 400, 600)
    assert model.predict(inp) == twin.predict(inp)

    model.enable_challenger(False)
    off = model.apply_challenger(KernelBayesReturnModel.fit_challenger(snapshot, now), now)
    assert not off["trained"] and off["reason"] == "disabled"
