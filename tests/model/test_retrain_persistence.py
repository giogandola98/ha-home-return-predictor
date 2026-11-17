"""Section 6 model fixes: merge on retrain (M1),
cheap observe_transition (M3), compact persistence (M4)."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta

import pytest
from model_helpers import CONFIG, approach_fixes, away_input, local

from hrp_core import KernelBayesReturnModel, ModelInput, PresenceTransition
from hrp_core.maturity import LogEntry, decode_entries, encode_entries
from hrp_core.synthetic import (
    DEFAULT_START,
    fixed_schedule_person,
    replay_online,
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
def test_horizon_change_discards_horizon_dependent_state(
    long_model: tuple[KernelBayesReturnModel, object],
) -> None:
    """A new horizon redefines the target: the prediction log goes, habits stay."""
    model, _tl = long_model
    state = model.to_dict()
    assert state["horizon_minutes"] == CONFIG.horizon_minutes
    assert model._log.resolved
    now = local(DEFAULT_START + timedelta(days=121), 0)

    same = KernelBayesReturnModel.from_dict(state, CONFIG)
    assert len(same._log.resolved) == len(model._log.resolved)

    other = dataclasses.replace(CONFIG, horizon_minutes=45.0)
    changed = KernelBayesReturnModel.from_dict(state, other)
    assert changed.episodes() == model.episodes() and changed.trips() == model.trips()
    assert changed._log.resolved == [] and changed._log.pending == []
    assert changed._calib_pending == []
    assert changed.maturity(now).resolved_predictions == 0
    assert changed.to_dict()["horizon_minutes"] == 45.0

    # Snapshots written before the horizon was stored: assume the configured one.
    legacy = dict(state)
    del legacy["horizon_minutes"]
    kept = KernelBayesReturnModel.from_dict(legacy, other)
    assert len(kept._log.resolved) == len(model._log.resolved)
