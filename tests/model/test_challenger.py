"""Optional gradient-boosting challenger (numpy GBM) and champion/challenger selection."""

from __future__ import annotations

import json
import math
import random
import time
from datetime import timedelta

import pytest
from model_helpers import CONFIG, away_input, local

from hrp_core import KernelBayesReturnModel, numpy_available, sklearn_available
from hrp_core import bayes_model as bayes_module
from hrp_core.challenger import (
    MAX_PERSISTED_BYTES,
    N_CHALLENGER_FEATURES,
    GBMChallenger,
    build_challenger_features,
)
from hrp_core.maturity import LogEntry
from hrp_core.selection import (
    GBM,
    KERNEL_BAYES,
    LEARNED_FUSION,
    Selector,
    chronological_split,
)
from hrp_core.stacker import build_features
from hrp_core.synthetic import DEFAULT_START, replay_online, rotating_shift_person


def _synthetic_rows(n: int, seed: int = 0) -> tuple[list[list[float]], list[int]]:
    """Rows where the outcome depends non-linearly on time of day and LR."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n):
        prior = rng.uniform(0.01, 0.99)
        lr = rng.choice([0.25, 0.5, 1.0, 1.8, 4.0, 7.0])
        hour = rng.uniform(0, 24)
        row = build_challenger_features(
            prior=prior,
            likelihood_ratio=lr,
            minutes_away=rng.uniform(0, 600),
            horizon_minutes=90,
            departure_hour=rng.uniform(0, 24),
            local_hour=hour,
            rest_day=rng.random() < 0.3,
            distance_m=rng.uniform(0, 20_000),
            speed_kmh=rng.uniform(-50, 50),
            has_gps=True,
            p_bayes=prior,
        )
        p = 0.9 if 17 <= hour <= 19 and lr > 1 else 0.1
        rows.append(row)
        labels.append(int(rng.random() < p))
    return rows, labels


# ----------------------------------------------------------------- pure selection
def test_chronological_split() -> None:
    train, hold = chronological_split(list(range(10)))
    assert train == list(range(8)) and hold == [8, 9]


def test_selection_needs_two_consecutive_wins_and_demotes_on_loss() -> None:
    sel = Selector()
    win = {KERNEL_BAYES: 0.10, LEARNED_FUSION: 0.10, GBM: 0.09}  # 10 % better
    small = {KERNEL_BAYES: 0.10, LEARNED_FUSION: 0.10, GBM: 0.098}  # only 2 %
    assert sel.update(win, 500) == KERNEL_BAYES  # one win is not enough
    assert sel.update(small, 500) == KERNEL_BAYES  # streak broken (< 3 %)
    assert sel.update(win, 500) == KERNEL_BAYES
    assert sel.update(win, 500) == GBM  # two consecutive wins
    assert sel.update(small, 500) == GBM  # still ahead of kernel_bayes: kept
    lose = {KERNEL_BAYES: 0.10, LEARNED_FUSION: 0.10, GBM: 0.11}
    assert sel.update(lose, 500) == KERNEL_BAYES  # demoted immediately
    assert sel.update(win, 500) == KERNEL_BAYES  # needs two fresh wins again
    restored = Selector.from_dict(json.loads(json.dumps(sel.to_dict())))
    assert restored.to_dict() == sel.to_dict()


def test_selection_min_entries_and_unscored_champion() -> None:
    sel = Selector()
    win = {KERNEL_BAYES: 0.10, LEARNED_FUSION: 0.10, GBM: 0.05}
    for _ in range(3):
        assert sel.update(win, 399) == KERNEL_BAYES  # < 400 resolved entries
    sel.update(win, 400)
    assert sel.update(win, 400) == GBM
    # gbm not scored (e.g. numpy missing): back to the reference model.
    assert sel.update({KERNEL_BAYES: 0.1, LEARNED_FUSION: 0.1}, 400) == KERNEL_BAYES
    # Learned fusion keeps its historical rule: 1 % on a single evaluation.
    assert sel.update({KERNEL_BAYES: 0.1, LEARNED_FUSION: 0.098}, 300) == LEARNED_FUSION


# ------------------------------------------------------------------ fallback
def test_no_numpy_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    tl = rotating_shift_person(days=21, seed=3)
    model = KernelBayesReturnModel(CONFIG)
    for _ in replay_online(model, tl):
        pass
    monkeypatch.setattr(bayes_module, "numpy_available", lambda: False)
    now = tl.absences[-1].returned_at
    inp = away_input(DEFAULT_START + timedelta(days=22), 400, 600)
    before = model.predict(inp)
    model.enable_challenger(True)
    assert model.challenger_due(now) is False
    summary = model.train_challenger(now)
    assert summary["trained"] is False and summary["reason"] == "numpy_unavailable"
    assert model.predict(inp) == before
    assert before.reason["active_model"] in (KERNEL_BAYES, LEARNED_FUSION)


# --------------------------------------------------------------- numpy required
def test_challenger_fit_predict_and_persist() -> None:
    pytest.importorskip("numpy")
    assert numpy_available() and sklearn_available()  # deprecated alias
    rows, labels = _synthetic_rows(2000)
    ch = GBMChallenger()
    ch.fit(rows, labels, [1.0] * len(rows))
    assert ch.trained and ch.trained_on == 2000
    test_rows, test_labels = _synthetic_rows(1000, seed=1)
    probs = ch.predict_many(test_rows)
    assert all(0.01 <= p <= 0.99 for p in probs)
    brier = sum((p - o) ** 2 for p, o in zip(probs, test_labels, strict=True)) / len(probs)
    assert brier < 0.15  # learns the hour x LR interaction (0.09 is the noise floor)

    data = ch.to_dict()
    assert data is not None and len(json.dumps(data)) <= MAX_PERSISTED_BYTES
    restored = GBMChallenger.from_dict(json.loads(json.dumps(data)))
    assert restored.trained and restored.trained_on == 2000
    assert restored.predict_many(test_rows[:200]) == probs[:200]
    # Unusable payloads give an untrained challenger (retrained later), never an error.
    for bad in (None, {}, {"model": {"garbage": 1}}, {"constant": "x"}, [1, 2]):
        assert not GBMChallenger.from_dict(bad).trained
    assert GBMChallenger().to_dict() is None


def test_challenger_single_class_is_constant() -> None:
    pytest.importorskip("numpy")
    rows, _ = _synthetic_rows(50)
    ch = GBMChallenger()
    ch.fit(rows, [1] * 50)
    assert ch.predict_proba(rows[0]) == 0.99
    restored = GBMChallenger.from_dict(json.loads(json.dumps(ch.to_dict())))
    assert restored.predict_proba(rows[3]) == 0.99


def test_training_5000_rows_under_two_seconds() -> None:
    pytest.importorskip("numpy")
    rows, labels = _synthetic_rows(5000)
    ch = GBMChallenger()
    started = time.perf_counter()
    ch.fit(rows, labels, [0.5 ** (i / 1000) for i in range(len(rows))])  # recency-like
    assert time.perf_counter() - started < 2.0


@pytest.fixture(scope="module")
def shift_model() -> tuple[KernelBayesReturnModel, object]:
    pytest.importorskip("numpy")
    tl = rotating_shift_person(days=35, seed=5)
    model = KernelBayesReturnModel(CONFIG)
    model.enable_challenger(True)
    for _ in replay_online(model, tl):
        pass
    return model, tl


def test_model_train_challenger(shift_model) -> None:  # type: ignore[no-untyped-def]
    model, tl = shift_model
    now = tl.absences[-1].returned_at
    assert len(model._challenger_rows()) >= 400
    assert model.challenger_due(now)  # never trained in this process
    summary = model.train_challenger(now)
    assert summary["trained"] and summary["reason"] == "ok"
    assert summary["rows"] >= 400 and summary["seconds"] < 5
    assert set(summary["brier"]) == {KERNEL_BAYES, LEARNED_FUSION, GBM}
    assert all(0 <= v < 0.25 for v in summary["brier"].values())
    assert summary["active_model"] in (KERNEL_BAYES, LEARNED_FUSION)  # 1 evaluation only
    assert not model.challenger_due(now + timedelta(hours=1))
    assert model.challenger_due(now + timedelta(hours=6))
    report = model.maturity(now)
    assert report.candidate_brier == summary["brier"]
    assert len(model._log.resolved[-1].x) == N_CHALLENGER_FEATURES
    assert model._log.resolved[-1].base is not None


def test_stale_challenger_result_is_discarded(shift_model) -> None:  # type: ignore[no-untyped-def]
    """A fit started before reset()/load_history(replace=True) must not be applied."""
    model, tl = shift_model
    now = tl.absences[-1].returned_at
    model = KernelBayesReturnModel.from_dict(json.loads(json.dumps(model.to_dict())), CONFIG)
    model.enable_challenger(True)
    fitted = model.fit_challenger(model.challenger_snapshot(), now)
    assert fitted["status"] == "ok"

    model.reset()  # e.g. the reset service ran while the fit was in flight
    assert model.generation == 1
    before = json.dumps(model.to_dict())
    summary = model.apply_challenger(fitted, now)
    assert summary["trained"] is False and summary["reason"] == "stale"
    assert json.dumps(model.to_dict()) == before  # selection/stacker untouched
    assert model._resolved_at_training == 0 and model._last_training is None
    assert model._challenger is not fitted["challenger"]

    # load_history(replace=True) also invalidates in-flight fits.
    snap = model.challenger_snapshot()
    model.load_history([], replace=True)
    assert model.generation == 2
    assert model.apply_challenger(model.fit_challenger(snap, now), now)["reason"] == "stale"
    # A merge (replace=False) keeps the generation.
    model.load_history([])
    assert model.generation == 2

    # Persisted, backwards compatible (missing key -> 0).
    data = json.loads(json.dumps(model.to_dict()))
    assert KernelBayesReturnModel.from_dict(data, CONFIG).generation == 2
    data.pop("generation")
    assert KernelBayesReturnModel.from_dict(data, CONFIG).generation == 0


def test_promoted_gbm_serves_and_survives_restart(shift_model) -> None:  # type: ignore[no-untyped-def]
    model, tl = shift_model
    now = tl.absences[-1].returned_at
    model = KernelBayesReturnModel.from_dict(json.loads(json.dumps(model.to_dict())), CONFIG)
    model.enable_challenger(True)
    model.train_challenger(now)
    # Force the promotion (synthetic data rarely lets the GBM win by 3 % twice).
    model.selector.champion = GBM
    model.selector.brier = {KERNEL_BAYES: 0.05, LEARNED_FUSION: 0.04, GBM: 0.03}
    day = DEFAULT_START + timedelta(days=36)
    inp = away_input(day, 400, 600)
    pred = model.predict(inp)
    assert pred.reason["active_model"] == GBM and pred.reason["fusion"] == "gbm"
    assert 0.01 <= pred.probability <= 0.99
    report = model.maturity(local(day, 600))
    assert report.active_model == GBM and report.fusion == "gbm"

    payload = json.dumps(model.to_dict())
    assert '"gbm"' in payload and "sklearn" not in payload
    restored = KernelBayesReturnModel.from_dict(json.loads(payload), CONFIG)
    assert json.dumps(restored.to_dict()) == payload
    assert restored.selector.champion == GBM
    # The exported trees are persisted: the champion serves at once after a restart
    # (and is still retrained on the next due check).
    again = restored.predict(inp)
    assert again.reason["active_model"] == GBM and again.probability == pred.probability
    assert restored.maturity(local(day, 600)).active_model == GBM
    assert restored.challenger_due(local(day, 600))

    # State without the persisted trees (written by the scikit-learn release, which
    # also named the candidate "sklearn_gbm"): the next-best candidate serves until
    # the challenger is retrained.
    legacy = json.loads(payload.replace('"gbm"', '"sklearn_gbm"'))
    legacy.pop("challenger")
    restored = KernelBayesReturnModel.from_dict(legacy, CONFIG)
    assert restored.selector.champion == GBM
    assert set(restored.selector.brier) == {KERNEL_BAYES, LEARNED_FUSION, GBM}
    fallback = restored.predict(inp)
    assert fallback.reason["active_model"] == LEARNED_FUSION
    assert restored.maturity(local(day, 600)).active_model == LEARNED_FUSION
    assert restored.challenger_due(local(day, 600))
    restored.train_challenger(local(day, 600))
    assert restored.predict(inp).reason["active_model"] in (
        GBM,
        KERNEL_BAYES,
        LEARNED_FUSION,
    )
    restored.enable_challenger(False)
    assert restored.predict(inp).reason["active_model"] != GBM
    assert not math.isnan(restored.predict(inp).probability)


def test_log_entry_round_trip_keeps_new_fields() -> None:
    now = local(DEFAULT_START, 600)
    entry = LogEntry(
        now,
        0.4,
        0.3,
        "approach",
        now,
        outcome=1,
        features=build_features(0.3, 4.0, 60, 90, 5000.0, True, 10.0),
        base=0.25,
        x=[0.5] * N_CHALLENGER_FEATURES,
    )
    assert LogEntry.from_dict(json.loads(json.dumps(entry.to_dict()))) == entry


def test_from_dict_rejects_cyclic_tree() -> None:
    from hrp_core.challenger import GBMChallenger

    payload = {
        "trained_on": 500,
        "model": {
            "baseline": 0.0,
            "trees": [{"f": [0, -1], "t": [0.5, 0.0], "l": [0, 1], "r": [1, 1], "v": [0.0, 0.1]}],
        },
    }
    assert not GBMChallenger.from_dict(payload).trained
