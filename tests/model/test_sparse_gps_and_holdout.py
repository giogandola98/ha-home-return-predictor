"""Sparse GPS reporting, holdouts that must contain returns, and the GPS floor."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from model_helpers import CONFIG, approach_fixes, away_input, local

from hrp_core import KernelBayesReturnModel, PositionFix
from hrp_core.evidence import compute_evidence
from hrp_core.maturity import LogEntry
from hrp_core.selection import GBM, KERNEL_BAYES, LEARNED_FUSION, informative_split
from hrp_core.stacker import N_FEATURES, Stacker

DAY = date(2026, 3, 2)


def _every_five_minutes(now_minutes: float, last_fix_age: float, n: int = 3):
    """A phone on a motorway reporting every 5 min, 50 km/h towards home."""
    now = local(DAY, now_minutes)
    newest = now - timedelta(minutes=last_fix_age)
    fixes = approach_fixes(newest, 8000, 50, minutes=5.0 * (n - 1), every=5.0)
    return now, fixes


def test_sparse_fixes_keep_the_approach_speed() -> None:
    # 3 fixes 5 min apart; 3 min after the newest only 2 are within the 10-min window
    now, fixes = _every_five_minutes(600, last_fix_age=3)
    ev = compute_evidence(fixes, now, 10, 250)
    assert ev.approach_class == "approach_fast"
    assert ev.speed_kmh == pytest.approx(50, abs=0.5)
    # dense fixes still use the plain window
    dense = approach_fixes(now, 5000, 30, minutes=30, every=1.0)
    assert compute_evidence(dense, now, 10, 250).n_fixes == 11


def test_sparse_fallback_needs_fresh_and_compact_fixes() -> None:
    # newest fix older than the window: the phone may have stopped
    now, fixes = _every_five_minutes(600, last_fix_age=11)
    assert compute_evidence(fixes, now, 10, 250).approach_class is None
    # last three fixes span more than twice the window
    now = local(DAY, 600)
    spread = [
        PositionFix(now - timedelta(minutes=m), 8000 - 100 * (25 - m), 15.0) for m in (25, 8, 2)
    ]
    assert compute_evidence(spread, now, 10, 250).approach_class is None


def test_informative_split_grows_until_it_holds_returns() -> None:
    # 100 entries, 5 returns (runs of 1) early on, none in the newest 20 %
    y = [0] * 100
    for start in (10, 30, 55, 62, 70):
        y[start : start + 3] = [1, 1, 1]
    split = informative_split(y)
    assert split is not None and split < 80
    held = y[split:]
    runs = sum(1 for i, v in enumerate(held) if v and (i == 0 or not held[i - 1]))
    assert runs == 3 and split == 55  # grown back just enough, return kept whole
    # already informative: the plain 80/20 split
    y2 = [0, 1] * 50
    assert informative_split(y2) == 80
    # no returns in the newest half: no evaluation
    y3 = [1] * 10 + [0] * 90
    assert informative_split(y3) is None
    assert informative_split([]) is None


def _entries(outcomes: list[int]) -> list[LogEntry]:
    start = local(DAY, 0)
    out = []
    for i, o in enumerate(outcomes):
        ts = start + timedelta(minutes=10 * i)
        feats = [-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0][:N_FEATURES]
        out.append(LogEntry(ts, 0.1, 0.1, None, ts + timedelta(minutes=90), o, feats))
    return out


def test_stacker_skips_a_holdout_without_returns() -> None:
    st = Stacker()
    st.weights = [0.1] * (N_FEATURES + 1)
    st.trained_on, st.holdout_brier_bayes = 7, 0.123
    trip = _entries([1, 0, 0] * 50 + [0] * 300)  # a long trip at the end
    assert st.fit(trip, trip[-1].ts, 28.0) is False
    assert st.weights == [0.1] * (N_FEATURES + 1)  # untouched
    assert st.trained_on == 7 and st.holdout_brier_bayes == 0.123
    assert st.fit(_entries([0, 0, 1] * 120), trip[-1].ts, 28.0) is True
    assert st.holdout_brier_bayes is not None


def test_challenger_on_a_trip_keeps_the_selection() -> None:
    model = KernelBayesReturnModel(CONFIG)
    model.enable_challenger(True)
    model.selector.champion = LEARNED_FUSION
    brier = {KERNEL_BAYES: 0.05, LEARNED_FUSION: 0.04, GBM: None}
    model.selector.brier = dict(brier)
    entries = _entries([1, 0, 0] * 100 + [0] * 400)
    snapshot = {
        "enabled": True,
        "generation": model.generation,
        "half_life_days": 28.0,
        "resolved_total": len(entries),
        "ts": [e.ts for e in entries],
        "x": [[0.0] * 12 for _ in entries],
        "f": [e.features for e in entries],
        "y": [int(e.outcome or 0) for e in entries],
    }
    now = entries[-1].ts
    fitted = model.fit_challenger(snapshot, now)
    assert fitted["status"] == "uninformative_holdout"
    summary = model.apply_challenger(fitted, now)
    assert summary["trained"] is False and summary["reason"] == "uninformative_holdout"
    assert model.selector.champion == LEARNED_FUSION and model.selector.brier == brier


def test_learned_model_never_undercuts_a_clear_gps_approach() -> None:
    model = KernelBayesReturnModel(CONFIG)
    model.selector.champion = LEARNED_FUSION
    model.stacker.weights = [-12.0] + [0.0] * N_FEATURES  # always ~P_MIN
    now = local(DAY, 600)
    approaching = away_input(DAY, 300, 600, approach_fixes(now, 6000, 50))
    pred = model.predict(approaching)
    assert pred.approach_class == "approach_fast" and pred.likelihood_ratio > 1
    assert pred.reason["gps_floor"] is True
    assert pred.probability == pytest.approx(pred.reason["probability_bayes"], abs=1e-4)
    # without GPS the learned model serves as it is
    still = model.predict(away_input(DAY, 300, 600))
    assert still.reason["gps_floor"] is False
    assert still.probability < still.reason["probability_bayes"]
