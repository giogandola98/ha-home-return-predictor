"""Sparse GPS reporting, holdouts that must contain returns, and the GPS floor."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from model_helpers import approach_fixes, local

from hrp_core import PositionFix
from hrp_core.evidence import compute_evidence
from hrp_core.maturity import LogEntry
from hrp_core.selection import informative_split
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


def test_returns_separated_by_a_stay_at_home_are_counted_apart() -> None:
    # absence A ends with 10 positive entries; after 3 h at home a short absence B
    # (5 entries, all positive) follows. The log only holds absences, so the runs touch.
    y = [0] * 85 + [1] * 10 + [1] * 5
    start = local(DAY, 0)
    times = [start + timedelta(minutes=10 * i) for i in range(95)]
    times += [times[-1] + timedelta(hours=3, minutes=10 * i) for i in range(1, 6)]
    assert informative_split(y, min_returns=2) is None  # by runs alone: one return
    split = informative_split(y, min_returns=2, times=times)
    assert split == 80  # the plain 20 % holdout already holds both returns


def test_keeping_a_return_whole_respects_the_maximum_holdout() -> None:
    y = [1] * 60 + [0, 1, 0, 1, 0] * 8  # a 60-entry run right below the 50 % bound
    split = informative_split(y, min_returns=8)
    assert split is not None and split >= int(len(y) * 0.5)


def test_burst_of_fixes_falls_back_to_the_extended_window() -> None:
    now = local(DAY, 600)
    older = [PositionFix(now - timedelta(minutes=m), 9000 - 800 * (15 - m), 15.0) for m in (15, 9)]
    burst = [PositionFix(now - timedelta(seconds=s), 9000 - 800 * 15, 15.0) for s in (50, 30, 10)]
    ev = compute_evidence([*older, *burst], now, 10, 250)
    assert ev.approach_class is not None and ev.speed_kmh > 30
