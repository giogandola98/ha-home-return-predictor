"""Sparse GPS reporting: the regression falls back to a wider window."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from model_helpers import approach_fixes, local

from hrp_core import PositionFix
from hrp_core.evidence import compute_evidence

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


def test_burst_of_fixes_falls_back_to_the_extended_window() -> None:
    now = local(DAY, 600)
    older = [PositionFix(now - timedelta(minutes=m), 9000 - 800 * (15 - m), 15.0) for m in (15, 9)]
    burst = [PositionFix(now - timedelta(seconds=s), 9000 - 800 * 15, 15.0) for s in (50, 30, 10)]
    ev = compute_evidence([*older, *burst], now, 10, 250)
    assert ev.approach_class is not None and ev.speed_kmh > 30
