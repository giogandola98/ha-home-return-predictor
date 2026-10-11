"""Maturity rises with online use."""

from __future__ import annotations

from datetime import timedelta

from model_helpers import CONFIG, local

from hrp_core import KernelBayesReturnModel
from hrp_core.maturity import LogEntry, weighted_brier
from hrp_core.synthetic import fixed_schedule_person, replay_online, rotating_shift_person


def _maturity_curve(timeline, weeks: int) -> list[float]:  # type: ignore[no-untyped-def]
    model = KernelBayesReturnModel(CONFIG)
    start = timeline.absences[0].left_at
    checkpoints = [start + timedelta(weeks=w) for w in range(1, weeks + 1)]
    curve: list[float] = []

    def on_end(ts) -> None:  # type: ignore[no-untyped-def]
        while checkpoints and ts >= checkpoints[0]:
            curve.append(model.maturity(checkpoints.pop(0)).maturity)

    for _ in replay_online(model, timeline, on_absence_end=on_end):
        pass
    return curve


def test_maturity_rises_fixed_schedule() -> None:
    curve = _maturity_curve(fixed_schedule_person(days=70, seed=1), 10)
    assert curve[7] > curve[0]
    assert max(curve) > 0.7 and curve[-1] > 0.7


def test_maturity_rises_shift_worker() -> None:
    curve = _maturity_curve(rotating_shift_person(days=63, seed=2), 9)
    assert curve[7] > curve[0]
    assert curve[-1] > 0.5


def test_brier_and_baseline() -> None:
    now = local(__import__("datetime").date(2026, 3, 2), 600)
    entries = [
        LogEntry(now, p, None, None, now, outcome=o)
        for p, o in ((0.9, 1), (0.1, 0), (0.8, 1), (0.2, 0))
    ]
    bs, ref = weighted_brier(entries, [e.p for e in entries], now, 28) or (0, 0)
    assert bs < ref and ref == 0.25
