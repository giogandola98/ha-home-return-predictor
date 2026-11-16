"""Shape tests of the synthetic generators (iPhone GPS, trips, couples, zones)."""

from __future__ import annotations

from datetime import timedelta

from hrp_core.house_metric import empty_periods
from hrp_core.synthetic import (
    _local,
    couple,
    fixed_schedule_person,
    iphone_fixes,
    trip,
    workplace_zone,
    zone_passby,
)


def _minutes(a, b) -> float:
    return (b - a).total_seconds() / 60.0


def test_iphone_fixes_are_sparse_on_the_move_and_silent_when_stopped() -> None:
    base = fixed_schedule_person(days=30, seed=3)
    tl = iphone_fixes(base)
    assert [a.left_at for a in tl.absences] == [a.left_at for a in base.absences]
    assert tl.style == "iphone" and len(tl.fixes) < len(base.fixes) / 2
    gaps = [_minutes(a.ts, b.ts) for a, b in zip(tl.fixes, tl.fixes[1:], strict=False)]
    bursts = sum(1 for g in gaps if g < 1.0)
    moving = [g for g in gaps if 1.0 <= g <= 10.0]
    assert bursts > 0  # bursts of fixes within a minute
    assert moving and 3.0 <= sorted(moving)[len(moving) // 2] <= 6.5  # ~5 min apart
    # a work day: a single fix while parked at the office (the long silent stretch)
    work = next(a for a in tl.absences if a.kind == "work")
    mid = tl.fixes_between(work.left_at + timedelta(hours=3), work.returned_at - timedelta(hours=3))
    assert len(mid) == 0


def test_trip_replaces_the_overlapping_absences() -> None:
    base = fixed_schedule_person(days=30, seed=3)
    start = _local(base.absences[10].left_at.date(), 9 * 60, base.tz)
    tl = trip(base, start, seed=5)
    trips = [a for a in tl.absences if a.kind == "trip"]
    assert len(trips) == 1
    t = trips[0]
    assert 2 * 1440 <= _minutes(t.left_at, t.returned_at) <= 5 * 1440
    assert 100 <= t.destination_km <= 600
    assert not any(
        a.left_at < t.returned_at and a.returned_at > t.left_at for a in tl.absences if a is not t
    )
    far = tl.fixes_between(t.left_at + timedelta(hours=12), t.returned_at - timedelta(hours=12))
    assert far and min(f.distance_m for f in far) > 50_000


def test_couple_returns_together_with_colocated_fixes() -> None:
    a, b = couple(days=60, seed=4, together_share=0.5)
    joint = [x for x in a.absences if x.kind == "together"]
    assert joint and {(x.left_at, x.returned_at) for x in joint} <= {
        (y.left_at, y.returned_at) for y in b.absences
    }
    # co-located: during a joint outing both phones report similar distances
    ab = next(x for x in joint if _minutes(x.left_at, x.returned_at) > 60)
    fa = a.fixes_between(ab.left_at, ab.returned_at)
    fb = b.fixes_between(ab.left_at, ab.returned_at)
    assert fa and fb
    for f in fa:
        if f.accuracy_m > 60:
            continue
        near = min(fb, key=lambda g: abs(_minutes(f.ts, g.ts)))
        if abs(_minutes(f.ts, near.ts)) < 1 and near.accuracy_m <= 60:
            assert abs(f.distance_m - near.distance_m) < 1500
    # a sizeable share of the empty-house periods ends with both arriving together
    presence = {
        "a": [(t.ts, t.is_home) for t in a.transitions],
        "b": [(t.ts, t.is_home) for t in b.transitions],
    }
    periods = empty_periods(presence, step_minutes=5)
    together = sum(1 for p in periods if len(p.arrivals) == 2)
    assert 0.15 <= together / len(periods) <= 0.6
    # without shared outings nobody arrives together
    a0, b0 = couple(days=60, seed=4, together_share=0.0)
    assert not [x for x in a0.absences if x.kind == "together"]


def test_zone_passby_never_returns_home() -> None:
    tl = zone_passby(days=60, seed=6, rate=0.5, style="android")  # dense: the closest fix
    assert len(zone_passby(days=60, seed=6, rate=0.5).fixes) < len(tl.fixes)
    by_home = [a for a in tl.absences if a.drive_by_at is not None]
    assert by_home
    for ab in by_home:
        near = tl.fixes_between(ab.drive_by_at - timedelta(minutes=10), ab.drive_by_at)
        assert near and 150 <= min(f.distance_m for f in near) <= 1500
        assert ab.returned_at - ab.drive_by_at > timedelta(hours=1)
    passes = [e for e in tl.zone_events if e.entered]
    assert passes and all(e.zone == "work" for e in passes)
    for e in passes:  # a pass through the work zone happens in the middle of an absence
        ab = next(a for a in tl.absences if a.left_at <= e.ts <= a.returned_at)
        assert ab.returned_at - e.ts > timedelta(minutes=20)


def test_workplace_zone_exit_to_home_time_is_stable() -> None:
    tl = workplace_zone(days=40, seed=7, travel_minutes=20)
    exits = [e for e in tl.zone_events if not e.entered]
    enters = [e for e in tl.zone_events if e.entered]
    assert len(exits) == len(enters) == len(tl.absences) > 20
    for ab, ex in zip(tl.absences, exits, strict=True):
        assert ab.left_at < ex.ts < ab.returned_at
        assert 18.5 <= _minutes(ex.ts, ab.returned_at) <= 21.5
