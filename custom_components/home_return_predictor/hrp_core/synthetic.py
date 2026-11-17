"""Deterministic synthetic presence/GPS data and an online replay driver.

Used by the pure tests and ``scripts/simulate.py``; not used by the integration.
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta, tzinfo

from .base import ModelInput, PositionFix, Prediction, PresenceTransition, ReturnModel
from .episodes import MIN_HOME_STAY_MINUTES

try:
    from zoneinfo import ZoneInfo

    DEFAULT_TZ: tzinfo = ZoneInfo("Europe/Rome")
except Exception:  # pragma: no cover - no tzdata available
    from datetime import timezone

    DEFAULT_TZ = timezone(timedelta(hours=1))

DEFAULT_START = date(2026, 1, 5)  # a Monday
SHIFTS: dict[str, tuple[int, int]] = {"morning": (6, 14), "afternoon": (14, 22), "night": (22, 6)}


@dataclass
class Absence:
    """Ground-truth absence."""

    left_at: datetime
    returned_at: datetime
    kind: str
    destination_km: float
    drive_by_at: datetime | None = None
    # Optional knobs (None = the historical random draw, so older timelines are unchanged)
    seed: int | None = None  # trajectory seed: absences sharing it share the trajectory
    kmh: float | None = None  # outbound speed
    back_kmh: float | None = None  # speed of the drive home
    drive_by_m: float | None = None  # closest distance of the drive-by
    zone: str | None = None  # known zone at the destination (enter/exit events)
    zone_passes: tuple[datetime, ...] = ()  # passes through ``pass_zone`` without stopping
    pass_zone: str = "work"


@dataclass(frozen=True)
class ZoneEvent:
    """Entering or leaving a known (non-home) zone."""

    ts: datetime
    zone: str
    entered: bool


@dataclass
class Timeline:
    """A synthetic person: absences, transitions and GPS fixes (all UTC)."""

    name: str
    tz: tzinfo
    absences: list[Absence] = field(default_factory=list)
    transitions: list[PresenceTransition] = field(default_factory=list)
    fixes: list[PositionFix] = field(default_factory=list)
    zone_events: list[ZoneEvent] = field(default_factory=list)
    homeward: list[datetime] = field(default_factory=list)  # start of each drive home
    style: str = "android"  # GPS reporting: "android" (dense) or "iphone" (sparse)
    _fix_times: list[datetime] = field(default_factory=list, repr=False)

    def recent_fixes(
        self, now: datetime, lookback: timedelta, since: datetime | None = None
    ) -> list[PositionFix]:
        """Fixes within ``lookback`` plus the newest older one (after ``since``), like the
        integration's buffer."""
        recent = self.fixes_between(now - lookback, now)
        lo = bisect.bisect_left(self._fix_times, now - lookback)
        if lo > 0 and (since is None or self.fixes[lo - 1].ts >= since):
            return [self.fixes[lo - 1], *recent]
        return recent

    def fixes_between(self, start: datetime, end: datetime) -> list[PositionFix]:
        """Fixes with start <= ts <= end, oldest first."""
        lo = bisect.bisect_left(self._fix_times, start)
        hi = bisect.bisect_right(self._fix_times, end)
        return self.fixes[lo:hi]


def _local(day: date, minutes: float, tz: tzinfo) -> datetime:
    """Local wall time ``minutes`` after midnight of ``day``, returned in UTC."""
    base = datetime(day.year, day.month, day.day, tzinfo=tz)
    naive = base.replace(tzinfo=None) + timedelta(minutes=minutes)
    return naive.replace(tzinfo=tz).astimezone(UTC)


class _FixWriter:
    """Generates noisy GPS fixes along piecewise trajectories (dense reporting)."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.fixes: list[PositionFix] = []

    def _emit(self, ts: datetime, true_m: float) -> None:
        r = self.rng.random()
        if r < 0.04:
            acc = self.rng.uniform(300, 1000)  # junk, should be filtered
        elif r < 0.14:
            acc = self.rng.uniform(60, 200)
        else:
            acc = self.rng.uniform(5, 40)
        noisy = max(0.0, true_m + self.rng.gauss(0, acc * 0.5))
        self.fixes.append(PositionFix(ts=ts, distance_m=noisy, accuracy_m=acc))

    def drive(self, start: datetime, d0: float, d1: float, kmh: float) -> datetime:
        """Drive from distance d0 to d1 (m); returns arrival time."""
        minutes = abs(d1 - d0) / 1000.0 / kmh * 60.0
        t = 0.0
        while t < minutes:
            self._emit(start + timedelta(minutes=t), d0 + (d1 - d0) * t / minutes)
            t += self.rng.uniform(1.0, 3.0)
        return start + timedelta(minutes=minutes)

    def stay(self, start: datetime, end: datetime, dist: float) -> None:
        t = start + timedelta(minutes=self.rng.uniform(3, 10))
        while t < end:
            self._emit(t, dist)
            t += timedelta(minutes=self.rng.uniform(8, 20))


class _IPhoneWriter(_FixWriter):
    """iPhone-like reporting: a fix every ~5 min on the move (sometimes a burst of 3
    fixes within a minute), one fix shortly after stopping, then silence."""

    def drive(self, start: datetime, d0: float, d1: float, kmh: float) -> datetime:
        minutes = abs(d1 - d0) / 1000.0 / kmh * 60.0
        t = self.rng.uniform(0.5, 2.0)
        while t < minutes:
            burst = self.rng.random() < IPHONE_BURST_RATE
            for k in range(3 if burst else 1):
                tk = min(t + 0.4 * k, minutes)
                self._emit(start + timedelta(minutes=tk), d0 + (d1 - d0) * tk / minutes)
            t += self.rng.uniform(4.0, 6.0)
        return start + timedelta(minutes=minutes)

    def stay(self, start: datetime, end: datetime, dist: float) -> None:
        t = start + timedelta(minutes=self.rng.uniform(1, 3))
        if t < end:
            self._emit(t, dist)


IPHONE_BURST_RATE = 0.1
WRITERS: dict[str, type[_FixWriter]] = {"android": _FixWriter, "iphone": _IPhoneWriter}


def _build(
    name: str,
    tz: tzinfo,
    absences: list[Absence],
    rng: random.Random,
    style: str = "android",
) -> Timeline:
    absences.sort(key=lambda a: a.left_at)
    clean: list[Absence] = []
    for ab in absences:
        if clean and ab.left_at < clean[-1].returned_at + timedelta(minutes=5):
            continue
        clean.append(ab)
    writer = WRITERS[style](rng)
    transitions: list[PresenceTransition] = []
    zone_events: list[ZoneEvent] = []
    homeward: list[datetime] = []
    for ab in clean:
        transitions.append(PresenceTransition(ab.left_at, False))
        transitions.append(PresenceTransition(ab.returned_at, True))
        total = (ab.returned_at - ab.left_at).total_seconds() / 60.0
        if total < 15:
            continue  # GPS bounce / tiny outing: no trajectory
        tr = random.Random(ab.seed) if ab.seed is not None else rng
        dest = ab.destination_km * 1000.0
        kmh = tr.uniform(40, 60)
        here = writer.drive(ab.left_at, 0.0, dest, ab.kmh or kmh)
        if ab.kmh:
            kmh = ab.kmh
        current = dest
        if ab.zone is not None:
            zone_events.append(ZoneEvent(here, ab.zone, True))
        for ts in ab.zone_passes:
            zone_events.append(ZoneEvent(ts, ab.pass_zone, True))
            zone_events.append(ZoneEvent(ts + timedelta(minutes=3), ab.pass_zone, False))
        if ab.drive_by_at is not None:
            closest = tr.uniform(200, 500)
            if ab.drive_by_m is not None:
                closest = ab.drive_by_m
            approach_start = ab.drive_by_at - timedelta(
                minutes=(current - closest) / 1000.0 / kmh * 60.0
            )
            writer.stay(here, approach_start, current)
            writer.drive(approach_start, current, closest, kmh)
            current = tr.uniform(5, 15) * 1000.0
            here = writer.drive(ab.drive_by_at, closest, current, kmh)
        back_kmh = tr.uniform(40, 60)
        if ab.back_kmh:
            back_kmh = ab.back_kmh
        back_start = ab.returned_at - timedelta(minutes=current / 1000.0 / back_kmh * 60.0)
        if back_start < here:
            back_start = here
        if ab.zone is not None:
            zone_events.append(ZoneEvent(back_start, ab.zone, False))
        writer.stay(here, back_start, current)
        writer.drive(back_start, current, 60.0, back_kmh)
        homeward.append(back_start)
    fixes = sorted(writer.fixes, key=lambda f: f.ts)
    return Timeline(
        name=name,
        tz=tz,
        absences=clean,
        transitions=transitions,
        fixes=fixes,
        zone_events=sorted(zone_events, key=lambda e: e.ts),
        style=style,
        homeward=homeward,
        _fix_times=[f.ts for f in fixes],
    )


def _outings(
    rng: random.Random,
    day: date,
    tz: tzinfo,
    windows: list[tuple[float, float]],
    dur: tuple[float, float],
) -> list[Absence]:
    out = []
    for lo, hi in windows:
        start = rng.uniform(lo, hi)
        minutes = rng.uniform(*dur)
        out.append(
            Absence(
                _local(day, start, tz),
                _local(day, start + minutes, tz),
                "errand",
                rng.uniform(2, 8),
            )
        )
    return out


def _maybe_bounce(rng: random.Random, day: date, tz: tzinfo, out: list[Absence]) -> None:
    if rng.random() < 0.1:  # < 10 min "absence": GPS noise / taking the bins out
        start = rng.uniform(8 * 60, 21 * 60)
        out.append(
            Absence(
                _local(day, start, tz), _local(day, start + rng.uniform(2, 8), tz), "bounce", 0.1
            )
        )


def fixed_schedule_person(
    days: int = 120,
    seed: int = 1,
    *,
    start: date = DEFAULT_START,
    tz: tzinfo = DEFAULT_TZ,
    drive_by_rate: float = 0.08,
    return_minutes: float = 17 * 60 + 45,
    change_day: int | None = None,
    new_return_minutes: float = 13 * 60,
) -> Timeline:
    """Office worker: leaves ~07:45 Mon-Fri, back ~17:45; weekend outings; errands."""
    rng = random.Random(seed)
    out: list[Absence] = []
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() < 5:
            ret_mean = (
                new_return_minutes if change_day is not None and i >= change_day else return_minutes
            )
            leave = _local(day, 7 * 60 + 45 + rng.gauss(0, 10), tz)
            back = _local(day, ret_mean + rng.gauss(0, 15), tz)
            drive_by = None
            if rng.random() < drive_by_rate:
                cand = _local(day, rng.uniform(10 * 60 + 30, 12 * 60 + 30), tz)
                if cand + timedelta(minutes=150) < back:
                    drive_by = cand
            out.append(Absence(leave, back, "work", rng.uniform(13, 17), drive_by))
            if rng.random() < 0.25:
                out += _outings(rng, day, tz, [(19 * 60, 20 * 60 + 30)], (30, 90))
        else:
            windows = [(9 * 60 + 30, 11 * 60 + 30)]
            if rng.random() < 0.5:
                windows.append((15 * 60 + 30, 18 * 60))
            out += _outings(rng, day, tz, windows, (40, 150))
        _maybe_bounce(rng, day, tz, out)
    return _build("fixed", tz, out, rng)


def rotating_shift_person(
    days: int = 120,
    seed: int = 2,
    *,
    start: date = DEFAULT_START,
    tz: tzinfo = DEFAULT_TZ,
    drive_by_rate: float = 0.08,
) -> Timeline:
    """Shift worker: 06-14 / 14-22 / 22-06 in irregular 2-4 day blocks, 1-2 days off."""
    rng = random.Random(seed)
    plan: list[str] = []
    current = rng.choice(list(SHIFTS))
    while len(plan) < days:
        plan += [current] * rng.randint(2, 4)
        plan += ["off"] * rng.randint(1, 2)
        current = rng.choice([s for s in SHIFTS if s != current])
    out: list[Absence] = []
    for i in range(days):
        day = start + timedelta(days=i)
        kind = plan[i]
        if kind == "off":
            windows = [(9 * 60 + 30, 12 * 60)]
            if rng.random() < 0.5:
                windows.append((16 * 60, 18 * 60 + 30))
            out += _outings(rng, day, tz, windows, (40, 150))
        else:
            s_h, e_h = SHIFTS[kind]
            begin = s_h * 60.0
            end = e_h * 60.0 + (1440.0 if e_h < s_h else 0.0)
            leave = _local(day, begin - 30 + rng.gauss(0, 8), tz)
            back = _local(day, end + 30 + rng.gauss(0, 12), tz)
            drive_by = None
            if rng.random() < drive_by_rate:
                drive_by = _local(day, begin + rng.uniform(120, 300), tz)
            out.append(Absence(leave, back, kind, rng.uniform(16, 20), drive_by))
            if rng.random() < 0.3:
                window = {
                    "morning": (17 * 60, 19 * 60 + 30),
                    "afternoon": (9 * 60 + 30, 11 * 60),
                    "night": (15 * 60, 17 * 60 + 30),
                }[kind]
                out += _outings(rng, day, tz, [window], (30, 90))
        _maybe_bounce(rng, day, tz, out)
    return _build("shift", tz, out, rng)


def _rebuild(timeline: Timeline, absences: list[Absence], seed: int, style: str) -> Timeline:
    return _build(timeline.name, timeline.tz, list(absences), random.Random(seed), style)


def iphone_fixes(timeline: Timeline, seed: int = 11) -> Timeline:
    """The same absences reported like an iPhone: a fix every ~5 min on the move,
    bursts of 3 fixes within a minute, silence once stopped."""
    return _rebuild(timeline, timeline.absences, seed, "iphone")


def trip(
    timeline: Timeline,
    start: datetime,
    *,
    seed: int = 12,
    days: float | None = None,
    km: float | None = None,
) -> Timeline:
    """Add a trip of 2-5 days to a destination 100-600 km away, leaving at ``start``.

    Absences overlapping the trip are dropped; the GPS keeps the timeline's style.
    """
    rng = random.Random(seed)
    days = days if days is not None else rng.uniform(2, 5)
    km = km if km is not None else rng.uniform(100, 600)
    end = start + timedelta(days=days)
    kept = [
        a
        for a in timeline.absences
        if a.returned_at + timedelta(minutes=10) < start or a.left_at > end + timedelta(minutes=10)
    ]
    speed = rng.uniform(80, 110)
    kept.append(Absence(start, end, "trip", km, kmh=speed, back_kmh=speed))
    return _rebuild(timeline, kept, seed, timeline.style)


def couple(
    days: int = 120,
    seed: int = 21,
    *,
    together_share: float = 0.4,
    start: date = DEFAULT_START,
    tz: tzinfo = DEFAULT_TZ,
    style: str = "iphone",
) -> tuple[Timeline, Timeline]:
    """Two persons with their own work days; ``together_share`` of the evening and
    weekend outings are made together (same times, shared trajectory -> co-located
    fixes)."""
    rng = random.Random(seed)
    a: list[Absence] = []
    b: list[Absence] = []

    def outing(day: date, lo: float, hi: float, dur: tuple[float, float]) -> None:
        begin = rng.uniform(lo, hi)
        minutes = rng.uniform(*dur)
        left, back = _local(day, begin, tz), _local(day, begin + minutes, tz)
        km = rng.uniform(2, 10)
        if rng.random() < together_share:
            shared = rng.randrange(1 << 30)
            a.append(Absence(left, back, "together", km, seed=shared))
            b.append(Absence(left, back, "together", km, seed=shared))
        else:
            (a if rng.random() < 0.5 else b).append(Absence(left, back, "errand", km))

    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() < 5:
            a.append(
                Absence(
                    _local(day, 7 * 60 + 45 + rng.gauss(0, 10), tz),
                    _local(day, 17 * 60 + 45 + rng.gauss(0, 15), tz),
                    "work",
                    rng.uniform(13, 17),
                )
            )
            b.append(
                Absence(
                    _local(day, 8 * 60 + 30 + rng.gauss(0, 10), tz),
                    _local(day, 18 * 60 + 15 + rng.gauss(0, 20), tz),
                    "work",
                    rng.uniform(6, 10),
                )
            )
            if rng.random() < 0.4:
                outing(day, 19 * 60 + 30, 20 * 60 + 30, (45, 150))
        else:
            outing(day, 9 * 60 + 30, 11 * 60 + 30, (60, 180))
            if rng.random() < 0.6:
                outing(day, 15 * 60 + 30, 18 * 60, (60, 180))
    return (
        _build("couple_a", tz, a, random.Random(seed + 1), style),
        _build("couple_b", tz, b, random.Random(seed + 2), style),
    )


def zone_passby(
    days: int = 120,
    seed: int = 31,
    *,
    rate: float = 0.3,
    start: date = DEFAULT_START,
    tz: tzinfo = DEFAULT_TZ,
    style: str = "iphone",
    zone: str = "work",
) -> Timeline:
    """Office worker whose days hold passes at 300-800 m from home (work days) and
    through the ``zone`` zone (weekend outings), none of them a return."""
    rng = random.Random(seed)
    out: list[Absence] = []
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() < 5:
            back = _local(day, 17 * 60 + 45 + rng.gauss(0, 15), tz)
            drive_by = None
            if rng.random() < rate:
                drive_by = _local(day, rng.uniform(10 * 60 + 30, 13 * 60), tz)
            out.append(
                Absence(
                    _local(day, 7 * 60 + 45 + rng.gauss(0, 10), tz),
                    back,
                    "work",
                    rng.uniform(13, 17),
                    drive_by,
                    drive_by_m=rng.uniform(300, 800),
                )
            )
        else:
            begin = rng.uniform(9 * 60 + 30, 11 * 60)
            minutes = rng.uniform(120, 240)
            passes: tuple[datetime, ...] = ()
            if rng.random() < rate:
                passes = (_local(day, begin + rng.uniform(20, minutes - 30), tz),)
            out.append(
                Absence(
                    _local(day, begin, tz),
                    _local(day, begin + minutes, tz),
                    "errand",
                    rng.uniform(5, 15),
                    zone_passes=passes,
                    pass_zone=zone,
                )
            )
    return _build("passby", tz, out, rng, style)


def workplace_zone(
    days: int = 120,
    seed: int = 41,
    *,
    zone: str = "work",
    zone_km: float = 12.0,
    travel_minutes: float = 20.0,
    start: date = DEFAULT_START,
    tz: tzinfo = DEFAULT_TZ,
    style: str = "iphone",
) -> Timeline:
    """Work days spent in the known zone ``zone``; leaving it is followed by the arrival
    home after a stable ``travel_minutes`` (+-5 %)."""
    rng = random.Random(seed)
    out: list[Absence] = []
    for i in range(days):
        day = start + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        speed = zone_km * 60.0 / travel_minutes * rng.uniform(0.95, 1.05)
        out.append(
            Absence(
                _local(day, 8 * 60 + rng.gauss(0, 10), tz),
                _local(day, 17 * 60 + 30 + rng.gauss(0, 30), tz),
                "work",
                zone_km,
                back_kmh=speed,
                zone=zone,
            )
        )
    return _build("workplace", tz, out, rng, style)


@dataclass
class StepRecord:
    """One online prediction with its ground truth."""

    ts: datetime
    prediction: Prediction
    truth: bool  # returned within the horizon of ts
    logged: bool


def replay_online(
    model: ReturnModel,
    timeline: Timeline,
    *,
    horizon_minutes: float = 90.0,
    step_minutes: float = 5.0,
    fix_lookback_minutes: float = 30.0,
    until: datetime | None = None,
    on_absence_end: Callable[[datetime], None] | None = None,
) -> Iterator[StepRecord]:
    """Drive the model like the integration would: transitions, predict, log, resolve."""
    step = timedelta(minutes=step_minutes)
    lookback = timedelta(minutes=fix_lookback_minutes)
    for ab in timeline.absences:
        if until is not None and ab.left_at >= until:
            return
        model.observe_transition(PresenceTransition(ab.left_at, False))
        t = ab.left_at + step
        while t < ab.returned_at:
            inp = ModelInput(
                now=t,
                is_home=False,
                left_at=ab.left_at,
                fixes=timeline.recent_fixes(t, lookback, ab.left_at),
            )
            pred = model.predict(inp)
            logged = model.log_prediction(inp, pred)
            model.resolve(t, False)
            truth = (ab.returned_at - t).total_seconds() / 60.0 <= horizon_minutes
            yield StepRecord(t, pred, truth, logged)
            t += step
        model.observe_transition(PresenceTransition(ab.returned_at, True))
        # Like the integration's periodic update: the arrival is scored once confirmed.
        model.resolve(ab.returned_at + timedelta(minutes=MIN_HOME_STAY_MINUTES), True)
        if on_absence_end is not None:
            on_absence_end(ab.returned_at)
