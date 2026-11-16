"""House-level, decision-oriented evaluation of a preheating policy (pure Python).

The question a heating automation asks is not "will this person be back within 90
minutes?" but "is the house going to be occupied again before the room can warm up?".
This module scores a policy "preheat when ..." on the periods during which the house
was empty, with a one-parameter thermal model:

* while the house is empty the heating is off and the room cools exponentially
  towards ``delta_t_k`` below the comfort temperature (time constant ``tau_hours``);
* once the policy switches on, the room warms at ``rate_k_per_hour``.

Per empty period the policy is sampled every ``step_minutes`` and scored with

* ``deficit_k``: degrees missing at the arrival (cooling minus the warm-up obtained by
  the last uninterrupted "on" run that lasts until the arrival);
* ``early_hours``: hours "on" before the useful window ``[arrival - need, arrival]``,
  where ``need`` is the warm-up time the cooling requires (wasted heating);
* ``lead_minutes``: length of that last "on" run.

Baselines: never setback (always on), a fixed daily schedule and the oracle (on
exactly ``need`` before the arrival). Per-person metrics on single absences (lead of the
first sustained "on", missed returns, false alarms per day) complement the house view.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, tzinfo

Policy = Callable[[datetime], bool]


@dataclass(frozen=True)
class ThermalParams:
    """Room model used by the metric (defaults: a living room with radiators)."""

    rate_k_per_hour: float = 0.55
    tau_hours: float = 50.0
    delta_t_k: float = 12.0

    def cooling_k(self, hours: float) -> float:
        """Degrees lost after ``hours`` with the heating off."""
        return self.delta_t_k * (1.0 - math.exp(-max(hours, 0.0) / self.tau_hours))

    def need_hours(self, hours_empty: float) -> float:
        """Warm-up time needed to recover the cooling of an absence of ``hours_empty``."""
        return self.cooling_k(hours_empty) / self.rate_k_per_hour


@dataclass(frozen=True)
class EmptyPeriod:
    """The house was empty from ``start`` until the first arrival at ``end``."""

    start: datetime
    end: datetime
    arrivals: tuple[str, ...] = ()  # persons home at ``end``

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


@dataclass(frozen=True)
class PeriodScore:
    period: EmptyPeriod
    cooling_k: float
    need_hours: float
    lead_minutes: float
    deficit_k: float
    early_hours: float
    on_hours: float


@dataclass
class HouseScore:
    """Scores of one policy on a set of empty periods."""

    periods: list[PeriodScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.periods)

    @property
    def total_deficit_k(self) -> float:
        return sum(p.deficit_k for p in self.periods)

    @property
    def mean_deficit_k(self) -> float:
        return self.total_deficit_k / self.n if self.n else 0.0

    @property
    def early_hours(self) -> float:
        return sum(p.early_hours for p in self.periods)

    @property
    def on_hours(self) -> float:
        return sum(p.on_hours for p in self.periods)


def step_value[T](series: Sequence[tuple[datetime, T]], default: T) -> Callable[[datetime], T]:
    """Callable returning the value of a sorted step series at any time."""
    times = [ts for ts, _ in series]
    values = [v for _, v in series]

    def value(t: datetime) -> T:
        i = bisect.bisect_right(times, t)
        return values[i - 1] if i else default

    return value


def empty_periods(
    presence: Mapping[str, Sequence[tuple[datetime, bool]]],
    start: datetime | None = None,
    end: datetime | None = None,
    *,
    step_minutes: float = 2.0,
    min_hours: float = 0.0,
) -> list[EmptyPeriod]:
    """Periods during which nobody was home, sampled every ``step_minutes``.

    ``presence`` maps a person to its sorted ``(ts, is_home)`` changes; before the first
    change a person counts as home. Defaults: from the latest first change to the
    earliest last change (the span every series covers). A period still open at ``end``
    is not returned (its arrival is unknown).
    """
    if not presence:
        return []
    start = start or max(s[0][0] for s in presence.values() if s)
    end = end or min(s[-1][0] for s in presence.values() if s)
    home = {who: step_value(series, True) for who, series in presence.items()}
    step = timedelta(minutes=step_minutes)
    out: list[EmptyPeriod] = []
    begin: datetime | None = None
    t = start
    while t <= end:
        at_home = tuple(who for who, f in home.items() if f(t))
        if not at_home and begin is None:
            begin = t
        elif at_home and begin is not None:
            period = EmptyPeriod(begin, t, at_home)
            if period.hours >= min_hours:
                out.append(period)
            begin = None
        t += step
    return out


def score_period(
    period: EmptyPeriod,
    policy: Policy,
    params: ThermalParams | None = None,
    *,
    step_minutes: float = 2.0,
) -> PeriodScore:
    """Score a policy (sampled every ``step_minutes``) on one empty period."""
    params = params or ThermalParams()
    step = timedelta(minutes=step_minutes)
    cooling = params.cooling_k(period.hours)
    need = cooling / params.rate_k_per_hour
    run_start: datetime | None = None
    on_steps = early_steps = 0
    t = period.start
    while t < period.end:
        on = bool(policy(t))
        if on:
            on_steps += 1
            if (period.end - t).total_seconds() / 3600.0 > need:
                early_steps += 1
            if run_start is None:
                run_start = t
        else:
            run_start = None
        t += step
    lead_h = (period.end - run_start).total_seconds() / 3600.0 if run_start else 0.0
    return PeriodScore(
        period=period,
        cooling_k=cooling,
        need_hours=need,
        lead_minutes=lead_h * 60.0,
        deficit_k=max(0.0, cooling - params.rate_k_per_hour * lead_h),
        early_hours=early_steps * step_minutes / 60.0,
        on_hours=on_steps * step_minutes / 60.0,
    )


def score_policy(
    periods: Sequence[EmptyPeriod],
    policy: Policy,
    params: ThermalParams | None = None,
    *,
    step_minutes: float = 2.0,
) -> HouseScore:
    return HouseScore([score_period(p, policy, params, step_minutes=step_minutes) for p in periods])


# ------------------------------------------------------------------ baselines
def never_setback(_: datetime) -> bool:
    """Heating kept on the whole time: no deficit, every hour before the window wasted."""
    return True


def fixed_schedule(on_from_minutes: float, on_until_minutes: float, tz: tzinfo) -> Policy:
    """On every day between two local wall times (minutes after midnight; may wrap)."""

    def policy(t: datetime) -> bool:
        local = t.astimezone(tz)
        m = local.hour * 60 + local.minute
        if on_from_minutes <= on_until_minutes:
            return on_from_minutes <= m < on_until_minutes
        return m >= on_from_minutes or m < on_until_minutes

    return policy


def oracle(periods: Sequence[EmptyPeriod], params: ThermalParams | None = None) -> Policy:
    """On exactly from ``arrival - need`` (knows every arrival in advance)."""
    params = params or ThermalParams()
    windows = [(p.end - timedelta(hours=params.need_hours(p.hours)), p.end) for p in periods]

    def policy(t: datetime) -> bool:
        return any(a <= t < b for a, b in windows)

    return policy


def threshold_policy(probability: Callable[[datetime], float], threshold: float) -> Policy:
    """On while the house return probability is at least ``threshold``."""
    return lambda t: probability(t) >= threshold


def house_probability(
    probabilities: Mapping[str, Callable[[datetime], float]],
    presence: Mapping[str, Callable[[datetime], bool]],
) -> Callable[[datetime], float]:
    """1 - prod(1 - p) over the persons away (the integration's house aggregate)."""

    def value(t: datetime) -> float:
        away = [p(t) for who, p in probabilities.items() if not presence[who](t)]
        return 1.0 - math.prod(1.0 - p for p in away)

    return value


# ------------------------------------------------------------- per person
@dataclass(frozen=True)
class AbsenceScore:
    left_at: datetime
    returned_at: datetime
    sustained_lead_minutes: float  # last "on" run lasting until the return (0 = missed)
    false_alarms: int  # "on" runs that ended before the return


@dataclass
class PersonScore:
    absences: list[AbsenceScore] = field(default_factory=list)
    days: float = 0.0  # observed days (for the false-alarm rate)

    @property
    def missed(self) -> int:
        return sum(1 for a in self.absences if a.sustained_lead_minutes <= 0)

    @property
    def median_lead_minutes(self) -> float | None:
        leads = sorted(a.sustained_lead_minutes for a in self.absences)
        if not leads:
            return None
        mid = len(leads) // 2
        return leads[mid] if len(leads) % 2 else (leads[mid - 1] + leads[mid]) / 2.0

    @property
    def false_alarms_per_day(self) -> float:
        n = sum(a.false_alarms for a in self.absences)
        return n / self.days if self.days > 0 else 0.0


def score_person(
    absences: Sequence[tuple[datetime, datetime]],
    signal: Policy,
    *,
    step_minutes: float = 2.0,
    min_minutes: float = 10.0,
    days: float | None = None,
) -> PersonScore:
    """Per-person metrics of a binary "returning" signal over finished absences."""
    step = timedelta(minutes=step_minutes)
    out = PersonScore()
    for left, back in absences:
        if (back - left).total_seconds() / 60.0 < min_minutes:
            continue
        run_start: datetime | None = None
        alarms = 0
        t = left
        while t < back:
            if signal(t):
                run_start = run_start or t
            else:
                if run_start is not None:
                    alarms += 1
                run_start = None
            t += step
        lead = (back - run_start).total_seconds() / 60.0 if run_start else 0.0
        out.absences.append(AbsenceScore(left, back, lead, alarms))
    if days is None and absences:
        span = max(b for _, b in absences) - min(a for a, _ in absences)
        days = span.total_seconds() / 86400.0
    out.days = days or 0.0
    return out
