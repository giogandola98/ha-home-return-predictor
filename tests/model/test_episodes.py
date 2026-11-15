"""Episode building and filtering."""

from __future__ import annotations

from datetime import date, timedelta

from model_helpers import CONFIG, away_input, local

from hrp_core import KernelBayesReturnModel, PresenceTransition
from hrp_core.episodes import MIN_HOME_STAY_MINUTES, build_episodes

DAY = date(2026, 3, 2)  # Monday


def _absence(start_min: float, minutes: float) -> list[PresenceTransition]:
    start = local(DAY, start_min)
    return [
        PresenceTransition(start, False),
        PresenceTransition(start + timedelta(minutes=minutes), True),
    ]


def test_short_absences_dropped_and_trips_separated() -> None:
    transitions = [
        PresenceTransition(local(DAY, 0), True),
        *_absence(8 * 60, 5),  # GPS bounce: dropped
        *_absence(9 * 60, 9.9),  # still < 10 min: dropped
        *_absence(10 * 60, 10.5),  # kept
        *_absence(12 * 60, 300),  # kept
        *_absence(24 * 60, 40 * 60),  # 40 h: trip
    ]
    built = build_episodes(transitions, CONFIG)
    assert [round(e.duration_minutes, 1) for e in built.episodes] == [10.5, 300.0]
    assert len(built.trips) == 1 and built.trips[0].duration_minutes == 2400
    assert built.episodes[0].weekday == 0
    assert built.is_home is True and built.away_since is None


def test_duplicates_and_open_absence() -> None:
    transitions = [
        PresenceTransition(local(DAY, 60), False),
        PresenceTransition(local(DAY, 70), False),  # duplicate away: ignored
        PresenceTransition(local(DAY, 200), True),
        PresenceTransition(local(DAY, 300), False),
    ]
    built = build_episodes(transitions, CONFIG)
    assert len(built.episodes) == 1 and built.episodes[0].duration_minutes == 140
    assert built.is_home is False and built.away_since == local(DAY, 300)


def test_live_observation_matches_filters() -> None:
    model = KernelBayesReturnModel(CONFIG)
    model.observe_transition(PresenceTransition(local(DAY, 0), True))
    model.observe_transition(PresenceTransition(local(DAY, 60), False))
    assert model.observe_transition(PresenceTransition(local(DAY, 65), True)) is None
    model.observe_transition(PresenceTransition(local(DAY, 120), False))
    ep = model.observe_transition(PresenceTransition(local(DAY, 240), True), is_holiday=True)
    assert ep is not None and ep.duration_minutes == 120 and ep.is_holiday
    model.observe_transition(PresenceTransition(local(DAY, 300), False))
    model.observe_transition(PresenceTransition(local(DAY, 300 + 37 * 60), True))
    assert len(model.episodes()) == 1 and len(model.trips()) == 1


def test_holidays_marked() -> None:
    built = build_episodes(_absence(9 * 60, 60), CONFIG, holidays=[local(DAY, 0)])
    assert built.episodes[0].is_holiday


def test_short_home_stay_does_not_split_absence() -> None:
    """Driving past home / GPS jitter at the zone edge: one absence, not two."""
    start = local(DAY, 8 * 60)
    transitions = [
        PresenceTransition(local(DAY, 0), True),
        PresenceTransition(start, False),
        PresenceTransition(start + timedelta(minutes=240), True),  # 1 min blip
        PresenceTransition(start + timedelta(minutes=241), False),
        PresenceTransition(start + timedelta(minutes=540), True),
        PresenceTransition(start + timedelta(minutes=600), False),  # 60 min at home
        PresenceTransition(start + timedelta(minutes=660), True),
    ]
    built = build_episodes(transitions, CONFIG)
    assert [e.duration_minutes for e in built.episodes] == [540.0, 60.0]
    # Still away after a blip: the open absence keeps its real start.
    built = build_episodes(transitions[:4], CONFIG)
    assert built.episodes == [] and built.away_since == start


def test_live_short_home_stay_continues_absence() -> None:
    """Live: leaving again within MIN_HOME_STAY_MINUTES continues the same absence."""
    model = KernelBayesReturnModel(CONFIG)
    left = local(DAY, 8 * 60)
    model.observe_transition(PresenceTransition(local(DAY, 0), True))
    model.observe_transition(PresenceTransition(left, False))
    for minute in range(8 * 60 + 150, 8 * 60 + 240, 10):  # far from home: no return expected
        inp = away_input(DAY, 8 * 60, minute)
        model.log_prediction(inp, model.predict(inp))
    pending = len(model._log.pending)
    assert pending > 0
    blip = left + timedelta(minutes=240)
    model.observe_transition(PresenceTransition(blip, True))
    model.resolve(blip + timedelta(minutes=2), True)
    model.observe_transition(PresenceTransition(blip + timedelta(minutes=3), False))
    assert model.episodes() == [] and model.away_since == left
    assert model._log.resolved == [] and len(model._log.pending) == pending
    # The real return settles everything against the real arrival.
    back = left + timedelta(minutes=540)
    ep = model.observe_transition(PresenceTransition(back, True))
    assert ep is not None and ep.duration_minutes == 540
    model.resolve(back + timedelta(minutes=MIN_HOME_STAY_MINUTES), True)
    assert len(model._log.resolved) == pending
    assert all(e.outcome == 0 for e in model._log.resolved)  # horizons elapsed before return
    assert len(model.episodes()) == 1
