"""Episode building and filtering."""

from __future__ import annotations

from datetime import date, timedelta

from model_helpers import CONFIG, local

from hrp_core import KernelBayesReturnModel, PresenceTransition
from hrp_core.episodes import build_episodes

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
