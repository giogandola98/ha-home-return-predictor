"""Home detection and history conversion."""

from __future__ import annotations

from datetime import timedelta

from homeassistant.core import HomeAssistant, State
from homeassistant.util import dt as dt_util

from custom_components.home_return_predictor.history import (
    HomeZone,
    async_load_person_history,
    state_is_home,
    states_to_history,
)

WORK = HomeZone("zone.work", "Work", 45.0, 9.0, 200)


def _state(state: str, lat: float | None = None, acc: float = 10, **kw) -> State:
    attrs = {"latitude": lat, "longitude": 9.0, "gps_accuracy": acc} if lat is not None else {}
    return State("person.alice", state, attrs, **kw)


def test_state_is_home_default_zone() -> None:
    """zone.home relies on the person state."""
    assert state_is_home(_state("home"), "zone.home", None) is True
    assert state_is_home(_state("not_home"), "zone.home", None) is False
    assert state_is_home(_state("unavailable"), "zone.home", None) is None


def test_state_is_home_custom_zone() -> None:
    """Other zones: state equals zone name, or coordinates inside the radius."""
    assert state_is_home(_state("Work"), "zone.work", WORK) is True
    assert state_is_home(_state("not_home", 45.001), "zone.work", WORK) is True  # ~111 m
    assert state_is_home(_state("not_home", 45.01), "zone.work", WORK) is False
    assert state_is_home(_state("not_home", 45.001, acc=500), "zone.work", WORK) is None
    assert state_is_home(_state("home"), "zone.work", WORK) is False


def test_states_to_history() -> None:
    """Transitions are deduplicated; fixes come from coordinates."""
    t0 = dt_util.utcnow() - timedelta(hours=3)
    states = [
        _state("home", 45.0, last_updated=t0, last_changed=t0),
        _state("unknown", last_updated=t0 + timedelta(minutes=5)),
        _state("not_home", 45.1, last_updated=t0 + timedelta(minutes=10)),
        _state("not_home", 45.2, last_updated=t0 + timedelta(minutes=20)),
        _state("home", 45.0, last_updated=t0 + timedelta(minutes=90)),
    ]
    transitions, fixes = states_to_history(states, "zone.home", WORK)
    assert [t.is_home for t in transitions] == [True, False, True]
    assert transitions[1].ts == t0 + timedelta(minutes=10)
    assert len(fixes) == 4
    assert fixes[0].distance_m == 0.0
    assert fixes[2].distance_m > fixes[1].distance_m


async def test_history_without_recorder(hass: HomeAssistant) -> None:
    """No recorder -> empty history, no error."""
    assert await async_load_person_history(hass, "person.alice", "zone.home", WORK, 30) == (
        [],
        [],
    )
