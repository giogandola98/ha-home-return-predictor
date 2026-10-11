"""Runtime behaviour: executor/locks, listeners, return event, saves, aggregation, bootstrap."""

from __future__ import annotations

import dataclasses
import threading
from collections.abc import Generator
from datetime import date, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from freezegun.api import FrozenDateTimeFactory
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant, State
from homeassistant.helpers.update_coordinator import UpdateFailed
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_capture_events

from custom_components.home_return_predictor.const import (
    DOMAIN,
    EVENT_RETURN_DETECTED,
    SAVE_DELAY_SECONDS,
)
from custom_components.home_return_predictor.coordinator import (
    HRPCoordinator,
    _Computed,
    build_reason,
)
from custom_components.home_return_predictor.history import (
    HomeZone,
    _event_dates,
    _load_history_chunked,
    states_to_history,
)
from custom_components.home_return_predictor.hrp_core import KernelBayesReturnModel
from custom_components.home_return_predictor.hrp_core.base import (
    Prediction,
    PresenceTransition,
)

from .conftest import ALICE, BOB, HOME_LAT, HOME_LON, person_attrs

COORD = "custom_components.home_return_predictor.coordinator"
WORK = HomeZone("zone.work", "Work", HOME_LAT, HOME_LON, 200)


def _offset(meters_north: float) -> float:
    return HOME_LAT + meters_north / 111_195


@pytest.fixture
def forced() -> Generator[dict[str, tuple[float, timedelta | None]]]:
    """Force (probability, eta offset) per person on top of the real model output."""
    values: dict[str, tuple[float, timedelta | None]] = {}
    original = HRPCoordinator._compute

    def _compute(self: HRPCoordinator, runtime: Any, model_input: Any) -> _Computed:
        computed = original(self, runtime, model_input)
        if computed.prediction is not None and runtime.entity_id in values:
            p, eta = values[runtime.entity_id]
            computed.prediction = dataclasses.replace(
                computed.prediction,
                probability=p,
                eta=model_input.now + eta if eta is not None else None,
            )
        return computed

    with patch.object(HRPCoordinator, "_compute", _compute):
        yield values


async def _leave(hass: HomeAssistant, person: str, name: str, meters: float = 5_000) -> None:
    hass.states.async_set(person, "not_home", person_attrs(name, _offset(meters), HOME_LON))
    await hass.async_block_till_done()


# --------------------------------------------------------------------------- H1


async def test_model_calls_run_in_executor(
    hass: HomeAssistant, setup_home: None, config_entry: MockConfigEntry
) -> None:
    """Live observe_transition runs in a worker thread, not in the event loop."""
    threads: list[bool] = []
    original = KernelBayesReturnModel.observe_transition

    def _observe(self: KernelBayesReturnModel, *args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread() is threading.main_thread())
        return original(self, *args, **kwargs)

    with patch.object(KernelBayesReturnModel, "observe_transition", _observe):
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        coordinator = config_entry.runtime_data
        await _leave(hass, ALICE, "Alice")
        await hass.async_block_till_done()
    assert coordinator.persons[ALICE].is_home is False
    assert threads and not any(threads)


async def test_setup_catch_up_observed_in_executor(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """A departure missed while stopped is fed to the model from a worker thread."""
    hass_storage[f"{DOMAIN}.{config_entry.entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{config_entry.entry_id}",
        "data": {"schema": 1, "persons": {ALICE: {"model": {}, "is_home": True}}},
    }
    hass.states.async_set(ALICE, "not_home", person_attrs("Alice", _offset(5_000)))
    calls: list[bool] = []
    original = KernelBayesReturnModel.observe_transition

    def _observe(self: KernelBayesReturnModel, *args: Any, **kwargs: Any) -> Any:
        calls.append(threading.current_thread() is threading.main_thread())
        return original(self, *args, **kwargs)

    with patch.object(KernelBayesReturnModel, "observe_transition", _observe):
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    assert calls == [False]


# --------------------------------------------------------------------------- H2


async def test_failed_first_refresh_does_not_leak_listeners(
    hass: HomeAssistant, setup_home: None, config_entry: MockConfigEntry
) -> None:
    """A failed first refresh unsubscribes; the retry subscribes exactly once."""
    created: list[HRPCoordinator] = []

    async def _fail(self: HRPCoordinator) -> Any:
        created.append(self)
        raise UpdateFailed("boom")

    with patch.object(HRPCoordinator, "_async_update_data", _fail):
        assert not await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.SETUP_RETRY
    failed = created[0]
    assert failed._unsub_tracking is None
    assert failed._unsub_stop is None

    await hass.config_entries.async_reload(config_entry.entry_id)
    await hass.async_block_till_done()
    assert config_entry.state is ConfigEntryState.LOADED
    live = config_entry.runtime_data
    assert live is not failed
    await _leave(hass, ALICE, "Alice")
    # Only the live coordinator tracks the departure (the failed one is unsubscribed).
    assert live.persons[ALICE].is_home is False
    assert failed.persons[ALICE].is_home is True


# --------------------------------------------------------------------------- H3


async def test_return_event_once_per_absence_with_hysteresis(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
    forced: dict[str, tuple[float, timedelta | None]],
) -> None:
    """One event per absence; returning turns off only below threshold - 0.05."""
    coordinator = loaded_entry.runtime_data
    events = async_capture_events(hass, EVENT_RETURN_DETECTED)
    freezer.tick(timedelta(minutes=1))
    await _leave(hass, BOB, "Bob")

    async def _refresh(p: float) -> str:
        forced[BOB] = (p, timedelta(minutes=20))
        await coordinator.async_refresh()
        await hass.async_block_till_done()
        return hass.states.get("binary_sensor.bob_returning").state

    assert await _refresh(0.5) == "off"
    assert await _refresh(0.65) == "on"
    assert len(events) == 1
    assert await _refresh(0.57) == "on"  # within hysteresis (0.6 - 0.05)
    assert await _refresh(0.54) == "off"
    assert await _refresh(0.7) == "on"
    assert len(events) == 1  # same absence: no second event
    runtime = coordinator.persons[BOB]
    assert runtime.event_left_at == runtime.left_at

    # New absence -> new event.
    hass.states.async_set(BOB, "home", person_attrs("Bob"))
    await hass.async_block_till_done()
    freezer.tick(timedelta(minutes=5))
    await _leave(hass, BOB, "Bob")
    assert await _refresh(0.7) == "on"
    assert len(events) == 2


async def test_return_event_not_refired_after_restart(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
    forced: dict[str, tuple[float, timedelta | None]],
    hass_storage: dict[str, Any],
) -> None:
    """event_left_at is persisted: a reload during the same absence does not re-fire."""
    coordinator = loaded_entry.runtime_data
    events = async_capture_events(hass, EVENT_RETURN_DETECTED)
    freezer.tick(timedelta(minutes=1))
    await _leave(hass, BOB, "Bob")
    forced[BOB] = (0.9, timedelta(minutes=10))
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert len(events) == 1
    left_at = coordinator.persons[BOB].left_at

    assert await hass.config_entries.async_reload(loaded_entry.entry_id)
    await hass.async_block_till_done()
    stored = hass_storage[f"{DOMAIN}.{loaded_entry.entry_id}"]["data"]["persons"][BOB]
    assert stored["event_left_at"] == left_at.isoformat()
    coordinator = loaded_entry.runtime_data
    assert coordinator.persons[BOB].event_left_at == left_at
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert hass.states.get("binary_sensor.bob_returning").state == "on"
    assert len(events) == 1


# --------------------------------------------------------------------------- H4


async def test_save_only_when_changed(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """Unchanged updates schedule no save; a change schedules one with the long delay."""
    coordinator = loaded_entry.runtime_data
    assert SAVE_DELAY_SECONDS == 600
    with patch.object(coordinator.storage._store, "async_delay_save") as delay_save:
        await coordinator.async_refresh()
        await coordinator.async_refresh()
        await coordinator.async_refresh()
        assert delay_save.call_count == 0
        await coordinator.async_set_mode(ALICE, "live")
        assert delay_save.call_count == 1
        assert delay_save.call_args.args[1] == SAVE_DELAY_SECONDS
        await coordinator.async_refresh()
        assert delay_save.call_count == 1


async def test_final_write_on_stop(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, hass_storage: dict[str, Any]
) -> None:
    """A pending delayed save is written on Home Assistant's final write."""
    coordinator = loaded_entry.runtime_data
    key = f"{DOMAIN}.{loaded_entry.entry_id}"
    hass_storage.pop(key, None)
    await coordinator.async_set_mode(ALICE, "live")
    assert key not in hass_storage  # delayed
    await hass.async_stop(force=True)
    assert hass_storage[key]["data"]["persons"][ALICE]["mode"] == "live"


# --------------------------------------------------------------------------- H5 / H6


async def test_house_eta_from_medium_probability(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    freezer: FrozenDateTimeFactory,
    forced: dict[str, tuple[float, timedelta | None]],
) -> None:
    """House ETA uses away persons with p >= 0.25 even when nobody is 'returning'."""
    coordinator = loaded_entry.runtime_data
    freezer.tick(timedelta(minutes=1))
    await _leave(hass, ALICE, "Alice")
    await _leave(hass, BOB, "Bob")
    forced[ALICE] = (0.45, timedelta(minutes=30))
    forced[BOB] = (0.2, timedelta(minutes=10))  # earlier but below 0.25
    await coordinator.async_refresh()
    await hass.async_block_till_done()
    assert hass.states.get("binary_sensor.house_someone_returning").state == "off"
    eta_state = hass.states.get("sensor.house_next_return_eta")
    expected = coordinator.data.persons[ALICE].prediction.eta
    assert coordinator.data.house.next_eta == expected
    assert dt_util.parse_datetime(eta_state.state) == expected.replace(microsecond=0)
    assert eta_state.attributes["eta_person"] == ALICE
    # Person ETA is published whenever away, even at low probability.
    assert hass.states.get("sensor.bob_return_eta").state not in ("unknown", "unavailable")
    assert hass.states.get("binary_sensor.alice_returning").state == "off"


async def test_unknown_person_excluded(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
) -> None:
    """A person never seen with a known state is neither home nor away."""
    hass.states.async_set(BOB, "unknown", {"friendly_name": "Bob"})
    hass.states.async_set(ALICE, "not_home", person_attrs("Alice", _offset(5_000)))
    with patch.object(KernelBayesReturnModel, "log_prediction", autospec=True) as log:
        log.return_value = False
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done()
        coordinator = config_entry.runtime_data
        assert all(call.args[0] is coordinator.persons[ALICE].model for call in log.call_args_list)
    result = coordinator.data.persons[BOB]
    assert result.prediction is None and result.state_unknown
    house = coordinator.data.house
    assert house.persons_unknown == [BOB]
    assert house.persons_home == []
    assert house.probability == pytest.approx(
        coordinator.data.persons[ALICE].prediction.probability
    )
    assert hass.states.get("sensor.bob_return_probability").state == "unknown"
    assert hass.states.get("sensor.bob_return_probability").attributes["state_unknown"] is True
    assert hass.states.get("binary_sensor.bob_returning").attributes["state_unknown"] is True
    assert hass.states.get("sensor.house_return_probability").attributes["persons_unknown"] == [BOB]


# --------------------------------------------------------------------------- H7


def _st(
    state: str, ts: datetime, lat: float | None = None, changed: datetime | None = None
) -> State:
    attrs = {"latitude": lat, "longitude": HOME_LON, "gps_accuracy": 10} if lat else {}
    return State(ALICE, state, attrs, last_updated=ts, last_changed=changed or ts)


def test_history_uses_last_changed_and_downsamples() -> None:
    """Transitions use last_changed; fixes are kept at most one per minute."""
    t0 = dt_util.utcnow() - timedelta(hours=5)
    states = [
        # include_start_time_state row: last_updated rewritten, last_changed exact
        _st("home", t0 + timedelta(minutes=1), HOME_LAT, changed=t0),
        _st("not_home", t0 + timedelta(minutes=10), _offset(1000)),
        *[
            _st(
                "not_home",
                t0 + timedelta(minutes=10, seconds=10 * i),
                _offset(1000 + i * 10),
                changed=t0 + timedelta(minutes=10),
            )
            for i in range(1, 13)
        ],
    ]
    transitions, fixes = states_to_history(states, "zone.home", WORK)
    assert transitions[0] == PresenceTransition(ts=t0, is_home=True)
    assert transitions[1].ts == t0 + timedelta(minutes=10)
    # 13 rows over 2 minutes past the first fix -> 1 + 2 fixes (+ the home fix)
    gaps = [(b.ts - a.ts).total_seconds() for a, b in zip(fixes, fixes[1:], strict=False)]
    assert all(g >= 60 for g in gaps)
    assert len(fixes) == 4


def test_history_custom_zone_inaccurate_fix_is_no_information() -> None:
    """In a non-default zone an inaccurate fix does not create a departure."""
    t0 = dt_util.utcnow() - timedelta(hours=5)
    inside = {"latitude": HOME_LAT, "longitude": HOME_LON, "gps_accuracy": 10}
    blurry = {"latitude": _offset(5000), "longitude": HOME_LON, "gps_accuracy": 900}
    states = [
        State(ALICE, "not_home", inside, last_updated=t0, last_changed=t0),
        State(ALICE, "not_home", blurry, last_updated=t0 + timedelta(minutes=5), last_changed=t0),
    ]
    transitions, _ = states_to_history(states, "zone.work", WORK)
    assert [t.is_home for t in transitions] == [True]


def test_history_chunked_queries(hass: HomeAssistant) -> None:
    """The recorder is queried in 7-day windows and chunks are reduced incrementally."""
    end = dt_util.utcnow()
    start = end - timedelta(days=20)
    windows: list[tuple[datetime, datetime]] = []

    def _fetch(_hass: Any, _entity: str, s: datetime, e: datetime) -> list[State]:
        windows.append((s, e))
        return [_st("not_home" if len(windows) % 2 else "home", s + timedelta(hours=1))]

    with patch("custom_components.home_return_predictor.history._fetch_states", _fetch):
        transitions, _ = _load_history_chunked(hass, ALICE, "zone.home", WORK, start, end)
    assert len(windows) == 3
    assert all(e - s <= timedelta(days=7) for s, e in windows)
    assert windows[0][0] == start and windows[-1][1] == end
    assert [t.is_home for t in transitions] == [False, True, False]


def test_holiday_multi_day_expansion() -> None:
    """All-day multi-day events cover every date (end exclusive)."""
    assert _event_dates({"start": "2026-08-10", "end": "2026-08-13"}) == [
        date(2026, 8, 10),
        date(2026, 8, 11),
        date(2026, 8, 12),
    ]
    assert _event_dates({"start": "2026-08-10", "end": "2026-08-11"}) == [date(2026, 8, 10)]
    tz = dt_util.get_default_time_zone()
    start = datetime(2026, 8, 10, 9, tzinfo=tz).isoformat()
    end = datetime(2026, 8, 11, 18, tzinfo=tz).isoformat()
    assert _event_dates({"start": start, "end": end}) == [date(2026, 8, 10), date(2026, 8, 11)]
    assert _event_dates({"start": "garbage"}) == []


# --------------------------------------------------------------------------- H8


async def test_bootstrap_left_at_from_history(
    hass: HomeAssistant, setup_home: None, config_entry: MockConfigEntry
) -> None:
    """Without stored data, an earlier departure from history replaces boot last_changed."""
    departed = dt_util.utcnow() - timedelta(hours=3)
    history = (
        [
            PresenceTransition(ts=departed - timedelta(hours=10), is_home=True),
            PresenceTransition(ts=departed, is_home=False),
        ],
        [],
    )
    hass.states.async_set(ALICE, "not_home", person_attrs("Alice", _offset(5_000)))
    with patch(f"{COORD}.async_load_person_history", return_value=history):
        assert await hass.config_entries.async_setup(config_entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
    runtime = config_entry.runtime_data.persons[ALICE]
    assert runtime.left_at == departed


# --------------------------------------------------------------------------- H9


def test_build_reason_uses_shift_label() -> None:
    """The reason text shows the model's shift label."""
    left = dt_util.utcnow() - timedelta(hours=2)
    prediction = Prediction(
        probability=0.4,
        prior_probability=0.3,
        likelihood_ratio=1.2,
        eta=None,
        eta_habit=None,
        eta_position=None,
        approach_speed_kmh=None,
        approach_class=None,
        distance_m=None,
        is_home=False,
        reason={"shift_label": "morning"},
    )
    assert "(morning)" in build_reason(prediction, left)
