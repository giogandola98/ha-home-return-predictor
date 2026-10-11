"""Entity, mode, prediction, aggregation and event tests."""

from __future__ import annotations

from datetime import timedelta

from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.helpers import entity_registry as er
from pytest_homeassistant_custom_component.common import MockConfigEntry, async_capture_events

from custom_components.home_return_predictor.const import (
    CONF_RETURN_THRESHOLD,
    EVENT_RETURN_DETECTED,
)

from .conftest import ALICE, BOB, HOME_LAT, HOME_LON, person_attrs

PERSON_ENTITIES = {
    "binary_sensor.{slug}_returning": "returning",
    "sensor.{slug}_return_probability": "return_probability",
    "sensor.{slug}_return_eta": "return_eta",
    "sensor.{slug}_model_maturity": "model_maturity",
    "sensor.{slug}_return_reason": "return_reason",
    "select.{slug}_mode": "mode",
}
HOUSE_ENTITIES = {
    "sensor.house_return_probability": "house_return_probability",
    "sensor.house_next_return_eta": "house_next_return_eta",
    "binary_sensor.house_someone_returning": "house_someone_returning",
}


async def test_entity_ids(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, entity_registry: er.EntityRegistry
) -> None:
    """All entities exist with stable entity_ids and unique_ids."""
    entry_id = loaded_entry.entry_id
    for slug in ("alice", "bob"):
        for pattern, key in PERSON_ENTITIES.items():
            entity_id = pattern.format(slug=slug)
            entry = entity_registry.async_get(entity_id)
            assert entry is not None, entity_id
            assert entry.unique_id == f"{entry_id}_{slug}_{key}"
            assert hass.states.get(entity_id) is not None
    for entity_id, key in HOUSE_ENTITIES.items():
        entry = entity_registry.async_get(entity_id)
        assert entry is not None, entity_id
        assert entry.unique_id == f"{entry_id}_{key}"

    house = hass.states.get("sensor.house_return_probability")
    assert float(house.state) == 100
    assert house.attributes["occupied"] is True
    assert set(house.attributes["persons_home"]) == {ALICE, BOB}
    assert house.attributes["mode"] == "learning"
    assert house.attributes["advisory"] is True
    assert hass.states.get("binary_sensor.alice_returning").state == "off"
    assert hass.states.get("sensor.alice_return_probability").attributes["mode"] == "learning"


async def test_select_changes_effective_mode(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Forcing live mode changes the effective mode everywhere."""
    assert hass.states.get("select.alice_mode").state == "auto"
    await hass.services.async_call(
        "select",
        "select_option",
        {"entity_id": "select.alice_mode", "option": "live"},
        blocking=True,
    )
    await hass.async_block_till_done()
    assert hass.states.get("select.alice_mode").state == "live"
    returning = hass.states.get("binary_sensor.alice_returning")
    assert returning.attributes["mode"] == "live"
    assert returning.attributes["advisory"] is False
    assert loaded_entry.runtime_data.persons[ALICE].effective_mode == "live"
    house = hass.states.get("sensor.house_return_probability")
    assert house.attributes["mode"] == "live"

    await hass.services.async_call(
        "select",
        "select_option",
        {"entity_id": "select.alice_mode", "option": "auto"},
        blocking=True,
    )
    await hass.async_block_till_done()
    # Fresh model: maturity 0 < threshold - hysteresis -> back to learning.
    assert hass.states.get("binary_sensor.alice_returning").attributes["mode"] == "learning"


def _offset(meters_north: float) -> float:
    return HOME_LAT + meters_north / 111_195


async def test_leave_approach_and_house(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    """Leaving and approaching changes probabilities; house aggregates correctly."""
    coordinator = loaded_entry.runtime_data

    # Bob leaves; Alice home -> house is 100 %.
    freezer.tick(timedelta(minutes=1))
    hass.states.async_set(BOB, "not_home", person_attrs("Bob", _offset(20_000), HOME_LON))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    assert coordinator.persons[BOB].is_home is False
    assert coordinator.persons[BOB].left_at is not None
    house = hass.states.get("sensor.house_return_probability")
    assert float(house.state) == 100
    assert house.attributes["persons_home"] == [ALICE]

    # Alice leaves too -> house = 1 - prod(1 - p_i).
    hass.states.async_set(ALICE, "not_home", person_attrs("Alice", _offset(20_000), HOME_LON))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    data = coordinator.data
    p_alice = data.persons[ALICE].prediction.probability
    p_bob = data.persons[BOB].prediction.probability
    expected = 1 - (1 - p_alice) * (1 - p_bob)
    house = hass.states.get("sensor.house_return_probability")
    assert house.attributes["occupied"] is False
    assert float(house.state) == round(expected * 100, 1)
    assert float(hass.states.get("sensor.alice_return_probability").state) == round(
        p_alice * 100, 1
    )
    distance_before = data.persons[ALICE].prediction.distance_m
    assert distance_before is not None and distance_before > 19_000

    # Alice drives home at ~60 km/h: fixes every minute through the tracker.
    distance = 20_000.0
    for _ in range(8):
        freezer.tick(timedelta(minutes=1))
        distance -= 1_000
        hass.states.async_set(
            "device_tracker.alice_phone",
            "not_home",
            {"latitude": _offset(distance), "longitude": HOME_LON, "gps_accuracy": 15},
        )
        await hass.async_block_till_done()
    # A low-accuracy fix is discarded.
    hass.states.async_set(
        "device_tracker.alice_phone",
        "not_home",
        {"latitude": _offset(50_000), "longitude": HOME_LON, "gps_accuracy": 900},
    )
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    alice = coordinator.data.persons[ALICE]
    assert alice.prediction.distance_m < distance_before
    assert alice.prediction.approach_speed_kmh is not None
    assert alice.prediction.approach_speed_kmh > 30
    assert alice.prediction.probability > p_alice
    attrs = hass.states.get("binary_sensor.alice_returning").attributes
    assert attrs["approach_speed_kmh"] > 30
    assert "km" in attrs["reason"]

    # Alice arrives -> house back to 100 %, transition fed to the model.
    hass.states.async_set(ALICE, "home", person_attrs("Alice"))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    assert float(hass.states.get("sensor.house_return_probability").state) == 100
    assert coordinator.persons[ALICE].fixes == type(coordinator.persons[ALICE].fixes)()


async def test_return_event_rising_edge(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, freezer: FrozenDateTimeFactory
) -> None:
    """The return event fires once on the rising edge."""
    hass.config_entries.async_update_entry(
        loaded_entry, options={**loaded_entry.options, CONF_RETURN_THRESHOLD: 0.0}
    )
    # OptionsFlowWithReload reloads only through the flow: reload explicitly.
    assert await hass.config_entries.async_reload(loaded_entry.entry_id)
    await hass.async_block_till_done()
    coordinator = loaded_entry.runtime_data
    events = async_capture_events(hass, EVENT_RETURN_DETECTED)

    freezer.tick(timedelta(minutes=1))
    hass.states.async_set(BOB, "not_home", person_attrs("Bob", _offset(5_000), HOME_LON))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    await coordinator.async_refresh()
    await hass.async_block_till_done()

    assert len(events) == 1
    data = events[0].data
    assert data["person"] == BOB
    assert data["name"] == "Bob"
    assert data["mode"] == "learning"
    assert data["advisory"] is True
    assert 0 <= data["probability"] <= 1
    assert hass.states.get("binary_sensor.bob_returning").state == "on"
    house = hass.states.get("binary_sensor.house_someone_returning")
    assert house.state == "on"
    assert house.attributes["returning_persons"] == [BOB]
    assert house.attributes["returning_names"] == ["Bob"]
