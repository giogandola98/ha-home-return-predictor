"""sensor.house_next_arrival and sensor.house_expected_absence."""

from __future__ import annotations

from datetime import datetime, timedelta

from freezegun.api import FrozenDateTimeFactory
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    CONF_ARRIVAL_QUANTILE,
    CONF_HOME_ZONE,
    CONF_PERSONS,
    DOMAIN,
)
from custom_components.home_return_predictor.hrp_core.base import PresenceTransition

from .conftest import ALICE, BOB, HOME_LAT, HOME_LON, person_attrs

NEXT = "sensor.house_next_arrival"
ABSENCE = "sensor.house_expected_absence"


def _habits(now: datetime, hours: float) -> list[PresenceTransition]:
    """40 past days: out at the current time of day, back after ``hours``."""
    out = []
    for i in range(1, 41):
        left = now - timedelta(days=i)
        out += [
            PresenceTransition(left, False),
            PresenceTransition(left + timedelta(hours=hours), True),
        ]
    return out


async def _setup(hass: HomeAssistant, options: dict) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], CONF_HOME_ZONE: "zone.home"},
        options=options,
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    now = dt_util.utcnow()
    coordinator = entry.runtime_data
    coordinator.persons[ALICE].model.load_history(_habits(now, 6))
    coordinator.persons[BOB].model.load_history(_habits(now, 10))
    return entry


async def _leave(hass: HomeAssistant, person: str, name: str) -> None:
    lat = HOME_LAT + 3_000 / 111_195
    hass.states.async_set(person, "not_home", person_attrs(name, lat, HOME_LON))
    await hass.async_block_till_done()


async def test_house_arrival_sensors(
    hass: HomeAssistant, setup_home: None, freezer: FrozenDateTimeFactory
) -> None:
    entry = await _setup(hass, {})
    coordinator = entry.runtime_data
    await coordinator.async_refresh()
    # someone home: unknown
    assert hass.states.get(NEXT).state == "unknown"
    assert hass.states.get(ABSENCE).state == "unknown"
    assert hass.states.get(NEXT).attributes["quantile"] == 0.25

    await _leave(hass, ALICE, "Alice")
    await _leave(hass, BOB, "Bob")
    freezer.tick(timedelta(hours=1))
    await coordinator.async_refresh()
    now = dt_util.utcnow()
    state = hass.states.get(NEXT)
    arrival = dt_util.parse_datetime(state.state)
    # Alice is habitually back 6 h after leaving: 5 h from now (q25 a bit earlier)
    assert now < arrival < now + timedelta(hours=5, minutes=30)
    assert arrival > now + timedelta(hours=3)
    attrs = state.attributes
    assert set(attrs["quantiles"]) == {"q10", "q25", "q50", "q75", "q90"}
    assert attrs["quantiles"]["q25"] == state.state
    assert [m for m, _ in attrs["cdf"]] == [15, 30, 60, 90, 120, 180, 240, 360, 480, 720, 1440]
    probs = [p for _, p in attrs["cdf"]]
    assert probs == sorted(probs) and probs[0] < 0.05 and probs[-1] > 0.9
    assert attrs["source"] in ("habit", "gps")
    assert set(attrs["persons"]) == {ALICE, BOB}
    assert attrs["persons"][ALICE]["q50"] < attrs["persons"][BOB]["q50"]
    assert attrs["mode"] == "learning" and attrs["advisory"] is True

    absence = hass.states.get(ABSENCE)
    assert 4.0 < float(absence.state) < 6.0  # median remaining empty time (hours)
    assert absence.attributes["unit_of_measurement"] == "h"
    assert absence.attributes["q25"] <= float(absence.state) <= absence.attributes["q75"]
    assert absence.attributes["absence_class"] == "day"
    assert dt_util.parse_datetime(absence.attributes["empty_since"]) < now

    # the old ETA sensor keeps its 90-min gate (nothing changes for its users)
    assert hass.states.get("sensor.house_next_return_eta").state == "unknown"

    # Alice comes back: unknown again
    hass.states.async_set(ALICE, "home", person_attrs("Alice"))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    assert hass.states.get(NEXT).state == "unknown"
    assert hass.states.get(ABSENCE).state == "unknown"


async def test_arrival_quantile_option(
    hass: HomeAssistant, setup_home: None, freezer: FrozenDateTimeFactory
) -> None:
    entry = await _setup(hass, {CONF_ARRIVAL_QUANTILE: 0.75})
    coordinator = entry.runtime_data
    await _leave(hass, ALICE, "Alice")
    await _leave(hass, BOB, "Bob")
    freezer.tick(timedelta(hours=1))
    await coordinator.async_refresh()
    state = hass.states.get(NEXT)
    assert state.attributes["quantile"] == 0.75
    assert state.state == state.attributes["quantiles"]["q75"]
