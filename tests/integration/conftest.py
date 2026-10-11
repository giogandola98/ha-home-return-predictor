"""Fixtures for Home Assistant integration tests."""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# Repo root on sys.path so Home Assistant can import ``custom_components``.
_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from homeassistant.core import HomeAssistant  # noqa: E402
from homeassistant.setup import async_setup_component  # noqa: E402
from pytest_homeassistant_custom_component.common import MockConfigEntry  # noqa: E402

from custom_components.home_return_predictor.const import (  # noqa: E402
    CONF_HOME_ZONE,
    CONF_PERSONS,
    DOMAIN,
)

ALICE = "person.alice"
BOB = "person.bob"
HOME_LAT = 45.0
HOME_LON = 9.0


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable custom integrations in every test."""


@pytest.fixture(autouse=True)
def no_history() -> Generator[None]:
    """Never hit the recorder during tests."""
    with patch(
        "custom_components.home_return_predictor.coordinator.async_load_person_history",
        return_value=([], []),
    ):
        yield


def person_attrs(name: str, lat: float = HOME_LAT, lon: float = HOME_LON, **extra: Any) -> dict:
    """Attributes of a person entity with one GPS tracker."""
    slug = name.lower()
    return {
        "friendly_name": name,
        "latitude": lat,
        "longitude": lon,
        "gps_accuracy": 10,
        "source": f"device_tracker.{slug}_phone",
        "device_trackers": [f"device_tracker.{slug}_phone"],
        **extra,
    }


@pytest.fixture
async def setup_home(hass: HomeAssistant) -> None:
    """Configure zone.home and two persons at home."""
    hass.config.latitude = HOME_LAT
    hass.config.longitude = HOME_LON
    hass.config.radius = 100
    assert await async_setup_component(hass, "zone", {})
    await hass.async_block_till_done()
    hass.states.async_set(ALICE, "home", person_attrs("Alice"))
    hass.states.async_set(BOB, "home", person_attrs("Bob"))


@pytest.fixture
def config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """A config entry for both persons."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], CONF_HOME_ZONE: "zone.home"},
        options={},
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
async def loaded_entry(
    hass: HomeAssistant, setup_home: None, config_entry: MockConfigEntry
) -> MockConfigEntry:
    """Set up the integration."""
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    return config_entry
