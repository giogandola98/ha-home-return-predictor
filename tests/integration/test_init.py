"""Setup, unload, storage and services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ServiceValidationError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    DOMAIN,
    MODE_LIVE,
    SERVICE_EXPORT_DIAGNOSTICS,
    SERVICE_RESET,
    SERVICE_RETRAIN,
)

from .conftest import ALICE, BOB


async def test_setup_and_unload(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, hass_storage: dict[str, Any]
) -> None:
    """Entry loads, then unloads and persists models."""
    assert loaded_entry.state is ConfigEntryState.LOADED
    coordinator = loaded_entry.runtime_data
    assert set(coordinator.persons) == {ALICE, BOB}
    coordinator.persons[ALICE].mode = MODE_LIVE

    assert await hass.config_entries.async_unload(loaded_entry.entry_id)
    await hass.async_block_till_done()
    assert loaded_entry.state is ConfigEntryState.NOT_LOADED

    stored = hass_storage[f"{DOMAIN}.{loaded_entry.entry_id}"]["data"]
    assert stored["schema"] == 1
    assert stored["persons"][ALICE]["mode"] == MODE_LIVE
    assert stored["persons"][ALICE]["is_home"] is True
    assert isinstance(stored["persons"][ALICE]["model"], dict)


async def test_restore_from_storage(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    """Stored mode and model are restored; a missed departure is caught up."""
    hass_storage[f"{DOMAIN}.{config_entry.entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{config_entry.entry_id}",
        "data": {
            "schema": 1,
            "persons": {ALICE: {"model": {}, "mode": MODE_LIVE, "is_home": True}},
        },
    }
    hass.states.async_set(ALICE, "not_home", {"friendly_name": "Alice"})
    assert await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()
    runtime = config_entry.runtime_data.persons[ALICE]
    assert runtime.mode == MODE_LIVE
    assert runtime.is_home is False
    assert runtime.left_at is not None


async def test_services(hass: HomeAssistant, loaded_entry: MockConfigEntry, tmp_path: Path) -> None:
    """Services are registered and callable."""
    for service in (SERVICE_RETRAIN, SERVICE_RESET, SERVICE_EXPORT_DIAGNOSTICS):
        assert hass.services.has_service(DOMAIN, service)

    await hass.services.async_call(DOMAIN, SERVICE_RETRAIN, {}, blocking=True)
    await hass.services.async_call(DOMAIN, SERVICE_RETRAIN, {"person": ALICE}, blocking=True)
    await hass.services.async_call(DOMAIN, SERVICE_RESET, {"person": [ALICE, BOB]}, blocking=True)

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN, SERVICE_RESET, {"person": "person.nobody"}, blocking=True
        )

    hass.config.config_dir = str(tmp_path)
    response = await hass.services.async_call(
        DOMAIN, SERVICE_EXPORT_DIAGNOSTICS, {}, blocking=True, return_response=True
    )
    path = Path(response["path"])
    assert path.parent == tmp_path
    content = path.read_text(encoding="utf-8")
    assert ALICE in content


async def test_service_without_entry(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """Services raise a translated error when the entry is not loaded."""
    assert await hass.config_entries.async_unload(loaded_entry.entry_id)
    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(DOMAIN, SERVICE_RETRAIN, {}, blocking=True)
