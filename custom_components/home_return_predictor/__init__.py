"""Home Return Predictor: predicts who is coming home, when, and how sure it is."""

from __future__ import annotations

import json
from pathlib import Path

import voluptuous as vol
from homeassistant.const import ATTR_ENTITY_ID, Platform
from homeassistant.core import HomeAssistant, ServiceCall, ServiceResponse, SupportsResponse
from homeassistant.exceptions import ServiceValidationError
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.device_registry import DeviceEntry
from homeassistant.helpers.json import ExtendedJSONEncoder
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import dt as dt_util

from .const import (
    ATTR_PERSON,
    DOMAIN,
    HOUSE_DEVICE_SLUG,
    SERVICE_EXPORT_DIAGNOSTICS,
    SERVICE_RESET,
    SERVICE_RETRAIN,
)
from .coordinator import HRPConfigEntry, HRPCoordinator
from .diagnostics import async_get_config_entry_diagnostics

PLATFORMS: list[Platform] = [Platform.BINARY_SENSOR, Platform.SELECT, Platform.SENSOR]

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

RETRAIN_SCHEMA = vol.Schema({vol.Optional(ATTR_PERSON): cv.entity_ids})
RESET_SCHEMA = vol.Schema({vol.Required(ATTR_PERSON): cv.entity_ids})


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Register services (once, independent of config entries)."""

    def _entry() -> HRPConfigEntry:
        entries: list[HRPConfigEntry] = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise ServiceValidationError(translation_domain=DOMAIN, translation_key="not_loaded")
        return entries[0]

    def _persons(coordinator: HRPCoordinator, call: ServiceCall) -> list[str]:
        requested: list[str] = call.data.get(ATTR_PERSON) or list(coordinator.persons)
        for person_id in requested:
            if person_id not in coordinator.persons:
                raise ServiceValidationError(
                    translation_domain=DOMAIN,
                    translation_key="unknown_person",
                    translation_placeholders={ATTR_ENTITY_ID: person_id},
                )
        return requested

    async def _retrain(call: ServiceCall) -> None:
        coordinator = _entry().runtime_data
        await coordinator.async_retrain(_persons(coordinator, call))

    async def _reset(call: ServiceCall) -> None:
        coordinator = _entry().runtime_data
        for person_id in _persons(coordinator, call):
            await coordinator.async_reset(person_id)

    async def _export(call: ServiceCall) -> ServiceResponse:
        entry = _entry()
        data = await async_get_config_entry_diagnostics(hass, entry)
        stamp = dt_util.now().strftime("%Y%m%d_%H%M%S")
        path = hass.config.path(f"{DOMAIN}_diagnostics_{stamp}.json")

        def _write() -> None:
            Path(path).write_text(
                json.dumps(data, cls=ExtendedJSONEncoder, indent=2), encoding="utf-8"
            )

        await hass.async_add_executor_job(_write)
        return {"path": path}

    hass.services.async_register(DOMAIN, SERVICE_RETRAIN, _retrain, schema=RETRAIN_SCHEMA)
    hass.services.async_register(DOMAIN, SERVICE_RESET, _reset, schema=RESET_SCHEMA)
    hass.services.async_register(
        DOMAIN,
        SERVICE_EXPORT_DIAGNOSTICS,
        _export,
        supports_response=SupportsResponse.OPTIONAL,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: HRPConfigEntry) -> bool:
    """Set up from a config entry."""
    coordinator = HRPCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass: HomeAssistant, entry: HRPConfigEntry) -> bool:
    """Unload a config entry and persist learnt models."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if unloaded:
        await entry.runtime_data.async_unload()
    return unloaded


async def async_remove_config_entry_device(
    hass: HomeAssistant, entry: HRPConfigEntry, device: DeviceEntry
) -> bool:
    """Allow removing devices of persons no longer configured."""
    entry_id = entry.entry_id
    active = {f"{entry_id}_{HOUSE_DEVICE_SLUG}"} | {
        f"{entry_id}_{pid.split('.', 1)[1]}" for pid in entry.runtime_data.person_ids
    }
    return not any(dom == DOMAIN and ident in active for dom, ident in device.identifiers)
