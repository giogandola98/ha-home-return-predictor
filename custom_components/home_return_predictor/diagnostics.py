"""Diagnostics support."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from homeassistant.components.diagnostics import async_redact_data
from homeassistant.const import CONF_LATITUDE, CONF_LONGITUDE
from homeassistant.core import HomeAssistant

from .const import DIAGNOSTICS_LOG_ENTRIES
from .coordinator import HRPConfigEntry

TO_REDACT = {CONF_LATITUDE, CONF_LONGITUDE}


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: HRPConfigEntry
) -> dict[str, Any]:
    """Return diagnostics for a config entry."""
    coordinator = entry.runtime_data
    data = coordinator.data
    persons: dict[str, Any] = {}
    for person_id, runtime in coordinator.persons.items():
        result = data.persons.get(person_id) if data else None
        prediction = coordinator.last_predictions.get(person_id)
        persons[person_id] = {
            "mode": runtime.mode,
            "effective_mode": runtime.effective_mode,
            "is_home": runtime.is_home,
            "left_at": runtime.left_at,
            "trackers": list(runtime.trackers),
            "recent_fixes": len(runtime.fixes),
            "episodes": await coordinator.async_episode_count(person_id),
            "event_left_at": runtime.event_left_at,
            "maturity": asdict(result.maturity) if result else None,
            "returning": result.returning if result else None,
            "reason": result.reason if result else None,
            "active_model": result.active_model if result else None,
            "last_prediction": asdict(prediction) if prediction else None,
            "prediction_log": coordinator.prediction_log(person_id, DIAGNOSTICS_LOG_ENTRIES),
        }
    return async_redact_data(
        {
            "entry": {"data": dict(entry.data), "options": dict(entry.options)},
            "home_zone": coordinator.home_zone_id,
            "challenger": {
                "requested": coordinator.challenger_requested,
                "enabled": coordinator.challenger_enabled,
                "numpy_available": coordinator.numpy_available,
                "sklearn_available": coordinator.numpy_available,  # deprecated alias
                "training": sorted(coordinator.training_persons),
            },
            "persons": persons,
            "house": asdict(data.house) if data else None,
        },
        TO_REDACT,
    )
