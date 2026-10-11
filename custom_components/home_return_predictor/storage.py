"""Persistent storage for learnt models.

Stored document (``.storage/home_return_predictor.<entry_id>``)::

    {
      "schema": 1,
      "persons": {
        "person.alice": {
          "model": {...},          # ReturnModel.to_dict()
          "mode": "auto",          # select value
          "is_home": false,        # presence at last save (catch up on restart)
          "left_at": "2026-...",   # start of the current absence (ISO) or null
          "event_left_at": "...",  # absence whose return event already fired
        }
      }
    }
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .const import SAVE_DELAY_SECONDS, STORAGE_KEY_PREFIX, STORAGE_VERSION

STORAGE_SCHEMA = 1

type StoredData = dict[str, Any]


class HRPStore(Store[StoredData]):
    """Store with a migration hook for future schema changes."""

    async def _async_migrate_func(
        self,
        old_major_version: int,
        old_minor_version: int,
        old_data: StoredData,
    ) -> StoredData:
        """Migrate stored data to the current version.

        Only version 1 exists so far; add ``if old_major_version < N`` steps here.
        """
        return old_data


class HRPStorage:
    """Thin wrapper around :class:`HRPStore`."""

    def __init__(self, hass: HomeAssistant, entry_id: str) -> None:
        """Initialize the storage for one config entry."""
        self._store = HRPStore(hass, STORAGE_VERSION, f"{STORAGE_KEY_PREFIX}.{entry_id}")

    async def async_load(self) -> StoredData:
        """Load stored data, returning an empty document when nothing is stored."""
        data = await self._store.async_load()
        if not isinstance(data, dict):
            return {"schema": STORAGE_SCHEMA, "persons": {}}
        data.setdefault("persons", {})
        return data

    def async_schedule_save(self, data_func: Callable[[], StoredData]) -> None:
        """Schedule a delayed save (coalesces bursts of changes).

        ``Store`` also writes a pending delayed save on Home Assistant's final
        write event, so nothing is lost on shutdown.
        """
        self._store.async_delay_save(data_func, SAVE_DELAY_SECONDS)

    async def async_save(self, data: StoredData) -> None:
        """Save immediately."""
        await self._store.async_save(data)

    async def async_remove(self) -> None:
        """Delete the stored file."""
        await self._store.async_remove()
