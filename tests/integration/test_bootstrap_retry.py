"""History bootstrap must be retried until it has succeeded once.

Regression: the bootstrap ran only when no model was stored, but the first update stores the
(empty) model snapshot. A recorder error or a restart during the bootstrap therefore left the
person on an empty model forever ("learning" for weeks).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import DOMAIN
from custom_components.home_return_predictor.history import HistoryLoadError
from custom_components.home_return_predictor.hrp_core.base import PresenceTransition

from .conftest import ALICE

COORD = "custom_components.home_return_predictor.coordinator"


def _history() -> tuple[list[PresenceTransition], list]:
    left = dt_util.utcnow() - timedelta(days=2)
    return (
        [
            PresenceTransition(ts=left - timedelta(hours=12), is_home=True),
            PresenceTransition(ts=left, is_home=False),
            PresenceTransition(ts=left + timedelta(hours=9), is_home=True),
        ],
        [],
    )


async def _setup(hass: HomeAssistant, entry: MockConfigEntry, loader: AsyncMock) -> None:
    with patch(f"{COORD}.async_load_person_history", loader):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)


async def _reload(hass: HomeAssistant, entry: MockConfigEntry, loader: AsyncMock) -> None:
    with patch(f"{COORD}.async_load_person_history", loader):
        assert await hass.config_entries.async_reload(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)


async def test_failed_bootstrap_is_retried_on_next_start(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
    hass_storage: dict[str, Any],
) -> None:
    hass.config.components.add("recorder")
    failing = AsyncMock(side_effect=HistoryLoadError("database is locked"))
    await _setup(hass, config_entry, failing)
    assert failing.await_count == 2  # both persons tried
    coordinator = config_entry.runtime_data
    assert coordinator.persons[ALICE].bootstrapped is False

    # The reload saves on unload: the empty model is stored but not marked bootstrapped,
    # so the next start reads the history again, and this time it works.
    working = AsyncMock(return_value=_history())
    await _reload(hass, config_entry, working)
    assert working.await_count == 2
    assert config_entry.runtime_data.persons[ALICE].bootstrapped is True

    await hass.config_entries.async_unload(config_entry.entry_id)
    stored = hass_storage[f"{DOMAIN}.{config_entry.entry_id}"]["data"]["persons"][ALICE]
    assert stored["model"]
    assert stored["bootstrapped"] is True


async def test_successful_bootstrap_is_not_repeated(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
) -> None:
    hass.config.components.add("recorder")
    working = AsyncMock(return_value=_history())
    await _setup(hass, config_entry, working)
    assert working.await_count == 2

    again = AsyncMock(return_value=_history())
    await _reload(hass, config_entry, again)
    assert again.await_count == 0
    assert config_entry.runtime_data.persons[ALICE].bootstrapped is True


async def test_without_recorder_bootstrap_stays_pending(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
) -> None:
    """No recorder yet: nothing was really read, so try again once it is available."""
    empty = AsyncMock(return_value=([], []))
    await _setup(hass, config_entry, empty)
    assert config_entry.runtime_data.persons[ALICE].bootstrapped is False
