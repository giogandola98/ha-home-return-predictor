"""Select platform: per-person operating mode."""

from __future__ import annotations

from homeassistant.components.select import ENTITY_ID_FORMAT, SelectEntity
from homeassistant.const import EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import KEY_MODE, MODES
from .coordinator import HRPConfigEntry, HRPCoordinator
from .entity import HRPPersonEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HRPConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up mode selects."""
    coordinator = entry.runtime_data
    async_add_entities(ModeSelect(coordinator, pid) for pid in coordinator.person_ids)


class ModeSelect(HRPPersonEntity, SelectEntity):
    """auto / learning / live; persisted in the integration store."""

    _attr_options = MODES
    _attr_entity_category = EntityCategory.CONFIG

    def __init__(self, coordinator: HRPCoordinator, person_id: str) -> None:
        """Initialize the select."""
        super().__init__(coordinator, person_id, KEY_MODE, ENTITY_ID_FORMAT)

    @property
    def available(self) -> bool:
        """Always available: the mode does not depend on predictions."""
        return True

    @property
    def current_option(self) -> str:
        """Return the selected mode."""
        return self.runtime.mode

    @property
    def extra_state_attributes(self) -> dict[str, str | None]:
        """Expose the effective mode."""
        return {"mode": self.runtime.effective_mode}

    async def async_select_option(self, option: str) -> None:
        """Change the mode."""
        await self.coordinator.async_set_mode(self.person_id, option)
