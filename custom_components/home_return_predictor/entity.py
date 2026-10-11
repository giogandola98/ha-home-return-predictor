"""Base entities and device info."""

from __future__ import annotations

from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, HOUSE_DEVICE_SLUG, NAME
from .coordinator import HouseResult, HRPCoordinator, PersonResult, PersonRuntime


class HRPEntity(CoordinatorEntity[HRPCoordinator]):
    """Common base: translated names, fixed entity_id independent of language."""

    _attr_has_entity_name = True

    def __init__(self, coordinator: HRPCoordinator, key: str, entity_id_format: str) -> None:
        """Initialize the entity."""
        super().__init__(coordinator)
        self._attr_translation_key = key
        self._key = key
        self._entity_id_format = entity_id_format


class HRPPersonEntity(HRPEntity):
    """Entity attached to a person device."""

    def __init__(
        self,
        coordinator: HRPCoordinator,
        person_id: str,
        key: str,
        entity_id_format: str,
    ) -> None:
        """Initialize the entity."""
        super().__init__(coordinator, key, entity_id_format)
        self.person_id = person_id
        slug = person_id.split(".", 1)[1]
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_{slug}_{key}"
        # Explicit entity_id so it never depends on the UI language.
        self.entity_id = entity_id_format.format(f"{slug}_{key}")
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry_id}_{slug}")},
            name=coordinator.person_name(person_id),
            manufacturer=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def runtime(self) -> PersonRuntime:
        """Runtime state of the person."""
        return self.coordinator.persons[self.person_id]

    @property
    def result(self) -> PersonResult | None:
        """Latest result for the person."""
        data = self.coordinator.data
        return data.persons.get(self.person_id) if data else None

    @property
    def available(self) -> bool:
        """Available when a result exists."""
        return super().available and self.result is not None


class HRPHouseEntity(HRPEntity):
    """Entity attached to the house device."""

    def __init__(self, coordinator: HRPCoordinator, key: str, entity_id_format: str) -> None:
        """Initialize the entity."""
        super().__init__(coordinator, key, entity_id_format)
        entry_id = coordinator.config_entry.entry_id
        self._attr_unique_id = f"{entry_id}_{key}"
        self.entity_id = entity_id_format.format(key)
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, f"{entry_id}_{HOUSE_DEVICE_SLUG}")},
            translation_key=HOUSE_DEVICE_SLUG,
            manufacturer=NAME,
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def house(self) -> HouseResult | None:
        """Latest aggregate."""
        data = self.coordinator.data
        return data.house if data else None

    @property
    def available(self) -> bool:
        """Available when a result exists."""
        return super().available and self.house is not None

    def house_attributes(self) -> dict[str, object]:
        """Attributes shared by all house entities (used by blueprints)."""
        house = self.house
        if house is None:
            return {}
        return {
            "occupied": house.occupied,
            "persons_home": house.persons_home,
            "returning_persons": house.returning_persons,
            "mode": house.mode,
            "advisory": house.advisory,
            "away_probability": round(house.away_probability * 100, 1),
        }
