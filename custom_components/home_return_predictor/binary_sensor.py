"""Binary sensor platform."""

from __future__ import annotations

from typing import Any

from homeassistant.components.binary_sensor import ENTITY_ID_FORMAT, BinarySensorEntity
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import KEY_HOUSE_SOMEONE_RETURNING, KEY_RETURNING
from .coordinator import HRPConfigEntry, HRPCoordinator
from .entity import HRPHouseEntity, HRPPersonEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HRPConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up binary sensors."""
    coordinator = entry.runtime_data
    entities: list[BinarySensorEntity] = [
        ReturningBinarySensor(coordinator, person_id) for person_id in coordinator.person_ids
    ]
    entities.append(HouseSomeoneReturningBinarySensor(coordinator))
    async_add_entities(entities)


class ReturningBinarySensor(HRPPersonEntity, BinarySensorEntity):
    """On while the person is predicted to be returning."""

    def __init__(self, coordinator: HRPCoordinator, person_id: str) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, person_id, KEY_RETURNING, ENTITY_ID_FORMAT)

    @property
    def is_on(self) -> bool | None:
        """Return True when returning."""
        result = self.result
        if result is None or result.state_unknown:
            return None
        return result.returning

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return prediction details."""
        result = self.result
        if result is None:
            return {}
        pred = result.prediction
        if pred is None:
            return {
                "state_unknown": True,
                "mode": result.effective_mode,
                "advisory": result.advisory,
                "reason": result.reason,
                "active_model": result.active_model,
            }

        def _round(value: float | None, digits: int) -> float | None:
            return round(value, digits) if value is not None else None

        return {
            "probability": round(pred.probability, 3),
            "prior_probability": _round(pred.prior_probability, 3),
            "likelihood_ratio": round(pred.likelihood_ratio, 2),
            "eta": pred.eta.isoformat() if pred.eta else None,
            "eta_habit": pred.eta_habit.isoformat() if pred.eta_habit else None,
            "eta_position": pred.eta_position.isoformat() if pred.eta_position else None,
            "approach_speed_kmh": _round(pred.approach_speed_kmh, 1),
            "distance_km": _round(
                pred.distance_m / 1000 if pred.distance_m is not None else None, 2
            ),
            "mode": result.effective_mode,
            "advisory": result.advisory,
            "reason": result.reason,
            "active_model": result.active_model,
            "state_unknown": False,
        }


class HouseSomeoneReturningBinarySensor(HRPHouseEntity, BinarySensorEntity):
    """On while at least one person is returning."""

    def __init__(self, coordinator: HRPCoordinator) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator, KEY_HOUSE_SOMEONE_RETURNING, ENTITY_ID_FORMAT)

    @property
    def is_on(self) -> bool | None:
        """Return True when someone is returning."""
        house = self.house
        return house.someone_returning if house else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return aggregate attributes."""
        attrs = self.house_attributes()
        if self.coordinator.data:
            attrs["returning_names"] = [
                self.coordinator.data.persons[pid].name
                for pid in attrs.get("returning_persons", [])
            ]
        return attrs
