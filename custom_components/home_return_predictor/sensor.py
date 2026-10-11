"""Sensor platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from homeassistant.components.sensor import (
    ENTITY_ID_FORMAT,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    KEY_HOUSE_NEXT_RETURN_ETA,
    KEY_HOUSE_RETURN_PROBABILITY,
    KEY_MODEL_MATURITY,
    KEY_RETURN_ETA,
    KEY_RETURN_PROBABILITY,
    KEY_RETURN_REASON,
)
from .coordinator import HRPConfigEntry, HRPCoordinator
from .entity import HRPHouseEntity, HRPPersonEntity

PARALLEL_UPDATES = 0


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


async def async_setup_entry(
    hass: HomeAssistant,
    entry: HRPConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up sensors."""
    coordinator = entry.runtime_data
    entities: list[SensorEntity] = []
    for person_id in coordinator.person_ids:
        entities += [
            ReturnProbabilitySensor(coordinator, person_id),
            ReturnEtaSensor(coordinator, person_id),
            ModelMaturitySensor(coordinator, person_id),
            ReturnReasonSensor(coordinator, person_id),
        ]
    entities += [HouseReturnProbabilitySensor(coordinator), HouseNextReturnEtaSensor(coordinator)]
    async_add_entities(entities)


class HRPPersonSensor(HRPPersonEntity, SensorEntity):
    """Base person sensor."""

    _key_name: str

    def __init__(self, coordinator: HRPCoordinator, person_id: str) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, person_id, self._key_name, ENTITY_ID_FORMAT)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """All person sensors expose the effective mode."""
        result = self.result
        if result is None:
            return {}
        return {
            "mode": result.effective_mode,
            "advisory": result.advisory,
            "state_unknown": result.state_unknown,
        }


class ReturnProbabilitySensor(HRPPersonSensor):
    """Probability of returning within the horizon."""

    _key_name = KEY_RETURN_PROBABILITY
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    @property
    def native_value(self) -> float | None:
        """Return the probability in percent."""
        result = self.result
        if result is None or result.prediction is None:
            return None
        return round(result.prediction.probability * 100, 1)


class ReturnEtaSensor(HRPPersonSensor):
    """Estimated arrival time."""

    _key_name = KEY_RETURN_ETA
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def native_value(self) -> datetime | None:
        """Return the fused ETA whenever the person is away (unknown at home)."""
        result = self.result
        if result is None or result.is_home is not False or result.prediction is None:
            return None
        return result.prediction.eta

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the ETA components."""
        attrs = super().extra_state_attributes
        if (result := self.result) is not None and (pred := result.prediction) is not None:
            attrs["eta_habit"] = _iso(pred.eta_habit)
            attrs["eta_position"] = _iso(pred.eta_position)
            attrs["probability"] = round(pred.probability, 3)
        return attrs


class ModelMaturitySensor(HRPPersonSensor):
    """How well the model knows the person."""

    _key_name = KEY_MODEL_MATURITY
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0
    _attr_entity_category = EntityCategory.DIAGNOSTIC

    @property
    def native_value(self) -> float | None:
        """Return maturity in percent."""
        result = self.result
        return round(result.maturity.maturity * 100, 1) if result else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return maturity details."""
        attrs = super().extra_state_attributes
        if (result := self.result) is not None:
            report = result.maturity
            attrs.update(
                episodes=report.episodes,
                effective_episodes=round(report.effective_episodes, 2),
                resolved_predictions=report.resolved_predictions,
                brier_score=report.brier_score,
                baseline_brier_score=report.baseline_brier_score,
                brier_skill_score=report.brier_skill_score,
                fusion=getattr(report, "fusion", None),
                active_model=result.active_model,
                candidate_brier=dict(getattr(report, "candidate_brier", None) or {}),
                challenger_enabled=self.coordinator.challenger_enabled,
                numpy_available=self.coordinator.numpy_available,
                sklearn_available=self.coordinator.numpy_available,  # deprecated alias
                selected_mode=result.mode,
                effective_mode=result.effective_mode,
            )
        return attrs


class ReturnReasonSensor(HRPPersonSensor):
    """Human readable explanation of the prediction."""

    _key_name = KEY_RETURN_REASON

    @property
    def native_value(self) -> str | None:
        """Return the reason text."""
        result = self.result
        return result.reason if result else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return the raw reason parts."""
        attrs = super().extra_state_attributes
        if (result := self.result) is not None:
            attrs["parts"] = result.prediction.reason if result.prediction else {}
            attrs["active_model"] = result.active_model
        return attrs


class HouseReturnProbabilitySensor(HRPHouseEntity, SensorEntity):
    """Probability that someone is home within the horizon."""

    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_suggested_display_precision = 0

    def __init__(self, coordinator: HRPCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, KEY_HOUSE_RETURN_PROBABILITY, ENTITY_ID_FORMAT)

    @property
    def native_value(self) -> float | None:
        """Return the house probability in percent (100 when occupied)."""
        house = self.house
        if house is None or house.probability is None:
            return None
        return round(house.probability * 100, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return aggregate attributes."""
        return self.house_attributes()


class HouseNextReturnEtaSensor(HRPHouseEntity, SensorEntity):
    """Earliest ETA among away persons with p >= HOUSE_ETA_MIN_PROBABILITY."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: HRPCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, KEY_HOUSE_NEXT_RETURN_ETA, ENTITY_ID_FORMAT)

    @property
    def native_value(self) -> datetime | None:
        """Return the earliest ETA."""
        house = self.house
        return house.next_eta if house else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return aggregate attributes and whose ETA is shown."""
        attrs = self.house_attributes()
        if (house := self.house) is not None:
            attrs["eta_person"] = house.eta_person
        return attrs
