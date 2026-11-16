"""Sensor platform."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from homeassistant.components.sensor import (
    ENTITY_ID_FORMAT,
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.const import PERCENTAGE, EntityCategory, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import (
    KEY_HOUSE_EXPECTED_ABSENCE,
    KEY_HOUSE_NEXT_ARRIVAL,
    KEY_HOUSE_NEXT_RETURN_ETA,
    KEY_HOUSE_RETURN_PROBABILITY,
    KEY_MODEL_MATURITY,
    KEY_RETURN_ETA,
    KEY_RETURN_PROBABILITY,
    KEY_RETURN_REASON,
    LONG_ABSENCE_HOURS,
    SHORT_ABSENCE_HOURS,
)
from .coordinator import HouseResult, HRPConfigEntry, HRPCoordinator
from .entity import HRPHouseEntity, HRPPersonEntity
from .hrp_core.arrival import CDF_POINTS, QUANTILES

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
    entities += [
        HouseReturnProbabilitySensor(coordinator),
        HouseNextReturnEtaSensor(coordinator),
        HouseNextArrivalSensor(coordinator),
        HouseExpectedAbsenceSensor(coordinator),
    ]
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


def _arrival_at(house: HouseResult, q: float) -> datetime | None:
    """Time at which the first arrival has probability ``q`` (never before the next
    update, so the timestamp stays in the future while the house is empty)."""
    if house.arrival is None or house.arrival_now is None:
        return None
    minutes = house.arrival.quantile(q)
    if minutes is None:
        return None
    return _in_minutes(house.arrival_now, minutes)


def _in_minutes(now: datetime, minutes: float | None) -> datetime | None:
    if minutes is None:
        return None
    return (now + timedelta(minutes=max(minutes, MIN_ARRIVAL_MINUTES))).replace(microsecond=0)


MIN_ARRIVAL_MINUTES = 2.0


class HouseNextArrivalSensor(HRPHouseEntity, SensorEntity):
    """Quantile ``arrival_quantile`` of the first arrival home (house empty)."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(self, coordinator: HRPCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, KEY_HOUSE_NEXT_ARRIVAL, ENTITY_ID_FORMAT)

    @property
    def native_value(self) -> datetime | None:
        """Arrival time at the configured quantile; unknown with someone home or no data."""
        house = self.house
        return _arrival_at(house, house.arrival_quantile) if house else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Quantiles, CDF and per-person quantiles of the arrival distribution."""
        house = self.house
        if house is None:
            return {}
        attrs: dict[str, Any] = {
            "quantile": house.arrival_quantile,
            "mode": house.mode,
            "advisory": house.advisory,
        }
        dist, now = house.arrival, house.arrival_now
        if dist is None or now is None:
            return attrs
        attrs["quantiles"] = {f"q{round(q * 100)}": _iso(_arrival_at(house, q)) for q in QUANTILES}
        attrs["cdf"] = [[m, round(dist.cdf(m), 3)] for m in CDF_POINTS]
        attrs["source"] = dist.source
        attrs["persons"] = {
            pid: {
                f"q{round(q * 100)}": _iso(_in_minutes(now, pdist.quantile(q))) for q in (0.25, 0.5)
            }
            for pid, pdist in house.arrival_persons.items()
        }
        return attrs


def absence_class(q25_hours: float | None) -> str | None:
    """short (< 4 h) / day (4-14 h) / long (> 14 h), judged on the q25."""
    if q25_hours is None:
        return "long"
    if q25_hours < SHORT_ABSENCE_HOURS:
        return "short"
    return "day" if q25_hours <= LONG_ABSENCE_HOURS else "long"


class HouseExpectedAbsenceSensor(HRPHouseEntity, SensorEntity):
    """Median remaining time with the house empty (hours)."""

    _attr_device_class = SensorDeviceClass.DURATION
    _attr_native_unit_of_measurement = UnitOfTime.HOURS
    _attr_suggested_display_precision = 1

    def __init__(self, coordinator: HRPCoordinator) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator, KEY_HOUSE_EXPECTED_ABSENCE, ENTITY_ID_FORMAT)

    def _hours(self, q: float) -> float | None:
        house = self.house
        if house is None or house.arrival is None:
            return None
        minutes = house.arrival.quantile(q)
        return round(minutes / 60.0, 2) if minutes is not None else None

    @property
    def native_value(self) -> float | None:
        """Median of the remaining empty time; unknown with someone home or no data."""
        return self._hours(0.5)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """q25 / q75 (hours), class and start of the empty period."""
        house = self.house
        if house is None or house.arrival is None:
            return {}
        q25 = self._hours(0.25)
        return {
            "q25": q25,
            "q75": self._hours(0.75),
            "absence_class": absence_class(q25),
            "empty_since": _iso(house.empty_since),
        }
