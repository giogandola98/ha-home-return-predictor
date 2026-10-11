"""Diagnostics tests."""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.diagnostics import (
    async_get_config_entry_diagnostics,
)

from .conftest import ALICE, BOB


async def test_diagnostics(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """Diagnostics contain per-person data."""
    coordinator = loaded_entry.runtime_data
    coordinator.last_predictions[ALICE].reason["latitude"] = 45.0
    data = await async_get_config_entry_diagnostics(hass, loaded_entry)
    assert set(data["persons"]) == {ALICE, BOB}
    alice = data["persons"][ALICE]
    assert alice["mode"] == "auto"
    assert alice["episodes"] == 0
    assert alice["maturity"]["maturity"] >= 0
    assert alice["last_prediction"]["is_home"] is True
    assert isinstance(alice["prediction_log"], list)
    assert data["house"]["occupied"] is True
    assert alice["last_prediction"]["reason"]["latitude"] == "**REDACTED**"
    assert alice["active_model"] == "kernel_bayes"
    assert alice["maturity"]["active_model"] == "kernel_bayes"
    assert "candidate_brier" in alice["maturity"]
    assert data["challenger"] == {
        "requested": False,
        "enabled": False,
        "sklearn_available": data["challenger"]["sklearn_available"],
        "training": [],
    }
