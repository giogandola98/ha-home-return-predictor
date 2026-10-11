"""Optional scikit-learn challenger: option, runtime install, background training."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Generator
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.requirements import RequirementsNotFound
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    CONF_ENABLE_SKLEARN,
    CONF_HALF_LIFE_DAYS,
    CONF_HORIZON_MINUTES,
    CONF_MATURITY_THRESHOLD,
    CONF_PERSONS,
    CONF_RETURN_THRESHOLD,
    CONF_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    SKLEARN_REQUIREMENT,
)
from custom_components.home_return_predictor.hrp_core import KernelBayesReturnModel

from .conftest import ALICE, BOB

COORD = "custom_components.home_return_predictor.coordinator"


@pytest.fixture
def enable_calls() -> Generator[list[tuple[int, bool]]]:
    """Record enable_challenger(model, enabled) calls."""
    calls: list[tuple[int, bool]] = []

    def _enable(self: KernelBayesReturnModel, enabled: bool) -> None:
        calls.append((id(self), enabled))

    with patch.object(KernelBayesReturnModel, "enable_challenger", _enable):
        yield calls


def _enabled_entry(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], "home_zone": "zone.home"},
        options={CONF_PERSONS: [ALICE, BOB], CONF_ENABLE_SKLEARN: True},
    )
    entry.add_to_hass(hass)
    return entry


async def _setup(hass: HomeAssistant, entry: MockConfigEntry) -> None:
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done(wait_background_tasks=True)


async def test_options_flow_enables_challenger(
    hass: HomeAssistant, loaded_entry: MockConfigEntry, enable_calls: list[tuple[int, bool]]
) -> None:
    """The boolean option is saved; the reload installs scikit-learn and enables it."""
    assert loaded_entry.runtime_data.challenger_enabled is False
    result = await hass.config_entries.options.async_init(loaded_entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    user_input = {
        CONF_PERSONS: [ALICE, BOB],
        CONF_RETURN_THRESHOLD: 0.6,
        CONF_MATURITY_THRESHOLD: 0.7,
        CONF_HORIZON_MINUTES: 90,
        CONF_HALF_LIFE_DAYS: 28,
        CONF_UPDATE_INTERVAL_MINUTES: 2,
        CONF_ENABLE_SKLEARN: True,
    }
    install = AsyncMock(return_value=None)
    with (
        patch(f"{COORD}.async_process_requirements", install),
        patch(f"{COORD}.sklearn_available", return_value=True),
    ):
        result = await hass.config_entries.options.async_configure(result["flow_id"], user_input)
        await hass.async_block_till_done(wait_background_tasks=True)
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert loaded_entry.options[CONF_ENABLE_SKLEARN] is True
    install.assert_awaited_once_with(hass, DOMAIN, [SKLEARN_REQUIREMENT])
    coordinator = loaded_entry.runtime_data
    assert coordinator.challenger_requested is True
    assert coordinator.challenger_enabled is True
    models = {id(rt.model) for rt in coordinator.persons.values()}
    assert {mid for mid, enabled in enable_calls if enabled} == models


async def test_disabled_by_default_never_installs(
    hass: HomeAssistant,
    setup_home: None,
    config_entry: MockConfigEntry,
    enable_calls: list[tuple[int, bool]],
) -> None:
    """Default off: no install attempt, challenger explicitly disabled on every model."""
    install = AsyncMock()
    with patch(f"{COORD}.async_process_requirements", install):
        await _setup(hass, config_entry)
    install.assert_not_awaited()
    coordinator = config_entry.runtime_data
    assert coordinator.challenger_enabled is False
    assert enable_calls and all(not enabled for _, enabled in enable_calls)
    attrs = hass.states.get("sensor.alice_model_maturity").attributes
    assert attrs["challenger_enabled"] is False
    assert attrs["active_model"] == "kernel_bayes"
    assert "candidate_brier" in attrs and "sklearn_available" in attrs
    assert hass.states.get("binary_sensor.alice_returning").attributes["active_model"] == (
        "kernel_bayes"
    )
    assert hass.states.get("sensor.alice_return_reason").attributes["active_model"] == (
        "kernel_bayes"
    )


async def test_enable_install_success(
    hass: HomeAssistant, setup_home: None, enable_calls: list[tuple[int, bool]]
) -> None:
    """Successful install enables the challenger on every person's model."""
    entry = _enabled_entry(hass)
    with (
        patch(f"{COORD}.async_process_requirements", AsyncMock(return_value=None)) as install,
        patch(f"{COORD}.sklearn_available", return_value=True),
        patch.object(KernelBayesReturnModel, "challenger_due", return_value=False),
    ):
        await _setup(hass, entry)
    install.assert_awaited_once_with(hass, DOMAIN, [SKLEARN_REQUIREMENT])
    coordinator = entry.runtime_data
    assert coordinator.challenger_enabled is True
    assert coordinator.sklearn_available is True
    models = {id(rt.model) for rt in coordinator.persons.values()}
    assert {mid for mid, enabled in enable_calls if enabled} == models
    attrs = hass.states.get("sensor.alice_model_maturity").attributes
    assert attrs["challenger_enabled"] is True
    assert attrs["sklearn_available"] is True


async def test_enable_install_failure(
    hass: HomeAssistant,
    setup_home: None,
    enable_calls: list[tuple[int, bool]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failed install only logs a warning; the champion keeps predicting."""
    entry = _enabled_entry(hass)
    install = AsyncMock(side_effect=RequirementsNotFound(DOMAIN, [SKLEARN_REQUIREMENT]))
    caplog.set_level(logging.WARNING)
    with (
        patch(f"{COORD}.async_process_requirements", install),
        patch(f"{COORD}.sklearn_available", return_value=False),
    ):
        await _setup(hass, entry)
    coordinator = entry.runtime_data
    assert coordinator.challenger_enabled is False
    assert all(not enabled for _, enabled in enable_calls)
    warnings = [r for r in caplog.records if "challenger stays disabled" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert hass.states.get("sensor.alice_return_probability").state not in (
        "unavailable",
        "unknown",
    )


async def test_training_scheduled_once_at_a_time(
    hass: HomeAssistant, setup_home: None, enable_calls: list[tuple[int, bool]]
) -> None:
    """Due challengers train in the background, one fit at a time, without the person lock.

    snapshot (locked) -> fit (unlocked, heavy) -> apply (locked); updates proceed
    while a fit is in flight.
    """
    entry = _enabled_entry(hass)
    release = threading.Event()
    guard = threading.Lock()
    trained: list[int] = []
    applied: list[int] = []
    running = 0
    max_running = 0

    def _due(self: KernelBayesReturnModel, now: datetime) -> bool:
        with guard:
            return id(self) not in trained

    def _snapshot(self: KernelBayesReturnModel) -> dict[str, Any]:
        return {"model": id(self)}

    def _fit(snapshot: dict[str, Any], now: datetime) -> dict[str, Any]:
        nonlocal running, max_running
        with guard:
            running += 1
            max_running = max(max_running, running)
        release.wait(5)
        with guard:
            running -= 1
            trained.append(snapshot["model"])
        return {"model": snapshot["model"]}

    def _apply(self: KernelBayesReturnModel, result: Any, now: datetime) -> dict[str, Any]:
        assert result["model"] == id(self)
        applied.append(id(self))
        return {"trained": True, "reason": "ok", "rows": 0, "seconds": 0.0, "brier": {}}

    with (
        patch(f"{COORD}.async_process_requirements", AsyncMock(return_value=None)),
        patch(f"{COORD}.sklearn_available", return_value=True),
        patch.object(KernelBayesReturnModel, "challenger_due", _due),
        patch.object(KernelBayesReturnModel, "challenger_snapshot", _snapshot),
        patch.object(KernelBayesReturnModel, "fit_challenger", staticmethod(_fit)),
        patch.object(KernelBayesReturnModel, "apply_challenger", _apply),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done()
        coordinator = entry.runtime_data
        for _ in range(200):  # wait for the challenger setup + refresh + first training
            if running:
                break
            await asyncio.sleep(0.01)
        assert running == 1
        assert coordinator.training_persons == {ALICE, BOB}
        # The fit runs without the person locks: an update completes meanwhile and
        # schedules nothing new.
        assert not any(rt.lock.locked() for rt in coordinator.persons.values())
        await asyncio.wait_for(coordinator.async_refresh(), 5)
        assert coordinator.last_update_success
        assert running == 1 and not trained
        release.set()
        await hass.async_block_till_done(wait_background_tasks=True)
    models = {id(rt.model) for rt in coordinator.persons.values()}
    assert sorted(trained) == sorted(models)  # each person exactly once
    assert sorted(applied) == sorted(models)
    assert max_running == 1
    assert coordinator.training_persons == set()
