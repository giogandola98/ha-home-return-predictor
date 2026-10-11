"""Config and options flow tests."""

from __future__ import annotations

from homeassistant.config_entries import SOURCE_USER
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    CONF_HALF_LIFE_DAYS,
    CONF_HOME_ZONE,
    CONF_HORIZON_MINUTES,
    CONF_MATURITY_THRESHOLD,
    CONF_PERSONS,
    CONF_RETURN_THRESHOLD,
    CONF_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
)

from .conftest import ALICE, BOB


async def test_user_flow(hass: HomeAssistant, setup_home: None) -> None:
    """Happy path creates an entry."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_PERSONS: [], CONF_HOME_ZONE: "zone.home"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {CONF_PERSONS: "no_persons"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_PERSONS: [ALICE], CONF_HOME_ZONE: "zone.home"}
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_PERSONS: [ALICE], CONF_HOME_ZONE: "zone.home"}
    assert result["result"].unique_id == DOMAIN


async def test_already_configured(hass: HomeAssistant, config_entry: MockConfigEntry) -> None:
    """A second instance is refused."""
    result = await hass.config_entries.flow.async_init(DOMAIN, context={"source": SOURCE_USER})
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_flow(hass: HomeAssistant, loaded_entry: MockConfigEntry) -> None:
    """Options are saved and the entry reloads with the new person list."""
    result = await hass.config_entries.options.async_init(loaded_entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"

    user_input = {
        CONF_PERSONS: [BOB],
        CONF_RETURN_THRESHOLD: 0.5,
        CONF_MATURITY_THRESHOLD: 0.8,
        CONF_HORIZON_MINUTES: 60,
        CONF_HALF_LIFE_DAYS: 14,
        CONF_UPDATE_INTERVAL_MINUTES: 5,
    }
    result = await hass.config_entries.options.async_configure(result["flow_id"], user_input)
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert loaded_entry.options == user_input

    coordinator = loaded_entry.runtime_data
    assert coordinator.person_ids == [BOB]
    assert coordinator.return_threshold == 0.5
    assert coordinator.model_config.horizon_minutes == 60
    assert coordinator.update_interval.total_seconds() == 300
