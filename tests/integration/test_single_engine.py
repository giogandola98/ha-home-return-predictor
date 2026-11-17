"""Single kernel engine: the retired challenger option and v1 learned-engine state."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    CONF_ENABLE_CHALLENGER,
    CONF_PERSONS,
    DOMAIN,
)
from custom_components.home_return_predictor.hrp_core import KernelBayesReturnModel, ModelConfig
from custom_components.home_return_predictor.hrp_core.base import PresenceTransition
from custom_components.home_return_predictor.hrp_core.maturity import LogEntry

from .conftest import ALICE, BOB, HOME_LAT, HOME_LON, person_attrs

N_CHALLENGER_FEATURES = 12  # row width of the retired challenger in v1 logs


def _v1_state_with_gbm_champion() -> dict[str, Any]:
    """A state written by the previous release: gbm champion, stacker, challenger."""
    model = KernelBayesReturnModel(ModelConfig())
    now = dt_util.utcnow()
    transitions = []
    for i in range(1, 31):
        left = now - timedelta(days=i)
        transitions += [
            PresenceTransition(left, False),
            PresenceTransition(left + timedelta(hours=8), True),
        ]
    model.load_history(transitions)
    for i in range(50):
        ts = now - timedelta(hours=5, minutes=10 * i)
        model._log.resolved.append(
            LogEntry(
                ts,
                0.1,
                0.1,
                None,
                ts + timedelta(minutes=90),
                0,
                [0.0] * 7,
                0.1,
                [0.0] * N_CHALLENGER_FEATURES,
            )  # fmt: skip
        )
    data = model.to_dict()
    data.update(
        version=1,
        stacker={"weights": [0.1] * 8, "active": True, "trained_on": 400},
        selection={"champion": "gbm", "streak": {}, "brier": {"gbm": 0.01}, "evaluations": 3},
        challenger_enabled=True,
        challenger={"trained_on": 4, "constant": -4.6},
        last_training=now.isoformat(),
    )
    return data


async def test_v1_state_with_gbm_champion_loads_on_the_kernel_engine(
    hass: HomeAssistant, setup_home: None, hass_storage: dict[str, Any]
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], "home_zone": "zone.home"},
        options={CONF_PERSONS: [ALICE, BOB], CONF_ENABLE_CHALLENGER: True},
    )
    entry.add_to_hass(hass)
    stored = _v1_state_with_gbm_champion()
    key = f"{DOMAIN}.{entry.entry_id}"
    hass_storage[key] = {
        "version": 1,
        "minor_version": 1,
        "key": key,
        "data": {"schema": 1, "persons": {ALICE: {"model": stored, "is_home": True}}},
    }
    install = AsyncMock(return_value=None)
    with patch("homeassistant.requirements.async_process_requirements", install):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
    install.assert_not_awaited()  # numpy is never installed any more
    coordinator = entry.runtime_data
    assert coordinator.challenger_enabled is False
    model = coordinator.persons[ALICE].model
    assert len(model.episodes()) == len(stored["episodes"])
    assert len(model._log.resolved) == 50  # prediction log kept
    # leave: a defined prediction from the kernel engine
    lat = HOME_LAT + 5_000 / 111_195
    hass.states.async_set(ALICE, "not_home", person_attrs("Alice", lat, HOME_LON))
    await hass.async_block_till_done()
    await coordinator.async_refresh()
    p = float(hass.states.get("sensor.alice_return_probability").state)
    assert 0 < p < 100
    attrs = hass.states.get("sensor.alice_model_maturity").attributes
    assert attrs["active_model"] == "kernel_bayes" and attrs["challenger_enabled"] is False
    assert hass.states.get("binary_sensor.alice_returning").attributes["active_model"] == (
        "kernel_bayes"
    )
    saved = coordinator.persons[ALICE].model.to_dict()
    assert saved["version"] == 2
    assert not {"stacker", "selection", "challenger", "last_training"} & set(saved)


async def test_options_flow_hides_and_keeps_the_retired_option(
    hass: HomeAssistant, setup_home: None
) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], "home_zone": "zone.home"},
        options={CONF_PERSONS: [ALICE, BOB], CONF_ENABLE_CHALLENGER: True},
    )
    entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(entry.entry_id)
    await hass.async_block_till_done()
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.FORM
    assert CONF_ENABLE_CHALLENGER not in {str(k) for k in result["data_schema"].schema}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_PERSONS: [ALICE, BOB],
            "return_threshold": 0.6,
            "maturity_threshold": 0.7,
            "horizon_minutes": 90,
            "half_life_days": 28,
            "update_interval_minutes": 2,
        },
    )
    await hass.async_block_till_done()
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert entry.options[CONF_ENABLE_CHALLENGER] is True  # read and ignored, never deleted
    assert entry.runtime_data.challenger_enabled is False
