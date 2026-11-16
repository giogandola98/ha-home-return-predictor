"""A stored gbm champion is demoted when the first training finds no informative holdout."""

from __future__ import annotations

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.home_return_predictor.const import (
    CONF_ENABLE_CHALLENGER,
    CONF_PERSONS,
    DOMAIN,
)
from custom_components.home_return_predictor.hrp_core import KernelBayesReturnModel, ModelConfig
from custom_components.home_return_predictor.hrp_core.challenger import (
    N_CHALLENGER_FEATURES,
    GBMChallenger,
)
from custom_components.home_return_predictor.hrp_core.maturity import LogEntry
from custom_components.home_return_predictor.hrp_core.selection import GBM, KERNEL_BAYES
from custom_components.home_return_predictor.hrp_core.stacker import N_FEATURES

from .conftest import ALICE, BOB

COORD = "custom_components.home_return_predictor.coordinator"


def _stored_gbm_champion() -> dict[str, Any]:
    """Model state: gbm champion, 600 resolved entries, the newest 400 without returns."""
    model = KernelBayesReturnModel(ModelConfig())
    model.enable_challenger(True)
    serving = GBMChallenger()
    serving.fit([[0.0] * N_CHALLENGER_FEATURES] * 4, [0, 0, 0, 0])  # constant model
    model._challenger = serving
    model.selector.champion = GBM
    model.selector.brier = {KERNEL_BAYES: 0.05, "learned_fusion": 0.05, GBM: 0.03}
    end = dt_util.utcnow() - timedelta(hours=1)
    outcomes = [1, 0, 0] * 66 + [1, 0] + [0] * 400  # a long trip at the end
    n = len(outcomes)
    for i, outcome in enumerate(outcomes):
        ts = end - timedelta(minutes=10 * (n - 1 - i))
        model._log.resolved.append(
            LogEntry(
                ts=ts,
                p=0.1,
                prior=0.1,
                approach_class=None,
                horizon_end=ts + timedelta(minutes=90),
                outcome=outcome,
                features=[-2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0][:N_FEATURES],
                base=0.1,
                x=[0.0] * N_CHALLENGER_FEATURES,
            )
        )
    data = model.to_dict()
    restored = KernelBayesReturnModel.from_dict(data, ModelConfig())
    assert restored._serving_model() == GBM  # precondition: gbm serves before training
    return data


async def test_first_training_on_a_trip_serves_kernel_bayes(
    hass: HomeAssistant, setup_home: None, hass_storage: dict[str, Any]
) -> None:
    """Stored gbm champion + holdout without returns -> kernel_bayes after the training."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Home Return Predictor",
        unique_id=DOMAIN,
        data={CONF_PERSONS: [ALICE, BOB], "home_zone": "zone.home"},
        options={CONF_PERSONS: [ALICE, BOB], CONF_ENABLE_CHALLENGER: True},
    )
    entry.add_to_hass(hass)
    hass_storage[f"{DOMAIN}.{entry.entry_id}"] = {
        "version": 1,
        "minor_version": 1,
        "key": f"{DOMAIN}.{entry.entry_id}",
        "data": {
            "schema": 1,
            "persons": {
                ALICE: {"model": _stored_gbm_champion(), "is_home": True, "bootstrapped": True}
            },
        },
    }
    summaries: list[dict[str, Any]] = []
    real_apply = KernelBayesReturnModel.apply_challenger

    def _apply(self: KernelBayesReturnModel, result: Any, now: Any) -> dict[str, Any]:
        summary = real_apply(self, result, now)
        summaries.append(summary)
        return summary

    with (
        patch(f"{COORD}.async_process_requirements", AsyncMock(return_value=None)),
        patch(f"{COORD}.numpy_available", return_value=True),
        patch.object(KernelBayesReturnModel, "apply_challenger", _apply),
    ):
        assert await hass.config_entries.async_setup(entry.entry_id)
        await hass.async_block_till_done(wait_background_tasks=True)
        coordinator = entry.runtime_data
        await coordinator.async_refresh()
        await hass.async_block_till_done(wait_background_tasks=True)

    assert summaries and summaries[0]["reason"] == "uninformative_holdout"
    model = coordinator.persons[ALICE].model
    assert model.selector.champion == KERNEL_BAYES
    assert model._serving_model() == KERNEL_BAYES
    attrs = hass.states.get("sensor.alice_model_maturity").attributes
    assert attrs["active_model"] == KERNEL_BAYES
    assert hass.states.get("binary_sensor.alice_returning").attributes["active_model"] == (
        KERNEL_BAYES
    )
