"""Config and options flow."""

from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlowWithReload
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
)

from .const import (
    CONF_ENABLE_CHALLENGER,
    CONF_HALF_LIFE_DAYS,
    CONF_HOLIDAY_CALENDAR,
    CONF_HOME_ZONE,
    CONF_HORIZON_MINUTES,
    CONF_MATURITY_THRESHOLD,
    CONF_PERSONS,
    CONF_RETURN_THRESHOLD,
    CONF_UPDATE_INTERVAL_MINUTES,
    DEFAULT_ENABLE_CHALLENGER,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HOME_ZONE,
    DEFAULT_HORIZON_MINUTES,
    DEFAULT_MATURITY_THRESHOLD,
    DEFAULT_RETURN_THRESHOLD,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    NAME,
)

PERSONS_SELECTOR = EntitySelector(EntitySelectorConfig(domain="person", multiple=True))


def _number(minimum: float, maximum: float, step: float, unit: str | None = None) -> NumberSelector:
    config = NumberSelectorConfig(min=minimum, max=maximum, step=step, mode=NumberSelectorMode.BOX)
    if unit:
        config["unit_of_measurement"] = unit
    return NumberSelector(config)


class HomeReturnPredictorConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle the initial setup (single instance)."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Pick persons and the home zone."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()
        errors: dict[str, str] = {}
        if user_input is not None:
            if not user_input.get(CONF_PERSONS):
                errors[CONF_PERSONS] = "no_persons"
            else:
                return self.async_create_entry(title=NAME, data=user_input)
        schema = vol.Schema(
            {
                vol.Required(CONF_PERSONS): PERSONS_SELECTOR,
                vol.Required(CONF_HOME_ZONE, default=DEFAULT_HOME_ZONE): EntitySelector(
                    EntitySelectorConfig(domain="zone")
                ),
            }
        )
        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(schema, user_input),
            errors=errors,
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: Any) -> HomeReturnPredictorOptionsFlow:
        """Return the options flow."""
        return HomeReturnPredictorOptionsFlow()


class HomeReturnPredictorOptionsFlow(OptionsFlowWithReload):
    """Tune thresholds and edit the person list (entry reloads on change)."""

    async def async_step_init(self, user_input: dict[str, Any] | None = None) -> ConfigFlowResult:
        """Manage options."""
        errors: dict[str, str] = {}
        if user_input is not None:
            if not user_input.get(CONF_PERSONS):
                errors[CONF_PERSONS] = "no_persons"
            else:
                return self.async_create_entry(data=user_input)
        entry = self.config_entry
        current = {
            CONF_PERSONS: entry.options.get(CONF_PERSONS) or entry.data.get(CONF_PERSONS, []),
            CONF_RETURN_THRESHOLD: DEFAULT_RETURN_THRESHOLD,
            CONF_MATURITY_THRESHOLD: DEFAULT_MATURITY_THRESHOLD,
            CONF_HORIZON_MINUTES: DEFAULT_HORIZON_MINUTES,
            CONF_HALF_LIFE_DAYS: DEFAULT_HALF_LIFE_DAYS,
            CONF_UPDATE_INTERVAL_MINUTES: DEFAULT_UPDATE_INTERVAL_MINUTES,
            CONF_ENABLE_CHALLENGER: DEFAULT_ENABLE_CHALLENGER,
        }
        current.update({k: v for k, v in entry.options.items() if k != CONF_PERSONS})
        if user_input is not None:
            current.update(user_input)
        schema = vol.Schema(
            {
                vol.Required(CONF_PERSONS): PERSONS_SELECTOR,
                vol.Required(CONF_RETURN_THRESHOLD): _number(0.05, 0.95, 0.05),
                vol.Required(CONF_MATURITY_THRESHOLD): _number(0.1, 0.95, 0.05),
                vol.Required(CONF_HORIZON_MINUTES): _number(15, 360, 5, "min"),
                vol.Required(CONF_HALF_LIFE_DAYS): _number(3, 180, 1, "d"),
                vol.Required(CONF_UPDATE_INTERVAL_MINUTES): _number(1, 30, 1, "min"),
                vol.Optional(CONF_HOLIDAY_CALENDAR): EntitySelector(
                    EntitySelectorConfig(domain="calendar")
                ),
                vol.Optional(CONF_ENABLE_CHALLENGER): BooleanSelector(),
            }
        )
        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(schema, current),
            errors=errors,
        )
