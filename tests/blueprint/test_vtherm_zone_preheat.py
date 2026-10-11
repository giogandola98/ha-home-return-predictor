"""Run the VTherm blueprint inside a real Home Assistant and compare with preset_logic.

The blueprint YAML is loaded, its ``!input`` tags are substituted with test values and
the resulting automation is set up in a test ``hass``. Each scenario sets fake
states, triggers the automation and checks the preset requested from the thermostat.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from homeassistant.util import dt as dt_util
from pytest_homeassistant_custom_component.common import async_mock_service

BLUEPRINT = (
    Path(__file__).resolve().parents[2]
    / "blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml"
)
CLIMATE = "climate.bedroom"


class _Input:
    def __init__(self, name: str) -> None:
        self.name = name


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_constructor("!input", lambda loader, node: _Input(loader.construct_scalar(node)))


def _load() -> dict[str, Any]:
    with BLUEPRINT.open(encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)  # noqa: S506 - restricted SafeLoader subclass


def _defaults(bp: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, spec in bp["blueprint"]["input"].items():
        if "input" in spec:  # section
            for sub_key, sub_spec in spec["input"].items():
                out[sub_key] = sub_spec.get("default")
        else:
            out[key] = spec.get("default")
    return out


def _substitute(node: Any, values: dict[str, Any]) -> Any:
    if isinstance(node, _Input):
        return values[node.name]
    if isinstance(node, dict):
        return {k: _substitute(v, values) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, values) for v in node]
    return node


def test_blueprint_yaml_structure() -> None:
    bp = _load()
    defaults = _defaults(bp)
    assert defaults["absent_preset"] == "frost"
    assert defaults["act_only_live"] is True
    assert defaults["house_probability_sensor"] == "sensor.house_return_probability"
    assert defaults["house_eta_sensor"] == "sensor.house_next_return_eta"
    # every !input used is declared
    used: set[str] = set()

    def walk(n: Any) -> None:
        if isinstance(n, _Input):
            used.add(n.name)
        elif isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk({k: v for k, v in bp.items() if k != "blueprint"})
    assert used <= set(defaults), used - set(defaults)
    # never sets temperatures
    assert "set_temperature" not in BLUEPRINT.read_text(encoding="utf-8")


async def _setup(hass: HomeAssistant, **inputs: Any) -> list:
    bp = _load()
    values = _defaults(bp)
    values.update({"climates": [CLIMATE], **inputs})
    config = _substitute({k: v for k, v in bp.items() if k != "blueprint"}, values)
    config["id"] = "zone"
    config["alias"] = "zone"
    calls = async_mock_service(hass, "climate", "set_preset_mode")
    assert await async_setup_component(hass, "automation", {"automation": [config]})
    await hass.async_block_till_done()
    calls.clear()  # ignore runs fired by the setup-time state triggers
    return calls


def _climate(hass: HomeAssistant, preset: str = "eco", **attrs: Any) -> None:
    hass.states.async_set(
        CLIMATE,
        attrs.pop("state", "heat"),
        {
            "preset_mode": preset,
            "preset_modes": ["none", "frost", "eco", "comfort", "boost"],
            **attrs,
        },
    )


def _house(
    hass: HomeAssistant,
    *,
    home: int = 0,
    prob: float = 0,
    eta_min: float | None = None,
    live: bool = True,
) -> None:
    eta = None if eta_min is None else dt_util.now() + timedelta(minutes=eta_min)
    hass.states.async_set("zone.home", str(home))
    hass.states.async_set(
        "sensor.house_return_probability",
        "100" if home else str(prob),
        {"occupied": bool(home)},
    )
    hass.states.async_set("sensor.house_next_return_eta", eta.isoformat() if eta else "unknown")
    hass.states.async_set(
        "binary_sensor.house_someone_returning",
        "on" if eta else "off",
        {"mode": "live" if live else "learning", "advisory": not live},
    )


async def _run(hass: HomeAssistant, calls: list) -> str | None:
    calls.clear()
    await hass.services.async_call(
        "automation", "trigger", {"entity_id": "automation.zone"}, blocking=True
    )
    await hass.async_block_till_done()
    return calls[-1].data["preset_mode"] if calls else None


def _schedule(hass: HomeAssistant, on: bool, next_in_min: float | None) -> None:
    nxt = None if next_in_min is None else dt_util.now() + timedelta(minutes=next_in_min)
    hass.states.async_set("schedule.bedroom", "on" if on else "off", {"next_event": nxt})


@pytest.mark.parametrize(
    ("house", "current", "expected"),
    [
        ({"prob": 0, "eta_min": None}, "eco", "frost"),
        ({"prob": 90, "eta_min": 20}, "frost", "comfort"),
        ({"prob": 50, "eta_min": 20}, "frost", "eco"),
        ({"prob": 90, "eta_min": 45}, "frost", None),  # outside 30-min window, already frost
        ({"prob": 90, "eta_min": 45}, "eco", "frost"),
        ({"prob": 90, "eta_min": -10}, "frost", "comfort"),  # late ETA
        ({"prob": 67, "eta_min": 10}, "comfort", None),  # hysteresis keeps comfort
        ({"prob": 64, "eta_min": 10}, "comfort", "eco"),
        ({"prob": 90, "eta_min": 10, "live": False}, "frost", None),  # act only live
        ({"home": 1}, "frost", "comfort"),
        ({"home": 1}, "comfort", None),
    ],
)
async def test_basic_table(hass: HomeAssistant, house, current, expected) -> None:
    calls = await _setup(hass)
    _house(hass, **house)
    _climate(hass, current)
    assert await _run(hass, calls) == expected


async def test_protection_window_open(hass: HomeAssistant) -> None:
    calls = await _setup(hass)
    _house(hass, home=1)
    _climate(hass, "frost", window_manager={"window_state": "on", "is_window_bypass": False})
    assert await _run(hass, calls) is None
    _climate(hass, "frost", window_manager={"window_state": "on", "is_window_bypass": True})
    assert await _run(hass, calls) == "comfort"
    _climate(hass, "frost", safety_manager={"safety_state": "on"})
    assert await _run(hass, calls) is None
    _climate(hass, "frost", power_manager={"overpowering_state": "on"})
    assert await _run(hass, calls) is None
    _climate(hass, "frost", state="off")
    assert await _run(hass, calls) is None


async def test_schedule_and_bed(hass: HomeAssistant) -> None:
    hass.states.async_set("input_boolean.bed", "off")
    calls = await _setup(
        hass, schedule_entity="schedule.bedroom", bed_switch="input_boolean.bed", zone_type="day"
    )
    # someone home, outside schedule -> unused (eco)
    _house(hass, home=1)
    _schedule(hass, False, 120)
    _climate(hass, "comfort")
    assert await _run(hass, calls) == "eco"
    # schedule active -> comfort
    _schedule(hass, True, 60)
    _climate(hass, "eco")
    assert await _run(hass, calls) == "comfort"
    # bed on -> day zone night preset (eco) even inside schedule
    hass.states.async_set("input_boolean.bed", "on")
    _climate(hass, "comfort")
    assert await _run(hass, calls) == "eco"


async def test_return_at_eta_checks_schedule(hass: HomeAssistant) -> None:
    calls = await _setup(hass, schedule_entity="schedule.bedroom", zone_type="night")
    _climate(hass, "frost")
    # nobody home, ETA in 25 min, schedule on now but ends in 10 min -> no preheat
    _house(hass, prob=90, eta_min=25)
    _schedule(hass, True, 10)
    assert await _run(hass, calls) is None
    # schedule turns on in 10 min (before the ETA) -> preheat
    _schedule(hass, False, 10)
    assert await _run(hass, calls) == "comfort"


async def test_preheat_sensor_wins(hass: HomeAssistant) -> None:
    hass.states.async_set("sensor.zone_predicted_duration", "60")
    calls = await _setup(hass, preheat_sensor="sensor.zone_predicted_duration")
    _house(hass, prob=90, eta_min=50)
    _climate(hass, "frost")
    assert await _run(hass, calls) == "comfort"
    hass.states.async_set("sensor.zone_predicted_duration", "unavailable")
    assert await _run(hass, calls) is None  # falls back to 30 min, already frost


async def test_home_upgrade_and_occupancy(hass: HomeAssistant) -> None:
    hass.states.async_set("binary_sensor.bedroom_occupancy", "off")
    calls = await _setup(
        hass, schedule_entity="schedule.bedroom", occupancy_sensor="binary_sensor.bedroom_occupancy"
    )
    _schedule(hass, False, 120)
    _climate(hass, "eco")
    # someone home, zone unused; another person returning soon but schedule off at ETA
    hass.states.async_set("zone.home", "1")
    eta = (dt_util.now() + timedelta(minutes=10)).isoformat()
    hass.states.async_set("sensor.house_return_probability", "100", {"occupied": True})
    hass.states.async_set("sensor.house_next_return_eta", eta)
    hass.states.async_set("binary_sensor.house_someone_returning", "on", {"advisory": False})
    assert await _run(hass, calls) is None  # stays eco (unused), no downgrade
    # occupancy forces comfort
    hass.states.async_set("binary_sensor.bedroom_occupancy", "on")
    assert await _run(hass, calls) == "comfort"


async def test_act_only_live_disabled(hass: HomeAssistant) -> None:
    calls = await _setup(hass, act_only_live=False)
    _house(hass, prob=90, eta_min=10, live=False)
    _climate(hass, "frost")
    assert await _run(hass, calls) == "comfort"


async def test_presence_bridge(hass: HomeAssistant) -> None:
    path = BLUEPRINT.with_name("presence_bridge.yaml")
    with path.open(encoding="utf-8") as fh:
        bp = yaml.load(fh, Loader=_Loader)  # noqa: S506
    values = {k: v.get("default") for k, v in bp["blueprint"]["input"].items()}
    values["target"] = "input_boolean.hint"
    config = _substitute({k: v for k, v in bp.items() if k != "blueprint"}, values)
    config.update(id="bridge", alias="bridge")
    hass.states.async_set("input_boolean.hint", "off")
    hass.states.async_set("zone.home", "0")
    hass.states.async_set("binary_sensor.house_someone_returning", "on", {"advisory": True})
    on_calls = async_mock_service(hass, "input_boolean", "turn_on")
    assert await async_setup_component(hass, "automation", {"automation": [config]})
    await hass.async_block_till_done()
    trigger = {"entity_id": "automation.bridge", "skip_condition": False}
    await hass.services.async_call("automation", "trigger", trigger, blocking=True)
    await hass.async_block_till_done()
    assert not on_calls  # learning -> ignored
    hass.states.async_set("binary_sensor.house_someone_returning", "on", {"advisory": False})
    await hass.async_block_till_done()
    assert on_calls and on_calls[-1].data["entity_id"] == ["input_boolean.hint"]


async def test_custom_presets(hass: HomeAssistant) -> None:
    calls = await _setup(hass, comfort_preset="boost", medium_preset="comfort", absent_preset="eco")
    _climate(hass, "frost")
    _house(hass, prob=90, eta_min=10)
    assert await _run(hass, calls) == "boost"
    _house(hass, prob=50, eta_min=10)
    assert await _run(hass, calls) == "comfort"
    _house(hass, prob=10, eta_min=10)
    assert await _run(hass, calls) == "eco"
