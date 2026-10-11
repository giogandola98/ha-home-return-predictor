"""Reference implementation of the VTherm zone pre-heat blueprint decision.

The blueprint ``blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml``
implements exactly this logic in Jinja (its ``variables`` block references the
functions below step by step). This module exists so the decision table can be unit
tested; it is pure standard-library Python and never imports Home Assistant.

Priority, highest first:

1. ``protection_active`` (window open, VTherm safety/power/off/locked) -> don't touch.
2. Bed switch on while someone is home -> day zones get ``night_preset``,
   night/always zones get ``comfort``.
3. Someone home -> zone in use (occupancy sensor on, no schedule, or schedule active
   now) -> ``comfort``, otherwise ``unused``.
4. Predicted return (only when ``prediction_enabled``) -> once ``now`` is inside the
   pre-heat window before the house ETA (or the ETA is already past) and the zone
   schedule is active at the expected arrival: probability >= high -> ``comfort``,
   >= medium -> ``medium`` preset (default eco), else ``absent`` (with hysteresis on
   the current preset).
   With per-person ``return_candidates`` (the ``candidates`` attribute of the house
   ETA sensor) every candidate is judged on its *own* probability, ETA and
   schedule-at-ETA and the highest-ranked preset wins; probabilities of different
   persons are never combined with another person's ETA. Without candidates the
   single (house probability, house ETA) pair is used.
   With someone home the result is the higher-ranked of (3) and (4): another
   person's return never downgrades a zone. With nobody home it is (4) or ``absent``.

The blueprint never sets temperatures; it only switches VTherm presets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

ZONE_DAY = "day"
ZONE_NIGHT = "night"
ZONE_ALWAYS = "always"
ZONE_TYPES = (ZONE_DAY, ZONE_NIGHT, ZONE_ALWAYS)

# Built-in VTherm preset ranks. "away" is treated like frost, "boost" above comfort.
_BUILTIN_RANKS: dict[str, int] = {
    "frost": 0,
    "away": 0,
    "eco": 1,
    "comfort": 2,
    "boost": 3,
}


@dataclass(frozen=True)
class PresetConfig:
    """Preset names used by the blueprint: each one is a user input of the blueprint,
    chosen per zone (any VTherm preset name, custom names allowed).

    ``absent`` defaults to ``frost`` because VTherm hides the ``away`` preset (it is
    not selectable) when its presence feature is disabled.
    """

    comfort: str = "comfort"  # likely return / zone in use
    medium: str = "eco"  # medium return probability
    absent: str = "frost"
    unused: str = "eco"  # zone outside its schedule while people are home
    night: str = "eco"  # day zones while the bed switch is on


@dataclass(frozen=True)
class Thresholds:
    """Probability thresholds (0..1) for the predicted-return candidate."""

    high: float = 0.7
    medium: float = 0.4
    hysteresis: float = 0.05


@dataclass(frozen=True)
class ZoneInputs:
    """Everything the blueprint reads for one zone at one evaluation."""

    now: datetime
    zone_type: str = ZONE_DAY
    someone_home: bool = False
    house_probability: float = 0.0  # 0..1, best return probability of an away person
    house_eta: datetime | None = None
    # Per-person (probability 0..1, ETA) of away persons; when non-empty it replaces
    # the (house_probability, house_eta) pair for the predicted-return rule.
    return_candidates: Sequence[tuple[float, datetime | None]] | None = None
    preheat_minutes: float = 30.0
    # None = no schedule entity; else (state_on, next_event) of an HA ``schedule``.
    schedule: tuple[bool, datetime | None] | None = None
    bed_mode: bool | None = None
    occupancy: bool | None = None
    protection_active: bool = False
    # False when "act only in live mode" is set and the model is not live: the
    # predicted-return candidate is ignored, home/schedule/bed logic still applies.
    prediction_enabled: bool = True
    presets: PresetConfig = field(default_factory=PresetConfig)
    thresholds: Thresholds = field(default_factory=Thresholds)


@dataclass(frozen=True)
class Decision:
    """Target preset (``None`` = leave the thermostat alone) and a readable reason."""

    preset: str | None
    reason: str


def schedule_active_at(state_on: bool, next_event: datetime | None, at: datetime) -> bool:
    """Approximate whether an HA ``schedule`` helper is active at ``at``.

    An HA schedule only exposes its current state and ``next_event`` (the next toggle).
    If on, it stays on until ``next_event`` (``None`` = forever). If off, it turns on at
    ``next_event`` and the block is *assumed to last* past ``at`` (we cannot see when
    that next block ends). Good enough for horizons of a few hours.
    """
    if state_on:
        return next_event is None or at < next_event
    return next_event is not None and at >= next_event


def preset_rank(preset: str | None, presets: PresetConfig) -> int:
    """Rank presets: absent/frost/away < medium/eco < comfort < boost.

    Configured names are ranked by role first, so custom names work too.
    Unknown names rank like eco. ``None``/``"none"`` rank -1 (lowest).
    """
    if preset is None or preset == "none":
        return -1
    if preset == presets.comfort:
        return 2
    if preset == presets.medium:
        return 1
    if preset == presets.absent:
        return 0
    return _BUILTIN_RANKS.get(preset, 1)


def _return_candidate(inputs: ZoneInputs, current_preset: str | None) -> tuple[str | None, str]:
    """Preset suggested by the predicted return, or ``None`` if it does not apply.

    Each (probability, ETA) candidate is evaluated on its own; the highest-ranked
    preset wins (first one on ties). Without candidates: the house pair.
    """
    if not inputs.prediction_enabled:
        return None, "prediction ignored (not live)"
    candidates = list(inputs.return_candidates or ()) or [
        (inputs.house_probability, inputs.house_eta)
    ]
    best: tuple[str | None, str] | None = None
    for p, eta in candidates:
        preset, reason = _single_return(inputs, p, eta, current_preset)
        if best is None or (
            preset is not None
            and preset_rank(preset, inputs.presets) > preset_rank(best[0], inputs.presets)
        ):
            best = (preset, reason)
    assert best is not None
    return best


def _single_return(
    inputs: ZoneInputs, p: float, eta: datetime | None, current_preset: str | None
) -> tuple[str | None, str]:
    """Return rule for one (probability, ETA) pair."""
    if eta is None:
        return None, "no return predicted"
    start = eta - timedelta(minutes=max(inputs.preheat_minutes, 0.0))
    if inputs.now < start:
        return None, f"return ETA {eta:%H:%M} outside pre-heat window"
    # Late ETA (already past): the arrival is expected "now".
    arrival = max(eta, inputs.now)
    if inputs.schedule is not None and not schedule_active_at(
        inputs.schedule[0], inputs.schedule[1], arrival
    ):
        return None, f"zone not scheduled at ETA {eta:%H:%M}"

    presets, th = inputs.presets, inputs.thresholds
    rank = preset_rank(current_preset, presets)
    high = th.high - th.hysteresis if rank >= 2 else th.high
    medium = th.medium - th.hysteresis if rank >= 1 else th.medium
    if p >= high:
        return presets.comfort, f"return likely ({p:.0%}) ETA {eta:%H:%M}"
    if p >= medium:
        return presets.medium, f"return possible ({p:.0%}) ETA {eta:%H:%M}"
    return presets.absent, f"return unlikely ({p:.0%})"


def decide_preset(inputs: ZoneInputs, current_preset: str | None) -> Decision:
    """Decide the VTherm preset for one zone (see module docstring for priority)."""
    presets = inputs.presets
    if inputs.protection_active:
        return Decision(None, "protection active (window/safety/power/off): not touching")

    if inputs.bed_mode and inputs.someone_home:
        if inputs.zone_type == ZONE_DAY:
            return Decision(presets.night, "bed mode: day zone set to night preset")
        return Decision(presets.comfort, "bed mode: night zone kept comfortable")

    if inputs.someone_home:
        if inputs.occupancy:
            base, base_reason = presets.comfort, "someone home, zone occupied"
        elif inputs.schedule is None:
            base, base_reason = presets.comfort, "someone home, zone has no schedule"
        elif schedule_active_at(inputs.schedule[0], inputs.schedule[1], inputs.now):
            base, base_reason = presets.comfort, "someone home, zone schedule active"
        else:
            base, base_reason = presets.unused, "someone home, zone outside schedule"
    else:
        base, base_reason = presets.absent, "nobody home"

    candidate, cand_reason = _return_candidate(inputs, current_preset)

    if inputs.someone_home:
        # Never downgrade because of another person's (unlikely) return.
        if candidate is not None and preset_rank(candidate, presets) > preset_rank(base, presets):
            return Decision(candidate, f"{cand_reason} (upgrade over: {base_reason})")
        return Decision(base, base_reason)

    if candidate is not None:
        return Decision(candidate, cand_reason)
    return Decision(presets.absent, f"{base_reason}; {cand_reason}")
