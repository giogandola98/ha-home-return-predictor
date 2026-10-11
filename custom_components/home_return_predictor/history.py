"""Home zone geometry, state conversion and recorder bootstrap.

Home detection rule (shared by live tracking and history bootstrap):

* home zone ``zone.home``: the person is home when its state is ``home``;
* any other zone: the person is home when its state equals the zone friendly
  name (that is how ``person`` reports non-home zones) or when its own
  coordinates are inside the zone radius; a fix less accurate than
  ``MAX_FIX_ACCURACY_M`` gives no information.

``unknown`` / ``unavailable`` states give ``None`` (no information).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any

from homeassistant.const import (
    ATTR_FRIENDLY_NAME,
    ATTR_GPS_ACCURACY,
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    STATE_HOME,
    STATE_UNAVAILABLE,
    STATE_UNKNOWN,
)
from homeassistant.core import HomeAssistant, State
from homeassistant.exceptions import HomeAssistantError
from homeassistant.util import dt as dt_util

from .const import (
    DEFAULT_HOME_ZONE,
    HISTORY_CHUNK_DAYS,
    HISTORY_FIX_MIN_INTERVAL_S,
    MAX_FIX_ACCURACY_M,
)
from .hrp_core import haversine_m
from .hrp_core.base import PositionFix, PresenceTransition

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HomeZone:
    """Snapshot of the configured home zone."""

    entity_id: str
    name: str
    latitude: float
    longitude: float
    radius: float

    @classmethod
    def from_state(cls, state: State | None) -> HomeZone | None:
        """Build from a zone state (None when missing or without coordinates)."""
        if state is None:
            return None
        try:
            return cls(
                entity_id=state.entity_id,
                name=str(state.attributes.get(ATTR_FRIENDLY_NAME, state.object_id)),
                latitude=float(state.attributes[ATTR_LATITUDE]),
                longitude=float(state.attributes[ATTR_LONGITUDE]),
                radius=float(state.attributes.get("radius", 0) or 0),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def distance_m(self, latitude: float, longitude: float) -> float:
        """Distance from the zone border (0 inside the zone)."""
        return max(
            0.0, haversine_m(latitude, longitude, self.latitude, self.longitude) - self.radius
        )


def _coordinates(state: State) -> tuple[float, float, float | None] | None:
    """Return (lat, lon, accuracy) from state attributes, if any."""
    attrs = state.attributes
    try:
        lat = float(attrs[ATTR_LATITUDE])
        lon = float(attrs[ATTR_LONGITUDE])
    except (KeyError, TypeError, ValueError):
        return None
    acc_raw = attrs.get(ATTR_GPS_ACCURACY)
    try:
        acc = float(acc_raw) if acc_raw is not None else None
    except (TypeError, ValueError):
        acc = None
    return lat, lon, acc


def state_is_home(state: State, zone_id: str, zone: HomeZone | None) -> bool | None:
    """Apply the home detection rule to a person state."""
    if state.state in (STATE_UNKNOWN, STATE_UNAVAILABLE):
        return None
    if zone_id == DEFAULT_HOME_ZONE:
        return state.state == STATE_HOME
    if zone is None:
        return False
    if state.state == zone.name:
        return True
    coords = _coordinates(state)
    if coords is None:
        return False
    lat, lon, acc = coords
    if acc is not None and acc > MAX_FIX_ACCURACY_M:
        return None  # inaccurate fix: no information either way
    return zone.distance_m(lat, lon) <= 0.0


def state_to_fix(state: State, zone: HomeZone | None) -> PositionFix | None:
    """Convert a state with coordinates into a PositionFix (None if unusable)."""
    if zone is None:
        return None
    coords = _coordinates(state)
    if coords is None:
        return None
    lat, lon, acc = coords
    if acc is not None and acc > MAX_FIX_ACCURACY_M:
        return None
    return PositionFix(ts=state.last_updated, distance_m=zone.distance_m(lat, lon), accuracy_m=acc)


class _HistoryReducer:
    """Incrementally reduce chronological person states to model inputs.

    Keeps only presence transitions and fixes downsampled to one every
    ``HISTORY_FIX_MIN_INTERVAL_S`` seconds, so raw state rows can be dropped
    chunk by chunk.
    """

    def __init__(self, zone_id: str, zone: HomeZone | None) -> None:
        self.zone_id = zone_id
        self.zone = zone
        self.transitions: list[PresenceTransition] = []
        self.fixes: list[PositionFix] = []
        self._last: bool | None = None
        self._last_state: str | None = None
        self._last_fix_ts: datetime | None = None

    def feed(self, states: Iterable[State]) -> None:
        min_gap = timedelta(seconds=HISTORY_FIX_MIN_INTERVAL_S)
        for state in states:
            is_home = state_is_home(state, self.zone_id, self.zone)
            if is_home is not None and is_home != self._last:
                # last_changed is exact when the state string changed; coordinate
                # based detection (non-default zones) only moves last_updated.
                changed = self._last_state is None or state.state != self._last_state
                ts = state.last_changed if changed else state.last_updated
                if self.transitions and ts < self.transitions[-1].ts:
                    ts = self.transitions[-1].ts
                self.transitions.append(PresenceTransition(ts=ts, is_home=is_home))
                self._last = is_home
            self._last_state = state.state
            fix = state_to_fix(state, self.zone)
            if fix is not None and (
                self._last_fix_ts is None or fix.ts >= self._last_fix_ts + min_gap
            ):
                self.fixes.append(fix)
                self._last_fix_ts = fix.ts


def states_to_history(
    states: list[State], zone_id: str, zone: HomeZone | None
) -> tuple[list[PresenceTransition], list[PositionFix]]:
    """Convert a chronological list of person states into model inputs."""
    reducer = _HistoryReducer(zone_id, zone)
    reducer.feed(states)
    return reducer.transitions, reducer.fixes


def _fetch_states(
    hass: HomeAssistant, entity_id: str, start: datetime, end: datetime
) -> list[State]:
    """Read every state row (attributes included) for one entity. Runs in executor."""
    from homeassistant.components.recorder import history

    result = history.get_significant_states(
        hass,
        start,
        end,
        entity_ids=[entity_id],
        include_start_time_state=True,
        significant_changes_only=False,
        minimal_response=False,
        no_attributes=False,
    )
    return [s for s in result.get(entity_id, []) if isinstance(s, State)]


def _load_history_chunked(
    hass: HomeAssistant,
    entity_id: str,
    zone_id: str,
    zone: HomeZone | None,
    start: datetime,
    end: datetime,
) -> tuple[list[PresenceTransition], list[PositionFix]]:
    """Query the recorder in windows and reduce each one (recorder executor)."""
    reducer = _HistoryReducer(zone_id, zone)
    window = timedelta(days=HISTORY_CHUNK_DAYS)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + window, end)
        reducer.feed(_fetch_states(hass, entity_id, chunk_start, chunk_end))
        chunk_start = chunk_end
    return reducer.transitions, reducer.fixes


async def async_load_person_history(
    hass: HomeAssistant,
    person_id: str,
    zone_id: str,
    zone: HomeZone | None,
    days: int,
) -> tuple[list[PresenceTransition], list[PositionFix]]:
    """Load presence transitions and fixes for a person from the recorder.

    Returns empty lists when the recorder is not available or has no data.
    """
    if "recorder" not in hass.config.components:
        _LOGGER.debug("Recorder not loaded, skipping history bootstrap for %s", person_id)
        return [], []
    from homeassistant.components.recorder import get_instance

    end = dt_util.utcnow()
    start = end - timedelta(days=days)
    try:
        return await get_instance(hass).async_add_executor_job(
            _load_history_chunked, hass, person_id, zone_id, zone, start, end
        )
    except (HomeAssistantError, KeyError, ValueError) as err:
        _LOGGER.warning("Could not read history for %s: %s", person_id, err)
        return [], []


def _event_dates(event: dict[str, Any]) -> list[date]:
    """Every local date covered by a calendar event (end exclusive)."""

    def _parse(raw: Any) -> tuple[date, bool] | None:
        text = str(raw or "")
        if (parsed := dt_util.parse_datetime(text)) is not None:
            local = dt_util.as_local(parsed)
            return local.date(), local.timetz().replace(tzinfo=None) == time.min
        if (parsed_date := dt_util.parse_date(text)) is not None:
            return parsed_date, True
        return None

    start = _parse(event.get("start"))
    if start is None:
        return []
    first = start[0]
    end = _parse(event.get("end"))
    if end is None:
        return [first]
    last = end[0] - timedelta(days=1) if end[1] else end[0]  # midnight end is exclusive
    last = max(last, first)
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


async def async_load_holidays(
    hass: HomeAssistant, calendar_id: str | None, days: int
) -> list[datetime]:
    """Return local midnights of calendar events over the last ``days`` days."""
    if not calendar_id:
        return []
    end = dt_util.now()
    start = end - timedelta(days=days)
    try:
        response = await hass.services.async_call(
            "calendar",
            "get_events",
            {"start_date_time": start.isoformat(), "end_date_time": end.isoformat()},
            target={"entity_id": calendar_id},
            blocking=True,
            return_response=True,
        )
    except HomeAssistantError as err:
        _LOGGER.warning("Could not read holiday calendar %s: %s", calendar_id, err)
        return []
    events: list[dict[str, Any]] = (response or {}).get(calendar_id, {}).get("events", [])
    tz = dt_util.get_default_time_zone()
    holidays: set[datetime] = set()
    for event in events:
        for day in _event_dates(event):
            holidays.add(datetime.combine(day, time.min, tzinfo=tz))
    return sorted(holidays)
