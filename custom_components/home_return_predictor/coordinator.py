"""Data update coordinator: per-person models, listeners, mode logic, aggregation.

Concurrency rules:

* every model call runs in the executor while holding the person's
  ``PersonRuntime.lock`` (the model is never used concurrently);
* lock order is always ``HRPCoordinator._lock`` (updates/observations/retrain)
  then ``PersonRuntime.lock``; nothing holds a person lock for long work:
  challenger training snapshots under the lock, fits without it and applies
  the result under the lock again.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import EVENT_HOMEASSISTANT_STOP, STATE_ON
from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
    State,
    callback,
)
from homeassistant.helpers.event import async_track_state_change_event
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
from homeassistant.requirements import RequirementsNotFound, async_process_requirements
from homeassistant.util import dt as dt_util

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
    DEFAULT_ACTIVE_MODEL,
    DEFAULT_ENABLE_CHALLENGER,
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_HOME_ZONE,
    DEFAULT_HORIZON_MINUTES,
    DEFAULT_MATURITY_THRESHOLD,
    DEFAULT_RETURN_THRESHOLD,
    DEFAULT_UPDATE_INTERVAL_MINUTES,
    DOMAIN,
    EVENT_RETURN_DETECTED,
    FIX_WINDOW_MINUTES,
    HISTORY_DAYS,
    HOUSE_ETA_MIN_PROBABILITY,
    MATURITY_HYSTERESIS,
    MODE_AUTO,
    MODE_LEARNING,
    MODE_LIVE,
    MODES,
    NUMPY_REQUIREMENT,
    RETURNING_OFF_HYSTERESIS,
)
from .history import (
    HomeZone,
    async_load_holidays,
    async_load_person_history,
    state_is_home,
    state_to_fix,
)
from .hrp_core import KernelBayesReturnModel, create_model, numpy_available
from .hrp_core.base import (
    MaturityReport,
    ModelConfig,
    ModelInput,
    PositionFix,
    Prediction,
    PresenceTransition,
    ReturnModel,
)
from .hrp_core.maturity import decode_entries
from .storage import STORAGE_SCHEMA, HRPStorage, StoredData

_LOGGER = logging.getLogger(__name__)

type HRPConfigEntry = ConfigEntry[HRPCoordinator]

REASON_STATE_UNKNOWN = "state unknown"
OBSERVE_FLUSH_TIMEOUT = 10.0  # seconds to wait for pending observations on unload


@dataclass
class PersonRuntime:
    """Mutable runtime state for one person."""

    entity_id: str
    slug: str
    model: ReturnModel  # KernelBayesReturnModel (also provides the challenger hooks)
    mode: str = MODE_AUTO
    effective_mode: str | None = None
    is_home: bool | None = None
    left_at: datetime | None = None
    event_left_at: datetime | None = None  # absence for which the return event fired
    fixes: deque[PositionFix] = field(default_factory=deque)
    trackers: tuple[str, ...] = ()
    returning: bool = False
    snapshot: dict[str, Any] | None = None  # last JSON-safe model state
    dirty: bool = True  # model changed since the last snapshot
    # Serialises every access to ``model`` (always in the executor) so the model
    # is never used concurrently. Only held for short jobs.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def add_fix(self, fix: PositionFix) -> bool:
        """Append a fix, skipping duplicates (person mirrors its source tracker)."""
        for old in reversed(self.fixes):
            if (
                abs((old.ts - fix.ts).total_seconds()) < 5
                and abs(old.distance_m - fix.distance_m) < 1
            ):
                return False
        self.fixes.append(fix)
        if len(self.fixes) > 1 and self.fixes[-2].ts > fix.ts:
            self.fixes = deque(sorted(self.fixes, key=lambda f: f.ts))
        return True

    def prune_fixes(self, now: datetime) -> None:
        """Drop fixes older than the tracking window."""
        cutoff = now - timedelta(minutes=FIX_WINDOW_MINUTES)
        while self.fixes and self.fixes[0].ts < cutoff:
            self.fixes.popleft()


@dataclass
class PersonResult:
    """Result of one update for one person (read by entities)."""

    entity_id: str
    slug: str
    name: str
    is_home: bool | None
    prediction: Prediction | None  # None while the presence state is unknown
    maturity: MaturityReport
    mode: str  # selected mode (auto/learning/live)
    effective_mode: str  # learning/live
    returning: bool
    reason: str

    @property
    def state_unknown(self) -> bool:
        """True when the person's presence is unknown (no prediction)."""
        return self.is_home is None

    @property
    def advisory(self) -> bool:
        """True when the prediction should only be informative."""
        return self.effective_mode == MODE_LEARNING

    @property
    def active_model(self) -> str:
        """Model whose probability is currently served (champion or promoted challenger)."""
        reason = self.prediction.reason if self.prediction is not None else {}
        return str(
            reason.get("active_model")
            or getattr(self.maturity, "active_model", None)
            or DEFAULT_ACTIVE_MODEL
        )


@dataclass
class HouseResult:
    """Aggregate over all persons with a known presence."""

    probability: float | None  # None when no person has a known presence
    occupied: bool
    persons_home: list[str]
    returning_persons: list[str]
    next_eta: datetime | None
    mode: str
    away_probability: float = 0.0  # combined return probability of persons away
    eta_person: str | None = None  # person whose ETA is ``next_eta``
    persons_unknown: list[str] = field(default_factory=list)
    # (person, probability 0..1, ETA) of away persons with p >= HOUSE_ETA_MIN_PROBABILITY,
    # sorted by ETA (None last): each return judged on its own (blueprint).
    candidates: list[tuple[str, float, datetime | None]] = field(default_factory=list)

    @property
    def advisory(self) -> bool:
        """True unless the aggregate relies on a live model."""
        return self.mode != MODE_LIVE

    @property
    def someone_returning(self) -> bool:
        """True when at least one person is returning."""
        return bool(self.returning_persons)


@dataclass
class HRPData:
    """Coordinator data."""

    persons: dict[str, PersonResult]
    house: HouseResult


@dataclass
class _Computed:
    """Executor result of one person update."""

    prediction: Prediction | None
    maturity: MaturityReport
    training_due: bool
    snapshot_changed: bool


def _numpy_installed() -> bool:
    """Cheap check (no import) whether numpy is installed."""
    return importlib.util.find_spec("numpy") is not None


def _fmt_dt(value: datetime | None) -> str | None:
    return dt_util.as_local(value).strftime("%H:%M") if value else None


def build_reason(prediction: Prediction, left_at: datetime | None) -> str:
    """Render a short human readable explanation of a prediction."""
    if prediction.is_home:
        return "at home"
    parts: list[str] = []
    if left_at is not None:
        label = prediction.reason.get("shift_label")
        parts.append(f"left {_fmt_dt(left_at)}" + (f" ({label})" if label else ""))
    if prediction.distance_m is not None:
        motion = ""
        speed = prediction.approach_speed_kmh
        if speed is not None:
            if speed > 1:
                motion = f"approaching {speed:.0f} km/h, "
            elif speed < -1:
                motion = f"moving away {-speed:.0f} km/h, "
            else:
                motion = "still, "
        parts.append(f"{motion}{prediction.distance_m / 1000:.1f} km")
    evidence = []
    if prediction.prior_probability is not None:
        evidence.append(f"habit P {prediction.prior_probability:.2f}")
    evidence.append(f"GPS LR {prediction.likelihood_ratio:.1f}")
    parts.append(", ".join(evidence))
    if prediction.eta is not None:
        parts.append(f"ETA {_fmt_dt(prediction.eta)}")
    if active := prediction.reason.get("active_model"):
        parts.append(f"model {active}")
    return "; ".join(parts)[:255]


class HRPCoordinator(DataUpdateCoordinator[HRPData]):
    """Coordinate all persons of the (single) config entry."""

    config_entry: HRPConfigEntry

    def __init__(self, hass: HomeAssistant, entry: HRPConfigEntry) -> None:
        """Initialize the coordinator."""
        options = entry.options
        super().__init__(
            hass,
            _LOGGER,
            config_entry=entry,
            name=DOMAIN,
            update_interval=timedelta(
                minutes=options.get(CONF_UPDATE_INTERVAL_MINUTES, DEFAULT_UPDATE_INTERVAL_MINUTES)
            ),
        )
        self.person_ids: list[str] = list(
            options.get(CONF_PERSONS) or entry.data.get(CONF_PERSONS, [])
        )
        self.home_zone_id: str = entry.data.get(CONF_HOME_ZONE, DEFAULT_HOME_ZONE)
        self.return_threshold: float = options.get(CONF_RETURN_THRESHOLD, DEFAULT_RETURN_THRESHOLD)
        self.maturity_threshold: float = options.get(
            CONF_MATURITY_THRESHOLD, DEFAULT_MATURITY_THRESHOLD
        )
        self.holiday_calendar: str | None = options.get(CONF_HOLIDAY_CALENDAR) or None
        self.model_config = ModelConfig(
            horizon_minutes=float(options.get(CONF_HORIZON_MINUTES, DEFAULT_HORIZON_MINUTES)),
            half_life_days=float(options.get(CONF_HALF_LIFE_DAYS, DEFAULT_HALF_LIFE_DAYS)),
            timezone=dt_util.get_default_time_zone(),
        )
        self.storage = HRPStorage(hass, entry.entry_id)
        self.persons: dict[str, PersonRuntime] = {}
        self.last_predictions: dict[str, Prediction] = {}
        self._lock = asyncio.Lock()
        self.challenger_requested: bool = bool(
            options.get(CONF_ENABLE_CHALLENGER, DEFAULT_ENABLE_CHALLENGER)
        )
        self.challenger_enabled: bool = False  # requested and numpy importable
        self.numpy_available: bool = False
        self._train_lock = asyncio.Lock()  # one challenger fit at a time
        self._training: set[str] = set()  # persons with a scheduled/running training
        self._unsub_tracking: CALLBACK_TYPE | None = None
        self._unsub_stop: CALLBACK_TYPE | None = None
        self._tracked: set[str] = set()
        # Pending ``_async_observe`` tasks; awaited before the final save.
        self._observe_tasks: set[asyncio.Task[None]] = set()
        self._unloading = False
        # Change counter of the persisted document; a save is scheduled only when
        # it moved since the last scheduled save.
        self._data_version = 0
        self._saved_version = 0

    # ------------------------------------------------------------------ helpers

    @property
    def training_persons(self) -> set[str]:
        """Persons whose challenger training is scheduled or running."""
        return set(self._training)

    @property
    def home_zone(self) -> HomeZone | None:
        """Current home zone geometry."""
        return HomeZone.from_state(self.hass.states.get(self.home_zone_id))

    def person_name(self, person_id: str) -> str:
        """Friendly name of a person."""
        state = self.hass.states.get(person_id)
        return state.name if state else person_id.split(".", 1)[1].replace("_", " ").title()

    def _is_holiday(self) -> bool:
        return bool(self.holiday_calendar) and self.hass.states.is_state(
            self.holiday_calendar, STATE_ON
        )

    @staticmethod
    def _trackers_of(state: State | None) -> tuple[str, ...]:
        if state is None:
            return ()
        trackers = list(state.attributes.get("device_trackers") or [])
        if (source := state.attributes.get("source")) and source not in trackers:
            trackers.append(source)
        return tuple(str(t) for t in trackers)

    @callback
    def _mark_changed(self) -> None:
        """Record that the persisted document changed."""
        self._data_version += 1

    async def _async_model_job(self, runtime: PersonRuntime, func: Any, *args: Any) -> Any:
        """Run one model call in the executor under the person lock."""
        async with runtime.lock:
            return await self.hass.async_add_executor_job(func, *args)

    # ------------------------------------------------------------------ setup

    async def _async_setup(self) -> None:
        """Load stored models, catch up presence, subscribe, bootstrap history."""
        # Registered first so a failed first refresh never leaks listeners.
        self.config_entry.async_on_unload(self._async_unsubscribe)
        stored = await self.storage.async_load()
        stored_persons: dict[str, Any] = stored.get("persons", {})
        to_bootstrap: list[str] = []
        for person_id in self.person_ids:
            pdata: dict[str, Any] = stored_persons.get(person_id) or {}
            model: ReturnModel | None = None
            if pdata.get("model"):
                try:
                    model = await self.hass.async_add_executor_job(
                        KernelBayesReturnModel.from_dict, pdata["model"], self.model_config
                    )
                except (KeyError, TypeError, ValueError) as err:
                    _LOGGER.warning("Discarding unreadable model of %s: %s", person_id, err)
            runtime = PersonRuntime(
                entity_id=person_id,
                slug=person_id.split(".", 1)[1],
                model=model or create_model(self.model_config),
                mode=pdata.get("mode") if pdata.get("mode") in MODES else MODE_AUTO,
                is_home=pdata.get("is_home"),
                left_at=_parse_dt(pdata.get("left_at")),
                event_left_at=_parse_dt(pdata.get("event_left_at")),
            )
            if model is None:
                to_bootstrap.append(person_id)
            else:
                runtime.snapshot = pdata["model"]
                runtime.dirty = False
            self.persons[person_id] = runtime
            state = self.hass.states.get(person_id)
            runtime.trackers = self._trackers_of(state)
            if state is not None:
                transition = self._handle_presence(runtime, state, at_setup=True)
                if transition is not None:
                    await self._async_model_job(
                        runtime, runtime.model.observe_transition, transition, self._is_holiday()
                    )
                    runtime.dirty = True
                fix = state_to_fix(state, self.home_zone)
                if fix is not None and runtime.is_home is False:
                    runtime.add_fix(fix)
        self._async_subscribe()
        self._unsub_stop = self.hass.bus.async_listen_once(
            EVENT_HOMEASSISTANT_STOP, self._async_handle_stop
        )
        self.config_entry.async_create_background_task(
            self.hass, self._async_setup_challenger(), f"{DOMAIN}_challenger_setup"
        )
        if to_bootstrap:
            self.config_entry.async_create_background_task(
                self.hass, self.async_retrain(to_bootstrap, bootstrap=True), f"{DOMAIN}_bootstrap"
            )

    @property
    def sklearn_available(self) -> bool:
        """Deprecated alias of ``numpy_available`` (attribute kept for one release)."""
        return self.numpy_available

    async def _async_setup_challenger(self) -> None:
        """Make numpy available on demand and switch the challenger on/off.

        numpy ships with Home Assistant on most installs: then nothing is installed.
        Otherwise it is requested without a version (HA's constraints pin it).
        """
        enabled = False
        if self.challenger_requested:
            available = await self.hass.async_add_executor_job(numpy_available)
            if not available:
                try:
                    await async_process_requirements(self.hass, DOMAIN, [NUMPY_REQUIREMENT])
                except (RequirementsNotFound, OSError) as err:
                    _LOGGER.warning(
                        "Could not install %s (%s); the gradient-boosting challenger stays "
                        "disabled and the built-in model keeps working",
                        NUMPY_REQUIREMENT,
                        err,
                    )
                else:
                    available = await self.hass.async_add_executor_job(numpy_available)
                    if not available:
                        _LOGGER.warning(
                            "%s reported installed but cannot be imported; the "
                            "gradient-boosting challenger stays disabled",
                            NUMPY_REQUIREMENT,
                        )
            self.numpy_available = enabled = available
        else:
            # Challenger off: only a cheap check, no import.
            self.numpy_available = await self.hass.async_add_executor_job(_numpy_installed)
        self.challenger_enabled = enabled
        for runtime in self.persons.values():
            await self._async_model_job(runtime, runtime.model.enable_challenger, enabled)
        # Entities read these flags from the coordinator: show them right away.
        if self.data is not None:
            self.async_update_listeners()
        if enabled:
            _LOGGER.debug("gradient-boosting challenger enabled")
            await self.async_request_refresh()

    @callback
    def _async_subscribe(self) -> None:
        """(Re)subscribe to persons and their device trackers."""
        entities = set(self.persons)
        for runtime in self.persons.values():
            entities.update(runtime.trackers)
        if entities == self._tracked and self._unsub_tracking:
            return
        if self._unsub_tracking:
            self._unsub_tracking()
        self._tracked = entities
        self._unsub_tracking = async_track_state_change_event(
            self.hass, sorted(entities), self._async_handle_state_event
        )

    @callback
    def _async_unsubscribe(self) -> None:
        """Remove every listener (idempotent)."""
        if self._unsub_tracking:
            self._unsub_tracking()
            self._unsub_tracking = None
        self._tracked = set()
        if self._unsub_stop:
            self._unsub_stop()
            self._unsub_stop = None

    async def _async_handle_stop(self, _event: Event) -> None:
        """Home Assistant stops: refresh changed snapshots; Store writes on final write."""
        self._unsub_stop = None  # listen_once already removed itself
        await self._async_wait_observers()
        await self._async_snapshot_dirty()
        self._schedule_save(force=True)

    async def _async_wait_observers(self) -> None:
        """Let pending observations reach the models before a final save."""
        if not (pending := set(self._observe_tasks)):
            return
        _done, not_done = await asyncio.wait(pending, timeout=OBSERVE_FLUSH_TIMEOUT)
        if not_done:
            _LOGGER.warning(
                "%d presence observations still pending at shutdown; not persisted",
                len(not_done),
            )

    async def async_unload(self) -> None:
        """Stop listening and persist everything."""
        self._unloading = True
        self._async_unsubscribe()
        await self._async_wait_observers()
        async with self._lock:
            await self._async_snapshot_dirty()
        await self.storage.async_save(self._store_data())
        self._saved_version = self._data_version

    # ------------------------------------------------------------------ events

    @callback
    def _async_handle_state_event(self, event: Event[EventStateChangedData]) -> None:
        entity_id = event.data["entity_id"]
        new_state = event.data["new_state"]
        if new_state is None:
            return
        zone = self.home_zone
        new_fix = False
        if (runtime := self.persons.get(entity_id)) is not None:
            trackers = self._trackers_of(new_state)
            if trackers != runtime.trackers:
                runtime.trackers = trackers
                self._async_subscribe()
            transition = self._handle_presence(runtime, new_state)
            if transition is not None:
                task = self.config_entry.async_create_task(
                    self.hass, self._async_observe(runtime, transition, self._is_holiday())
                )
                self._observe_tasks.add(task)
                task.add_done_callback(self._observe_tasks.discard)
        fix = state_to_fix(new_state, zone)
        if fix is not None:
            for runtime in self.persons.values():
                # Fixes only matter during an absence (cleared again on arrival).
                if (
                    runtime.is_home is False
                    and (entity_id == runtime.entity_id or entity_id in runtime.trackers)
                    and runtime.add_fix(fix)
                ):
                    new_fix = True
        if new_fix:
            self.config_entry.async_create_task(self.hass, self.async_request_refresh())

    @callback
    def _handle_presence(
        self, runtime: PersonRuntime, state: State, at_setup: bool = False
    ) -> PresenceTransition | None:
        """Track home/away changes; return the transition to feed to the model."""
        is_home = state_is_home(state, self.home_zone_id, self.home_zone)
        if is_home is None or is_home == runtime.is_home:
            return None
        previous = runtime.is_home
        # last_changed is exact for state changes (and across restarts); coordinate
        # based detection (non-default zones) only moves last_updated.
        ts = state.last_changed if previous is None or at_setup else state.last_updated
        runtime.is_home = is_home
        runtime.left_at = None if is_home else ts
        self._mark_changed()
        if previous is None:
            return None
        return PresenceTransition(ts=ts, is_home=is_home)

    async def _async_observe(
        self, runtime: PersonRuntime, transition: PresenceTransition, holiday: bool
    ) -> None:
        async with self._lock:
            await self._async_model_job(
                runtime, runtime.model.observe_transition, transition, holiday
            )
            runtime.dirty = True
            if transition.is_home:
                runtime.fixes.clear()
        if not self._unloading:
            await self.async_request_refresh()

    # ------------------------------------------------------------------ update

    def _compute(self, runtime: PersonRuntime, model_input: ModelInput | None) -> _Computed:
        """Run the model (executor, under runtime.lock). ``None`` input = state unknown."""
        model = runtime.model
        now = model_input.now if model_input is not None else dt_util.utcnow()
        prediction: Prediction | None = None
        changed = runtime.dirty or runtime.snapshot is None
        if model_input is not None:
            resolved = model.resolve(model_input.now, model_input.is_home)
            prediction = model.predict(model_input)
            logged = model.log_prediction(model_input, prediction)
            changed = changed or bool(resolved) or logged
        maturity = model.maturity(now)
        if changed:
            runtime.snapshot = model.to_dict()
            runtime.dirty = False
        training_due = self.challenger_enabled and model.challenger_due(now)
        return _Computed(prediction, maturity, training_due, changed)

    @staticmethod
    def _snapshot(runtime: PersonRuntime) -> bool:
        """Refresh the snapshot of a changed model (executor, under runtime.lock)."""
        if runtime.dirty or runtime.snapshot is None:
            runtime.snapshot = runtime.model.to_dict()
            runtime.dirty = False
            return True
        return False

    async def _async_snapshot_dirty(self, person_ids: list[str] | None = None) -> None:
        """Refresh snapshots of changed models, one person (lock) at a time."""
        for person_id in person_ids or list(self.persons):
            runtime = self.persons[person_id]
            if await self._async_model_job(runtime, self._snapshot, runtime):
                self._mark_changed()

    def _reset_model(self, runtime: PersonRuntime) -> None:
        """Reset a model, keeping the challenger switch (executor, under runtime.lock)."""
        runtime.model.reset()
        runtime.model.enable_challenger(self.challenger_enabled)

    def _effective_mode(self, runtime: PersonRuntime, maturity: float) -> str:
        if runtime.mode != MODE_AUTO:
            return runtime.mode
        if runtime.effective_mode == MODE_LIVE:
            threshold = self.maturity_threshold - MATURITY_HYSTERESIS
        else:
            threshold = self.maturity_threshold
        return MODE_LIVE if maturity >= threshold else MODE_LEARNING

    def _is_returning(self, runtime: PersonRuntime, prediction: Prediction | None) -> bool:
        """Returning with off-hysteresis (stays on until p < threshold - 0.05)."""
        if prediction is None or runtime.is_home is not False:
            return False
        threshold = self.return_threshold
        if runtime.returning:
            threshold -= RETURNING_OFF_HYSTERESIS
        return prediction.probability >= threshold

    async def _async_update_data(self) -> HRPData:
        now = dt_util.utcnow()
        holiday = self._is_holiday()
        results: dict[str, PersonResult] = {}
        async with self._lock:
            for person_id, runtime in self.persons.items():
                runtime.prune_fixes(now)
                is_home = runtime.is_home
                model_input = (
                    None
                    if is_home is None
                    else ModelInput(
                        now=now,
                        is_home=is_home,
                        left_at=None if is_home else runtime.left_at,
                        is_holiday=holiday,
                        fixes=tuple(runtime.fixes),
                    )
                )
                try:
                    computed: _Computed = await self._async_model_job(
                        runtime, self._compute, runtime, model_input
                    )
                except (ArithmeticError, KeyError, TypeError, ValueError) as err:
                    raise UpdateFailed(
                        translation_domain=DOMAIN,
                        translation_key="update_failed",
                        translation_placeholders={"person": person_id, "error": str(err)},
                    ) from err
                if computed.snapshot_changed:
                    self._mark_changed()
                prediction = computed.prediction
                if prediction is not None:
                    self.last_predictions[person_id] = prediction
                else:
                    self.last_predictions.pop(person_id, None)
                runtime.effective_mode = self._effective_mode(runtime, computed.maturity.maturity)
                returning = self._is_returning(runtime, prediction)
                result = PersonResult(
                    entity_id=person_id,
                    slug=runtime.slug,
                    name=self.person_name(person_id),
                    is_home=is_home,
                    prediction=prediction,
                    maturity=computed.maturity,
                    mode=runtime.mode,
                    effective_mode=runtime.effective_mode,
                    returning=returning,
                    reason=(
                        build_reason(prediction, runtime.left_at if not is_home else None)
                        if prediction is not None
                        else REASON_STATE_UNKNOWN
                    ),
                )
                if returning and self._should_fire(runtime):
                    self._fire_return_event(result)
                runtime.returning = returning
                results[person_id] = result
                if computed.training_due:
                    self._schedule_training(runtime, now)
        self._schedule_save()
        return HRPData(persons=results, house=self._aggregate(results))

    @callback
    def _should_fire(self, runtime: PersonRuntime) -> bool:
        """One return event per absence (keyed on left_at, persisted across restarts)."""
        if runtime.left_at is None:
            return not runtime.returning  # unknown departure: rising edge only
        if runtime.event_left_at == runtime.left_at:
            return False
        runtime.event_left_at = runtime.left_at
        self._mark_changed()
        return True

    # ------------------------------------------------------------------ challenger

    @callback
    def _schedule_training(self, runtime: PersonRuntime, now: datetime) -> None:
        """Train the challenger of a person in the background (once at a time)."""
        if runtime.entity_id in self._training:
            return
        self._training.add(runtime.entity_id)
        self.config_entry.async_create_background_task(
            self.hass,
            self._async_train(runtime, now),
            f"{DOMAIN}_train_challenger_{runtime.slug}",
        )

    async def _async_train(self, runtime: PersonRuntime, now: datetime) -> None:
        """Snapshot (locked) -> fit (unlocked, heavy) -> apply (locked)."""
        model = runtime.model
        try:
            async with self._train_lock:
                snapshot = await self._async_model_job(runtime, model.challenger_snapshot)
                fitted = await self.hass.async_add_executor_job(
                    type(model).fit_challenger, snapshot, now
                )
                summary = await self._async_model_job(runtime, model.apply_challenger, fitted, now)
                runtime.dirty = True
        except asyncio.CancelledError:
            raise
        except Exception:  # a failed challenger must never stop the champion
            _LOGGER.warning("Challenger training failed for %s", runtime.entity_id, exc_info=True)
            return
        finally:
            self._training.discard(runtime.entity_id)
        _LOGGER.debug("Challenger training for %s: %s", runtime.entity_id, summary)
        await self._async_snapshot_dirty([runtime.entity_id])
        self._schedule_save()

    def _fire_return_event(self, result: PersonResult) -> None:
        prediction = result.prediction
        if prediction is None:
            return
        eta = prediction.eta
        self.hass.bus.async_fire(
            EVENT_RETURN_DETECTED,
            {
                "person": result.entity_id,
                "name": result.name,
                "probability": round(prediction.probability, 3),
                "eta": eta.isoformat() if eta else None,
                "mode": result.effective_mode,
                "advisory": result.advisory,
            },
        )

    @staticmethod
    def _aggregate(results: dict[str, PersonResult]) -> HouseResult:
        persons_home = [pid for pid, r in results.items() if r.is_home is True]
        away = {
            pid: r.prediction
            for pid, r in results.items()
            if r.is_home is False and r.prediction is not None
        }
        unknown = [pid for pid, r in results.items() if r.state_unknown]
        returning = [pid for pid, r in results.items() if r.returning]
        away_probability = 1.0 - math.prod(1.0 - p.probability for p in away.values())
        probability: float | None
        if persons_home:
            probability = 1.0
        elif away:
            probability = away_probability
        else:
            probability = None
        candidates = sorted(
            (
                (pid, p.probability, p.eta)
                for pid, p in away.items()
                if p.probability >= HOUSE_ETA_MIN_PROBABILITY
            ),
            key=lambda c: (c[2] is None, c[2] or datetime.min, c[0]),
        )
        eta_candidates = [(eta, pid) for pid, _p, eta in candidates if eta is not None]
        next_eta, eta_person = min(eta_candidates) if eta_candidates else (None, None)
        known = [pid for pid in results if pid not in unknown]
        mode_source = returning or known or list(results)
        mode = (
            MODE_LIVE
            if any(results[pid].effective_mode == MODE_LIVE for pid in mode_source)
            else MODE_LEARNING
        )
        return HouseResult(
            probability=probability,
            occupied=bool(persons_home),
            persons_home=persons_home,
            returning_persons=returning,
            next_eta=next_eta,
            mode=mode,
            away_probability=away_probability,
            eta_person=eta_person,
            persons_unknown=unknown,
            candidates=candidates,
        )

    # ------------------------------------------------------------------ storage

    def _store_data(self) -> StoredData:
        return {
            "schema": STORAGE_SCHEMA,
            "persons": {
                pid: {
                    "model": rt.snapshot,
                    "mode": rt.mode,
                    "is_home": rt.is_home,
                    "left_at": rt.left_at.isoformat() if rt.left_at else None,
                    "event_left_at": rt.event_left_at.isoformat() if rt.event_left_at else None,
                }
                for pid, rt in self.persons.items()
            },
        }

    @callback
    def _schedule_save(self, force: bool = False) -> None:
        """Schedule a delayed save when the persisted document changed."""
        if not force and self._data_version == self._saved_version:
            return
        self._saved_version = self._data_version
        # data_func runs at write time, so later changes are included as well.
        self.storage.async_schedule_save(self._store_data)

    # ------------------------------------------------------------------ actions

    async def async_set_mode(self, person_id: str, mode: str) -> None:
        """Change the selected mode of a person."""
        self.persons[person_id].mode = mode
        self._mark_changed()
        self._schedule_save()
        await self.async_refresh()

    async def async_reset(self, person_id: str) -> None:
        """Forget everything learnt for a person."""
        async with self._lock:
            runtime = self.persons[person_id]
            await self._async_model_job(runtime, self._reset_model, runtime)
            runtime.fixes.clear()
            runtime.returning = False
            runtime.effective_mode = None
            runtime.dirty = True
        await self.async_refresh()

    async def async_retrain(
        self, person_ids: list[str] | None = None, bootstrap: bool = False
    ) -> None:
        """Merge recorder history into the models (bootstrap or ``retrain`` service)."""
        holidays = await async_load_holidays(self.hass, self.holiday_calendar, HISTORY_DAYS)
        for person_id in person_ids or list(self.persons):
            runtime = self.persons[person_id]
            transitions, fixes = await async_load_person_history(
                self.hass, person_id, self.home_zone_id, self.home_zone, HISTORY_DAYS
            )
            _LOGGER.debug(
                "History for %s: %d transitions, %d fixes", person_id, len(transitions), len(fixes)
            )
            async with self._lock:
                if transitions or fixes:
                    await self._async_model_job(
                        runtime, runtime.model.load_history, transitions, fixes, holidays
                    )
                    runtime.dirty = True
                last = transitions[-1] if transitions else None
                if (
                    last is not None
                    and not last.is_home
                    and runtime.is_home is not True
                    and (
                        runtime.left_at is None
                        # Bootstrap: left_at came from last_changed at boot, which is
                        # the restart time when the absence started earlier.
                        or (bootstrap and last.ts < runtime.left_at)
                    )
                ):
                    runtime.left_at = last.ts
                    self._mark_changed()
                await self._async_snapshot_dirty([person_id])
        self._schedule_save()
        await self.async_request_refresh()

    def prediction_log(self, person_id: str, limit: int) -> list[dict[str, Any]]:
        """Return the last ``limit`` prediction log entries (resolved and pending)."""
        snapshot = self.persons[person_id].snapshot or {}
        log = snapshot.get("log")
        if not isinstance(log, dict):
            return []
        entries = [*decode_entries(log.get("resolved")), *decode_entries(log.get("pending"))]
        entries.sort(key=lambda e: e.ts)
        return [e.to_dict() for e in entries[-limit:]]

    async def async_episode_count(self, person_id: str) -> int:
        """Number of stored episodes (model call under the person lock)."""
        runtime = self.persons[person_id]
        return len(await self._async_model_job(runtime, runtime.model.episodes))


def _parse_dt(value: Any) -> datetime | None:
    return dt_util.parse_datetime(value) if isinstance(value, str) and value else None
