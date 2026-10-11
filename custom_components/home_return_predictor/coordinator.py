"""Data update coordinator: per-person models, listeners, mode logic, aggregation."""

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
from homeassistant.const import STATE_ON
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
    CONF_ENABLE_SKLEARN,
    CONF_HALF_LIFE_DAYS,
    CONF_HOLIDAY_CALENDAR,
    CONF_HOME_ZONE,
    CONF_HORIZON_MINUTES,
    CONF_MATURITY_THRESHOLD,
    CONF_PERSONS,
    CONF_RETURN_THRESHOLD,
    CONF_UPDATE_INTERVAL_MINUTES,
    DEFAULT_ACTIVE_MODEL,
    DEFAULT_ENABLE_SKLEARN,
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
    MATURITY_HYSTERESIS,
    MODE_AUTO,
    MODE_LEARNING,
    MODE_LIVE,
    MODES,
    SKLEARN_REQUIREMENT,
)
from .history import (
    HomeZone,
    async_load_holidays,
    async_load_person_history,
    state_is_home,
    state_to_fix,
)
from .hrp_core import KernelBayesReturnModel, create_model, sklearn_available
from .hrp_core.base import (
    MaturityReport,
    ModelConfig,
    ModelInput,
    PositionFix,
    Prediction,
    PresenceTransition,
    ReturnModel,
)
from .storage import STORAGE_SCHEMA, HRPStorage, StoredData

_LOGGER = logging.getLogger(__name__)

type HRPConfigEntry = ConfigEntry[HRPCoordinator]


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
    fixes: deque[PositionFix] = field(default_factory=deque)
    trackers: tuple[str, ...] = ()
    returning: bool = False
    snapshot: dict[str, Any] | None = None  # last JSON-safe model state
    dirty: bool = True  # model changed since the last snapshot
    # Serialises every access to ``model`` (executor jobs, including challenger
    # training, and in-loop mutations) so the model is never used concurrently.
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
    prediction: Prediction
    maturity: MaturityReport
    mode: str  # selected mode (auto/learning/live)
    effective_mode: str  # learning/live
    returning: bool
    reason: str

    @property
    def advisory(self) -> bool:
        """True when the prediction should only be informative."""
        return self.effective_mode == MODE_LEARNING

    @property
    def active_model(self) -> str:
        """Model whose probability is currently served (champion or promoted challenger)."""
        return str(
            self.prediction.reason.get("active_model")
            or getattr(self.maturity, "active_model", None)
            or DEFAULT_ACTIVE_MODEL
        )


@dataclass
class HouseResult:
    """Aggregate over all persons."""

    probability: float
    occupied: bool
    persons_home: list[str]
    returning_persons: list[str]
    next_eta: datetime | None
    mode: str
    away_probability: float = 0.0  # combined return probability of persons away

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


def _sklearn_installed() -> bool:
    """Cheap check (no import) whether scikit-learn is installed."""
    return importlib.util.find_spec("sklearn") is not None


def _fmt_dt(value: datetime | None) -> str | None:
    return dt_util.as_local(value).strftime("%H:%M") if value else None


def build_reason(prediction: Prediction, left_at: datetime | None) -> str:
    """Render a short human readable explanation of a prediction."""
    if prediction.is_home:
        return "at home"
    parts: list[str] = []
    if left_at is not None:
        label = prediction.reason.get("label") or prediction.reason.get("schedule")
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
            options.get(CONF_ENABLE_SKLEARN, DEFAULT_ENABLE_SKLEARN)
        )
        self.challenger_enabled: bool = False  # requested and scikit-learn installed
        self.sklearn_available: bool = False
        self._train_lock = asyncio.Lock()  # one challenger training at a time
        self._training: set[str] = set()  # persons with a scheduled/running training
        self._unsub_tracking: CALLBACK_TYPE | None = None
        self._tracked: set[str] = set()

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

    # ------------------------------------------------------------------ setup

    async def _async_setup(self) -> None:
        """Load stored models, catch up presence, subscribe, bootstrap history."""
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
                left_at=dt_util.parse_datetime(pdata["left_at"]) if pdata.get("left_at") else None,
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
                self._handle_presence(runtime, state, observe_now=True)
                fix = state_to_fix(state, self.home_zone)
                if fix is not None and runtime.is_home is not True:
                    runtime.add_fix(fix)
        self._async_subscribe()
        self.config_entry.async_create_background_task(
            self.hass, self._async_setup_challenger(), f"{DOMAIN}_challenger_setup"
        )
        if to_bootstrap:
            self.config_entry.async_create_background_task(
                self.hass, self.async_retrain(to_bootstrap), f"{DOMAIN}_bootstrap"
            )

    async def _async_setup_challenger(self) -> None:
        """Install scikit-learn on demand and switch the challenger on/off."""
        enabled = False
        if self.challenger_requested:
            try:
                await async_process_requirements(self.hass, DOMAIN, [SKLEARN_REQUIREMENT])
            except (RequirementsNotFound, OSError) as err:
                _LOGGER.warning(
                    "Could not install %s (%s); the scikit-learn challenger stays disabled "
                    "and the built-in model keeps working",
                    SKLEARN_REQUIREMENT,
                    err,
                )
            else:
                enabled = True
        if self.challenger_requested:
            self.sklearn_available = await self.hass.async_add_executor_job(sklearn_available)
        else:
            # Do not import scikit-learn (memory) when the challenger is off.
            self.sklearn_available = await self.hass.async_add_executor_job(_sklearn_installed)
        if enabled and not self.sklearn_available:
            _LOGGER.warning(
                "%s reported installed but cannot be imported; challenger disabled",
                SKLEARN_REQUIREMENT,
            )
            enabled = False
        self.challenger_enabled = enabled
        for runtime in self.persons.values():
            async with runtime.lock:
                await self.hass.async_add_executor_job(runtime.model.enable_challenger, enabled)
        if enabled:
            _LOGGER.debug("scikit-learn challenger enabled")
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

    async def async_unload(self) -> None:
        """Stop listening and persist everything."""
        if self._unsub_tracking:
            self._unsub_tracking()
            self._unsub_tracking = None
        async with self._lock:
            await self._async_snapshot_dirty()
        await self.storage.async_save(self._store_data())

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
            self._handle_presence(runtime, new_state)
        fix = state_to_fix(new_state, zone)
        if fix is not None:
            for runtime in self.persons.values():
                # Fixes only matter during an absence (cleared again on arrival).
                if (
                    runtime.is_home is not True
                    and (entity_id == runtime.entity_id or entity_id in runtime.trackers)
                    and runtime.add_fix(fix)
                ):
                    new_fix = True
        if new_fix:
            self.config_entry.async_create_task(self.hass, self.async_request_refresh())

    @callback
    def _handle_presence(
        self, runtime: PersonRuntime, state: State, observe_now: bool = False
    ) -> None:
        """Detect home/away changes and feed them to the model."""
        is_home = state_is_home(state, self.home_zone_id, self.home_zone)
        if is_home is None or is_home == runtime.is_home:
            return
        previous = runtime.is_home
        # last_changed is exact for state changes (and across restarts); coordinate
        # based detection (non-default zones) only moves last_updated.
        ts = state.last_changed if previous is None or observe_now else state.last_updated
        runtime.is_home = is_home
        runtime.left_at = None if is_home else ts
        if previous is None:
            return
        transition = PresenceTransition(ts=ts, is_home=is_home)
        holiday = self._is_holiday()
        if observe_now:
            # Setup: nothing else touches the model yet.
            runtime.model.observe_transition(transition, holiday)
            runtime.dirty = True
            return
        self.config_entry.async_create_task(
            self.hass, self._async_observe(runtime, transition, holiday)
        )

    async def _async_observe(
        self, runtime: PersonRuntime, transition: PresenceTransition, holiday: bool
    ) -> None:
        async with self._lock, runtime.lock:
            runtime.model.observe_transition(transition, holiday)
            runtime.dirty = True
            if transition.is_home:
                runtime.fixes.clear()
        self._schedule_save()
        await self.async_request_refresh()

    # ------------------------------------------------------------------ update

    @staticmethod
    def _compute(
        runtime: PersonRuntime, model_input: ModelInput
    ) -> tuple[Prediction, MaturityReport]:
        """Run the model (executor)."""
        model = runtime.model
        resolved = model.resolve(model_input.now, model_input.is_home)
        prediction = model.predict(model_input)
        logged = model.log_prediction(model_input, prediction)
        maturity = model.maturity(model_input.now)
        if resolved or logged or runtime.dirty:
            runtime.snapshot = model.to_dict()
            runtime.dirty = False
        return prediction, maturity

    @staticmethod
    def _snapshot(runtime: PersonRuntime) -> None:
        """Refresh the snapshot of a changed model (executor, under runtime.lock)."""
        if runtime.dirty or runtime.snapshot is None:
            runtime.snapshot = runtime.model.to_dict()
            runtime.dirty = False

    async def _async_snapshot_dirty(self, person_ids: list[str] | None = None) -> None:
        """Refresh snapshots of changed models, one person (lock) at a time."""
        for person_id in person_ids or list(self.persons):
            runtime = self.persons[person_id]
            async with runtime.lock:
                await self.hass.async_add_executor_job(self._snapshot, runtime)

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

    async def _async_update_data(self) -> HRPData:
        now = dt_util.utcnow()
        holiday = self._is_holiday()
        results: dict[str, PersonResult] = {}
        async with self._lock:
            for person_id, runtime in self.persons.items():
                runtime.prune_fixes(now)
                is_home = bool(runtime.is_home)
                model_input = ModelInput(
                    now=now,
                    is_home=is_home,
                    left_at=None if is_home else runtime.left_at,
                    is_holiday=holiday,
                    fixes=tuple(runtime.fixes),
                )
                try:
                    async with runtime.lock:
                        prediction, maturity = await self.hass.async_add_executor_job(
                            self._compute, runtime, model_input
                        )
                        model = runtime.model
                        training_due = self.challenger_enabled and model.challenger_due(now)
                except (ArithmeticError, KeyError, TypeError, ValueError) as err:
                    raise UpdateFailed(
                        translation_domain=DOMAIN,
                        translation_key="update_failed",
                        translation_placeholders={"person": person_id, "error": str(err)},
                    ) from err
                self.last_predictions[person_id] = prediction
                runtime.effective_mode = self._effective_mode(runtime, maturity.maturity)
                returning = (not is_home) and prediction.probability >= self.return_threshold
                result = PersonResult(
                    entity_id=person_id,
                    slug=runtime.slug,
                    name=self.person_name(person_id),
                    is_home=runtime.is_home,
                    prediction=prediction,
                    maturity=maturity,
                    mode=runtime.mode,
                    effective_mode=runtime.effective_mode,
                    returning=returning,
                    reason=build_reason(prediction, model_input.left_at),
                )
                if returning and not runtime.returning:
                    self._fire_return_event(result)
                runtime.returning = returning
                results[person_id] = result
                if training_due:
                    self._schedule_training(runtime, now)
        self._schedule_save()
        return HRPData(persons=results, house=self._aggregate(results))

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
        try:
            async with self._train_lock, runtime.lock:
                future = self.hass.async_add_executor_job(runtime.model.train_challenger, now)
                try:
                    summary = await asyncio.shield(future)
                except asyncio.CancelledError:
                    # Keep the model locked until the worker thread really finishes.
                    await future
                    raise
                runtime.dirty = True
        except asyncio.CancelledError:
            raise
        except Exception:  # a failed challenger must never stop the champion
            _LOGGER.warning("Challenger training failed for %s", runtime.entity_id, exc_info=True)
            return
        finally:
            self._training.discard(runtime.entity_id)
        _LOGGER.debug("Challenger training for %s: %s", runtime.entity_id, summary)
        self._schedule_save()

    def _fire_return_event(self, result: PersonResult) -> None:
        eta = result.prediction.eta
        self.hass.bus.async_fire(
            EVENT_RETURN_DETECTED,
            {
                "person": result.entity_id,
                "name": result.name,
                "probability": round(result.prediction.probability, 3),
                "eta": eta.isoformat() if eta else None,
                "mode": result.effective_mode,
                "advisory": result.advisory,
            },
        )

    @staticmethod
    def _aggregate(results: dict[str, PersonResult]) -> HouseResult:
        persons_home = [pid for pid, r in results.items() if r.is_home]
        returning = [pid for pid, r in results.items() if r.returning]
        away_probability = 1.0 - math.prod(
            1.0 - r.prediction.probability for r in results.values() if not r.is_home
        )
        probability = 1.0 if persons_home else away_probability
        etas = [e for pid in returning if (e := results[pid].prediction.eta) is not None]
        mode_source = returning or list(results)
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
            next_eta=min(etas) if etas else None,
            mode=mode,
            away_probability=away_probability,
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
                }
                for pid, rt in self.persons.items()
            },
        }

    @callback
    def _schedule_save(self) -> None:
        self.storage.async_schedule_save(self._store_data)

    # ------------------------------------------------------------------ actions

    async def async_set_mode(self, person_id: str, mode: str) -> None:
        """Change the selected mode of a person."""
        self.persons[person_id].mode = mode
        self._schedule_save()
        await self.async_refresh()

    async def async_reset(self, person_id: str) -> None:
        """Forget everything learnt for a person."""
        async with self._lock:
            runtime = self.persons[person_id]
            async with runtime.lock:
                await self.hass.async_add_executor_job(self._reset_model, runtime)
            runtime.fixes.clear()
            runtime.returning = False
            runtime.effective_mode = None
            runtime.dirty = True
        await self.async_refresh()

    async def async_retrain(self, person_ids: list[str] | None = None) -> None:
        """Rebuild models from recorder history."""
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
                async with runtime.lock:
                    if transitions or fixes:
                        await self.hass.async_add_executor_job(
                            runtime.model.load_history, transitions, fixes, holidays
                        )
                        runtime.dirty = True
                    last = transitions[-1] if transitions else None
                    if last is not None and not last.is_home and runtime.left_at is None:
                        runtime.left_at = last.ts
                await self._async_snapshot_dirty()
        self._schedule_save()
        await self.async_request_refresh()

    def prediction_log(self, person_id: str, limit: int) -> list[dict[str, Any]]:
        """Return the last ``limit`` prediction log entries (resolved and pending)."""
        snapshot = self.persons[person_id].snapshot or {}
        log = snapshot.get("log")
        if not isinstance(log, dict):
            return []
        entries = [*log.get("resolved", []), *log.get("pending", [])]
        entries.sort(key=lambda e: str(e.get("ts", "")))
        return entries[-limit:]
