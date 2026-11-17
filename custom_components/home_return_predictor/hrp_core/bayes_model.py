"""KernelBayesReturnModel: kernel survival prior x calibrated GPS likelihood ratio."""

from __future__ import annotations

import bisect
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

from .arrival import ArrivalDistribution, arrival_tod_density, habit_arrival, with_evidence
from .base import (
    AbsenceEpisode,
    MaturityReport,
    ModelConfig,
    ModelInput,
    PositionFix,
    Prediction,
    PresenceTransition,
    ReturnModel,
)
from .episodes import (
    KIND_EPISODE,
    KIND_TRIP,
    MIN_HOME_STAY_MINUTES,
    build_episodes,
    classify_episode,
    make_episode,
    minutes_between,
    to_local,
    to_utc,
)
from .evidence import (
    EXTENDED_WINDOW_FACTOR,
    CalibrationSegments,
    LRCalibrator,
    compute_evidence,
)
from .fusion import (
    apply_feasibility,
    combine_eta,
    fuse_probability,
    min_travel_minutes,
    position_eta_minutes,
)
from .maturity import (
    LogEntry,
    PredictionLog,
    compact_entry,
    decode_entries,
    encode_entries,
    maturity_report,
    resolve_entries,
)
from .prior import (
    EpisodeKey,
    PriorResult,
    duration_baseline,
    episode_keys,
    estimate_prior,
)

KERNEL_BAYES = "kernel_bayes"  # the only engine (name kept for the attributes)
FUSION_ALIAS = {KERNEL_BAYES: "bayes"}

# v2: single engine. The learned-engine blocks of v1 (stacker, selection, challenger,
# last_training, since_refit) are no longer written and are ignored when loading;
# episodes, trips, GPS calibration and the prediction log are kept.
STATE_VERSION = 2
EPISODE_RETENTION_HALF_LIVES = 10.0
MERGE_TOLERANCE = timedelta(minutes=10)  # same departure in stored and recorder data


def shift_label(local_hour: int) -> str:
    """Descriptive label of a departure hour."""
    if 4 <= local_hour < 10:
        return "morning"
    if 10 <= local_hour < 16:
        return "afternoon"
    if 16 <= local_hour < 22:
        return "evening"
    return "night"


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _episode_to_dict(ep: AbsenceEpisode) -> dict[str, Any]:
    return {
        "left_at": ep.left_at.isoformat(),
        "returned_at": ep.returned_at.isoformat(),
        "weekday": ep.weekday,
        "is_holiday": ep.is_holiday,
    }


def _episode_from_dict(data: dict[str, Any]) -> AbsenceEpisode:
    return AbsenceEpisode(
        left_at=datetime.fromisoformat(data["left_at"]),
        returned_at=datetime.fromisoformat(data["returned_at"]),
        weekday=int(data["weekday"]),
        is_holiday=bool(data.get("is_holiday", False)),
    )


class KernelBayesReturnModel(ReturnModel):
    """Per-person return predictor (see docs/PLAN.md section 2)."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._episodes: list[AbsenceEpisode] = []
        self._trips: list[AbsenceEpisode] = []
        self._is_home: bool | None = None
        self._away_since: datetime | None = None
        self._away_is_holiday = False
        # Arrival whose scoring waits MIN_HOME_STAY_MINUTES (a quick new departure
        # continues the absence): (arrived_at, left_at, away_is_holiday).
        self._arrival: tuple[datetime, datetime, bool] | None = None
        self.calibrator = LRCalibrator()
        self._log = PredictionLog()
        self._calib_pending: list[LogEntry] = []
        self._segments = CalibrationSegments()
        self._tod_density: tuple[Any, Any] | None = None  # arrival() cache, not persisted
        # Bumped whenever the learnt state is discarded (reset / replacing history).
        self._generation = 0
        # Prior keys cache; set to None whenever episodes/trips change.
        self._key_cache: tuple[list[EpisodeKey], list[EpisodeKey]] | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        generation = self._generation
        self.__init__(self.config)  # type: ignore[misc]
        self._generation = generation + 1

    @property
    def generation(self) -> int:
        """Counter of discarded learnt states (reset / ``load_history(replace=True)``)."""
        return self._generation

    def load_history(
        self,
        transitions: Iterable[PresenceTransition],
        fixes: Iterable[PositionFix] = (),
        holidays: Iterable[datetime | date] = (),
        replace: bool = False,
    ) -> None:
        built = build_episodes(transitions, self.config, holidays)
        if replace:
            self._generation += 1
            self._episodes, self._trips = built.episodes, built.trips
            self._is_home, self._away_since = built.is_home, built.away_since
            self._away_is_holiday = False
            self._segments = CalibrationSegments()
        else:
            self._episodes, self._trips = self._merge(built.episodes, built.trips)
            if built.is_home is not None:
                if built.away_since != self._away_since:
                    self._away_is_holiday = False
                self._is_home, self._away_since = built.is_home, built.away_since
        self._key_cache = None
        fix_list = sorted(fixes, key=lambda f: to_utc(f.ts))
        if fix_list and (replace or not self.calibrator.has_observations()):
            self._warm_calibration(fix_list)

    def _merge(
        self, episodes: list[AbsenceEpisode], trips: list[AbsenceEpisode]
    ) -> tuple[list[AbsenceEpisode], list[AbsenceEpisode]]:
        """Union of stored and rebuilt episodes/trips; the rebuilt version wins."""
        rebuilt = sorted(e.left_at for e in (*episodes, *trips))

        def duplicated(ep: AbsenceEpisode) -> bool:
            i = bisect.bisect_left(rebuilt, ep.left_at - MERGE_TOLERANCE)
            return i < len(rebuilt) and rebuilt[i] <= ep.left_at + MERGE_TOLERANCE

        def union(stored: list[AbsenceEpisode], new: list[AbsenceEpisode]) -> list[AbsenceEpisode]:
            kept = [e for e in stored if not duplicated(e)]
            return sorted((*kept, *new), key=lambda e: e.left_at)

        return union(self._episodes, episodes), union(self._trips, trips)

    def _warm_calibration(self, fixes: list[PositionFix]) -> None:
        """Replay historical fixes per absence to calibrate the GPS LR from scratch."""
        cfg = self.config
        self.calibrator.reset()
        times = [to_utc(f.ts) for f in fixes]
        # fixes the regression may use, incl. the sparse-reporting fallback
        window = timedelta(minutes=EXTENDED_WINDOW_FACTOR * cfg.regression_window_minutes)
        idx = 0
        segments = CalibrationSegments()
        for ep in sorted((*self._episodes, *self._trips), key=lambda e: e.left_at):
            while idx < len(times) and times[idx] < ep.left_at:
                idx += 1
            start = idx
            key = ep.left_at.isoformat()
            j = idx
            while j < len(times) and times[j] < ep.returned_at:
                lo = start
                while lo < j and times[lo] < times[j] - window:
                    lo += 1
                start = lo
                ev = compute_evidence(
                    fixes[lo : j + 1],
                    times[j],
                    cfg.regression_window_minutes,
                    cfg.max_fix_accuracy_m,
                )
                cls = ev.approach_class
                if cls is not None and segments.observe(times[j], cls, key, cfg.horizon_minutes):
                    returned = minutes_between(times[j], ep.returned_at) <= cfg.horizon_minutes
                    self.calibrator.update(cls, returned)
                j += 1
            idx = j

    def observe_transition(
        self, transition: PresenceTransition, is_holiday: bool = False
    ) -> AbsenceEpisode | None:
        ts = to_utc(transition.ts)
        if not transition.is_home:
            if self._is_home is False and self._away_since is not None:
                return None  # duplicate away report
            self._is_home = False
            arrival = self._arrival
            if arrival is not None and minutes_between(arrival[0], ts) < MIN_HOME_STAY_MINUTES:
                # Brief home blip (drive-by, zone-edge jitter): the absence goes on.
                self._arrival = None
                arrived, self._away_since, self._away_is_holiday = arrival
                self._forget_episode(self._away_since, arrived)
                return None
            self._settle_arrival()
            self._away_since = ts
            self._away_is_holiday = is_holiday
            return None

        was_away = self._is_home is False
        self._is_home = True
        left = self._away_since
        self._away_since = None
        if not was_away or left is None:
            if self._arrival is None:
                self._settle(arrival=ts)
            return None
        self._settle_arrival()
        # Scored once the stay lasted MIN_HOME_STAY_MINUTES (see resolve).
        self._arrival = (ts, left, self._away_is_holiday)
        ep = make_episode(left, ts, self.config.timezone, self._away_is_holiday or is_holiday)
        kind = classify_episode(ep, self.config)
        if kind == KIND_EPISODE:
            self._episodes.append(ep)
        elif kind == KIND_TRIP:
            self._trips.append(ep)
        else:
            return None
        self._prune(ts)
        return ep

    def _forget_episode(self, left_at: datetime, returned_at: datetime) -> None:
        """Drop the episode/trip closed by an arrival that turned out to be a blip."""

        def keep(e: AbsenceEpisode) -> bool:
            return not (e.left_at == left_at and e.returned_at == returned_at)

        self._episodes = [e for e in self._episodes if keep(e)]
        self._trips = [e for e in self._trips if keep(e)]
        self._key_cache = None

    def _settle_arrival(self) -> int:
        """Score the pending predictions against a confirmed arrival."""
        if self._arrival is None:
            return 0
        arrived = self._arrival[0]
        self._arrival = None
        return self._settle(arrival=arrived)

    @property
    def away_since(self) -> datetime | None:
        """Start of the current absence (UTC); a brief home blip does not restart it."""
        return self._away_since if self._is_home is False else None

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(days=EPISODE_RETENTION_HALF_LIVES * self.config.half_life_days)
        self._episodes = [e for e in self._episodes if e.left_at >= cutoff]
        self._trips = [e for e in self._trips if e.left_at >= cutoff]
        self._key_cache = None

    # ----------------------------------------------------------------- prediction
    def _keys(self) -> tuple[list[EpisodeKey], list[EpisodeKey]]:
        """Cached prior keys; rebuilt when the episode lists change (cache only)."""
        if self._key_cache is None:
            self._key_cache = (
                episode_keys(self._episodes, self.config.timezone),
                episode_keys(self._trips, self.config.timezone),
            )
        return self._key_cache

    def _departure_holiday(self, model_input: ModelInput, left_at: datetime | None) -> bool:
        """Holiday flag of the departure day (the habits are keyed on the departure, like
        the weekday); today's flag only when the model did not see the departure."""
        if (
            left_at is not None
            and self._away_since is not None
            and abs(minutes_between(self._away_since, left_at)) < 1.0
        ):
            return self._away_is_holiday
        return model_input.is_holiday

    def predict(self, model_input: ModelInput) -> Prediction:
        cfg = self.config
        now = to_utc(model_input.now)
        if model_input.is_home:
            return Prediction(
                probability=1.0,
                prior_probability=None,
                likelihood_ratio=1.0,
                eta=None,
                eta_habit=None,
                eta_position=None,
                approach_speed_kmh=None,
                approach_class=None,
                distance_m=None,
                is_home=True,
                reason={
                    "at_home": True,
                    "fusion": FUSION_ALIAS[self._serving_model()],
                    "active_model": self._serving_model(),
                },
            )

        left_at = to_utc(model_input.left_at) if model_input.left_at else self._away_since
        minutes_away = max(minutes_between(left_at, now), 0.0) if left_at else 0.0
        ref = to_local(left_at if left_at else now, cfg.timezone)
        ep_keys, trip_keys = self._keys()
        prior = estimate_prior(
            ep_keys,
            trip_keys,
            now=now,
            left_at=left_at,
            minutes_away=minutes_away,
            horizon_minutes=cfg.horizon_minutes,
            weekday=ref.weekday(),
            is_holiday=self._departure_holiday(model_input, left_at),
            half_life_days=cfg.half_life_days,
            tz=cfg.timezone,
        )
        ev = compute_evidence(
            model_input.fixes, now, cfg.regression_window_minutes, cfg.max_fix_accuracy_m
        )
        lr = self.calibrator.likelihood_ratio(ev.approach_class)
        p_bayes = fuse_probability(prior.probability, lr)
        active = self._serving_model()

        pos_min = position_eta_minutes(ev.distance_m, ev.speed_kmh, cfg.horizon_minutes)
        # Physical bound, whatever the horizon: too far to be home before travel_min.
        # A stationary phone reports rarely, so fall back to the last known fix and its age.
        feas_distance, feas_age = ev.distance_m, 0.0
        if feas_distance is None and model_input.last_fix is not None:
            feas_distance = model_input.last_fix.distance_m
            feas_age = max(minutes_between(to_utc(model_input.last_fix.ts), now), 0.0)
        travel_min = min_travel_minutes(feas_distance)
        if travel_min is not None:
            travel_min = max(travel_min - feas_age, 0.0)

        # Single engine: habit prior x calibrated GPS likelihood ratio.
        probability = p_bayes

        habit_min = (
            prior.eta_duration_minutes - minutes_away
            if prior.eta_duration_minutes is not None
            else None
        )
        habit_iqr = prior.eta_iqr_minutes
        habit = (
            self._arrival_model(now, minutes_away, prior)
            if habit_min is None and left_at is not None
            else None
        )
        if habit is not None:
            # Past the usual time (or beyond 12 h): the arrival distribution's median
            # instead of no habit ETA at all.
            dist = with_evidence(
                habit, prior=prior.probability, p_bayes=p_bayes, eta_minutes=pos_min,
                min_minutes=travel_min,
            )  # fmt: skip
            q50, q25, q75 = dist.quantile(0.5), dist.quantile(0.25), dist.quantile(0.75)
            if q50 is not None:
                habit_min = q50
                habit_iqr = q75 - q25 if q25 is not None and q75 is not None else None
        eta_min, eta_source = combine_eta(habit_min, habit_iqr, pos_min)
        probability, eta_min, feasible = apply_feasibility(
            probability, eta_min, feas_distance, cfg.horizon_minutes, feas_age
        )

        def at(minutes: float | None) -> datetime | None:
            return now + timedelta(minutes=minutes) if minutes is not None else None

        reason: dict[str, Any] = {
            "left_at_local": to_local(left_at, cfg.timezone).strftime("%H:%M") if left_at else None,
            "shift_label": shift_label(to_local(left_at, cfg.timezone).hour) if left_at else None,
            "minutes_away": round(minutes_away, 1),
            "prior": round(prior.probability, 4),
            "approach_class": ev.approach_class,
            "approach_speed_kmh": round(ev.speed_kmh, 1) if ev.speed_kmh is not None else None,
            "distance_km": round(ev.distance_m / 1000.0, 2) if ev.distance_m is not None else None,
            "lr": round(lr, 3),
            "eta_source": eta_source,
            "fusion": FUSION_ALIAS[active],
            "active_model": active,
            "probability_bayes": round(p_bayes, 4),
            "feasible": feasible,
            "min_travel_minutes": round(travel_min, 1) if travel_min is not None else None,
        }
        return Prediction(
            probability=probability,
            prior_probability=prior.probability,
            likelihood_ratio=lr,
            eta=at(eta_min),
            eta_habit=at(habit_min),
            eta_position=at(pos_min),
            approach_speed_kmh=ev.speed_kmh,
            approach_class=ev.approach_class,
            distance_m=ev.distance_m,
            is_home=False,
            prior_effective_samples=prior.effective_samples,
            reason=reason,
        )

    def arrival(self, model_input: ModelInput, prediction: Prediction) -> Any:
        """Arrival distribution while away (``arrival.py``); None at home or without habits.

        Habits from the kernel prior's weighted durations; the GPS evidence of
        ``prediction`` (habit x GPS probability, position ETA, physical minimum travel
        time) moves and truncates the mass.
        """
        if model_input.is_home or prediction.is_home:
            return None
        cfg = self.config
        now = to_utc(model_input.now)
        left_at = to_utc(model_input.left_at) if model_input.left_at else self._away_since
        minutes_away = max(minutes_between(left_at, now), 0.0) if left_at else 0.0
        ref = to_local(left_at if left_at else now, cfg.timezone)
        ep_keys, trip_keys = self._keys()
        prior = estimate_prior(
            ep_keys,
            trip_keys,
            now=now,
            left_at=left_at,
            minutes_away=minutes_away,
            horizon_minutes=cfg.horizon_minutes,
            weekday=ref.weekday(),
            is_holiday=self._departure_holiday(model_input, left_at),
            half_life_days=cfg.half_life_days,
            tz=cfg.timezone,
        )
        habit = self._arrival_model(now, minutes_away, prior)
        if habit is None:
            return None
        reason = prediction.reason or {}
        eta_pos = (
            minutes_between(now, to_utc(prediction.eta_position))
            if prediction.eta_position is not None
            else None
        )
        return with_evidence(
            habit,
            prior=prediction.prior_probability,
            p_bayes=reason.get("probability_bayes"),
            eta_minutes=eta_pos,
            min_minutes=reason.get("min_travel_minutes"),
        )

    def _arrival_model(
        self, now: datetime, minutes_away: float, prior: PriorResult
    ) -> ArrivalDistribution | None:
        """Habit-only arrival distribution (``arrival.habit_arrival``), cached density."""
        cfg = self.config
        ep_keys, trip_keys = self._keys()
        # The arrival time-of-day density only moves with the recency weights: cache it
        # per hour and per set of absences.
        cache_key = (len(ep_keys), len(trip_keys), self._generation, int(now.timestamp() // 3600))
        if self._tod_density is None or self._tod_density[0] != cache_key:
            pool = (*ep_keys, *trip_keys)
            density = arrival_tod_density(pool, now, cfg.half_life_days) if pool else None
            self._tod_density = (cache_key, density)
        return habit_arrival(
            ep_keys,
            trip_keys,
            prior.durations,
            now=now,
            minutes_away=minutes_away,
            half_life_days=cfg.half_life_days,
            tz=cfg.timezone,
            density=self._tod_density[1],
        )

    def _serving_model(self) -> str:
        """Always the kernel prior x GPS evidence (single engine since state v2)."""
        return KERNEL_BAYES

    # ------------------------------------------------------------ logging/scoring
    def log_prediction(self, model_input: ModelInput, prediction: Prediction) -> bool:
        if model_input.is_home or prediction.is_home:
            return False
        now = to_utc(model_input.now)
        horizon_end = now + timedelta(minutes=self.config.horizon_minutes)
        left_at = to_utc(model_input.left_at) if model_input.left_at else self._away_since
        minutes_away = max(minutes_between(left_at, now), 0.0) if left_at else 0.0
        cls = prediction.approach_class
        if self._segments.observe(now, cls, _iso(left_at), self.config.horizon_minutes):
            # Calibration observation: first time this class is seen in this
            # absence (still/away) or approach segment (approach classes).
            self._calib_pending.append(
                compact_entry(
                    LogEntry(
                        now, prediction.probability, prediction.prior_probability, cls, horizon_end
                    )
                )
            )
        entry = LogEntry(
            now,
            prediction.probability,
            prediction.prior_probability,
            cls,
            horizon_end,
        )
        if self._log.last_logged is not None and (
            minutes_between(self._log.last_logged, now)
            < self.config.prediction_log_interval_minutes
        ):
            return False
        ep_keys, trip_keys = self._keys()
        entry.base = duration_baseline(
            ep_keys,
            trip_keys,
            now=now,
            minutes_away=minutes_away,
            horizon_minutes=self.config.horizon_minutes,
            half_life_days=self.config.half_life_days,
        )
        return self._log.add(compact_entry(entry), self.config.prediction_log_interval_minutes)

    def _settle(self, *, arrival: datetime | None = None, now: datetime | None = None) -> int:
        resolved, self._log.pending = resolve_entries(self._log.pending, arrival=arrival, now=now)
        done_cal, self._calib_pending = resolve_entries(
            self._calib_pending, arrival=arrival, now=now
        )
        for entry in done_cal:
            if entry.approach_class is not None:
                self.calibrator.update(entry.approach_class, entry.outcome == 1)
        ref = arrival or now
        if ref is None or not resolved:
            return 0
        self._log.settle(resolved, ref)
        return len(resolved)

    def resolve(self, now: datetime, is_home: bool) -> int:
        now = to_utc(now)
        if is_home:
            count = 0
            if self._arrival is not None:
                if minutes_between(self._arrival[0], now) < MIN_HOME_STAY_MINUTES:
                    return 0  # a quick new departure would continue the absence
                count = self._settle_arrival()
            # Drop orphans of unknown outcome.
            self._log.pending = [e for e in self._log.pending if e.horizon_end >= now]
            self._calib_pending = [e for e in self._calib_pending if e.horizon_end >= now]
            return count
        return self._settle(now=now)

    def maturity(self, now: datetime) -> MaturityReport:
        return maturity_report(
            self._episodes,
            self._log.resolved,
            to_utc(now),
            self.config.half_life_days,
            fusion=FUSION_ALIAS[self._serving_model()],
            active_model=self._serving_model(),
        )

    def episodes(self) -> list[AbsenceEpisode]:
        return list(self._episodes)

    def trips(self) -> list[AbsenceEpisode]:
        """Stored long absences (> trip_absence_hours)."""
        return list(self._trips)

    # -------------------------------------------------------------- persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            # Outcome definition of the logged predictions and of the learnt fusion.
            "horizon_minutes": self.config.horizon_minutes,
            "episodes": [_episode_to_dict(e) for e in self._episodes],
            "trips": [_episode_to_dict(e) for e in self._trips],
            "is_home": self._is_home,
            "away_since": _iso(self._away_since),
            "away_is_holiday": self._away_is_holiday,
            "arrival": (
                [self._arrival[0].isoformat(), self._arrival[1].isoformat(), self._arrival[2]]
                if self._arrival is not None
                else None
            ),
            "calibrator": self.calibrator.to_dict(),
            "log": self._log.to_dict(),
            "calib_pending": encode_entries(self._calib_pending),
            "segments": self._segments.to_dict(),
            "generation": self._generation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: ModelConfig) -> KernelBayesReturnModel:
        model = cls(config)
        if not data:
            return model
        model._episodes = [_episode_from_dict(e) for e in data.get("episodes", [])]
        model._trips = [_episode_from_dict(e) for e in data.get("trips", [])]
        model._is_home = data.get("is_home")
        model._away_since = _parse(data.get("away_since"))
        model._away_is_holiday = bool(data.get("away_is_holiday", False))
        if arrival := data.get("arrival"):
            arrived, left = datetime.fromisoformat(arrival[0]), datetime.fromisoformat(arrival[1])
            model._arrival = (arrived, left, bool(arrival[2]))
        model.calibrator = LRCalibrator.from_dict(data.get("calibrator", {}))
        # v1 blocks of the retired learned engines (stacker, selection, challenger,
        # last_training) are ignored: the kernel engine serves.
        model._log = PredictionLog.from_dict(data.get("log", {}))
        model._calib_pending = decode_entries(data.get("calib_pending"))
        model._segments = CalibrationSegments.from_dict(data.get("segments") or {})
        model._generation = int(data.get("generation", 0))
        stored_horizon = data.get("horizon_minutes")  # missing: older snapshot, keep
        if stored_horizon is not None and float(stored_horizon) != config.horizon_minutes:
            model.discard_scoring()
        return model

    def discard_scoring(self) -> None:
        """Forget everything scored against the horizon; keep habits and GPS calibration.

        Called when the horizon changed: logged outcomes describe another target.
        Episodes and trips are horizon-independent and stay.
        """
        self._log = PredictionLog()
        self._calib_pending = []
        self._segments = CalibrationSegments()
        self._generation += 1
