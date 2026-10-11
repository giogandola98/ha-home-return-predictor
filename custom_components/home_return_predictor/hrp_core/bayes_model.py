"""KernelBayesReturnModel: kernel survival prior x calibrated GPS likelihood ratio."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Any

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
    build_episodes,
    classify_episode,
    make_episode,
    minutes_between,
    to_local,
    to_utc,
)
from .evidence import LRCalibrator, compute_evidence
from .fusion import combine_eta, fuse_probability, position_eta_minutes
from .maturity import LogEntry, PredictionLog, maturity_report, resolve_entries
from .prior import EpisodeKey, episode_keys, estimate_prior
from .stacker import Stacker, build_features

STATE_VERSION = 1
EPISODE_RETENTION_HALF_LIVES = 10.0
STACKER_REFIT_EVERY = 150  # newly resolved entries between stacker refits


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
        self.calibrator = LRCalibrator()
        self.stacker = Stacker()
        self._log = PredictionLog()
        self._calib_pending: list[LogEntry] = []
        self._seen_key: str | None = None
        self._seen_classes: list[str] = []
        self._since_refit = 0
        # Prior keys cache; set to None whenever episodes/trips change.
        self._key_cache: tuple[list[EpisodeKey], list[EpisodeKey]] | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        self.__init__(self.config)  # type: ignore[misc]

    def load_history(
        self,
        transitions: Iterable[PresenceTransition],
        fixes: Iterable[PositionFix] = (),
        holidays: Iterable[datetime | date] = (),
    ) -> None:
        built = build_episodes(transitions, self.config, holidays)
        self._episodes, self._trips = built.episodes, built.trips
        self._key_cache = None
        self._is_home, self._away_since = built.is_home, built.away_since
        self._away_is_holiday = False
        self._seen_key, self._seen_classes = None, []
        fix_list = sorted(fixes, key=lambda f: to_utc(f.ts))
        if fix_list:
            self._warm_calibration(fix_list)

    def _warm_calibration(self, fixes: list[PositionFix]) -> None:
        """Replay historical fixes per absence to calibrate the GPS LR from scratch."""
        cfg = self.config
        self.calibrator.reset()
        times = [to_utc(f.ts) for f in fixes]
        window = timedelta(minutes=cfg.regression_window_minutes)
        idx = 0
        for ep in sorted((*self._episodes, *self._trips), key=lambda e: e.left_at):
            while idx < len(times) and times[idx] < ep.left_at:
                idx += 1
            start = idx
            seen: set[str] = set()
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
                if cls is not None and cls not in seen:
                    seen.add(cls)
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
            self._away_since = ts
            self._away_is_holiday = is_holiday
            self._seen_key, self._seen_classes = None, []
            return None

        was_away = self._is_home is False
        self._is_home = True
        self._settle(arrival=ts)
        left = self._away_since
        self._away_since = None
        if not was_away or left is None:
            return None
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
                reason={"at_home": True, "fusion": "bayes"},
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
            is_holiday=model_input.is_holiday,
            half_life_days=cfg.half_life_days,
            tz=cfg.timezone,
        )
        ev = compute_evidence(
            model_input.fixes, now, cfg.regression_window_minutes, cfg.max_fix_accuracy_m
        )
        lr = self.calibrator.likelihood_ratio(ev.approach_class)
        features = self._features(now, prior.probability, lr, minutes_away, ev.distance_m)
        p_bayes = fuse_probability(prior.probability, lr)
        fusion = "learned" if self.stacker.active else "bayes"
        probability = self.stacker.predict(features) if self.stacker.active else p_bayes

        habit_min = (
            prior.eta_duration_minutes - minutes_away
            if prior.eta_duration_minutes is not None
            else None
        )
        pos_min = position_eta_minutes(ev.distance_m, ev.speed_kmh, cfg.horizon_minutes)
        eta_min, eta_source = combine_eta(habit_min, prior.eta_iqr_minutes, pos_min)

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
            "fusion": fusion,
            "probability_bayes": round(p_bayes, 4),
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

    def _features(
        self,
        now: datetime,
        prior: float,
        lr: float,
        minutes_away: float,
        distance_m: float | None,
    ) -> list[float]:
        local_now = to_local(now, self.config.timezone)
        return build_features(
            prior,
            lr,
            minutes_away,
            self.config.horizon_minutes,
            distance_m,
            distance_m is not None,
            local_now.hour + local_now.minute / 60.0,
        )

    # ------------------------------------------------------------ logging/scoring
    def log_prediction(self, model_input: ModelInput, prediction: Prediction) -> bool:
        if model_input.is_home or prediction.is_home:
            return False
        now = to_utc(model_input.now)
        horizon_end = now + timedelta(minutes=self.config.horizon_minutes)
        left_at = to_utc(model_input.left_at) if model_input.left_at else self._away_since
        features: list[float] = []
        if prediction.prior_probability is not None:
            minutes_away = max(minutes_between(left_at, now), 0.0) if left_at else 0.0
            features = self._features(
                now,
                prediction.prior_probability,
                prediction.likelihood_ratio,
                minutes_away,
                prediction.distance_m,
            )
        key = _iso(left_at)
        if key != self._seen_key:
            self._seen_key, self._seen_classes = key, []
        cls = prediction.approach_class
        if cls is not None and cls not in self._seen_classes:
            # Calibration observation: first time this class is seen in this absence.
            self._seen_classes.append(cls)
            self._calib_pending.append(
                LogEntry(
                    now, prediction.probability, prediction.prior_probability, cls, horizon_end
                )
            )
        entry = LogEntry(
            now,
            prediction.probability,
            prediction.prior_probability,
            cls,
            horizon_end,
            features=features,
        )
        return self._log.add(entry, self.config.prediction_log_interval_minutes)

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
        self._since_refit += len(resolved)
        if self._since_refit >= STACKER_REFIT_EVERY:
            self._since_refit = 0
            self.stacker.fit(self._log.resolved, ref, self.config.half_life_days)
        return len(resolved)

    def resolve(self, now: datetime, is_home: bool) -> int:
        now = to_utc(now)
        if is_home:
            # Arrival is resolved by observe_transition; drop orphans of unknown outcome.
            self._log.pending = [e for e in self._log.pending if e.horizon_end >= now]
            self._calib_pending = [e for e in self._calib_pending if e.horizon_end >= now]
            return 0
        return self._settle(now=now)

    def maturity(self, now: datetime) -> MaturityReport:
        return maturity_report(
            self._episodes,
            self._log.resolved,
            to_utc(now),
            self.config.half_life_days,
            fusion="learned" if self.stacker.active else "bayes",
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
            "episodes": [_episode_to_dict(e) for e in self._episodes],
            "trips": [_episode_to_dict(e) for e in self._trips],
            "is_home": self._is_home,
            "away_since": _iso(self._away_since),
            "away_is_holiday": self._away_is_holiday,
            "calibrator": self.calibrator.to_dict(),
            "stacker": self.stacker.to_dict(),
            "log": self._log.to_dict(),
            "calib_pending": [e.to_dict() for e in self._calib_pending],
            "seen_key": self._seen_key,
            "seen_classes": list(self._seen_classes),
            "since_refit": self._since_refit,
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
        model.calibrator = LRCalibrator.from_dict(data.get("calibrator", {}))
        model.stacker = Stacker.from_dict(data.get("stacker", {}))
        model._log = PredictionLog.from_dict(data.get("log", {}))
        model._calib_pending = [LogEntry.from_dict(e) for e in data.get("calib_pending", [])]
        model._seen_key = data.get("seen_key")
        model._seen_classes = list(data.get("seen_classes", []))
        model._since_refit = int(data.get("since_refit", 0))
        return model
