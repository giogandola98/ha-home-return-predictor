"""KernelBayesReturnModel: kernel survival prior x calibrated GPS likelihood ratio."""

from __future__ import annotations

import bisect
import time
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
from .challenger import (
    N_CHALLENGER_FEATURES,
    GBMChallenger,
    bayes_from_challenger_features,
    build_challenger_features,
    numpy_available,
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
from .evidence import CalibrationSegments, LRCalibrator, compute_evidence
from .fusion import combine_eta, fuse_probability, position_eta_minutes
from .maturity import (
    LogEntry,
    PredictionLog,
    compact_entry,
    decode_entries,
    encode_entries,
    maturity_report,
    resolve_entries,
)
from .prior import EpisodeKey, duration_baseline, episode_keys, estimate_prior, is_rest_day
from .selection import (
    FUSION_ALIAS,
    GBM,
    KERNEL_BAYES,
    LEARNED_FUSION,
    RULES,
    Selector,
    chronological_split,
    recency_weights,
    weighted_brier,
)
from .stacker import (
    ACTIVATION_MARGIN,
    N_FEATURES,
    Stacker,
    apply_weights,
    build_features,
    fit_weights,
)

STATE_VERSION = 1
EPISODE_RETENTION_HALF_LIVES = 10.0
STACKER_REFIT_EVERY = 150  # newly resolved entries between stacker refits
CHALLENGER_RETRAIN_HOURS = 6.0
CHALLENGER_RETRAIN_ENTRIES = 150
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
        self.calibrator = LRCalibrator()
        self.stacker = Stacker()
        self._log = PredictionLog()
        self._calib_pending: list[LogEntry] = []
        self._segments = CalibrationSegments()
        self._since_refit = 0
        self.selector = Selector()
        # Challenger: the enable flag, the selection state and the exported trees
        # (when small enough) are persisted.
        self._challenger_enabled = False
        self._challenger = GBMChallenger()
        self._last_training: datetime | None = None
        self._resolved_total = 0  # resolved entries seen in this process
        self._resolved_at_training = 0
        # Bumped whenever the learnt state is discarded (reset / replacing history);
        # a challenger fit started on an older generation is stale and discarded.
        self._generation = 0
        # Prior keys cache; set to None whenever episodes/trips change.
        self._key_cache: tuple[list[EpisodeKey], list[EpisodeKey]] | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        enabled = self._challenger_enabled
        generation = self._generation
        self.__init__(self.config)  # type: ignore[misc]
        self._challenger_enabled = enabled
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
        window = timedelta(minutes=cfg.regression_window_minutes)
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
            self._away_since = ts
            self._away_is_holiday = is_holiday
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
            is_holiday=model_input.is_holiday,
            half_life_days=cfg.half_life_days,
            tz=cfg.timezone,
        )
        ev = compute_evidence(
            model_input.fixes, now, cfg.regression_window_minutes, cfg.max_fix_accuracy_m
        )
        lr = self.calibrator.likelihood_ratio(ev.approach_class)
        p_bayes = fuse_probability(prior.probability, lr)
        active = self._serving_model()
        if active == LEARNED_FUSION:
            features = self._features(now, prior.probability, lr, minutes_away, ev.distance_m)
            probability = self.stacker.predict(features)
        elif active == GBM:
            row = self._challenger_row(
                now,
                left_at,
                minutes_away,
                prior.probability,
                lr,
                ev.distance_m,
                ev.speed_kmh,
                model_input.is_holiday,
            )
            probability = self._challenger.predict_proba(row)
        else:
            probability = p_bayes

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
            "fusion": FUSION_ALIAS[active],
            "active_model": active,
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

    def _challenger_row(
        self,
        now: datetime,
        left_at: datetime | None,
        minutes_away: float,
        prior: float,
        lr: float,
        distance_m: float | None,
        speed_kmh: float | None,
        is_holiday: bool,
    ) -> list[float]:
        tz = self.config.timezone
        local_now = to_local(now, tz)
        local_dep = to_local(left_at, tz) if left_at is not None else None
        ref = local_dep or local_now
        return build_challenger_features(
            prior=prior,
            likelihood_ratio=lr,
            minutes_away=minutes_away,
            horizon_minutes=self.config.horizon_minutes,
            departure_hour=(
                local_dep.hour + local_dep.minute / 60.0 if local_dep is not None else None
            ),
            local_hour=local_now.hour + local_now.minute / 60.0,
            rest_day=is_rest_day(ref.weekday(), is_holiday),
            distance_m=distance_m,
            speed_kmh=speed_kmh,
            has_gps=distance_m is not None,
            p_bayes=fuse_probability(prior, lr),
        )

    def _serving_model(self) -> str:
        """Champion, or the next-best candidate while the champion cannot serve."""
        champ = self.selector.champion
        if champ != GBM:
            return champ
        if self._challenger_enabled and self._challenger.trained:
            return GBM
        brier = self.selector.brier
        learned, bayes = brier.get(LEARNED_FUSION), brier.get(KERNEL_BAYES)
        if learned is not None and bayes is not None and learned < bayes:
            return LEARNED_FUSION
        return KERNEL_BAYES

    # ------------------------------------------------------------ logging/scoring
    def log_prediction(self, model_input: ModelInput, prediction: Prediction) -> bool:
        if model_input.is_home or prediction.is_home:
            return False
        now = to_utc(model_input.now)
        horizon_end = now + timedelta(minutes=self.config.horizon_minutes)
        left_at = to_utc(model_input.left_at) if model_input.left_at else self._away_since
        features: list[float] = []
        row: list[float] = []
        minutes_away = max(minutes_between(left_at, now), 0.0) if left_at else 0.0
        if prediction.prior_probability is not None:
            features = self._features(
                now,
                prediction.prior_probability,
                prediction.likelihood_ratio,
                minutes_away,
                prediction.distance_m,
            )
            row = self._challenger_row(
                now,
                left_at,
                minutes_away,
                prediction.prior_probability,
                prediction.likelihood_ratio,
                prediction.distance_m,
                prediction.approach_speed_kmh,
                model_input.is_holiday,
            )
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
            features=features,
            x=row,
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
        self._since_refit += len(resolved)
        self._resolved_total += len(resolved)
        return len(resolved)

    def _maybe_refit(self, now: datetime) -> None:
        """Refit the stacker after enough new resolved entries (called by resolve only).

        In challenger mode ``train_challenger`` evaluates and installs the learned
        fusion weights, so the stacker is not refit here: the weights served are
        always the ones the selector evaluated.
        """
        if self._since_refit < STACKER_REFIT_EVERY:
            return
        self._since_refit = 0
        if self._challenger_mode():
            return
        self.stacker.fit(self._log.resolved, now, self.config.half_life_days)
        self.selector.update(
            {
                KERNEL_BAYES: self.stacker.holdout_brier_bayes,
                LEARNED_FUSION: self.stacker.holdout_brier_learned,
            },
            self.stacker.trained_on,
        )

    # ----------------------------------------------------------------- challenger
    def _challenger_rows(self) -> list[LogEntry]:
        return [
            e
            for e in self._log.resolved
            if e.outcome is not None
            and len(e.x) == N_CHALLENGER_FEATURES
            and len(e.features) == N_FEATURES
        ]

    def _challenger_mode(self) -> bool:
        """True when train_challenger (not the stacker refit) drives selection."""
        return (
            self._challenger_enabled
            and numpy_available()
            and len(self._challenger_rows()) >= RULES[GBM].min_entries
        )

    def enable_challenger(self, enabled: bool) -> None:
        """Switch the optional gradient-boosting challenger on or off."""
        self._challenger_enabled = bool(enabled)
        if not enabled:
            self._challenger = GBMChallenger()
            self._last_training = None

    def challenger_due(self, now: datetime) -> bool:
        """True when ``train_challenger`` should run (cheap; call from the event loop)."""
        if not self._challenger_enabled or not numpy_available():
            return False
        if self._last_training is None:
            return len(self._challenger_rows()) >= RULES[GBM].min_entries
        elapsed_h = minutes_between(self._last_training, to_utc(now)) / 60.0
        new = self._resolved_total - self._resolved_at_training
        return elapsed_h >= CHALLENGER_RETRAIN_HOURS or new >= CHALLENGER_RETRAIN_ENTRIES

    def challenger_snapshot(self) -> Any:
        """Cheap copy of what ``fit_challenger`` needs (call under the person lock).

        Only list copies: the logged feature rows are never mutated after logging,
        so the snapshot can be fit in another thread while the model keeps running.
        """
        rows = self._challenger_rows()
        return {
            "enabled": self._challenger_enabled,
            "generation": self._generation,
            "half_life_days": self.config.half_life_days,
            "resolved_total": self._resolved_total,
            "ts": [e.ts for e in rows],
            "x": [e.x for e in rows],
            "f": [e.features for e in rows],
            "y": [int(e.outcome or 0) for e in rows],
        }

    @staticmethod
    def fit_challenger(snapshot: Any, now: datetime) -> Any:
        """Heavy part of the training: fit and evaluate all candidates (no model state).

        Holdout = last 20 % of the resolved entries (chronological); the stacker and
        the GBM are fit on the first 80 % for the evaluation, then the GBM is refit
        on everything for serving. The learned-fusion weights evaluated here are
        the ones installed (and served) by ``apply_challenger``.
        """
        started = time.perf_counter()
        now = to_utc(now)
        n = len(snapshot["y"])
        result: dict[str, Any] = {
            "status": "ok",
            "rows": n,
            "resolved_total": snapshot["resolved_total"],
            "generation": snapshot.get("generation", 0),
            "brier": {},
        }

        def done(status: str) -> dict[str, Any]:
            result["status"] = status
            result["seconds"] = round(time.perf_counter() - started, 3)
            return result

        if not snapshot["enabled"]:
            return done("disabled")
        if not numpy_available():
            return done("numpy_unavailable")
        if n < RULES[GBM].min_entries:
            return done("insufficient_data")

        hl = snapshot["half_life_days"]
        ts, xs, fs, ys = snapshot["ts"], snapshot["x"], snapshot["f"], snapshot["y"]
        idx_train, idx_hold = chronological_split(range(n))
        w_all = recency_weights(ts, now, hl)
        w_train = [w_all[i] for i in idx_train]
        w_hold = [w_all[i] for i in idx_hold]
        y_train = [ys[i] for i in idx_train]
        y_hold = [ys[i] for i in idx_hold]

        weights = fit_weights([fs[i] for i in idx_train], y_train, w_train)
        candidate = GBMChallenger()
        candidate.fit([xs[i] for i in idx_train], y_train, w_train)
        result["brier"] = {
            KERNEL_BAYES: weighted_brier(
                [bayes_from_challenger_features(xs[i]) for i in idx_hold], y_hold, w_hold
            ),
            LEARNED_FUSION: weighted_brier(
                [apply_weights(weights, fs[i]) for i in idx_hold], y_hold, w_hold
            ),
            GBM: weighted_brier(candidate.predict_many([xs[i] for i in idx_hold]), y_hold, w_hold),
        }
        result["weights"] = weights
        serving = GBMChallenger()
        serving.fit(xs, ys, w_all)
        result["challenger"] = serving
        return done("ok")

    def apply_challenger(self, result: Any, now: datetime) -> dict[str, Any]:
        """Cheap: install the fitted challenger / fusion weights and update selection."""
        status = result["status"]
        brier = dict(result.get("brier") or {})

        def summary(trained: bool, reason: str) -> dict[str, Any]:
            return {
                "trained": trained,
                "reason": reason,
                "rows": result["rows"],
                "seconds": result.get("seconds", 0.0),
                "brier": brier if trained else {},
                "active_model": self._serving_model(),
            }

        if int(result.get("generation", 0)) != self._generation:
            # Reset / history replaced while the fit ran: the result describes a
            # model that no longer exists; leave selection and counters untouched.
            return summary(False, "stale")
        if status == "ok" and not self._challenger_enabled:
            status = "disabled"  # switched off while training ran
        if status in ("disabled", "numpy_unavailable"):
            return summary(False, status)
        self._last_training = to_utc(now)
        self._resolved_at_training = int(result["resolved_total"])
        if status != "ok":
            return summary(False, status)

        self.selector.update(brier, result["rows"])
        st = self.stacker
        st.weights = list(result["weights"])
        st.trained_on = int(result["rows"])
        st.holdout_brier_learned = brier.get(LEARNED_FUSION)
        st.holdout_brier_bayes = brier.get(KERNEL_BAYES)
        learned, bayes = st.holdout_brier_learned, st.holdout_brier_bayes
        st.active = (
            learned is not None and bayes is not None and learned < bayes * (1 - ACTIVATION_MARGIN)
        )
        self._challenger = result["challenger"]
        return summary(True, "ok")

    def train_challenger(self, now: datetime) -> dict[str, Any]:
        """Train + evaluate the challenger and update the selection (blocking).

        Equivalent to ``apply_challenger(fit_challenger(challenger_snapshot(), now))``.
        """
        return self.apply_challenger(self.fit_challenger(self.challenger_snapshot(), now), now)

    def resolve(self, now: datetime, is_home: bool) -> int:
        now = to_utc(now)
        if is_home:
            # Arrival is resolved by observe_transition; drop orphans of unknown outcome.
            self._log.pending = [e for e in self._log.pending if e.horizon_end >= now]
            self._calib_pending = [e for e in self._calib_pending if e.horizon_end >= now]
            self._maybe_refit(now)
            return 0
        count = self._settle(now=now)
        self._maybe_refit(now)
        return count

    def maturity(self, now: datetime) -> MaturityReport:
        return maturity_report(
            self._episodes,
            self._log.resolved,
            to_utc(now),
            self.config.half_life_days,
            fusion=FUSION_ALIAS[self._serving_model()],
            active_model=self._serving_model(),
            candidate_brier=dict(self.selector.brier),
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
            "calib_pending": encode_entries(self._calib_pending),
            "segments": self._segments.to_dict(),
            "since_refit": self._since_refit,
            "selection": self.selector.to_dict(),
            "challenger_enabled": self._challenger_enabled,
            "challenger": self._challenger.to_dict(),
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
        model.calibrator = LRCalibrator.from_dict(data.get("calibrator", {}))
        model.stacker = Stacker.from_dict(data.get("stacker", {}))
        model._log = PredictionLog.from_dict(data.get("log", {}))
        model._calib_pending = decode_entries(data.get("calib_pending"))
        model._segments = CalibrationSegments.from_dict(data.get("segments") or {})
        model._since_refit = int(data.get("since_refit", 0))
        if "selection" in data:
            model.selector = Selector.from_dict(data["selection"] or {})
        elif model.stacker.active:  # state written before the generic selection existed
            model.selector.champion = LEARNED_FUSION
        model._challenger_enabled = bool(data.get("challenger_enabled", False))
        if model._challenger_enabled:
            model._challenger = GBMChallenger.from_dict(data.get("challenger"))
        model._generation = int(data.get("generation", 0))
        return model
