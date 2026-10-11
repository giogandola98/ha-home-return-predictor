"""Model contract shared by the Home Assistant glue and any model implementation.

This module (and the whole ``hrp_core`` package) must never import Home Assistant:
it is pure standard-library Python so it can be unit tested and simulated alone.
Only relative imports are allowed inside ``hrp_core``.

All datetimes are timezone-aware. Implementations must treat "local time of day"
(used for departure-hour similarity) via ``dt.astimezone(tz)`` where ``tz`` is the
model's configured timezone (see ``ModelConfig.timezone``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Tunable parameters (mirrors the integration options flow)."""

    horizon_minutes: float = 90.0
    half_life_days: float = 28.0
    timezone: tzinfo | None = None  # None -> use each datetime's own tzinfo
    min_absence_minutes: float = 10.0
    trip_absence_hours: float = 36.0
    max_fix_accuracy_m: float = 250.0
    regression_window_minutes: float = 10.0
    prediction_log_interval_minutes: float = 10.0


@dataclass(frozen=True)
class PresenceTransition:
    """A change of the person's home/away state."""

    ts: datetime
    is_home: bool


@dataclass(frozen=True)
class PositionFix:
    """A GPS fix already converted to distance from the home zone."""

    ts: datetime
    distance_m: float
    accuracy_m: float | None = None


@dataclass(frozen=True)
class AbsenceEpisode:
    """One completed absence from home."""

    left_at: datetime
    returned_at: datetime
    weekday: int  # 0 = Monday, in model timezone, of left_at
    is_holiday: bool = False

    @property
    def duration_minutes(self) -> float:
        return (self.returned_at - self.left_at).total_seconds() / 60.0


@dataclass(frozen=True)
class ModelInput:
    """Everything the model needs to make one prediction."""

    now: datetime
    is_home: bool
    left_at: datetime | None  # start of current absence; None if unknown / at home
    is_holiday: bool = False
    fixes: Sequence[PositionFix] = ()  # recent fixes, oldest first


@dataclass
class Prediction:
    """Output of ``ReturnModel.predict``. Pure data, no side effects."""

    probability: float  # fused P(return within horizon), 0..1 (1.0 when at home)
    prior_probability: float | None  # habit prior alone
    likelihood_ratio: float  # GPS evidence LR (1.0 = neutral)
    eta: datetime | None  # fused ETA
    eta_habit: datetime | None
    eta_position: datetime | None
    approach_speed_kmh: float | None  # >0 approaching, <0 moving away
    approach_class: str | None
    distance_m: float | None
    is_home: bool
    prior_effective_samples: float = 0.0
    reason: dict[str, Any] = field(default_factory=dict)  # machine-readable parts


@dataclass
class MaturityReport:
    """How well the model knows this person."""

    maturity: float  # 0..1
    effective_episodes: float
    episodes: int
    resolved_predictions: int
    brier_score: float | None
    baseline_brier_score: float | None
    brier_skill_score: float | None
    fusion: str = "bayes"  # "bayes" or "learned" (active probability fusion)


class ReturnModel(ABC):
    """Abstract per-person return predictor.

    Lifecycle used by the integration:
      1. ``load_history`` on bootstrap / retrain (replaces episodes).
      2. ``observe_transition`` on every person home/away change.
      3. ``predict`` every update interval (pure).
      4. ``log_prediction`` after predict (the model rate-limits and later
         resolves the entry against the real outcome; it also feeds the GPS
         likelihood calibration).
      5. ``resolve`` periodically, so entries whose horizon has elapsed while the
         person is still away are scored as "no return".
      6. ``to_dict`` / ``from_dict`` for persistence (JSON-safe).
    """

    @abstractmethod
    def load_history(
        self,
        transitions: Iterable[PresenceTransition],
        fixes: Iterable[PositionFix] = (),
        holidays: Iterable[datetime] = (),
    ) -> None:
        """Rebuild episodes (and optionally warm the GPS calibration) from history."""

    @abstractmethod
    def observe_transition(
        self, transition: PresenceTransition, is_holiday: bool = False
    ) -> AbsenceEpisode | None:
        """Record a live presence change; returns the episode closed by an arrival."""

    @abstractmethod
    def predict(self, model_input: ModelInput) -> Prediction:
        """Return the current prediction. Must not mutate state."""

    @abstractmethod
    def log_prediction(self, model_input: ModelInput, prediction: Prediction) -> bool:
        """Store the prediction for later scoring; returns True if it was logged."""

    @abstractmethod
    def resolve(self, now: datetime, is_home: bool) -> int:
        """Score pending predictions whose outcome is known; returns count resolved."""

    @abstractmethod
    def maturity(self, now: datetime) -> MaturityReport:
        """Return the maturity report."""

    @abstractmethod
    def episodes(self) -> list[AbsenceEpisode]:
        """Return stored (non-trip) episodes, oldest first."""

    @abstractmethod
    def reset(self) -> None:
        """Forget everything learnt."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any], config: ModelConfig) -> ReturnModel:
        """Restore from ``to_dict`` output."""
