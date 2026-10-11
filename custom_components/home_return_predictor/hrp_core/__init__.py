"""Pure-Python prediction core of Home Return Predictor (no Home Assistant imports)."""

from __future__ import annotations

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
from .bayes_model import KernelBayesReturnModel
from .challenger import sklearn_available
from .evidence import haversine_m


def create_model(config: ModelConfig | None = None) -> ReturnModel:
    """Factory used by the integration: the default per-person model."""
    return KernelBayesReturnModel(config or ModelConfig())


__all__ = [
    "AbsenceEpisode",
    "KernelBayesReturnModel",
    "MaturityReport",
    "ModelConfig",
    "ModelInput",
    "PositionFix",
    "Prediction",
    "PresenceTransition",
    "ReturnModel",
    "create_model",
    "haversine_m",
    "sklearn_available",
]
