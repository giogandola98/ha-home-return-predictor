"""Constants for Home Return Predictor."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "home_return_predictor"
NAME: Final = "Home Return Predictor"

# Config entry data
CONF_PERSONS: Final = "persons"
CONF_HOME_ZONE: Final = "home_zone"
DEFAULT_HOME_ZONE: Final = "zone.home"

# Options
CONF_RETURN_THRESHOLD: Final = "return_threshold"
CONF_MATURITY_THRESHOLD: Final = "maturity_threshold"
CONF_HORIZON_MINUTES: Final = "horizon_minutes"
CONF_HALF_LIFE_DAYS: Final = "half_life_days"
CONF_UPDATE_INTERVAL_MINUTES: Final = "update_interval_minutes"
CONF_HOLIDAY_CALENDAR: Final = "holiday_calendar"  # optional calendar.* entity
# Challenger option. The stored key keeps its historical name (the challenger used
# scikit-learn until scikit-learn proved uninstallable on HA OS) so that existing
# config entries keep their setting; only the UI label changed.
CONF_ENABLE_CHALLENGER: Final = "enable_sklearn_challenger"

DEFAULT_RETURN_THRESHOLD: Final = 0.6
DEFAULT_MATURITY_THRESHOLD: Final = 0.7
DEFAULT_HORIZON_MINUTES: Final = 90
DEFAULT_HALF_LIFE_DAYS: Final = 28
DEFAULT_UPDATE_INTERVAL_MINUTES: Final = 2
DEFAULT_ENABLE_CHALLENGER: Final = False
MATURITY_HYSTERESIS: Final = 0.1
RETURNING_OFF_HYSTERESIS: Final = 0.05  # returning stays on until p < threshold - this
HOUSE_ETA_MIN_PROBABILITY: Final = 0.25  # away persons considered for the house ETA

# Optional gradient-boosting challenger: needs numpy, installed at runtime only when
# enabled and not already importable. No version: HA's package constraints pin numpy
# (a musl wheel exists for the pinned version, unlike scikit-learn).
NUMPY_REQUIREMENT: Final = "numpy"
DEFAULT_ACTIVE_MODEL: Final = "kernel_bayes"

# Modes
MODE_AUTO: Final = "auto"
MODE_LEARNING: Final = "learning"
MODE_LIVE: Final = "live"
MODES: Final = [MODE_AUTO, MODE_LEARNING, MODE_LIVE]

# Event
EVENT_RETURN_DETECTED: Final = f"{DOMAIN}_return_detected"

# Services
SERVICE_RETRAIN: Final = "retrain"
SERVICE_RESET: Final = "reset"
SERVICE_EXPORT_DIAGNOSTICS: Final = "export_diagnostics"

# Storage
STORAGE_VERSION: Final = 1
STORAGE_KEY_PREFIX: Final = DOMAIN

# Entity keys (translation_key == key; unique_id = f"{entry_id}_{person_slug}_{key}")
KEY_RETURNING: Final = "returning"
KEY_RETURN_PROBABILITY: Final = "return_probability"
KEY_RETURN_ETA: Final = "return_eta"
KEY_MODEL_MATURITY: Final = "model_maturity"
KEY_RETURN_REASON: Final = "return_reason"
KEY_MODE: Final = "mode"
KEY_HOUSE_RETURN_PROBABILITY: Final = "house_return_probability"
KEY_HOUSE_NEXT_RETURN_ETA: Final = "house_next_return_eta"
KEY_HOUSE_SOMEONE_RETURNING: Final = "house_someone_returning"
HOUSE_DEVICE_SLUG: Final = "house"

# Recorder bootstrap
HISTORY_DAYS: Final = 120
HISTORY_CHUNK_DAYS: Final = 7  # recorder query window (bounds memory)
HISTORY_FIX_MIN_INTERVAL_S: Final = 60  # history fixes downsampled to 1 per minute

# Runtime tuning
SAVE_DELAY_SECONDS: Final = 600  # coalesce saves; final write on unload / HA stop
FIX_WINDOW_MINUTES: Final = 30
MAX_FIX_ACCURACY_M: Final = 250.0
DIAGNOSTICS_LOG_ENTRIES: Final = 50

# Service fields
ATTR_PERSON: Final = "person"
