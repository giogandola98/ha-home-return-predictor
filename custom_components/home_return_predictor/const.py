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

DEFAULT_RETURN_THRESHOLD: Final = 0.6
DEFAULT_MATURITY_THRESHOLD: Final = 0.7
DEFAULT_HORIZON_MINUTES: Final = 90
DEFAULT_HALF_LIFE_DAYS: Final = 28
DEFAULT_UPDATE_INTERVAL_MINUTES: Final = 2
MATURITY_HYSTERESIS: Final = 0.1

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

# Runtime tuning
SAVE_DELAY_SECONDS: Final = 60
FIX_WINDOW_MINUTES: Final = 30
MAX_FIX_ACCURACY_M: Final = 250.0
DIAGNOSTICS_LOG_ENTRIES: Final = 50

# Service fields
ATTR_PERSON: Final = "person"
