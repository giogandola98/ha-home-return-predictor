"""Presence transitions -> absence episodes (with noise and trip filtering)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, tzinfo

from .base import AbsenceEpisode, ModelConfig, PresenceTransition

KIND_DROP = "drop"
KIND_TRIP = "trip"
KIND_EPISODE = "episode"


def to_utc(dt: datetime) -> datetime:
    """Normalise an aware datetime to UTC (safe arithmetic across DST)."""
    return dt.astimezone(UTC)


def to_local(dt: datetime, tz: tzinfo | None) -> datetime:
    """Convert to the model timezone (or keep the datetime's own tzinfo)."""
    return dt.astimezone(tz) if tz is not None else dt


def minutes_between(start: datetime, end: datetime) -> float:
    """Elapsed real minutes from ``start`` to ``end`` (DST safe)."""
    return (to_utc(end) - to_utc(start)).total_seconds() / 60.0


def holiday_dates(holidays: Iterable[datetime | date], tz: tzinfo | None) -> set[date]:
    """Local calendar dates of the given holidays."""
    out: set[date] = set()
    for item in holidays:
        if isinstance(item, datetime):
            out.add(to_local(item, tz).date())
        else:
            out.add(item)
    return out


def make_episode(
    left_at: datetime, returned_at: datetime, tz: tzinfo | None, is_holiday: bool = False
) -> AbsenceEpisode:
    """Build an episode stored in UTC, weekday taken in the model timezone."""
    return AbsenceEpisode(
        left_at=to_utc(left_at),
        returned_at=to_utc(returned_at),
        weekday=to_local(left_at, tz).weekday(),
        is_holiday=is_holiday,
    )


def classify_episode(episode: AbsenceEpisode, config: ModelConfig) -> str:
    """Return KIND_DROP (GPS noise), KIND_TRIP (long absence) or KIND_EPISODE."""
    minutes = episode.duration_minutes
    if minutes < config.min_absence_minutes:
        return KIND_DROP
    if minutes > config.trip_absence_hours * 60.0:
        return KIND_TRIP
    return KIND_EPISODE


@dataclass
class EpisodeBuild:
    """Result of replaying a transition history."""

    episodes: list[AbsenceEpisode] = field(default_factory=list)
    trips: list[AbsenceEpisode] = field(default_factory=list)
    is_home: bool | None = None
    away_since: datetime | None = None


def build_episodes(
    transitions: Iterable[PresenceTransition],
    config: ModelConfig,
    holidays: Iterable[datetime | date] = (),
) -> EpisodeBuild:
    """Pair away -> home transitions into filtered episodes and trips."""
    hol = holiday_dates(holidays, config.timezone)
    result = EpisodeBuild()
    for tr in sorted(transitions, key=lambda t: to_utc(t.ts)):
        if tr.is_home == result.is_home:
            continue  # duplicate state report
        if not tr.is_home:
            result.away_since = to_utc(tr.ts)
        elif result.away_since is not None:
            left = result.away_since
            ep = make_episode(
                left, tr.ts, config.timezone, to_local(left, config.timezone).date() in hol
            )
            kind = classify_episode(ep, config)
            if kind == KIND_EPISODE:
                result.episodes.append(ep)
            elif kind == KIND_TRIP:
                result.trips.append(ep)
            result.away_since = None
        result.is_home = tr.is_home
    return result
