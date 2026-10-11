"""Prediction log, Brier skill vs climatology and the maturity score."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base import AbsenceEpisode, MaturityReport
from .episodes import minutes_between, to_utc
from .prior import DAY_MINUTES, recency_weight

MAX_RESOLVED = 3000
MAX_RESOLVED_AGE_DAYS = 90.0
MIN_RESOLVED_FOR_SKILL = 30
EPISODE_SCALE = 12.0
BSS_FULL = 0.25


@dataclass
class LogEntry:
    """One logged prediction (or GPS calibration observation)."""

    ts: datetime
    p: float
    prior: float | None
    approach_class: str | None
    horizon_end: datetime
    outcome: int | None = None
    features: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "p": self.p,
            "prior": self.prior,
            "cls": self.approach_class,
            "hend": self.horizon_end.isoformat(),
            "outcome": self.outcome,
            "f": list(self.features),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogEntry:
        return cls(
            ts=datetime.fromisoformat(data["ts"]),
            p=float(data["p"]),
            prior=None if data.get("prior") is None else float(data["prior"]),
            approach_class=data.get("cls"),
            horizon_end=datetime.fromisoformat(data["hend"]),
            outcome=data.get("outcome"),
            features=[float(x) for x in data.get("f", [])],
        )


def resolve_entries(
    pending: list[LogEntry], *, arrival: datetime | None = None, now: datetime | None = None
) -> tuple[list[LogEntry], list[LogEntry]]:
    """Split pending entries into (resolved, still pending).

    With ``arrival``: entries whose window contains it -> 1, elapsed windows -> 0
    (everything pending gets resolved). With ``now`` only (still away): entries
    whose window has elapsed -> 0.
    """
    resolved: list[LogEntry] = []
    keep: list[LogEntry] = []
    for entry in pending:
        if arrival is not None:
            if entry.ts <= arrival <= entry.horizon_end:
                entry.outcome = 1
                resolved.append(entry)
            elif entry.horizon_end < arrival:
                entry.outcome = 0
                resolved.append(entry)
            # entries logged after the arrival are stale: drop
        elif now is not None and entry.horizon_end < now:
            entry.outcome = 0
            resolved.append(entry)
        else:
            keep.append(entry)
    return resolved, keep


class PredictionLog:
    """Rate-limited prediction log with resolved history."""

    def __init__(self) -> None:
        self.pending: list[LogEntry] = []
        self.resolved: list[LogEntry] = []
        self.last_logged: datetime | None = None

    def add(self, entry: LogEntry, interval_minutes: float) -> bool:
        if (
            self.last_logged is not None
            and minutes_between(self.last_logged, entry.ts) < interval_minutes
        ):
            return False
        self.pending.append(entry)
        self.last_logged = entry.ts
        return True

    def settle(self, resolved: list[LogEntry], now: datetime) -> None:
        self.resolved.extend(resolved)
        cutoff = to_utc(now) - timedelta(days=MAX_RESOLVED_AGE_DAYS)
        self.resolved = [e for e in self.resolved if e.ts >= cutoff][-MAX_RESOLVED:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending": [e.to_dict() for e in self.pending],
            "resolved": [e.to_dict() for e in self.resolved],
            "last_logged": self.last_logged.isoformat() if self.last_logged else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionLog:
        log = cls()
        log.pending = [LogEntry.from_dict(e) for e in data.get("pending", [])]
        log.resolved = [LogEntry.from_dict(e) for e in data.get("resolved", [])]
        last = data.get("last_logged")
        log.last_logged = datetime.fromisoformat(last) if last else None
        return log


def weighted_brier(
    entries: Sequence[LogEntry], probs: Sequence[float], now: datetime, half_life_days: float
) -> tuple[float, float] | None:
    """Recency-weighted (Brier, climatology Brier) of ``probs`` vs entry outcomes."""
    num = den = pos = 0.0
    weights: list[float] = []
    for entry, p in zip(entries, probs, strict=True):
        w = recency_weight(minutes_between(entry.ts, now) / DAY_MINUTES, half_life_days)
        weights.append(w)
        o = float(entry.outcome or 0)
        num += w * (p - o) ** 2
        den += w
        pos += w * o
    if den <= 0:
        return None
    base = pos / den
    return num / den, base * (1.0 - base)


def effective_episodes(
    episodes: Sequence[AbsenceEpisode], now: datetime, half_life_days: float
) -> float:
    return sum(
        recency_weight(minutes_between(e.left_at, now) / DAY_MINUTES, half_life_days)
        for e in episodes
    )


def maturity_report(
    episodes: Sequence[AbsenceEpisode],
    resolved: Sequence[LogEntry],
    now: datetime,
    half_life_days: float,
    fusion: str = "bayes",
) -> MaturityReport:
    """maturity = (1 - e^(-n_eff/12)) x clamp(BSS / 0.25, 0, 1)."""
    n_eff = effective_episodes(episodes, now, half_life_days)
    data_factor = 1.0 - math.exp(-n_eff / EPISODE_SCALE)
    bs = ref = bss = None
    skill = 0.0
    scores = weighted_brier(resolved, [e.p for e in resolved], now, half_life_days)
    if scores is not None:
        bs, ref = scores
        if ref > 1e-9:
            bss = 1.0 - bs / ref
            if len(resolved) >= MIN_RESOLVED_FOR_SKILL:
                skill = min(max(bss / BSS_FULL, 0.0), 1.0)
    return MaturityReport(
        maturity=data_factor * skill,
        effective_episodes=n_eff,
        episodes=len(episodes),
        resolved_predictions=len(resolved),
        brier_score=bs,
        baseline_brier_score=ref,
        brier_skill_score=bss,
        fusion=fusion,
    )
