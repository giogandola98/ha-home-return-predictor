"""Prediction log, Brier skill vs a duration-only baseline and the maturity score."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from .base import AbsenceEpisode, MaturityReport
from .episodes import minutes_between, to_utc
from .fusion import P_MAX, P_MIN, clamp, logit, sigmoid
from .prior import DAY_MINUTES, recency_weight

MAX_RESOLVED = 3000
MAX_RESOLVED_AGE_DAYS = 90.0
MIN_RESOLVED_FOR_SKILL = 30
EPISODE_SCALE = 15.0  # data factor: 1 - e^(-n_eff / 15)
# Full skill at BSS 0.5 vs the duration-only baseline (habits + GPS halve the Brier
# of "how long has the person been away"); vs climatology it used to be 0.25.
BSS_FULL = 0.5
# Return-based skill (the per-person metrics of house_metric.py, on the prediction
# log): share of real returns the signal p >= SIGNAL_P caught in the horizon before
# the arrival, minus FALSE_ALARM_WEIGHT x false-alarm runs per return; full at
# RECALL_FULL, defined after MIN_RETURNS_FOR_SKILL returns.
SIGNAL_P = 0.5
FALSE_ALARM_WEIGHT = 0.5
RECALL_FULL = 0.8
MIN_RETURNS_FOR_SKILL = 5
ABSENCE_GAP_MINUTES = 25.0  # > 2 x the 10-min log interval: home in between


@dataclass
class LogEntry:
    """One logged prediction (or GPS calibration observation)."""

    ts: datetime
    p: float
    prior: float | None
    approach_class: str | None
    horizon_end: datetime
    outcome: int | None = None
    features: list[float] = field(default_factory=list)  # stacker features
    base: float | None = None  # duration-only baseline P, computed at log time
    x: list[float] = field(default_factory=list)  # challenger features

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "p": self.p,
            "prior": self.prior,
            "cls": self.approach_class,
            "hend": self.horizon_end.isoformat(),
            "outcome": self.outcome,
            "f": list(self.features),
            "b": self.base,
            "x": list(self.x),
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
            base=None if data.get("b") is None else float(data["b"]),
            x=[float(v) for v in data.get("x", [])],
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


# ------------------------------------------------------------------ compact storage
LOG_DECIMALS = 4
# Challenger feature positions duplicated from the stacker features (x index -> f index);
# x[11] (z_bayes) is derived from f[0] + f[1], and x[2] (away horizons, capped at 16)
# equals f[2] (capped at 8) below the stacker cap. Only the other x columns are stored:
# 4 values (x[2] = f[2]) or 5 values (x[2] stored first); 0 / 12 values = verbatim.
_X_FROM_F: dict[int, int] = {0: 0, 1: 1, 5: 5, 6: 6, 8: 3, 10: 4}
_X_DERIVED = 11
_X_AWAY = 2
_F_AWAY_CAP = 8.0
_X_OWN: tuple[int, ...] = (3, 4, 7, 9)
_N_F = 7
_N_X = 12


def round_value(value: float | None) -> float | int | None:
    """Rounded to LOG_DECIMALS; integral values become ints (shorter JSON)."""
    if value is None:
        return None
    v = round(float(value), LOG_DECIMALS)
    return int(v) if v.is_integer() else v


def round_float(value: float | None) -> float | None:
    """In-memory counterpart of ``round_value`` (what a JSON round trip yields)."""
    return None if value is None else float(round(float(value), LOG_DECIMALS))


def _z_bayes(f: Sequence[float]) -> float:
    return round(logit(clamp(sigmoid(f[0] + f[1]), P_MIN, P_MAX)), LOG_DECIMALS)


def _pack_x(f: Sequence[float], x: Sequence[float]) -> list[float | int | None]:
    """Store only the challenger columns not recoverable from ``f`` (if possible)."""
    rx = [float(round_value(v) or 0) for v in x]
    rf = [float(round_value(v) or 0) for v in f]
    if (
        len(rf) == _N_F
        and len(rx) == _N_X
        and all(rx[i] == rf[j] for i, j in _X_FROM_F.items())
        and rx[_X_DERIVED] == _z_bayes(rf)
    ):
        own = [round_value(rx[i]) for i in _X_OWN]
        if rx[_X_AWAY] == rf[_X_AWAY] and rf[_X_AWAY] < _F_AWAY_CAP:
            return own
        return [round_value(rx[_X_AWAY]), *own]
    return [round_value(v) for v in rx]


def _unpack_x(f: Sequence[float], packed: Sequence[float]) -> list[float]:
    n_own = len(_X_OWN)
    if len(packed) not in (n_own, n_own + 1) or len(f) != _N_F:
        return [float(v) for v in packed]
    x = [0.0] * _N_X
    for i, j in _X_FROM_F.items():
        x[i] = float(f[j])
    x[_X_AWAY] = float(packed[0]) if len(packed) > n_own else float(f[_X_AWAY])
    for i, v in zip(_X_OWN, packed[-n_own:], strict=True):
        x[i] = float(v)
    x[_X_DERIVED] = _z_bayes(f)
    return x


def _epoch(dt: datetime) -> int:
    return int(to_utc(dt).timestamp())


def _from_epoch(value: float) -> datetime:
    return datetime.fromtimestamp(int(value), UTC)


def encode_entries(entries: Sequence[LogEntry]) -> dict[str, list[Any]]:
    """Column arrays: int epoch seconds, floats rounded to 4 decimals."""
    classes: list[str | None] = []
    codes: list[int] = []
    for e in entries:
        if e.approach_class not in classes:
            classes.append(e.approach_class)
        codes.append(classes.index(e.approach_class))
    horizons = [round((e.horizon_end - e.ts).total_seconds()) for e in entries]
    return {
        "ts": [_epoch(e.ts) for e in entries],
        # one int when every entry has the same horizon (the usual case)
        "h": horizons[0] if len(set(horizons)) == 1 else horizons,
        "p": [round_value(e.p) for e in entries],
        "prior": [round_value(e.prior) for e in entries],
        "classes": classes,
        "cls": codes,
        "o": [e.outcome for e in entries],
        "b": [round_value(e.base) for e in entries],
        "f": [[round_value(v) for v in e.features] for e in entries],
        "x": [_pack_x(e.features, e.x) for e in entries],
    }


def decode_entries(data: Any) -> list[LogEntry]:
    """Inverse of ``encode_entries``; also reads the legacy list-of-dicts format."""
    if isinstance(data, list):
        return [LogEntry.from_dict(e) for e in data]
    if not isinstance(data, dict) or not data.get("ts"):
        return []
    classes = data.get("classes") or []
    h = data["h"]
    horizons = h if isinstance(h, list) else [h] * len(data["ts"])
    out: list[LogEntry] = []
    for i, ts_raw in enumerate(data["ts"]):
        ts = _from_epoch(ts_raw)
        prior = data["prior"][i]
        base = data["b"][i]
        features = [float(v) for v in data["f"][i]]
        outcome = data["o"][i]
        out.append(
            LogEntry(
                ts=ts,
                p=float(data["p"][i]),
                prior=None if prior is None else float(prior),
                approach_class=classes[data["cls"][i]],
                horizon_end=ts + timedelta(seconds=int(horizons[i])),
                outcome=None if outcome is None else int(outcome),
                features=features,
                base=None if base is None else float(base),
                x=_unpack_x(features, data["x"][i]),
            )
        )
    return out


def compact_entry(entry: LogEntry) -> LogEntry:
    """The entry exactly as it reads back from storage (rounded, whole seconds).

    Applied when an entry is logged, so a persistence round trip changes nothing.
    """
    return decode_entries(encode_entries([entry]))[0]


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
            "pending": encode_entries(self.pending),
            "resolved": encode_entries(self.resolved),
            "last_logged": self.last_logged.isoformat() if self.last_logged else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PredictionLog:
        log = cls()
        log.pending = decode_entries(data.get("pending"))
        log.resolved = decode_entries(data.get("resolved"))
        last = data.get("last_logged")
        log.last_logged = datetime.fromisoformat(last) if last else None
        return log


def weighted_brier(
    entries: Sequence[LogEntry], probs: Sequence[float], now: datetime, half_life_days: float
) -> tuple[float, float] | None:
    """Recency-weighted (Brier of ``probs``, Brier of the baseline) vs entry outcomes.

    The baseline is each entry's duration-only survival probability stored at log
    time (an honest out-of-sample forecast); entries logged before the baseline
    existed fall back to the weighted climatology rate.
    """
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
    clim = pos / den
    ref = 0.0
    for entry, w in zip(entries, weights, strict=True):
        b = clim if entry.base is None else entry.base
        ref += w * (b - float(entry.outcome or 0)) ** 2
    return num / den, ref / den


def effective_episodes(
    episodes: Sequence[AbsenceEpisode], now: datetime, half_life_days: float
) -> float:
    return sum(
        recency_weight(minutes_between(e.left_at, now) / DAY_MINUTES, half_life_days)
        for e in episodes
    )


@dataclass(frozen=True)
class ReturnSkill:
    returns: float  # recency-weighted returns in the log
    recall: float | None
    false_alarms_per_return: float | None
    count: int  # returns (unweighted)


def return_skill(resolved: Sequence[LogEntry], now: datetime, half_life_days: float) -> ReturnSkill:
    """Caught returns and false alarms of the signal ``p >= SIGNAL_P`` in the log.

    The log holds absences only: a gap longer than ``ABSENCE_GAP_MINUTES`` or a
    positive entry followed by a negative one starts a new absence. A return is caught
    when one of its positive entries (horizon before the arrival) had p >= SIGNAL_P;
    a false alarm is a run of p >= SIGNAL_P with no positive entry.
    """
    entries = sorted((e for e in resolved if e.outcome is not None), key=lambda e: e.ts)
    absences: list[list[LogEntry]] = []
    for e in entries:
        if (
            absences
            and minutes_between(absences[-1][-1].ts, e.ts) <= ABSENCE_GAP_MINUTES
            and not (absences[-1][-1].outcome == 1 and e.outcome == 0)
        ):
            absences[-1].append(e)
        else:
            absences.append([e])
    w_ret = w_caught = w_false = 0.0
    count = 0
    for ab in absences:
        w = recency_weight(minutes_between(ab[-1].ts, now) / DAY_MINUTES, half_life_days)
        runs, run_has_pos, in_run = 0, False, False
        for e in ab:
            if e.p >= SIGNAL_P:
                if not in_run:
                    in_run, run_has_pos = True, False
                run_has_pos = run_has_pos or e.outcome == 1
            elif in_run:
                runs += 0 if run_has_pos else 1
                in_run = False
        if in_run and not run_has_pos:
            runs += 1
        w_false += w * runs
        if any(e.outcome == 1 for e in ab):
            count += 1
            w_ret += w
            if any(e.outcome == 1 and e.p >= SIGNAL_P for e in ab):
                w_caught += w
    if w_ret <= 0:
        return ReturnSkill(0.0, None, None, count)
    return ReturnSkill(w_ret, w_caught / w_ret, w_false / w_ret, count)


def maturity_report(
    episodes: Sequence[AbsenceEpisode],
    resolved: Sequence[LogEntry],
    now: datetime,
    half_life_days: float,
    fusion: str = "bayes",
    active_model: str = "kernel_bayes",
    candidate_brier: dict[str, float | None] | None = None,
) -> MaturityReport:
    """maturity = (1 - e^(-n_eff/15)) x return skill.

    Return skill = clamp((recall - 0.5 x false alarms per return) / 0.8, 0, 1) of the
    signal p >= 0.5 over the logged absences (``return_skill``), 0 before 5 returns.
    The Brier scores are still reported; they no longer drive the maturity (a few
    long absences dominate a per-entry Brier).
    """
    n_eff = effective_episodes(episodes, now, half_life_days)
    data_factor = 1.0 - math.exp(-n_eff / EPISODE_SCALE)
    bs = ref = bss = None
    skill = 0.0
    scores = weighted_brier(resolved, [e.p for e in resolved], now, half_life_days)
    if scores is not None:
        bs, ref = scores
        if ref > 1e-9:
            bss = 1.0 - bs / ref
    rs = return_skill(resolved, now, half_life_days)
    if rs.count >= MIN_RETURNS_FOR_SKILL and rs.recall is not None:
        raw = rs.recall - FALSE_ALARM_WEIGHT * (rs.false_alarms_per_return or 0.0)
        skill = min(max(raw / RECALL_FULL, 0.0), 1.0)
    return MaturityReport(
        maturity=data_factor * skill,
        effective_episodes=n_eff,
        episodes=len(episodes),
        resolved_predictions=len(resolved),
        brier_score=bs,
        baseline_brier_score=ref,
        brier_skill_score=bss,
        fusion=fusion,
        active_model=active_model,
        candidate_brier=dict(candidate_brier or {}),
        return_recall=rs.recall,
        false_alarms_per_return=rs.false_alarms_per_return,
    )
