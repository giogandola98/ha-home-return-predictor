"""Champion / challenger selection on a chronological holdout (pure Python).

Candidates:
  * ``kernel_bayes``   - kernel survival prior x calibrated GPS LR (reference model)
  * ``learned_fusion`` - the prior-centred logistic stacker
  * ``sklearn_gbm``    - optional HistGradientBoosting challenger

Each evaluation scores every available candidate on the same holdout (the last
20 % of resolved entries, recency-weighted Brier; the trainable candidates are fit
on the first 80 %). Rules:

  * a candidate replaces the current champion only after beating the champion's
    Brier by its margin on ``wins`` consecutive evaluations, with at least
    ``min_entries`` resolved entries (sklearn: 3 %, 2 evaluations, 400 entries;
    learned fusion keeps its historical rule: 1 %, 1 evaluation, 300 entries);
  * a non-reference champion is demoted to ``kernel_bayes`` immediately when it
    falls behind ``kernel_bayes`` on an evaluation (or is not scored at all).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .episodes import minutes_between
from .prior import DAY_MINUTES, recency_weight

KERNEL_BAYES = "kernel_bayes"
LEARNED_FUSION = "learned_fusion"
SKLEARN_GBM = "sklearn_gbm"
CANDIDATES: tuple[str, ...] = (KERNEL_BAYES, LEARNED_FUSION, SKLEARN_GBM)
FUSION_ALIAS: dict[str, str] = {
    KERNEL_BAYES: "bayes",
    LEARNED_FUSION: "learned",
    SKLEARN_GBM: "sklearn",
}
HOLDOUT_FRACTION = 0.2


@dataclass(frozen=True)
class PromotionRule:
    """Evidence a challenger needs before it replaces the champion."""

    min_entries: int
    margin: float  # relative Brier improvement over the champion
    wins: int  # consecutive evaluations


RULES: dict[str, PromotionRule] = {
    LEARNED_FUSION: PromotionRule(min_entries=300, margin=0.01, wins=1),
    SKLEARN_GBM: PromotionRule(min_entries=400, margin=0.03, wins=2),
}


def chronological_split[T](
    items: Sequence[T], holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[list[T], list[T]]:
    """(train, holdout): the holdout is the most recent fraction of ``items``."""
    split = int(len(items) * (1.0 - holdout_fraction))
    return list(items[:split]), list(items[split:])


def recency_weights(times: Sequence[datetime], now: datetime, half_life_days: float) -> list[float]:
    return [recency_weight(minutes_between(t, now) / DAY_MINUTES, half_life_days) for t in times]


def weighted_brier(
    probs: Sequence[float], outcomes: Sequence[int], weights: Sequence[float]
) -> float | None:
    num = den = 0.0
    for p, o, w in zip(probs, outcomes, weights, strict=True):
        num += w * (p - float(o)) ** 2
        den += w
    return num / den if den > 0 else None


class Selector:
    """Hysteresis state of the champion/challenger selection (JSON-safe)."""

    def __init__(self) -> None:
        self.champion = KERNEL_BAYES
        self.streak: dict[str, int] = {}
        self.brier: dict[str, float | None] = {}
        self.evaluations = 0

    def update(self, briers: dict[str, float | None], n_entries: int) -> str:
        """Record one evaluation and return the (possibly new) champion.

        ``briers`` maps candidate -> holdout Brier (None / missing = not scored).
        """
        self.evaluations += 1
        self.brier = {c: briers.get(c) for c in CANDIDATES}
        ref = self.brier.get(KERNEL_BAYES)
        champ = self.champion
        champ_b = self.brier.get(champ)
        if champ != KERNEL_BAYES and (champ_b is None or (ref is not None and champ_b > ref)):
            champ, champ_b = KERNEL_BAYES, ref
            self.streak = {}
        promoted: list[tuple[float, str]] = []
        for cand, rule in RULES.items():
            b = self.brier.get(cand)
            if cand == champ:
                self.streak.pop(cand, None)
                continue
            if (
                b is not None
                and champ_b is not None
                and n_entries >= rule.min_entries
                and b <= champ_b * (1.0 - rule.margin)
            ):
                self.streak[cand] = self.streak.get(cand, 0) + 1
                if self.streak[cand] >= rule.wins:
                    promoted.append((b, cand))
            else:
                self.streak.pop(cand, None)
        if promoted:
            champ = min(promoted)[1]
            self.streak = {}
        self.champion = champ
        return champ

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": self.champion,
            "streak": dict(self.streak),
            "brier": dict(self.brier),
            "evaluations": self.evaluations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Selector:
        sel = cls()
        champ = data.get("champion")
        sel.champion = champ if champ in CANDIDATES else KERNEL_BAYES
        sel.streak = {c: int(v) for c, v in (data.get("streak") or {}).items() if c in CANDIDATES}
        sel.brier = {
            c: (None if v is None else float(v))
            for c, v in (data.get("brier") or {}).items()
            if c in CANDIDATES
        }
        sel.evaluations = int(data.get("evaluations", 0))
        return sel
