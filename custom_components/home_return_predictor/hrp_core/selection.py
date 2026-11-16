"""Champion / challenger selection on a chronological holdout (pure Python).

Candidates:
  * ``kernel_bayes``   - kernel survival prior x calibrated GPS LR (reference model)
  * ``learned_fusion`` - the prior-centred logistic stacker
  * ``gbm``            - optional gradient-boosting challenger (numpy, ``gbm.py``)

Each evaluation scores every available candidate on the same holdout (the last
20 % of resolved entries, recency-weighted Brier; the trainable candidates are fit
on the first 80 %). Rules:

  * a candidate replaces the current champion only after beating the champion's
    Brier by its margin on ``wins`` consecutive evaluations, with at least
    ``min_entries`` resolved entries (gbm: 3 %, 2 evaluations, 400 entries;
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
GBM = "gbm"
CANDIDATES: tuple[str, ...] = (KERNEL_BAYES, LEARNED_FUSION, GBM)
# Names written by earlier releases (the challenger used scikit-learn).
LEGACY_CANDIDATES: dict[str, str] = {"sklearn_gbm": GBM}
FUSION_ALIAS: dict[str, str] = {
    KERNEL_BAYES: "bayes",
    LEARNED_FUSION: "learned",
    GBM: "gbm",
}
HOLDOUT_FRACTION = 0.2
# The holdout must contain real returns: on a stretch without any (a trip, a long
# absence) the candidate that always says "no" wins, whatever it does near home.
MIN_HOLDOUT_RETURNS = 3
MAX_HOLDOUT_FRACTION = 0.5  # the holdout may grow back to this share to find them
RUN_GAP_MINUTES = 25.0  # > 2 x the 10-min log interval: the person was home in between


@dataclass(frozen=True)
class PromotionRule:
    """Evidence a challenger needs before it replaces the champion."""

    min_entries: int
    margin: float  # relative Brier improvement over the champion
    wins: int  # consecutive evaluations


RULES: dict[str, PromotionRule] = {
    LEARNED_FUSION: PromotionRule(min_entries=300, margin=0.01, wins=1),
    GBM: PromotionRule(min_entries=400, margin=0.03, wins=2),
}


def chronological_split[T](
    items: Sequence[T], holdout_fraction: float = HOLDOUT_FRACTION
) -> tuple[list[T], list[T]]:
    """(train, holdout): the holdout is the most recent fraction of ``items``."""
    split = int(len(items) * (1.0 - holdout_fraction))
    return list(items[:split]), list(items[split:])


def informative_split(
    outcomes: Sequence[int],
    holdout_fraction: float = HOLDOUT_FRACTION,
    min_returns: int = MIN_HOLDOUT_RETURNS,
    max_fraction: float = MAX_HOLDOUT_FRACTION,
    times: Sequence[datetime] | None = None,
) -> int | None:
    """Start index of a chronological holdout holding ``min_returns`` returns, or None.

    ``outcomes`` are the resolved entries in time order (1 = returned within the
    horizon). A return is a run of consecutive positive entries; with ``times`` a gap
    longer than ``RUN_GAP_MINUTES`` (the log only covers absences: a gap means the
    person was home in between) also ends a run. The holdout is the newest
    ``holdout_fraction``, grown back (up to ``max_fraction``) until it holds
    ``min_returns`` returns, keeping a return whole when the bound allows; None when
    even the largest holdout does not hold enough returns.
    """
    n = len(outcomes)
    if times is not None and len(times) != n:
        raise ValueError("times and outcomes differ in length")
    split = int(n * (1.0 - holdout_fraction))
    lowest = int(n * (1.0 - max(max_fraction, holdout_fraction)))
    if split >= n:
        return None

    def joined(i: int) -> bool:
        """Entry ``i`` continues the run of entry ``i - 1``."""
        if not (outcomes[i] and outcomes[i - 1]):
            return False
        return times is None or minutes_between(times[i - 1], times[i]) <= RUN_GAP_MINUTES

    returns = sum(1 for i in range(split, n) if outcomes[i] and (i == split or not joined(i)))
    while returns < min_returns and split > lowest:
        split -= 1
        # a prepended positive entry starts a new run unless it joins the next one
        if outcomes[split] and not joined(split + 1):
            returns += 1
    if returns < min_returns:
        return None
    while split > lowest and joined(split):
        split -= 1  # keep a return whole on the holdout side
    return split


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


def canonical_candidate(name: Any) -> Any:
    """Map a stored candidate name to the current one (``sklearn_gbm`` -> ``gbm``)."""
    return LEGACY_CANDIDATES.get(name, name) if isinstance(name, str) else name


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

    def demote(self) -> str:
        """Fall back to the reference model (no evaluation could vouch for another one)."""
        self.champion = KERNEL_BAYES
        self.streak = {}
        return self.champion

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
        champ = canonical_candidate(data.get("champion"))
        sel.champion = champ if champ in CANDIDATES else KERNEL_BAYES
        sel.streak = {
            canonical_candidate(c): int(v)
            for c, v in (data.get("streak") or {}).items()
            if canonical_candidate(c) in CANDIDATES
        }
        sel.brier = {
            canonical_candidate(c): (None if v is None else float(v))
            for c, v in (data.get("brier") or {}).items()
            if canonical_candidate(c) in CANDIDATES
        }
        sel.evaluations = int(data.get("evaluations", 0))
        return sel
