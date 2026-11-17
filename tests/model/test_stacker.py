"""Learned (stacked) fusion: centred on Bayes, selected only when it helps."""

from __future__ import annotations

import random

import pytest
from model_helpers import local

from hrp_core.evidence import LRCalibrator
from hrp_core.fusion import fuse_probability
from hrp_core.maturity import LogEntry
from hrp_core.stacker import CENTRE, Stacker, bayes_from_features, build_features


def test_untrained_stacker_equals_bayes() -> None:
    st = Stacker()
    rng = random.Random(0)
    for _ in range(200):
        prior, lr = rng.uniform(0, 1), rng.choice([0.05, 0.25, 1.0, 4.0, 20.0])
        f = build_features(
            prior, lr, rng.uniform(0, 900), 90, rng.choice([None, 5000.0]), True, rng.uniform(0, 24)
        )
        assert st.predict(f) == pytest.approx(fuse_probability(prior, lr), abs=1e-12)
        assert bayes_from_features(f) == pytest.approx(fuse_probability(prior, lr), abs=1e-12)


def test_few_entries_keep_bayes() -> None:
    now = local(__import__("datetime").date(2026, 3, 2), 600)
    entries = [
        LogEntry(
            now,
            0.5,
            0.5,
            None,
            now,
            outcome=i % 2,
            features=build_features(0.5, 1.0, 30, 90, None, False, 10),
        )
        for i in range(50)
    ]
    st = Stacker()
    st.fit(entries, now, 28)
    assert not st.active
    assert st.weights == pytest.approx(list(CENTRE), abs=1e-6)  # balanced data: stays centred


class InflatedCalibrator(LRCalibrator):
    """Deliberately miscalibrated GPS: approach LRs 5x too high, never corrected."""

    def likelihood_ratio(self, approach_class: str | None) -> float:
        lr = super().likelihood_ratio(approach_class)
        return min(lr * 5.0, 20.0) if approach_class and "approach" in approach_class else lr

    def update(self, approach_class: str, returned: bool) -> None:
        return None
