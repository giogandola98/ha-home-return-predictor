"""Learned (stacked) fusion: centred on Bayes, selected only when it helps."""

from __future__ import annotations

import json
import random
from datetime import timedelta

import pytest
from model_helpers import CONFIG, local

from hrp_core import KernelBayesReturnModel
from hrp_core.evidence import LRCalibrator
from hrp_core.fusion import fuse_probability
from hrp_core.maturity import LogEntry
from hrp_core.stacker import CENTRE, Stacker, bayes_from_features, build_features
from hrp_core.synthetic import replay_online, rotating_shift_person


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


def test_learned_fusion_beats_miscalibrated_bayes() -> None:
    tl = rotating_shift_person(days=80, seed=6, drive_by_rate=0.6)
    model = KernelBayesReturnModel(CONFIG)
    model.calibrator = InflatedCalibrator()
    cutoff = tl.absences[0].left_at + timedelta(days=50)
    learned_sq = bayes_sq = 0.0
    n = 0
    for rec in replay_online(model, tl):
        if rec.ts < cutoff:
            continue
        o = 1.0 if rec.truth else 0.0
        learned_sq += (rec.prediction.probability - o) ** 2
        bayes_sq += (rec.prediction.reason["probability_bayes"] - o) ** 2
        n += 1
    st = model.stacker
    assert st.active and model.maturity(tl.absences[-1].returned_at).fusion == "learned"
    assert st.holdout_brier_learned is not None and st.holdout_brier_bayes is not None
    assert st.holdout_brier_learned < st.holdout_brier_bayes
    assert learned_sq / n < bayes_sq / n
    assert st.weights[2] < 1.0  # learned to trust the (inflated) GPS LR less

    restored = KernelBayesReturnModel.from_dict(json.loads(json.dumps(model.to_dict())), CONFIG)
    assert restored.stacker.weights == st.weights and restored.stacker.active
