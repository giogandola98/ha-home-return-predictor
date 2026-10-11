#!/usr/bin/env python3
"""Simulate months of synthetic data and print the model's learning curve.

Standalone: needs no Home Assistant. Usage::

    python scripts/simulate.py --days 120 --seed 1
    python scripts/simulate.py --sklearn   # also train the scikit-learn challenger
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "custom_components" / "home_return_predictor")
)

from hrp_core import KernelBayesReturnModel, ModelConfig, sklearn_available  # noqa: E402
from hrp_core.synthetic import (  # noqa: E402
    DEFAULT_TZ,
    Timeline,
    fixed_schedule_person,
    replay_online,
    rotating_shift_person,
)

THRESHOLD = 0.6
ROLLING_DAYS = 7


def simulate(
    timeline: Timeline,
    config: ModelConfig,
    step: float,
    use_sklearn: bool = False,
    stats: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    """Replay one person online; return one summary row per local day."""
    model = KernelBayesReturnModel(config)
    model.enable_challenger(use_sklearn)
    per_day: dict[date, list[tuple[float, bool]]] = defaultdict(list)
    reports: dict[date, tuple[float, int, str]] = {}
    stats = stats if stats is not None else {}

    def maybe_train(ts: datetime) -> None:
        if use_sklearn and model.challenger_due(ts):
            summary = model.train_challenger(ts)
            if summary["trained"]:
                stats["trainings"] = stats.get("trainings", 0) + 1
                stats["train_seconds"] = stats.get("train_seconds", 0.0) + summary["seconds"]
                stats["max_train_seconds"] = max(
                    stats.get("max_train_seconds", 0.0), summary["seconds"]
                )

    def capture(ts: datetime) -> None:
        maybe_train(ts)
        report = model.maturity(ts)
        reports[ts.astimezone(timeline.tz).date()] = (
            report.maturity,
            report.episodes,
            report.active_model,
        )

    for rec in replay_online(
        model,
        timeline,
        horizon_minutes=config.horizon_minutes,
        step_minutes=step,
        on_absence_end=capture,
    ):
        if rec.ts.minute < step:  # roughly hourly while away
            maybe_train(rec.ts)
        per_day[rec.ts.astimezone(timeline.tz).date()].append(
            (rec.prediction.probability, rec.truth)
        )

    stats["state_bytes"] = len(json.dumps(model.to_dict()))
    rows: list[dict[str, object]] = []
    history: list[list[tuple[float, bool]]] = []
    last = (0.0, 0, "kernel_bayes")
    for day in sorted(per_day):
        recs = per_day[day]
        history.append(recs)
        window = [r for d in history[-ROLLING_DAYS:] for r in d]
        last = reports.get(day, last)
        acc = sum((p >= THRESHOLD) == t for p, t in recs) / len(recs)
        brier = sum((p - t) ** 2 for p, t in window) / len(window)
        rows.append(
            {
                "date": day,
                "accuracy": acc,
                "brier": brier,
                "maturity": last[0],
                "episodes": last[1],
                "active": last[2],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=float, default=5.0, help="minutes between predictions")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    parser.add_argument(
        "--sklearn", action="store_true", help="train the scikit-learn challenger when due"
    )
    args = parser.parse_args()
    if args.sklearn and not sklearn_available():
        parser.error("--sklearn needs scikit-learn installed")

    config = ModelConfig(timezone=DEFAULT_TZ)
    people = [
        fixed_schedule_person(days=args.days, seed=args.seed),
        rotating_shift_person(days=args.days, seed=args.seed + 1),
    ]
    started = time.perf_counter()
    stats: dict[str, dict[str, float]] = {tl.name: {} for tl in people}
    results = {
        tl.name: simulate(tl, config, args.step, args.sklearn, stats[tl.name]) for tl in people
    }

    if not args.quiet:
        print(
            f"{'date':<10} {'person':<6} {'acc@0.6':>7} {'brier7d':>7} {'maturity':>8} "
            f"{'eps':>4} active"
        )
        for i in range(max(len(r) for r in results.values())):
            for name, rows in results.items():
                if i < len(rows):
                    r = rows[i]
                    print(
                        f"{r['date']!s:<10} {name:<6} {r['accuracy']:>7.3f} "
                        f"{r['brier']:>7.3f} {r['maturity']:>8.3f} {r['episodes']:>4} "
                        f"{r['active']}"
                    )

    print("\nLearning curve summary (accuracy = mean over the 7 days ending at that day)")
    print(f"{'person':<6} {'day':>4} {'acc@0.6':>7} {'brier7d':>7} {'maturity':>8} active")
    for name, rows in results.items():
        for day in sorted({d for d in (7, 30, 60, 90, 120) if d <= len(rows)} | {len(rows)}):
            chunk = rows[max(0, day - 7) : day]
            acc = sum(float(r["accuracy"]) for r in chunk) / len(chunk)  # type: ignore[arg-type]
            r = rows[day - 1]
            print(
                f"{name:<6} {day:>4} {acc:>7.3f} {r['brier']:>7.3f} {r['maturity']:>8.3f} "
                f"{r['active']}"
            )
    if args.sklearn:
        for name, st in stats.items():
            n = int(st.get("trainings", 0))
            mean = st.get("train_seconds", 0.0) / n if n else 0.0
            print(
                f"{name}: {n} challenger trainings, mean {mean:.2f} s, "
                f"max {st.get('max_train_seconds', 0.0):.2f} s"
            )
    for name, st in stats.items():
        print(f"{name}: persisted state {st['state_bytes'] / 1000:.0f} KB (to_dict JSON)")
    print(
        f"\nsimulated {args.days} days x {len(people)} people in "
        f"{time.perf_counter() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
