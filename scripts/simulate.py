#!/usr/bin/env python3
"""Simulate months of synthetic data and print the model's learning curve.

Standalone: needs no Home Assistant. Usage::

    python scripts/simulate.py --days 120 --seed 1
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "custom_components" / "home_return_predictor")
)

from hrp_core import KernelBayesReturnModel, ModelConfig  # noqa: E402
from hrp_core.synthetic import (  # noqa: E402
    DEFAULT_TZ,
    Timeline,
    fixed_schedule_person,
    replay_online,
    rotating_shift_person,
)

THRESHOLD = 0.6
ROLLING_DAYS = 7


def simulate(timeline: Timeline, config: ModelConfig, step: float) -> list[dict[str, object]]:
    """Replay one person online; return one summary row per local day."""
    model = KernelBayesReturnModel(config)
    per_day: dict[date, list[tuple[float, bool]]] = defaultdict(list)
    reports: dict[date, tuple[float, int, str]] = {}

    def capture(ts: datetime) -> None:
        report = model.maturity(ts)
        reports[ts.astimezone(timeline.tz).date()] = (
            report.maturity,
            report.episodes,
            report.fusion,
        )

    for rec in replay_online(
        model,
        timeline,
        horizon_minutes=config.horizon_minutes,
        step_minutes=step,
        on_absence_end=capture,
    ):
        per_day[rec.ts.astimezone(timeline.tz).date()].append(
            (rec.prediction.probability, rec.truth)
        )

    rows: list[dict[str, object]] = []
    history: list[list[tuple[float, bool]]] = []
    last = (0.0, 0, "bayes")
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
                "fusion": last[2],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--step", type=float, default=5.0, help="minutes between predictions")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args()

    config = ModelConfig(timezone=DEFAULT_TZ)
    people = [
        fixed_schedule_person(days=args.days, seed=args.seed),
        rotating_shift_person(days=args.days, seed=args.seed + 1),
    ]
    started = time.perf_counter()
    results = {tl.name: simulate(tl, config, args.step) for tl in people}

    if not args.quiet:
        print(
            f"{'date':<10} {'person':<6} {'acc@0.6':>7} {'brier7d':>7} {'maturity':>8} "
            f"{'eps':>4} fusion"
        )
        for i in range(max(len(r) for r in results.values())):
            for name, rows in results.items():
                if i < len(rows):
                    r = rows[i]
                    print(
                        f"{r['date']!s:<10} {name:<6} {r['accuracy']:>7.3f} "
                        f"{r['brier']:>7.3f} {r['maturity']:>8.3f} {r['episodes']:>4} "
                        f"{r['fusion']}"
                    )

    print("\nLearning curve summary (accuracy = mean over the 7 days ending at that day)")
    print(f"{'person':<6} {'day':>4} {'acc@0.6':>7} {'brier7d':>7} {'maturity':>8} fusion")
    for name, rows in results.items():
        for day in sorted({d for d in (7, 30, 60, 90, 120) if d <= len(rows)} | {len(rows)}):
            chunk = rows[max(0, day - 7) : day]
            acc = sum(float(r["accuracy"]) for r in chunk) / len(chunk)  # type: ignore[arg-type]
            r = rows[day - 1]
            print(
                f"{name:<6} {day:>4} {acc:>7.3f} {r['brier']:>7.3f} {r['maturity']:>8.3f} "
                f"{r['fusion']}"
            )
    print(
        f"\nsimulated {args.days} days x {len(people)} people in "
        f"{time.perf_counter() - started:.1f} s"
    )


if __name__ == "__main__":
    main()
