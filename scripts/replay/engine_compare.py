#!/usr/bin/env python3
"""Compare the serving engines on the house metric with synthetic households.

One model per person is replayed online (like the integration: transitions, predict,
log, resolve, background challenger training when due). At every step the probability
of each engine is read from the same trained model by switching the served candidate:

* ``kernel_bayes``   - kernel prior x calibrated GPS likelihood ratio;
* ``learned_fusion`` - the stacker weights currently installed;
* ``gbm``            - the gradient-boosting challenger (once trained; numpy needed).

The house is scored with ``hrp_core.house_metric`` (policy "on while the house
probability >= threshold") on the empty periods after the warm-up, plus per-person
metrics of the binary signal ``p >= --signal``. Needs no Home Assistant and no data::

    python scripts/replay/engine_compare.py --days 150 --warmup 60 --seed 21
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor")
)

from hrp_core import KernelBayesReturnModel, ModelConfig, ModelInput, numpy_available  # noqa: E402
from hrp_core.base import PresenceTransition  # noqa: E402
from hrp_core.episodes import MIN_HOME_STAY_MINUTES  # noqa: E402
from hrp_core.house_metric import (  # noqa: E402
    ThermalParams,
    empty_periods,
    house_probability,
    score_person,
    score_policy,
    step_value,
    threshold_policy,
)
from hrp_core.selection import GBM, KERNEL_BAYES, LEARNED_FUSION  # noqa: E402
from hrp_core.synthetic import DEFAULT_TZ, Timeline, couple, trip  # noqa: E402

ENGINES = (KERNEL_BAYES, LEARNED_FUSION, GBM)


def replay(
    timeline: Timeline, config: ModelConfig, step_minutes: float, use_gbm: bool
) -> dict[str, list[tuple[datetime, float]]]:
    """Online replay; probability series per engine (only while away)."""
    model = KernelBayesReturnModel(config)
    model.enable_challenger(use_gbm)
    series: dict[str, list[tuple[datetime, float]]] = {e: [] for e in ENGINES}
    step = timedelta(minutes=step_minutes)
    lookback = timedelta(minutes=30)

    def maybe_train(ts: datetime) -> None:
        if use_gbm and model.challenger_due(ts):
            model.train_challenger(ts)

    for ab in timeline.absences:
        model.observe_transition(PresenceTransition(ab.left_at, False))
        t = ab.left_at + step
        while t < ab.returned_at:
            inp = ModelInput(
                now=t,
                is_home=False,
                left_at=ab.left_at,
                fixes=timeline.fixes_between(t - lookback, t),
            )
            champion = model.selector.champion
            for engine in ENGINES:
                if engine == GBM and not model._challenger.trained:
                    continue
                model.selector.champion = engine
                series[engine].append((t, model.predict(inp).probability))
            model.selector.champion = champion
            pred = model.predict(inp)
            model.log_prediction(inp, pred)
            model.resolve(t, False)
            if t.minute < step_minutes:
                maybe_train(t)
            t += step
        series_end = ab.returned_at
        for engine in ENGINES:  # back home: the engine's value no longer matters
            series[engine].append((series_end, 0.0))
        model.observe_transition(PresenceTransition(ab.returned_at, True))
        model.resolve(ab.returned_at + timedelta(minutes=MIN_HOME_STAY_MINUTES), True)
        maybe_train(ab.returned_at)
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=60, help="days not scored")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--together", type=float, default=0.4)
    ap.add_argument("--step", type=float, default=5.0)
    ap.add_argument("--trip-day", type=int, default=100, help="<0: no trip")
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.25, 0.4, 0.6, 0.68])
    ap.add_argument("--signal", type=float, default=0.6, help="per-person returning threshold")
    ap.add_argument("--no-gbm", action="store_true")
    args = ap.parse_args()
    use_gbm = not args.no_gbm and numpy_available()

    tz = DEFAULT_TZ
    a, b = couple(days=args.days, seed=args.seed, together_share=args.together)
    if args.trip_day >= 0:
        start = a.absences[0].left_at.replace(hour=7, minute=0) + timedelta(days=args.trip_day)
        a, b = trip(a, start, seed=args.seed, days=3.0), trip(b, start, seed=args.seed, days=3.0)
    people = {"person A": a, "person B": b}
    config = ModelConfig(timezone=tz)
    started = time.perf_counter()
    series = {name: replay(tl, config, args.step, use_gbm) for name, tl in people.items()}
    elapsed = time.perf_counter() - started

    scored_from = a.absences[0].left_at + timedelta(days=args.warmup)
    presence = {name: [(t.ts, t.is_home) for t in tl.transitions] for name, tl in people.items()}
    periods = [p for p in empty_periods(presence, min_hours=3.0) if p.start >= scored_from]
    home = {name: step_value(s, True) for name, s in presence.items()}
    params = ThermalParams()
    print(
        f"{args.days} days (scored after day {args.warmup}), {len(periods)} empty periods "
        f">= 3 h, replay {elapsed:.0f} s, gbm {'on' if use_gbm else 'off'}"
    )
    print(f"{'engine':<15} {'thr':>5} {'deficit K':>9} {'early h':>8} {'on h':>6}")
    for engine in ENGINES:
        if not all(series[n][engine] for n in people):
            continue
        # GBM is only defined once trained: score from its first value on
        first = max(series[n][engine][0][0] for n in people)
        sel = [p for p in periods if p.start >= first]
        if len(sel) < len(periods):
            print(f"  ({engine}: {len(periods) - len(sel)} early periods skipped, not trained)")
        prob = {n: step_value(series[n][engine], 0.0) for n in people}
        p_house = house_probability(prob, home)
        for thr in args.threshold:
            s = score_policy(sel, threshold_policy(p_house, thr), params)
            print(
                f"{engine:<15} {thr:>5.2f} {s.total_deficit_k:>9.1f} {s.early_hours:>8.1f} "
                f"{s.on_hours:>6.1f}"
            )
    print(f"\nper person, signal p >= {args.signal} (absences >= 30 min after the warm-up)")
    for name, tl in people.items():
        absences = [
            (x.left_at, x.returned_at)
            for x in tl.absences
            if x.left_at >= scored_from and x.kind != "bounce"
        ]
        for engine in ENGINES:
            if not series[name][engine]:
                continue
            f = step_value(series[name][engine], 0.0)
            ps = score_person(absences, lambda t, f=f: f(t) >= args.signal, min_minutes=30)
            print(
                f"  {name} {engine:<15} absences {len(ps.absences):3d}  missed {ps.missed:3d}  "
                f"median lead {ps.median_lead_minutes or 0:5.0f} min  "
                f"false alarms/day {ps.false_alarms_per_day:4.2f}"
            )


if __name__ == "__main__":
    main()
