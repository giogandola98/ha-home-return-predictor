#!/usr/bin/env python3
"""House-level evaluation of recorded predictor outputs (``hrp_core.house_metric``).

For every period with the house empty (>= ``--min-hours``) the policy "preheat while the
house return probability >= threshold" is scored: degrees missing at the arrival and
hours heated before the useful window. Baselines: never setback, fixed schedule,
oracle. Input = a Home Assistant history export (``GET /api/history/period/...``)
holding ``person.*`` and ``sensor.<person>_return_probability`` (in %); passed as an
argument, nothing personal lives in the repository::

    python scripts/replay/house_eval.py --history history.json --threshold 0.4 0.68

Persons are reported as "person A", "person B", ... (sorted entity ids).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor")
)

from hrp_core.house_metric import (  # noqa: E402
    HouseScore,
    ThermalParams,
    empty_periods,
    fixed_schedule,
    house_probability,
    never_setback,
    oracle,
    score_policy,
    step_value,
    threshold_policy,
)


def load(path: str) -> tuple[dict[str, list], dict[str, list]]:
    """Presence and probability step series per person object id."""
    series = {s[0]["entity_id"]: s for s in json.loads(Path(path).read_text()) if s}
    presence: dict[str, list[tuple[datetime, bool]]] = {}
    probability: dict[str, list[tuple[datetime, float]]] = {}
    for eid in sorted(series):
        if not eid.startswith("person."):
            continue
        slug = eid.split(".", 1)[1]
        presence[slug] = [
            (datetime.fromisoformat(s["last_changed"]), s["state"] == "home")
            for s in series[eid]
            if s["state"] not in ("unknown", "unavailable")
        ]
        prob = []
        for s in series.get(f"sensor.{slug}_return_probability", []):
            try:
                prob.append((datetime.fromisoformat(s["last_changed"]), float(s["state"]) / 100))
            except ValueError:
                continue
        probability[slug] = prob
    return presence, probability


def report(name: str, score: HouseScore, labels: Callable[[datetime], str], verbose: bool) -> None:
    print(
        f"{name:<28} periods {score.n}  deficit total {score.total_deficit_k:5.1f} K "
        f"(mean {score.mean_deficit_k:4.2f} K)  early {score.early_hours:5.1f} h  "
        f"on {score.on_hours:5.1f} h"
    )
    if verbose:
        for p in score.periods:
            print(
                f"    {labels(p.period.start)} {p.period.hours:5.1f} h  cool {p.cooling_k:4.1f} K "
                f"need {p.need_hours:4.1f} h  lead {p.lead_minutes:4.0f} min  "
                f"deficit {p.deficit_k:4.1f} K  early {p.early_hours:4.1f} h"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", required=True)
    ap.add_argument("--threshold", type=float, nargs="+", default=[0.4, 0.68])
    ap.add_argument("--rate", type=float, default=ThermalParams.rate_k_per_hour)
    ap.add_argument("--tau", type=float, default=ThermalParams.tau_hours)
    ap.add_argument("--delta-t", type=float, default=ThermalParams.delta_t_k)
    ap.add_argument("--min-hours", type=float, default=3.0)
    ap.add_argument("--max-hours", type=float, default=None, help="drop longer periods (trips)")
    ap.add_argument("--utc-offset", type=float, default=1.0, help="local time for the labels")
    ap.add_argument("--fixed", default="16:00-23:00", help="fixed-schedule baseline window")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    params = ThermalParams(args.rate, args.tau, args.delta_t)
    tz = timezone(timedelta(hours=args.utc_offset))
    presence, probability = load(args.history)
    periods = [
        p
        for p in empty_periods(presence, min_hours=args.min_hours)
        if args.max_hours is None or p.hours <= args.max_hours
    ]
    home = {who: step_value(s, True) for who, s in presence.items()}
    prob = {who: step_value(s, 0.0) for who, s in probability.items()}
    p_house = house_probability(prob, home)

    def labels(t: datetime) -> str:
        return t.astimezone(tz).strftime("%a %H:%M")

    print(
        f"thermal model: {params.rate_k_per_hour} K/h, tau {params.tau_hours} h, "
        f"dT {params.delta_t_k} K; {len(periods)} empty periods >= {args.min_hours} h"
    )
    for thr in args.threshold:
        report(
            f"p_house >= {thr}",
            score_policy(periods, threshold_policy(p_house, thr), params),
            labels,
            args.verbose,
        )
    a, b = (int(x[:2]) * 60 + int(x[3:]) for x in args.fixed.split("-"))
    report("never setback", score_policy(periods, never_setback, params), labels, args.verbose)
    report(
        f"fixed {args.fixed}",
        score_policy(periods, fixed_schedule(a, b, tz), params),
        labels,
        args.verbose,
    )
    report("oracle", score_policy(periods, oracle(periods, params), params), labels, args.verbose)


if __name__ == "__main__":
    main()
