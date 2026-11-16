#!/usr/bin/env python3
"""House metric of the rule "preheat when now >= q(first arrival) - L" on real presence.

The per-person models are rebuilt from a Home Assistant history export (``person.*``
states with their GPS attributes) and replayed on every empty-house period of at least
``--min-hours``. ``L`` is the warm-up the room needs if the heating starts now
(``cooling so far / rate``, what a thermal model would report). Training data:

* ``loo`` (default): every absence of the history except the one being scored
  (short histories: a few weeks hold too few absences for an online replay);
* ``online``: only the absences that ended before the scored period.

Inputs (passed as arguments, nothing personal in the repository)::

    python scripts/replay/arrival_eval.py --states states.json --history history.json

``states.json`` = ``GET /api/states`` (``zone.home`` coordinates). Periods are labelled
by weekday and local time only; persons never appear in the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor")
)

from hrp_core import KernelBayesReturnModel, ModelConfig, ModelInput, PositionFix  # noqa: E402
from hrp_core.arrival import first_arrival  # noqa: E402
from hrp_core.base import PresenceTransition  # noqa: E402
from hrp_core.evidence import haversine_m  # noqa: E402
from hrp_core.house_metric import (  # noqa: E402
    HouseScore,
    ThermalParams,
    empty_periods,
    score_period,
)

STEP_MINUTES = 5.0


def load(states_path: str, history_path: str, tz: timezone):
    states = {s["entity_id"]: s for s in json.loads(Path(states_path).read_text())}
    home = states["zone.home"]["attributes"]
    lat0, lon0 = home["latitude"], home["longitude"]
    persons = {}
    for series in json.loads(Path(history_path).read_text()):
        if not series or not series[0]["entity_id"].startswith("person."):
            continue
        changes, fixes = [], []
        for s in series:
            if s["state"] in ("unknown", "unavailable"):
                continue
            ts = datetime.fromisoformat(s.get("last_updated") or s["last_changed"])
            is_home = s["state"] == "home"
            if not changes or changes[-1][1] != is_home:
                changes.append((ts, is_home))
            a = s.get("attributes") or {}
            if not is_home and a.get("latitude") is not None:
                dist = haversine_m(lat0, lon0, a["latitude"], a["longitude"])
                fixes.append(PositionFix(ts, dist, a.get("gps_accuracy")))
        persons[series[0]["entity_id"]] = (changes, sorted(fixes, key=lambda f: f.ts))
    return persons


def absences(changes):
    out, left = [], None
    for ts, is_home in changes:
        if not is_home and left is None:
            left = ts
        elif is_home and left is not None:
            out.append((left, ts))
            left = None
    return out


def train(changes, exclude: tuple[datetime, datetime], online: bool, tz) -> KernelBayesReturnModel:
    model = KernelBayesReturnModel(ModelConfig(timezone=tz))
    transitions = []
    for left, back in absences(changes):
        if left <= exclude[1] and back >= exclude[0]:
            continue  # the absence being scored
        if online and back > exclude[0]:
            continue
        transitions += [PresenceTransition(left, False), PresenceTransition(back, True)]
    model.load_history(transitions)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--quantile", type=float, nargs="+", default=[0.25])
    ap.add_argument("--mode", choices=("loo", "online"), default="loo")
    ap.add_argument("--rate", type=float, default=ThermalParams.rate_k_per_hour)
    ap.add_argument("--tau", type=float, default=ThermalParams.tau_hours)
    ap.add_argument("--delta-t", type=float, default=ThermalParams.delta_t_k)
    ap.add_argument("--min-hours", type=float, default=3.0)
    ap.add_argument("--trip-hours", type=float, default=36.0, help="longer periods = trips")
    ap.add_argument("--utc-offset", type=float, default=1.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    tz = timezone(timedelta(hours=args.utc_offset))
    params = ThermalParams(args.rate, args.tau, args.delta_t)
    persons = load(args.states, args.history, tz)
    periods = empty_periods({p: c for p, (c, _) in persons.items()}, min_hours=args.min_hours)
    print(
        f"thermal model: {params.rate_k_per_hour} K/h, tau {params.tau_hours} h, "
        f"dT {params.delta_t_k} K; {len(periods)} empty periods >= {args.min_hours} h; "
        f"training {args.mode}"
    )
    flags: dict[float, list[list[bool]]] = {q: [] for q in args.quantile}
    for period in periods:
        runs = {}
        for pid, (changes, fixes) in persons.items():
            ab = next((a for a in absences(changes) if a[0] <= period.start < a[1]), None)
            if ab is None:
                continue
            runs[pid] = (train(changes, ab, args.mode == "online", tz), ab[0], fixes)
        per_q: dict[float, list[bool]] = {q: [] for q in args.quantile}
        t = period.start
        while t < period.end:
            dists = []
            for model, left, fixes in runs.values():
                recent = [f for f in fixes if t - timedelta(minutes=30) <= f.ts <= t]
                older = [f for f in fixes if f.ts <= t and f.ts >= left]
                inp = ModelInput(
                    now=t,
                    is_home=False,
                    left_at=left,
                    fixes=recent,
                    last_fix=older[-1] if older else None,
                )
                dist = model.arrival(inp, model.predict(inp))
                if dist is not None:
                    dists.append(dist)
            house = first_arrival(dists)
            need_min = (
                params.cooling_k((t - period.start).total_seconds() / 3600)
                / params.rate_k_per_hour
                * 60
            )
            for q in args.quantile:
                m = house.quantile(q) if house is not None else None
                per_q[q].append(m is not None and m <= need_min)
            t += timedelta(minutes=STEP_MINUTES)
        for q in args.quantile:
            flags[q].append(per_q[q])

    for q in args.quantile:
        trips, other = HouseScore(), HouseScore()
        for period, fl in zip(periods, flags[q], strict=True):
            start = period.start

            def policy(t: datetime, fl=fl, start=start) -> bool:
                i = int((t - start).total_seconds() // (STEP_MINUTES * 60))
                return fl[min(i, len(fl) - 1)]

            s = score_period(period, policy, params, step_minutes=STEP_MINUTES)
            (trips if period.hours > args.trip_hours else other).periods.append(s)
            if args.verbose:
                print(
                    f"    q{q:.2f} {start.astimezone(tz).strftime('%a %H:%M')} "
                    f"{period.hours:5.1f} h  need {s.need_hours:4.1f} h  lead "
                    f"{s.lead_minutes:4.0f} min  deficit {s.deficit_k:4.1f} K  "
                    f"early {s.early_hours:4.1f} h"
                )
        print(
            f"now >= q{q:.2f} - L: {other.n} periods (no trips): deficit total "
            f"{other.total_deficit_k:.1f} K, early {other.early_hours:.1f} h | "
            f"{trips.n} trips: deficit {trips.total_deficit_k:.1f} K, "
            f"early {trips.early_hours:.1f} h"
        )


if __name__ == "__main__":
    main()
