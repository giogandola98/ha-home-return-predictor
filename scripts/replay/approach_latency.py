#!/usr/bin/env python3
"""Minutes from the start of the drive home to the first approach class (synthetic).

For every drive home of at least ``--min-drive`` minutes the GPS evidence is evaluated
every minute with the integration's fix buffer (last 30 min plus the newest older fix)
and the first time an approach class appears is recorded. Needs no data::

    python scripts/replay/approach_latency.py --days 60 --style iphone
"""

from __future__ import annotations

import argparse
import statistics
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor")
)

from hrp_core import evidence  # noqa: E402
from hrp_core.evidence import APPROACH_CLASSES, compute_evidence  # noqa: E402
from hrp_core.synthetic import (  # noqa: E402
    Timeline,
    couple,
    fixed_schedule_person,
    iphone_fixes,
    workplace_zone,
)


def latencies(tl: Timeline, min_drive: float) -> tuple[list[float], int]:
    """First-approach latency per drive home (minutes) and drives never detected."""
    out, missed = [], 0
    lookback = timedelta(minutes=30)
    for ab, start in zip(tl.absences, tl.homeward, strict=False):
        drive = (ab.returned_at - start).total_seconds() / 60.0
        if drive < min_drive or start <= ab.left_at:
            continue
        t, found = start, None
        while t < ab.returned_at:
            ev = compute_evidence(tl.recent_fixes(t, lookback, ab.left_at), t, 10.0, 250.0)
            if ev.approach_class in APPROACH_CLASSES:
                found = (t - start).total_seconds() / 60.0
                break
            t += timedelta(minutes=1)
        if found is None:
            missed += 1
        else:
            out.append(found)
    return out, missed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--style", choices=("iphone", "android"), default="iphone")
    ap.add_argument("--min-drive", type=float, default=12.0)
    ap.add_argument("--no-departure", action="store_true", help="disable the silence rule")
    args = ap.parse_args()
    if args.no_departure:
        evidence.DEPARTURE_SILENCE_MINUTES = float("inf")
    people = [
        fixed_schedule_person(days=args.days, seed=args.seed),
        workplace_zone(days=args.days, seed=args.seed + 1, style="android"),
        *couple(days=args.days, seed=args.seed + 2, style="android"),
    ]
    if args.style == "iphone":
        people = [iphone_fixes(p, seed=args.seed + 10 + i) for i, p in enumerate(people)]
    all_lat, all_missed = [], 0
    for tl in people:
        lat, missed = latencies(tl, args.min_drive)
        all_lat += lat
        all_missed += missed
        if lat:
            med, p90 = statistics.median(lat), sorted(lat)[int(0.9 * (len(lat) - 1))]
            print(
                f"{tl.name:<10} drives {len(lat) + missed:3d}  median {med:4.1f} min  "
                f"p90 {p90:4.1f} min  missed {missed}"
            )
    print(
        f"all        drives {len(all_lat) + all_missed:3d}  median "
        f"{statistics.median(all_lat):4.1f} min  missed {all_missed}"
    )


if __name__ == "__main__":
    main()
