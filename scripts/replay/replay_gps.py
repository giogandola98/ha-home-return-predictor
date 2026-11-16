#!/usr/bin/env python3
"""Replay recorded GPS fixes and count the updates that see an approach.

Compares the plain regression window with the sparse-reporting fallback
(``EXTENDED_WINDOW_FACTOR``). Inputs are exports from Home Assistant's REST API,
passed as arguments; no personal data lives in the repository::

    python scripts/replay/replay_gps.py --states states.json --history history.json \
        --start 2026-01-01T18:00:00+01:00 --end 2026-01-01T19:00:00+01:00

``states.json`` = ``GET /api/states`` (only ``zone.home`` is read);
``history.json`` = ``GET /api/history/period/...`` (lists of states per entity; only
``device_tracker.*`` entities with coordinates are used). Trackers are reported as
"person A", "person B", ... in input order.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "custom_components" / "home_return_predictor")
)

import hrp_core.evidence as evidence  # noqa: E402
from hrp_core.base import PositionFix  # noqa: E402
from hrp_core.evidence import compute_evidence, haversine_m  # noqa: E402


def load_fixes(states_path: str, history_path: str) -> list[list[PositionFix]]:
    states = {s["entity_id"]: s for s in json.loads(Path(states_path).read_text())}
    home = states["zone.home"]["attributes"]
    lat0, lon0 = home["latitude"], home["longitude"]
    out = []
    for series in json.loads(Path(history_path).read_text()):
        if not series or not series[0]["entity_id"].startswith("device_tracker."):
            continue
        fixes = []
        for s in series:
            a = s.get("attributes") or {}
            if s["state"] == "home" or a.get("latitude") is None:
                continue
            fixes.append(
                PositionFix(
                    datetime.fromisoformat(s["last_updated"]),
                    haversine_m(lat0, lon0, a["latitude"], a["longitude"]),
                    a.get("gps_accuracy"),
                )
            )
        if fixes:
            out.append(sorted(fixes, key=lambda f: f.ts))
    return out


def approach_seen(fixes: list[PositionFix], now: datetime, factor: float, window: float) -> bool:
    evidence.EXTENDED_WINDOW_FACTOR = factor
    recent = [f for f in fixes if f.ts <= now and (now - f.ts).total_seconds() <= 1800]
    cls = compute_evidence(recent, now, window, 250.0).approach_class
    return cls is not None and cls.startswith("approach")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states", required=True)
    ap.add_argument("--history", required=True)
    ap.add_argument("--start", required=True, type=datetime.fromisoformat)
    ap.add_argument("--end", required=True, type=datetime.fromisoformat)
    ap.add_argument("--step-minutes", type=float, default=2.0)
    ap.add_argument("--window-minutes", type=float, default=10.0)
    args = ap.parse_args()
    default_factor = evidence.EXTENDED_WINDOW_FACTOR
    for k, fixes in enumerate(load_fixes(args.states, args.history)):
        t, total, old, new = args.start, 0, 0, 0
        while t < args.end:
            total += 1
            old += approach_seen(fixes, t, 1.0, args.window_minutes)
            new += approach_seen(fixes, t, default_factor, args.window_minutes)
            t += timedelta(minutes=args.step_minutes)
        print(
            f"person {chr(65 + k)}: {total} updates, approach seen "
            f"{old} (plain window) -> {new} (sparse fallback)"
        )


if __name__ == "__main__":
    main()
