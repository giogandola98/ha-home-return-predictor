"""Truth-table tests for the blueprint reference logic (hrp_core.preset_logic)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from hrp_core.preset_logic import (
    Decision,
    PresetConfig,
    Thresholds,
    ZoneInputs,
    decide_preset,
    preset_rank,
    schedule_active_at,
)

D = datetime(2026, 10, 12, tzinfo=UTC)  # a Monday, midnight


def at(hh: int, mm: int = 0, day: int = 0) -> datetime:
    return D + timedelta(days=day, hours=hh, minutes=mm)


# Schedules expressed as an HA schedule helper would expose them at a given "now".
def sched(now: datetime, blocks: list[tuple[datetime, datetime]]) -> tuple[bool, datetime | None]:
    """(state_on, next_event) for a schedule made of ``blocks`` (sorted, non overlapping)."""
    for start, end in blocks:
        if start <= now < end:
            return True, end
        if now < start:
            return False, start
    return False, None


def bedroom_blocks() -> list[tuple[datetime, datetime]]:
    # night zone: evening/night 20:00 -> 07:00
    return [(at(-4), at(7)), (at(20), at(7, day=1)), (at(20, day=1), at(7, day=2))]


def living_blocks() -> list[tuple[datetime, datetime]]:
    # day zone: day/evening 07:00 -> 23:00
    return [(at(7), at(23)), (at(7, day=1), at(23, day=1))]


# --------------------------------------------------------------------------- schedule


@pytest.mark.parametrize(
    ("state_on", "next_event", "when", "expected"),
    [
        (True, at(23), at(22), True),
        (True, at(23), at(23), False),
        (True, at(23), at(23, 30), False),
        (True, None, at(23, 30), True),
        (False, at(20), at(19), False),
        (False, at(20), at(20), True),
        (False, at(20), at(23), True),  # approximation: block assumed to last
        (False, None, at(23), False),
    ],
)
def test_schedule_active_at(state_on, next_event, when, expected):
    assert schedule_active_at(state_on, next_event, when) is expected


def test_preset_rank_order():
    p = PresetConfig()
    assert preset_rank("frost", p) == preset_rank("away", p) < preset_rank("eco", p)
    assert preset_rank("eco", p) < preset_rank("comfort", p) < preset_rank("boost", p)
    assert preset_rank(None, p) < preset_rank("frost", p)
    custom = PresetConfig(comfort="cosy", medium="saving", absent="hors_gel")
    assert (
        preset_rank("hors_gel", custom)
        < preset_rank("saving", custom)
        < preset_rank("cosy", custom)
    )


# --------------------------------------------------------------------------- main table


def inp(**kw) -> ZoneInputs:
    kw.setdefault("now", at(12))
    return ZoneInputs(**kw)


CASES = [
    # id, inputs, current, expected preset
    ("nobody_no_eta", inp(), None, "frost"),
    ("nobody_eta_far", inp(house_probability=0.9, house_eta=at(14)), None, "frost"),
    (
        "nobody_eta_in_window_high",
        inp(house_probability=0.9, house_eta=at(12, 20)),
        None,
        "comfort",
    ),
    ("nobody_eta_in_window_medium", inp(house_probability=0.5, house_eta=at(12, 20)), None, "eco"),
    ("nobody_eta_in_window_low", inp(house_probability=0.2, house_eta=at(12, 20)), None, "frost"),
    ("window_edge_exact", inp(house_probability=0.9, house_eta=at(12, 30)), None, "comfort"),
    ("window_edge_before", inp(house_probability=0.9, house_eta=at(12, 31)), None, "frost"),
    ("late_eta_still_likely", inp(house_probability=0.8, house_eta=at(11, 0)), None, "comfort"),
    (
        "preheat_60",
        inp(house_probability=0.8, house_eta=at(13), preheat_minutes=60),
        None,
        "comfort",
    ),
    ("home_no_schedule", inp(someone_home=True), None, "comfort"),
    ("home_schedule_active", inp(someone_home=True, schedule=(True, at(23))), None, "comfort"),
    ("home_schedule_inactive", inp(someone_home=True, schedule=(False, at(18))), None, "eco"),
    (
        "home_unused_custom",
        inp(someone_home=True, schedule=(False, at(18)), presets=PresetConfig(unused="frost")),
        None,
        "frost",
    ),
    (
        "home_occupancy_overrides_schedule",
        inp(someone_home=True, schedule=(False, at(18)), occupancy=True),
        None,
        "comfort",
    ),
    ("nobody_occupancy_ignored", inp(occupancy=True), None, "frost"),
    ("protection", inp(someone_home=True, protection_active=True), "eco", None),
    (
        "protection_over_return",
        inp(protection_active=True, house_probability=0.99, house_eta=at(12, 5)),
        "frost",
        None,
    ),
    ("bed_day_zone", inp(someone_home=True, bed_mode=True, zone_type="day"), None, "eco"),
    (
        "bed_day_zone_custom_night",
        inp(someone_home=True, bed_mode=True, zone_type="day", presets=PresetConfig(night="frost")),
        None,
        "frost",
    ),
    (
        "bed_night_zone",
        inp(someone_home=True, bed_mode=True, zone_type="night", schedule=(False, at(20))),
        None,
        "comfort",
    ),
    ("bed_always_zone", inp(someone_home=True, bed_mode=True, zone_type="always"), None, "comfort"),
    (
        "bed_beats_return",
        inp(
            someone_home=True,
            bed_mode=True,
            zone_type="day",
            house_probability=0.95,
            house_eta=at(12, 10),
        ),
        None,
        "eco",
    ),
    ("bed_ignored_when_nobody_home", inp(bed_mode=True, zone_type="night"), None, "frost"),
    (
        "never_downgrade_low_return",
        inp(someone_home=True, house_probability=0.1, house_eta=at(12, 10)),
        "comfort",
        "comfort",
    ),
    (
        "never_downgrade_medium_return_unused",
        inp(
            someone_home=True,
            schedule=(False, at(18)),
            house_probability=0.1,
            house_eta=at(12, 10),
            presets=PresetConfig(unused="eco"),
        ),
        "eco",
        "eco",
    ),
    (
        "upgrade_unused_on_return",
        inp(
            someone_home=True,
            schedule=(True, at(23)),
            occupancy=False,
            house_probability=0.9,
            house_eta=at(12, 10),
        ),
        None,
        "comfort",
    ),
    (
        "prediction_disabled_nobody",
        inp(house_probability=0.95, house_eta=at(12, 10), prediction_enabled=False),
        None,
        "frost",
    ),
    (
        "prediction_disabled_home_logic_kept",
        inp(someone_home=True, prediction_enabled=False),
        None,
        "comfort",
    ),
    ("custom_absent", inp(presets=PresetConfig(absent="away")), None, "away"),
]


@pytest.mark.parametrize(
    ("inputs", "current", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_truth_table(inputs, current, expected):
    decision = decide_preset(inputs, current)
    assert isinstance(decision, Decision)
    assert decision.preset == expected, decision.reason
    assert decision.reason


def test_schedule_checked_at_eta_not_now():
    # nobody home, zone schedule off now (starts 13:00), ETA 13:10 -> scheduled at ETA
    d = decide_preset(
        inp(now=at(12, 45), house_probability=0.9, house_eta=at(13, 10), schedule=(False, at(13))),
        "frost",
    )
    assert d.preset == "comfort"
    # schedule on now but ends before ETA -> no preheat
    d = decide_preset(
        inp(now=at(12, 45), house_probability=0.9, house_eta=at(13, 10), schedule=(True, at(13))),
        "frost",
    )
    assert d.preset == "frost"
    assert "not scheduled" in d.reason


def test_late_eta_uses_now_for_schedule():
    # ETA 22:50 already past at 23:05; living room schedule ended 23:00 -> no preheat
    now = at(23, 5)
    d = decide_preset(
        inp(
            now=now,
            zone_type="day",
            house_probability=0.9,
            house_eta=at(22, 50),
            schedule=sched(now, living_blocks()),
        ),
        "frost",
    )
    assert d.preset == "frost"


# --------------------------------------------------------------------------- 23:30 example


@pytest.mark.parametrize("someone_home", [False, True])
def test_return_at_2330_bedroom_preheats_living_room_does_not(someone_home):
    eta = at(23, 30)
    for now, expect_bedroom in [(at(22, 30), False), (at(23, 0), True), (at(23, 10), True)]:
        bedroom = decide_preset(
            inp(
                now=now,
                zone_type="night",
                someone_home=someone_home,
                house_probability=0.85,
                house_eta=eta,
                preheat_minutes=30,
                schedule=sched(now, bedroom_blocks()),
            ),
            "eco",
        )
        living = decide_preset(
            inp(
                now=now,
                zone_type="day",
                someone_home=someone_home,
                house_probability=0.85,
                house_eta=eta,
                preheat_minutes=30,
                schedule=sched(now, living_blocks()),
            ),
            "eco",
        )
        if someone_home:
            # bedroom schedule (20-07) is active anyway; living room follows its own
            # schedule: comfort until 23:00, then unused (eco) - never the ETA.
            assert bedroom.preset == "comfort"
            assert living.preset == ("comfort" if now < at(23) else "eco"), living.reason
        else:
            assert bedroom.preset == ("comfort" if expect_bedroom else "frost"), bedroom.reason
            assert living.preset == "frost", living.reason


def test_bedroom_without_people_and_schedule_not_yet_on():
    # bedroom schedule starts 20:00, return predicted 20:15, now 19:50 -> preheat
    now = at(19, 50)
    d = decide_preset(
        inp(
            now=now,
            zone_type="night",
            house_probability=0.75,
            house_eta=at(20, 15),
            schedule=sched(now, bedroom_blocks()),
        ),
        "frost",
    )
    assert d.preset == "comfort"


# --------------------------------------------------------------------------- hysteresis


def _nobody(p: float) -> ZoneInputs:
    return inp(house_probability=p, house_eta=at(12, 10))


@pytest.mark.parametrize(
    ("p", "current", "expected"),
    [
        (0.69, None, "eco"),
        (0.69, "eco", "eco"),
        (0.69, "comfort", "comfort"),  # within high - hysteresis
        (0.651, "comfort", "comfort"),
        (0.64, "comfort", "eco"),
        (0.39, "frost", "frost"),
        (0.39, "eco", "eco"),
        (0.351, "eco", "eco"),
        (0.34, "eco", "frost"),
        (0.36, "comfort", "eco"),  # comfort also keeps >= eco with lowered medium
        (0.70, "frost", "comfort"),
    ],
)
def test_hysteresis(p, current, expected):
    assert decide_preset(_nobody(p), current).preset == expected


def test_no_ping_pong_on_noisy_probability():
    """Probability oscillating around the high threshold must not toggle presets."""
    current: str | None = "frost"
    history = []
    for p in [0.72, 0.68, 0.71, 0.67, 0.69, 0.70, 0.66, 0.71]:
        current = decide_preset(_nobody(p), current).preset
        history.append(current)
    assert history == ["comfort"] * len(history)


def test_no_ping_pong_around_medium():
    current: str | None = "frost"
    history = []
    for p in [0.41, 0.38, 0.42, 0.37, 0.39]:
        current = decide_preset(_nobody(p), current).preset
        history.append(current)
    assert history == ["eco"] * len(history)


def test_custom_thresholds():
    th = Thresholds(high=0.9, medium=0.6, hysteresis=0.0)
    base = replace(_nobody(0.8), thresholds=th)
    assert decide_preset(base, None).preset == "eco"
    assert decide_preset(replace(base, house_probability=0.9), None).preset == "comfort"
    assert decide_preset(replace(base, house_probability=0.5), None).preset == "frost"


def test_boost_current_counts_as_comfort_for_hysteresis():
    assert decide_preset(_nobody(0.66), "boost").preset == "comfort"


def test_all_presets_configurable():
    presets = PresetConfig(
        comfort="boost", medium="comfort", absent="eco", unused="frost", night="frost"
    )
    assert decide_preset(replace(_nobody(0.9), presets=presets), None).preset == "boost"
    assert decide_preset(replace(_nobody(0.5), presets=presets), None).preset == "comfort"
    assert decide_preset(replace(_nobody(0.1), presets=presets), None).preset == "eco"
    home_unused = inp(someone_home=True, schedule=(False, at(18)), presets=presets)
    assert decide_preset(home_unused, None).preset == "frost"
    bed = inp(someone_home=True, bed_mode=True, zone_type="day", presets=presets)
    assert decide_preset(bed, None).preset == "frost"
    # hysteresis by role: current "comfort" here is the medium preset
    assert decide_preset(replace(_nobody(0.37), presets=presets), "comfort").preset == "comfort"


# --------------------------------------------------------------------------- per-person candidates


def test_candidates_never_mix_probability_and_eta():
    """Alice p=0.30 ETA 17:00, Bob p=0.80 ETA 21:00: the house sensors say 86 % at 17:00.

    The combined probability must not be paired with Alice's (earliest) ETA:
    pre-heat follows Bob (comfort from 20:30), not 86 % at 17:00.
    """
    alice, bob = (0.30, at(17)), (0.80, at(21))
    house_p = 1 - (1 - alice[0]) * (1 - bob[0])  # 0.86
    for now, expected in [(at(16, 40), "frost"), (at(20, 0), "frost"), (at(20, 35), "comfort")]:
        legacy = inp(now=now, house_probability=house_p, house_eta=at(17))
        with_candidates = replace(legacy, return_candidates=[alice, bob])
        assert decide_preset(with_candidates, "frost").preset == expected, now
    # Without the attribute (old integration): the single pair is still used.
    assert (
        decide_preset(
            inp(now=at(16, 40), house_probability=house_p, house_eta=at(17)), "frost"
        ).preset
        == "comfort"
    )


def test_two_low_candidates_do_not_sum():
    """Two persons at 35 % each: combined 58 % (eco), but neither return is likely."""
    a, b = (0.35, at(12, 10)), (0.35, at(12, 15))
    house_p = 1 - 0.65 * 0.65
    legacy = inp(house_probability=house_p, house_eta=at(12, 10))
    assert decide_preset(legacy, "frost").preset == "eco"
    d = decide_preset(replace(legacy, return_candidates=[a, b]), "frost")
    assert d.preset == "frost", d.reason


def test_candidates_highest_rank_wins_with_own_schedule_and_hysteresis():
    now = at(19, 50)
    schedule = sched(now, bedroom_blocks())  # off now, on from 20:00
    cands = [
        (0.95, at(19, 55)),  # in window but bedroom not scheduled at 19:55
        (0.50, at(20, 10)),  # medium, scheduled at ETA -> eco
        (0.99, at(23, 0)),  # outside the pre-heat window
    ]
    base = inp(now=now, zone_type="night", schedule=schedule, return_candidates=cands)
    assert decide_preset(base, "frost").preset == "eco"
    # Hysteresis per candidate against the current preset.
    cands2 = [(0.67, at(20, 10)), (0.2, at(20, 5))]
    assert decide_preset(replace(base, return_candidates=cands2), "comfort").preset == "comfort"
    assert decide_preset(replace(base, return_candidates=cands2), "frost").preset == "eco"
    # Candidate without ETA is ignored; empty list falls back to the house pair.
    assert decide_preset(replace(base, return_candidates=[(0.9, None)]), None).preset == "frost"
    fallback = replace(base, return_candidates=[], house_probability=0.9, house_eta=at(20, 10))
    assert decide_preset(fallback, None).preset == "comfort"


def test_candidates_upgrade_only_when_someone_home():
    # Zone outside its schedule now (unused), scheduled from 12:05 (at the ETAs).
    home = inp(
        someone_home=True, schedule=(False, at(12, 5)), return_candidates=[(0.9, at(12, 10))]
    )
    assert decide_preset(home, "eco").preset == "comfort"
    low = replace(home, return_candidates=[(0.3, at(12, 10)), (0.3, at(12, 6))])
    assert decide_preset(low, "eco").preset == "eco"  # unused preset kept, never frost
