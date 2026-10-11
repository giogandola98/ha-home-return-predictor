# Home Return Predictor

A custom Home Assistant integration that **learns when each member of the household
comes home** and publishes, per person and for the whole house, the **probability of
returning** within a horizon (default 90 minutes), the **estimated time of arrival
(ETA)** and how **reliable** the model is (maturity). It is designed to pre-heat zones
with [Versatile Thermostat](https://github.com/jmcollin78/versatile_thermostat) (VTherm)
through the included blueprint, but the sensors can be used in any automation.

- Domain: `home_return_predictor`
- No external dependencies, no cloud services: everything runs locally.
- Install via HACS (custom repository) or by manual copy.
- Related: [Room Thermal Model](https://github.com/giogandola98/ha-room-thermal-model),
  a companion integration that learns how long each room takes to heat up and feeds
  the blueprint's pre-heat minutes (see [Thermal plugins](#thermal-plugins-comparison-and-recommended-setup)).

---

## Contents

1. [What it does / does not do](#what-it-does--does-not-do)
2. [How it works](#how-it-works)
3. [Probability and maturity](#probability-and-maturity)
4. [Modes: learning, live, auto](#modes-learning-live-auto)
5. [Installation](#installation)
6. [Configuration and options](#configuration-and-options)
7. [Entities](#entities)
8. [Event and services](#event-and-services)
9. [Blueprints](#blueprints)
10. [Control hierarchy with VTherm](#control-hierarchy-with-vtherm)
11. [Decision priority](#decision-priority)
12. [Multi-zone and night examples](#multi-zone-and-night-examples)
13. [Thermal plugins: comparison and recommended setup](#thermal-plugins-comparison-and-recommended-setup)
14. [Automation examples](#automation-examples)
15. [The first days of learning](#the-first-days-of-learning)
16. [Challenger model (gradient boosting, optional)](#challenger-model-gradient-boosting-optional)
17. [For developers: simulation and tests](#for-developers-simulation-and-tests)
18. [Known limitations](#known-limitations)

---

## What it does / does not do

**Does**

- Rebuilds each `person`'s absences from recorder history (up to 120 days) and keeps
  learning in real time.
- While a person is away, estimates the probability that they return within the
  horizon and their ETA, combining **habits** with **GPS position** (when available).
- Measures its own reliability (Brier score against a baseline) and switches from
  *learning* to *live* once it has earned it.
- Aggregates at house level: probability that someone returns, earliest ETA,
  "someone is returning".
- Provides a blueprint that chooses the **preset** of each VTherm zone.

**Does not**

- **The integration does not control any entity.** It only publishes sensors and
  events. Only the blueprint (or your own automations) act on thermostats, and you
  choose which presets to use for each zone.
- Never sets temperatures: the blueprint only changes VTherm presets.
- Does not model room thermals (how long a room takes to heat up): see
  [Thermal plugins](#thermal-plugins-comparison-and-recommended-setup).
- Does not replace VTherm's window detection or safety features.

---

## How it works

For each person the model combines two sources of information.

1. **Habit prior.** Past absences (< 10 min discarded as GPS noise, > 36 h treated as
   trips) are weighted by similarity to the current one:
   - similar departure time (circular kernel, roughly ±75 min);
   - same weekday / same workday-vs-holiday class (optional holiday calendar);
   - recency (configurable half-life, default 28 days), so a change of habits is
     absorbed within a few weeks.

   The probability is a survival estimate: "among similar absences that were still
   ongoing after the same amount of time, how many ended within the horizon?".
   It also works with **shift work**: someone leaving at 06:00 returns at a different
   time than someone leaving at 14:00, and the model tells them apart by departure
   time. With little similar data the estimate is shrunk towards an average value
   instead of becoming extreme.
2. **GPS evidence.** From recent positions (accuracy worse than 250 m discarded) it
   computes the speed of approach to home and classifies it (moving away, stationary,
   approaching slowly/fast…). Each class has a **self-calibrating likelihood ratio**:
   it learns how much "approaching" really means "returning" for that person (someone
   who often passes near home without coming in lowers the weight of that signal).
3. **Fusion.** `logit(P) = logit(prior) + log(LR)`, clamped between 1% and 99%. No
   hand-written rules per type of person: clear habits dominate, fuzzy habits leave
   more room for GPS.
4. **ETA.** Uncertainty-weighted average of the habit ETA and the distance/speed ETA
   when moving faster than 5 km/h.

A person **at home** has probability 100% and `returning` off.

---

## Probability and maturity

These are two different numbers and should be read together.

| | Question it answers | Where |
|---|---|---|
| **Probability** | "How likely is it that they return within 90 minutes, right now?" | `sensor.<p>_return_probability` |
| **Maturity** | "How far can I trust these probabilities, in general?" | `sensor.<p>_model_maturity` |

Maturity grows with:

- **amount of data**: number of absences (recency-weighted);
- **demonstrated skill**: the model logs its own predictions and later compares them
  with what actually happened. It computes the **Brier score** (mean squared error,
  lower = better) and compares it with a deliberately simple **baseline**: the return
  probability computed only from "how long they have been away", without departure
  time or weekday. The *Brier skill score* (`1 − brier / baseline`) therefore measures
  how much the learnt habits are worth; at least 30 verified predictions are needed.

A 90% probability with 20% maturity is a guess; with 85% maturity it is a prediction
you can act on.

---

## Modes: learning, live, auto

Each person has a `select.<p>_mode`:

| Mode | Behaviour |
|---|---|
| `learning` | Everything is computed and published (including `returning`), but with attribute `advisory: true`. Meant for observing without acting. |
| `live` | Fully operational predictions, `advisory: false`. |
| `auto` (default) | Switches to live when maturity exceeds the threshold (default 70%) and back to learning if it drops below threshold − 10 points (**hysteresis**: no flip-flopping). |

The house sensors have `mode: live` if at least one of the relevant people (those
returning, otherwise everyone) is live, and `advisory` accordingly.

The VTherm blueprint has the input **"Act on predictions only in live mode"**,
**enabled by default**. While the model is learning, the blueprint behaves like a
classic presence + schedule + bed automation (so it can replace a zone's existing
automations right away), but **ignores predicted returns**: no pre-heating on
immature predictions. When the person switches to live, pre-heating turns on by
itself. Disable the input if you want pre-heating from day one, accepting a few
unnecessary heating cycles in the first weeks.

If you prefer to keep your current automations while the model learns, don't create
the blueprint automations yet: watch `binary_sensor.<p>_returning` and the maturity,
and enable the blueprint once you are convinced.

---

## Installation

### Via HACS

1. HACS → ⋮ menu → **Custom repositories**.
2. URL: `https://github.com/giogandola98/ha-home-return-predictor`, category
   **Integration** → Add.
3. Search for "Home Return Predictor" → Download → **restart Home Assistant**.

### Manual

Copy `custom_components/home_return_predictor/` to
`<config>/custom_components/home_return_predictor/` and restart.

Requirements: Home Assistant 2026.1 or later, `recorder` enabled, `person` entities
with trackers (ideally the Companion app with GPS).

---

## Configuration and options

Settings → Devices & services → Add integration → **Home Return Predictor**: choose
the people and the home zone (default `zone.home`). History is imported from the
recorder immediately.

Options (Configure):

| Option | Default | Meaning |
|---|---|---|
| Return threshold | 0.6 | Probability above which `returning` turns on |
| Maturity threshold | 0.7 | Maturity for the automatic switch to live |
| Horizon | 90 min | Window of the question "will they be back within…?" |
| Half-life | 28 days | How quickly old habits are forgotten |
| Update interval | 2 min | Periodic recalculation (in addition to event-driven updates) |
| Holiday calendar | — | Optional `calendar.*`: days with events are treated as holidays |
| Challenger model (gradient boosting, numpy) | off | Trains an alternative numpy model in the background, used only if it is better (see [below](#challenger-model-gradient-boosting-optional)) |

---

## Entities

For each person `<p>`:

| Entity | Type | Content |
|---|---|---|
| `binary_sensor.<p>_returning` | binary | On when the probability exceeds the threshold. Attributes: `probability`, `eta`, `mode`, `advisory`, `reason`, `active_model` |
| `sensor.<p>_return_probability` | % | Probability of returning within the horizon (100 when at home) |
| `sensor.<p>_return_eta` | timestamp | Estimated ETA (published while the person is away) |
| `sensor.<p>_model_maturity` | % | Maturity. Attributes: `episodes`, `brier_score`, `baseline_brier_score`, `brier_skill_score`, `active_model`, `candidate_brier`, `challenger_enabled`, `numpy_available` (`sklearn_available`: deprecated alias) |
| `sensor.<p>_return_reason` | text | Human-readable explanation of the current estimate |
| `select.<p>_mode` | select | `auto` / `learning` / `live` |

House:

| Entity | Content |
|---|---|
| `sensor.house_return_probability` | % that at least one away person returns within the horizon; **100 when someone is home** |
| `sensor.house_next_return_eta` | Earliest ETA among away people with probability ≥ 25% (attribute `eta_person`), so the blueprint can also use the "medium probability" band |
| `binary_sensor.house_someone_returning` | On when at least one person is returning |

All house entities have the attributes `occupied`, `persons_home`,
`returning_persons`, `mode`, `advisory`, `away_probability` and `candidates` (the
binary sensor also `returning_names`). `candidates` is the list of away people with
probability ≥ 25%, sorted by ETA: `[{person, probability, eta}]` with `probability`
in **%** (like the sensors) and `eta` in ISO 8601 (or `null`).

---

## Event and services

**Event** `home_return_predictor_return_detected`, fired when a person switches to
"returning":

```yaml
person: person.alice
name: Alice
probability: 0.82
eta: "2026-10-12T18:40:00+02:00"
mode: live
advisory: false
```

**Services**

| Service | Effect |
|---|---|
| `home_return_predictor.retrain` | Rebuilds the model from history (people optional) |
| `home_return_predictor.reset` | Clears the learnt model for the given people |
| `home_return_predictor.export_diagnostics` | Exports the model state (coordinates redacted) |

---

## Blueprints

| File | Purpose |
|---|---|
| `blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml` | Chooses the VTherm preset of **one zone** (one automation per zone) |
| `blueprints/automation/home_return_predictor/presence_bridge.yaml` | Mirrors "someone is returning" (and optionally "someone is home") into an `input_boolean`, for tools that only accept an on/off entity |

**Import.** Use Settings → Automations → Blueprints → **Import blueprint** with these
URLs:

- `https://github.com/giogandola98/ha-home-return-predictor/blob/main/blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml`
- `https://github.com/giogandola98/ha-home-return-predictor/blob/main/blueprints/automation/home_return_predictor/presence_bridge.yaml`

[![Import VTherm zone preheat](https://my.home-assistant.io/badges/blueprint_import.svg)](https://my.home-assistant.io/redirect/blueprint_import/?blueprint_url=https://github.com/giogandola98/ha-home-return-predictor/blob/main/blueprints/automation/home_return_predictor/vtherm_zone_preheat.yaml)

Alternatively copy the YAML files to `<config>/blueprints/automation/home_return_predictor/`.
Then create one automation per zone.

### VTherm blueprint inputs

| Input | Default | Notes |
|---|---|---|
| VTherm thermostats | — | One or more `climate` entities of the zone |
| Zone type | `day` | `day`, `night`, `always` (always used, e.g. bathroom) |
| Zone schedule | — | `schedule` helper: when the zone is used. Empty = always |
| Zone occupancy sensor | — | `binary_sensor`: when on, with someone home, forces comfort |
| Bed / night switch | — | `input_boolean`/`switch`/`binary_sensor` |
| House return probability | `sensor.house_return_probability` | |
| House next return ETA | `sensor.house_next_return_eta` | |
| House "someone returning" sensor | `binary_sensor.house_someone_returning` | `mode`/`advisory` attributes |
| Someone-home source | `zone.home` | Zone state = number of people; also person/group/binary_sensor |
| Act on predictions only in live mode | **on** | See [Modes](#modes-learning-live-auto) |
| Fixed pre-heat minutes | 30 | |
| Pre-heat minutes sensor | — | When numeric it overrides the fixed value (e.g. `sensor.<zone>_preheat_minutes` from [Room Thermal Model](https://github.com/giogandola98/ha-room-thermal-model), or `sensor.<zone>_predicted_duration` from ha-preheat) |
| **Comfort preset** | `comfort` | Zone in use, bed (night zones), return very likely |
| **Medium-probability preset** | `eco` | Return with medium probability |
| **Absent preset** | `frost` | Nobody home and no likely return |
| **Unused-zone preset** | `eco` | Someone home but zone outside its schedule |
| **Night preset for day zones** | `eco` | Day zones with the bed switch on |
| High probability threshold | 70% | ≥ threshold → comfort preset |
| Medium probability threshold | 40% | ≥ threshold → medium-probability preset |
| Hysteresis | 5% | Prevents ping-pong between presets |
| Write decisions to the logbook | off | Useful to understand why a preset was chosen |

All presets are chosen **per zone** from `frost`/`eco`/`comfort`/`boost`, or by typing
another VTherm preset name. If the chosen preset does not exist on a thermostat, that
thermostat is left untouched.

> **Why `frost` and not `away`?** In VTherm the `away` preset is hidden and **not
> selectable** when VTherm's presence feature is disabled (as it should be, see
> below). That is why absence uses `frost` by default.

The blueprint runs every 2 minutes, at HA startup and on every change of its input
entities. Its Jinja logic is identical to
`custom_components/home_return_predictor/hrp_core/preset_logic.py`, which is covered
by tests.

---

## Control hierarchy with VTherm

Golden rule: **only one "brain" changes a zone's presets.**

```
Home Return Predictor  →  who returns, when, how certain (sensors only)
          │
VTherm zone blueprint  →  the ONLY caller of climate.set_preset_mode
          │
Versatile Thermostat   →  preset temperatures, regulation, windows, safety
```

- **Disable VTherm's presence feature** (thermostat *Features* menu:
  `use_presence_feature` off) and also **motion**. If they stay enabled, VTherm
  changes temperatures based on presence on its own and the two logics contradict
  each other (and `away` would become the only way to handle absence).
- **Keep enabled** VTherm's window detection and safety, and power management if you
  use it: the blueprint respects them and does not touch the thermostat while they
  are active.
- The blueprint **never sets temperatures**. Changing the temperature manually in
  VTherm sets the preset to `none`: on the next run the blueprint restores the
  computed preset. For extended manual control, disable the zone's automation.
- When you enable the blueprint, disable any old automations that change presets of
  the same zones: two automations on the same zone will fight each other.

---

## Decision priority

For each thermostat in the zone, from highest to lowest:

| # | Condition | Result |
|---|---|---|
| 1 | VTherm protections: window open (without bypass), safety, power overload, `safety`/`power` preset, HVAC off, thermostat locked | **Touches nothing** |
| 2 | Bed switch on **and** someone home | Day zones → night preset; night/always zones → comfort |
| 3 | Someone home | Zone in use (occupancy on, no schedule, or schedule active now) → comfort; otherwise → unused-zone preset |
| 4 | Predicted return (if allowed by the mode) | If now ≥ ETA − pre-heat minutes (or the ETA has already passed) **and** the zone schedule is active at the ETA: p ≥ high → comfort, p ≥ medium → medium-probability preset, otherwise → absent preset |

Additional rules:

- With someone home, another person's return can only **raise** the preset, never
  lower it (order: absent/frost < eco < comfort < boost).
- With nobody home, rule 4 applies, or the absent preset.
- **Hysteresis**: if the current preset is comfort, it stays comfort while
  p ≥ high − 5; if it is eco (or higher), it stays at least eco while p ≥ medium − 5.
- **Each return is judged on its own**: the blueprint reads the `candidates`
  attribute of the house ETA sensor and applies rule 4 to each person with *their*
  probability, *their* ETA and the schedule at *their* ETA; the highest preset wins.
  The combined house probability is never paired with another person's ETA (e.g.
  Alice 30% at 17:00 and Bob 80% at 21:00: comfort from 20:30, not 86% at 17:00), and
  two unlikely returns do not add up.
- Without the `candidates` attribute (older integration) or with an empty list, the
  blueprint uses the house probability / house ETA pair.
- With someone home the house probability is 100%; the blueprint then uses the
  `away_probability` attribute if the integration provides it, otherwise it assumes
  100% when an ETA exists (i.e. someone is already returning).
- The schedule is evaluated from the current state and the `next_event` of the
  `schedule` helper: if it is off and turns on before the ETA, it is assumed to still
  be on at arrival (a fair approximation for horizons of a few hours).

---

## Multi-zone and night examples

Typical setup, one automation per zone:

| Zone | Type | Schedule | Other |
|---|---|---|---|
| Living room + kitchen | `day` | 07:00–23:00 | bed switch |
| Bedroom | `night` | 20:00–07:00 | bed switch |
| Bathroom | `always` | 06:00–08:00, 19:00–23:30 | — |
| Office | `day` | 08:30–18:00 weekdays | occupancy sensor |

**Return predicted at 23:30, house empty, pre-heat 30 min, p = 85%.**

- At 22:30 all zones are on `frost` (outside the pre-heat window).
- At 23:00 the **bedroom** switches to `comfort`: we are inside the window and its
  schedule (20–07) is active at 23:30.
- The **living room** stays on `frost`: its schedule ends at 23:00, so it will not be
  in use at arrival.
- The **bathroom** stays on `frost`: its schedule ends at 23:30, so it is no longer
  active at arrival time. If its schedule ran until midnight, it would switch to
  `comfort` at 23:00 like the bedroom.

**Same evening, but someone is already home.** Bedroom on comfort (schedule active),
living room on comfort until 23:00 and then on the "unused zone" preset (eco): the
23:30 return does not bring it back to comfort because the zone is not scheduled at
arrival.

**Bed.** When you turn on the bed switch with someone home: living room and office
switch to the night preset (eco), bedroom and bathroom to comfort, regardless of
schedules and returns. If everyone leaves and the bed switch is accidentally left on,
it is ignored.

---

## Thermal plugins: comparison and recommended setup

Home Return Predictor knows **when** someone arrives; it does not know **how long a
room takes to heat up**. For the pre-heat minutes you can use a fixed value or a
thermal plugin.

| Project | What it does | Compatibility with preset-based VTherm |
|---|---|---|
| [Room Thermal Model](https://github.com/giogandola98/ha-room-thermal-model) (companion integration) | Generic per-zone learnt RC thermal model (time constant, heating rate, dead time), trained on all history, with outdoor temperature / weather forecast. **Does not control anything**: sensors only. Per zone `sensor.<zone>_preheat_minutes`, `sensor.<zone>_model_maturity`, and with an arrival sensor (e.g. `sensor.house_next_return_eta`) `sensor.<zone>_preheat_start`, `binary_sensor.<zone>_preheat_now` | **Recommended**: use `sensor.<zone>_preheat_minutes` as the blueprint's pre-heat minutes sensor. Designed to pair with this integration, no second controller |
| [Ecronika/ha-preheat](https://github.com/Ecronika/ha-preheat) (v2.11.x) | Learns per-zone thermal mass, heat loss and dead time (Radiator/Floor/AC profiles). **Does not write setpoints**: it is driven by its own blueprint (comfort/eco). Per-zone entities `binary_sensor.<zone>_preheat_active`, `sensor.<zone>_next_preheat_start`, `sensor.<zone>_predicted_duration` (minutes); house `sensor.preheat_house_next_arrival`, `binary_sensor.preheat_house_incoming`. Requires a per-zone occupancy `binary_sensor`; `schedule_entity` and `global_presence` optional. Has its own arrival predictor | **Good** alternative source of minutes: use `sensor.<zone>_predicted_duration` in our blueprint. **Do not** run its control blueprint at the same time (one controller only). Our ETA cannot be injected as its target; `binary_sensor.house_someone_returning` can be passed as a presence hint via the *presence bridge* blueprint (inferred usage, not documented by ha-preheat) |
| [nic6911/ha-predictive-heating](https://github.com/nic6911/ha-predictive-heating) | Aimed at underfloor heating; writes setpoints with `climate.set_temperature` (an advisory mode exists); entities `sensor.*_recommended_setpoint`, `*_predicted_temperature`; no presence/occupancy/schedule/arrival inputs | **Not recommended** with radiators + VTherm: a manual temperature change sets the VTherm preset to `none` and conflicts with preset-based control |
| [silkyclouds/ha-smart-preheat](https://github.com/silkyclouds/ha-smart-preheat) | YAML package for a single room; physical formula `minutes = 60·ΔT·C / (P − UA·(T_target − T_out))` with hand-tuned C, UA, P; exposes `sensor.preheat_minutes_needed`; writes setpoints itself via `script.preheat_apply` | Only as a **source of minutes**, removing the control script |

**Recommended setup for radiators + thermostatic valves + VTherm**

1. **Home Return Predictor**: who returns, when, how certain.
2. **VTherm zone blueprint**: the only controller, presets only.
3. **Pre-heat minutes**: `sensor.<zone>_preheat_minutes` from
   [Room Thermal Model](https://github.com/giogandola98/ha-room-thermal-model)
   (learnt per zone); alternatively `sensor.<zone>_predicted_duration` from Ecronika
   ha-preheat (radiator profile) without its control blueprint; or, to start with, a
   fixed number (30–45 min for radiators).
4. **VTherm**: presence and motion **disabled**; windows and safety **enabled**.

---

## Automation examples

**Notify when someone is returning (live predictions only)**

```yaml
alias: Notify predicted return
triggers:
  - trigger: event
    event_type: home_return_predictor_return_detected
conditions:
  - condition: template
    value_template: "{{ not trigger.event.data.advisory }}"
actions:
  - action: notify.mobile_app_phone
    data:
      title: On the way
      message: >
        {{ trigger.event.data.name }} should arrive at
        {{ as_timestamp(trigger.event.data.eta) | timestamp_custom('%H:%M') }}
        ({{ (trigger.event.data.probability * 100) | round }}%).
```

**Use the sensor in learning mode as a simple notice**

```yaml
alias: Return notice (also while learning)
triggers:
  - trigger: state
    entity_id: binary_sensor.alice_returning
    to: "on"
actions:
  - action: persistent_notification.create
    data:
      title: >
        {{ 'Prediction (learning)' if state_attr('binary_sensor.alice_returning', 'advisory') else 'Return predicted' }}
      message: >
        Probability {{ (state_attr('binary_sensor.alice_returning', 'probability') * 100) | round }}%,
        reason: {{ state_attr('binary_sensor.alice_returning', 'reason') }}
```

---

## The first days of learning

What to look at in `sensor.<p>_model_maturity` and the other sensors:

- **`episodes`**: number of absences used. After the history import it should already
  be > 0; if it is 0, check that the person has recorder history and that the home
  zone is correct.
- **`brier_score` vs `baseline_brier_score`**: the first must drop below the second.
  While they are similar the model does not beat "the average" and stays in learning.
- **`brier_skill_score`**: > 0 means the model adds information; at 0.5 the skill
  contribution to maturity is maxed out.
- **`sensor.<p>_return_reason`**: explains the estimate compactly, e.g.
  `left 06:10 (morning); approaching 45 km/h, 12.3 km; habit P 0.42, GPS LR 4.0`.
  It is the quickest way to understand odd predictions.
- In the blueprint, temporarily enable **"Write decisions to the logbook"** to see why
  each zone picked a given preset.

With regular habits it usually takes 2–3 weeks to reach live; with variable shifts
(e.g. one person has fixed hours, the other works rotating shifts) it takes longer for
the shift worker. You can force `live` from the select if you trust it earlier.

---

## Challenger model (gradient boosting, optional)

A **champion/challenger** scheme: the built-in model (kernel Bayesian, pure Python)
always remains the *champion*. If you enable the **Challenger model (gradient
boosting, numpy)** option (off by default), the integration trains a *challenger* in
the background on past verified predictions, and uses it **only when it is
demonstrably better**.

**Why not scikit-learn.** Early versions used scikit-learn's
`HistGradientBoostingClassifier`. On a real Home Assistant OS install (Alpine Linux,
musl, Python 3.14) installation failed: scikit-learn does not publish *musllinux*
wheels, so pip tries to build it from source and stops (no meson or compiler in the
container). numpy, on the other hand, has musl wheels and is already a Home Assistant
dependency (version pinned in its constraints). The challenger is therefore now a
home-grown gradient boosting implementation that uses only numpy (`hrp_core/gbm.py`).

- **What it is**: histogram gradient boosting for binary log-loss: features binned
  into ≤ 64 quantile bins, trees of depth ≤ 3 with at least 20 samples per leaf,
  learning rate 0.1, ≤ 80 trees, L2 regularisation on leaves, recency weights. The
  features are the same as in the prediction log (prior probability, GPS likelihood
  ratio, time away, departure time and current time, holiday, distance, speed,
  Bayesian probability).
- **Nothing to install (usually)**: if numpy is already importable, as in normal Home
  Assistant installs, no installation is attempted. Otherwise `numpy` is requested
  without a version (HA's constraints resolve it). If that fails, a warning is logged
  and the built-in model carries on; nothing breaks.
- **Cost on an Intel N100**: a single thread, background training at most every 6
  hours (or after 150 new verified predictions), one training at a time, never in the
  Home Assistant event loop. Measured on an N100 (120-day simulation, log of 3000
  predictions): on average ~0.35 s (max ~0.5 s) per training (evaluation + full
  retraining). Prediction is pure Python (< 0.1 ms) and does not use numpy.
- **Survives restarts**: the exported trees are a small JSON (about 40 KB per person)
  saved with the state, so a promoted challenger keeps serving right after a restart,
  without waiting for retraining.
- **Automatic switch only when better**: the challenger becomes active only if, on
  recent data **not seen in training** (the last 20% of verified predictions), it has
  a Brier score at least **3%** better for **2 consecutive evaluations** and with at
  least **400 verified predictions**. If it then does worse than the built-in model,
  it switches back to the built-in model **immediately**.
- **Which model is active**: see the attributes of `sensor.<p>_model_maturity`:
  `active_model` (`kernel_bayes` = built-in, `learned_fusion` = learnt fusion, `gbm` =
  challenger), `candidate_brier` (Brier of each candidate at the last evaluation;
  lower is better), `challenger_enabled`, `numpy_available` (`sklearn_available` is
  kept for one release as a deprecated alias). `active_model` also appears on
  `binary_sensor.<p>_returning`, on `sensor.<p>_return_reason` (and in the explanation
  text) and in the diagnostics. State saved under the old name `sklearn_gbm` is read
  as `gbm`; the option key (`enable_sklearn_challenger`) is unchanged to preserve the
  existing setting.

---

## For developers: simulation and tests

The model core (`custom_components/home_return_predictor/hrp_core/`) is pure Python,
with no Home Assistant imports.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements_test.txt
.venv/bin/python scripts/simulate.py          # months of synthetic data, accuracy/maturity curve
.venv/bin/python scripts/simulate.py --gbm    # also the gradient boosting challenger (training times)
.venv/bin/python -m pytest                    # all tests
.venv/bin/python -m pytest tests/model -q     # model and preset_logic only (no HA)
.venv/bin/python -m pytest tests/blueprint -q # VTherm blueprint run in a test HA
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

- `tests/model/test_preset_logic.py`: truth table of the preset logic (day/night/always
  zones, schedule evaluated at the ETA, bed, priorities, never downgrade, hysteresis
  without ping-pong, occupancy, protections).
- `tests/blueprint/`: loads the YAML blueprint, runs it in a test Home Assistant with
  fake states and checks the requested presets, to guarantee that the Jinja matches
  `preset_logic.py`.

---

## Known limitations

- A reliable tracker is needed for each person; without GPS the model uses habits
  only.
- The first return after a radical change of habits (new job, holidays) will be
  poorly predicted; the half-life controls how quickly it adapts.
- The house probability means "someone returns"; with several people away the ETA is
  that of the first predicted return.
- Zone schedules are read with the approximation described above (a block starting
  before the ETA is assumed to still be active at arrival).
- Conditions on optional entities (schedule, bed, occupancy) are picked up
  immediately on on/off changes; everything else within 2 minutes at most.
- Guests and untracked people are invisible to the model: use the "someone home"
  source (e.g. an `input_boolean` or a group) to include them.
- Other models: all Home Assistant code talks to the model through the abstract
  `ReturnModel` interface (`hrp_core/base.py`: `load_history`,
  `observe_transition`, `predict`, `log_prediction`, `resolve`, `maturity`,
  `to_dict`/`from_dict`); in the future other backends (e.g. survival models or other
  ML libraries) can be added as new implementations or as additional challengers,
  without touching entities and blueprints.
