# Implementation plan — Home Return Predictor

> Written before any code. Execution starts immediately after (per instructions).
> Note: the formal plan-mode tool blocks on user approval; since the brief says
> "start without waiting for approval", the plan lives here instead.

## 1. Repository layout

```
ha-home-return-predictor/
├── custom_components/home_return_predictor/
│   ├── __init__.py            # setup/unload entry, services registration
│   ├── manifest.json          # iot_class calculated, deps person/zone/recorder, no requirements
│   ├── const.py               # domain, option keys + defaults, entity keys, event name
│   ├── config_flow.py         # persons (multi entity selector) + home zone; options flow
│   ├── coordinator.py         # DataUpdateCoordinator, per-person runtime, listeners, mode logic
│   ├── history.py             # recorder bootstrap -> presence transitions + GPS fixes (executor)
│   ├── storage.py             # helpers.storage.Store wrapper, schema version 1 + migration hook
│   ├── entity.py              # base CoordinatorEntity, DeviceInfo per person + "house" device
│   ├── sensor.py / binary_sensor.py / select.py
│   ├── diagnostics.py         # async_get_config_entry_diagnostics (redacted coordinates)
│   ├── services.yaml, strings.json, icons.json, translations/{en,it}.json
│   └── hrp_core/              # PURE PYTHON, stdlib only, relative imports only
│       ├── base.py            # ReturnModel ABC + dataclasses (THE contract, written first)
│       ├── episodes.py        # presence transitions -> AbsenceEpisode, filters (<10 min, >36 h)
│       ├── prior.py           # kernel-weighted survival prior + habit ETA
│       ├── evidence.py        # haversine, approach-speed regression, Beta-calibrated LR
│       ├── fusion.py          # log-odds fusion, clamp, ETA combination
│       ├── maturity.py        # prediction log, Brier, climatology baseline, maturity score
│       ├── bayes_model.py     # KernelBayesReturnModel(ReturnModel) + (de)serialisation
│       └── preset_logic.py    # reference implementation of the blueprint decision table
├── blueprints/automation/home_return_predictor/
│   ├── vtherm_zone_preheat.yaml
│   └── (preheat bridge blueprint if a thermal plugin can consume our signals)
├── scripts/simulate.py        # months of synthetic data, daily accuracy curve, no HA needed
├── tests/
│   ├── model/                 # pure tests (import hrp_core via sys.path, no HA)
│   └── integration/           # pytest-homeassistant-custom-component: config flow, setup, entities
├── .github/workflows/{ci.yml,validate.yml}   # ruff + pytest ; hassfest + HACS action
├── hacs.json, pyproject.toml (ruff + pytest config), README.md (Italian), LICENSE
```

`hrp_core` sits inside the integration (so HACS ships it) but never imports
`homeassistant`; tests and `simulate.py` put the integration dir on `sys.path`
and `import hrp_core` directly, so the parent `__init__.py` (which imports HA) is
never executed for pure tests.

## 2. Model choices

Per person, one `KernelBayesReturnModel` implementing `ReturnModel`:

1. **Episodes**: presence transitions (home/not_home) -> absences
   `(left_at, returned_at, weekday, is_holiday)`. < 10 min dropped (GPS noise);
   > 36 h stored as `trip` (excluded from kernel, only used for a small long-absence hazard).
2. **Prior** P(return in (d, d+H] | left at X, away d, day Y): weighted survival
   estimate over completed episodes with duration > d (risk set). Weight =
   circular Gaussian on departure time-of-day (σ ≈ 75 min) × day weight
   (same weekday 1.0, same weekday/holiday class 0.5, other 0.15) × recency
   2^(−age/half_life). Shrunk toward a global hazard with a pseudo-weight, so few or
   heterogeneous neighbours (shift worker) give a less extreme prior. Habit ETA =
   weighted conditional median of return time.
3. **GPS evidence**: haversine distance to home zone, fixes with accuracy > 250 m
   dropped, approach speed = −slope of an OLS fit over the last 10 min (≥ 3 fixes,
   ≥ 2 min span). Speed -> class (away_fast / away / still / approach_slow /
   approach / approach_fast). Each class has Beta-style pseudo-counts for
   P(class | return within H) and P(class | no return) initialised from default LRs
   (≈0.2 … 8), updated online once per class per absence episode -> self-calibrated LR.
   "Pass near home and leave" episodes push the LR of approach classes down.
4. **Fusion**: `logit(P) = logit(prior) + log(LR)`, P clamped to [0.01, 0.99]. No
   hand-written person-type rules: a sharp prior (fixed schedule) dominates, a
   flat prior (shifts) lets the LR move it.
5. **ETA**: inverse-variance combination of habit ETA (spread from weighted IQR) and
   position ETA (distance/speed when speed > 5 km/h, variance ∝ ETA²).
6. **Maturity**: prediction log (sampled every ≥ 10 min while away), each resolved
   against the truth (returned within H?). Brier score vs a duration-only survival baseline (see §5)
   -> Brier skill score. maturity = data factor
   (1 − e^(−n_eff/12), n_eff = recency-weighted episodes) × skill factor
   (clamp(BSS / 0.25, 0, 1), requires ≥ 30 resolved predictions).
7. **Modes**: select auto/learning/live. auto -> live when maturity ≥ threshold
   (default 0.7), back to learning when maturity < threshold − 0.1 (hysteresis).

## 3. Dependencies and parallelism

```
[me] base.py contract + const entity/option names + PLAN  (serial, first)
        │
        ├── Agent A: hrp_core impl + pure tests + scripts/simulate.py
        ├── Agent B: HA integration (all HA files) + integration tests
        ├── Agent C: preset_logic.py + tests + blueprints + README  (needs plugin research)
        ├── Agent R: research on thermal plugins + VTherm (already running)
        └── [me]   : packaging — pyproject, CI workflows, hacs.json, .gitignore, LICENSE
        │
        ▼
 integrate & run full suite (me) -> independent verification agent (did not write code)
 -> code-review skill -> fixes -> GitHub repo (private at first, later made public), small commits, tag v0.1.0
```

## 4. Verification per piece

| Piece | How it is verified |
|---|---|
| hrp_core | pytest with synthetic fixed-schedule + rotating-shift people: shift learnt from departure time, maturity rises, habit change absorbed (decay), drive-by gives no strong false positive; serialisation round trip |
| simulate.py | runs standalone (no HA import), prints daily accuracy / Brier / maturity curve |
| preset_logic | truth-table tests: day/night/always zones, schedule evaluated at ETA, bed switch, priority order, "never downgrade while someone home", hysteresis |
| integration | pytest-homeassistant-custom-component: config flow (user + options + duplicate abort), setup/unload, entities created with stable unique_ids, select changes mode, services registered, diagnostics |
| packaging | ruff clean, hassfest + HACS action in CI, manifest keys checked |
| final | separate subagent reviews everything against the brief; `/code-review` pass |

## 5. Follow-up plan: champion/challenger + model fixes (added after first build)

Requested by the user mid-build: optional scikit-learn model trained in the background,
promoted automatically only when it beats the stdlib model.

1. **Model fixes** (hrp_core, pure):
   - Maturity baseline is too easy (climatology). Replace with a still-trivial but
     informed baseline: duration-only survival P(return in (d, d+H] | away d) ignoring
     time of day / weekday (recency-weighted Kaplan–Meier style). Maturity then measures
     the value of learnt habits, so it no longer saturates in ~2 weeks.
   - GPS calibration "once per class per absence" misses the real return on drive-by
     days. Change to once per class per *approach segment* (a segment resets after the
     class changes to still/away for ≥ 10 min or after a horizon has elapsed).
   - Odd-time returns (sharp prior ≈ 0.005) stay low even with strong approach:
     floor the prior at 0.02 before fusion, and let the calibrated LR cap rise to 50.
2. **Challenger** `hrp_core/challenger.py`: `HistGradientBoostingClassifier`
   (max_iter ≤ 100, max_leaf_nodes ≤ 15, early stopping off, single thread via
   threadpoolctl) trained on the resolved prediction log features (logit prior, ln LR,
   minutes away, departure-hour sin/cos, weekday class, log distance, speed, Bayes P).
   sklearn imported lazily; module importable without it. Not pickled: retrained from
   the persisted log at startup.
3. **Selection** (pure, in `hrp_core`): rolling chronological holdout; candidates
   bayes / learned fusion / sklearn; promote challenger only if it beats the current
   champion's Brier by ≥ 3 % on ≥ 2 consecutive evaluations with ≥ 400 resolved
   entries; demote immediately if it falls behind. Active model exposed in reason +
   MaturityReport (`active_model`, per-candidate Brier).
4. **HA wiring**: option `enable_sklearn_challenger` (default off). When enabled,
   `homeassistant.requirements.async_process_requirements(hass, DOMAIN,
   ["scikit-learn>=1.5"])` installs it at runtime (respects HA constraints); failure →
   repair-free warning log, champion continues. Training every ≥ 6 h or after 150 new
   resolved entries, in `hass.async_add_executor_job`, guarded by a lock so only one
   training runs at a time. Attributes `active_model`, `candidate_brier` on maturity.
5. **Verification**: unit tests (skip if sklearn missing) for training, selection
   hysteresis, no-sklearn fallback; integration test for the option and failed install;
   simulate.py `--sklearn` flag prints which model is active per day; timing check
   that one training on 5000 rows takes < 2 s.

## 6. Fix plan after independent verification + code review

Model side (hrp_core, tests/model):
- M1 `load_history(..., replace: bool = False)`: default MERGES recorder episodes with stored
  ones (dedupe by departure within 10 min) and keeps calibrator counts; `replace=True` = old
  behaviour (used by reset-from-recorder only). Retrain no longer wipes months of learning.
- M2 learned fusion: the weights evaluated by the selector are the weights served.
- M3 `observe_transition` stays cheap (no stacker refit / sklearn probe); refits move to `resolve`.
- M4 compact persistence: prediction log stored as rounded column arrays; target < 400 KB/person.
- M5 training split for short locks: `challenger_snapshot() -> Any` (cheap copy),
  `fit_challenger(snapshot, now) -> Any` (static-ish, heavy, touches no model state),
  `apply_challenger(result) -> dict` (cheap). `train_challenger` remains as a wrapper.

HA side:
- H1 all model calls (incl. observe_transition) in executor under the per-person lock;
  training uses M5 so locks are held only for snapshot/apply.
- H2 listener unsubscribe registered with `config_entry.async_on_unload`.
- H3 one return event per absence (keyed on left_at, persisted) + off-hysteresis 0.05.
- H4 save only when state changed, delay 10 min, final save on unload/stop.
- H5 house ETA published from away persons with p >= 0.25 (earliest; attribute `eta_person`),
  person ETA published whenever away, so the blueprint medium tier works.
- H6 unknown-state persons excluded from prediction/aggregation.
- H7 history: `last_changed` for transitions, multi-day holiday expansion, inaccurate fix in
  non-default zone = no information; chunked queries (7-day windows), fixes downsampled.
- H8 bootstrap fixes left_at from history when no stored data.
- H9 reason text reads `shift_label`; hacs.json minimum HA raised to the tested range.
- Docs/CI: README baseline description, CI runs the whole test suite (incl. blueprint tests).

## 7. Fix plan after re-verification

- R1 model generation counter: `reset()` / `load_history(replace=True)` bump it; the challenger
  snapshot carries it and `apply_challenger` discards stale results.
- R2 per-person return candidates: house ETA sensor gets attribute `candidates`
  (`[{person, probability, eta}]`, away persons with p >= 0.25). `decide_preset` accepts a list
  of candidates and takes the highest-ranked preset (each candidate judged on its own
  probability, ETA and schedule-at-ETA); the blueprint mirrors it with a Jinja loop and falls
  back to the single house ETA/probability when the attribute is missing.
- R3 pending observe tasks are awaited before the final save on unload.
- R4 chunked recorder bootstrap ignores the synthetic first state of each chunk for fixes.

## 8. Plan: replace scikit-learn with an in-house numpy GBM (after first real install)

Real-world finding (logs/ from the user's HA OS, Python 3.14, Alpine/musl): scikit-learn
publishes no musllinux wheels at all, so HA tries to compile it and fails (no meson /
compiler). numpy and scipy do ship musl wheels, and HA pins numpy in package_constraints.

- G1 `hrp_core/gbm.py`: histogram gradient boosting for binary log-loss in numpy only.
  Quantile binning (≤ 64 bins), depth-limited trees (max_depth 3, min_samples_leaf 20),
  shrinkage 0.1, ≤ 80 trees, L2 on leaf values, sample weights. Vectorised histogram
  building (np.bincount) so the GIL is released and an N100 trains 3000×11 in < 1 s.
  Trees exported to plain Python for serving (no numpy at predict time), as today.
- G2 `challenger.py` uses `gbm.py`; candidate renamed `gbm` (stored `sklearn_gbm` selection
  state maps to `gbm`). `numpy_available()` replaces `sklearn_available()` (keep the old
  name as alias for one release). Selection rules unchanged.
- G3 HA: option renamed in UI to "challenger model (gradient boosting)" (same config key
  for compatibility); runtime requirement becomes `numpy` (resolved by HA's constraint
  to the version HA already pins, musl wheel available). Failure → warning, built-in model.
  No install attempt at all if numpy is already importable.
- G4 tests: GBM learns a non-linear synthetic function better than logistic; matches
  sklearn's HistGradientBoosting Brier within a tolerance in a dev-only test (skipped
  when sklearn missing); timing < 1 s for 3000 rows; challenger/selection tests ported;
  scikit-learn removed from requirements_test (kept optional for the comparison test).
- G5 README + simulate.py `--gbm` (keep `--sklearn` as alias).
