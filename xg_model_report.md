# xG Model Report — 2025-26 Regular Season

**Status: COMPLETE.** Items A, B, C resolved; Part 2 (steps 8–11) trained and validated.
No duplicate drop applied (item A). Final training set **111,078 rows / 7,577 goals / 6.821%**.
All seven protected files unchanged (md5 check at the end).

---

# Pre-Part-2 resolutions (A, B, C)

## Item A — Duplicate Pairs → all 10 are TWO REAL events (no drop)

Checked all 10 "duplicate" pairs against NHL API play-by-play. **Every pair is two distinct,
real events** — two distinct `eventId`s with consecutive `sortOrder`s, and either the official
shot-on-goal counter increments by 1 between them or one is a shot-on-goal and the other a
missed-shot. They are genuine same-second rapid-fire sequences (shot + immediate rebound),
logged in one clock-second only because the feed has 1-second resolution.

| Pair | game_id | evidence |
|---|---|---|
| 1 | 2025020165 | ids 306/309, SOG h=27→28 |
| 2 | 2025020410 | ids 105/104, SOG a=26→27 |
| 3 | 2025020478 | ids 301/302, SOG h=3→4 |
| 4 | 2025020555 | id 451 **shot-on-goal** + 459 **missed-shot** (wide-left) |
| 5 | 2025020622 | ids 156/155, SOG a=12→13 |
| 6 | 2025020899 | ids 103/104, SOG h=4→5 |
| 7 | 2025020937 | ids 405/352, SOG h=14→15 |
| 8 | 2025021092 | ids 160/161, SOG h=15→16 |
| 9 | 2025021243 | ids 154/430, SOG h=15→16 |
| 10 | 2025021287 | ids 155/154, SOG h=11→12 |

**Per your confirmation (Option 1): all 20 rows kept as real shots. No duplicate drop applied.**
The `time_since_last_shot ≈ 0` / `is_rebound = 1` copy is the second shot of the sequence and its
feature values are correct. `data_audit_report.md` was given a correction note showing the
audit's "12 duplicate rows" = these 10 + 1 extra regular-season group (2025020323, two events with
*different* shot types backhand/deflected) + 1 playoff group (2025030215, out of scope) — all
verified as distinct events, zero true duplicates.

## Item B — `shot_type_id = 0` (24 rows, all goals)

Stop-condition checks: exactly **24** category-0 rows, **all 24 are goals** — neither trigger fired.

1. **What each is:** all 24 are NHL API `goal` events for which the feed recorded **no `shotType`**
   (a normal-goal-with-missing-type gap; the API simply omits `shotType` on some goals — often
   scrambles/tips in tight/where the shooter attribution is loose). Situation codes are ordinary
   (mostly 5v5 `1551`, some PP/PK); **none are penalty shots or awarded goals** (see step-5 note).
2. **API backfill:** matched all 24 to their exact goal event (time gap 0). **0 of 24** carry a
   `shotType` in the feed → **0 backfillable**. (Verified carefully: e.g. McDavid game 2025020499
   scored twice; the wrist goal is a *different* shot — our category-0 row is his P2 (−88,7) goal,
   which has no shotType.)
3. **Imputation:** all 24 set to the modeling-set mode = **wrist (id 1)** (most common category,
   47,961 rows). Kept in training and scoring.
4. **`shot_type_imputed`** boolean added → True on exactly these 24 rows, False elsewhere.
5. **Penalty shots (reference only):** the NHL play-by-play has **no distinct penalty-shot event
   type** (penalty shots surface as ordinary goal/shot/miss events; a `penalty` event with
   `descKey='ps'` sits separately), and the enriched modeling set carries no penalty-shot flag.
   **No reliable identifier is available**, so none are counted. No exclusions made regardless.

## Item C — Persisted, deterministic build

`model/build_xg_features_2025_26.py` performs the full pipeline (scope filter, **no** duplicate
drop with an inline comment citing the item-A API verification, shot-type imputation + flag,
handedness join, off-wing via `y_norm > 0 = shooter-left`, garbage-strength flagging) and writes
`model/xg_features_2025_26.parquet`. The parquet holds all 111,099 in-scope rows; the 21 garbage
rows carry `train_drop_reason = "garbage_strength_state"` (Part 2 trains only on
`train_drop_reason` NULL). **Re-run twice → identical counts: 111,078 train rows / 7,577 goals**
(reproducibility stop-condition satisfied; goal count exactly 7,577).

---

# Part 2 — Training & Validation

## Step 8 — Validation design & tuning

- **Holdout:** chronologically last 20% of regular-season games by game date (games sorted by
  `game_date`, then `game_id`) — **263 games / 22,089 rows**, starting **2026-03-14**. Never used
  for tuning. Tune set: 1,049 games / 88,989 rows.
- **Tuning:** 5-fold **GroupKFold by `game_id`** on the tune set (no game split across folds),
  XGBoost `binary:logistic`, `tree_method="hist"`, `enable_categorical=True` (`shot_type_id`
  categorical), **early stopping** (50 rounds) on fold validation log loss.
- **Grid (randomized modest search, 16 configs)** over: `max_depth`∈{4,5,6,7,8},
  `learning_rate`∈{0.03,0.05,0.08,0.1}, `min_child_weight`∈{5,10,20,40},
  `subsample`∈{0.7,0.8,0.9,1.0}, `colsample_bytree`∈{0.7,0.8,0.9,1.0}.

**Top 5 by CV log loss:**

| CV logloss | ±std | n_est | max_depth | lr | min_child_wt | subsample | colsample |
|---|---|---|---|---|---|---|---|
| **0.21785** | 0.00236 | 292 | **4** | **0.05** | **5** | **0.8** | **1.0** |
| 0.21796 | 0.00232 | 172 | 4 | 0.08 | 5 | 0.9 | 0.9 |
| 0.21815 | 0.00236 | 134 | 6 | 0.05 | 5 | 0.8 | 1.0 |
| 0.21816 | 0.00245 | 89 | 6 | 0.08 | 10 | 1.0 | 1.0 |
| 0.21816 | 0.00234 | 223 | 7 | 0.03 | 5 | 1.0 | 0.8 |

All 16 configs fell in a tight band (0.2178–0.2187); **shallow trees (depth 4) win**, as expected
for xG. **Chosen: max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8,
colsample_bytree=1.0, ~292 trees.** No monotonic constraints (per spec).

**Partial dependence (final full-data model) — shapes are sensible:**

| distance (ft) | 2 | 5 | 10 | 15 | 20 | 30 | 40 | 55 | 75 |
|---|---|---|---|---|---|---|---|---|---|
| mean xG | .212 | .143 | .104 | .088 | .087 | .072 | .038 | .020 | .010 |

Monotonic-decreasing in distance. Angle: highest at small angles (.076–.078 around 0.2–0.6 rad),
falling to ~.036 by 1.57 rad and flat beyond (behind/beside the net) — the expected xG angle shape.

## Step 9 — Holdout evaluation (new model vs old `xg` column)

Holdout: 22,089 shots, **1,537 actual goals**. (`xg_old` clipped to [1e-6, 1−1e-6] for log loss.)

| Metric | No-skill baseline | **NEW model** | OLD `xg` column |
|---|---|---|---|
| Log loss | 0.25258 | **0.22033** | **0.21837** |
| AUC | — | **0.7690** | **0.7755** |
| Brier | — | **0.05978** | **0.05960** |
| Predicted goals (actual 1,537) | — | 1,507.0 | 1,557.3 |

**Where the new model is WORSE than the old column (reported honestly):**
- Overall **log loss, AUC, and Brier are all marginally worse** than the old column
  (Δlogloss +0.0020, ΔAUC −0.0065).
- On **total predicted goals** the old column (1,557, +20 vs actual) is closer than the new model
  (1,507, −30 vs actual) — the new model slightly under-predicts overall.
- By strength group the new model under-predicts **5v5** (act 1,120 vs new 1,096.8) and over-predicts
  **extra-attacker** (act 31 vs new 41.4).

**Where the new model is BETTER / equal:**
- **Tip-in** calibration: act 143 vs new **149.8** — the old column badly over-predicts tips (190.6).
- Low-probability calibration (decile 0): new 0.0052 vs actual 0.0054 (old predicts 0.0062 vs
  actual 0.0005 — old is miscalibrated at the very low end).
- Both models comfortably beat the no-skill baseline (0.2526).

> **Important caveat on the head-to-head:** the old `xg` column is a prior multi-season model whose
> training data almost certainly **included these 2025-26 holdout games** (it is baked into
> `shots_augmented`, spanning 2010→2025). The new model's holdout scores are true out-of-sample,
> the old column's are effectively in-sample. So the old column's small edge is at least partly a
> leakage advantage, not a like-for-like win. The new model is competitive despite training on a
> single season (89k tune rows) vs the old model's ~1.7M.

**Calibration deciles (new model) — well calibrated:**

| decile | n | pred | actual |
|---|---|---|---|
| 0 | 2209 | 0.0052 | 0.0054 |
| 3 | 2209 | 0.0283 | 0.0321 |
| 5 | 2208 | 0.0558 | 0.0512 |
| 7 | 2209 | 0.0999 | 0.1014 |
| 9 | 2209 | 0.2105 | 0.2236 |

**By strength group (n / actual / new-pred / old-pred):** 5v5 17,703/1,120/1,096.8/1,168.5 ·
PP 2,927/309/284.4/266.7 · PK 501/39/36.6/38.1 · extra-attacker 493/31/41.4/37.9 ·
3v3-4v4 465/38/47.8/46.1.

## Step 10 — Out-of-fold scoring + saved model

- **OOF** over the full cleaned set (111,078 rows) via 5-fold GroupKFold by game with the tuned
  params — every shot scored by a model that never saw its game. **OOF log loss 0.21843, AUC
  0.7648; total OOF xG = 7,569.3 vs 7,577 actual goals** (aggregate calibration within 8 goals
  over the whole season).
- **`model/shots_xg_2025_26.parquet`** — 111,078 rows (= cleaned training set): `game_id`,
  `abs_secs`, `shooter_id`, `xg_new` (OOF), `fold`, `xg_old`.
- **`model/xg_model_2025_26.json`** — one XGBoost model fit on the full cleaned set (tuned params,
  292 trees) for scoring future games.

## Step 11 — Player sanity check

All situations (ixG = summed shot xG; new = OOF):

| Player | G | ixG new | ixG old | G−ixG new | G−ixG old | imputed goals |
|---|---|---|---|---|---|---|
| McDavid | 42 | 39.98 | 40.03 | +2.02 | +1.97 | **1** |
| Kucherov | 37 | 24.92 | 24.00 | +12.08 | +13.00 | 0 |
| MacKinnon | 45 | 36.27 | 35.96 | +8.73 | +9.04 | **1** |
| Brady Tkachuk | 18 | 25.79 | 26.33 | −7.79 | −8.33 | 0 |
| Mark Stone | 26 | 21.74 | 23.00 | +4.26 | +3.00 | 0 |

5v5 only: McDavid G25 new 23.54 / old 24.70 · Kucherov G25 new 16.09 / old 16.35 ·
MacKinnon G31 new 23.20 / old 24.27 · Brady Tkachuk G11 new 14.04 / old 14.58 ·
Mark Stone G14 new 11.26 / old 11.66.

- **None of the five differ from the old column by more than 20%** (largest gap: Mark Stone −5.5%;
  Kucherov +3.8%, MacKinnon +0.8%, McDavid −0.1%, Brady Tkachuk −2.1%). No flags.
- **`shot_type_imputed` goals among the five:** McDavid **1**, MacKinnon **1** (each has one of the
  24 category-0 goals, now imputed to wrist). Both players' new-vs-old gaps are tiny, so the
  imputation had no material effect on their ixG. No adjustments made.
- Kucherov's large G−ixG (+12) and Brady Tkachuk's negative (−7.8) reflect genuine finishing
  over/under-performance, consistent under both models (not a model artifact).

**League top 20 by new ixG (old beside), goals:**

| shooter_id | ixG new | ixG old | G |
|---|---|---|---|
| 8478402 (McDavid) | 39.98 | 40.03 | 42 |
| 8481540 | 37.16 | 36.13 | 51 |
| 8480027 | 36.97 | 36.37 | 44 |
| 8477492 (MacKinnon) | 36.27 | 35.96 | 45 |
| 8479337 | 35.55 | 34.78 | 37 |
| 8482740 | 34.73 | 35.06 | 44 |
| 8478864 | 33.69 | 32.15 | 39 |
| 8477946 | 31.61 | 31.16 | 30 |
| 8475166 | 31.52 | 31.88 | 29 |
| 8484801 | 31.15 | 29.02 | 40 |
| 8478414 | 31.03 | 31.87 | 22 |
| 8476881 | 30.87 | 31.93 | 22 |
| 8477404 | 30.75 | 31.89 | 35 |
| 8479420 | 30.30 | 30.00 | 37 |
| 8482699 | 30.12 | 29.79 | 40 |
| 8480002 | 30.09 | 30.34 | 27 |
| 8482093 | 30.04 | 29.62 | 27 |
| 8478398 | 29.73 | 30.19 | 36 |
| 8481604 | 29.65 | 28.21 | 37 |
| 8478498 | 29.33 | 30.50 | 23 |

New and old ixG track closely across the top of the league (typically within ~1 goal), a good
consistency signal.

---

## Outputs written
- `model/xg_model_2025_26.json` — full-data XGBoost model (future scoring).
- `model/shots_xg_2025_26.parquet` — 111,078 rows: keys + `xg_new` (OOF) + `fold` + `xg_old`.
- `model/xg_features_2025_26.parquet` — cleaned modeling table (item C), with `shot_type_imputed`
  and `train_drop_reason`.
- `model/player_handedness.csv` (Part 1) and `model/game_dates_2025_26.csv` (holdout dates).
- `model/build_xg_features_2025_26.py` — deterministic feature builder.

## md5 integrity check (unchanged throughout — all protected files)
| File | md5 |
|---|---|
| model/shots_augmented.parquet | `47af38575046913c9699837bd3a1cd44` ✓ |
| model/season_cache/shifts_20252026.parquet | `919143f2d1eba4e93c81a87759940984` ✓ |
| model/shots_enriched_2025_26.parquet | `23bb76c3f95ca7146bbb9ac2ed7a11e7` ✓ |
| model/composite_ratings_sim.csv | `d02630fd68286488dd951351c84dc85c` ✓ |
| model/rapm_results.csv | `53cc52783cb279da0c8f9ee9c07f5637` ✓ |
| model/team_strength.csv | `136632ddafc0d6234045ac2202594d55` ✓ |
| model/season_simulation_results.csv | `06ef68fffb575ad505262ce7f3ed1e20` ✓ |

## Choices I made that you did not specify
1. **Randomized 16-config search** over the 5-param space (rather than a full 1,280-cell grid) to
   keep the "modest grid" tractable; reported the space, the top-5, and the winner.
2. **Fixed `n_estimators = 292`** (the mean CV best-iteration of the winning config) for the
   holdout, OOF, and full-data fits, instead of re-running early stopping on each — keeps the three
   models on one comparable tree count.
3. **Strength-group definitions** for step-9 calibration: extra-attacker = `skaters_for==6`;
   5v5 = (5,5); PP = shooter advantage excluding extra-attacker; PK = shooter disadvantage;
   3v3/4v4 = (3,3),(4,4). ("other" was empty after cleaning.)
4. **Holdout ordering tie-break** = `game_date` then `game_id`.
5. **`xg_old` clipped to [1e-6, 1−1e-6]** for log loss (it contained exact 0.0 values that make log
   loss undefined); AUC/Brier unaffected.
6. **Partial dependence** = average marginal prediction over a 4,000-row random sample.
7. Added `game_date` and `xg_old` columns to `xg_features_2025_26.parquet` (needed for the holdout
   split and the benchmark) — neither is a model feature.
8. Cached game dates to `model/game_dates_2025_26.csv` via 32 club-schedule API calls.
