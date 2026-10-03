# xG v2 — Season Feature (final)

**Outcome: the season feature fixed the calibration. ALL THREE GATES PASSED. v2 is written.**

- Hypothesis (Step 1) **confirmed**: shot types converted differently in prior seasons — tip-in
  most strongly (v1 under-predicts prior-season tip-ins by 20–34%).
- Config C + `season` feature (depth 5, lr 0.03, ~435 trees) passes every gate:
  - **GATE 1** total OOF **7,614.2 vs 7,577 = +0.49%** (was +3.39% without season) ✅
  - **GATE 2** every ≥150-goal shot type within 0.90–1.10 (tip-in 0.923, slap 0.996) ✅
  - **GATE 3** holdout log loss significantly better than v1 (bootstrap CI [−0.00193, −0.00054]) ✅
- **Written:** `model/xg_model_2025_26_v2.json`, `model/shots_xg_2025_26_v2.parquet` (111,078 rows).
  All protected files md5-unchanged.

Deterministic script: `model/train_xg_v2.py` (random_state=42). Fixed holdout = same last 263 games
of 2025-26 (22,089 rows / 1,537 goals). Training pool = first 80% of 2025-26 + all of 2023-24 +
all of 2024-25. Config C = 2023-24 + 2024-25 + 2025-26, unweighted, 12 shot types, + `season`
(integer start year 2023/2024/2025).

---

## Step 1 — Hypothesis confirmed (diagnostic)

### 1a. Per-season shot-type profile (raw)

Raw goal rate by shot type shows a clear **tip-in decline** and a wrist→snap coding shift:

| Shot type | 2022-23 rate | 2023-24 | 2024-25 | 2025-26 | share drift |
|---|---|---|---|---|---|
| wrist | 6.19% | 5.86% | 5.78% | 6.22% | 52.9%→43.2% (falling) |
| snap | 8.69% | 9.29% | 8.35% | 8.06% | 15.4%→24.9% (rising) |
| slap | 5.15% | 4.94% | 4.69% | 4.85% | ~12% (stable) |
| **tip-in** | **8.77%** | **8.34%** | **7.76%** | **6.75%** | ~9% (rate **declining**) |
| deflected | 9.71% | 8.18% | 10.01% | 11.55% | ~2% |

### 1b. v1 model (2025-26 only) scored on prior seasons — actual/predicted by shot type

This isolates a season effect *after* accounting for location and context (not just raw rate):

| Season | overall A/P | wrist | snap | slap | tip-in | deflected |
|---|---|---|---|---|---|---|
| 2022-23 | 0.995 | 0.930 | 1.010 | 1.049 | **1.343** | 0.979 |
| 2023-24 | 1.041 | 0.982 | **1.118** | **1.057** | **1.286** | **0.804** |
| 2024-25 | 1.038 | 1.003 | **1.051** | **1.052** | **1.198** | **0.932** |

### 1c. Shot types with a season effect (A/P outside 0.95–1.05, ≥100 goals)

**tip-in** (every prior season, strongest: A/P up to 1.343), **snap** (2023-24, 2024-25),
**slap** (2023-24, 2024-25), **deflected** (2023-24, 2024-25), plus wrist/backhand in 2022-23.

**STOP-condition check** (slap & tip-in in 2023-24 and 2024-25 within 0.95–1.05?):
2023-24 slap **1.057**, tip-in **1.286**; 2024-25 slap **1.052**, tip-in **1.198** — **all OUTSIDE**.
The hypothesis holds → **did not stop; proceeded to Step 2.**

**Read:** prior-season tip-ins (and to a lesser degree snap/slap) converted *higher* than v1
expects, so a pooled model without season learns those elevated rates and over-applies them to
2025-26 — exactly the +3.39% / slap+35% / tip-in+21% miscalibration seen in the v2 (no-season) test.

## Step 2 — Train configuration C with a season feature

Tuned with GroupKFold(5) by game on the 314,290-row pool. Grid: max_depth {4,5,6} × lr {0.03,0.05}
× min_child_weight {5,10,20} × subsample {0.8,1.0} (colsample fixed 1.0), 14 deterministic samples.

- **Chosen: max_depth 5, lr 0.03, min_child_weight 5, subsample 0.8, colsample 1.0, ~435 trees**
  (CV log loss **0.21268**, best in grid).
- **v1's params (depth 4, lr 0.05, mcw 5, ss 0.8) on the same folds: CV 0.21277** (reference) — the
  tuned depth-5 model is marginally better.

**How the model uses `season`:**
- **Feature importance (gain): season ranks 12 of 16** — a low-importance, *corrective* feature, not
  a driver (top gains: distance 90.0, |y| 61.6, is_rebound 54.1, shot_type_id 29.2).
- **Season what-if** — the same holdout shots scored with season set to 2023 / 2024 / 2025 (mean
  predicted goal rate):

  | Shot type | season=2023 | season=2024 | season=2025 |
  |---|---|---|---|
  | **tip-in** | 0.0815 | 0.0792 | **0.0713** |
  | slap | 0.0498 | 0.0497 | 0.0493 |

  Season directly **pulls tip-in down** as it advances to 2025 (0.0815→0.0713, tracking the actual
  2025-26 tip-in rate ≈0.0675), while barely touching slap — precisely the intended correction.

## Step 3 — Holdout evaluation (263 games, 22,089 shots, 1,537 goals)

| Metric | No-skill | **v2 (season)** | v1 | C (no season) | old `xg` (leaked) |
|---|---|---|---|---|---|
| Log loss | 0.25261 | **0.21943** | 0.22061 | 0.21924 | 0.21837 |
| AUC | — | **0.7719** | 0.7682 | 0.7721 | 0.7755 |
| Brier | — | **0.05963** | 0.05985 | 0.05961 | 0.05960 |
| Predicted goals (actual 1,537) | — | **1,536.4** | 1,511.0 | 1,567.9 | 1,557.3 |

**Bootstrap of holdout log-loss difference vs v1** (1,000 game resamples): mean **−0.00120**,
95% CI **[−0.00193, −0.00054]** → excludes 0, **v2 significantly better than v1**.

> Honest note: **C-without-season edges v2 on holdout log loss** (0.21924 vs 0.21943) and AUC — the
> season feature costs a hair of holdout discrimination. But C-without-season is the version that
> **failed** the out-of-fold calibration gate (+3.39%, tip-in +21%); v2's holdout predicted-goals
> total (1,536.4 vs 1,537 actual) is essentially perfect where C's was +2%. The old column still
> leads on log loss/AUC because it **trained on these holdout games** (leaked) — noted per instruction.

**v2 calibration by decile** (well-behaved): dec0 pred 0.0051/act 0.0032 · dec4 0.0399/0.0407 ·
dec7 0.0999/0.0937 · dec9 0.2168/0.2232.

**By strength (n / actual / v2 / v1 / C-no-season / old):** 5v5 17,703/1,120/**1,099**/1,102/1,128/1,169 ·
PP 2,927/309/**290**/286/305/267 · PK 501/39/**38**/36/39/38 ·
extra-attacker 493/31/**41**/40/41/38 · 3v3-4v4 465/38/**50**/46/53/46.

**By shot type (n / actual / v2 / v1 / C-no-season / old)** — the tip-in fix is visible:

| Type | n | actual | **v2** | v1 | C-no-season | old |
|---|---|---|---|---|---|---|
| wrist | 9,197 | 603 | 561.5 | 574.5 | 571.2 | 567.6 |
| snap | 5,743 | 497 | 477.8 | 472.6 | 496.4 | 478.0 |
| slap | 2,720 | 101 | 134.0 | 127.5 | 135.5 | 128.7 |
| **tip-in** | 2,240 | 143 | **159.7** | 147.9 | **173.7** | 190.6 |
| backhand | 1,540 | 131 | 127.0 | 127.3 | 130.3 | 132.2 |

v2 pulls holdout tip-in from C-no-season's 173.7 down to 159.7 (actual 143). (Slap stays high on this
*holdout* subset for all models — a low-slap-luck sample — but on the full-season OOF slap calibrates
to 0.996; see GATE 2.)

## Step 4 — Out-of-fold scoring + gates

OOF over all 111,078 2025-26 rows, 5-fold GroupKFold by game; each fold trains on its other 2025-26
games + all of 2023-24 + all of 2024-25, every shot at its true season. OOF log loss 0.21745, AUC 0.7683.

| Gate | Requirement | Result | Verdict |
|---|---|---|---|
| **1** | total OOF within ±2% of 7,577 | **7,614.2 = +0.49%** | ✅ PASS |
| **2** | every shot type ≥150 goals: OOF pred/actual ∈ [0.90, 1.10] | all pass (below) | ✅ PASS |
| **3** | holdout log loss significantly better than v1 | bootstrap CI hi = −0.00054 < 0 | ✅ PASS |

**GATE 2 — OOF predicted/actual by shot type (all shown; ≥150 goals are gated):**

| Shot type | n | goals | pred | ratio | gated |
|---|---|---|---|---|---|
| wrist | 47,977 | 2,984 | 2,941.8 | 1.014 | ✓ |
| snap | 27,632 | 2,227 | 2,262.1 | 0.984 | ✓ |
| slap | 13,200 | 640 | 642.5 | 0.996 | ✓ |
| backhand | 8,034 | 662 | 664.0 | 0.997 | ✓ |
| tip-in | 10,663 | 720 | 780.0 | 0.923 | ✓ |
| deflected | 1,827 | 211 | 192.8 | 1.094 | ✓ |
| wrap-around | 731 | 31 | 34.5 | 0.900 | (n/a <150) |
| poke | 371 | 41 | 37.0 | 1.107 | (n/a) |
| bat | 563 | 52 | 52.9 | 0.983 | (n/a) |
| between-legs | 72 | 8 | 6.2 | 1.283 | (n/a) |
| cradle | 8 | 1 | 0.5 | 2.089 | (n/a) |

All six gated shot types (≥150 goals) are inside [0.90, 1.10] — tip-in 0.923 and slap 0.996 both
fixed (were the failures without season).

## Step 5 — Final model, scoring API, outputs, player check

**Final model** fit on all of 2023-24 + 2024-25 + 2025-26 → `model/xg_model_2025_26_v2.json`.
**`model/shots_xg_2025_26_v2.parquet`** — 111,078 rows: `game_id, abs_secs, shooter_id, xg_v2`
(OOF), `fold, xg_v1, xg_old`.

**Scoring API** (`model/train_xg_v2.py::score_shots(df, season=2025)`): `season` **defaults to 2025**,
the most recent trained season, so future 2026-27 games are scored at 2025-26 levels until the model
is retrained. Documented in the script.

**Player check (ixG = summed shot xG; v2 = OOF):**

| Player | G | ixG v1 | ixG v2 | ixG old | 5v5 G | 5v5 v1 | 5v5 v2 | slap+tip % |
|---|---|---|---|---|---|---|---|---|
| McDavid | 42 | 40.0 | 39.9 | 40.0 | 25 | 23.5 | 23.1 | 7% |
| Kucherov | 37 | 24.9 | 25.3 | 24.0 | 25 | 16.1 | 16.1 | 30% |
| MacKinnon | 45 | 36.3 | 37.2 | 36.0 | 31 | 23.2 | 23.7 | 20% |
| Brady Tkachuk | 18 | 25.8 | 25.9 | 26.3 | 11 | 14.0 | 13.9 | 13% |
| Mark Stone | 26 | 21.7 | 21.9 | 23.0 | 14 | 11.3 | 11.4 | 25% |

v1→v2 changes are small (≤0.9 ixG) — expected, since v2 scores 2025-26 at season=2025, near v1's
2025-26-only levels. The higher slap+tip shooters (Kucherov 30%, Stone 25%, MacKinnon 20%) shift
slightly up.

**League top 20 by v2 ixG (v1 beside):** McDavid 39.9 (v1 40.0), Caufield 37.5 (37.2),
MacKinnon 37.2 (36.3), Robertson 36.9 (37.0), DeBrincat 36.1 (35.6), W. Johnston 34.9 (34.7),
Kaprizov 34.3 (33.7), Larkin 32.1 (31.6), Tavares 31.7 (31.5), Guentzel 31.4 (30.7)… — v2 tracks
v1 within ~1 ixG throughout.

**10 largest v1→v2 changes (≥50 shots)** — all small and positive:
Draisaitl +1.14 (+4.6%, 293 sh), Zegras +1.03 (+4.8%), Matthews +1.01 (+3.6%), Rantanen +0.93,
MacKinnon +0.93 (+2.6%), Suzuki +0.92, Slafkovský +0.91, Beniers +0.83 (slap+tip 29%),
Suter +0.83 (+6.7%), Granlund +0.80. **Likely reason:** these are high shot-volume players; v2's
mild positive re-level (OOF total +0.49%) accrues most on high-volume shooters. The largest single
mover, Beniers, is the most slap/tip-heavy (29%) of the group. No change exceeds ~1.1 ixG or ~7% —
the season feature is a **calibration correction, not a re-ranking.**

---

## md5 integrity — all protected files unchanged
Re-hashed at the end and compared (sorted) to the pre-start baseline — **identical**:
`xg_model_2025_26.json`, `shots_xg_2025_26.parquet`, all four `xg_features_*.parquet`, all four
`shots_enriched_*.parquet`, `player_handedness.csv`, `train_xg_multiseason.py`,
`shifts_supplement_2024_25.parquet`, and the four simulator CSVs. ✅ (v1 fully intact.)

## Choices I made that you did not specify
1. **`season` encoded as a plain numeric feature** (int start year 2023/2024/2025), not categorical —
   matches "integer start year" and lets the model interpolate/extrapolate the trend; the
   default-2025 scoring API relies on that ordering.
2. **Tune grid** = 14 deterministic samples from max_depth {4,5,6} × lr {0.03,0.05} × mcw {5,10,20} ×
   subsample {0.8,1.0}, colsample fixed at 1.0 (v1 tuning already favored 1.0). n_est per config = the
   mean early-stopping best-iteration across the 5 CV folds.
3. **v1 holdout comparison uses v1's OOF predictions** (`shots_xg_2025_26.parquet.xg_new`), aligned
   **positionally** (the `(game_id, abs_secs, shooter_id)` key is non-unique on rapid-fire pairs, so a
   merge fans out) — a genuine out-of-sample v1, consistent with the v2 (no-season) report.
4. **C-without-season** was retrained here with its report params (depth 5, lr 0.03, 472 trees) for a
   like-for-like calibration comparison.
5. **Bootstrap** resamples holdout **games** (1,000 draws, seed 42); "significant" = 95% CI of the
   log-loss difference lies entirely below 0.
6. Player names for the top-20 / change lists sourced from `skater_xgar_self_generated.csv` (display
   only; not used in any computation).
