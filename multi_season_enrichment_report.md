# Multi-Season Enrichment Report — 2024-25, 2023-24, 2022-23

**Goal:** enrich three additional regular seasons and build their xG feature tables using the
exact 2025-26 logic, as additional training data. **No model training in this step.**

**Status: COMPLETE. No stop condition was triggered.** All protected files (22 checked) are
md5-unchanged. Regular season only; shootout / period-5 rows excluded.

---

## Step 1 — Generalized builders + equivalence (PASSED exactly)

Created `model/build_shots_enriched.py` and `model/build_xg_features.py`, season-parameterized
(via `--season`) versions of the two 2025-26 scripts. The 2025-26 scripts were left untouched.
The enrichment generalization parameterizes exactly the 4 constants the audit named
(`SHIFTS_IN`, `OUT`, `GID_LO`, `GID_HI`); `GOALIE_MEAN_SECS=180` is unchanged and re-validated
per season (Step 3). The feature generalization parameterizes `ENRICHED`, `DATES`, `OUT`, and the
game-id range; `HAND` (player_handedness.csv) is shared.

**Equivalence test (run both new scripts for 2025-26, compare to the protected outputs, sorted on
the shot key, every column):**
- `build_shots_enriched.py --season 2025_26` → **IDENTICAL** to `shots_enriched_2025_26.parquet`
  (111,896 rows, all columns incl. list columns, all flags: goalie_set 98, boundary 12,090,
  no-shifts 797, two-goalies 8, gt6 12 — all match).
- `build_xg_features.py --season 2025_26` → **IDENTICAL** to `xg_features_2025_26.parquet`
  (111,099 rows; 111,078 train; 7,577 goals; 21 garbage; 24 imputed).

Outputs of the equivalence run were written only to scratch temp paths, never to the protected
files. **Equivalence exact → proceeded to Step 2.**

## Step 2 — Data gaps filled

**2024-25 shifts — 57 missing games recovered.** The contiguous block 2024021235–2024021291
returns 0 rows from the JSON shift-chart endpoint. Recovered via the HTML TOI fallback imported
from `model/train_rapm.py` (`fetch_shifts_html` + `shift_record_to_dict`). Importing `train_rapm`
is side-effect-free (module body is only imports, constant assignments, and a `__main__` guard —
verified by AST), so it was imported directly rather than copied. Written to a NEW file
**`model/shifts_supplement_2024_25.parquet`** (43,224 rows, 57 games, same schema as the cache).
The enrichment reads the season cache **plus** this supplement.
**Validation: all 57/57 games pass** — exactly 2 team_ids, both goalies present, plausible per-team
TOI, 37–39 players/game. 0 failures.

**2022-23 — 2 games excluded.** `2022020544` and `2022020843` are absent from `shots_augmented`
(the shot table). They are **not rebuilt** (per instruction); 2022-23 proceeds at 1,310/1,312
games. Recorded here as a known exclusion.

**Handedness — 383 new shooters appended.** Fetched `shootsCatches` from
`/v1/player/{id}/landing` and appended to `model/player_handedness.csv` (933 → 1,316 rows).
Existing rows verified **byte-identical** (appended textually; the original 933-row byte prefix is
unchanged and the reloaded first-933 DataFrame equals the original). **All 383 resolved (0 null).**
Per-season coverage: 2024-25 908/908, 2023-24 907/907, 2022-23 937/937 shooters resolved — **0
unresolved (0 nulls) in any season.**

## Step 3 — Enrichment + goalie re-validation

Enriched each season → `shots_enriched_2024_25.parquet`, `shots_enriched_2023_24.parquet`,
`shots_enriched_2022_23.parquet`. **Row parity exact** (OUTPUT_ROWS == INPUT_SHOTS) every season.

| Enrichment metric | 2024-25 | 2023-24 | 2022-23 |
|---|---|---|---|
| Input = output rows | 111,812 | 114,666 | 113,809 |
| Goalie set size | 103 | 98 | 105 |
| Shots on a shift boundary | 12,188 | 12,807 | 18,512 |
| Shooter-not-on-ice (flag) | 521 | 615 | 653 |
| No-shifts (shootout, 0v0) | 508 | 607 | 641 |
| Two-goalies-against (resolved) | 0 | 1 | 2 |
| >6 skaters a side (flag) | 3 | 12 | 4 |
| Empty-net-for (pulled goalie) | 3,311 | 3,201 | 2,989 |
| Self-overlapping shift rows de-duped | 951 | 2,915 | 644 |

Players-per-side (skaters + goalie, excl. shootout) is dominated by 6 (5 skaters + goalie):
2024-25 106,290 · 2023-24 108,399 · 2022-23 107,330 at exactly 6; residual >6 (raw-shift
anomalies, flagged) only 23 / 51 / 23 rows respectively — same pattern as 2025-26.

**Goalie detection re-validated against NHL API `position` (STOP condition — clean):**

| Season | Detected goalies | All `position=='G'`? | Skater-as-goalie | Goalie-as-skater | Players w/ per-game mean in [150,210]s |
|---|---|---|---|---|---|
| 2024-25 | 103 | ✅ yes | **0** | **0** | 3 (all goalies, pulled-early games; pgm_max=1200) |
| 2023-24 | 98 | ✅ yes | **0** | **0** | 1 (goalie) |
| 2022-23 | 105 | ✅ yes | **0** | **0** | 3 (all goalies) |

The 180 s threshold holds cleanly at the **player** level: skater max per-game-mean shift is
**84 s**, goalie minimum is **283 s** — a wide gap with **no player in between**. The earlier
"thin 176↔185" concern was a *per-game* artifact (a goalie pulled early logs a ~185 s game; a
goalie's short relief game logs ~176 s) — both belong to players who are goalies in their full
games, so the season-level union catches every one. Every player with a per-game mean in the
[150,210] band is a confirmed goalie (`position='G'`) already in the set. **No misclassification;
threshold unchanged; proceeded to Step 4.**

## Step 4 — Feature tables

Built `xg_features_2024_25.parquet`, `xg_features_2023_24.parquet`, `xg_features_2022_23.parquet`
with the exact 2025-26 rules (no duplicate drop; shot-type-0 → wrist with `shot_type_imputed`;
off_wing via y_norm>0 = shooter-left; garbage-strength flagged out of training; same 15-feature
list; same exclusions — no `xg`, `on_goal`, ids, teams, on-ice arrays, or position).

**Garbage strength states dropped (STOP threshold 0.5% — all far under):**

| Season | Rows dropped | % of scope | Goals dropped |
|---|---|---|---|
| 2024-25 | 18 | 0.016% | 2 |
| 2023-24 | 44 | 0.039% | 3 |
| 2022-23 | 24 | 0.021% | 3 |

**Same-second pairs — verified against NHL API play-by-play (the audit's 0 / 14 / 8 *rows*):**

| Season | Pairs | Two-real-events | True duplicates |
|---|---|---|---|
| 2024-25 | 0 | 0 | 0 |
| 2023-24 | 7 (=14 rows) | **7** | **0** |
| 2022-23 | 4 (=8 rows) | **4** | **0** |

Every pair is two distinct events (distinct `eventId`s with consecutive `sortOrder`s, and the SOG
counter increments by 1 or the two events are different types — e.g. 2023020320: missed-shot +
shot-on-goal; 2023020641: eids 509/510). **No true duplicates → no drop**, consistent with the
2025-26 item-A precedent.

**shot_type_id = 0 — API backfill attempted, 0 resolvable (same as 2025-26):**

| Season | Category-0 rows | All goals? | API-backfillable | Imputed to wrist |
|---|---|---|---|---|
| 2024-25 | 20 | ✅ | 0 | 20 |
| 2023-24 | 10 | ✅ | 0 | 10 |
| 2022-23 | 17 | ✅ | 0 | 17 |

All category-0 rows are goals the NHL feed logged with no `shotType`; the goal event carries no
type in the API for any of them, so all are imputed to the mode (wrist=1) and flagged
`shot_type_imputed`.

**Off-wing sign spot check (2 games/season vs raw API coordinates):** the normalization is
unchanged across seasons — `x_norm = |xCoord|` and **y_norm > 0 = shooter's LEFT** confirmed at
**100%** on the left side every season (2024-25 100/100, 2023-24 100/100, 2022-23 100% left /
98% right). The single 2022-23 "right-side" exception is a merge artifact on a same-second pair
(our shot is a centerline `y_norm=0` shot merged to the shooter's other same-second shot), **not a
sign flip.** No disagreement.

---

## Four-season side-by-side (training rows only)

| Metric | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| **Training rows** | **113,145** | **114,015** | **111,286** | 111,078 |
| **Goals** | **7,798** | **7,636** | **7,375** | 7,577 |
| **Goal rate** | **6.89%** | **6.70%** | **6.63%** | 6.82% |
| shot_type_imputed | 17 | 10 | 20 | 24 |
| Garbage dropped | 24 (0.021%) | 44 (0.039%) | 18 (0.016%) | 21 (0.019%) |
| **Combined 4-season training rows** | | | | **449,524** |
| **Combined goals** | | | | **30,386** |

**Strength-state mix (% of training rows):**

| Group | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| 5v5 | 78.3% | 78.3% | 80.1% | 78.7% |
| PP (shooter advantage) | 15.0% | 14.9% | 13.2% | 14.4% |
| PK (shooter short) | 2.5% | 2.5% | 2.2% | 2.2% |
| extra-attacker (6vX) | 2.0% | 2.2% | 2.5% | 2.3% |
| 4v4 / 3v3 | 2.2% | 2.1% | 2.1% | 2.3% |

**Shot-type mix (% of training rows):**

| Type | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| wrist | 52.9% | 54.5% | 48.4% | 43.2% |
| snap | 15.4% | 14.0% | 20.7% | 24.9% |
| slap | 12.2% | 11.9% | 11.7% | 11.9% |
| backhand | 7.4% | 7.1% | 7.0% | 7.2% |
| tip-in | 8.5% | 8.9% | 8.8% | 9.6% |
| deflected | 2.0% | 2.0% | 1.7% | 1.6% |
| wrap-around | 0.9% | 0.8% | 0.7% | 0.7% |

> **Note for whoever trains the pooled model:** there is a real season-to-season drift in shot-type
> *coding* — `wrist` falls from ~53% (2022-23) to ~43% (2025-26) while `snap` rises from ~15% to
> ~25%. This is an NHL scorer-classification shift, not a data defect, but it means `shot_type_id`
> is not stationary across seasons; a season indicator or shot-type-agnostic geometry may be worth
> considering. Strength-state and goal-rate mixes are stable.

---

## Files written (all new; nothing protected modified)
- `model/build_shots_enriched.py`, `model/build_xg_features.py` — generalized builders.
- `model/shifts_supplement_2024_25.parquet` — 57 recovered games (43,224 rows).
- `model/shots_enriched_2024_25.parquet`, `_2023_24.parquet`, `_2022_23.parquet`.
- `model/xg_features_2024_25.parquet`, `_2023_24.parquet`, `_2022_23.parquet`.
- `model/game_dates_2024_25.csv`, `_2023_24.csv`, `_2022_23.csv`.
- `model/player_handedness.csv` — appended 383 rows (existing 933 byte-identical).

## md5 integrity — all protected files unchanged
Re-hashed at the end and compared to the pre-start baseline: **all 22 protected files unchanged**
— `shots_augmented.parquet`, every `model/season_cache/*.parquet` (all shifts/shots/pp_shots
caches, including the three candidate seasons' caches), the four 2025-26 model artifacts
(`shots_enriched_2025_26`, `xg_features_2025_26`, `shots_xg_2025_26`, `xg_model_2025_26.json`), and
the four simulator CSVs (`composite_ratings_sim.csv`, `rapm_results.csv`, `team_strength.csv`,
`season_simulation_results.csv`). ✓

## Stop conditions — none triggered
- Step-1 equivalence: **exact** ✓
- Goalie misclassification: **none** (0/0/0) ✓
- Garbage strength loss > 0.5%: **no** (max 0.039%) ✓
- Off-wing sign disagreement: **none** (100% left-side every season) ✓
