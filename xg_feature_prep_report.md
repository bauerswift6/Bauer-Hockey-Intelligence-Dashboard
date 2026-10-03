# xG Feature Prep Report — Part 1 Checkpoint (2025-26 Regular Season)

**Task:** Feature preparation and verification for the from-scratch single-season xG model.
No model was trained in Part 1. No protected file was modified (md5 check at the end).

**Source:** `model/shots_enriched_2025_26.parquet`
**Scope filter:** `game_id` ∈ [2025020001, 2025021312], exclude period-5 / shootout rows
(`abs_secs == 4800`, `goalie_id` NULL).
**Filtered row count: 111,099 — exactly matches the expected 111,099 (diff 0).**
0 period-5 rows and 0 null-goalie rows remain after the filter.

---

## Final numbers (after cleaning)

| Quantity | Value |
|---|---|
| Filtered rows (pre-clean) | 111,099 |
| Dropped — exact duplicates | 10 |
| Dropped — garbage strength states | 21 |
| Overlap between the two drop sets | 0 |
| **Final modeling rows** | **111,068** |
| **Goals** | **7,577** |
| **Goal rate** | **0.0682 (6.82%)** |

---

## 1. Duplicates

Exact-duplicate key = (`game_id`, `abs_secs`, `shooter_id`, `x_norm`, `y_norm`, `shot_type_id`).

- **10 duplicate groups (20 rows, all pairs). 0 are goals.** Plan: drop 10, keep one copy each.
- **Caveat worth noting:** within each of these 10 groups the two rows are *not* byte-identical —
  they differ in `time_since_last_shot` and `xg` (one group also in `on_goal`). That is the
  signature of two genuine shot attempts logged at the same clock-second and coordinate (a shot
  + immediate rebound / second whack), not a pure data-duplication. By the exact-duplicate key
  you specified they qualify, so I plan to drop them (keep first), but flagging that this is a
  defensible-but-debatable 10-row call (0 goals, 0.009% of rows — negligible either way).
- **Near matches (kept, not dropped):** ~319 rows share (`game_id`, `abs_secs`, `shooter_id`) but
  differ in x/y and/or shot type — genuine distinct-location shots in the same second (e.g. game
  2025020014 @ 3032s: one shot at (75,4) type snap, another at (77,5) type wrist). These are
  **kept**.

## 2. Shot Type Mapping

Mapping found in **`model/train_xg.py:93`** (`SHOT_TYPE_MAP`, the script that built
`shots_augmented.parquet`). ID 0 is the fallback for a missing/unrecognized `shotType`.

| id | label | rows | goals | goal rate |
|---|---|---|---|---|
| 0 | unknown | 24 | 24 | **100.00%** ⚠ |
| 1 | wrist | 47,961 | 2,960 | 6.17% |
| 2 | snap | 27,638 | 2,228 | 8.06% |
| 3 | slap | 13,204 | 640 | 4.85% |
| 4 | backhand | 8,034 | 662 | 8.24% |
| 5 | tip-in | 10,666 | 720 | 6.75% |
| 6 | deflected | 1,827 | 211 | 11.55% |
| 7 | wrap-around | 731 | 31 | 4.24% |
| 8 | poke | 371 | 41 | 11.05% |
| 9 | bat | 563 | 52 | 9.24% |
| 10 | cradle | 8 | 1 | 12.50% |
| 11 | between-legs | 72 | 8 | 11.11% |

**NHL API cross-check:** for 6 regular-season games (2025020001, …007, …100, …500, …1000, …1312),
matched every shot to `/v1/gamecenter/{id}/play-by-play` by (game, period, time-in-period, shooter)
and compared our `shot_type_id` to the API `shotType` through the map. **516/516 shots matched,
0 not found, 100.0% agreement. No disagreements** between the codebase mapping and the API.

**⚠ Flag (not a mapping error):** `shot_type_id == 0` (unknown) is 24 rows and **all 24 are goals**
(100%). These are goals the API logged without a `shotType`. As a categorical level the model could
memorize "unknown ⇒ goal," a mild leakage-ish artifact. Left as-is for now (24 rows); raising it so
you can decide in Part 2 whether to fold level 0 into another category or handle it specially.

## 3. Shooter Handedness

No local file stored `shoots`/`catches`. Fetched `shootsCatches` from
`/v1/player/{id}/landing` for **all 933 distinct shooters** (throttled, 8 workers, ~40s),
cached to **`model/player_handedness.csv`** (`player_id, shoots`).

- **Coverage: 933 / 933 resolved (100%). 0 unknown.** L = 583, R = 350.
- **Shots with an unknown-handedness shooter: 0.** No defaulting needed; `off_wing` has no nulls.

## 4. Off-Wing Sign Convention

Off-wing = left-shot from the shooter's **right** side, or right-shot from the shooter's **left**
side (shooter facing the net at (89, 0), `y_norm` signed).

**Check (a) — geometric trace from raw NHL API coordinates (5 games, 431 joined shots):**
- `x_norm` = |`xCoord`| (corr 1.000) — the play is reflected/rotated so every shot attacks +x.
- Attacking +x: `y_norm` = `yCoord` (corr 1.000). Attacking −x: `y_norm` = −`yCoord` (corr 1.000).
  (i.e. a 180° rotation for shots attacking the −x net — both x and y flip.)
- Assigning shooter-left in the raw frame (facing the attacking net, left = +y when attacking +x,
  left = −y when attacking −x) and mapping to `y_norm`: **100% of shooter-LEFT shots have
  `y_norm > 0`; 0% of shooter-RIGHT shots have `y_norm > 0`** (mean `y_norm` = +21.6 vs −21.4).

**Check (b) — known-direction PP one-timer pattern (behavioral corroboration):**
- Share of each handed group's lateral shots taken from the LEFT side (`y_norm > 0`):

| Situation | L-shot % left | R-shot % left |
|---|---|---|
| 5v5 (even strength) | 62.3% | 32.4% |
| 5v4 (power play) | 41.1% | **60.1%** |

  At even strength handed shooters skew to their **on-wing** (L-shots left, R-shots right); on the
  power play they **flip to their off-wing** for one-timers (R-shots move to the left circle, L-shots
  to the right) — the classic, expected PP setup. Consistent with (a). (Off-wing PP lateral shots
  also convert slightly higher: 7.91% vs 7.38% on-wing.)

**Conclusion (both checks agree): `y_norm > 0` = shooter's LEFT side; `y_norm < 0` = shooter's RIGHT.**
→ `off_wing = (shoots == 'L' & y_norm < 0) | (shoots == 'R' & y_norm > 0)`;
shots on the centerline (`y_norm == 0`) are treated as not-off-wing (0).

## 5. Strength State

Built `skaters_for`, `skaters_against` (shooter's perspective). Full table with keep/drop:

| for | against | rows | goals | class |
|---|---|---|---|---|
| 5 | 5 | 87,382 | 5,368 | **KEEP** |
| 5 | 4 | 15,134 | 1,447 | **KEEP** |
| 4 | 5 | 2,455 | 175 | **KEEP** |
| 6 | 5 | 2,313 | 184 | **KEEP** (pulled goalie, 100% empty_net_for) |
| 3 | 3 | 1,346 | 173 | **KEEP** |
| 4 | 4 | 1,234 | 82 | **KEEP** |
| 5 | 3 | 543 | 89 | **KEEP** |
| 4 | 3 | 348 | 36 | **KEEP** |
| 6 | 4 | 293 | 22 | **KEEP** (pulled goalie, 98% empty_net_for) |
| 3 | 4 | 20 | 1 | **KEEP** (real state, per spec) |
| 3 | 5 | 10 | 0 | **KEEP** (real state, per spec) |
| 5 | 6 | 6 | 0 | DROP — defending team 6 skaters; no real empty-net-against shots exist in this set |
| 6 | 7 | 7 | 0 | DROP — count > 6 |
| 6 | 6 | 2 | 0 | DROP — both goalies pulled, impossible |
| 7 | 10 | 2 | 0 | DROP — counts > 6 |
| 6 | 8 | 1 | 1 | DROP — count > 6 |
| 7 | 6 | 1 | 0 | DROP — count > 6 |
| 10 | 5 | 1 | 0 | DROP — count > 6 |
| 6 | 3 | 1 | 0 | DROP — 3-skater differential impossible |

- **Garbage: 8 combos, 21 rows, 1 goal** — dropped. (12 of the 21 are also flagged by the
  enrichment's own `flag_skaters_gt6`; none had `flag_two_goalies_against`.)
- **Kept: 11 combos, 111,078 rows, 7,577 goals.** 3v4 and 3v5 kept as instructed.
- Drop rule: count outside 3–6, OR |for − against| > 2, OR both = 6, OR against = 6 (would be
  empty-net-against, which the enrichment established does not occur in this set → spurious 6th skater).

## 6. Score State Perspective

Reconstructed the running score differential from the **shooting team's** perspective (capped ±3)
by replaying goals in `abs_secs` order, and compared to the `score_state` column.

- **5 games spot-checked (2025020001, …007, …100, …500, …1000): 100.0% match in every game.**
- Confirmed: `score_state` is already **shooter-perspective** (positive = shooter's team leading)
  and already **capped at ±3**. **No conversion needed.**

## 7. Final Feature List

Geometry confirmed exact: `distance_from_net == hypot(89 − x_norm, y_norm)` and
`angle_from_center == atan2(|y_norm|, 89 − x_norm)` — max abs diff **0.0** on both.

| Feature | dtype | nulls |
|---|---|---|
| distance_from_net | float64 | 0 |
| angle_from_center | float64 | 0 |
| x_norm | float64 | 0 |
| abs_y_norm ( = \|y_norm\| ) | float64 | 0 |
| shot_type_id | category (12 levels) | 0 |
| is_rebound | int64 | 0 |
| time_since_last_shot | float64 | 0 |
| is_rush | int64 | 0 |
| score_state (shooter persp., ±3) | int64 | 0 |
| period (1–3 reg, 4 = OT) | int64 | 0 |
| time_in_period ( = abs_secs − (period−1)·1200 ) | int64 | 0 |
| is_home (shooter's team) | int64 | 0 |
| skaters_for | int64 | 0 |
| skaters_against | int64 | 0 |
| off_wing (1/0; null if handedness unknown — none here) | float64 | 0 |
| **is_goal** (target) | int64 | 0 |

`off_wing` distribution: 46,238 off-wing (1), 64,830 on-wing/centre (0), 0 null.
`time_in_period` range 2–1200 s; `period` ∈ {1,2,3,4}; `score_state` ∈ [−3, 3].

**Exclusion check — confirmed NONE of these leaked into the feature table:**
`xg` (prior model), `on_goal`, `shooter_id`, `goalie_id`, `goalie_for_id`, `shooting_team_id`,
`defending_team_id`, `team`, `game_id`, `on_ice_for`, `on_ice_against`, shooter position. ✓

---

## md5 integrity check (start vs end of Part 1 — all unchanged)

| File | md5 (start == end) |
|---|---|
| model/shots_augmented.parquet | `47af38575046913c9699837bd3a1cd44` ✓ |
| model/season_cache/shifts_20252026.parquet | `919143f2d1eba4e93c81a87759940984` ✓ |
| model/shots_enriched_2025_26.parquet | `23bb76c3f95ca7146bbb9ac2ed7a11e7` ✓ |
| model/composite_ratings_sim.csv | `d02630fd68286488dd951351c84dc85c` ✓ |
| model/rapm_results.csv | `53cc52783cb279da0c8f9ee9c07f5637` ✓ |
| model/team_strength.csv | `136632ddafc0d6234045ac2202594d55` ✓ |
| model/season_simulation_results.csv | `06ef68fffb575ad505262ce7f3ed1e20` ✓ |

New file written this part (allowed / required by step 3): **`model/player_handedness.csv`**.

---

## Choices I made that you did not explicitly specify
1. **Duplicate tie-break = keep first.** The 10 dup pairs differ in `time_since_last_shot`/`xg`;
   I keep the first occurrence (0 goals, negligible).
2. **`shot_type_id` kept as a 12-level categorical** (including level 0 = unknown). Flagged the
   0-level 100%-goal anomaly for your Part 2 decision rather than silently recoding it.
3. **`abs_y_norm` = |y_norm|** used for the "abs(y_norm)" feature; kept `x_norm` signed-free
   (it is already 0–99). Signed side information is carried by `off_wing`, not by raw y.
4. **`off_wing` for centreline shots (`y_norm == 0`) = 0** (not-off-wing), not null.
5. **`skaters_against == 6` treated as garbage** (dropped), justified by the enrichment finding of
   zero real empty-net-against shots — so a defending "6" is a spurious extra skater.
6. Cleaned modeling table held in memory / scratch only — **not** written to `model/` (Part 2 owns
   the output parquets); it will be re-derived deterministically in Part 2.

---

**STOP — Part 1 complete. Awaiting your go-ahead before starting Part 2 (training & validation).**
