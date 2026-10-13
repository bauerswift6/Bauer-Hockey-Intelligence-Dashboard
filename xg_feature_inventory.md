# xG Feature Inventory — 2025-26 Regular Season

**Purpose:** Inventory of what is *actually available* in the shot data before designing
the single-season xG model. No features engineered, nothing trained, no files modified.

**Source:** `model/shots_enriched_2025_26.parquet` — 111,896 rows (17 original + 17
enrichment columns).
**Analysis set:** 797 flagged shootout rows (period 5, `abs_secs`=4800) excluded →
**111,099 shots** across **1,312 games**. All figures below are on this core set.

---

## 0. Two findings to read first

1. **`xg` is already a column** — it is a *prior xG model's output* baked into the file
   (range 0.000–0.967, mean 0.070). It is **not** an input feature. Treat it as a
   baseline to benchmark the rebuild against, and **exclude it from any feature set** to
   avoid target leakage.
2. **Goal rate depends on the denominator.** Over all shot attempts in this file the goal
   rate is **6.82%**; over shots *on goal* (`on_goal==1`) it is **10.44%** — the latter is
   the standard shooting-% sanity check and lands squarely in the typical 9–11% band. The
   file is an unblocked-attempt (Fenwick-style) set: **34.7%** of rows are off-target
   (`on_goal==0`). Decide up front whether the model scores all attempts or on-goal only.

---

## 1. Full Column Inventory

Null rate is over the 111,099 core rows. "Card." = distinct value count.

| # | Column | dtype | Null % | Card. | 5 sample values |
|---|--------|-------|--------|-------|-----------------|
| 1 | `x_norm` | float64 | 0.00% | 100 | 82, 5, 28, 77, 54 |
| 2 | `y_norm` | float64 | 0.00% | 85 | 9, 40, -35, -3, -13 |
| 3 | `distance_from_net` | float64 | 0.00% | 2082 | 11.40, 93.04, 70.33, 12.37, 37.34 |
| 4 | `angle_from_center` | float64 | 0.00% | 2342 | 0.910, 0.444, 0.521, 0.245, 0.356 (radians) |
| 5 | `shot_type_id` | int64 | 0.00% | **12** | 1, 1, 3, 5, 2 (integer code; see §3) |
| 6 | `is_rush` | int64 | 0.00% | 2 | 0, 1, 0, 0, 0 |
| 7 | `is_rebound` | int64 | 0.00% | 2 | 0, 0, 0, 0, 0 |
| 8 | `time_since_last_shot` | float64 | 0.00% | 379 | 600, 65, 50, 20, 102 (seconds, capped 600) |
| 9 | `period` | int64 | 0.00% | 4 | 1, 2, 3, 4 |
| 10 | `score_state` | int64 | 0.00% | **7** | -3…+3 (capped) |
| 11 | `is_home` | int64 | 0.00% | 2 | 0, 1 |
| 12 | `is_goal` | int64 | 0.00% | 2 | 0, 1 — **target** |
| 13 | `shooter_id` | int64 | 0.00% | 933 | 8477960, 8476879, … |
| 14 | `on_goal` | int64 | 0.00% | 2 | 0, 1 |
| 15 | `game_id` | int64 | 0.00% | 1312 | 2025020007, … |
| 16 | `abs_secs` | int64 | 0.00% | 3888 | 44, 109, 159, 179, 281 (absolute game seconds) |
| 17 | `xg` | float64 | 0.00% | 35391 | 0.106, 0.0, 0.009, … — **prior model output, do not use as feature** |
| 18 | `shooting_team_id` | int64 | 0.00% | 32 | 26, 54, … |
| 19 | `defending_team_id` | int64 | 0.00% | 32 | 54, 26, … |
| 20 | `skaters_for` | int16 | 0.00% | 6 | 3–6 (5 sample: 5,5,5,5,5) |
| 21 | `skaters_against` | int16 | 0.00% | 7 | 3–10 (5 sample: 5,5,5,5,4) |
| 22 | `strength_state` | str | 0.00% | **19** | 5v5, 5v4, … (full list in §5) |
| 23 | `empty_net_for` | bool | 0.00% | 2 | True/False (2,664 True = shooting team pulled goalie) |
| 24 | `empty_net_against` | bool | 0.00% | **1** | all **False** in core (see note) |
| 25 | `goalie_id` | Int64 | 0.00% | 98 | 8478499, … — facing goalie |
| 26 | `goalie_for_id` | Int64 | 2.40% | 98 | 8476341, … (null = shooting team's net empty) |
| 27 | `on_ice_for` | list[int] | 0.00% | — | `[8471685, 8476441, 8477960, 8482730, 8483808]` (len 3–10) |
| 28 | `on_ice_against` | list[int] | 0.00% | — | `[8475191, 8475913, 8476448, 8478396, 8480727]` (len 3–10) |
| 29 | `flag_shift_boundary` | bool | 0.00% | 2 | 12,090 True (10.88%) |
| 30 | `flag_no_shifts_for` | bool | 0.00% | 1 | 0 True (all shootout, excluded) |
| 31 | `flag_no_shifts_against` | bool | 0.00% | 1 | 0 True |
| 32 | `flag_two_goalies_against` | bool | 0.00% | 2 | 8 True (0.01%) |
| 33 | `flag_shooter_not_onice` | bool | 0.00% | 2 | 11 True (0.01%) |
| 34 | `flag_skaters_gt6` | bool | 0.00% | 2 | 12 True (0.01%) |

**Notes**
- `empty_net_against` is entirely False on the core set: the only True rows were the 797
  shootout rows now excluded. Regulation shots *at* an empty net are absent by design of the
  underlying xG dataset (documented in the reconciliation note). No shot in the modeling set
  faces an empty net — the facing goalie (`goalie_id`) is always present.
- `on_ice_for` / `on_ice_against` hold **distinct skater** player_ids (goalies excluded;
  they are in `goalie_id`/`goalie_for_id`). List lengths reach 10 only for the ~12–19
  flagged raw-shift anomalies; normal values are 3–6.

---

## 2. Geometric Features

**Both raw coordinates AND pre-computed distance/angle are present — and the derived
fields are exact.**

- `x_norm` ∈ [0, 99], `y_norm` ∈ [-42, 42]. **Coordinates are normalized to a single
  half of the rink** — every shot is folded onto one end with the net at **(89, 0)** (all
  `x_norm` are non-negative; 0 rows are negative or >100). So shots at either net are
  already on a common frame; **no per-net normalization is needed**.
- `distance_from_net` is **exactly** `hypot(89 − x_norm, y_norm)` (max abs diff vs. raw =
  `0.0` across all rows). Range 1.0–98.0 ft, mean 32.9.
- `angle_from_center` is **exactly** `atan2(|y_norm|, 89 − x_norm)` (max abs diff `0.0`),
  in **radians**, range 0–π. Behind-the-net shots (`x_norm` > 89, 1,638 rows) all correctly
  carry angle > π/2.

**Implication:** distance and angle are already engineered and consistent, so no geometry
work is required. `x_norm`/`y_norm` are also available if you want raw coordinates,
interaction terms, or a spatial (grid / smoothed) treatment. Note that because `y` is
signed but `angle` uses `|y|`, left/right asymmetry is preserved in `y_norm` but folded out
of `angle_from_center` — model off `y_norm` if you want side information.

---

## 3. Shot Context Features

| Feature | Field(s) in data | Status | Completeness |
|---|---|---|---|
| Shot type | `shot_type_id` | **Direct** — integer code, 12 categories | 100% (24 rows code 0 = unspecified) |
| Rebound | `is_rebound` (binary) **and** `time_since_last_shot` (sec) | **Direct** | 100% |
| Rush | `is_rush` (binary) | **Direct** (already engineered) | 100% |
| Time since zone entry | — | **Absent** — would need zone-entry events, which the public play-by-play does not carry (see §6) | — |
| Score state | `score_state` (−3…+3) | **Direct** | 100%, **capped at ±3** (blowouts collapsed) |
| Period | `period` (1–4) | **Direct** | 100% |
| Time in period | `abs_secs` (absolute game seconds) | **Trivially engineered**: `abs_secs − (period−1)·1200` → 0–1200 (P4/OT 0–299) | 100% |
| Home / away | `is_home` (binary) | **Direct** | 100% (56,694 home / 54,405 away) |

**Detail:**
- **`shot_type_id`** distribution (code → count): 1→47,961 · 2→27,638 · 3→13,204 ·
  5→10,666 · 4→8,034 · 6→1,827 · 7→731 · 9→563 · 8→371 · 11→72 · 0→24 · 10→8.
  The **human-readable label mapping is NOT in the file** — the codes correspond to NHL API
  shot types (wrist/slap/snap/backhand/tip/deflection/wrap-around, etc.) and must be joined
  from an external lookup if you want named categories. Code 0 (24 rows) is unspecified.
- **`is_rush` / `is_rebound`** are pre-derived binaries (5,059 rush; 9,051 rebound).
  `time_since_last_shot` (0–600 s, capped) is the continuous companion to `is_rebound`.
- **`score_state`** is from the shooter's perspective and **clipped to ±3** — you cannot
  distinguish "up 3" from "up 5".

---

## 4. Target Variable

- **Column:** `is_goal` (int64, 0/1). Fully populated, no nulls.
- **Encoding:** 1 = goal, 0 = non-goal. Consistency check passed: **0** rows have
  `is_goal==1` with `on_goal==0` (every goal is also on goal).
- **Counts:** 7,578 goals / 103,521 non-goals.
- **Goal rate (all attempts):** 7,578 / 111,099 = **6.82%**.
- **Goal rate (shots on goal only, `on_goal==1`):** 7,578 / 72,554 = **10.44%** — within
  the typical 9–11% league range. ✅

Use `on_goal` deliberately: modeling all unblocked attempts gives ~6.8% base rate;
restricting to shots on goal gives ~10.4%. `on_goal` itself is a strong feature but is only
known *after* the shot outcome, so treat it as a filter/target-scope choice, not a
predictor.

---

## 5. Strength State — full row counts (nothing collapsed)

| strength_state | rows | | strength_state | rows |
|---|---|---|---|---|
| **5v5** | 87,382 | | 4v3 | 348 |
| **5v4** | 15,134 | | 3v4 | 20 |
| **4v5** | 2,455 | | 3v5 | 10 |
| **6v5** | 2,313 | | 6v7 | 7 |
| **3v3** | 1,346 | | 5v6 | 6 |
| **4v4** | 1,234 | | 6v6 | 2 |
| **5v3** | 543 | | 7v10 | 2 |
| **6v4** | 293 | | 7v6, 10v5, 6v8, 6v3 | 1 each |

Read from the shooter's perspective (`skaters_for` v `skaters_against`; goalie excluded).

- **Dense enough to model alone:** 5v5 (87k), 5v4 (15k), 4v5 (2.5k), 6v5 (2.3k pulled-goalie
  offense), 3v3 (1.3k OT), 4v4 (1.2k).
- **Sparse — likely need pooling:** 5v3 (543), 6v4 (293), 4v3 (348).
- **Too sparse / anomalous — 3v4, 3v5, and everything ≤7 rows** (6v7, 5v6, 6v6, 7v10, 7v6,
  10v5, 6v8, 6v3). The last group (7v10, 10v5, 6v8, etc.) are the ~12–19 flagged raw-shift
  anomalies, not real game states — filter via the `flag_*` columns.

No collapsing applied here, per request — just the distribution.

---

## 6. Missing But Expected (the ceiling)

Features that MoneyPuck / Evolving-Hockey / Natural Stat Trick style models commonly use
that are **not present in this data at all**:

**Pre-shot movement & passing (biggest gap)**
- **Pass / cross-ice pass before shot ("Royal Road", cross-slot)** — absent. The public
  play-by-play carries no pass, carry, or zone-entry events (confirmed in the data audit),
  so one-timer detection and pre-shot movement cannot be built.
- **Shot angle change / rebound geometry** (velocity of angle between consecutive shots,
  which drives MoneyPuck's rebound term) — not available. We have `is_rebound` and
  `time_since_last_shot`, but **not** the location/angle of the previous shot, so the
  angular delta cannot be computed.
- **Time / distance since zone entry** — absent (no zone-entry events).

**Prior-event context**
- **Type of the event immediately preceding the shot** (faceoff, giveaway, takeaway, hit,
  blocked shot) — absent. No preceding-event stream.
- **Faceoff-following flag** and faceoff win/zone — absent.
- **Time since last event of any kind** — only `time_since_last_shot` (shots), not general
  events.

**Shooter / on-ice attributes**
- **Shooter handedness (L/R) and off-wing flag** — absent (no handedness field; would need
  an external roster join on `shooter_id`).
- **Shooter position (F/D)** — absent. On-ice arrays give player_ids but no positions.
- **Player-level shooting talent / shooter fixed effect** — not a column, but *derivable*
  from `shooter_id` (933 shooters present) if you add a shooter term.

**Shot physical context**
- **Screen / traffic / goalie-sightline indicators** — absent.
- **Deflection vs. clean shot beyond the `shot_type_id` code** — only whatever the shot-type
  code encodes; no separate screen/tip/traffic flags.
- **Blocked-shot indicator / shot distance-to-blocker** — the set appears to be unblocked
  attempts only; there is no explicit blocked flag (`on_goal` splits on-goal vs. missed).

**Already present (so NOT missing, for reference)**
- Distance, angle, raw x/y, shot type, rebound flag + recency, rush flag, score state,
  period, time, home/away, strength state, empty-net flags, facing goalie, and full on-ice
  skater arrays.

**Bottom line on the ceiling:** geometry + basic shot context + strength state + on-ice
personnel are all solid and complete. The model's ceiling is limited primarily by the
**absence of pre-shot movement/passing data and preceding-event context**, plus
**shooter handedness/position**, which together are the main features separating public
"advanced" xG models from a distance/angle/context baseline. None of these are recoverable
from this file alone; handedness/position could be added via an external roster join on
`shooter_id`, but passing/zone-entry/pre-shot-movement cannot be recovered from the public
play-by-play at all.
