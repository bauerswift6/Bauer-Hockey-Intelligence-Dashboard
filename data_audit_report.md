# Data Audit Report — Self-Generated Single-Season Analytics Layer

**Scope:** Read-only accuracy/completeness audit of the two raw inputs that feed the
xG / RAPM / QoC-QoT / goalie-GSAX rebuild:

- `model/shots_augmented.parquet`
- `model/season_cache/shifts_20252026.parquet`

Validated against the live NHL Public API (`api-web.nhle.com`) schedule/standings/gamecenter
endpoints. **No files were modified.** Composite/WAR/Contract layers and the simulator
pipeline were not read or touched.

**Audit date:** 2026-08-27 (NHL offseason — 2025-26 season fully concluded 2026-06-14;
2026-27 regular season has not yet begun).

---

## TL;DR — headline findings

| # | Finding | Severity |
|---|---------|----------|
| 1 | **`shots_augmented.parquet` is a 16-season training set (2010→2025), not single-season.** 1,750,736 rows / 20,527 games. The 2025-26 slice is 117,843 rows / 1,379 games. | Structural — expected, but any single-season rebuild MUST filter to `game_id ≥ 2025020001`. |
| 2 | **2025-26 regular season is 100% complete in both files (1312/1312 games).** | ✅ Clean |
| 3 | **Playoffs are incomplete in both files. Ingestion stopped ~2026-05-14 (Round 2).** Entire Conference Finals (R3) + Stanley Cup Final (R4) are absent — 15 games missing from shots, and shifts additionally drops the first 20 Round-1 games (35 missing total). | ⚠️ High — GSAX/RAPM will silently exclude the deepest playoff runs |
| 4 | **`shots_augmented.parquet` contains no `goalie_id`, no strength-state, and no on-ice player columns.** Goalie attribution (GSAX) and 5v5/PP/PK on-ice sets (RAPM, QoC/QoT) must be reconstructed by time-overlapping shots against the shifts file. | Structural — design constraint, not corruption |
| 5 | **Join integrity of the raw layer is clean.** 0 orphan shooters, 0 orphan shift players, 0 blank names, all 98 goalies present in shifts, all IDs resolve to player metadata. The `I_F_plusMinus`-style silent-zero failure does **not** exist in this raw layer. | ✅ Clean |
| 6 | xG-input null rates are **0.00%** across all model input columns; 12 duplicate shot rows (0.01%); team abbreviations fully consistent (32 franchises, Utah = `UTA`/id 68, no stale ARI/ATL). | ✅ Clean |

---

## 1. Season Coverage

### `shots_augmented.parquet` — MULTI-SEASON
This is not a single-season file. It holds **16 seasons, 2010-11 → 2025-26**:

- Total: **1,750,736 rows across 20,527 distinct games** (regular + playoffs each year).
- 2025-26 slice (`game_id` 2025020001–2025999999): **117,843 rows / 1,379 games**
  - Regular season: **1,312 distinct games** (game IDs 2025020001–2025021312)
  - Playoffs: **67 distinct games** (2025030111–2025030246)

### `shifts_20252026.parquet` — SINGLE-SEASON
- **1,021,535 rows / 1,359 distinct games**, all 2025-26.
  - Regular season: **1,312 distinct games**
  - Playoffs: **47 distinct games**

### Schedule reconciliation (live NHL API)
| Segment | League actual (API) | shots present | shifts present |
|---|---|---|---|
| Regular season | 1,312 games (`standings/now` ΣGP=2624÷2; seasonId 20252026) | **1,312 ✅** | **1,312 ✅** |
| Playoffs (final) | **82 games** (R1=45, R2=22, R3=9, R4=6) | 67 (−15) | 47 (−35) |

### Date ranges
| | First game | Last game |
|---|---|---|
| Regular season (both files, complete) | 2025-10-07 (`2025020001`, CHI @ FLA) | 2026-04-16 (`2025021312`, SEA @ COL) |
| Latest playoff game present (both files) | — | **2026-05-14** (`2025030246`, VGK @ ANA, Round 2) |
| Latest completed game per API | — | **2026-06-14** (`2025030416`, CAR @ VGK, SC Final G6) |

**The regular season is fully covered. The playoffs are truncated after Round 2 in both
files.**

---

## 2. Game-Level Completeness

### Missing games vs. full schedule
Regular season: **0 missing** in either file.

**Playoff games missing from `shots_augmented.parquet` (15):** all of Round 3 + Round 4.

| game_id | date | matchup |
|---|---|---|
| 2025030311–315 | 2026-05-21 → 05-29 | MTL @ CAR series (Conf Final) |
| 2025030321–324 | 2026-05-20 → 05-26 | VGK @ COL series (Conf Final) |
| 2025030411–416 | 2026-06-02 → 06-14 | VGK vs CAR (Stanley Cup Final, 6 games) |

**Playoff games missing from `shifts_20252026.parquet` (35):** everything above **plus**
the first 20 Round-1 games:

| game_id | date | matchup |
|---|---|---|
| 2025030111–116 | 2026-04-19 → 05-01 | BOS @ BUF (6 games) |
| 2025030121–127 | 2026-04-19 → 05-03 | MTL @ TBL (7 games) |
| 2025030131–134 | 2026-04-18 → 04-25 | OTT @ CAR (4 games) |
| 2025030141–143 | 2026-04-18 → 04-22 | PHI @ PIT (3 games) |
| + all R3 (9) and R4 (6) as listed above | 2026-05-20 → 06-14 | |

> Note the shifts file has a **non-contiguous** gap: it is missing Round-1 series 1–4 but
> *has* series 5–8 and all of Round 2. This is the signature of a partial/failed backfill
> rather than a clean "stopped at date X" cutoff.

### Low-event / partial-pull flags
- **Shots per game (2025-26):** mean 85.5, median 85, min 48, max 154.
  **0 games below the 40-attempt threshold.** No partial-pull shot games detected.
  (85/game ≈ league unblocked-attempt (Fenwick) rate — see note in §4 on blocked shots.)
- **Shifts:** distinct players per team-game mean 19.1 (min **18**, max 20); shift *rows*
  per team-game mean 375.8 (min 301, max 620). **0 team-games with an incomplete roster
  (<10 players).** Every present game has full 18-skater + goalie rosters on both sides.

### Row-count sanity
| File | Rows | Games | Per-game | Expectation | Verdict |
|---|---|---|---|---|---|
| shots (2025-26 slice) | 117,843 | 1,379 | 85.5 shots | ~85–95 unblocked attempts/gm | ✅ in range |
| shots (all seasons) | 1,750,736 | 20,527 | 85.3 | consistent across years | ✅ |
| shifts | 1,021,535 | 1,359 | 751.7 (≈376/team) | ~19 players × ~20 shifts × 2 | ✅ in range |

---

## 3. Join Integrity

**The raw layer joins cleanly. No silent zero-default failures found.**

| Check | Result |
|---|---|
| Distinct shooters in shots | 934 |
| Distinct players in shifts | 1,038 |
| Shooters **not** present in shifts | **0** |
| Shift players who never shoot | 104 (expected: goalies + low-usage skaters) |
| Shooters not resolving to player metadata (MoneyPuck `skaters.csv` — the source `app.py` uses for name/pos on Players pages) | **0** |
| Shift players not resolving to metadata | **0** |
| Shift players with blank/null embedded `first_name`/`last_name` | **0 of 1,038** |

**On the `I_F_plusMinus` pattern:** that bug is a *missing column in the composite layer*
(out of scope). At the **raw** layer audited here, there is no analogous silent-zero risk —
every `shooter_id` and every `player_id` resolves to a named, metadata-backed player. The
shifts file even carries `first_name`/`last_name` inline, so name rendering for
shift-derived stats needs no external join at all.

**Surfaced-vs-raw:** Composite leaderboards are out of scope, so "never surfaces on a
leaderboard" can't be evaluated against them here. Using the metadata source that backs
name rendering as the proxy: **0 raw player_ids fail to resolve**, and the 104 shift-only
(non-shooting) players are goalies and fringe skaters as expected — no orphans in either
direction.

### Goalie attribution (GSAX-critical)
- **`shots_augmented.parquet` has no `goalie_id` column.** Columns present:
  `x_norm, y_norm, distance_from_net, angle_from_center, shot_type_id, is_rush,
  is_rebound, time_since_last_shot, period, score_state, is_home, is_goal, shooter_id,
  on_goal, game_id, abs_secs, xg`.
- Therefore GSAX **must** reconstruct the facing goalie by intersecting each shot's
  `abs_secs` with the opposing team's goalie shift in `shifts_20252026.parquet`
  (`abs_start`/`abs_end`). `app.py`'s own comments confirm this is the intended design
  ("my xG model applied to shots faced via goalie shifts intersection").
- **Feasibility confirmed:** all **98** MoneyPuck-listed goalies are present in the shifts
  file (0 missing). The reconstruction join has full goalie coverage for every game that
  is present. The only gap is the missing playoff games in §2 — for those, no goalie shifts
  exist to attribute against.

---

## 4. Data Quality Spot Checks

### Duplicates
- **Shots:** 12 duplicate rows on (`game_id, shooter_id, abs_secs, x_norm, y_norm`) — 0.01%
  of the 2025-26 slice. Negligible but non-zero; worth a `drop_duplicates` in the rebuild.
- **Shifts:** **0** duplicates on (`game_id, player_id, period, abs_start, abs_end`).

### xG-input null rates (2025-26 slice)
| Column | Null % |
|---|---|
| x_norm, y_norm | 0.00% |
| distance_from_net | 0.00% |
| angle_from_center | 0.00% |
| shot_type_id | 0.00% |
| period | 0.00% |
| score_state | 0.00% |
| is_home | 0.00% |
| xg | 0.00% |

All model inputs are fully populated. **No column exceeds the "few percent" flag threshold.**

### Strength-state & on-ice fields (RAPM / QoC / QoT inputs)
- **There is no strength-state column in `shots_augmented.parquet`** (only `score_state`,
  which is score differential, not manpower). No `is_pp`/`is_pk`/`strength` field exists.
- **There are no on-ice player arrays in `shots_augmented.parquet`** (no `onIce`/`on_ice`
  columns; the only player reference is the single `shooter_id`).
- Consequence: the 5v5 / PP / PK strength state **and** the on-ice skater sets that RAPM and
  QoC/QoT require are **not present as fields** — they must be derived by overlapping each
  shot's `abs_secs` with all shift intervals in the shifts file (counting skaters per side
  → strength; enumerating players → on-ice set). Since shift rosters are complete (§2, min
  18 skaters+G per team), this reconstruction is feasible for every present game, but note
  it is a *derivation*, not a stored field, so its correctness lives in the rebuild code,
  not the raw data.

### Team abbreviation consistency
- **32 franchises, 32 team_ids — fully consistent, no stale/relocated duplicates.**
- Utah correctly present as `UTA` (team_id 68). **No `ARI`/`PHX`/`ATL` stragglers**, no
  double-coding of a single franchise. Abbrev set:
  ANA, BOS, BUF, CAR, CBJ, CGY, CHI, COL, DAL, DET, EDM, FLA, LAK, MIN, MTL, NJD, NSH, NYI,
  NYR, OTT, PHI, PIT, SEA, SJS, STL, TBL, TOR, UTA, VAN, VGK, WPG, WSH.

> Aside for the xG rebuild: 85 shots/game indicates these are **unblocked** attempts
> (Fenwick). If the xG model is intended to also weigh blocked attempts, note blocked shots
> are not in this file. (Standard practice excludes blocks from xG, so this is likely
> intentional — flagged only for confirmation.)

---

## 5. Freshness

- **Most recent completed NHL game (live API):** `2025030416`, **2026-06-14**, CAR @ VGK —
  Stanley Cup Final Game 6. The 2025-26 season is entirely finished; the 2026-27 season has
  not started, so there is no in-progress season to be "behind" on.
- **Most recent game present in our data:** `2025030246`, **2026-05-14** (Round 2), in
  *both* files.
- **Gap: ~31 days / 2 full playoff rounds (15 games).** The entire Conference Finals and
  Stanley Cup Final are absent.

### Pipeline status interpretation
- File mtimes: `shots_augmented.parquet` last written **2026-05-19**;
  `shifts_20252026.parquet` last written **2026-06-11**.
- The shots file's content (through May 14) and mtime (May 19) line up: it ran through
  mid-Round-2, then stopped.
- **The shifts file is the anomaly:** it was rewritten on **2026-06-11** (near the end of
  the Final) yet still contains nothing past May 14 *and* is missing the first 20 Round-1
  games. A job touched the file weeks after the missing games were played but did not
  backfill them — pointing to a **stalled/broken ingestion for playoff rounds**, not merely
  a scheduler that fell behind.
- Net: the ingestion pipeline appears **stopped since mid-May 2026**. Because the season has
  since ended, no live catch-up is possible via `schedule/now` — the missing 15 (shots) /
  35 (shifts) playoff games would need an explicit historical backfill against
  `gamecenter/{gameId}/play-by-play` and `/shiftcharts` for the specific IDs listed in §2.

---

## Recommendations for the rebuild (informational only — nothing changed)

1. **Filter shots to the target season** (`game_id ≥ 2025020001`) — the file is 16 seasons deep.
2. **Decide playoff scope before building.** If playoffs are in scope, backfill the missing
   IDs in §2 first; otherwise restrict the rebuild to the complete regular season (1312/1312).
3. **Reconstruct strength-state, on-ice sets, and facing goalie from the shifts file** —
   none exist as columns in `shots_augmented.parquet`. Roster completeness supports this for
   every present game.
4. Drop the 12 duplicate shot rows defensively.
5. Investigate why the 2026-06-11 shifts rewrite did not backfill playoffs — the pipeline is
   the root gap, not the data format.

---

# Reconciliation Note — On-Ice / Strength / Goalie Enrichment (2026-09-13)

Follow-up build that fills the three fields the audit found missing from
`shots_augmented.parquet`, by time-overlapping shots against
`shifts_20252026.parquet`.

- **Script:** `model/build_shots_enriched_2025_26.py` (auditable, self-contained)
- **Output:** `model/shots_enriched_2025_26.parquet`
- **Scope:** regular season 2025-26 only (`game_id` 2025020001–2025021312). Playoffs
  (2025030xxx) never touched. Both source parquets untouched (read-only).

## Row-count parity ✅
| | Rows |
|---|---|
| Filtered input (`shots_augmented`, reg-season slice) | **111,896** |
| Output (`shots_enriched_2025_26.parquet`) | **111,896** |

Exact match — no shots dropped, only enriched. Output adds 17 columns:
`shooting_team_id, defending_team_id, skaters_for, skaters_against, strength_state,
empty_net_for, empty_net_against, goalie_id (Int64), goalie_for_id (Int64),
on_ice_for [list], on_ice_against [list]`, plus six `flag_*` edge-case columns.

## Method decisions

**Overlap rule — `abs_start < t <= abs_end` (exclusive start, inclusive end).**
A shot is a live-play event, so it belongs to the shift already in progress, not the
line arriving at the change. This was chosen empirically over the two naive rules:

| Rule | shooter-not-on-ice | shots inflated to >5 skaters |
|---|---|---|
| `[start, end)` (excl. end) | 5,871 | 2,494 |
| `[start, end]` (incl. both) | 2 | 12,803 |
| **`(start, end]` (chosen)** | **11** | **legit pulled-goalie only** |

The chosen rule credits the outgoing line at a boundary, keeping the shooter on the
ice for their own shot while not double-counting the changing line.

**Goalie identification — season-level set (no position field exists).** A player is a
goalie if their *mean* shift duration exceeds 180 s in **any** game. This sits in the
clean statistical gap (skater max mean 156.8 s ↔ goalie min mean 207.0 s) and reproduces
MoneyPuck's 98-goalie roster **exactly** (0 discrepancy). Season-level (not per-game) is
required: 7 goalies made brief relief appearances (e.g. Halverson, 1 shift / 6 s) that a
per-game rule would misclassify as skaters.

**Duplicate/overlapping shift rows de-duped.** 146 raw shift rows (0.01%, 59 games) are
self-overlapping — the same player with two rows both covering an instant. On-ice sets
are counted by **distinct** `player_id`, which collapses impossible states (7v6, 8v7)
and cut `two_goalies_against` from 201→8 and >6-skater rows from ~2,500→12.

## Edge-case counts (from the build summary)

| Edge case | Count | Handling |
|---|---|---|
| Shots exactly on a shift boundary (`t == start` or `t == end`) | 12,090 | Resolved deterministically by the `(start, end]` rule above |
| Zero shifts found for a team (data gap) | 797 | **All are period-5 shootout rows** (`abs_secs`=4800, no ice time). Flagged `flag_no_shifts_*`, `strength_state`="0v0", `goalie_id`=NULL — never defaulted |
| Two goalies on-ice for defending team at once | 8 | Resolved to the goalie whose shift has the longer duration at `t`; flagged `flag_two_goalies_against` |
| Shooter not in on-ice set at `t` | 808 | 797 are the shootout rows + 11 true source quirks; flagged `flag_shooter_not_onice` |
| >6 distinct skaters a side (residual raw anomaly) | 12 | Flagged `flag_skaters_gt6` for inspection (0.01%) |

## Empty-net handling ✅ (no misattribution)

- `empty_net_for` = 3,461 (shooting team's own goalie pulled — the pressing team firing
  at a defended net; strength 6v5 / 6v4). These are valid scoring chances and correctly
  keep a defending `goalie_id`.
- `empty_net_against` = 797, **all period-5 shootout**. Regulation empty-net-against is
  **0** — and that is correct, not a bug: `shots_augmented` is an xG-oriented dataset and,
  per standard xG construction, excludes shots taken *at* an empty net. Confirmed
  independently: 334 team-games have a genuine late goalie pull (median pull at 58:29),
  yet the leading team registers **0** shot events inside those pulled-goalie windows.
- `goalie_id` is NULL for exactly the 797 no-coverage rows and for **0** rows where a
  goalie was on ice — i.e. no bench/backup goalie was ever misattributed to an empty net.

## Validation results ✅

- **Row parity:** 111,896 == 111,896 (above).
- **Skater-count bounds:** total distinct players per side (skaters + goalie), excluding
  the 797 shootout rows — 105,625 shots sit at exactly 6 (5 skaters + goalie); the rest at
  4–5 (short-handed / pulled goalie). Only 14 (for) / 19 (against) rows exceed 6 total, all
  flagged. No side has an impossible roster in the 99.98% mainline.
- **Strength sequencing, 10 games spanning the season** (2025020001 → 2025021312): every
  game shows a realistic profile — 54–69 5v5 shots, power-play (5v4) shots 7–21, penalty-kill
  (4v5) shots present, and empty-net-for shots (0–9) scaling with game situation. Sample:

  | game_id | shots | 5v5 | PP (5v4) | PK (4v5) | EN-for |
  |---|---|---|---|---|---|
  | 2025020001 | 82 | 69 | 9 | 2 | 2 |
  | 2025020330 | 98 | 61 | 19 | 4 | 9 |
  | 2025020990 | 85 | 56 | 21 | 5 | 3 |
  | 2025021312 | 85 | 64 | 14 | 2 | 2 |

**Status:** `shots_enriched_2025_26.parquet` is ready to serve as the on-ice / strength /
goalie foundation for the xG, RAPM, QoC-QoT, and GSAX rebuilds. Downstream consumers should
exclude the 797 period-5 shootout rows (flagged) and may inspect the ~12–19 flagged
raw-shift anomalies if strict roster integrity is required.

---

# Correction Note — "12 duplicate shot rows" re-checked (2026-09-28)

The Phase-0 audit (§4) flagged **12 duplicate shot rows** on the key
(`game_id`, `shooter_id`, `abs_secs`, `x_norm`, `y_norm`). During the xG rebuild these were
re-checked against NHL API play-by-play. **They are NOT duplicates — all 12 are genuine,
distinct events** (rapid-fire same-second shots). Reporting only; no rows were dropped.

**Reconciliation of the 12 (2025-26 slice, incl. playoffs, 5-column key):**
- **10** coincide with the regular-season pairs verified in the xG rebuild's item A
  (games 2025020165, …0410, …0478, …0555, …0622, …0899, …0937, …1092, …1243, …1287). Each is
  two distinct `eventId`s with consecutive `sortOrder`s and the official SOG counter incrementing
  by 1 (or a shot-on-goal + a missed-shot). Genuine shot + immediate rebound.
- **1 extra regular-season group** (game **2025020323**, shooter 8480002, 16:07 P1, coord (83,6)):
  two events — eventId 302 `backhand` and 1291 `deflected` (awaySOG 11→12). **Different shot
  types ⇒ unambiguously two shots.** It did not appear in item A's 10 only because item A used a
  6-column key that includes `shot_type_id`, making this a "near-match" rather than an exact-key
  match. Both rows are retained as separate shots.
- **1 playoff group** (game **2025030215**, shooter 8481523): eventId 143 and 401, both `snap`
  shot-on-goal, consecutive `sortOrder`, awaySOG 5→6. Two real shots. Outside the regular-season
  rebuild scope, so not in the modeling set regardless.

**Conclusion:** the "12 duplicate rows" were a false positive of the coordinate/second-level key.
The NHL feed genuinely records multiple shots by the same player at the same coordinate within a
single (1-second-resolution) clock tick. In the 2025-26 regular-season modeling set, **no
duplicate drop is applied** — all such rows are kept as distinct events. See
`xg_model_report.md` item A for the full per-pair evidence.
