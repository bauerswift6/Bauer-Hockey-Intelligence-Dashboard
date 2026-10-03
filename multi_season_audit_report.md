# Multi-Season xG Training-Data Readiness Audit

**Question:** can 2024-25, 2023-24, 2022-23 (regular season) be added to the 2025-26 xG model,
given the same Phase-0 audit + Phase-1 enrichment treatment 2025-26 received?

**Diagnostic only.** No file was modified or created except this report. All NHL API calls were
read-only; nothing was written to disk or any cache.

Read first: `data_audit_report.md`, `xg_model_report.md`, `model/build_shots_enriched_2025_26.py`,
`model/build_xg_features_2025_26.py`, plus `model/train_xg.py` and `model/train_rapm.py`.

---

## Verdict summary

| Season | Shots | Shifts | Verdict | Concrete fix |
|---|---|---|---|---|
| **2023-24** | 1,312 / 1,312 ✅ | 1,312 / 1,312 ✅ | **READY** | Parameterize 2 builders (constants only). Fetch handedness for 208 players; build a game-date cache. **0 games to fetch.** |
| **2024-25** | 1,312 / 1,312 ✅ | 1,255 / 1,312 ⚠ (57 missing) | **NEEDS FETCH** | Fetch **57** contiguous games' shifts (2024021235–2024021291) via the existing HTML TOI fallback (JSON endpoint returns empty; HTML reports are HTTP 200). Then parameterize + enrich. Handedness: 130 players. Re-validate goalie threshold (thin margin). |
| **2022-23** | 1,310 / 1,312 ⚠ (2 missing) | 1,310 / 1,310 present ✅ | **NEEDS FETCH (minor)** | Fetch **2** games (2022020544, 2022020843) — absent from `shots_augmented`; their shifts are available via the JSON API. Otherwise complete. Handedness: 310 players. Near-ready (1,310/1,312 = 99.85%). |

**Shared enrichment infrastructure already exists and is season-agnostic**: shift caches for all
three seasons are present with the exact 2025-26 schema, the shot table is one multi-season file,
the shot-type mapping is identical, and the HTML shift fallback + JSON fetch live in
`model/train_rapm.py`. No new *logic* is required — only constant parameterization and the small
fetches above.

---

## 1. Shots (`shots_augmented.parquet`)

The shot table is a single multi-season file; all seasons share one schema/dtype set:
`x_norm, y_norm, distance_from_net, angle_from_center` (float64), `shot_type_id, is_rush,
is_rebound, period, score_state, is_home, is_goal, shooter_id, on_goal, game_id, abs_secs` (int64),
`time_since_last_shot, xg` (float64). **Identical to 2025-26.**

| Metric (excl. shootout period 5) | 2024-25 | 2023-24 | 2022-23 |
|---|---|---|---|
| Rows (all / excl SO) | 111,812 / 111,304 | 114,666 / 114,059 | 113,809 / 113,169 |
| Distinct games | 1,312 | 1,312 | **1,310** |
| Goals / goal rate | 7,377 / 6.63% | 7,639 / 6.70% | 7,801 / 6.89% |
| Null in any xG feature input | none | none | none |
| `distance == hypot(89−x,y)` max diff | 0.0 | 0.0 | 0.0 |
| `angle == atan2(\|y\|,89−x)` max diff | 0.0 | 0.0 | 0.0 |
| Net / normalization | (89,0), x∈[0,99], y∈[−42,42] | same | same |
| `shot_type_id` values | 0–11 | 0–11 | 0–11 |
| `shot_type_id == 0` rows (all goals?) | 20 (100% goals) | 10 (100% goals) | 17 (100% goals) |
| Duplicate pairs (game+sec+shooter+xy) | 0 | 14 | 8 |
| — same + shot_type | 0 | 8 | 6 |

- **Completeness vs 1,312 (NHL API standings, ΣGP÷2 = 1,312 for every season):**
  - 2024-25: 1,312 present — complete.
  - 2023-24: 1,312 present — complete.
  - 2022-23: **1,310 present — 2 missing: `2022020544`, `2022020843`.** Both are real games (the
    API returns full data for them); they are simply absent from `shots_augmented`.
- **Shot-type mapping cross-check (3 games/season vs API `shotType`):** 2024-25 265/267 (99.3%),
  2023-24 260/261 (99.6%), 2022-23 241/241 (100%). The 0–11 mapping (0=unknown, 1=wrist … 11=
  between-legs) is **identical across seasons**; the 1–3 mismatches are the same-second rapid-fire
  matching artifact, not a mapping disagreement. `shot_type_id==0` is the same "goal with no
  `shotType` in the feed" pattern as 2025-26 (all goals) — item-B treatment applies unchanged.
- **Duplicate pairs (counted, not yet API-verified):** 2023-24 has 14 rows (8 with matching
  shot_type), 2022-23 has 8 (6), 2024-25 has 0. These are the same same-second-rapid-fire signature
  that in 2025-26 proved to be **real distinct events** (item A). They must be API-verified per
  season before any drop — and the 2025-26 precedent says keep them.

## 2. Shifts (the critical item)

**Shift data already exists** for all three seasons in `model/season_cache/`, with the **exact
2025-26 schema** (`game_id, player_id, team_id, team_abbrev, first_name, last_name, period,
abs_start, abs_end`):

| File | Rows | Distinct games | Reg-season shot games covered | Corrupt (teams≠2) | Near-empty (<50 rows) |
|---|---|---|---|---|---|
| `shifts_20242025.parquet` | 1,018,816 | 1,341 | **1,255 / 1,312 (57 missing)** | 0 | 0 |
| `shifts_20232024.parquet` | 1,062,112 | 1,400 | 1,312 / 1,312 ✅ | 0 | 0 |
| `shifts_20222023.parquet` | 1,067,775 | 1,398 | 1,310 / 1,310 ✅ | 0 | 0 |

(Game counts exceed 1,312 because each cache also holds that season's playoffs. Players/game =
36–40 both sides, i.e. full rosters — no partial games among those present.)

- **2024-25 gap:** the 57 missing regular-season games are a **contiguous block,
  `2024021235`–`2024021291`**. The **JSON shift-chart endpoint returns 0 rows** for them
  (tested 2024021235/2024021260/2024021291 → 0 rows). **The HTML TOI reports ARE available**
  (tested TH/TV for 2024021235 and 2024021291 → HTTP 200, ~200 KB each; boxscores HTTP 200), so
  the existing HTML fallback can recover all 57.
- **2022-23:** shifts cover every present shot game. The 2 games missing from *shots*
  (2022020544, 2022020843) also have no shift rows in the cache, but the **JSON shift API returns
  complete data** for both (851 and 818 rows, 2 teams) — so both shots and shifts for these 2 games
  are fetchable.
- **No corruption** (0 games with >2 team_ids) in any of the three caches — the >2-team defect seen
  earlier in 2025-26 does not appear here.

**Shift-chart API test (read-only, 5+ games):** controls (2024020001, 2023020001, 2022020001)
returned 751–829 rows / 2 teams — healthy. Gap games behaved as above.

**HTML TOI fallback parser — exists and is reusable as-is:** `model/train_rapm.py`
- `fetch_shifts()` (line 144): JSON primary, falls back to HTML when JSON is empty.
- `fetch_shifts_html()` (line 298): boxscore jersey→playerId map + home (`TH`) / visitor (`TV`)
  HTML TOI reports at `nhl.com/scores/htmlreports/{season}/{TH|TV}{gid6}.HTM`.
- `_parse_toi_report()` (line 232): parses the HTML (period OT→4, SO skipped).
- `shift_record_to_dict()` (line 330): normalizes to the **exact shift-cache schema**.
- `bulk_pull_shifts()` (line 358): parallel puller.
It derives the season from the game_id and is not 2025-26-specific, so **it runs on these seasons
unchanged** — this is the same fallback built for the empty-JSON 2025-26 games.

## 3. Enrichment readiness

**`model/build_shots_enriched_2025_26.py` — no logic changes needed; parameterize 4 constants:**

| Line | Constant | 2025-26 hardcode | Change to parameterize |
|---|---|---|---|
| 46 | `SHIFTS_IN` | `season_cache/shifts_20252026.parquet` | per-season shift cache |
| 47 | `OUT` | `shots_enriched_2025_26.parquet` | per-season output name |
| 49 | `GID_LO, GID_HI` | `2025020001, 2025021312` | per-season game_id range |
| 50 | `GOALIE_MEAN_SECS` | `180` | keep 180, but re-validate per season (below) |

(`SHOTS_IN`, line 45, is the shared multi-season `shots_augmented.parquet` — no change.) The core
loop uses **`team_id`**, not abbreviations, and is season-agnostic. Cleanest change: add a
`--season` arg mapping to `(SHIFTS_IN, OUT, GID_LO, GID_HI)`.

**`model/build_xg_features_2025_26.py` — parameterize 4 constants:** `ENRICHED` (line 42),
`DATES` (line 44, needs a per-season `game_dates_*.csv`), `OUT` (line 45), `GID_LO/HI` (line 47).
`HAND` (line 43) is shared but must be extended (see §4). `mode_id` (line 86) is computed
dynamically (wrist=1 in every season), so no change.

**Goalie detection (mean-shift-length gap) — re-validate per season; the 180 s threshold holds for
all three but 2024-25's margin is thin:**

| Season | Skater max mean-shift | Goalie min mean-shift | Gap around 180 | Goalies detected |
|---|---|---|---|---|
| 2023-24 | 88.1 s | 193.8 s | wide (clean) | 98 |
| 2022-23 | 176.0 s | 191.0 s | 15 s | 105 |
| 2024-25 | **176.0 s** | **185.0 s** | **9 s (thin)** | 99 |

180 s still separates skaters from goalies in every season, but 2024-25's 176↔185 gap is narrow —
worth an explicit per-season check (as done for 2025-26's 157↔207) before trusting it.

**Team relocation (ARI → UTA):** `team_abbrev` differs by season — 2022-23 & 2023-24 use **ARI**
(Arizona), 2024-25 & 2025-26 use **UTA** (Utah). Because the enrichment and feature builders join
on **`team_id`** (ARI = 53, UTA = 68, distinct — no id collision) and never on the abbreviation,
**relocation does not break enrichment or feature construction.** It would only break a *cross-season*
aggregation keyed on `team_abbrev` (treating ARI and UTA as different franchises) — not part of this
pipeline, but flagged for any future franchise-level rollup.

## 4. Handedness coverage

`model/player_handedness.csv` holds 933 players (the 2025-26 shooters).

| Season | Distinct shooters | Not in handedness.csv |
|---|---|---|
| 2024-25 | 908 | 130 |
| 2023-24 | 907 | 208 |
| 2022-23 | 937 | 310 |
| **Union of all 3** | — | **383** |

Fetch handedness (`shootsCatches` from `/v1/player/{id}/landing`, the same method used in Part 1)
for the **383** distinct players not yet covered, appending to `player_handedness.csv`. Small,
one-time fetch; unknowns stay null (no defaulting), exactly as before.

## 5. Old xG model training window

`model/train_xg.py` (v3, "15-season window" in the header but the `SEASONS` list, lines 51–56, is
**16 seasons: 20102011 → 20252026**) trains on 2010-11 through **2025-26, regular season *and*
playoffs**, with year-decay `SEASON_WEIGHTS` anchored at `20252026 = 1.00` (lines 63–79). The
per-season shot caches (`season_cache/shots_20242025.parquet`, etc.) confirm those seasons are its
inputs.

**Yes — 2025-26 regular-season games were in the old model's training data** (weight 1.00). This
confirms the caveat in `xg_model_report.md` §9: the old `xg` column's slight holdout edge over the
new single-season model is at least partly an in-sample (leakage) advantage, since the old model
trained on the very holdout games the new model was scored on out-of-sample.

Note the candidate seasons 2024-25 (0.82), 2023-24 (0.67), 2022-23 (0.55) are already weighted
inputs to the *old* model — so adding them to the *new* single-season model is really about giving
the new model the multi-season data volume the old model already had.

---

## Per-season readiness & fix sizing

### 2023-24 — **READY**
- Shots complete (1,312), shifts complete (1,312/1,312), schema identical, 0 nulls, geometry exact,
  clean goalie gap, no corruption.
- **Games to fetch: 0.**
- Work: parameterize the 2 builders (constants only, no logic); fetch handedness for 208 players;
  build `game_dates_20232024.csv` (32 club-schedule API calls, the Part-1 method). Then run
  enrichment → features. API-verify the 14 duplicate pairs before any drop (precedent: keep).

### 2024-25 — **NEEDS FETCH (57 shift games)**
- Shots complete (1,312). Shifts missing the contiguous block **2024021235–2024021291 (57 games)**;
  JSON endpoint empty, **HTML TOI reports available (HTTP 200)**.
- **Games to fetch: 57**, via `train_rapm.py`'s `bulk_pull_shifts` / `fetch_shifts_html` (runs
  as-is) → append to a 2024-25 shift set. Then parameterize + enrich.
- Also: re-validate the 180 s goalie threshold (thin 176↔185 margin); fetch handedness for 130
  players; build `game_dates_20242025.csv`.

### 2022-23 — **NEEDS FETCH (2 games) / near-ready**
- Shifts complete for all present shot games. **2 games (2022020544, 2022020843) are missing from
  `shots_augmented` entirely**; both are fetchable (shots via the shot pipeline; shifts via the JSON
  API, 851/818 rows).
- **Games to fetch: 2** (shots + shifts) for 100% completeness, or proceed at 1,310/1,312
  (99.85%) and document the 2 omissions.
- Also: fetch handedness for 310 players; build `game_dates_20222023.csv`; API-verify the 8
  duplicate pairs.

### Common work (once, across all three)
- Add a `--season` parameter to `build_shots_enriched_2025_26.py` (4 constants) and
  `build_xg_features_2025_26.py` (4 constants). **No algorithm changes.**
- Extend `player_handedness.csv` by **383** players (one batched `/landing` fetch).
- Build three `game_dates_<season>.csv` caches (32 club-schedule calls each).
- Per-season: API-verify duplicate pairs (0 / 14 / 8) — do not drop without verification.

**Nothing is BLOCKED.** The heaviest single item is the 57-game HTML shift fetch for 2024-25, for
which the parser already exists and the source reports are confirmed available.
