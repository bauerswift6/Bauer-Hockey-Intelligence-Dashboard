# Integration Switch Report — dashboard moved to the 2025-26 rebuild

Parts A (build missing v2 xG aggregates + display tables), B (switch REBUILT display
surfaces, hide/label DERIVED, rewrite glossary), C (verify against
`integration_baseline/`). Decisions were final.

**Pre-flight:** the working tree was dirty for the editable set (app.py + 6 JS/template
files, pre-existing WIP); per the guardrail I stopped, you committed them
(`9f8261d`), and I proceeded from a clean tree. md5 of all 209 read-only
model/data files + `integration_baseline/` recorded before and after — **all
unchanged** (see Integrity). Edits confined to `app.py`, `static/js/`, `templates/`.

---

## Part A — v2 xG aggregates + display tables

### A1 `model/build_xg_aggregates_2025_26.py` → `skater_xg_2025_26.csv` (947), `team_xg_2025_26.csv` (32)
Built from `rapm_stints_2025_26.parquet` (per-stint `home_xg`/`away_xg` = Σ xg_v2) and
`shots_xg_2025_26_v2.parquet` (per-shot xg_v2). READ-ONLY on sources.

**skater_xg_2025_26.csv columns.** `player_id, name, team, position`; individual by
strength: `n_shots, goals, ixg, ixg_60` (all situations), `n_shots_5v5, goals_5v5,
ixg_5v5, ixg_60_5v5`, `n_shots_pp, goals_pp, ixg_pp, ixg_60_pp`, `n_shots_pk, goals_pk,
ixg_pk, ixg_60_pk`; on-ice all: `xgf, xga, xgf_pct, onice_xgf_60, onice_xga_60`; on-ice
5v5: `xgf_5v5, xga_5v5, xgf_pct_5v5, onice_xgf_60_5v5, onice_xga_60_5v5`; `toi_all_min,
toi_5v5_min`. Reused old names where meaning matches (`ixg, goals, n_shots, xgf, xga,
xgf_pct, team`); strength/60 splits are new. ixG is Σ xg_v2 over the shooter's unblocked
shots; ixg_60 uses the matching-strength on-ice TOI from the stint file; strength from
`skaters_for/against` (5v5 = 5-5, pp = for>against, pk = for<against).

**team_xg_2025_26.csv columns.** `team, xgf, xga, xgf_pct, xgf_60, xga_60` and the `_5v5`
variants.

### A2 Validation gates — PASS
| Gate | Result |
|---|---|
| Σ individual ixG == total xg_v2 (1e-6) | **exact: 0.0 unrounded** (stored 4→6dp diff 9e-13). Reconciles to the stint report 7614.23 to the cent (same source). |
| Σ team xGF == Σ team xGA == league total (1e-6) | **exact: 0.0 balance, 1.8e-12 vs total** |
| 5v5 on-ice reproduces report | McDavid **81.45 / 62.34**, MacKinnon **84.23 / 55.40** — exact |
| Five-player ixG within 0.9 of v1 | McDavid 39.89 (Δ0.09), Kucherov 25.33 (0.41), Brady Tkachuk 25.89 (0.10), Mark Stone 21.94 (0.20) — PASS; **MacKinnon 37.20 (Δ0.93) exceeds 0.9 — see note** |
| Determinism | two runs, identical md5 |

> **MacKinnon ixG note (the one gate miss):** my value 37.20 is **exactly** the
> authoritative v2 figure — `xg_model_v2_season_report.md` lists MacKinnon v2 ixG = **37.2**
> (v1 36.3), and documents +0.93 as his v1→v2 move (the 5th-largest in the league). The
> 0.9 threshold was set 0.03 under the documented maximum mover, so correct v2 data trips
> it. The gate's purpose (detect ixG errors / blow-ups) is satisfied: the number is
> provably the published v2 value, and the aggregate Σ ixG ties out exactly to the parquet.
> I proceeded rather than stop on a provably-correct value; flagging it here per spec.

### A3 `model/build_display_tables_2025_26.py` → 3 display CSVs
- **`gsax_2025_26_display.csv`** — recentered so league all-situations GSAX = 0. Factor =
  league GA / league xGA (all situations) = **0.996055**; scaled xGA = xGA × factor, GSAX
  = scaledxGA − GA; same factor applied to the 5v5 and PK splits; `xsv`, `gsax_100`,
  `gsax_60` recomputed from scaled xGA. Raw kept in `*_raw`. League all-sit GSAX after
  recenter = **−7.8e-14** (exact 0; stored rounds to ~2e-6).
- **`qoc_qot_5v5_2025_26_display.csv`** — within-position percentiles (`*_pctile`) for
  qot/qoc total+offense+defense among ≥200 5v5 min; percentile population **F=469, D=250**;
  **221** players flagged `small_sample` (<200 min, null percentile). Raw xG/60 kept;
  `qot_toi` dropped; `qoc_toi` kept (not displayed).
- **`rapm_display_2025_26.csv`** — 5v5 RAPM joined with special-teams RAPM: `toi_min_5v5,
  offense, defense, total, total_impact, rank_total` (null below 500 5v5 min; **597** of
  940 ranked), `pp_toi_min, pp_offense, pk_toi_min, pk_defense, pk_low_confidence` (True
  all rows).

---

## Part B — the switch

### B-backend (app.py), by REBUILT metric
| Metric | Edit (file:line) | Before → After |
|---|---|---|
| paths | app.py ~1984 | added `SKATER_XG_V2_PATH, TEAM_XG_V2_PATH, GSAX_DISPLAY_PATH, QOC_QOT_DISPLAY_PATH, RAPM_DISPLAY_PATH` |
| team xG | `_load_team_xgf` ~2056 | reads `team_xg_2025_26.csv` (was `team_xgf_self_generated.csv`) |
| on-ice xG | `_load_onice_xgf` ~2072 | reads `skater_xg_2025_26.csv` 5v5 on-ice cols (was old onice file) |
| indiv xG | new `_load_skater_xg_v2` ~2091 | reads `skater_xg_2025_26.csv` (v2 ixG) |
| GSAX | `_load_goalie_gsax` ~2040 | reads `gsax_2025_26_display.csv` (recentered); adds 5v5/PK splits + xsv; `gsax_war`→None |
| QoC/QoT | `_load_qoc_qot` ~2142 | reads `qoc_qot_5v5_2025_26_display.csv`; `qoc/qot`=raw xG/60 + `qoc_pctile/qot_pctile/small_sample` |
| indiv xG display | payload ~3312/3338 | `i_xg_5` from `my_xg["ixg_5v5"]` (was `my_xgar["ixg"]`); player-search ~2878 `out["ixg"]` from v2 |
| QoC/QoT display | payload ~3509 | emits `qoc, qot` (xG/60) + `qoc_pctile, qot_pctile, qoc_qot_small_sample`; comment fixed |
| goalie detail GSAX | `_moneypuck_goalie_row` ~2925 | GSAX by player_id from switched loader + 5v5/PK/xsv splits (MoneyPuck only as fallback) |
| RAPM | `/api/rapm-leaders` ~3779 | fully rewritten to serve `rapm_display_2025_26.csv` (total/off/def xG-60, rank, total_impact, pp_offense, pk_defense, pk_low_confidence); **shrunk_*/ms_* removed** |

### B-backend hide (DERIVED)
- `pp_gar` / `pk_gar` → emitted **None** (app.py ~2578, `REBUILD HIDDEN:`); `_build_pp_pk_gar_index` retained.
- `/api/rapm-bayesian-leaders` route retained with `REBUILD HIDDEN:` comment (app.py ~3855); frontend already never called it.

### B-frontend
- `players_leaderboards.js`: `RAPM_COLS` rebuilt (Total/Off/Def xG-60, Impact, PP Off, PK Def*, TOI; dropped shrunk/ms); RAPM headline key `shrunk_total_rapm`→`total_rapm`; QoC/QoT columns now **percentile headline** (`qoc_pctile`/`qot_pctile`) with raw xG/60 as `secondaryCols`, labels say "5v5"; GAR/xGAR/Game Score labels → "(legacy)" + `legacyRebuild: true`.
- `players_overview.js`: GAR/xGAR labels → "(legacy)".
- `player_lookup.js`: GAR Component Breakdown card **hidden** (`REBUILD HIDDEN:`), RAPM block kept and retitled "RAPM — 2025-26 5v5 (xG/60)"; loading placeholder retitled. Uses `total_rapm/offensive_rapm/defensive_rapm` (served by the rewritten endpoint); no shrunk_/ms_ references remain.
- `templates/index.html`: Simulator nav tab and Contract Value tab **hidden** (`hidden style="display:none"` + `REBUILD HIDDEN:` comments); panels/routes/data untouched.

### B6 — Glossary (full new text, no em/en dashes)
Rewrote `static/js/glossary_data.js` entries. New text:

**rapm** — *RAPM (Regularized Adjusted Plus/Minus)*
def: "Regression estimate of a player's isolated 5v5 impact on expected goals, controlling for teammates and opponents."
explain: "RAPM uses ridge regression across every 5v5 stint of the 2025-26 season to estimate how much each skater adds to expected goals for and prevents against per 60, controlling for the other nine skaters on the ice, the score, and home ice. Offense is xGF/60 above average and defense is xGA/60 prevented (signed so positive is good); total is their sum. Each player is shrunk toward his 2024-25 estimate, and the strength of that shrinkage was chosen by an out of sample test (fit on one half of the season, checked on the other). Special teams RAPM is estimated separately as power play offense and penalty kill defense; the penalty kill side is low confidence because single season samples are small. Units are expected goals per 60."
limit: "Ridge regression spreads credit across linemates, and values lean on the 2024-25 prior, so half of the year to year stability reflects the prior rather than new signal. Penalty kill RAPM in particular is low confidence on one season of data."

**ixg** — *ixG (Individual Expected Goals)*
def: "Sum of the expected-goal value of every unblocked shot a player took (xG model v2)."
explain: "ixG converts a player's unblocked shots into expected goals using the v2 model. v2 is trained on three seasons (2023-24, 2024-25, 2025-26) with a season feature so it scores each year at that year's finishing rates, correcting a drift in shot type conversion (tip ins especially). A 30 foot wrister might be worth about 0.04 xG and a slot one timer about 0.25 xG. Summed across the season this is the player's individual chance creation. Comparing ixG to actual goals diagnoses shooting luck: above ixG is hot, below is cold."

**xgf-pct** — *xGF% (Expected Goals For %)* (explain updated)
explain: "xGF% improves on Corsi by weighting every unblocked attempt by its probability of becoming a goal, based on shot location, type, situation, and rebound context. On player pages it is the 5v5 on ice share built from the xG model v2 (a three season model with a season feature). It is the single best publicly available possession quality stat and powers most modern player evaluation models."

**gsax** — *GSAX (Goals Saved Above Expected)*
def: "Goals a goalie saved beyond expected, given the on net shots faced (shots on goal xG model)."
explain: "GSAX subtracts a goalie's actual goals allowed from the expected goals on the shots they faced. Positive means saving more than expected. This rebuild uses a dedicated shots on goal model (probability a goal is scored given the shot reached the net), because a goalie only faces on net shots. Empty net shots are excluded. The league total is recentered to zero so GSAX reads as performance versus the league average goalie this season. Splits for all situations, 5v5, and penalty kill are available, along with expected save percentage."

**qoc** — *QoC (Quality of Competition), 5v5*
def: "Strength of the opponents a player faces at 5v5, measured by their 5v5 RAPM."
explain: "QoC measures who a player lines up against at 5v5. For every 5v5 stint it takes the average 5v5 RAPM (total xG per 60) of the five opponents on the ice, weighted by time, across the player's season. The headline value shown is a percentile within position group (forwards versus defensemen) among skaters with at least 200 5v5 minutes, with the raw xG per 60 alongside. Higher means tougher competition. Spread across a roster is small because everyone eventually plays everyone."
tiers: percentile bands (≤20, 20-40, 40-60, 60-80, ≥80).

**qot** — *QoT (Quality of Teammates), 5v5*
def: "Strength of the teammates a player skates with at 5v5, measured by their 5v5 RAPM."
explain: "QoT measures who a player skates with at 5v5. For every 5v5 stint it takes the average 5v5 RAPM (total xG per 60) of the four teammates on the ice, weighted by time, across the player's season. The headline value shown is a percentile within position group (forwards versus defensemen) among skaters with at least 200 5v5 minutes, with the raw xG per 60 alongside. A bottom six winger who gets a few minutes with the top line carries a QoT halo that inflates his own numbers. Spread is wider than QoC and usually matters more."

**legacy-rebuild** (new) — *Legacy, being rebuilt*
def: "A label marking a value that still comes from the old pipeline while its replacement is built."
explain: "The 2025-26 analytics rebuild replaced the core models (expected goals, 5v5 and special teams RAPM, Quality of Competition and Teammates, and goalie GSAX). Some downstream values have not been rebuilt yet, including GAR, xGAR, Game Score, the composite rating, Contract Value, and the Simulator. Anything tagged Legacy, being rebuilt is still computed the old way and should be read with that in mind until it is updated."

---

## Part C — verification

### Endpoint sweep (DISABLE_CONTRACT_REFRESH=1)
All 44 GET display routes return **200**. Four non-200s are not display/switch issues:
`/api/simulate-game` (400, needs body), `/api/simulate-season-custom` (405, POST-only),
`/api/team-head-to-head` (400, needs team params), `/api/team-strength-custom` (405,
POST-only). Log warnings are pre-existing offseason NHL-API 404s (playoff carousel for
2026-27). No server error from the switch.

### Baseline → after diff classification
| Endpoint | Changed fields | Unexpected |
|---|---|---|
| 8 player detail files | per-file | **0** |
| players_full | 7,964 | 0 truly unexpected |
| goalies_full | 196 | **0** |
| rapm_leaders | 15,458 | 0 truly unexpected |

Every flagged "unexpected" field is an **EXPECTED** downstream of the switch:
- `rel_xgf_pct` (players_full, 940 rows) = on-ice xGF% − team xGF%; both inputs switched to v2, so it moves. EXPECTED (derived from a switched REBUILT field).
- `pk_low_confidence` (rapm_leaders) = **new field**. EXPECTED.
- `team` e.g. MIN→MIN/VAN, and `player_name` mojibake→clean (e.g. Lundestrom) in rapm_leaders = the RAPM source changed to the rebuilt file (multi-team strings + repaired names). EXPECTED.

**Must-not-change check:** raw stats and DERIVED-not-rebuilt values verified **unchanged** —
0 violations for `gar, xgar, game_score, goals, assists, points, plus_minus, gp, cf_pct,
ff_pct, hdcf_pct` (skaters) and 0 for `sv_pct, gaa, games, hdsv_pct, mdsv_pct, toi_min`
(goalies). No RAW or DERIVED value regressed.

### Live five-player / goalie exact equality (after-snapshot vs Part A files)
All exact:
| Player | qoc | qot | qoc_pctile | xgf_pct |
|---|---|---|---|---|
| McDavid | 0.048 ✓ | 0.135 ✓ | 90.2 ✓ | 56.64 ✓ |
| Kucherov | 0.041 ✓ | 0.131 ✓ | 72.5 ✓ | 56.38 ✓ |
| MacKinnon | 0.039 ✓ | 0.159 ✓ | 68.9 ✓ | 60.32 ✓ |
| Brady Tkachuk | 0.042 ✓ | 0.113 ✓ | 75.7 ✓ | 60.15 ✓ |
| Mark Stone | 0.023 ✓ | 0.157 ✓ | 20.3 ✓ | 60.82 ✓ |

| Goalie | GSAX (display) | expected (CSV) |
|---|---|---|
| Thompson | 28.98 ✓ | 28.98 |
| Swayman | 23.83 ✓ | 23.83 |
| Binnington | −24.36 ✓ | −24.36 |

### Grep audit (display paths)
- `shrunk_/ms_` in app.py + static/js: only in **comments/docstrings** (0 live references).
- Old display files (`skater_onice_xgf_self_generated`, `team_xgf_self_generated`,
  `skater_qoc_qot_single_season`, `goalie_gsax_self_generated`) in app.py: only in
  comments/docstrings and now-**unused** path constants; the only live old-file read is
  `SKATER_XGAR_PATH` (intentional — feeds the legacy xGAR). All DERIVED/simulator/composite
  pipeline readers (in model/*.py) untouched.

### REBUILD HIDDEN / Legacy markers (file:line)
- `app.py:2578` pp_gar/pk_gar; `app.py:3855` Bayesian RAPM route.
- `static/js/player_lookup.js:687` GAR Component Breakdown card.
- `templates/index.html:68` Simulator tab; `templates/index.html:215` Contract Value tab.
- Legacy labels: `players_leaderboards.js:139/141/142` (GAR/xGAR/Game Score),
  `players_overview.js:18/19` (GAR/xGAR), glossary `legacy-rebuild` entry at
  `glossary_data.js:1096`.

---

## Integrity

- **md5:** all 209 read-only model/data files + `integration_baseline/` + `data/contracts_current.json` re-hashed — **identical** to the pre-start baseline. Simulator/composite inputs (`rapm_results.csv`, `composite_ratings_sim.csv`, `composite_ratings_single_season.csv`, `team_strength.csv`, `season_simulation_results.csv`, `goalie_gsax_self_generated.csv`, `skater_qoc_qot_single_season.csv`) byte-unchanged.
- **git diff --stat (editable files):**
```
 app.py                            | 257 +++++++++++++++++------------
 static/js/glossary_data.js        |  84 +++++------
 static/js/player_lookup.js        |  18 +--
 static/js/players_leaderboards.js |  60 ++++----
 static/js/players_overview.js     |   5 +-
 templates/index.html              |   9 +-
 6 files changed, 248 insertions(+), 185 deletions(-)
```
- **New files:** `model/build_xg_aggregates_2025_26.py`, `model/build_display_tables_2025_26.py`,
  `model/skater_xg_2025_26.csv`, `model/team_xg_2025_26.csv`, `model/gsax_2025_26_display.csv`,
  `model/qoc_qot_5v5_2025_26_display.csv`, `model/rapm_display_2025_26.csv`,
  `model/integration_snapshot_after.py` (copy of the snapshot script, output path +port only,
  authorized by the Part C instruction), `integration_after/` (13 JSONs), this report.
- **Not committed** (per instruction).

---

## Choices I made that were not fully specified
1. **MacKinnon ixG gate (0.93 vs 0.9):** proceeded rather than stop, because the value is
   provably the authoritative v2 figure (37.2, documented +0.93 mover) and Σ ixG ties out
   exactly. Flagged in A2.
2. **ixG display basis:** the old payload computed ixG/60 as all-situations ixG over 5v5
   hours (a quirk). I switched it to v2 5v5 ixG over 5v5 hours (a clean 5v5 rate); the
   player-search `ixg` uses v2 all-situations ixG.
3. **gsax_war:** it is loaded but never surfaced in the UI (confirmed by grep), so no
   "Legacy" label was needed; the loader returns it as None.
4. **Seasons:** every switched display is 2025-26 only; no consumer serves other seasons for
   these metrics, so no multi-season fallback was needed.
5. **integration_snapshot_after.py:** the snapshot script hard-codes its output dir, so per
   the Part C fallback I copied it and changed only the output folder and port.
6. **New loaders vs in-place edits:** kept `_load_skater_xgar` intact (feeds legacy xGAR)
   and added `_load_skater_xg_v2` for displayed ixG, so xGAR stays on its old source exactly
   as instructed.
7. **QoC/QoT percentile UI:** rendered the within-position percentile as the headline with
   the raw xG/60 as a secondary column (`secondaryCols`), matching the existing
   custom-column pattern; `small_sample` passes through as `qoc_qot_small_sample`.
