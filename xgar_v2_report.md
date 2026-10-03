# xGAR recomputed on v2 ixG

The displayed xGAR stopgap was computed on the OLD xG; displayed ixG is now v2.
This step recomputes xGAR with the identical formula on v2 ixG, repoints the xGAR
display, and removes the "(legacy)" label from xGAR only. GAR and Game Score keep
their legacy labels. `build_self_generated_stats.py` was read only (imported, never
modified; no TEMP STOPGAP block touched). Not committed.

## Step 1 — the stopgap formula (documented)

**Source:** `model/build_self_generated_stats.py::build_skater_xgar` (lines 150-225).

- **xG input:** the OLD `xg` column in `shots_augmented.parquet`, 2025-26 regular
  season (`season==2025 and game_type==02`), team-attributed shots. Aggregated per
  shooter: `ixg = Σ xg`, `goals = Σ is_goal`, `n_shots`, `sog_my = Σ on_goal`.
- **Game states:** ALL situations (every unblocked shot; no strength filter).
- **TOI + position/team:** from MoneyPuck "all" rows. `toi_sec` = summed all-situations
  icetime; position/team from the player's most-iced stint.
- **Replacement baseline (from our own data, not a hardcoded constant):** group
  skaters into F (C/L/R) and D; rank within each (team, group) by `toi_sec`
  descending; the replacement pool is forwards ranked **>13** on their team and
  defensemen ranked **>7** (the 14th+ F / 8th+ D), pooled league-wide by group.
  `replacement_ixG_60[group] = Σ pool ixG / (Σ pool toi_sec / 3600)`.
- **Formula:** `xGAR = ixG - replacement_ixG_60[pos_group] * (toi_sec / 3600)`
  (offense only, goals scale; players with no group fall back to the F baseline).
- **Output / dashboard read:** `model/skater_xgar_self_generated.csv`, column `xgar`,
  read by `_load_skater_xgar`.
- **Dependencies:** individual xG and TOI only. **No GAR scaling, no composite, no
  WAR.** → proceeding (no STOP).

## Step 2 — replicate, then swap

`model/build_xgar_v2_2025_26.py` implements the formula as a function of the ixG
input, imports `build_self_generated_stats` for the exact shared inputs.

**Equivalence (OLD ixG):**
- vs the stopgap `build_skater_xgar` output (same current inputs): **max |Δ xGAR| =
  0.00e+00** over 933 players.
- vs the on-disk `skater_xgar_self_generated.csv` (the currently-displayed values):
  **max |Δ| = 0.00e+00** over 933 players (MoneyPuck inputs stable; exact reproduction).

**Replacement baseline, old vs new:**

| group | old (OLD xG) | new (v2 ixG) |
|---|---|---|
| F | 0.7143 | 0.6577 |
| D | 0.1621 | 0.1549 |

(v2 recalibrates individual xG slightly, so the fringe-player replacement level is a
touch lower; this is applied with the same pool and TOI.)

**Output:** `model/xgar_v2_2025_26.csv` (947 skaters), same columns as the stopgap
output (`player_id, name, team, position, n_shots, ixg, goals, sog_my,
goals_above_expected, xgar`). v2 ixG/goals/n_shots from `skater_xg_2025_26.csv`
(all situations), `sog_my` from enriched on-goal, TOI/position from MoneyPuck.

**Determinism:** two runs, identical md5 (`ed2ad429…`).

## Step 3 — display

- Added `_load_xgar_v2()` (reads `model/xgar_v2_2025_26.csv`) and repointed the three
  xGAR **display** emissions to it: the gar/xgar leaderboard row builder, the
  player-search detail, and the full player payload. **The GAR off/def split still
  reads the old xGAR ixG via `_load_skater_xgar` (unchanged)** — xGAR as an input to
  GAR was left on its current source per the guardrail; only the display read moved.
- Removed the "(legacy)" label and `legacyRebuild` flag from xGAR in
  `players_leaderboards.js` and `players_overview.js`. GAR and Game Score keep their
  legacy labels.
- **Glossary — new xGAR entry (full text):**
  - name: "xGAR (Expected Goals Above Replacement)"
  - def: "Individual expected goals (xG model v2) above a replacement level baseline. Offense only, goals scale."
  - explain: "xGAR takes a player's individual expected goals from the v2 model and subtracts a replacement level baseline scaled to his ice time. Replacement level is measured from this season's own data as the expected goals per 60 of fringe players (forwards outside their team's top 13 by ice time and defensemen outside the top 7), pooled by position. So xGAR is how many expected goals of offense a player created above what a readily available replacement would have created in the same minutes. It is a stopgap offense only measure on the same goals scale as GAR, now built on v2 ixG so it matches the ixG shown elsewhere. Comparing GAR to xGAR still reads as luck: GAR above xGAR means hot finishing likely to regress."
  - why: "A clean read on offensive value that is not inflated by finishing luck, useful for buy low and sell high calls."
  - limit: "Offense only and a stopgap. It inherits xG model noise, and on one season the GAR to xGAR gap can be partly real signal rather than pure luck. The full value rebuild will replace it."
  - (no em dashes or en dashes)
- **Legacy-rebuild entry:** xGAR removed from its list; now reads "GAR, Game Score, the
  composite rating, Contract Value, and the Simulator" with the note "(xGAR now uses
  v2 ixG and is no longer legacy.)"

## Step 4 — verify

- **Routes:** app boots; all GET display routes return 200 (checked players-full,
  goalies-full, leaders, rapm-leaders, gar-leaders, xgar-leaders, player detail).
- **Diff `integration_after_xgar/` vs `integration_after/`:** across all 8 player
  detail files and players-full / goalies-full / leaders / rapm-leaders, the ONLY
  changed fields are `xgar` (and `sustainability_score`, which is defined as
  `gar - xgar` and so follows the displayed xGAR). **0 UNEXPECTED changes** to any
  RAW stat or other DERIVED value.

**Five-player check (live, matches `xgar_v2_2025_26.csv` exactly):**

| Player | old xGAR | new xGAR | v2 ixG | match |
|---|---|---|---|---|
| McDavid | +18.50 | +19.23 | 39.89 | ✓ |
| Kucherov | +7.24 | +8.39 | 25.33 | ✓ |
| MacKinnon | +16.83 | +17.68 | 37.20 | ✓ |
| Brady Tkachuk | +14.26 | +14.74 | 25.89 | ✓ |
| Mark Stone | +9.97 | +9.30 | 21.94 | ✓ |

Baseline used: F 0.6577, D 0.1549 (v2).

**Spearman(new, old) xGAR, ≥500 total TOI min (n=616): 0.9862.** 5 largest rank
movers (all low-TOI players near the baseline, tiny absolute values):
- Claude Giroux: old #565 → new #417 (−1.76 → +0.07)
- Paul Cotter: #376 → #506 (+0.45 → −0.69)
- Oliver Moore: #408 → #533 (+0.08 → −1.07)
- Robert Thomas: #513 → #389 (−0.87 → +0.37)
- Alexandre Texier: #316 → #437 (+1.27 → −0.07)

## Integrity

- **md5:** all 230 protected model/data files + `integration_baseline/` +
  `integration_after/` + `data/contracts_current.json` re-hashed — **unchanged**.
  `build_self_generated_stats.py` and `skater_xgar_self_generated.csv` byte-identical;
  simulator / composite / Contract Value inputs untouched.
- **New files:** `model/build_xgar_v2_2025_26.py`, `model/xgar_v2_2025_26.csv`,
  `integration_after_xgar/` (13 JSONs), `xgar_v2_report.md`.
- **Incremental code diff (this step only, vs the pre-step saved diff):** confined to
  `app.py` (new `XGAR_V2_PATH` + `_load_xgar_v2` + 3 display repoints) and
  `static/js/players_leaderboards.js`, `players_overview.js`, `glossary_data.js`
  (xGAR label de-legacy + xGAR glossary rewrite + legacy-rebuild list update). Diff
  grew from 701 to 775 lines; every added hunk is xGAR-scoped. No other file changed.
- **Not committed.**

## Choices I made that were not fully specified
1. **Display-only repoint:** I did not repoint `_load_skater_xgar` wholesale (its ixG
   feeds the GAR off/def split, a non-display GAR input). I added `_load_xgar_v2` and
   switched only the three xGAR display emissions, leaving the GAR-split input on the
   old source as the guardrail requires.
2. **v2 inputs:** v2 ixG/goals/n_shots come from `skater_xg_2025_26.csv` (all
   situations, as the stopgap uses); `sog_my` (not in that file) is taken from the
   enriched on-goal counts. Only ixG changes the xGAR value; TOI, position, team, and
   the replacement-pool definition are identical to the stopgap (MoneyPuck).
3. **sustainability_score:** it is defined as `gar - xgar` in a (hidden) Contract
   Value row and therefore moves with the displayed xGAR; I treated that as an allowed
   consequence of the xGAR switch, not an unexpected change.
4. **Equivalence reference:** I compared against both the live `build_skater_xgar`
   output and the on-disk file; both are exact (0.00), confirming no MoneyPuck drift.

---

# Self generated TOI

Two fixes before committing: (1) switch the xGAR TOI (and the replacement-pool
ranking TOI) from MoneyPuck to self generated TOI from the validated stint file;
(2) label `sustainability_score` legacy. Edits limited to
`model/build_xgar_v2_2025_26.py`, its output `model/xgar_v2_2025_26.csv`, the
`sustainability_score` label and tooltip, and the xGAR glossary entry. Not committed.

## Step 1 — self generated TOI (and the other MoneyPuck dependencies)

`model/build_xgar_v2_2025_26.py` now takes **all inputs self generated**:
- **TOI (all situations):** per skater, Σ stint `duration_secs` over every stint in
  `model/rapm_stints_2025_26.parquet` where the skater is an on-ice skater (home or
  away). Feeds both `TOI_hours` in the formula and the replacement-pool ranking.
- **team / position / name:** from `model/skater_xg_2025_26.csv` (team = primary team
  by most all-situations on-ice stint TOI; position from `model/player_positions.csv`;
  name from the shift file). ixG / goals / n_shots from the same file; `sog_my` from
  `shots_enriched_2025_26.parquet` on-goal counts.
- **Multi-team rule:** a traded skater is assigned to the ONE team where he logged the
  most all-situations on-ice stint TOI and is ranked within that team only (one pool
  membership per player), matching `skater_xg_2025_26.csv.team`.

> **MoneyPuck dependency beyond TOI — reported (did not STOP).** The stopgap formula
> also consumed MoneyPuck **position** (F/D grouping and per-position baseline) and
> **team** (within-team top-13 / top-7 ranking). These are identity fields, and the
> self-generation rule makes external sources validation-only, so leaving them on
> MoneyPuck would itself violate the rule this fix enforces. They are trivially
> available self generated (already in `skater_xg_2025_26.csv`), so I moved them to
> the self-generated sources rather than stop with a half-self-generated formula. No
> MoneyPuck field feeds any output. Flagging per the guardrail.

**Determinism:** two runs, identical md5 (`03045bc5…`).

**Baseline (self-generated inputs):** replacement ixG/60 **F = 0.6292, D = 0.1587**
(the previous displayed file, which used MoneyPuck team/position/TOI, had F = 0.6577,
D = 0.1549 — the shift is driven mostly by self-sourcing team/position, not TOI;
see Step 2).

## Step 2 — comparison (validation only; MoneyPuck read, never written)

**Stint TOI vs MoneyPuck TOI, per skater (n = 940):** correlation **1.0000**, mean
absolute difference **0.36 min**. The 10 largest absolute differences are all Chicago
skaters (Vlasic +25.8, Rinzel +22.3, Crevier +20.2, Bedard +19.3, Levshunov +18.8,
Nazar +18.5, Teravainen +17.7, Mikheyev +17.2, Grzelcyk +16.9, Bertuzzi +15.7 min —
stint TOI slightly higher), a localized CHI shift-data quirk, all under half a minute
in aggregate effect.

**Replacement baseline and pool (isolating the TOI swap: self team/position held
constant, only TOI source varies):**
| | F baseline | D baseline | pool size | moved in | moved out |
|---|---|---|---|---|---|
| MoneyPuck TOI | 0.6292 | 0.1589 | 300 | — | — |
| self TOI | 0.6292 | 0.1587 | 300 | 0 | 0 |

The TOI swap alone changes essentially nothing (identical pool, identical baseline to
3 decimals) because stint TOI and MoneyPuck TOI are ~identical.

**Five-player xGAR, MoneyPuck TOI vs self TOI (self team/position held constant):**
identical to 2 dp — McDavid +20.12 / +20.12, Kucherov +9.12 / +9.12, MacKinnon
+18.52 / +18.52, Brady Tkachuk +15.22 / +15.22, Mark Stone +9.85 / +9.85.

**Spearman(self-TOI, MoneyPuck-TOI) xGAR, ≥500 TOI min (n = 616): 0.999976**;
max |Δ xGAR| = 0.202. 5 largest movers (all Chicago, matching the TOI quirk):
Mikheyev #331→#346 (Δ−0.18), Ryan Greene #405→#418 (Δ−0.16), Oliver Moore #524→#534
(Δ−0.12), Ryan Donato #228→#236 (Δ−0.16), Burakovsky #571→#579 (Δ−0.15).

> Net effect vs the previously displayed file (which used MoneyPuck team/position/TOI):
> five-player xGAR moved e.g. McDavid +19.23 → +20.12, Stone +9.97 → +9.85. That net
> change is driven by self-sourcing **team/position** (lower F baseline 0.6577 → 0.6292),
> not by the TOI swap, which is immaterial on its own.

## Step 3 — labels and glossary

- `sustainability_score` columns relabeled **"Sustain (legacy)"** in
  `static/js/contract_value.js` (both list and scatter column defs), and its tooltip
  now opens with "Legacy, being rebuilt: this mixes legacy GAR with v2 xGAR ... uses
  the legacy composite GAR and will be updated when GAR is rebuilt." (plain language,
  no em/en dashes). This is the only place `sustainability_score` is displayed.
- **No `sustainability_score` glossary entry exists**, so none was edited (noted).
- **xGAR glossary entry updated** — new `explain` (full text):
  "xGAR takes a player's individual expected goals from the v2 model and subtracts a
  replacement level baseline scaled to his ice time. Ice time, team, and position are
  all self generated: ice time is each skater's all situations on ice time computed
  from our own shift data (the validated stint file), not an external source.
  Replacement level is measured from this season's own data as the expected goals per
  60 of fringe players (forwards outside their team's top 13 by ice time and
  defensemen outside the top 7), pooled by position. So xGAR is how many expected
  goals of offense a player created above what a readily available replacement would
  have created in the same minutes. It is a stopgap offense only measure on the same
  goals scale as GAR, now built on v2 ixG so it matches the ixG shown elsewhere.
  Comparing GAR to xGAR still reads as luck: GAR above xGAR means hot finishing likely
  to regress."

## Step 4 — verify

- **All GET display routes return 200** (players-full, goalies-full, leaders,
  rapm-leaders, gar-leaders, xgar-leaders, player detail, goalie-analytics,
  goalies-extended, skater-onice-leaders, composite-ratings, sim-player-pool,
  team-strength, standings, root).
- **Snapshot `integration_after_selftoi/` vs `integration_after_xgar/`:** the only
  code-attributable changes are **`xgar` (and its ranks) and `sustainability_score`**
  (896 changed values across the files). All other diffs are **live NHL-API drift**,
  not code: the two snapshots were captured a day apart and the **2026-27 season rolled
  over** (`current_season.season` 20252026 → 20262027). The distinct non-xGAR changed
  fields are exclusively live counting/season stats — `plus_minus` (104), `gp`,
  `gamesPlayed`, `w`, `wins`, `savePctg`, `gaa`, `sv_pct`, `qs_pct`, `so`, `l`, `otl`,
  `starts`, `season`, `years_in_league`, `last5` length. **No rebuilt or derived field
  (RAPM, QoC/QoT, GSAX, ixG, xGF%, GAR, Game Score, composite) changed.** Classified
  EXPECTED (live drift), none UNEXPECTED-from-code.
- **Five-player live xGAR == `model/xgar_v2_2025_26.csv` exactly:** McDavid 20.12,
  Kucherov 9.12, MacKinnon 18.52, Brady Tkachuk 15.22, Mark Stone 9.85.

## Integrity

- **md5:** all protected files unchanged except the two allowed edits this step —
  `model/build_xgar_v2_2025_26.py` and its output `model/xgar_v2_2025_26.csv`.
  `build_self_generated_stats.py`, `rapm_stints_2025_26.parquet`,
  `skater_xgar_self_generated.csv`, `integration_after/`, and `integration_after_xgar/`
  are byte-identical.
- **Incremental code diff (this step only, vs the saved pre-step diff):** **app.py
  unchanged this step**; only `static/js/contract_value.js` (sustainability label +
  tooltip) and `static/js/glossary_data.js` (xGAR explain) changed. Diff grew 775 →
  812 lines, every added hunk in scope. New/updated files:
  `model/build_xgar_v2_2025_26.py`, `model/xgar_v2_2025_26.csv`,
  `integration_after_selftoi/`.
- **Not committed.**

## Choices I made that were not fully specified
1. **Moved team and position to self-generated too** (not just TOI): the formula's
   F/D grouping and within-team ranking consumed MoneyPuck position/team. Rather than
   STOP (they are identity fields, not a MoneyPuck-only stat, and are already derived
   self generated in `skater_xg_2025_26.csv`), I sourced them self generated so the
   output has zero MoneyPuck dependency, consistent with the self-generation rule.
2. **TOI definition:** all-situations on-ice stint time (Σ duration over every stint
   the skater is on the ice), the self-generated analogue of MoneyPuck's "all"
   icetime; matches `skater_xg_2025_26.csv.toi_all_min`.
3. **Comparison isolation:** Step 2's "MoneyPuck TOI vs self TOI" holds self
   team/position fixed and varies only the TOI source, so the reported Spearman/movers
   isolate the TOI effect (near-nil); the net change vs the previously displayed file
   (team/position also self-sourced) is reported separately.
4. **Live-NHL drift in the snapshot diff** is from the 2026-27 season starting between
   the two snapshots (a day apart); it is not caused by this change and is classified
   EXPECTED.
