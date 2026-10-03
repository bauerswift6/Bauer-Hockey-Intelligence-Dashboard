# Goalie GSAX — 2025-26 Regular Season (shots-on-goal xG model)

Deterministic: `model/train_xg_sog.py` (SOG xG model) + `model/build_gsax_2025_26.py` (attribution + GSAX). Read-only w.r.t. all protected inputs. GSAX uses a **separate shots-on-goal** xG model (P(goal | shot reached the net)), since a goalie only faces on-net shots.

## Step 1 — Old GSAX pipeline inventory (diagnostic; nothing modified)

**Builder:** `model/build_self_generated_stats.py::build_goalie_gsax`.

**Inputs:** `shots_augmented.parquet` (cols used: `game_id, abs_secs, xg, is_goal,
shooter_team`), `season_cache/shifts_20252026.parquet` (goalie shifts), and the
live MoneyPuck goalies season-summary CSV (for the goalie id set).

**What it computes**
- xG used: the **old `xg` column** (v1 unblocked-shot xG baked into
  shots_augmented) — NOT the v2 model and NOT a shots-on-goal model.
- Shots counted: **every unblocked shot** in the goalie's shift window whose
  shooter is on the opposing team (attribution by `abs_secs ∈ [abs_start,
  abs_end]` of the goalie shift). It does **not** filter to on-goal, so
  `shots_faced` counts unblocked attempts (misses included).
- GSAX = `Σ xg − Σ is_goal` (all situations only; no 5v5 / PK split). Empty-net
  handling is implicit (shots_augmented omits shots at an empty net).

**Output:** `model/goalie_gsax_self_generated.csv`
**Columns:** `player_id, name, team, shots_faced, xg_against, goals_against,
gsax, gsax_war` (gsax_war = gsax / GOALS_PER_WAR).

**Dependencies:** does NOT read `rapm_results.csv`. Reads shifts + MoneyPuck.

**Where the dashboard reads it (file:line)**
- `app.py:1976-1977` — `GOALIE_GSAX_PATH = model/goalie_gsax_self_generated.csv`.
- `app.py:2031-2044` — `_load_goalie_gsax()` → `{gsax, gsax_war, xg_against,
  goals_against, shots_faced}` per player_id.
- `app.py:795-815` — goalie-leaderboard endpoint emits `gsax`.
- `app.py:1788-1833` — full goalie table emits `gsax` and `gsax_60`
  (`gsax_60 = gsax / (icetime/3600)`; falls back to `xGoals − goals` if missing).
- `static/js/goalies_page.js:74-79` — columns `gsax`, `gsax_60`.
- `static/js/players_overview.js:26` — `gsax` (goalies).
- `static/js/player_compare.js:178-181` — `GSAX`, `GSAx/60`.

**Difference vs this build:** the old metric is **unblocked-attempt** based on
**v1 xG**, all-situations only; the new build is **on-goal (SOG)** based on a
**dedicated SOG v2-style model**, split by situation. Different denominator and
scale, written to a **separate** file — no dashboard wiring touched.

## Step 2 — Shots-on-goal dataset

On-net = shots that reached the net = `on_goal == 1` (saves + goals; all goals have on_goal==1). Missed shots (incl. posts/crossbars) excluded. On-net flag joined from `shots_enriched_*` (positionally aligned to `xg_features_*`, verified). The shot_type_id=0 goals (24/20/10, imputed to wrist & flagged) are kept exactly as v2.

**Empty-net:** empty-net-against shots (no defending goalie) are excluded from training and GSAX. In v2 training data they were **already absent** — shots_augmented omits shots taken at an empty net by construction; `empty_net_against` among regular-season non-shootout shots = **0** per season, so **0** were excluded in each of 2023-24 / 2024-25 / 2025-26.

| season | on-net shots | goals | goal rate |
|---|---|---|---|
| 2023-24 | 79,014 | 7,636 | 0.0966 |
| 2024-25 | 73,681 | 7,375 | 0.1001 |
| 2025-26 | 72,542 | 7,577 | 0.1044 |

**2025-26 on-net counts by shot type:** wrist 32843, snap 18726, slap 7689, backhand 5481, tip-in 5425, deflected 1327, wrap-around 442, poke 299, bat 260, cradle 3, between-legs 47

## Step 3 — Shots-on-goal xG model

- Features: identical to v2 (incl. integer `season`). Holdout: same last 263 games of 2025-26 (14264 on-net shots, 1537 goals).
- Tuned (same grid/procedure as train_xg_v2): **{'max_depth': 5, 'learning_rate': 0.03, 'min_child_weight': 5, 'subsample': 0.8, 'colsample_bytree': 1.0}**, **423 trees**.
- Holdout: log loss **0.29538** (constant-rate baseline 0.34214), AUC 0.7657, Brier 0.08654, predicted 1506.2 vs actual 1537 (-2.00%).
- OOF (5-fold GroupKFold by game, cross-season pooled, season=2025): total **7607.0** vs actual 7577 (+0.40%), log loss 0.29035, AUC 0.7635.
- **Calibration gates:** GATE1 OOF ±2% → PASS (+0.40%); GATE2 per-shot-type [0.90,1.10] (≥100 goals) → PASS; GATE3 holdout ±3% → PASS (-2.00%).

**GATE2 — OOF actual/pred by shot type:**

| shot type | n | goals | pred | ratio | gated |
|---|---|---|---|---|---|
| wrist | 32843 | 2984 | 2931.4 | 1.018 | ✓ |
| snap | 18726 | 2227 | 2269.9 | 0.981 | ✓ |
| slap | 7689 | 640 | 629.8 | 1.016 | ✓ |
| backhand | 5481 | 662 | 658.8 | 1.005 | ✓ |
| tip-in | 5425 | 720 | 780.5 | 0.923 | ✓ |
| deflected | 1327 | 211 | 210.2 | 1.004 | ✓ |
| wrap-around | 442 | 31 | 33.3 | 0.932 |  |
| poke | 299 | 41 | 35.9 | 1.142 |  |
| bat | 260 | 52 | 50.6 | 1.027 |  |
| cradle | 3 | 1 | 0.6 | 1.748 |  |
| between-legs | 47 | 8 | 6.0 | 1.335 |  |

- Final model (all 3 seasons, tuned settings) → `model/xg_sog_model_2025_26.json` (scoring defaults season=2025).

### Danger-band calibration (holdout)

| xg_sog band | n | actual rate | predicted rate |
|---|---|---|---|
| [0.00,0.05) | 5358 | 0.0239 | 0.0237 |
| [0.05,0.15) | 4785 | 0.0959 | 0.0937 |
| [0.15,0.30) | 3703 | 0.2098 | 0.2051 |
| [0.30,1.01) | 418 | 0.4139 | 0.4107 |

## Step 4 — Goalie attribution & verification

- Scored on-net shots: **72,542**. Attributed to a stint goalie: **72,542**.
- Shots with no stint match: **0**; null defending goalie: **0** (these would be empty-net only — expected 0 since empty-net shots were already excluded).
- Internal consistency: stint attribution matches the enriched facing `goalie_id` on **72,542/72,542** (100.00%) shots (both derived from the same shifts/boundary rule).
- **Joseph Woll overlap:** 10 scored shots fell in two-goalie stints (flag_two_goalies). Woll was credited with 1215 shots; per the stint build, two-goalie stints resolve to the goalie whose shift is longer at that instant (rapm_stints_report.md), so Woll's shots inherit that resolution.

**NHL API verification** (play-by-play `goalieInNetId`, 50 games = every 26th game_id): 2,649 shots checked, 2,649 matched to an API event, 0 unmatched (no on-net API event within ±2s). Agreement: **2,649/2,649 = 100.00%**.

## Step 5 — GSAX

- 98 goalies → `model/gsax_2025_26.csv`.

### Sanity checks

- **Identity check:** per-team Σ goalie all-sit GSAX vs team (Σxg_sog − GA) faced — max deviation **2.49e-14** (≤1e-6 required).
- **League total GSAX:** **+30.01** = total xg_sog (7607.0) − goals (7577). Non-zero because the OOF model total is +0.40% vs actual (the calibration gap, within the ±2% gate).

**Top 10 by all-sit GSAX**

| # | goalie | team | shots | sv% | all_gsax |
|---|---|---|---|---|---|
| 1 | Logan Thompson | WSH | 1587 | 0.912 | +29.646 |
| 2 | Jeremy Swayman | BOS | 1572 | 0.907 | +24.502 |
| 3 | Ilya Sorokin | NYI | 1530 | 0.906 | +23.964 |
| 4 | Andrei Vasilevskiy | TBL | 1485 | 0.911 | +22.222 |
| 5 | Igor Shesterkin | NYR | 1425 | 0.912 | +19.859 |
| 6 | Scott Wedgewood | COL | 1092 | 0.921 | +18.117 |
| 7 | Jet Greaves | CBJ | 1539 | 0.908 | +15.535 |
| 8 | Devin Cooley | CGY | 847 | 0.909 | +15.198 |
| 9 | Jesper Wallstedt | MIN | 1021 | 0.915 | +14.102 |
| 10 | Ukko-Pekka Luukkonen | BUF | 931 | 0.909 | +13.333 |

**Bottom 10 by all-sit GSAX**

| # | goalie | team | shots | sv% | all_gsax |
|---|---|---|---|---|---|
| 1 | Jordan Binnington | STL | 1010 | 0.873 | -23.947 |
| 2 | Leevi Meriläinen | OTT | 457 | 0.860 | -19.581 |
| 3 | Yaroslav Askarov | SJS | 1342 | 0.883 | -19.475 |
| 4 | Sergei Bobrovsky | FLA | 1250 | 0.877 | -18.898 |
| 5 | Samuel Ersson | PHI | 744 | 0.870 | -18.492 |
| 6 | Kevin Lankinen | VAN | 1285 | 0.875 | -16.732 |
| 7 | Adin Hill | VGK | 587 | 0.871 | -16.404 |
| 8 | Jacob Markstrom | NJD | 1113 | 0.883 | -16.264 |
| 9 | Cam Talbot | DET | 797 | 0.883 | -10.572 |
| 10 | Tristan Jarry | EDM/PIT | 830 | 0.882 | -10.181 |

**Top 10 by GSAX/100 (≥800 shots)**

| # | goalie | team | shots | sv% | all_gsax_100 |
|---|---|---|---|---|---|
| 1 | Logan Thompson | WSH | 1587 | 0.912 | +1.868 |
| 2 | Devin Cooley | CGY | 847 | 0.909 | +1.794 |
| 3 | Scott Wedgewood | COL | 1092 | 0.921 | +1.659 |
| 4 | Ilya Sorokin | NYI | 1530 | 0.906 | +1.566 |
| 5 | Jeremy Swayman | BOS | 1572 | 0.907 | +1.559 |
| 6 | Andrei Vasilevskiy | TBL | 1485 | 0.911 | +1.496 |
| 7 | Ukko-Pekka Luukkonen | BUF | 931 | 0.909 | +1.432 |
| 8 | Igor Shesterkin | NYR | 1425 | 0.912 | +1.394 |
| 9 | Jesper Wallstedt | MIN | 1021 | 0.915 | +1.381 |
| 10 | Alex Lyon | BUF | 980 | 0.906 | +1.228 |

**Bottom 10 by GSAX/100 (≥800 shots)**

| # | goalie | team | shots | sv% | all_gsax_100 |
|---|---|---|---|---|---|
| 1 | Jordan Binnington | STL | 1010 | 0.873 | -2.371 |
| 2 | Sergei Bobrovsky | FLA | 1250 | 0.877 | -1.512 |
| 3 | Jacob Markstrom | NJD | 1113 | 0.883 | -1.461 |
| 4 | Yaroslav Askarov | SJS | 1342 | 0.883 | -1.451 |
| 5 | Kevin Lankinen | VAN | 1285 | 0.875 | -1.302 |
| 6 | Tristan Jarry | EDM/PIT | 830 | 0.882 | -1.227 |
| 7 | Elvis Merzlikins | CBJ | 812 | 0.883 | -0.958 |
| 8 | Frederik Andersen | CAR | 849 | 0.874 | -0.767 |
| 9 | Arturs Silovs | PIT | 1020 | 0.887 | -0.709 |
| 10 | Joey Daccord | SEA | 1309 | 0.896 | -0.660 |

- **Distribution of GSAX/100 (≥800 shots, n=45):** mean +0.225, std 1.029, range [-2.371, +1.868].

- **Reliability (same halves as RAPM, ≥400 shots each half, n=36):** GSAX/100 half-to-half corr **0.213**; raw SV% corr 0.377.
- **Danger check:** see the danger-band calibration table in Step 3 (holdout actual vs predicted rate by xg_sog band).
- **Old vs new (≥800 shots, n=45):** Spearman rank corr **0.948** (reference only — old is unblocked/v1, new is on-goal/SOG).

  5 largest rank movers (old_rank → new_rank):
  - Juuse Saros: 31 → 17 (old GSAX -1.6, new +6.6)
  - Joey Daccord: 29 → 39 (old GSAX +0.2, new -8.6)
  - John Gibson: 7 → 16 (old GSAX +18.1, new +7.3)
  - Jake Oettinger: 18 → 26 (old GSAX +9.5, new +2.0)
  - Stuart Skinner: 21 → 28 (old GSAX +7.0, new +0.2)

## Choices I made that were not fully specified

- **On-net flag** taken from `shots_enriched.on_goal` (goals included; all goals have on_goal==1), joined positionally to `xg_features` (alignment verified). GSAX uses exactly the scored OOF on-net set; on-net shots dropped from xG training for impossible strength states (no xg_sog) are excluded from GSAX so the identity check holds exactly.
- **Attribution** uses the stint file's resolved goalie (two-goalie stints already resolved to the longer-shift goalie at the stint build), so Woll's overlap is handled upstream.
- **PK** = goalie's team shorthanded in states 4v5/3v5/3v4 (shooter on the power play); 5v5 requires both goalies in net and 5 skaters a side.
- **TOI/GP** from the stint file (goalie on ice), consistent with the RAPM layer.
- **NHL API verification** matches each on-net shot to a play-by-play shot-on-goal/goal event by shooter id and abs_secs within ±2s; shots with no such event are reported as unmatched, not counted against agreement.
## md5 integrity

All protected files re-hashed and compared to the pre-run baseline — see the session log for the full diff (expected: identical). New outputs this step: `model/train_xg_sog.py`, `model/build_gsax_2025_26.py`, `model/shots_xg_sog_2025_26.parquet`, `model/xg_sog_model_2025_26.json`, `model/xg_sog_meta_2025_26.json`, `model/gsax_2025_26.csv`, `gsax_report.md`.

