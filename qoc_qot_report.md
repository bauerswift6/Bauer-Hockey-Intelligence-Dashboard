# 5v5 QoC / QoT — 2025-26 Regular Season (RAPM-based rebuild)

Deterministic build: `model/build_qoc_qot_2025_26.py`. Read-only w.r.t. all protected inputs. Output: `model/qoc_qot_5v5_2025_26.csv`.

## Step 1 — Old QoC/QoT pipeline inventory (diagnostic; nothing modified)

**Builder:** `model/build_qoc_qot.py` (exists).

**Inputs**
- `model/season_cache/shifts_20252026.parquet` — columns `game_id, player_id,
  team_abbrev, abs_start, abs_end` (regular season filtered in-script via
  `(game_id // 10000) % 100 == 2`).
- `model/composite_ratings_single_season.csv` — columns `player_id,
  composite_war` (the GAR rating), `player_name`, `team`.

**What it computes**
- Rating used: **`composite_war` (GAR)**, NOT RAPM.
- Game states: **ALL strengths** — raw pairwise shift-overlap across every shift,
  no 5v5 / strength filter (goalies are included in the overlap because the shift
  file is not role-filtered; only same-player self-overlap is removed).
- Weighting: shared on-ice **overlap seconds** between each focal player and each
  partner, summed across games. `same_team=True → QoT`, `False → QoC`. Partners
  with no GAR are dropped from numerator and denominator.
- Output kept only for players who are themselves in the composite table.

**Output file:** `model/skater_qoc_qot_single_season.csv`
**Columns:** `player_id, player_name, team, qoc, qot, qoc_toi_secs, qot_toi_secs`
(qoc/qot rounded to 3; the two `*_toi_secs` are total shared seconds).

**Dependencies:** reads `season_cache/shifts_20252026.parquet` (yes) and
`composite_ratings_single_season.csv`. **Does NOT read `rapm_results.csv`** or any
RAPM file; nothing else in `season_cache/`.

**Where the dashboard reads it (file:line)**
- `app.py:2095-2096` — `QOC_QOT_PATH = model/skater_qoc_qot_single_season.csv`.
- `app.py:2099-2112` — `_load_qoc_qot()` loads `qoc, qot, qoc_toi_secs,
  qot_toi_secs` per player_id (cached as `qoc_qot_lookup`).
- `app.py:3313` — `my_qq = _load_qoc_qot().get(pid)` in the player-payload builder.
- `app.py:3460-3461` — emits `"qoc"` and `"qot"` (rounded 3) into the player API
  payload (null when missing).
- `static/js/players_leaderboards.js:156-157` — leaderboard columns `qoc`/`qot`
  (label "QoC"/"QoT", group "usage", source "skaters", 3-decimal signed fmt).
- `static/js/glossary_data.js:1057-1092` — glossary entries `qoc`/`qot` (prose
  only, no data read).

**Note:** the old metric is GAR-based and all-strengths; the new build below is
**5v5-only and RAPM-based**, so values are on a different scale (RAPM xG/60, ~0)
and are written to a **separate** file — no dashboard wiring is touched here.

## Step 2 — Definitions & counts

- 5v5 stint set (both goalies, 5v5): **445,437** stints, **63845:44** total — matches rapm_5v5_report.md (445,437 / 63845:44) exactly.
- Skaters: **940**, all present in `rapm_5v5_2025_26.csv` (0 missing).
- Teammates = the other 4 skaters on the player's team; opponents = the 5 opposing skaters; goalies never included.
- qot_* / qoc_* = duration-weighted average, over the player's 5v5 stints, of the mean teammate / opponent rating (offense, defense, total from `rapm_5v5_2025_26.csv`). qot_toi / qoc_toi use each skater's 5v5 TOI-per-GP.
- **League duration-weighted average RAPM total across all 5v5 player-seconds: +0.0347** (this is what 'average' QoC/QoT is centered near).

## Step 3 — Hand check & determinism

| player | qot_total (main) | qot_total (independent) | qoc_total (main) | qoc_total (independent) | agree<1e-9 |
|---|---|---|---|---|---|
| McDavid | 0.134902095 | 0.134902095 | 0.047870857 | 0.047870857 | ✅ |
| Mark Stone | 0.157446508 | 0.157446508 | 0.022963311 | 0.022963311 | ✅ |

- Hand check passes to <1e-9 via an independent per-stint loop.
- Output md5: `2e173456cd1c7d27225e9e1c1be95c44` (determinism confirmed by re-running the script and comparing — see session log).

## Step 4 — Sanity checks (≥500 5v5 min unless noted)

### Five players

| player | pos | qot_tot | qot_off | qot_def | qoc_tot | qoc_off | qoc_def | qot_toi | qoc_toi | rank qot | rank qoc | top linemate (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| McDavid | C | +0.135 | +0.132 | +0.002 | +0.048 | +0.040 | +0.008 | 15.45 | 14.47 | 58 | 61 | Evan Bouchard (913) |
| Kucherov | R | +0.131 | +0.114 | +0.017 | +0.041 | +0.038 | +0.003 | 14.46 | 14.30 | 62 | 184 | Brandon Hagel (512) |
| MacKinnon | C | +0.159 | +0.140 | +0.019 | +0.039 | +0.034 | +0.005 | 15.53 | 14.41 | 30 | 212 | Martin Necas (1053) |
| Brady Tkachuk | L | +0.113 | +0.077 | +0.036 | +0.042 | +0.037 | +0.005 | 14.48 | 14.20 | 82 | 161 | Dylan Cozens (451) |
| Mark Stone | R | +0.157 | +0.109 | +0.048 | +0.023 | +0.027 | -0.004 | 15.48 | 14.39 | 31 | 507 | Ivan Barbashev (475) |

_Ranks among 597 players with ≥500 5v5 min._

### Distributions by position

| group | n | metric | mean | std | min | max |
|---|---|---|---|---|---|---|
| F | 384 | qot_total | +0.032 | 0.070 | -0.136 | +0.307 |
| F | 384 | qot_offense | +0.028 | 0.054 | -0.098 | +0.259 |
| F | 384 | qot_defense | +0.004 | 0.037 | -0.113 | +0.104 |
| F | 384 | qoc_total | +0.035 | 0.011 | +0.004 | +0.063 |
| F | 384 | qoc_offense | +0.033 | 0.011 | +0.003 | +0.060 |
| F | 384 | qoc_defense | +0.002 | 0.004 | -0.011 | +0.013 |
| F | 384 | qot_toi | +14.447 | 0.586 | +12.799 | +16.277 |
| F | 384 | qoc_toi | +14.054 | 0.249 | +13.384 | +14.583 |
| D | 213 | qot_total | +0.039 | 0.068 | -0.126 | +0.249 |
| D | 213 | qot_offense | +0.038 | 0.048 | -0.060 | +0.208 |
| D | 213 | qot_defense | +0.000 | 0.034 | -0.085 | +0.100 |
| D | 213 | qoc_total | +0.034 | 0.011 | +0.002 | +0.059 |
| D | 213 | qoc_offense | +0.032 | 0.012 | -0.002 | +0.059 |
| D | 213 | qoc_defense | +0.002 | 0.004 | -0.009 | +0.016 |
| D | 213 | qot_toi | +13.450 | 0.428 | +12.484 | +14.889 |
| D | 213 | qoc_toi | +14.044 | 0.204 | +13.470 | +14.488 |

### Leaderboards

**Top 10 qot_total**

| # | player | pos | team | 5v5 min | value |
|---|---|---|---|---|---|
| 1 | Martin Necas | C | COL | 1270 | +0.307 |
| 2 | Artturi Lehkonen | L | COL | 998 | +0.254 |
| 3 | Cale Makar | D | COL | 1300 | +0.249 |
| 4 | Gabriel Landeskog | L | COL | 793 | +0.225 |
| 5 | Devon Toews | D | COL | 1189 | +0.224 |
| 6 | Jalen Chatfield | D | CAR | 1230 | +0.214 |
| 7 | Valeri Nichushkin | R | COL | 948 | +0.208 |
| 8 | Alexander Nikishin | D | CAR | 1218 | +0.202 |
| 9 | Andrei Svechnikov | R | CAR | 1027 | +0.201 |
| 10 | Anthony Cirelli | C | TBL | 873 | +0.195 |

**Bottom 10 qot_total**

| # | player | pos | team | 5v5 min | value |
|---|---|---|---|---|---|
| 1 | Eeli Tolvanen | R | SEA | 980 | -0.136 |
| 2 | Marcus Pettersson | D | VAN | 1471 | -0.126 |
| 3 | Zack Ostapchuk | C | SJS | 518 | -0.116 |
| 4 | Linus Karlsson | C | VAN | 852 | -0.111 |
| 5 | Andre Burakovsky | L | CHI | 1010 | -0.105 |
| 6 | Liam Ohgren | L | VAN/MIN | 828 | -0.105 |
| 7 | Elias Pettersson | D | VAN | 954 | -0.103 |
| 8 | Ryan Greene | C | CHI | 1068 | -0.102 |
| 9 | Tom Willander | D | VAN | 1024 | -0.100 |
| 10 | Jaden Schwartz | L | SEA | 646 | -0.099 |

**Top 10 qoc_total**

| # | player | pos | team | 5v5 min | value |
|---|---|---|---|---|---|
| 1 | Auston Matthews | C | TOR | 918 | +0.063 |
| 2 | Nick Suzuki | C | MTL | 1206 | +0.063 |
| 3 | Bo Horvat | C | NYI | 915 | +0.063 |
| 4 | Cole Caufield | R | MTL | 1088 | +0.062 |
| 5 | Jason Dickinson | C | CHI/EDM | 831 | +0.059 |
| 6 | Dylan Larkin | C | DET | 1036 | +0.059 |
| 7 | Mike Matheson | D | MTL | 1442 | +0.059 |
| 8 | Ryan O'Reilly | C | NSH | 1149 | +0.058 |
| 9 | Louis Crevier | D | CHI | 1159 | +0.057 |
| 10 | Alex Vlasic | D | CHI | 1415 | +0.057 |

**Bottom 10 qoc_total**

| # | player | pos | team | 5v5 min | value |
|---|---|---|---|---|---|
| 1 | Ben Hutton | D | VGK | 782 | +0.002 |
| 2 | Mathieu Joseph | R | STL/LAK | 564 | +0.004 |
| 3 | Cole Reinhardt | L | VGK/FLA | 616 | +0.005 |
| 4 | Mike Reilly | D | CAR | 568 | +0.007 |
| 5 | Colton Sissons | C | VGK | 686 | +0.007 |
| 6 | Lars Eller | C | OTT | 679 | +0.008 |
| 7 | Eric Robinson | L | CAR | 684 | +0.008 |
| 8 | Luke Kunin | C | FLA | 531 | +0.009 |
| 9 | Ty Emberson | D | EDM | 1016 | +0.009 |
| 10 | Ryan Lomberg | L | CGY | 510 | +0.010 |

### Team view — TOI-weighted avg qoc_total

- Teams: 32. Spread: min +0.0256 (VGK), max +0.0425 (TOR), range 0.0169, std 0.0045.
_QoC spread across teams is expected to be small — everyone plays everyone._

### Correlations (≥500 5v5 min)

- qot_total vs qot_toi: **0.176**
- qoc_total vs qoc_toi: **0.798**
- qot_total vs own RAPM total: **0.295**

