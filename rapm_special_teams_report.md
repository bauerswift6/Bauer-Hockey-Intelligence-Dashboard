# Special-Teams RAPM on xG — 2025-26 Regular Season

Deterministic: `model/train_rapm_special_teams.py` (imports `train_rapm_5v5.py`; does not modify it). Joint per-stint model: PP offense for the man-advantage skaters, PK defense for the shorthanded skaters. Same stint files, machinery, halves, bootstrap seed, and step-down kappa rule as 5v5. Read-only w.r.t. all protected files.

## Step 1 — Special-teams data

**2025-26** — 56,669 ST stints (12286:13 total).

| state | stints | minutes | PP attempts | PP goals | PP xG | SH attempts | SH goals | SH xG |
|---|---|---|---|---|---|---|---|---|
| 5v4 | 55,738 | 11915:54 | 15107 | 1444 | 1457.0 | 2455 | 175 | 174.8 |
| 5v3 | 505 | 203:29 | 543 | 89 | 78.3 | 10 | 0 | 0.5 |
| 4v3 | 426 | 166:50 | 338 | 36 | 42.0 | 20 | 1 | 1.3 |
| **all** | 56,669 | 12286:13 | 15988 | 1569 | 1577.3 | 2485 | 176 | 176.6 |

_SH (shorthanded) columns are reference only — not modeled._

**2024-25** — 54,355 ST stints (11578:26 total).

| state | stints | minutes | PP attempts | PP goals | PP xG | SH attempts | SH goals | SH xG |
|---|---|---|---|---|---|---|---|---|
| 5v4 | 53,551 | 11269:36 | 13993 | 1377 | 1333.5 | 2410 | 178 | 163.4 |
| 5v3 | 491 | 188:50 | 457 | 82 | 65.7 | 7 | 0 | 0.1 |
| 4v3 | 313 | 120:00 | 236 | 40 | 31.8 | 9 | 2 | 0.5 |
| **all** | 54,355 | 11578:26 | 14686 | 1499 | 1431.0 | 2426 | 180 | 164.0 |

_SH (shorthanded) columns are reference only — not modeled._

## Step 3 — 2024-25 prior (plain ridge, kappa=0)

- 2024-25 ST stints: 54,355, players: 872. Chosen lambda (xG prior): **16,000** (goals prior: 32,000).
- Saved `model/rapm_st_prior_2024_25.csv`.

**2024-25 top 10 PP offense (≥100 PP min) — sanity only:**

| # | player | pos | PP min | pp_offense |
|---|---|---|---|---|
| 1 | Nico Hischier | C | 211 | +0.98 |
| 2 | Jesper Bratt | L | 225 | +0.92 |
| 3 | Lucas Raymond | L | 250 | +0.89 |
| 4 | Dylan Larkin | C | 256 | +0.76 |
| 5 | John Tavares | C | 211 | +0.73 |
| 6 | Tomas Hertl | C | 196 | +0.70 |
| 7 | Jake Guentzel | C | 266 | +0.67 |
| 8 | Aleksander Barkov | C | 213 | +0.60 |
| 9 | Alex DeBrincat | R | 244 | +0.59 |
| 10 | Shea Theodore | D | 176 | +0.59 |

**2024-25 top 10 PK defense (≥100 PK min) — sanity only:**

| # | player | pos | PK min | pk_defense |
|---|---|---|---|---|
| 1 | Jonas Siegenthaler | D | 107 | +0.62 |
| 2 | Ilya Mikheyev | R | 156 | +0.60 |
| 3 | Ryan McDonagh | D | 204 | +0.57 |
| 4 | Kaiden Guhle | D | 115 | +0.56 |
| 5 | Jaccob Slavin | D | 231 | +0.53 |
| 6 | K'Andre Miller | D | 164 | +0.53 |
| 7 | Brett Kulak | D | 140 | +0.52 |
| 8 | Erik Haula | L | 101 | +0.46 |
| 9 | Connor Clifton | D | 200 | +0.45 |
| 10 | Warren Foegele | L | 123 | +0.41 |

## Step 4 — Out-of-sample kappa test

- Same split as 5v5: first half 656 games, second half 656. ST rows: half A 29,441, half B 27,228.
- Baseline (intercept+covariates only): A→B 208.433, B→A 186.941, avg **197.687**.

| kappa | lambda A | lambda B | MSE A→B | MSE B→A | avg MSE |
|---|---|---|---|---|---|
| 0.0 | 16,000 | 16,000 | 207.095 | 185.744 | **196.4198** |
| 0.25 | 16,000 | 16,000 | 206.925 | 185.597 | **196.2610** |
| 0.5 | 32,000 | 32,000 | 206.914 | 185.574 | **196.2441** |
| 0.75 | 32,000 | 32,000 | 206.786 | 185.487 | **196.1368** |
| 1.0 | 32,000 | 32,000 | 206.719 | 185.464 | **196.0917** |

**Bootstrap 95% CIs** (1,000 resamples, seed 42, both directions pooled; 1308 ST games). Negative ⇒ the larger kappa has lower MSE.

| difference | mean | 95% CI |
|---|---|---|
| 0.25 − 0.0 | -0.15887 | [-0.20333, -0.10946] |
| 0.5 − 0.0 | -0.17774 | [-0.29224, -0.06044] |
| 0.75 − 0.0 | -0.28481 | [-0.44731, -0.12405] |
| 1.0 − 0.0 | -0.32956 | [-0.53727, -0.11801] |
| 0.5 − 0.25 | -0.01888 | [-0.10458, +0.07034] |
| 0.75 − 0.5 | -0.10707 | [-0.16080, -0.04954] |
| 1.0 − 0.75 | -0.04474 | [-0.09800, +0.01138] |

### Selection walk

- Start at lowest-avg-MSE kappa = **1.0** (avg 196.0917).
- 1.0 vs 0.75: CI [-0.09800,+0.01138] includes 0 ⇒ step down to 0.75.
- 0.75 vs 0.5: CI [-0.16080,-0.04954] excludes 0 (0.5 worse) ⇒ **stop at 0.75**.
- Final check: 0.75 − 0 CI [-0.44731,-0.12405] excludes 0 ⇒ **0.75 beats kappa 0**.

- **Selected kappa = 0.75.**

### Reliability decomposition

| role | kappa | n | (a) half-half | (b) data-driven (−κ·prior) | (c) H1 vs prior | (c) H2 vs prior |
|---|---|---|---|---|---|---|
| pp_offense | 0.0 | 216 | 0.493 | 0.493 | 0.418 | 0.431 |
| pp_offense | 0.75 | 216 | 0.843 | 0.327 | 0.877 | 0.881 |
| pk_defense | 0.0 | 173 | 0.201 | 0.201 | 0.251 | 0.193 |
| pk_defense | 0.75 | 173 | 0.718 | 0.179 | 0.818 | 0.804 |

_Role minimum: 50 min in each half. Column (b) nets out the shared prior._

- Corr(2024-25 prior pp_offense, 2025-26 κ=0 full-season pp_offense), ≥100 min both seasons (n=206): **0.470**.
- Corr(2024-25 prior pk_defense, 2025-26 κ=0 full-season pk_defense), ≥100 min both seasons (n=143): **0.282**.

## Step 5 — Final fit & outputs

- Positions: read `model/player_positions.csv` (read-only); **0** missing players fetched into `model/player_positions_st_supplement.csv` (protected cache untouched).
- Chosen kappa **0.75**, full-season lambda **16,000** (κ=0 full-season lambda 16,000; goals version lambda 16,000).
- Saved `model/rapm_st_2025_26.csv` (887 players).
- xG vs goals corr: pp_offense (≥100 PP min, n=258) **0.555**; pk_defense (≥100 PK min, n=223) **0.458**.

### Sanity — five players (κ=0 beside)

| player | pos | PP min | pp_off | pp_impact | PP rank | κ0 pp_off | PK min | pk_def | pk_impact | PK rank | κ0 pk_def |
|---|---|---|---|---|---|---|---|---|---|---|---|
| McDavid | C | 294 | +1.00 | +4.89 | 1 | +0.86 | 84 | +0.48 | +0.67 | below min | +0.42 |
| Kucherov | R | 314 | +0.57 | +2.99 | 27 | +0.34 | 2 | +0.01 | +0.00 | below min | +0.01 |
| MacKinnon | C | 337 | +0.81 | +4.55 | 10 | +0.64 | 2 | +0.07 | +0.00 | below min | +0.03 |
| Brady Tkachuk | L | 193 | +0.58 | +1.87 | 24 | +0.50 | 4 | +0.03 | +0.00 | below min | +0.02 |
| Mark Stone | R | 202 | +0.87 | +2.95 | 6 | +0.76 | 69 | -0.11 | -0.13 | below min | -0.02 |

### Sanity — leaderboards

**Top 20 pp_offense (≥100 PP min)**

| # | player | pos | team | role min | value |
|---|---|---|---|---|---|
| 1 | Connor McDavid | C | EDM | 294 | +1.00 |
| 2 | Roope Hintz | C | DAL | 163 | +0.96 |
| 3 | Ryan Nugent-Hopkins | C | EDM | 252 | +0.91 |
| 4 | Dylan Larkin | C | DET | 238 | +0.90 |
| 5 | Tyler Toffoli | C | SJS | 228 | +0.88 |
| 6 | Mark Stone | R | VGK | 202 | +0.87 |
| 7 | Tomas Hertl | C | VGK | 278 | +0.87 |
| 8 | Jesper Bratt | L | NJD | 236 | +0.86 |
| 9 | Jason Robertson | L | DAL | 316 | +0.83 |
| 10 | Nathan MacKinnon | C | COL | 337 | +0.81 |
| 11 | Jake Guentzel | C | TBL | 321 | +0.81 |
| 12 | Alex DeBrincat | R | DET | 268 | +0.80 |
| 13 | Nico Hischier | C | NJD | 244 | +0.80 |
| 14 | Artemi Panarin | L | NYR/LAK | 260 | +0.77 |
| 15 | Lucas Raymond | L | DET | 257 | +0.77 |
| 16 | Wyatt Johnston | C | DAL | 287 | +0.75 |
| 17 | Leon Draisaitl | C | EDM | 232 | +0.73 |
| 18 | Mitch Marner | R | VGK | 286 | +0.73 |
| 19 | Adam Fox | D | NYR | 182 | +0.71 |
| 20 | John Tavares | C | TOR | 218 | +0.68 |

**Bottom 10 pp_offense (≥100 PP min)**

| # | player | pos | team | role min | value |
|---|---|---|---|---|---|
| 1 | Simon Holmstrom | R | NYI | 110 | -0.61 |
| 2 | Artturi Lehkonen | L | COL | 159 | -0.60 |
| 3 | Jordan Kyrou | R | STL | 154 | -0.58 |
| 4 | Alex Killorn | L | ANA | 120 | -0.54 |
| 5 | Brandon Montour | D | SEA | 120 | -0.51 |
| 6 | Will Smith | C | SJS | 223 | -0.48 |
| 7 | Filip Hronek | D | VAN | 152 | -0.48 |
| 8 | Pavel Buchnevich | L | STL | 189 | -0.46 |
| 9 | JJ Peterka | R | UTA | 157 | -0.43 |
| 10 | Justin Faulk | D | STL/DET | 150 | -0.43 |

**Top 20 pk_defense (≥100 PK min)**

| # | player | pos | team | role min | value |
|---|---|---|---|---|---|
| 1 | Noah Cates | L | PHI | 138 | +0.77 |
| 2 | Niko Mikkola | D | FLA | 160 | +0.76 |
| 3 | Ilya Mikheyev | R | CHI | 198 | +0.70 |
| 4 | Seth Jones | D | FLA | 117 | +0.66 |
| 5 | Nate Schmidt | D | UTA | 162 | +0.55 |
| 6 | Jaccob Slavin | D | CAR | 109 | +0.53 |
| 7 | Brayden McNabb | D | VGK | 149 | +0.50 |
| 8 | Jonas Siegenthaler | D | NJD | 140 | +0.50 |
| 9 | Brent Burns | D | COL | 179 | +0.48 |
| 10 | Tyler Kleven | D | OTT | 109 | +0.47 |
| 11 | Valeri Nichushkin | R | COL | 122 | +0.45 |
| 12 | Dougie Hamilton | D | NJD | 111 | +0.45 |
| 13 | Cale Makar | D | COL | 146 | +0.45 |
| 14 | K'Andre Miller | D | CAR | 133 | +0.44 |
| 15 | Matt Boldy | L | MIN | 127 | +0.43 |
| 16 | Eetu Luostarinen | C | FLA | 139 | +0.42 |
| 17 | Ryan Shea | D | PIT | 180 | +0.42 |
| 18 | Pius Suter | C | STL | 108 | +0.40 |
| 19 | Brock Nelson | C | COL | 147 | +0.39 |
| 20 | Ryan McDonagh | D | TBL | 148 | +0.38 |

**Bottom 10 pk_defense (≥100 PK min)**

| # | player | pos | team | role min | value |
|---|---|---|---|---|---|
| 1 | Jamie Oleksiak | D | SEA | 128 | -1.01 |
| 2 | Darnell Nurse | D | EDM | 122 | -0.81 |
| 3 | Chandler Stephenson | C | SEA | 144 | -0.80 |
| 4 | Mikael Backlund | C | CGY | 197 | -0.75 |
| 5 | Jacob Trouba | D | ANA | 249 | -0.74 |
| 6 | Parker Kelly | C | COL | 172 | -0.71 |
| 7 | Erik Gudbranson | D | CBJ | 101 | -0.68 |
| 8 | Charlie Coyle | C | CBJ | 167 | -0.66 |
| 9 | J.T. Compher | L | DET | 125 | -0.64 |
| 10 | J.T. Miller | C | NYR | 119 | -0.64 |

### Sanity — team clustering (top 20)

- pp_offense: distinct teams **10**, most from one team **3** (EDM).
- pk_defense: distinct teams **13**, most from one team **4** (COL).

### Sanity — team reconciliation

- pp_offense vs team PP xGF/60 − league (32 teams): **0.966**.
- pk_defense vs team PK xGA/60 (sign-flipped) (32 teams): **0.952**.

### Sanity — distributions (by position, role minimums)

| role | group | n | mean | std | min | max |
|---|---|---|---|---|---|---|
| pp_offense | F | 206 | +0.19 | 0.34 | -0.61 | +1.00 |
| pp_offense | D | 52 | +0.07 | 0.28 | -0.51 | +0.71 |
| pk_defense | F | 108 | -0.07 | 0.33 | -0.80 | +0.77 |
| pk_defense | D | 115 | -0.02 | 0.33 | -1.01 | +0.76 |

## Lambda grid & chosen lambdas

- Grid: `[500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]` (5v5 grid extended one step each way). Edges: 500, 512000.

| fit | lambda |
|---|---|
| 2024-25 prior (xG) | 16,000 |
| 2024-25 prior (goals) | 32,000 |
| half A (κ=0.0) | 16,000 |
| half B (κ=0.0) | 16,000 |
| half A (κ=0.25) | 16,000 |
| half B (κ=0.25) | 16,000 |
| half A (κ=0.5) | 32,000 |
| half B (κ=0.5) | 32,000 |
| half A (κ=0.75) | 32,000 |
| half B (κ=0.75) | 32,000 |
| half A (κ=1.0) | 32,000 |
| half B (κ=1.0) | 32,000 |
| full season κ=0 | 16,000 |
| full season κ=0.75 (final) | 16,000 |

_No lambda at a grid edge._

## md5 integrity — protected files (at report time)

All protected files re-hashed; must match the pre-run baseline (incl. `player_positions.csv`, untouched — 0 new players went to the supplement). New outputs this step: `model/rapm_st_prior_2024_25.csv`, `model/rapm_st_2025_26.csv`, `model/train_rapm_special_teams.py`, `rapm_special_teams_report.md`.

| file | md5 |
|---|---|
| `shots_augmented.parquet` | `47af38575046913c9699837bd3a1cd44` |
| `season_cache/shifts_20252026.parquet` | `919143f2d1eba4e93c81a87759940984` |
| `season_cache/shifts_20242025.parquet` | `2135d4bb361296b4034f7b7e43a76cb3` |
| `shifts_supplement_2024_25.parquet` | `6c22bae580e9dafdddabfbaf92bb9ca0` |
| `shots_enriched_2025_26.parquet` | `23bb76c3f95ca7146bbb9ac2ed7a11e7` |
| `shots_enriched_2024_25.parquet` | `ac4c0476722259c2889853e115822707` |
| `shots_enriched_2023_24.parquet` | `d4d14123b6acf97b9ece8a0c0c26b84e` |
| `xg_features_2025_26.parquet` | `bd8ea39c198b409e27ea20260b994b9c` |
| `xg_features_2024_25.parquet` | `ad61c9a731276f09d8daf695e5ff8bfd` |
| `xg_features_2023_24.parquet` | `3517d6c8c5f0cdb36050a574056b153b` |
| `shots_xg_2025_26.parquet` | `0abc0a0f883e94ed1487fa851c444c78` |
| `shots_xg_2025_26_v2.parquet` | `1373443f0e479bbb7a63ff8649d5aeb7` |
| `xg_model_2025_26.json` | `0656cdef6f5e36156da305ddd4638713` |
| `xg_model_2025_26_v2.json` | `7c9fe42daadb8ec396af64da9c0f6e8d` |
| `player_handedness.csv` | `fae143b3fc84e75ccb14825e5f05a783` |
| `rapm_stints_2025_26.parquet` | `cac68b5832a06fc887162567db880fe8` |
| `build_rapm_stints_2025_26.py` | `42b9634c3f18895d69ee354a87e19200` |
| `composite_ratings_sim.csv` | `d02630fd68286488dd951351c84dc85c` |
| `rapm_results.csv` | `53cc52783cb279da0c8f9ee9c07f5637` |
| `team_strength.csv` | `136632ddafc0d6234045ac2202594d55` |
| `season_simulation_results.csv` | `06ef68fffb575ad505262ce7f3ed1e20` |
| `train_rapm.py` | `3c0ff2dbed73fbc21b15097c90d4eb92` |
| `apply_rapm_shrinkage.py` | `c20a8fec90a6113e3fe80dc7a598630e` |
| `build_composite.py` | `32519c3cb5fa4bedabb719bd74c50164` |
| `build_self_generated_stats.py` | `5f5ce7973c05b3a55392e47d5cbcfe50` |
| `build_qoc_qot.py` | `ef6b4bb264d51c2afca9805d03d6aaba` |
| `build_rapm_stints.py` | `0cf9a5806173a4f80c6302adf498bcf3` |
| `score_xg_2024_25_v2.py` | `be5614fdbbe4c5e627085607874eb078` |
| `shots_xg_2024_25_v2.parquet` | `fb597368e033634b02203aa885a56783` |
| `rapm_stints_2024_25.parquet` | `30cd6542ca9d70b8781503bf9ae54987` |
| `rapm_prior_2024_25.csv` | `9a6bfc8045409ecc1d29f176983aa827` |
| `player_positions.csv` | `380ac77daa93c5cf9164b8e5c820f88e` |
| `train_rapm_5v5.py` | `cd4cc38bc1e84e439fa99912d9a7c15f` |
| `rapm_kappa_tiebreak.py` | `551456113f9b1e17782f84db1ee36431` |
| `rapm_5v5_2025_26.csv` | `4a34b4a48c0400f840bc47b45cbf5267` |
| `rapm_5v5_report.md` | `e9b4d32b9f21f76e3e4576fcb8b19b18` |
| `rapm_kappa_tiebreak_report.md` | `6328990c2c86b8d7eb81ba331bb6183b` |

## Choices I made that were not fully specified

- **Lambda grid** = 5v5 grid extended one step each way: `[500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000]`. Chosen by GroupKFold-by-game weighted MSE.
- **player_positions.csv is in the protected do-not-modify list**, yet Step 5 asks to append missing players. I reconciled by treating the cache as read-only and writing any newly fetched players to a separate `model/player_positions_st_supplement.csv` (same UA header/retry as 5v5), combined in memory. Count added is reported. Unknown positions never defaulted.
- **Score buckets** capped at ±2 from the PP team's perspective, tied = reference.
- **Goals version** uses the chosen kappa and the final xG lambda, with a goals-based 2024-25 ST prior built identically (used only when kappa>0).
- **Team reconciliation** predicted value = Σ(player role-rating × player-team role TOI) / team role TOI; actual PP = team PP xGF/60 − league; actual PK = −(team PK xGA/60 − league) so positive is better, matching pk_defense's sign.
- **Primary team / team list** ranked by combined (PP+PK) special-teams TOI.
