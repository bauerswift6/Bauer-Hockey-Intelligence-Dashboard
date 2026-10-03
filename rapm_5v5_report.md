# 5v5 RAPM on xG — 2025-26 Regular Season

Deterministic build: `model/train_rapm_5v5.py` (+ `model/build_rapm_stints.py`, `model/score_xg_2024_25_v2.py`). Read-only w.r.t. all protected inputs.

## Step 2 — Model data (2025-26)

- 5v5 stints (both goalies, 5v5): **445,437**
- Total 5v5 time: **63845:44** (1064.1 h)
- Design rows (two per stint): **890,874**
- Target: attacking team xG/60; weight: stint seconds. Covariates (unpenalized): intercept, home-attacking, score buckets (lead2+/lead1/tied[ref]/trail1/trail2+). Penalty on player offense/defense columns only.

## Step 3 — 2024-25 prior (plain ridge, kappa=0)

- 2024-25 5v5 stints: **453,064**, players: **920**
- Lambda grid: [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
- Chosen lambda (2024-25 xG prior): **32,000**  (goals prior lambda: 128,000)
- Saved `model/rapm_prior_2024_25.csv`.

**2024-25 top 10 by total (≥500 5v5 min) — sanity only:**

| # | player | pos | 5v5 min | off | def | total |
|---|---|---|---|---|---|---|
| 1 | Nathan MacKinnon | C | 1349 | +0.66 | +0.03 | +0.69 |
| 2 | Adam Fox | D | 1265 | +0.41 | +0.21 | +0.62 |
| 3 | Jack Hughes | C | 951 | +0.41 | +0.16 | +0.57 |
| 4 | Robert Thomas | C | 1032 | +0.48 | +0.07 | +0.56 |
| 5 | Brady Tkachuk | L | 985 | +0.43 | +0.10 | +0.54 |
| 6 | Connor McDavid | C | 1153 | +0.54 | -0.04 | +0.50 |
| 7 | Zach Hyman | L | 1080 | +0.45 | +0.05 | +0.50 |
| 8 | Leon Draisaitl | C | 1181 | +0.47 | +0.03 | +0.49 |
| 9 | Seth Jarvis | R | 957 | +0.18 | +0.30 | +0.48 |
| 10 | Shayne Gostisbehere | D | 994 | +0.29 | +0.15 | +0.44 |

## Step 4 — Out-of-sample test (choose kappa)

- Games split by date: first half **656** games (2025-10-07…2026-01-03), second half **656** (2026-01-04…2026-04-16).
- Baseline (intercept+covariates only): A→B 123.4882, B→A 119.9108, avg **121.6995**.

| kappa | lambda (fit A) | lambda (fit B) | MSE A→B | MSE B→A | avg MSE |
|---|---|---|---|---|---|
| 0.0 | 32,000 | 32,000 | 123.3074 | 119.7614 | **121.5344** |
| 0.25 | 64,000 | 64,000 | 123.2879 | 119.7258 | **121.5069** |
| 0.5 | 64,000 | 64,000 | 123.2738 | 119.7076 | **121.4907** |
| 0.75 | 64,000 | 64,000 | 123.2698 | 119.7003 | **121.4850** |
| 1.0 | 64,000 | 64,000 | 123.2757 | 119.7039 | **121.4898** |

- Best kappa>0 by avg MSE: **0.75** (avg 121.4850) vs kappa=0 (avg 121.5344).
- Game-level bootstrap (1,000; both directions) of MSE(best kappa>0) − MSE(kappa=0): observed **-0.04934**, 95% CI **[-0.06054, -0.03682]**.
- **Selection rule → chosen kappa = 0.75** (best kappa>0 (0.75) beats kappa=0 with a CI excluding 0).

- Player half-to-half stability (≥200 5v5 min each half, n=566):

| kappa | corr(total_H1, total_H2) |
|---|---|
| 0.0 | 0.332 |
| 0.75 | 0.739 |

## Step 5 — Final fit (full 2025-26)

- Chosen kappa **0.75**, lambda re-chosen on full season: **64,000**.
- kappa=0 full-season lambda: 32,000. Goals-version lambda (same as xG): 64,000.
- Saved `model/rapm_5v5_2025_26.csv` (940 players).
- Correlation xG-total vs goals-total (≥500 5v5 min, n=597): **0.538**.

### Sanity — five players

| player | pos | 5v5 min | off | def | total | total_impact (xG) | rank | κ0 off | κ0 def | κ0 total |
|---|---|---|---|---|---|---|---|---|---|---|
| McDavid | C | 1389 | +0.55 | -0.09 | +0.46 | +10.69 | 11 | +0.51 | -0.06 | +0.45 |
| Kucherov | R | 1113 | +0.53 | -0.12 | +0.42 | +7.74 | 20 | +0.43 | -0.14 | +0.28 |
| MacKinnon | C | 1313 | +0.75 | -0.02 | +0.74 | +16.10 | 1 | +0.69 | -0.05 | +0.63 |
| Brady Tkachuk | L | 751 | +0.49 | +0.12 | +0.61 | +7.63 | 5 | +0.36 | +0.09 | +0.46 |
| Mark Stone | R | 803 | +0.13 | +0.21 | +0.35 | +4.62 | 33 | +0.13 | +0.18 | +0.31 |

### Sanity — leaderboards (≥500 5v5 min)

**Top 20 by total**

| # | player | pos | team | 5v5 min | off | def | total |
|---|---|---|---|---|---|---|---|
| 1 | Nathan MacKinnon | C | COL | 1313 | +0.75 | -0.02 | +0.74 |
| 2 | Jordan Spence | D | OTT | 1212 | +0.43 | +0.24 | +0.67 |
| 3 | Jordan Kyrou | R | STL | 922 | +0.36 | +0.27 | +0.63 |
| 4 | Adam Fox | D | NYR | 1013 | +0.40 | +0.23 | +0.63 |
| 5 | Brady Tkachuk | L | OTT | 751 | +0.49 | +0.12 | +0.61 |
| 6 | Brandon Hagel | L | TBL | 984 | +0.51 | +0.04 | +0.55 |
| 7 | Robert Thomas | C | STL | 899 | +0.43 | +0.12 | +0.55 |
| 8 | Damon Severson | D | CBJ | 1319 | +0.22 | +0.25 | +0.47 |
| 9 | Moritz Seider | D | DET | 1580 | +0.13 | +0.34 | +0.46 |
| 10 | Zach Werenski | D | CBJ | 1551 | +0.39 | +0.07 | +0.46 |
| 11 | Connor McDavid | C | EDM | 1389 | +0.55 | -0.09 | +0.46 |
| 12 | Jack Hughes | C | NJD | 1015 | +0.38 | +0.07 | +0.45 |
| 13 | Lane Hutson | D | MTL | 1504 | +0.40 | +0.05 | +0.45 |
| 14 | Adam Pelech | D | NYI | 1426 | +0.11 | +0.33 | +0.44 |
| 15 | Quinn Hughes | D | MIN/VAN | 1606 | +0.40 | +0.03 | +0.43 |
| 16 | Jack Eichel | C | VGK | 1080 | +0.40 | +0.03 | +0.43 |
| 17 | Rasmus Dahlin | D | BUF | 1436 | +0.27 | +0.15 | +0.43 |
| 18 | Thomas Novak | C | PIT | 1021 | +0.40 | +0.03 | +0.43 |
| 19 | Mark Jankowski | L | CAR | 656 | +0.14 | +0.28 | +0.42 |
| 20 | Nikita Kucherov | R | TBL | 1113 | +0.53 | -0.12 | +0.42 |

**Top 10 offense**

| # | player | pos | team | 5v5 min | off | def | total |
|---|---|---|---|---|---|---|---|
| 1 | Nathan MacKinnon | C | COL | 1313 | +0.75 | -0.02 | +0.74 |
| 2 | Connor McDavid | C | EDM | 1389 | +0.55 | -0.09 | +0.46 |
| 3 | Nikita Kucherov | R | TBL | 1113 | +0.53 | -0.12 | +0.42 |
| 4 | Brandon Hagel | L | TBL | 984 | +0.51 | +0.04 | +0.55 |
| 5 | Brady Tkachuk | L | OTT | 751 | +0.49 | +0.12 | +0.61 |
| 6 | Jordan Spence | D | OTT | 1212 | +0.43 | +0.24 | +0.67 |
| 7 | Robert Thomas | C | STL | 899 | +0.43 | +0.12 | +0.55 |
| 8 | Sebastian Aho | C | CAR | 1050 | +0.42 | -0.08 | +0.34 |
| 9 | Quinn Hughes | D | MIN/VAN | 1606 | +0.40 | +0.03 | +0.43 |
| 10 | Lane Hutson | D | MTL | 1504 | +0.40 | +0.05 | +0.45 |

**Top 10 defense**

| # | player | pos | team | 5v5 min | off | def | total |
|---|---|---|---|---|---|---|---|
| 1 | Moritz Seider | D | DET | 1580 | +0.13 | +0.34 | +0.46 |
| 2 | Pontus Holmberg | R | TBL | 897 | +0.01 | +0.33 | +0.34 |
| 3 | Adam Pelech | D | NYI | 1426 | +0.11 | +0.33 | +0.44 |
| 4 | Mark Jankowski | L | CAR | 656 | +0.14 | +0.28 | +0.42 |
| 5 | Jordan Kyrou | R | STL | 922 | +0.36 | +0.27 | +0.63 |
| 6 | Garnet Hathaway | R | PHI | 602 | -0.23 | +0.27 | +0.04 |
| 7 | Oskar Bäck | C | DAL | 715 | -0.20 | +0.27 | +0.06 |
| 8 | Jaccob Slavin | D | CAR | 663 | -0.05 | +0.26 | +0.22 |
| 9 | Jonas Brodin | D | MIN | 1093 | -0.12 | +0.26 | +0.14 |
| 10 | Vincent Desharnais | D | SJS | 762 | -0.13 | +0.26 | +0.13 |

**Bottom 10 by total**

| # | player | pos | team | 5v5 min | off | def | total |
|---|---|---|---|---|---|---|---|
| 1 | Ty Dellandrea | C | SJS | 515 | -0.36 | -0.29 | -0.65 |
| 2 | Chandler Stephenson | C | SEA | 1094 | -0.23 | -0.36 | -0.58 |
| 3 | Ben Chiarot | D | DET | 1485 | -0.25 | -0.25 | -0.50 |
| 4 | Simon Benoit | D | TOR | 1107 | -0.30 | -0.18 | -0.48 |
| 5 | Liam Ohgren | L | VAN/MIN | 828 | -0.28 | -0.19 | -0.46 |
| 6 | Anthony Duclair | L | NYI | 719 | -0.18 | -0.27 | -0.44 |
| 7 | Ryan Lindgren | D | SEA | 1164 | -0.28 | -0.13 | -0.41 |
| 8 | Barclay Goodrow | C | SJS | 755 | -0.34 | -0.07 | -0.41 |
| 9 | Ryan Pulock | D | NYI | 1285 | -0.23 | -0.15 | -0.38 |
| 10 | Drew Helleson | D | ANA | 901 | -0.24 | -0.14 | -0.37 |

### Sanity — team clustering (top 20 by total)

- Distinct teams in top 20: **16**; most from any one team: **2** (OTT).

### Sanity — team reconciliation

- Teams: 32. Correlation(TOI-weighted Σ player totals, actual 5v5 xG diff/60): **0.991**.

### Sanity — distribution (≥500 5v5 min, by position group)

| group | n | metric | mean | std | min | max |
|---|---|---|---|---|---|---|
| F | 384 | offense | +0.04 | 0.16 | -0.40 | +0.75 |
| F | 384 | defense | -0.00 | 0.12 | -0.40 | +0.33 |
| F | 384 | total | +0.04 | 0.19 | -0.65 | +0.74 |
| D | 213 | offense | +0.01 | 0.14 | -0.32 | +0.43 |
| D | 213 | defense | +0.01 | 0.12 | -0.28 | +0.34 |
| D | 213 | total | +0.02 | 0.20 | -0.50 | +0.67 |

## Choices I made that were not fully specified

- **Lambda grid**: [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000] (log-spaced), chosen by GroupKFold-by-game weighted MSE, per the spec. Reported per fit above.
- **2024-25 xG for the prior** scored OOF with the v2 config (depth 5, lr 0.03, 435 trees, season=2024), 5-fold GroupKFold by game, each fold trained on the other 2024-25 folds + all 2023-24 + all 2025-26 → `model/shots_xg_2024_25_v2.parquet` (script `score_xg_2024_25_v2.py`). Total xG within 0.5% of goals.
- **Score buckets** capped at ±2 (lead2+/lead1/tied/trail1/trail2+), tied = reference.
- **Positions** fetched live from the NHL API player-landing endpoint and cached to `model/player_positions.csv` (no cached goalie-validation file existed); unknown positions left null, never defaulted. F = C/L/R, D = D.
- **Players absent from a fit's data** get coefficient 0 (kappa=0) or kappa·prior (kappa>0), which falls out of the ridge normal equations naturally.
- **Goals version** uses the same chosen kappa and the same xG-fit lambda, with a goals-based 2024-25 prior built identically (kappa>0 only).
- **Team reconciliation** predicted value = Σ(player total × player-team 5v5 TOI) / team 5v5 stint-seconds, compared to actual team 5v5 xG differential/60.
## md5 integrity — protected files (at report time)

All protected inputs, model artifacts, and the four simulator CSVs are re-hashed below and compared to the pre-build baseline recorded at the session start — **all unchanged**. New outputs written this step: `model/rapm_stints_2024_25.parquet`, `model/shots_xg_2024_25_v2.parquet`, `model/rapm_prior_2024_25.csv`, `model/rapm_5v5_2025_26.csv`, `model/player_positions.csv`, and the scripts.

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

