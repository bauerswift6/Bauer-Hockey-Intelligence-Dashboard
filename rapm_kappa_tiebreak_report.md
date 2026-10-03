# 5v5 RAPM — kappa tiebreak & reliability diagnostic

Deterministic: `model/rapm_kappa_tiebreak.py` (imports `train_rapm_5v5.py`, does not modify it). Same data, model, halves, lambda procedure, and bootstrap seed as the previous run.

## Step 1 — Kappa selection table (reproduction) + bootstrap

| kappa | MSE A→B | MSE B→A | avg MSE | prev avg | |Δ| |
|---|---|---|---|---|---|
| 0.0 | 123.3074 | 119.7614 | **121.5344** | 121.5344 | 0.0000 |
| 0.25 | 123.2879 | 119.7258 | **121.5069** | 121.5069 | 0.0000 |
| 0.5 | 123.2738 | 119.7076 | **121.4907** | 121.4907 | 0.0000 |
| 0.75 | 123.2698 | 119.7003 | **121.4850** | 121.4850 | 0.0000 |
| 1.0 | 123.2757 | 119.7039 | **121.4898** | 121.4898 | 0.0000 |
| baseline | 123.4882 | 119.9108 | **121.6995** | 121.6995 | 0.0000 |

**Bootstrap 95% CIs of MSE differences** (1,000 resamples, seed 42, both directions pooled). Negative ⇒ the first (larger) kappa has lower MSE.

| difference | mean | 95% CI |
|---|---|---|
| 0.25 − 0.0 | -0.02733 | [-0.03391, -0.02011] |
| 0.5 − 0.0 | -0.04354 | [-0.05177, -0.03413] |
| 0.75 − 0.0 | -0.04929 | [-0.06054, -0.03682] |
| 1.0 − 0.0 | -0.04459 | [-0.05890, -0.02865] |
| 0.5 − 0.25 | -0.01621 | [-0.02017, -0.01225] |
| 0.75 − 0.5 | -0.00575 | [-0.00966, -0.00186] |
| 1.0 − 0.75 | +0.00470 | [+0.00075, +0.00864] |

_Consistency: 0.75 − 0 reproduces the previous bootstrap ([-0.06054, -0.03682] vs previously [-0.06054, -0.03682])._

### Selection walk (step-down tiebreak)

- Start at lowest-avg-MSE kappa = **0.75** (avg 121.4850).
- 0.75 vs 0.5: CI((0.75)−(0.5)) = [-0.00966,-0.00186] excludes 0 with 0.5 worse ⇒ **stop at 0.75**.
- Final check: 0.75 − 0 CI [-0.06054,-0.03682] excludes 0 ⇒ **0.75 beats kappa 0**.

- **Selected kappa = 0.75** (previous pick was 0.75).

## Step 2 — Reliability diagnostic

- Players with ≥200 5v5 min in each half: **566**.

| kappa | (a) corr(total_H1,total_H2) | (b) corr data-driven (total − κ·prior) | (c) corr(total_H1, prior) | (c) corr(total_H2, prior) |
|---|---|---|---|---|
| 0.0 | 0.332 | 0.332 | 0.374 | 0.300 |
| 0.5 | 0.645 | 0.289 | 0.740 | 0.686 |
| 0.75 | 0.739 | 0.269 | 0.822 | 0.782 |

_Reproduces previous half-to-half: kappa 0 ≈ 0.332, kappa 0.75 ≈ 0.739. Column (b) nets out the shared prior: it is the repeatable 2025-26 signal only._

- Corr(2024-25 prior total, 2025-26 κ=0 full-season total), ≥500 5v5 min both seasons (n=500): **0.407**.

## Step 4 — Lambda grid & chosen lambdas

- Grid (all fits): `[1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]` — edges 1000 and 256000.

| fit | chosen lambda |
|---|---|
| 2024-25 prior (xG, κ=0) | 32,000 |
| 2024-25 prior (goals, κ=0) | 128,000 |
| half fit A (κ=0.0) | 32,000 |
| half fit B (κ=0.0) | 32,000 |
| half fit A (κ=0.25) | 64,000 |
| half fit B (κ=0.25) | 64,000 |
| half fit A (κ=0.5) | 64,000 |
| half fit B (κ=0.5) | 64,000 |
| half fit A (κ=0.75) | 64,000 |
| half fit B (κ=0.75) | 64,000 |
| half fit A (κ=1.0) | 64,000 |
| half fit B (κ=1.0) | 64,000 |
| full season κ=0 | 32,000 |
| full season κ=0.75 (this run) | 64,000 |

_No lambda sits at a grid edge (all interior); none changed._

## Step 3 — Refit decision

- Selected kappa = 0.75 (unchanged). **No new model outputs written** — `model/rapm_5v5_2025_26.csv` already reflects the chosen kappa.

## md5 integrity — protected files (at report time)

All protected files below (incl. the four simulator CSVs) are re-hashed here and must match the pre-run baseline. This step wrote only `rapm_kappa_tiebreak_report.md`.

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
| `rapm_5v5_2025_26.csv` | `4a34b4a48c0400f840bc47b45cbf5267` |
| `rapm_5v5_report.md` | `e9b4d32b9f21f76e3e4576fcb8b19b18` |

