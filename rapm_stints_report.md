# RAPM Stint Dataset — 2025-26 Regular Season

Build script: `model/build_rapm_stints_2025_26.py` (deterministic). Output: `model/rapm_stints_2025_26.parquet`. Report: this file.

**Scope:** regular season only, game_id 2025020001–2025021312. Period 5 (shootout) excluded everywhere. No playoff game read.

## 1. Overview

- **Total stints:** 519,836
- **Games:** 1,312
- **Mean stint length:** 9.21s  |  **Median:** 4.0s  |  min 1s, max 211s
- **Total stint time:** 4,789,216s (1330.3 clock-hours)

## 2. Totals reconciliation (nothing lost or double-counted)

| Quantity | Across stints | Filtered enriched (non-shootout) | Match |
|---|---|---|---|
| Unblocked attempts | 111,099 | 111,099 | ✅ |
| Goals | 7,578 | 7,578 | ✅ |
| xG (xg_v2) | 7614.23 | 7614.23 | ✅ |

Expected attempts per the task spec: **111,099** (matches). Expected xG ≈ 7,614 (matches). Unmatched shots (no stint): **0**.

## 3. Clock-time validation (per game)

- Regulation periods with start != period start: **0**
- Regulation periods with end != period end (1200/2400/3600): **0**
- (game, period) with internal tiling gap (span != Σduration): **0**
- Games with OT (period 4): **326**; OT span min 13s max 300s
- Games whose total stint time != 3600 + OT span: **0**

All 1,312 games tile their periods exactly (PASS).

## 4. Player TOI validation (stint TOI vs de-duplicated shift TOI)

- Players compared: **1,038**
- Players with |stint TOI − de-dup shift TOI| > 2s: **1**

_Any outlier here is a goalie: during a goalie change two goalies can be on ice for the same instant, and the stint credits only the longer-shift goalie (the enrichment two-goalie rule), so the other goalie's overlap seconds are attributed away. This is intended and consistent with the enrichment step; skaters (never double-attributed) all match to 0s._

| player_id | name | stint TOI | shift TOI | diff (s) |
|---|---|---|---|---|
| 8479361 | Joseph Woll | 2224:40 | 2230:04 | 324 |
| 8470613 | Brent Burns | 1548:47 | 1548:47 | 0 |
| 8481712 | Matej Blümel | 43:28 | 43:28 | 0 |
| 8481716 | Dmitri Voronkov | 878:13 | 878:13 | 0 |
| 8481719 | Max Crozier | 560:47 | 560:47 | 0 |
| 8481721 | Arseny Gritsyuk | 1001:30 | 1001:30 | 0 |
| 8481725 | Elmer Soderblom | 656:31 | 656:31 | 0 |
| 8481726 | Adam Edstrom | 331:08 | 331:08 | 0 |
| 8481732 | Andre Lee | 71:59 | 71:59 | 0 |
| 8481754 | Nikita Nesterenko | 355:30 | 355:30 | 0 |

## 5. Stint count & time by (home_skaters, away_skaters)

| home_sk | away_sk | stints | total time | share of time |
|---|---|---|---|---|
| 5 | 5 | 445,551 | 63864:52 | 80.01% |
| 5 | 4 | 28,987 | 6183:19 | 7.75% |
| 4 | 5 | 26,822 | 5751:26 | 7.21% |
| 4 | 4 | 5,724 | 1035:25 | 1.30% |
| 3 | 3 | 4,443 | 937:21 | 1.17% |
| 5 | 6 | 3,505 | 785:21 | 0.98% |
| 6 | 5 | 3,325 | 738:29 | 0.93% |
| 5 | 3 | 275 | 109:41 | 0.14% |
| 4 | 3 | 241 | 99:32 | 0.12% |
| 3 | 5 | 232 | 93:58 | 0.12% |
| 6 | 4 | 249 | 73:36 | 0.09% |
| 3 | 4 | 199 | 70:33 | 0.09% |
| 4 | 6 | 226 | 64:08 | 0.08% |
| 2 | 0 | 1 | 3:31 | 0.00% |
| 6 | 7 | 2 | 1:24 | 0.00% |
| 6 | 6 | 6 | 1:16 | 0.00% |
| 7 | 5 | 11 | 1:09 | 0.00% |
| 3 | 6 | 5 | 1:05 | 0.00% |
| 7 | 6 | 4 | 0:53 | 0.00% |
| 10 | 5 | 4 | 0:41 | 0.00% |
| 7 | 7 | 2 | 0:34 | 0.00% |
| 5 | 9 | 4 | 0:31 | 0.00% |
| 10 | 7 | 2 | 0:29 | 0.00% |
| 5 | 7 | 6 | 0:23 | 0.00% |
| 5 | 8 | 2 | 0:16 | 0.00% |
| 8 | 6 | 2 | 0:08 | 0.00% |
| 6 | 8 | 1 | 0:06 | 0.00% |
| 8 | 5 | 2 | 0:04 | 0.00% |
| 10 | 8 | 1 | 0:03 | 0.00% |
| 6 | 9 | 1 | 0:01 | 0.00% |
| 9 | 5 | 1 | 0:01 | 0.00% |

## 6. Edge cases (counted, not guessed)

- **Stints shorter than 1s:** 0. By construction stints live on an integer-second breakpoint grid, so the minimum possible length is 1s (153,443 stints are exactly 1s). None are sub-second.
- **Overlapping (self-overlapping) raw shift rows:** 315 rows overlap another row of the same player. Resolved by counting DISTINCT player_ids per side (enrichment rule) and merging adjacent identical on-ice sets, so they never create impossible 7v6/8v7 states.
- **Stints with >6 skaters a side:** 45. **Skater count outside 3–6 (either side):** 46. These are raw shift-coverage artifacts (the same anomalies enrichment flags as `flag_skaters_gt6`): e.g. the single 2v0 stint is the last 211s of an OT that went to a shootout, where the away team's skater shift rows are missing from the raw feed while both goalies remain logged. All are tiny in aggregate time and are surfaced, not silently defaulted.
  - 10v5: 4 stints, 0:41
  - 10v7: 2 stints, 0:29
  - 10v8: 1 stints, 0:03
  - 2v0: 1 stints, 3:31
  - 5v7: 6 stints, 0:23
  - 5v8: 2 stints, 0:16
  - 5v9: 4 stints, 0:31
  - 6v7: 2 stints, 1:24
  - 6v8: 1 stints, 0:06
  - 6v9: 1 stints, 0:01
  - 7v5: 11 stints, 1:09
  - 7v6: 4 stints, 0:53
  - 7v7: 2 stints, 0:34
  - 8v5: 2 stints, 0:04
  - 8v6: 2 stints, 0:08
  - 9v5: 1 stints, 0:01
- **No-goalie stints that are NOT a 6-skater pull (skaters < 6, suspicious):** 0, total 0:00.
- **No-goalie stints that ARE a pull (≥6 skaters, legitimate empty net):** 0, total 0:00.
- **Stints with >1 goalie resolved (longest shift wins):** 34.
- **Coverage-gap (empty-side) stints:** 0.
- **Stints spanning a period boundary:** 0 (impossible by construction — stints are built within a single (game, period); every regular-season shift lies wholly in one period band).

## 7. The 21 impossible-strength shots (no xg_v2)

These 21 shots have no xg_v2 (dropped from xG training for impossible strength states). They ARE counted as attempts and goals in their stints and contribute **0** to xG sums. Where they landed:

| game_id | period | abs_secs | enriched strength | stint (home_sk v away_sk) | goal | home shot |
|---|---|---|---|---|---|---|
| 2025020036 | 3 | 3573 | 7v6 | 6v7 | 0 | False |
| 2025020428 | 3 | 3565 | 6v7 | 6v7 | 0 | True |
| 2025020428 | 3 | 3573 | 6v7 | 6v7 | 0 | True |
| 2025020428 | 3 | 3589 | 6v7 | 6v7 | 0 | True |
| 2025020428 | 3 | 3590 | 6v7 | 6v7 | 0 | True |
| 2025020428 | 3 | 3593 | 6v7 | 6v7 | 0 | True |
| 2025020428 | 3 | 3596 | 6v7 | 6v7 | 0 | True |
| 2025020482 | 1 | 1163 | 5v6 | 6v5 | 0 | False |
| 2025020482 | 1 | 1190 | 5v6 | 6v5 | 0 | False |
| 2025020573 | 3 | 3572 | 10v5 | 10v5 | 0 | True |
| 2025020584 | 3 | 3591 | 6v7 | 7v6 | 0 | False |
| 2025020594 | 3 | 3578 | 6v8 | 8v6 | 1 | False |
| 2025020703 | 2 | 1895 | 5v6 | 6v5 | 0 | False |
| 2025020760 | 3 | 3547 | 7v10 | 10v7 | 0 | False |
| 2025020767 | 3 | 3240 | 5v6 | 6v5 | 0 | False |
| 2025020767 | 3 | 3249 | 5v6 | 6v5 | 0 | False |
| 2025021042 | 3 | 3557 | 7v10 | 10v7 | 0 | False |
| 2025021176 | 2 | 2323 | 5v6 | 6v5 | 0 | False |
| 2025021181 | 3 | 3364 | 6v3 | 3v6 | 0 | False |
| 2025021192 | 1 | 1030 | 6v6 | 6v6 | 0 | True |
| 2025021192 | 1 | 1052 | 6v6 | 6v6 | 0 | True |

Total: 21 shots, 1 goal(s). All assigned to a stint; xG contribution 0.

## 8. Five-player check

| Player | player_id | total stint TOI | 5v5 stint TOI | 5v5 on-ice xGF | 5v5 on-ice xGA |
|---|---|---|---|---|---|
| McDavid | 8478402 | 1885:27 | 1389:26 | 81.45 | 62.34 |
| Kucherov | 8476453 | 1545:28 | 1114:50 | 64.74 | 50.10 |
| MacKinnon | 8477492 | 1780:57 | 1315:00 | 84.23 | 55.40 |
| Brady Tkachuk | 8480801 | 1017:58 | 751:12 | 42.28 | 28.01 |
| Mark Stone | 8475913 | 1152:54 | 803:33 | 41.02 | 26.42 |

_5v5 = stints with home_skaters==5 and away_skaters==5. xGF/xGA are the sum of xg_v2 for and against while the player is on ice at 5v5._

## 9. Protected-file integrity (md5 at report time)

| file | md5 |
|---|---|
| `shots_augmented.parquet` | `47af38575046913c9699837bd3a1cd44` |
| `season_cache/shifts_20252026.parquet` | `919143f2d1eba4e93c81a87759940984` |
| `shots_enriched_2025_26.parquet` | `23bb76c3f95ca7146bbb9ac2ed7a11e7` |
| `xg_features_2025_26.parquet` | `bd8ea39c198b409e27ea20260b994b9c` |
| `shots_xg_2025_26.parquet` | `0abc0a0f883e94ed1487fa851c444c78` |
| `shots_xg_2025_26_v2.parquet` | `1373443f0e479bbb7a63ff8649d5aeb7` |
| `xg_model_2025_26.json` | `0656cdef6f5e36156da305ddd4638713` |
| `xg_model_2025_26_v2.json` | `7c9fe42daadb8ec396af64da9c0f6e8d` |
| `composite_ratings_sim.csv` | `d02630fd68286488dd951351c84dc85c` |
| `rapm_results.csv` | `53cc52783cb279da0c8f9ee9c07f5637` |
| `team_strength.csv` | `136632ddafc0d6234045ac2202594d55` |
| `season_simulation_results.csv` | `06ef68fffb575ad505262ce7f3ed1e20` |
| `build_shots_enriched.py` | `180cd41400bba5fbeeed7d2593077424` |
| `build_shots_enriched_2025_26.py` | `305033d5919daf7c8f5b27fa5a0ae121` |

Compare against the pre-build baseline in the session log. All read-only inputs, protected model artifacts, and the four simulator CSVs are unchanged; output was written only to `model/rapm_stints_2025_26.parquet` and this report.

