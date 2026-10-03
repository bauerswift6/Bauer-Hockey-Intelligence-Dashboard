# xG v2 — Multi-Season Training Test

**Outcome: a winning configuration was found and it beats v1 significantly on the holdout —
but it FAILED the Step-4 calibration gate (out-of-fold total xG +3.39% vs actual, > 2%). Per the
instructions the run STOPPED before writing any v2 outputs and applied no recalibration. Awaiting
your decision.**

- Winner: **C-unweighted-as-is** (2025-26 + 2024-25 + 2023-24, unweighted, 12 shot types).
- Holdout log loss **0.21934** (v2, re-tuned) vs **0.22061** (v1 on same rows) — v2 better,
  bootstrap-significant.
- **2025-26 out-of-fold total xG = 7,834.0 vs 7,577 actual = +3.39%** → exceeds the 2% gate → **STOP.**
- **No files written:** `xg_model_2025_26_v2.json` and `shots_xg_2025_26_v2.parquet` were NOT
  created. All protected files md5-unchanged.

Everything is in the deterministic script `model/train_xg_multiseason.py` (random_state=42).
Fixed holdout = the same last 263 games of 2025-26 by date (22,089 rows / 1,537 goals), identical
to v1.

---

## Step 1 — 14-configuration grid (v1 hyperparameters, early stopping)

All configs use v1's params (depth 4, lr 0.05, min_child_weight 5, subsample 0.8, colsample 1.0)
with a game-grouped 85/15 validation split for early stopping. Ranked by holdout log loss:

| Rank | Config | Seasons | Weight | Shot types | LogLoss | AUC | Brier | PredG (act 1537) | Trees | Pool rows |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **C-unweighted-asis** | +24-25,+23-24 | unwtd | as-is | **0.21952** | 0.7718 | 0.05966 | 1562 | 336 | 314,290 |
| 2 | C-recency-asis | +24-25,+23-24 | recency | as-is | 0.21955 | 0.7717 | 0.05969 | 1551 | 374 | 314,290 |
| 2 | D-recency-asis | all 4 | recency | as-is | 0.21955 | 0.7716 | 0.05967 | 1556 | 464 | 427,435 |
| 4 | B-recency-asis | +24-25 | recency | as-is | 0.21970 | 0.7706 | 0.05970 | 1529 | 340 | 200,275 |
| 5 | B-unweighted-asis | +24-25 | unwtd | as-is | 0.21974 | 0.7706 | 0.05969 | 1535 | 340 | 200,275 |
| 5 | D-unweighted-asis | all 4 | unwtd | as-is | 0.21974 | 0.7706 | 0.05969 | 1562 | 628 | 427,435 |
| 7 | C-recency-merged | +24-25,+23-24 | recency | merged | 0.21984 | 0.7704 | 0.05973 | 1527 | 373 | 314,290 |
| 8 | C-unweighted-merged | +24-25,+23-24 | unwtd | merged | 0.21993 | 0.7703 | 0.05973 | 1531 | 375 | 314,290 |
| 9 | D-recency-merged | all 4 | recency | merged | 0.22002 | 0.7695 | 0.05973 | 1528 | 464 | 427,435 |
| 10 | B-unweighted-merged | +24-25 | unwtd | merged | 0.22009 | 0.7690 | 0.05973 | 1520 | 371 | 200,275 |
| 10 | B-recency-merged | +24-25 | recency | merged | 0.22009 | 0.7690 | 0.05974 | 1516 | 340 | 200,275 |
| 12 | D-unweighted-merged | all 4 | unwtd | merged | 0.22054 | 0.7678 | 0.05984 | 1524 | 307 | 427,435 |
| 13 | **A-unweighted-asis** | 2025-26 only | — | as-is | **0.22057** | 0.7683 | 0.05982 | 1498 | 165 | 88,989 |
| 14 | A-unweighted-merged | 2025-26 only | — | merged | 0.22101 | 0.7662 | 0.05989 | 1494 | 165 | 88,989 |

**Sanity check:** A-unweighted-as-is = **0.22057**, essentially reproducing v1's **0.22033**
(tiny gap from early-stopping picking 165 trees vs v1's fixed 292). ✓

**Reads:** (1) adding prior seasons helps — every B/C/D as-is config beats A. (2) The sweet spot is
**C (3 seasons)**; the 4th season (D) does not improve on C and needs many more trees. (3) **Merging
wrist+snap hurts** everywhere (each merged config is worse than its as-is twin). (4) Weighting is
a wash (recency ≈ unweighted, within 0.0003).

## Step 2 — Is the difference real? (game-level bootstrap, 1,000 resamples vs A-as-is)

95% CI of the per-shot log-loss difference (config − A); "better" = CI entirely below 0.

| Config | Δ log loss (mean) | 95% CI | Significantly better than A? |
|---|---|---|---|
| **C-unweighted-asis** | −0.00105 | **[−0.00177, −0.00030]** | **YES** |
| C-recency-asis | −0.00102 | [−0.00172, −0.00032] | YES |
| D-recency-asis | −0.00101 | [−0.00166, −0.00033] | YES |

All three top configs are significantly better than A (each CI excludes 0). **Selection:** lowest
holdout log loss among the significant set = **C-unweighted-as-is** (0.21952). It is also the
**simplest** under the tie-break (fewest seasons among {C,C,D}; unweighted over recency; as-is over
merged), so it wins outright on both criteria. The three are within each other's bootstrap noise,
which only reinforces the simplest choice.

## Step 3 — Re-tune the winner + holdout evaluation

Re-tuned C-unweighted-as-is with GroupKFold(5) by game on its 314,290-row pool (16-config grid over
depth {4,5,6,7} × lr {0.03,0.05} × min_child_weight {5,10,20} × subsample {0.8,1.0}). More data did
support a slightly deeper tree:

- **Chosen: max_depth 5, lr 0.03, min_child_weight 5, subsample 0.8, colsample 1.0, ~472 trees**
  (CV log loss 0.21271, marginally ahead of the depth-4 field at ~0.21278).

**Holdout head-to-head (same 22,089 rows; old column is still leaked — it trained on these games):**

| Metric | No-skill | **v2 (winner, re-tuned)** | v1 | old `xg` (leaked) |
|---|---|---|---|---|
| Log loss | 0.25261 | **0.21934** | 0.22061 | 0.21837 |
| AUC | — | **0.7721** | 0.7682 | 0.7755 |
| Brier | — | **0.05961** | 0.05985 | 0.05960 |
| Predicted goals (actual 1,537) | — | **1,567.9 (+2.0%)** | 1,511.0 (−1.7%) | 1,557.3 |

**v2 clearly beats v1** on log loss, AUC, and Brier. The leaked old column still edges v2 on log
loss/AUC — expected, since it trained on these holdout games (noted per instruction). Note v2's
holdout predicted goals already run **+2.0% high**, foreshadowing the Step-4 gate.

**v2 calibration by decile (well-behaved):**

| decile | 0 | 2 | 4 | 6 | 8 | 9 |
|---|---|---|---|---|---|---|
| pred | 0.0048 | 0.0192 | 0.0405 | 0.0761 | 0.1426 | 0.2269 |
| actual | 0.0036 | 0.0136 | 0.0398 | 0.0797 | 0.1426 | 0.2245 |

**By strength (n / actual / v2 / v1 / old):** 5v5 17,703 / 1,120 / **1,128** / 1,102 / 1,169 ·
PP 2,927 / 309 / **306** / 286 / 267 · PK 501 / 39 / **39** / 36 / 38 ·
extra-attacker 493 / 31 / **41** / 40 / 38 · 3v3-4v4 465 / 38 / **53** / 46 / 46.

**By shot type (n / actual / v2 / v1 / old):** wrist 9,197 / 603 / **572** / 575 / 568 ·
snap 5,743 / 497 / **497** / 473 / 478 · slap 2,720 / 101 / **136** / 128 / 129 ·
tip-in 2,240 / 143 / **173** / 148 / 191 · backhand 1,540 / 131 / **131** / 127 / 132.

**Where v2's over-prediction concentrates:** **slap (+35%: 136 vs 101 actual)** and
**tip-in (+21%: 173 vs 143)**, plus **3v3/4v4 (+39%)** and **extra-attacker (+32%)**. snap and
wrist are well-calibrated. This is the mechanism behind the total bias — see Step 4.

## Step 4 — Out-of-fold scoring of 2025-26 (calibration gate → STOP)

5-fold GroupKFold by game over 2025-26; each fold trained on (its other 2025-26 games + 2024-25 +
2023-24, unweighted, the re-tuned params).

- **Total OOF v2 xG = 7,834.0 vs 7,577 actual goals = +3.39%.**
- OOF log loss 0.21756, AUC 0.7678.
- **+3.39% exceeds the ±2% gate → STOPPED.** No recalibration applied (per instruction). The final
  full-pool model and the OOF parquet were **not written**.

**Diagnosis:** the winner discriminates better than v1 (lower log loss, higher AUC, bootstrap-
significant) but is **positively biased in aggregate**. The bias traces to shot types whose quality/
coding shifted across seasons — the prior seasons inflate 2025-26 predictions for **slap** and
**tip-in** shots specifically (Step-3 table), and the depth-5 re-tune amplifies it. v1 (depth 4,
2025-26 only) *under*-predicted by ~1.7%; v2 swings to *over*-predict by 2–3.4%.

## Step 5 — Player check

**Not reached** — the Step-4 gate stopped the run before player scoring (v2 OOF was not persisted).

---

## Decision needed from you

The winner is genuinely better at *ranking* shots (significant holdout log-loss/AUC gain) but is
+3.39% biased in *total*. Options, none applied without your say-so:

1. **Recalibrate** the winner (e.g., a single global scale ≈ 7577/7834 = 0.967, or isotonic/Platt on
   the OOF) and re-check — cheapest fix, keeps the discrimination gain.
2. **Use the winner config but v1's depth-4 params** (no Step-3 re-tune) — its Step-1 holdout
   predicted goals were 1,562 (+1.6%), i.e. likely inside or near the 2% gate; the re-tune to
   depth 5 is what widened the bias. Would need an OOF re-check.
3. **Drop slap/tip-in cross-season leakage** — e.g. a season indicator feature, or investigating the
   shot-type coding drift (wrist 53%→43%, snap 15%→25% across 2022-23→2025-26) that the prior
   seasons carry.
4. **Keep v1** — the gain is real but small (~0.0007 holdout log loss) and may not be worth the
   calibration risk.

## md5 integrity — all protected files unchanged
Re-hashed at the end and `diff`'d against the pre-start baseline — **identical**:
`xg_model_2025_26.json`, `shots_xg_2025_26.parquet`, all four `xg_features_*.parquet`, all four
`shots_enriched_*.parquet`, `player_handedness.csv`, and the four simulator CSVs
(`composite_ratings_sim.csv`, `rapm_results.csv`, `team_strength.csv`, `season_simulation_results.csv`).
No `_v2` output files exist. ✓

## Choices I made that you did not specify
1. **Early stopping in Step 1** optimizes *unweighted* validation log loss (validation eval-set gets
   no sample weights), so weighted and unweighted configs are compared on the same unweighted metric
   that the holdout uses.
2. **Merged shot types = snap→wrist** (id 2 folded into id 1), kept as a categorical.
3. **v1 alignment is positional**, not a key-merge: `(game_id, abs_secs, shooter_id)` is non-unique
   (the same-second rapid-fire pairs share it), so a merge fans out; v1's `shots_xg` is row-aligned
   to `xg_features_2025_26` (verified: keys + xg_old match positionally).
4. **Step-3 re-tune grid** = 16 deterministically-sampled configs from a depth {4–7} × lr {0.03,0.05}
   × mcw {5,10,20} × subsample {0.8,1.0} space (colsample fixed at 1.0, which v1 tuning already
   favored).
5. **"Significantly better"** operationalized as the 95% bootstrap CI of (config − A) lying entirely
   below 0.
6. Bootstrap resamples **holdout games** (not shots), 1,000 draws, seed 42.
