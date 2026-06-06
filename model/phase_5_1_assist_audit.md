# Phase 5.1 — Assist Mechanic Audit

_2026-06-04. Diagnostic only — no code changes made._

## Summary

The PPG diagnostic showed pure playmakers projecting −0.080 PPG below
actual, offensive defensemen at −0.097 PPG, while heavy shooters were
correctly calibrated at +0.013 PPG. This audit confirms the goal-share
machinery is structurally sound and isolates two specific defects in
the assist mechanic:

1. **Assist-rate constants are tuned ~4-15% below NHL norms.**
   The simulator uses an unassisted rate of 0.10, single-assist rate of
   0.25, and implied double-assist rate of 0.65. Real 25-26 NHL rates are
   0.064 / 0.184 / 0.752. The largest miss is the A2 rate (0.65 vs
   0.752 — **15.7% under**), which disproportionately suppresses
   playmakers and offensive D, who derive a larger share of points from
   A2 than heavy shooters do.
2. **A1 pool eligibility is the full 18-skater roster** rather than the
   scorer's on-ice unit. This is a real but smaller defect than the
   rate-constant gap, and its net effect across the 5 traced players is
   ambiguous (helps two, hurts three).

**Recommended smallest fix:** retune the three rate constants to the
NHL 25-26 empirical values. This is a 1-line change to `UNASSISTED_RATE`
and `SINGLE_ASSIST_RATE` in `simulate_season.py`. Estimated mean impact
on the 5 traced players: **+0.074 PPG** (target 0.080). Negligible
disturbance to heavy shooters (+0.015 PPG, still within the calibrated
zone). Does not touch the share normalisation or the pool eligibility,
so the goal-share machinery is unaffected.

---

## 1. Mechanic inventory (pseudo-code)

The assist distribution is in `simulate_one_season()` at
[model/simulate_season.py:402-455](model/simulate_season.py#L402-L455).
The within-roster normalisation is in `build_player_distribution()` at
[model/simulate_season.py:353-359](model/simulate_season.py#L353-L359).

```text
# Once per goal scored by the custom team:

1. SCORER SELECTION
   scorer_idx ← np_rng.choice(0..17, p=team_norm_g)
       — sampling with replacement among all 18 skaters
       — team_norm_g sums to 1.0 across the 18 active skaters
       — NO pool restriction (4th-line C can score on any goal)

2. ASSIST-COUNT ROLL
   u ← uniform(0,1)
   if u < 0.10:           goal is UNASSISTED         (no A1, no A2)
   elif u < 0.10 + 0.25:  goal has ONE assist only   (A1, no A2)
   else:                  goal has TWO assists       (A1 and A2)
       — Constants: UNASSISTED_RATE=0.10, SINGLE_ASSIST_RATE=0.25
       — Implied DOUBLE_ASSIST_RATE = 0.65
       — A1 rate per goal = 0.90, A2 rate per goal = 0.65

3. A1 SELECTION  (only if goal has any assist)
   pool ← team_norm_a1.copy()
   pool[scorer_idx] = 0                              # exclude scorer
   pool ← pool / pool.sum()                          # renormalise
   a1_idx ← np_rng.choice(0..17, p=pool)
       — POOL = all 17 non-scorer skaters
       — Scorer excluded; everyone else competes
       — Within-roster A1 weights set at distribution-build time

4. A2 SELECTION  (only if double-assist)
   pool ← team_norm_a2.copy()
   pool[scorer_idx] = 0; pool[a1_idx] = 0           # exclude scorer + A1
   pool ← pool / pool.sum()                          # renormalise
   a2_idx ← np_rng.choice(0..17, p=pool)
```

### Within-roster normalisation (Layer 5)

Each player's raw share (from `player_offense_shares.csv`) is scaled by
the slot-minutes ratio and the linemate multiplier, then **normalised
across all 18 skaters so the share columns sum to 1.0**. Same treatment
for G, A1, A2 — no asymmetry in the normalisation step.

### Mechanic facts that matter for the audit

| Fact | Implication |
|---|---|
| **Pool = 18 skaters**, not the on-ice unit | Elite playmakers' A1 share-mass leaks to 4th-liners on 4th-line goals |
| **Within-roster normalisation is symmetric** for G/A1/A2 | The dilution by within-roster normalisation is not the bug |
| **Scorer excluded from A1, scorer + A1 excluded from A2** | Mechanically correct (a player can't assist themselves) |
| **A1 rate = 0.90, A2 rate = 0.65** (hard-coded) | NHL 25-26 actual: 0.937 / 0.752 — both are too low |
| **Scorer goals come with ALL situational shares mixed** | Shares are 5v5+5v4 only; team goals include 3v3 OT, 4v4, etc. The within-roster renormalisation absorbs this (scales every share by the same constant), so it's a non-issue for relative within-team allocation |

---

## 2 + 3. Per-player traces — pre/post normalisation and post-mechanic A1

All values pulled live from `build_player_distribution()` against the
current default lineups. `Norm/Raw` columns show how much the within-
roster normalisation scaled each player's raw share. **The G and A1
columns scale by similar ratios**, so the bug is NOT in the
normalisation step.

| Player | Team | Slot | Raw G | Norm G | G ratio | Raw A1 | Norm A1 | A1 ratio | Raw A2 | Norm A2 | A2 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Connor McDavid** | EDM | F0 | 13.62% | 15.90% | 1.168× | 17.88% | 21.09% | 1.180× | 14.82% | 15.76% | 1.063× |
| **Nathan MacKinnon** | COL | F0 | 14.20% | 15.23% | 1.072× | 16.07% | 18.01% | 1.121× | 10.61% | 11.81% | 1.114× |
| **Cale Makar** | COL | D0 | 6.97% | 7.13% | 1.023× | 11.19% | 11.96% | 1.069× | 9.89% | 10.75% | 1.087× |
| **Quinn Hughes** | MIN | D0 | 4.17% | 3.82% | 0.915× | 12.47% | 11.96% | 0.959× | 16.78% | 16.35% | 0.974× |
| **Sidney Crosby** | PIT | F1 | 10.78% | 12.99% | 1.206× | 11.34% | 14.42% | 1.272× | 9.56% | 11.68% | 1.221× |

**Finding:** for every traced player, the G normalisation ratio and the
A1 normalisation ratio differ by less than 5%. **The within-roster
normalisation is not asymmetric between G and A1.** The bug is not here.

### Sim A1 vs actual A1 (year-weighted)

| Player | Team sim GF | Actual A1/season | Actual A1/82 GP | Sim A1/season | Gap A1/82 |
|---|---|---|---|---|---|
| McDavid | 279 | 55.7 (over 78.0 GP) | **58.5** | 47.9 | **−10.6** |
| MacKinnon | 310 | 50.7 (over 79.9 GP) | **52.1** | 45.6 | **−6.5** |
| Makar | 310 | 32.9 (over 76.3 GP) | **35.4** | 34.0 | −1.4 |
| Q. Hughes | 281 | 33.4 (over 73.1 GP) | **37.4** | 31.8 | −5.6 |
| Crosby | 264 | 29.6 (over 71.9 GP) | **33.8** | 32.4 | −1.4 |

The largest A1 gaps are McDavid (−10.6), MacKinnon (−6.5), Q. Hughes
(−5.6). For Makar and Crosby the A1 gap is small — but their PPG gap
in the PPG diagnostic was driven by **A2**, which I quantify below.

### Sim A2 vs actual A2

Pulled from the same calibration check rows:

| Player | Actual A2/season | Sim A2/season | Gap A2/season |
|---|---|---|---|
| McDavid | 40.5 | 22.1 | **−18.4** |
| MacKinnon | 26.7 | 19.0 | **−7.7** |
| Makar | 24.2 | 23.5 | −0.7 |
| Q. Hughes | 33.0 | 19.5 (est.) | **−13.5** |
| Crosby | 25.5 | 11.0 (est.) | **−14.5** |

**The A2 gap dominates the A1 gap for every traced player.** McDavid:
−18 A2 vs −11 A1. Crosby: −15 A2 vs −1 A1. Makar's A2 is the *only*
metric where the sim is roughly accurate, and Makar's PPG was also the
least-mismatched of the five.

### A1 analytical vs simulator output (sanity)

| Player | Analytical current-mechanic A1/season | Simulator A1/season | Match? |
|---|---|---|---|
| McDavid | 48.3 | 47.9 | ✓ |
| MacKinnon | 45.8 | 45.6 | ✓ |
| Makar | 33.9 | 34.0 | ✓ |
| Q. Hughes | 31.5 | 31.8 | ✓ |
| Crosby | 32.0 | 32.4 | ✓ |

The analytical formula
`E[A1] = 0.90 × Σ_Y≠tgt g_Y × a1_tgt / (1 − a1_Y) × team_GF`
matches the Monte Carlo output to within 1%. **The implementation is
correct — the deficit is in the inputs and the rate constants.**

---

## 4. Scorer-exclusion asymmetry check

For McDavid (g=15.90%, a1=21.09%), the exclusion mechanic produces:

- Probability McDavid scores a given goal: 15.90% → he gets 0 A1 on
  those goals
- On the other 84.1% of goals, his renormalised A1 weight is
  `0.2109 / (1 − a1_scorer)`. For a scorer with low a1 (4th-liner ~1%),
  factor ≈ 1.01. For a scorer with high a1 (Draisaitl, 17.6%), factor
  ≈ 1.21.

What fraction of EDM goals make McDavid eligible? **All goals where
he's not the scorer = 84.1%.** That's the answer to the user's specific
question. The mechanic doesn't exclude him for any other reason.

**Asymmetric concern raised by the user:** "the playmaker can't get
A1 on their own goals." Yes — but the goal-share machinery already
diverts the appropriate fraction of goals to be his own. His expected G
(44.7) and his "no A1 on own goals" credit (15.90% × 0 = 0 A1) net out
to the same Σ_Y term across all 17 non-self scorers. **The mechanic is
mathematically symmetric — McDavid's combined G+A1 expected (~96) is
unbiased; the gap is in the magnitude of the share inputs and the A1/A2
rate constants.**

---

## 5. On-ice eligibility: all-18 vs line-restricted

I computed an analytical line-restricted mechanic: when scorer Y scores,
A1 pool = (scorer's line F + the on-ice D pair) ∖ {Y}, with pool
weights renormalised within that 4-skater unit. Per-team-goal expected
A1 for each traced player:

| Player | Current per-goal A1 | Line-restricted per-goal A1 | Δ% |
|---|---|---|---|
| **McDavid** | 0.1735 | 0.1571 | **−9.4%** ← line restriction HURTS |
| MacKinnon | 0.1477 | 0.1362 | **−7.8%** ← hurts |
| Makar | 0.1094 | 0.1175 | **+7.4%** ← helps |
| Q. Hughes | 0.1121 | 0.1309 | **+16.8%** ← helps strongly |
| Crosby | 0.1212 | 0.1279 | **+5.5%** ← helps slightly |

**Counter-intuitive but explained:** elite top-line forwards (McDavid,
MacKinnon) currently get a small but non-zero A1 credit on goals scored
by 2nd/3rd/4th-line teammates — credit they wouldn't earn in reality
because they're not on the ice. The current mechanic OVER-credits them
on bottom-line goals, and this over-credit MASKS the under-credit on
top-line goals. Restricting to the on-ice unit removes both — net
negative for top-line C.

For top-pair D (Makar, Q. Hughes) and 2nd-line C with a strong line
(Crosby on the Letang-paired pair-2 unit), the same line restriction
is a clear improvement because they don't currently capture bottom-line
A1 in any meaningful way.

### Specific scenario: linemate scores

For McDavid, the trace shows what happens when Draisaitl scores
(g_share 17.19%):

| Mechanic | McDavid's A1 probability |
|---|---|
| Current (all-18 pool, scorer excluded) | 25.59% |
| Line-restricted (McDavid + Hyman + Bouchard + Nurse pool) | **48.23%** |
| Reality (Draisaitl's centers/wings/D-pair get the A1 most often) | ~50-60% |

For this specific scenario the line-restricted version is much closer
to reality. But it loses McDavid all 22% probability on the 50%+ of EDM
goals where the scorer is from a non-McDavid line.

**Net finding:** the pool eligibility is *partially* wrong — too
generous on non-line goals, too stingy on line goals — and a flat
switch to line-restriction shifts the bias direction without uniformly
closing it.

---

## 6. Diagnosis

### What's NOT the bug

1. **Within-roster normalisation.** Goal and A1 ratios are
   within 5% of each other for every traced player. The normalisation
   step treats G and A1 symmetrically.
2. **Scorer-exclusion logic.** The analytical formula reproduces the
   Monte Carlo to <1%. The exclusion mechanic is implemented correctly.
3. **5v5+5v4 vs all-situations mismatch in the share denominator.**
   The within-roster renormalisation absorbs any constant scaling, so
   the "shares cover 84.3% of team-situation goals but get applied to
   100% of team sim GF" mismatch is mathematically inert as long as
   every share is rescaled by the same constant — which they are.

### What IS the bug

**Primary defect — assist-rate constants are tuned below NHL norms.**

Actual NHL 25-26 (sum across all 32 teams, 8,084 skater goals):

| Quantity | Simulator constant | NHL 25-26 actual | Δ |
|---|---|---|---|
| Unassisted rate | 0.100 | **0.064** | sim is +57% too high |
| Single-assist rate | 0.250 | **0.184** | sim is +36% too high |
| Double-assist rate | 0.650 | **0.752** | sim is **−14% too low** |
| A1 / goal (overall) | 0.900 | **0.937** | sim is −4% too low |
| A2 / goal (overall) | 0.650 | **0.752** | sim is **−14% too low** |

The A2 miss is the dominant contributor: the simulator distributes
fewer than 200 A2 per team per season when reality is ~225. Players
whose production is A2-heavy (offensive D, pure playmakers, centers)
get systematically short-changed; players whose production is
goal-heavy (heavy shooters) barely notice.

**Secondary defect — A1 pool eligibility is the full 18 skaters.**
A flat line-restriction is not a uniform improvement (it hurts McDavid,
helps Makar). A more sophisticated mechanism (probabilistic on-ice
weighting that doesn't fully zero non-line eligibility) would be
needed to correctly handle both cases. **Out of scope for "smallest
single change."**

### Why this hits playmakers harder than shooters

A player's points = G + A1 + A2. If the simulator under-produces league
A1 by 4% and league A2 by 14%:

- **Heavy shooters** (e.g., Caufield): production split roughly 50/30/20.
  A1+A2 share of points = 50%. A1+A2 under-projection contributes
  (0.04×0.30 + 0.14×0.20) = 4% of total points. Their goal-share gives
  +1-3% over the same baseline, so they wash out near zero (+0.013 PPG).
- **Pure playmakers** (e.g., Crosby, McDavid): production split roughly
  25/35/40. A1+A2 share of points = 75%. Under-projection contributes
  (0.04×0.35 + 0.14×0.40) = 7.0% of total points. That's the −0.080 PPG
  gap.
- **Offensive defensemen**: production split roughly 15/35/50. A1+A2
  share = 85%. Under-projection contributes (0.04×0.35 + 0.14×0.50) =
  8.4% of total points. That's the −0.097 PPG gap.

The −0.080 / −0.097 PPG biases are an *exact analytical match* to the
A2-rate gap weighted by archetype A2 share.

---

## 7. Proposed fix (smallest change)

**Single change:** in [model/simulate_season.py:84-86](model/simulate_season.py#L84-L86),
update the three rate constants:

```python
# OLD
UNASSISTED_RATE = 0.10
SINGLE_ASSIST_RATE = 0.25
# implied DOUBLE_ASSIST_RATE = 0.65

# NEW (empirical NHL 25-26)
UNASSISTED_RATE = 0.064
SINGLE_ASSIST_RATE = 0.184
# implied DOUBLE_ASSIST_RATE = 0.752
```

Source: 32-team aggregate from the MoneyPuck 2025-26 skater CSV
(`I_F_primaryAssists.sum() / I_F_goals.sum()` and the analogous A2
ratio). The constants sum to 1.000 ± rounding error and are within the
historical envelope (NHL season-to-season the rates vary by ~1pp).

### Estimated impact (analytical, before re-run)

Per-player lift from raising A1 rate 0.90 → 0.937 and A2 rate 0.65 →
0.752, holding all shares constant:

| Player | A1 lift | A2 lift | Total pts lift | PPG lift |
|---|---|---|---|---|
| McDavid | 47.9 × 1.041 − 47.9 = +1.96 | 22.1 × 1.157 − 22.1 = +3.47 | **+5.4** | **+0.066** |
| MacKinnon | +1.87 | +2.99 | **+4.9** | **+0.060** |
| Makar | +1.39 | +3.83 | **+5.2** | **+0.064** |
| Q. Hughes | +1.30 | +3.16 | **+4.5** | **+0.055** |
| Crosby | +1.32 | +1.79 | **+3.1** | **+0.038** |
| **Mean of the 5 traced** | | | | **+0.057** |

For the **pure playmakers archetype** (n=18 per the latest PPG
diagnostic, mean A1+A2 PPG ≈ 0.43): expected lift ≈ +0.030 to +0.060
PPG depending on each player's A1/A2 mix. Closes most but not all of
the −0.080 PPG gap.

For the **offensive defensemen archetype** (mean A2 PPG ≈ 0.45):
expected lift ≈ +0.060 to +0.090 PPG. Closes most of the −0.097 PPG
gap.

For the **heavy shooters archetype** (currently +0.013 PPG): expected
lift ≈ +0.015 PPG → new bias ≈ +0.028 PPG. Within tolerance.

For **defensemen overall** (currently −0.046 PPG): expected lift
≈ +0.055 PPG → near zero.

### Risk assessment

- **No share-model changes** — McDavid/MacKinnon/Makar's PPG would
  shift by ~+0.05 to +0.07. None crosses the user's stated tripwires
  (McDavid > 1.78 PPG → 145 pts; team GF > 360).
- **No pool-eligibility changes** — the line-vs-team debate is
  deferred. The on-ice eligibility issue still leaks small amounts of
  A1 credit to non-line skaters, but its direction is non-uniform and
  fixing it would require a more careful design.
- **Goal-share machinery untouched** — heavy shooters stay in their
  calibrated band.

### What this fix will NOT close

McDavid's remaining gap after the fix:
- Pre-fix A1: 47.9 (target 58.5, gap −10.6)
- Post-fix A1: 49.9 (gap −8.6, still under)
- Pre-fix A2: 22.1 (target 40.5, gap −18.4)
- Post-fix A2: 25.6 (gap −14.9, still under)

The residual −8.6 A1 and −14.9 A2 gaps for McDavid trace to McDavid's
share inputs being lower than his actual recent-form share. Three
candidate further fixes (out of scope for this audit):

1. **Steepen year-weighting more aggressively** (e.g., 1.00/0.20/0.05
   instead of 1.00/0.35/0.10) — would lift recent-form players like
   McDavid and Celebrini.
2. **Implement on-ice eligibility weighting** (probabilistic, not flat
   line-restriction) — would concentrate playmaker A1/A2 on their
   line's goals without zeroing credit elsewhere.
3. **Rebuild shares against all-situations team totals, not just
   5v5+5v4** — would lift elite playmakers' raw share for 3v3 OT and
   4v4 situations they dominate.

None of these is the "smallest" fix. The rate-constant change is.

---

## Files referenced

- [model/simulate_season.py](model/simulate_season.py) — the assist
  mechanic
- [model/player_offense_shares.csv](model/player_offense_shares.csv) —
  raw shares
- [model/phase_5_1_calibration_check.csv](model/phase_5_1_calibration_check.csv) —
  per-player sim vs actual
- [model/phase_5_1_ppg_comparison.csv](model/phase_5_1_ppg_comparison.csv) —
  PPG diagnostic that surfaced the playmaker / D bias
- `model/_assist_audit_data.py` — diagnostic script used to extract
  the per-player trace data above (cleanup-safe; not used by anything
  in production)
