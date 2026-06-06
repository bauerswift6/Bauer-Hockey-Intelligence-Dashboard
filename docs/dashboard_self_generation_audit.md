# Dashboard self-generation audit — which metrics are mine vs third-party

_Audit only — no code changes made. Generated 2026-06-06._

The goal of this audit is to identify every analytical metric on the dashboard
and trace it to its source: either a model file I produce in `model/` or an
external derived value pulled from MoneyPuck / Evolving Hockey / Natural Stat
Trick. Raw facts (goals, shots, GP, ice time, hits, blocks) pulled from the
NHL API or MoneyPuck CSVs are acceptable; derived analytical values are not.

The headline finding: **the Phase 2.5 composite-rating, RAPM, xG, season
simulator, team-strength, and contract-classifier models are all my own.
However, the "GAR" and "xGAR" values displayed on the Players Leaderboards,
Player Search profile, Compare, and Contract Value tabs are all scaled
MoneyPuck `gameScore` and `I_F_flurryScoreVenueAdjustedxGoals` values
relabeled — not derived from my RAPM or composite_war pipeline.** The same
holds for the Goalies page GSAX leaderboard, which uses MoneyPuck's `xGoals`
not my own xG model. Several "Powered by Evolving Hockey" attributions and
EH branding remain on the page even though every EH and NST scrape attempt
returns `None` (their pages require login / block bots), so the live data
always comes from MoneyPuck behind the EH label.

---

## Section 1 — Statistics I generate myself

These are computed entirely by my code from raw inputs. Each is backed by a
file in `model/` that I produce, and is served by an endpoint that reads my
file rather than calling a third-party derived value.

| Metric (as shown to user) | Endpoint | Backing file | Calculation |
|---|---|---|---|
| **Composite Rating** | `/api/composite-ratings` | `model/composite_ratings.csv` (`build_composite.py`) | Weighted z-score blend of 6 components — RAPM, individual impact, relative xGF, playmaking, PP, PK — each derived from raw shifts/shots/goals |
| RAPM component | `/api/composite-ratings`, `/api/rapm-leaders` | `model/rapm_results.csv` (`train_rapm.py`) | Ridge regression of 5v5 on-ice xG differential against player one-hot encodings over 6 seasons |
| Individual component | `/api/composite-ratings` | `model/composite_ratings.csv` | Individual xG/60 z-scored within position pool |
| Relative xGF component | `/api/composite-ratings` | `model/composite_ratings.csv` | Player's 5v5 xGF% minus their team's xGF% without them |
| Playmaking component | `/api/composite-ratings` | `model/composite_ratings.csv` | A1/60 z-scored within position pool |
| PP / PK components | `/api/composite-ratings` | `model/composite_ratings.csv` | Per-60 PP/PK xG impact split |
| **Bayesian RAPM** (experimental) | `/api/rapm-bayesian-leaders` | `model/rapm_bayesian_results.csv` (`train_rapm_bayesian.py`) | Prior-informed ridge using individual xG/60 + on-ice xGA/60 as informative priors |
| **xG model** (per-shot) | `/api/xg-model-stats` (and consumed internally by training) | `model/xg_model.pkl` + `model/shots_augmented.parquet` (`train_xg.py`) | Gradient-boosted classifier with isotonic calibration on 6-season shot events; AUC + log-loss reported via the model-stats endpoint |
| Composite WAR (sim variant) | `/api/sim-team-roster/<team>`, `/api/sim-player-pool`, simulator lineup editor | `model/composite_ratings_sim.csv` (`build_composite_sim.py`) | Rescaled version of composite_rating in WAR units (centered ~0, ~6 goals/WAR), excludes prospects without enough sample |
| Expected GP share | `/api/sim-team-roster/<team>`, `/api/sim-player-pool` | `model/composite_ratings_sim.csv` | Year-weighted GP / 82 from MoneyPuck game-logs (my computation, not an external rating) |
| Average TOI / game | `/api/sim-team-roster/<team>` | `model/player_offense_shares.csv` (`build_player_offense_shares.py`) | Year-weighted icetime / GP from raw MoneyPuck rows |
| Player goal/A1/A2 share | (simulator inputs) | `model/player_offense_shares.csv` | Year-weighted player goals / team goals (5v5+5v4), with the 25-26 Phase 5.1 dilution fix |
| **Team strength** | `/api/team-strength`, `/api/team-strength-custom`, simulator | `model/team_strength.csv` (`build_team_strength.py`) | Weighted z-score blend of top6_F / bottom6_F / top-pair D / bottom-pair D / goalie components; goalie pre-cap ±1.5 WAR; final magnitude compression × 0.85 |
| Top6/Bot6 forward strength | `/api/team-strength` | `model/team_strength.csv` | Ice-time-weighted average of composite_war × expected_gp_share for the TOI-sorted top 6 / bottom 6 forwards |
| Top-pair / Bottom-pair D strength | `/api/team-strength` | `model/team_strength.csv` | Same approach, top 4 / bottom 2 defensemen by TOI |
| Goalie strength (team-level) | `/api/team-strength` | `model/team_strength.csv` | (xGoals − goals)/6 blended over the top 2 goalies by 25-26 GP, capped at ±1.5 raw WAR units |
| **Per-game scoring engine** | `/api/simulate-game`, `/api/simulate-season-custom`, `/api/sim-team-roster/<team>` | `model/simulate.py` (NegBin(mean, 1.3×mean) per side) | Mean = LEAGUE_GPG + ALPHA(0.525) × (TS_h − TS_a) + HOME_ICE(0.15); variance = 1.3× mean |
| Series outcome simulator | (consumed by season backtest, playoffs-only sim) | `model/simulate.py` `simulate_series()` | Best-of-7 with home/away rotation; per-game uses NegBin scoring engine |
| **Season Cup probability backtest** | `/api/season-projections` (`backtest` block) | `model/season_simulation_results.csv` (`monte_carlo_full_season` at 10k sims) | 10,000 Monte Carlo full-season sims with the locked team_strength and assist constants; per-team playoff%/division%/round2/conf-final/final/cup |
| Live playoff-now Monte Carlo | `/api/season-projections` (`playoffs_now` block), refreshable via `?force=true` | `model/playoff_projection_now.csv` + on-demand `_rerun_playoffs_now_mc()` | Fold live R3 carousel state, lock decided rounds, MC remaining series at 10k sims |
| **Phase 5.1 per-player season projection** | `/api/simulate-season-custom` | `model/simulate_season.py` four-layer share model | Layer 1 historical share → Layer 2 slot-minutes ratio → Layer 3 linemate-quality multiplier → Layer 4 thin-sample blend → within-roster normalization → per-game distribution |
| Assist distribution | `/api/simulate-season-custom` | `model/simulate_season.py` constants `UNASSISTED_RATE=0.064`, `SINGLE_ASSIST_RATE=0.184`, implied `DOUBLE=0.752` | Empirical NHL 25-26 32-team aggregate (8,084 goals / 7,571 A1 / 6,080 A2) |
| Linemate quality multiplier | `/api/simulate-season-custom` | `model/simulate_season.py` `1 + LINEMATE_K(0.15) × (avg_linemate_war − baseline)` | Applied to goal share + A1 share only |
| **Sustainability Score** | Contract Value tab | `model/`-derived if we were using composite_war; **currently** computed in `app.py` as `GAR − xGAR` where both are MoneyPuck-scaled (flagged in section 3) | Conceptually mine — but the GAR/xGAR inputs are external (see section 3) |
| **Surplus Value** | Contract Value tab | Computed in `app.py` `_merge_contracts_with_moneypuck()` | `GAR × MARKET_VALUE_PER_GAR($1.85M) − cap_hit`. Logic is mine; GAR input is external (flagged) |
| Off / Def GAR split | Contract Value tab | `app.py` `_split_off_def_gar()` | Uses MoneyPuck ixG/60 to allocate skater "GAR" between offense and defense weights. Logic is mine; inputs are external |
| PP / PK GAR | Contract Value tab | `app.py` `_build_pp_pk_gar_index()` | Situation-specific MoneyPuck gameScore / scale. Logic is mine; inputs are external |
| Contract type (ELC / Bridge / Market Rate / Veteran) | Contract Value tab | `app.py` `_classify_contract()` | Rule-based classification from cap_hit + age I derive |
| Age curve projection (Year-3 GAR / Surplus) | Contract Value tab | `app.py` `_age_curve_projection()` | Position + age decay curves I define |
| Aggregate team Surplus / GAR-per-dollar | `/api/team-cap-efficiency` | `app.py` rollup over my contract values | Sum of player surplus by team |
| Lineup editor team_strength recompute | `/api/team-strength-custom` | `model/build_team_strength.py` `compute_team_strength()` with `roster_override` | Same TS pipeline applied to a user-modified roster |
| Default lineup assignment (TOI-sort) | `/api/sim-team-roster/<team>` | `model/build_team_strength.py` `_attach_war()` + sort by `avg_toi_per_game` | The Phase 5.1 TOI-based sort (Fix 1) |
| Custom-season per-player projection | `/api/simulate-season-custom` | `model/simulate_season.py` `monte_carlo_custom_season()` | The user's full pipeline including all Phase 5.1 fixes |
| "Similar players" (closest matches) | `/api/player/<id>/similar` | `app.py` Euclidean-distance search | Distance over `(gar_proxy, zone_start_pct)` — see flag in section 3 about the gar_proxy input |

---

## Section 2 — External dependencies that are acceptable (raw data only)

These sources provide facts, not derived ratings. Every analytical use of them
goes through my own pipeline.

| Source | Endpoints | What I pull | Why it's OK |
|---|---|---|---|
| **NHL Public API** (`api-web.nhle.com`, `search.d3.nhle.com`) | `/api/season-info`, `/api/games`, `/api/game-details/<id>`, `/api/leaders`, `/api/standings`, `/api/playoff-form`, `/api/playoff-bracket`, `/api/playoff-stats-leaders`, `/api/transactions` (partial), `/api/team-head-to-head`, `/api/player-search`, `/api/player/<id>`, `/api/player/<id>/shot-chart` (NHL play-by-play coords) | Standings (W/L/OTL/GF/GA/pts), boxscores, schedule, rosters, game logs, player bios + headshots, draft details, playoff brackets, transactions, raw shot coordinates | These are facts and identity data the NHL itself reports. The dashboard does not derive any analytic from them beyond simple counts/totals |
| **MoneyPuck CSV — raw counts** | `/api/players-full`, `/api/goalies-full`, `/api/leaders`, contract pipeline, `/api/skater-onice-leaders`, etc. | Goals, primary/secondary assists, points, shots, ice time, games played, penalty minutes, zone starts, rush/rebound counts, on-ice CF/FF/HD shot attempts | These are derived from raw NHL play-by-play and are factual. We can re-derive them ourselves from `shots_dataset.parquet`, but the MP CSV is convenient. Considered raw |
| **MoneyPuck `corsiPercentage`, `xGoalsPercentage` (5v5 team)** | `/api/team-analytics`, `/api/playoff-team-analytics`, `/api/team-special-teams`, `/api/team-style` | CF%, xGF%, HDCF% at the team level | CF% is a ratio of raw counts (mine to compute or theirs — same number). **xGF% uses MoneyPuck's xG model rather than mine** — see flag in section 3. CF% is fine; xGF% is borderline |
| **MoneyPuck PDO components** (5v5) | `/api/team-style` | Shooting %, save %, PDO | SH% and SV% are ratios of raw goals/shots and goals/saves; PDO = SH% + SV%. Logic is mine, inputs are raw |
| **MoneyPuck shot danger zones** | Several | `lowDangerShots`, `mediumDangerShots`, `highDangerShots` | MoneyPuck's danger classification is a model — but a zone-based one (location-derived). Counts of shots in each zone are facts about where shots came from. Their HDSV%/MDSV% breakdown is acceptable as a structural raw input |
| **PuckPedia contracts (Tier 1 attempt)** | `/api/contract-values` | Player cap hit + expiry year if scrape succeeds | Currently always fails (Cloudflare); falls back to `data/contracts_manual.json` |
| **Manual contracts JSON** (`data/contracts_manual.json`) | `/api/contract-values` | Curated set of ~70 cap-hit + expiry rows | My own curated dataset of factual contract terms |
| **Sportsnet RSS** | `/api/news`, `/api/transactions` (RSS branch) | News article titles, links, pub dates | Just news headlines. Filtered to last 14 days and through a section-landing denylist |

---

## Section 3 — External dependencies that need to be addressed (derived analytics)

These are analytical metrics displayed to the user but pulled from third-party
models. **Each is misattributed or duplicates an existing model I already own.**

### 3.1 "GAR" (skater) — displayed on Players Overview, Players Leaderboards, Player Search profile, Compare, Contract Value tabs

**Source today:** `MoneyPuck gameScore / GAR_PROXY_SCALE (=10.0)` — see
[app.py:2965](app.py#L2965) and [app.py:2176](app.py#L2176).

`gameScore` is a single-number player rating MoneyPuck publishes; dividing by
a constant doesn't make it mine. The "GAR" label aligns the user with
Evolving Hockey terminology, but **the value is not derived from my
RAPM/composite pipeline.** Same goes for `gar_per_million` on the Contract
Value tab.

**Gap in the dashboard if removed:** large — appears in 6+ places, drives
Contract Value's Surplus Value, drives the "GAR: 36th of 727 skaters" pill
under player profile photos.

**Easiest self-generated replacement:**
- Server-side I already produce `composite_ratings_sim.csv::composite_war`
  in proper WAR units (~6 goals/WAR centered ~0)
- Frontend `gar` field could be sourced from `composite_war` for skaters
  rather than scaled gameScore. The composite uses RAPM + individual +
  relative + playmaking + special-teams components — exactly what GAR is
  supposed to represent
- Contract Value's Surplus Value formula would then use `composite_war ×
  MARKET_VALUE_PER_GAR`, which is methodologically cleaner

### 3.2 "xGAR" (skater) — displayed in the same places as GAR

**Source today:** `MoneyPuck I_F_flurryScoreVenueAdjustedxGoals /
XGAR_PROXY_SCALE (=3.0)` — see [app.py:2193](app.py#L2193) and
[app.py:2931](app.py#L2931).

This is MoneyPuck's flurry-and-score-and-venue-adjusted xG total. It uses
**MoneyPuck's xG model**, not mine. My xG model lives in `model/xg_model.pkl`
and is trained against the same raw shot data, but the dashboard does not
display anything computed from it (it's used internally as a feature for
my RAPM and composite, never surfaced directly).

**Gap if removed:** medium — xGAR is a secondary column on the leaderboard
and the "xgar_per_million" / Sustainability Score = GAR − xGAR inputs.

**Easiest self-generated replacement:**
- Replay the shot dataset through `xg_model.pkl` to produce a per-player
  total of my xG values (sum of predicted goal probabilities for each shot)
- Use my individual xG component (which is computed from `xg_model.pkl`)
  rescaled to the GAR scale as the "xGAR" replacement
- Sustainability Score becomes (my composite-derived GAR) − (my-xG-derived
  xGAR) — both internal

### 3.3 "GSAX" (goalie) — displayed on Goalies page, Goalie Detail, player goalie profiles, Contract Value (goalies)

**Source today:** `MoneyPuck xGoals − MoneyPuck goals` — see
[app.py:1659+](app.py#L1659) and [app.py:2103](app.py#L2103).

`xGoals` here is MoneyPuck's xG against the goalie. The subtraction is mine
(trivial); the underlying xG is theirs. Since I have my own xG model trained
on the same shot data, the displayed GSAX should be `(my_xG_against −
goals_against)` instead.

**Gap if removed:** large — GSAX is the headline goalie metric, surfaces in
the leaderboard, the SV%/workload/quality-starts table, the team_strength
goalie blend (already partially mine since I cap at ±1.5 WAR), and the
goalie rankings on player profiles.

**Easiest self-generated replacement:**
- Run all 25-26 shots through `xg_model.pkl` to produce per-goalie sum of
  predicted goal probabilities, then subtract actual goals
- Surface as `gsax` field; existing display logic unchanged
- For the team_strength goalie blend, optionally pass this in instead of
  the MP value — would close a loop since the goalie cap is mine but the
  underlying GSAX is currently MP-derived

### 3.4 "xGF%" / "xGoalsPercentage" (team-level + skater on-ice) — displayed on Teams, Players Leaderboards (on-ice xGF%), Playoff Team Analytics

**Source today:** MoneyPuck team-level `xGoalsPercentage` and skater-level
`onIce_xGoalsPercentage`. These are computed from MoneyPuck's xG model.

**Gap if removed:** medium — it's a primary possession-quality column on
multiple team and skater tables, and on the on-ice leaderboard.

**Easiest self-generated replacement:**
- Aggregate my xG model output (`xg_model.pkl` predictions on
  `shots_dataset.parquet`) per team per skater into "on-ice xG for" and
  "on-ice xG against" totals
- Compute xGF% = xGF / (xGF + xGA) using my model's numbers
- Same column name, value from my pipeline

### 3.5 "GAR" (goalie) on Contract Value tab

**Source today:** `MoneyPuck (xGoals − goals)` shown as goalie "GAR" — see
[app.py:2152](app.py#L2152).

Same external-xG problem as GSAX (3.3). The label "GAR" for a goalie is
actually Goals Saved Above Expected expressed in saved goals; the
methodology is correct but the xG input is external.

**Easiest self-generated replacement:** same fix as 3.3 — re-derive xG with
my model.

### 3.6 `gameScore` — displayed directly in some places + drives the GAR proxy

**Source today:** MoneyPuck-computed gameScore (per-game contribution score
that MP publishes per skater per season).

The Players-full endpoint exposes both `game_score` and `gar`. The
leaderboard surfaces `game_score` as the Game Score column.

**Easiest self-generated replacement:** Game Score has a known formula
(Domenic Galamini): `0.75·G + 0.7·A1 + 0.55·A2 + 0.075·SOG + 0.075·BLK +
0.05·PIM + 0.15·PD − 0.15·PT + 0.01·FOW − 0.01·FOL + 0.05·CF − 0.05·CA
+ 0.15·GF − 0.15·GA`. We can compute this ourselves from raw counts.
Without doing so, "Game Score" remains an MP-derived single-number rating.

### 3.7 `I_F_flurryScoreVenueAdjustedxGoals` (also called "Individual Adjusted xG")

**Source today:** Used as the `xGAR` numerator and as a column in the
Players-full row builder; flagged for sustainability-score input
([app.py:2931](app.py#L2931)).

This is MoneyPuck's adjusted xG — venue-adjusted and score-state-adjusted.
We don't expose those adjustments to the user as separate methodological
choices.

**Easiest self-generated replacement:** sum per-shot predictions from
`xg_model.pkl` per player; apply our own venue/score-state adjustments if
desired (probably not necessary at the dashboard level).

### 3.8 Branding / attribution issues

These don't return wrong values — they return misleading labels.

| Issue | Where | Fix |
|---|---|---|
| Footer says "Data: NHL Public API, Evolving Hockey, MoneyPuck" | [templates/index.html:1242](templates/index.html#L1242) | EH is never actually queried successfully — remove or relabel as "MoneyPuck (analytical CSVs)" |
| Players section displays "Advanced stats powered by Evolving Hockey" banner | [static/js/util.js:117](static/js/util.js#L117), [static/js/players_leaderboards.js:376](static/js/players_leaderboards.js#L376) | The banner is shown when EH scrape fails (always). The data underneath is MoneyPuck — banner should reflect that |
| GAR leaderboard label says "Source: Evolving Hockey" or shows "Source: MoneyPuck" depending on which fallback wins | [static/js/players_leaderboards.js:378](static/js/players_leaderboards.js#L378) | Always MoneyPuck in practice. Label is honest in fallback case but a leftover EH preference in the source ordering |
| Glossary still mentions "Evolving Hockey" + "Natural Stat Trick" as defining sources of advanced metrics | [static/js/glossary_data.js](static/js/glossary_data.js) | Definitions are fine; would need wording adjustments if we relabel anywhere |

### 3.9 Endpoints that always fall back to MoneyPuck despite labeling EH first

| Endpoint | Label sequence | Reality |
|---|---|---|
| `/api/gar-leaders` | EH → MP → NST | Always MP gameScore proxy |
| `/api/xgar-leaders` | EH → MP → NST | Always MP adjusted xG proxy |
| `/api/team-analytics` | EH → MP → NST | Always MP CF%/xGF%/HDCF% (xGF% is external xG-derived — flagged) |
| `/api/goalie-analytics` | EH → MP | Always MP-derived GSAX (xG-derived — flagged) |
| `/api/line-analytics` | EH → NST | Both fail; endpoint returns blocked-state message ("requires EH subscription") |

The "EH first" preference is a leftover from when the project hoped to use EH
data; **in practice every advanced-stats response comes from MoneyPuck.** The
fallback chain functions as a "try EH, never succeed, use MP" pattern. The
labels show MP correctly when MP wins, but the chain ordering preserves an EH
preference that no longer functions.

---

## Summary by major dashboard section

### Morning Brief
- **Mine:** Recent games carousel logic, league headlines filter (14-day +
  denylist), playoff bracket display
- **Acceptable raw:** NHL API standings/games/schedule, Sportsnet RSS
- **External derived:** none (the "Playoff Form" last-10 is W/L/OTL from
  raw scores)

### Players section
- **Mine:** Composite Rating column + 6 component columns, RAPM (total/off/def)
  column, RAPM panel on Player Lookup profile, similar-players logic,
  sustainability score formula (but inputs are MP), surplus value formula
  (but inputs are MP), age-curve / contract-type classifier, all per-player
  Phase 5.1 simulator projections
- **Acceptable raw:** goals, assists, points, shots, GP, TOI, plus-minus,
  zone starts, rush/rebound counts, on-ice CF/FF counts, hits, blocks
- **External derived (flagged):** GAR column, xGAR column, gameScore column,
  on-ice xGF%, GAR/$1M, xGAR/$1M

### Teams section
- **Mine:** Team Strength leaderboard with full 5-component breakdown,
  team-strength-custom endpoint for lineup editor, division/playoff/Cup
  probabilities from my season backtest
- **Acceptable raw:** standings (W/L/OTL/GF/GA/pts), CF%, raw HDCF counts,
  PDO components (SH%, SV%)
- **External derived (flagged):** xGoalsPercentage (team-level xGF%), HDxGF
  variants if shown

### Goalies section
- **Mine:** Sortable leaderboard rendering and ranking logic, Quality Start
  calculation from NHL game logs, team_strength goalie blend (cap is mine)
- **Acceptable raw:** raw shots-against, goals-against, ice time, GP,
  high/medium-danger shot counts, save totals
- **External derived (flagged):** GSAX (= MP xG − goals); HDSV% and MDSV%
  are zone-restricted save percentages — those are fine; the GSAX number
  needs replacing with my-xG-based version

### Playoffs section
- **Mine:** Playoff projection probabilities (round2/conf-final/final/cup
  via `monte_carlo_full_season` and `monte_carlo_playoffs_only`), live
  playoff-now re-run option
- **Acceptable raw:** NHL playoff carousel for bracket/series state, NHL
  playoff stat leaders
- **External derived (flagged):** Playoff Team Analytics table's
  xGoalsPercentage column

### Simulator section
- **Mine:** Team Strength rankings, Lineup Editor + roster swap (Phase 5.1
  with goalies now in the swap pool after today's fix), Season Simulator
  Monte Carlo per-team projection with full four-layer share model, custom
  team_strength recompute, NB(1.3) per-game goal generation
- **Acceptable raw:** NHL active rosters (player_id, position) and MoneyPuck
  25-26 goalie names + GP for the swap pool's goalie cards
- **External derived (flagged):** the `composite_war` field shown on swap
  result rows for goalies is technically MP-xG-derived since I currently
  pass `goalie_war` through it (this is goalie-only)

### Glossary
- **Mine:** All 74 stat definitions written by me; methodology explanations
- **No external derived metrics** (this is informational)

### Contract Value tab
- **Mine:** Contract-type classifier, age-curve projection, surplus-value
  formula structure, $/GAR market value constant, team-level cap-efficiency
  rollup
- **Acceptable raw:** Cap hit, expiry year, age (derived from NHL bio date)
- **External derived (flagged):** GAR, xGAR, off/def GAR (uses MP ixG/60 as
  the allocation weight), PP GAR, PK GAR, sustainability score (inputs are
  MP), goalie GAR (= GSAX = MP xG − goals). **Essentially every analytical
  input on the Contract Value tab is currently a scaled MoneyPuck value.**

---

## Recommended priority order for self-generating the remaining external derived metrics

The biggest single lever is **replacing the `GAR_PROXY` and `XGAR_PROXY` skater
fields with values derived from `composite_ratings_sim.csv::composite_war` and
my xG model.** That single change makes the Players Leaderboards, Player
Search profile, Compare, and Contract Value tabs fully self-generated for
skaters, since every downstream metric (`gar_per_million`, surplus value,
sustainability score, off/def GAR split) reads from the `gar` and `xgar`
fields.

1. **GAR + xGAR (skaters) → composite_war + my-xG-derived equivalent.** Single
   replacement point in `_build_skater_full_row()` and
   `_merge_contracts_with_moneypuck()`. Closes ~6 of the 8 flagged metrics.
2. **GSAX (goalies) → my-xG-derived version.** Single replacement point in
   `api_goalies_extended`, `api_goalies_full`, and the contract merge for
   goalies. Closes the goalie GSAX/GAR flag.
3. **xGF% (team-level + on-ice) → my-xG-derived team aggregation.** Larger
   surface but mostly a re-aggregation off `xg_model.pkl` predictions on the
   shot dataset.
4. **gameScore column → self-compute the Galamini formula from raw counts.**
   Easy mechanical replacement.
5. **Footer attribution / EH labels.** Remove "Evolving Hockey" from the
   footer and the players_leaderboards.js banners. Re-source the "Source:"
   labels to my model files where appropriate.

After (1)+(2)+(3)+(4), every analytical number on the dashboard would be
either a raw count or a calculation produced by my code from raw counts.

---

## Audit conclusion

**My calibrated models exist and are accurate.** The Phase 2/2.5/2.6/2.7/3
RAPM + composite pipeline is complete, the xG model is trained and validated,
and the Phase 5.1 simulator (team_strength + per-game NB engine + four-layer
player share model) is fully self-generated. The dashboard already uses these
where it surfaces "Composite Rating", "RAPM", "Team Strength", "Playoff
Probability", "Cup Probability", "Season Projection".

**However, the marquee "GAR", "xGAR", "Game Score", and "GSAX" columns —
which the user sees most often — are still scaled MoneyPuck values relabeled
to look like analytical outputs.** This is the gap. My composite_war and my
xG model are sufficient to replace them; the change is mostly mechanical
plumbing.

**The Contract Value tab is the largest concentrated cluster of external
derived metrics**, since every per-dollar and surplus calculation flows from
the external GAR and xGAR. Fixing (1) and (2) above closes the Contract Value
gap entirely.

Branding cleanup ("Powered by Evolving Hockey" labels, footer attribution,
EH-first source ordering) is independent of the value-correctness work and
can be done at any point. No EH or NST endpoint actually returns data — the
labels are misleading even today.

The simulator, team strength, playoff probabilities, and per-player Phase 5.1
projections are all self-generated and validated. Section 1's metrics are the
"core" that I can defensibly call my own model. Section 3's metrics are
labeled as mine but trace to MoneyPuck; they need replacement to make the
"my own model" claim true end-to-end.
