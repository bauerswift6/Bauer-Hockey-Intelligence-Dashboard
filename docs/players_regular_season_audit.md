# Players section — regular season 2025-26 audit

_Diagnostic only — no source data was rebuilt and no displayed value has
changed yet. Findings are presented for confirmation before any fix._

## Headline

**Four of the six analytical CSVs feeding the Players section currently
include 2025-26 playoff data mixed in with regular season data.** The
MoneyPuck-fed counting stats and the NHL API player-profile current-season
block are clean (regular season only). The contamination comes from the
shot- and shift-parquet sources I use for the four model builders, none of
which filter by `game_type`.

The dashboard's "Stats reflect 2025-26 regular season only" note on the
Players Overview and Leaderboards headers is therefore inaccurate today —
the goal of this audit is to make it accurate.

## NHL game_id encoding

NHL game IDs are 10 digits, `YYYYTTGGGG`:

| Digits | Meaning |
|---|---|
| 1-4 (`YYYY`) | season start year — `2025` = 2025-26 season |
| 5-6 (`TT`) | game type — `01` pre-season, `02` regular season, `03` playoffs, `04` all-star |
| 7-10 (`GGGG`) | sequence within season+type |

Existing builders use `game_id // 1_000_000` to filter by season year. **None
of them check the game-type digits**, so playoffs pass through silently.

## Source data — playoff content quantified

Both root parquets carry 2025-26 playoff rows alongside regular season:

| File | 25-26 regular | 25-26 playoff | Playoff share |
|---|---|---|---|
| `model/shots_augmented.parquet` | 111,896 shots | **5,947 shots** | 5.0% |
| `model/shots_dataset.parquet` | 111,896 shots | **5,947 shots** | 5.0% |
| `model/season_cache/shifts_20252026.parquet` | 605,621 shift records (807 unique games) | **39,523 shift records (47 unique games)** | 6.1% |
| `model/home_id_by_game.json` | 1,312 RS games | **67 playoff games** | — |

Note the shifts parquet only covers 807 of 1,312 regular season games — the
fetch was paused before the season completed, so the shifts cache is also
incomplete on the regular season side. That's a separate issue from the
playoff-contamination question this audit asks; I'm flagging but not
diagnosing here.

## Per-source verdict table

| # | Source | Verdict |
|---|---|---|
| 1 | `model/composite_ratings_single_season.csv` | **CONTAMINATED — includes playoffs** |
| 2 | `model/rapm_single_season.csv` | **CONTAMINATED — includes playoffs** |
| 3 | `model/skater_xgar_single_season.csv` | **CONTAMINATED — includes playoffs** |
| 4 | `model/skater_game_score_single_season.csv` | **regular season only ✓** |
| 5 | `model/skater_onice_xgf_self_generated.csv` | **CONTAMINATED — includes playoffs** |
| 6 | `model/skater_qoc_qot_single_season.csv` | **CONTAMINATED — includes playoffs** |
| 7 | MoneyPuck CSV pulls used by `/api/players-full` (`_mp_skaters`, `_mp_goalies`, `_mp_teams`) | **regular season only ✓** |
| 8 | NHL API `featuredStats.regularSeason.subSeason` used by `/api/player/<id>` | **regular season only ✓** |

## Builder-by-builder analysis

### 1. `model/composite_ratings_single_season.csv` — CONTAMINATED

**Builder:** `model/build_composite_single_season.py` (monkey-patches
`model/build_composite.py`).

**Inputs and contamination paths:**

- **Component 1 — RAPM (weight 0.20):** reads `model/rapm_single_season.csv`,
  which is itself contaminated (see #2 below).
- **Component 2 — Individual xG impact (weight 0.33):** the iGSAX sub-metric
  scores every shot in `shots_dataset.parquet` through the xG model and
  filters via [build_composite.py:148](model/build_composite.py#L148):

  ```python
  shots = shots[(shots["game_id"] // 1_000_000).isin(COMPONENT_GAME_YEARS)].copy()
  ```

  Season-year filter only — playoff shots pass through.
- **Component 4 — Playmaking (weight 0.18) → PP sub-metric:** re-fetches NHL
  play-by-play for the game IDs returned by
  [build_composite.py:191](model/build_composite.py#L191) `_season_game_ids()`:

  ```python
  def _season_game_ids(year: int) -> list[int]:
      ids = [int(f"{year}02{n:04d}") for n in range(1, 1313)]
      for rnd in range(1, 5):
          for m in range(1, {1: 8, 2: 4, 3: 2, 4: 1}[rnd] + 1):
              for g in range(1, 8):
                  ids.append(int(f"{year}030{rnd}{m}{g}"))   # ← playoff IDs
      return ids
  ```

  **Explicitly enumerates playoff games and appends them to the fetch list.**
- **Components 2-6 MoneyPuck inputs (counts):** pulled via the
  `MP_URL = ".../{season}/regular/{kind}.csv"` pattern — regular season only.

**Net effect:** the composite z-scoring is computed over a pool that mixes
regular season RAPM + playoff-augmented individual xG and playmaking. For
players who had deep playoff runs, this inflates their composite_rating /
composite_war / GAR.

**Specific fix needed:** In both `build_composite.py:148`, the shot filter
needs an additional clause `& (((shots["game_id"] // 10_000) % 100) == 2)`.
In `_season_game_ids()` the playoff-ID block (lines 193-196) needs to be
removed (or guarded by a `single_season=True` flag in the single-season
wrapper).

### 2. `model/rapm_single_season.csv` — CONTAMINATED

**Builder:** `model/train_rapm_single_season.py` (monkey-patches
`model/train_rapm.py`).

**Inputs and contamination paths:**

- **Shots:** the augmented parquet, loaded wholesale (the single-season
  wrapper patches `SEASONS = ["20252026"]` but no game-type filter is added).
- **Shifts:** `load_or_fetch_shifts()` iterates `SEASONS` and reads
  `season_cache/shifts_20252026.parquet`, which contains 47 playoff games of
  shift data alongside 807 regular season games. The fetch routine itself
  uses [train_rapm.py:104](model/train_rapm.py#L104) `season_game_ids()`:

  ```python
  def season_game_ids(season: str) -> list[int]:
      """All regular-season + playoff game IDs for a season. ..."""
      year = season[:4]
      ids = [int(f"{year}02{n:04d}") for n in range(1, 1313)]
      for rnd in range(1, 5):
          n_matchups = {1: 8, 2: 4, 3: 2, 4: 1}[rnd]
          for m in range(1, n_matchups + 1):
              for g in range(1, 8):
                  ids.append(int(f"{year}030{rnd}{m}{g}"))   # ← playoff IDs
      return ids
  ```

  The docstring even says "All regular-season + playoff game IDs".
- **Segment construction (`build_all_segments`):** consumes shifts × shots
  without a game-type filter; playoff segments contribute to the ridge
  regression like any other segment.

**Net effect:** RAPM ridge coefficients are trained on a pool of segments
that includes ~6-7% playoff segments. For players whose teams ran deep
into playoffs (CAR, MTL, VGK, COL, etc.) those segments are higher-leverage
in the regression because score-state weights are unchanged in playoffs.

**Specific fix needed:** Two places. First, in the source shifts parquet
(or at load time in the trainer), filter to `game_type == 2`. Second, in
`season_game_ids()`, restrict to regular season for single-season runs
(or unconditionally, since multi-season composite_sim is for talent
forecasting which traditionally regularizes toward regular-season form).

### 3. `model/skater_xgar_single_season.csv` — CONTAMINATED

**Builder:** `model/build_self_generated_stats.py`, `build_skater_xgar()`
function.

**Filter today** ([build_self_generated_stats.py:82-83](model/build_self_generated_stats.py#L82)):

```python
shots["season"] = (shots["game_id"].astype(int) // 1_000_000).astype(int)
shots = shots[shots["season"] == 2025].copy()
```

**Season-year filter only — no game-type filter.** Each player's xGAR sums
their personal shots from both regular season and playoff.

**Quantified for Connor McDavid:**

| game_type | shots | sum xG | goals |
|---|---|---|---|
| Regular season | 421 | 40.95 | 43 |
| **Playoffs** | **27** | **2.33** | **1** |
| Total (current file) | 448 | 43.28 | 44 |

McDavid's published `xgar` = 4.76 (= 28.54 / 6 in the file, where the
file's `ixg=28.54` is the subset of his 448 shots that have a successfully
attributed shooter team via the shifts-derived `teams_per_game` lookup;
many of his RS shots are missing because shifts cache is incomplete).
If the playoff shots are removed, his published xGAR would drop by
roughly the playoff `sum_xg / 6` = **0.39 wins** — about an 8% downward
shift.

**Specific fix needed:** add `& (((shots["game_id"] // 10_000) % 100) == 2)`
to the season-filter line in `load_shared_inputs()`.

### 4. `model/skater_game_score_single_season.csv` — CLEAN ✓

**Builder:** `model/build_self_generated_stats.py`, `build_game_score()`
function.

**Input:** the MoneyPuck skaters CSV pulled from
`MP_SKATERS_URL = "...moneypuck.com/.../2025/regular/skaters.csv"` —
**regular season only by URL**. MoneyPuck publishes a separate
`.../2025/playoffs/skaters.csv` which I don't fetch.

**Verdict: regular season only. No change needed.**

### 5. `model/skater_onice_xgf_self_generated.csv` — CONTAMINATED

**Builder:** `model/build_self_generated_stats.py`, `build_onice_xgf()`
function.

**Inputs:** the same 2025-season-filtered `shots` and the
`season_cache/shifts_20252026.parquet` shifts. **Neither side filters
game_type.** The per-shift x-shot intersection therefore counts playoff
shots and playoff shifts.

**Specific fix needed:** filter both `shots` and `shifts` to
`game_type == 2` before the intersection loop.

### 6. `model/skater_qoc_qot_single_season.csv` — CONTAMINATED

**Builder:** `model/build_qoc_qot.py`.

**Inputs:** loads `season_cache/shifts_20252026.parquet` wholesale and
groups by `game_id` for pairwise overlap. **No game-type filter at all.**

When I ran the build, the script reported:

```
Computing pairwise overlaps across 854 games …
```

854 = 807 RS games + **47 playoff games**. McDavid's pairwise-overlap
calculation includes shifts from EDM's playoff series, which were
disproportionately overlap with COL/VGK/etc. top lines (deeper-round
opponents had higher pre-existing QoC weighting).

**Quantified for Connor McDavid:**

| game_type | shifts | total minutes |
|---|---|---|
| Regular season | 1,115 | 1,153 |
| **Playoffs** | **151** | **141** |

About **11% of McDavid's tracked shift time is playoffs.** QoC and QoT
weighted-average GAR over those shifts. His current QoT = +0.901 would
shift modestly if playoffs were removed (likely lower by ~0.05 since deep
playoff opponents had QoT-equivalent partner GARs).

**Specific fix needed:** add a single line at the top of `main()`:

```python
shifts = shifts[((shifts["game_id"] // 10_000) % 100) == 2].copy()
```

### 7. MoneyPuck CSV pulls (`_mp_skaters` / `_mp_goalies` / `_mp_teams`) — CLEAN ✓

**Helpers:** [app.py:643-666](app.py#L643).

**URL:** `MP_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular"` —
**hardcoded `/regular/`**. MoneyPuck's situational-row publishing (the
`situation` column with `all` / `5on5` / `5on4` / `4on5` / `other`) is over
the regular season window only at this URL.

**Verdict: regular season only.** This is the entire raw-counts spine of
`/api/players-full` (G/A/PTS/Shots/Hits/Blocks/PIM/TOI/Zone Starts/PP and
PK situation rows). All counting stats and on-ice raw counts that don't
go through my xG model are clean.

### 8. NHL API `/api/player/<id>` current-season block — CLEAN ✓

**Code:** [app.py:2771-2774](app.py#L2771):

```python
feat = landing.get("featuredStats") or {}
cur = (feat.get("regularSeason") or {}).get("subSeason") or {}
career_totals = (feat.get("regularSeason") or {}).get("career") or {}
```

**Explicitly reads the `regularSeason.subSeason` branch.** The NHL landing
endpoint also exposes a `featuredStats.playoffs.subSeason` block which the
profile code does NOT touch.

The `career_totals` row also gates by `gameTypeId == 2` at
[app.py:2575](app.py#L2575) (used for the years-in-league count).

**Verdict: regular season only.**

## Summary table of fixes needed

If you confirm, the changes needed are mechanical and small. Six locations
across four files; no methodology changes.

| File | Change | Lines |
|---|---|---|
| `model/build_self_generated_stats.py` | `load_shared_inputs()` — add `& (((shots["game_id"] // 10_000) % 100) == 2)` to the shots filter. Also filter `shifts` to `game_type == 2`. | ~line 83 |
| `model/build_composite.py` | `compute_igsax_by_player()` — add the same `game_type == 2` clause to the shot filter at line 148. | ~line 148 |
| `model/build_composite.py` | `_season_game_ids()` — remove the playoff-ID generator block at lines 193-196 (or gate behind a flag). | 193-196 |
| `model/train_rapm.py` | `season_game_ids()` — same removal at lines 109-113. | 109-113 |
| `model/train_rapm.py` | Shift loader (or main pipeline) — filter the merged shifts DataFrame to `game_type == 2` before segment construction. | ~line 755 |
| `model/build_qoc_qot.py` | `main()` — add `shifts = shifts[((shifts["game_id"] // 10_000) % 100) == 2].copy()` after the shifts load. | ~line 132 |

## Magnitude of the expected shift after the fix

For McDavid specifically (deep playoff runner in 25-26):

| Metric | Current (mixed) | After RS-only filter (estimated) | Δ |
|---|---|---|---|
| xGAR | 4.76 | ~4.37 | **−0.39 (~−8%)** |
| RAPM offensive | +0.36 | likely slightly lower (small) | small |
| QoT | +0.901 | likely slightly lower (deep playoff opponents had high pre-existing QoT-equivalents weighting) | small |
| Composite/GAR | +2.10 | likely +1.95 to +2.05 (driven by iGSAX + RAPM shifts) | **moderate** |

For a player who missed the playoffs (e.g., NYR, NSH, BUF skaters who were
out by April): **no shift**, because they have zero playoff segments to
remove.

This means **fixing the contamination will not lower everyone equally** —
it will compress the gap between deep-playoff players and missed-playoff
players in favor of the latter group's relative ranking. This is the
correct directional outcome for a "regular season only" view.

## Confirmation prompt

Per your instruction in the audit brief:

> If any source includes playoff data, do not make changes — report the
> findings first so I can confirm the fix before it goes live. Rebuilding
> any of these files would shift all the displayed values for players who
> played in the 2025-26 playoffs.

**Four CSVs are contaminated:** `composite_ratings_single_season.csv`,
`rapm_single_season.csv`, `skater_xgar_single_season.csv`,
`skater_onice_xgf_self_generated.csv`, `skater_qoc_qot_single_season.csv`.

**Two are already clean:** `skater_game_score_single_season.csv` (MoneyPuck
URL is `/regular/`), and all `/api/player/<id>` current-season fields (NHL
API explicitly reads `featuredStats.regularSeason.subSeason`).

To fix:
1. Apply the six small filter changes listed in the table above.
2. Rebuild in dependency order: RAPM first (single-season trainer ~3 min),
   then composite single-season (~30s), then self-generated xGAR + on-ice
   xGF (~2.5 min), then QoC/QoT (~3s).
3. Clear the `composite_war_lookup`, `skater_xgar_lookup`, `onice_xgf_lookup`,
   and `qoc_qot_lookup` cached dicts in app.py (or restart Flask to drop
   the in-memory cache).

After the fix, McDavid's GAR will drop ~5-8%, his QoT will drop ~0.05, his
xGAR will drop ~0.39. Players who missed the playoffs will move up in
relative rank.

I have not changed any source data or any displayed value. Awaiting
confirmation before applying the fix.
