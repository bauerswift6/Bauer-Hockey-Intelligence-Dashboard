"""
RAPM ridge regression model — Phase 2.

Pipeline:
  1. Load shots_dataset.parquet (built in Phase 1) and apply xG model to score every shot.
  2. Pull shift data for every unique game ID from the NHL shiftcharts endpoint
     (https://api.nhle.com/stats/rest/en/shiftcharts) — cache to shifts_dataset.parquet.
  3. Build "segments" — windows where the same 10 skaters are on the ice.
     Filter to 5v5 even strength, segments ≥ 10 sec.
  4. Attribute xGF/xGA to each segment via overlap with scored shots.
  5. Apply score-state weighting (>3 goal diff → 0.5).
  6. Filter to players with ≥ 1000 EV minutes across the 3 seasons.
  7. Build the player-segment sparse matrix using duplicated-row encoding
     (one row per perspective per segment) so off/def coefficients are identified.
  8. Run RidgeCV for offensive and defensive RAPM. Total = off − def.
  9. Save model/rapm_results.csv with player_id, name, team, season, total/off/def RAPM, TOI, shifts.

Run from project root:
    python3 model/train_rapm.py

Re-runs use the cached shifts parquet unless --refresh-shifts is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
from scipy import sparse
from sklearn.linear_model import RidgeCV

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
CACHE_DIR = MODEL_DIR / "season_cache"

SHOTS_PATH = MODEL_DIR / "shots_dataset.parquet"
SHIFTS_PATH = MODEL_DIR / "shifts_dataset.parquet"
XG_MODEL_PATH = MODEL_DIR / "xg_model.pkl"
RAPM_CSV_PATH = MODEL_DIR / "rapm_results.csv"
RAPM_META_PATH = MODEL_DIR / "rapm_meta.json"

NHL_SHIFTS = "https://api.nhle.com/stats/rest/en/shiftcharts"
NHL_API = "https://api-web.nhle.com/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# 16 seasons — 2010-11 through 2025-26 (2025-refresh).
SEASONS = [
    "20102011", "20112012", "20122013", "20132014", "20142015",
    "20152016", "20162017", "20172018", "20182019", "20192020",
    "20202021", "20212022", "20222023", "20232024", "20242025",
    "20252026",
]
MIN_SEGMENT_SECS = 10
MIN_EV_MINUTES = 500
ALPHAS = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

# Year-weighting (2025-refresh): decay-slot anchored at 2025-26 = 1.00; the decay
# sequence shifted one slot older vs. v4. COVID/lockout games-played fractions
# stay attached to their actual calendar seasons. New floor weight (0.04) for the
# 16th-oldest slot. Implemented as RidgeCV sample_weight × score-state weight.
SEASON_WEIGHTS = {
    "20252026": 1.00,
    "20242025": 0.82,
    "20232024": 0.67,
    "20222023": 0.55,
    "20212022": 0.42,                # full 82-game season (post-COVID)
    "20202021": 0.34 * (56 / 82),    # 56-game COVID-shortened season
    "20192020": 0.28 * (69.5 / 82),  # ~69.5-game COVID-suspended season
    "20182019": 0.23,
    "20172018": 0.19,
    "20162017": 0.15,
    "20152016": 0.12,
    "20142015": 0.10,
    "20132014": 0.07,
    "20122013": 0.06 * (48 / 82),    # 48-game lockout-shortened season
    "20112012": 0.05,
    "20102011": 0.04,                # new oldest floor (16th-slot)
}
DEFAULT_WEIGHT = 0.04
# Goalie adjustment cap (goals per 60)
GOALIE_ADJ_CAP = 0.5
# MoneyPuck season codes for the 16-season goalie career GSAX build (2010-2025)
MP_GOALIE_SEASONS = [str(y) for y in range(2010, 2026)]


def season_str_from_game_id(game_id: int) -> str:
    start = int(game_id) // 1_000_000
    return f"{start}{start + 1}"


def season_game_ids(season: str) -> list[int]:
    """All regular-season + playoff game IDs for a season. Older 30/31-team
    and lockout seasons simply 404 on the unused tail."""
    year = season[:4]
    ids = [int(f"{year}02{n:04d}") for n in range(1, 1313)]
    for rnd in range(1, 5):
        n_matchups = {1: 8, 2: 4, 3: 2, 4: 1}[rnd]
        for m in range(1, n_matchups + 1):
            for g in range(1, 8):
                ids.append(int(f"{year}030{rnd}{m}{g}"))
    return ids


# ---------------------------------------------------------------------------
# Stage 1: score every shot with the Phase 1 model
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "x_norm", "y_norm",
    "distance_from_net", "angle_from_center",
    "shot_type_id",
    "is_rush", "is_rebound",
    "time_since_last_shot",
    "period",
    "score_state",
    "is_home",
]


def score_shots() -> pd.DataFrame:
    print(f"Loading shots dataset → {SHOTS_PATH}", flush=True)
    df = pd.read_parquet(SHOTS_PATH)
    print(f"  {len(df):,} shots loaded.", flush=True)

    print(f"Loading xG model → {XG_MODEL_PATH}", flush=True)
    model = joblib.load(XG_MODEL_PATH)
    print(f"Scoring shots …", flush=True)
    X = df[FEATURE_COLS].astype("float32")
    df["xg"] = model.predict_proba(X)[:, 1]
    print(f"  Mean xG: {df['xg'].mean():.4f} (vs goal rate {df['is_goal'].mean():.4f})", flush=True)
    return df


# ---------------------------------------------------------------------------
# Stage 2: pull shift data
# ---------------------------------------------------------------------------
def fetch_shifts(game_id: int, max_retries: int = 2) -> list[dict] | None:
    url = NHL_SHIFTS
    params = {"cayenneExp": f"gameId={game_id}"}
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json().get("data", [])
        except Exception:
            if attempt < max_retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            return None
    return None


def parse_mmss(s: str) -> int:
    if not s or ":" not in str(s):
        return 0
    try:
        m, sec = str(s).split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return 0


def shift_record_to_dict(rec: dict) -> dict | None:
    """Normalize a shiftcharts record into our minimal schema."""
    # detailCode 0 = regular shift; other codes are event markers (goals etc.)
    if rec.get("detailCode") != 0:
        return None
    pid = rec.get("playerId")
    if not pid:
        return None
    period = rec.get("period") or 1
    start = parse_mmss(rec.get("startTime"))
    end = parse_mmss(rec.get("endTime"))
    if end <= start:
        return None
    abs_start = (period - 1) * 1200 + start
    abs_end = (period - 1) * 1200 + end
    return {
        "game_id": int(rec.get("gameId")),
        "player_id": int(pid),
        "team_id": int(rec.get("teamId") or 0),
        "team_abbrev": rec.get("teamAbbrev") or "",
        "first_name": rec.get("firstName") or "",
        "last_name": rec.get("lastName") or "",
        "period": int(period),
        "abs_start": int(abs_start),
        "abs_end": int(abs_end),
    }


def bulk_pull_shifts(game_ids: list[int], workers: int = 8) -> pd.DataFrame:
    print(f"Pulling shifts for {len(game_ids):,} games "
          f"with {workers} parallel workers …", flush=True)
    rows = []
    completed = 0
    skipped = 0
    start = time.time()

    def _process(gid):
        recs = fetch_shifts(gid)
        if not recs:
            return None
        out = []
        for r in recs:
            d = shift_record_to_dict(r)
            if d:
                out.append(d)
        return out

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, gid): gid for gid in game_ids}
        for fut in as_completed(futures):
            try:
                shifts = fut.result()
            except Exception:
                shifts = None
            completed += 1
            if not shifts:
                skipped += 1
            else:
                rows.extend(shifts)
            if completed % 250 == 0:
                elapsed = time.time() - start
                rate = completed / max(elapsed, 0.1)
                eta = (len(game_ids) - completed) / max(rate, 0.1)
                print(f"  {completed:,}/{len(game_ids):,} games — "
                      f"{len(rows):,} shifts — "
                      f"{rate:.1f} games/s — ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - start
    print(f"Pulled {completed} games in {elapsed:.1f}s "
          f"({skipped} returned no data — typically future / cancelled)", flush=True)
    return pd.DataFrame(rows)


def load_or_fetch_shifts(workers: int = 8) -> pd.DataFrame:
    """For each of the 15 seasons: load the cached per-season shift parquet if
    it exists, otherwise fetch every game's shifts from the NHL shiftcharts API
    and cache it. Returns the concatenated shift DataFrame."""
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for season in SEASONS:
        path = CACHE_DIR / f"shifts_{season}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"  [{season}] cached — {len(df):,} shift records (skipping fetch).", flush=True)
        else:
            print(f"  [{season}] fetching shifts …", flush=True)
            df = bulk_pull_shifts(season_game_ids(season), workers=workers)
            if df.empty:
                print(f"  [{season}] WARNING: no shift data — skipping season.", flush=True)
                continue
            df.to_parquet(path, index=False)
            print(f"  [{season}] cached → {path.name}", flush=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Stage 3: build a player → position lookup so we can identify goalies
# ---------------------------------------------------------------------------
def build_goalie_gsax() -> dict:
    """Build {goalie playerId -> career GSAX per 60} aggregated across the 6
    MoneyPuck goalie CSV seasons (2019-2024). GSAX = xGoals − goals; per-60 over
    total icetime. Used to remove goalie quality from skater on-ice xGA."""
    from io import StringIO
    print("Building 15-season goalie career GSAX/60 from MoneyPuck …", flush=True)
    tot_gsax, tot_ice = {}, {}
    for s in MP_GOALIE_SEASONS:
        url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{s}/regular/goalies.csv"
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                print(f"  goalie CSV {s}: HTTP {r.status_code} — skipped", flush=True)
                continue
            df = pd.read_csv(StringIO(r.text))
        except Exception as e:
            print(f"  goalie CSV {s} failed: {e}", flush=True)
            continue
        df = df[df["situation"] == "all"]
        for _, row in df.iterrows():
            try:
                pid = int(row["playerId"])
            except (TypeError, ValueError):
                continue
            xg = float(pd.to_numeric(row.get("xGoals", 0), errors="coerce") or 0)
            g = float(pd.to_numeric(row.get("goals", 0), errors="coerce") or 0)
            ice = float(pd.to_numeric(row.get("icetime", 0), errors="coerce") or 0)
            tot_gsax[pid] = tot_gsax.get(pid, 0.0) + (xg - g)
            tot_ice[pid] = tot_ice.get(pid, 0.0) + ice
    gsax60 = {pid: tot_gsax[pid] / (ice / 3600.0)
              for pid, ice in tot_ice.items() if ice > 0}
    if gsax60:
        vals = sorted(gsax60.values())
        print(f"  {len(gsax60):,} goalies — career GSAX/60 range "
              f"{vals[0]:.3f} to {vals[-1]:.3f}", flush=True)
    return gsax60


def build_position_lookup() -> dict:
    """Build {playerId -> 'G' | 'F' | 'D'} from MoneyPuck per-season CSVs.
    Pull 6 seasons of skaters + goalies, prefer most recent."""
    print("Building player position lookup from MoneyPuck …", flush=True)
    pid_to_pos = {}
    seasons = [str(y) for y in range(2010, 2026)]
    for s in seasons:
        for kind in ("skaters", "goalies"):
            url = f"https://moneypuck.com/moneypuck/playerData/seasonSummary/{s}/regular/{kind}.csv"
            try:
                r = requests.get(url, headers=HEADERS, timeout=20)
                if r.status_code != 200:
                    continue
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                for _, row in df.iterrows():
                    pid = row.get("playerId")
                    pos = row.get("position", "")
                    if pid and pid == pid:
                        try:
                            pid_to_pos[int(pid)] = "G" if kind == "goalies" else (
                                "D" if pos == "D" else "F"
                            )
                        except (TypeError, ValueError):
                            continue
            except Exception as e:
                print(f"  Warning: failed to load {kind} {s}: {e}", flush=True)
    print(f"  Loaded {len(pid_to_pos):,} player positions.", flush=True)
    return pid_to_pos


# ---------------------------------------------------------------------------
# Stage 4: build segments per game
# ---------------------------------------------------------------------------
def build_segments_for_game(game_shifts: pd.DataFrame, game_shots: pd.DataFrame,
                             home_team_id: int, away_team_id: int,
                             pid_to_pos: dict) -> list[dict]:
    """Convert per-player shift records + shot events into per-segment rows.

    Vectorized: pre-converts shifts to numpy arrays and uses np.searchsorted to
    locate active shifts at each window boundary.

    Returns list of dicts: {abs_start, abs_end, dur_sec, home_skaters, away_skaters,
                            home_xgf, home_xga, score_diff_home}.
    """
    if game_shifts.empty:
        return []

    # Pre-extract numpy arrays per team
    home_mask = (game_shifts["team_id"] == home_team_id).values
    away_mask = (game_shifts["team_id"] == away_team_id).values

    starts = game_shifts["abs_start"].values
    ends = game_shifts["abs_end"].values
    pids = game_shifts["player_id"].values.astype(np.int64)
    is_g = np.array([pid_to_pos.get(int(p)) == "G" for p in pids], dtype=bool)

    home_starts = starts[home_mask]; home_ends = ends[home_mask]
    home_pids = pids[home_mask];     home_is_g = is_g[home_mask]
    away_starts = starts[away_mask]; away_ends = ends[away_mask]
    away_pids = pids[away_mask];     away_is_g = is_g[away_mask]

    # Window boundaries = unique sorted set of all start/end times
    times = np.unique(np.concatenate([starts, ends]))
    if times.size < 2:
        return []

    # Shots: pre-extract numpy arrays for fast windowed sum
    if not game_shots.empty:
        shot_t = game_shots["abs_secs"].values.astype(np.int64)
        shot_xg = game_shots["xg"].values.astype(np.float64)
        shot_home = (game_shots["is_home"].values == 1)
        shot_goal = (game_shots["is_goal"].values == 1)
        # Sort by time for searchsorted
        order = np.argsort(shot_t)
        shot_t = shot_t[order]; shot_xg = shot_xg[order]
        shot_home = shot_home[order]; shot_goal = shot_goal[order]
        # Cumulative sums for O(1) window queries
        cum_home_xgf = np.concatenate([[0.0], np.cumsum(np.where(shot_home, shot_xg, 0.0))])
        cum_home_xga = np.concatenate([[0.0], np.cumsum(np.where(~shot_home, shot_xg, 0.0))])
        cum_home_g = np.concatenate([[0], np.cumsum(np.where(shot_goal & shot_home, 1, 0))])
        cum_away_g = np.concatenate([[0], np.cumsum(np.where(shot_goal & ~shot_home, 1, 0))])
    else:
        shot_t = np.array([], dtype=np.int64)
        cum_home_xgf = np.array([0.0])
        cum_home_xga = np.array([0.0])
        cum_home_g = np.array([0])
        cum_away_g = np.array([0])

    segments = []

    for i in range(times.size - 1):
        t0 = int(times[i])
        t1 = int(times[i + 1])
        dur = t1 - t0
        if dur < MIN_SEGMENT_SECS:
            continue  # score state still tracks via cumulative — no need for per-window update

        # Active shifts at t0: start <= t0 < end
        h_active = (home_starts <= t0) & (home_ends > t0)
        a_active = (away_starts <= t0) & (away_ends > t0)

        h_pids_on = home_pids[h_active]
        h_is_g_on = home_is_g[h_active]
        a_pids_on = away_pids[a_active]
        a_is_g_on = away_is_g[a_active]

        home_skaters = h_pids_on[~h_is_g_on]
        home_goalie_ids = h_pids_on[h_is_g_on]
        away_skaters = a_pids_on[~a_is_g_on]
        away_goalie_ids = a_pids_on[a_is_g_on]
        home_goalie = len(home_goalie_ids) > 0
        away_goalie = len(away_goalie_ids) > 0

        if not (len(home_skaters) == 5 and len(away_skaters) == 5 and home_goalie and away_goalie):
            continue

        # Sum xG within [t0, t1) via cumulative arrays + searchsorted
        if shot_t.size:
            lo = int(np.searchsorted(shot_t, t0, side="left"))
            hi = int(np.searchsorted(shot_t, t1, side="left"))
            home_xgf = float(cum_home_xgf[hi] - cum_home_xgf[lo])
            home_xga = float(cum_home_xga[hi] - cum_home_xga[lo])
            home_score_now = int(cum_home_g[lo])
            away_score_now = int(cum_away_g[lo])
        else:
            home_xgf = 0.0; home_xga = 0.0
            home_score_now = 0; away_score_now = 0

        segments.append({
            "abs_start": t0,
            "abs_end": t1,
            "dur_sec": dur,
            "home_skaters": tuple(int(p) for p in home_skaters),
            "away_skaters": tuple(int(p) for p in away_skaters),
            "home_xgf": home_xgf,
            "home_xga": home_xga,
            "score_diff_home": home_score_now - away_score_now,
            "home_goalie_id": int(home_goalie_ids[0]),
            "away_goalie_id": int(away_goalie_ids[0]),
        })

    return segments


def build_all_segments(shifts_df: pd.DataFrame, shots_df: pd.DataFrame,
                       home_id_by_game: dict, pid_to_pos: dict) -> pd.DataFrame:
    """Build segment-level dataset across every game."""
    print(f"\nBuilding 5v5 segments across {shifts_df['game_id'].nunique():,} games …", flush=True)

    # Group shots and shifts by game
    shifts_by_game = {gid: g for gid, g in shifts_df.groupby("game_id")}
    shots_by_game = {gid: g for gid, g in shots_df.groupby("game_id")} if "game_id" in shots_df.columns else {}

    all_segments = []
    games_processed = 0
    games_skipped = 0
    t0 = time.time()
    for gid, gshifts in shifts_by_game.items():
        home_id = home_id_by_game.get(gid)
        if not home_id:
            games_skipped += 1
            continue
        teams_in_game = list(set(gshifts["team_id"].tolist()))
        away_id = next((t for t in teams_in_game if t != home_id), None)
        if not away_id:
            games_skipped += 1
            continue

        gshots = shots_by_game.get(gid, pd.DataFrame())
        if not gshots.empty:
            # Compute absolute seconds for each shot (period * 1200 + parsed time)
            # Phase 1 didn't store this directly — need to reconstruct.
            # We DID store period and time_since_last_shot but not the absolute time-in-period.
            # Workaround: store nothing — fall back to event ordinal counting? No, that won't align.
            # Actually we never stored time_in_period for shots. Big oversight.
            # Skip xG-segment alignment for now and handle separately below.
            pass

        segments = build_segments_for_game(gshifts, gshots, home_id, away_id, pid_to_pos)
        for seg in segments:
            seg["game_id"] = gid
        all_segments.extend(segments)
        games_processed += 1
        if games_processed % 250 == 0:
            elapsed = time.time() - t0
            rate = games_processed / max(elapsed, 0.1)
            print(f"  {games_processed} games segmented — "
                  f"{len(all_segments):,} segments — "
                  f"{rate:.1f} games/s", flush=True)

    print(f"  Done. {games_processed} games processed, {games_skipped} skipped.", flush=True)
    print(f"  Total 5v5 segments: {len(all_segments):,}", flush=True)
    return pd.DataFrame(all_segments)


# ---------------------------------------------------------------------------
# Stage 5: build sparse matrix and train
# ---------------------------------------------------------------------------
def build_player_index(segments_df: pd.DataFrame, shifts_df: pd.DataFrame,
                       pid_to_pos: dict) -> tuple[dict, dict]:
    """Compute total EV TOI per player from segments. Filter to ≥1000 min.
    Return (player_id -> col_idx, col_idx -> player_id)."""
    print(f"\nComputing EV TOI per player …", flush=True)
    toi = {}
    for _, r in segments_df.iterrows():
        for pid in r["home_skaters"]:
            toi[pid] = toi.get(pid, 0) + r["dur_sec"]
        for pid in r["away_skaters"]:
            toi[pid] = toi.get(pid, 0) + r["dur_sec"]

    qualifying = {pid: secs for pid, secs in toi.items() if secs / 60 >= MIN_EV_MINUTES and pid_to_pos.get(pid) != "G"}
    print(f"  {len(qualifying):,} skaters with ≥ {MIN_EV_MINUTES} EV min "
          f"(out of {len(toi):,} total).", flush=True)

    pid_list = sorted(qualifying.keys())
    pid_to_col = {pid: i for i, pid in enumerate(pid_list)}
    col_to_pid = {i: pid for pid, i in pid_to_col.items()}
    return pid_to_col, col_to_pid, qualifying


def build_design_matrix(segments_df: pd.DataFrame, pid_to_col: dict,
                        goalie_gsax: dict | None = None
                        ) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """Build the doubled-COLUMN design matrix (the EH/MoneyPuck standard).

    Each player gets TWO columns:
      - col[player_idx]: 'on offense' (player on the team-of-interest's offense)
      - col[player_idx + P]: 'on defense' (player on the opposing team, defending)

    Each segment → 2 rows (one per perspective):
      row A: from home's perspective. Home players +1 in off cols. Away players +1 in def cols.
             Targets: y_xgf=home_xgf/60, y_xga=home_xga/60
      row B: from away's perspective. Away players +1 in off cols. Home players +1 in def cols.
             Targets: y_xgf=away_xgf/60=home_xga/60, y_xga=away_xga/60=home_xgf/60

    With this encoding, off and def coefficients are INDEPENDENTLY identified
    (no symmetry collapse). Returns (X, y_xgf60, y_xga60, weights).
    """
    print(f"\nBuilding sparse design matrix (doubled-column encoding) …", flush=True)
    n_segs = len(segments_df)
    n_rows = n_segs * 2
    P = len(pid_to_col)
    n_cols = 2 * P  # P offensive cols, P defensive cols

    rows_list = []
    cols_list = []
    vals_list = []
    y_xgf = np.zeros(n_rows, dtype=np.float32)
    y_xga = np.zeros(n_rows, dtype=np.float32)
    weights = np.ones(n_rows, dtype=np.float32)

    goalie_gsax = goalie_gsax or {}
    n_goalie_adjusted = 0
    seg_records = segments_df.to_dict(orient="records")
    for i, r in enumerate(seg_records):
        dur_min = r["dur_sec"] / 60.0
        if dur_min <= 0:
            continue
        # Per-60 normalization (raw)
        home_xgf60 = r["home_xgf"] * (60.0 / dur_min)
        home_xga60 = r["home_xga"] * (60.0 / dur_min)

        # --- Goalie adjustment: remove the goalie's above-average save contribution
        # from the on-ice xGA only (xGF targets stay raw — offense isn't goalie-driven).
        # home_xga is defended by the HOME goalie; the away team's xGA (= home_xgf as a
        # quantity) is defended by the AWAY goalie. Subtract capped career GSAX/60.
        hg = goalie_gsax.get(r.get("home_goalie_id", 0), 0.0)
        ag = goalie_gsax.get(r.get("away_goalie_id", 0), 0.0)
        hg = max(-GOALIE_ADJ_CAP, min(GOALIE_ADJ_CAP, hg))
        ag = max(-GOALIE_ADJ_CAP, min(GOALIE_ADJ_CAP, ag))
        if hg != 0.0 or ag != 0.0:
            n_goalie_adjusted += 1
        home_xga60_adj = max(0.0, home_xga60 - hg)   # home goalie defends home's xGA
        away_xga60_adj = max(0.0, home_xgf60 - ag)   # away goalie defends away's xGA

        # --- Year-weighting: season recency × score-state weight (sample_weight) ---
        season = season_str_from_game_id(r["game_id"])
        season_w = SEASON_WEIGHTS.get(season, DEFAULT_WEIGHT)
        score_w = 0.5 if abs(r["score_diff_home"]) > 3 else 1.0
        weight = season_w * score_w

        home_in = [pid_to_col[p] for p in r["home_skaters"] if p in pid_to_col]
        away_in = [pid_to_col[p] for p in r["away_skaters"] if p in pid_to_col]

        # Row A — home is the team-of-interest
        rA = i * 2
        # Home players on offense → off cols
        for c in home_in:
            rows_list.append(rA); cols_list.append(c); vals_list.append(1.0)
        # Away players defending → def cols (= P + idx)
        for c in away_in:
            rows_list.append(rA); cols_list.append(c + P); vals_list.append(1.0)
        y_xgf[rA] = home_xgf60          # offense — raw
        y_xga[rA] = home_xga60_adj      # defense — goalie-adjusted
        weights[rA] = weight

        # Row B — away is the team-of-interest
        rB = i * 2 + 1
        for c in away_in:
            rows_list.append(rB); cols_list.append(c); vals_list.append(1.0)
        for c in home_in:
            rows_list.append(rB); cols_list.append(c + P); vals_list.append(1.0)
        # From away's perspective: their xGF = home's xGA (raw), their xGA = home's xGF
        # (goalie-adjusted by the away goalie).
        y_xgf[rB] = home_xga60          # away's offense — raw
        y_xga[rB] = away_xga60_adj      # away's defense — goalie-adjusted
        weights[rB] = weight

    X = sparse.csr_matrix(
        (vals_list, (rows_list, cols_list)),
        shape=(n_rows, n_cols), dtype=np.float32
    )
    print(f"  X shape: {X.shape}  ({P} off cols + {P} def cols),  nnz: {X.nnz:,}", flush=True)
    print(f"  Mean xGF/60: {y_xgf.mean():.3f},  Mean xGA/60 (goalie-adj): {y_xga.mean():.3f}", flush=True)
    print(f"  Goalie-adjusted segments: {n_goalie_adjusted:,} / {n_segs:,}", flush=True)
    print(f"  Sample-weight range: {weights.min():.3f}–{weights.max():.3f} "
          f"(season recency × score-state)", flush=True)
    return X, y_xgf, y_xga, weights


def fit_ridge(X, y, weights, alphas=ALPHAS, label="off"):
    print(f"\nFitting RidgeCV for {label} target across {len(alphas)} alphas …", flush=True)
    t0 = time.time()
    model = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
    model.fit(X, y, sample_weight=weights)
    print(f"  done in {time.time()-t0:.1f}s — alpha selected: {model.alpha_}", flush=True)
    return model


# ---------------------------------------------------------------------------
# Stage 6: assemble results
# ---------------------------------------------------------------------------
def assemble_results(off_model, def_model, col_to_pid, qualifying_toi,
                     shifts_df, off_alpha, def_alpha) -> pd.DataFrame:
    """Pull offensive RAPM from the offense regression's off-cols (first P coefs),
    pull defensive RAPM from the defense regression's def-cols (last P coefs),
    NEGATED so that POSITIVE = better defender (suppresses xGA).
    Total RAPM = offensive RAPM + (negated) defensive RAPM."""
    print(f"\nAssembling results …", flush=True)
    P = len(col_to_pid)

    # Player metadata from shifts
    name_team = (shifts_df.sort_values("game_id")
                 .groupby("player_id")
                 .agg(first_name=("first_name", "last"),
                      last_name=("last_name", "last"),
                      team=("team_abbrev", "last"),
                      shift_count=("game_id", "size"))
                 .to_dict(orient="index"))

    rows = []
    for col_idx, pid in col_to_pid.items():
        # From the off regression (target=team_xGF/60), the off-column coef tells us
        # how much player adds to their team's xGF when on offense.
        off_rapm = float(off_model.coef_[col_idx])

        # From the def regression (target=team_xGA/60), the def-column coef tells us
        # how much the player ALLOWS xGA to opp when defending. Higher coef = worse.
        # Negate so that a positive defensive_rapm value = better defender.
        def_raw = float(def_model.coef_[col_idx + P])
        def_rapm = -def_raw

        # Total RAPM = pure offensive contribution + pure defensive contribution
        total_rapm = off_rapm + def_rapm

        toi_min = qualifying_toi.get(pid, 0) / 60.0
        meta = name_team.get(pid, {})
        rows.append({
            "player_id": pid,
            "player_name": f"{meta.get('first_name','')} {meta.get('last_name','')}".strip(),
            "team": meta.get("team", ""),
            "season": "20102025",
            "total_rapm": round(total_rapm, 4),
            "offensive_rapm": round(off_rapm, 4),
            "defensive_rapm": round(def_rapm, 4),
            "toi_minutes": round(toi_min, 1),
            "shift_count": int(meta.get("shift_count", 0)),
        })
    out = pd.DataFrame(rows).sort_values("total_rapm", ascending=False).reset_index(drop=True)
    out["off_alpha"] = off_alpha
    out["def_alpha"] = def_alpha
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-shifts", action="store_true",
                        help="Re-pull shift data even if cached parquet exists.")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Fetch any missing seasons of shift data into the "
                             "per-season cache, then exit (no training).")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.fetch_only:
        print("Fetching 15 seasons of shift data (per-season cache) …", flush=True)
        t0 = time.time()
        shifts_df = load_or_fetch_shifts(workers=args.workers)
        print(f"\nTotal: {len(shifts_df):,} shift records across "
              f"{shifts_df['game_id'].nunique():,} games — {time.time()-t0:.0f}s.", flush=True)
        print("--fetch-only: seasons cached, skipping training.", flush=True)
        return

    if not SHOTS_PATH.exists():
        print(f"ERROR: {SHOTS_PATH} not found. Run Phase 1 (model/train_xg.py) first.", flush=True)
        sys.exit(1)
    if not XG_MODEL_PATH.exists():
        print(f"ERROR: {XG_MODEL_PATH} not found. Run Phase 1 first.", flush=True)
        sys.exit(1)

    # Stage 1: load shots WITH abs_secs.
    # Prefer the cached shots_augmented.parquet (has abs_secs + already xG-scored)
    # so re-runs never re-fetch PBP. Fall back to the PBP re-parse only if it's missing.
    augmented_path = MODEL_DIR / "shots_augmented.parquet"
    home_meta = MODEL_DIR / "home_id_by_game.json"

    if augmented_path.exists() and home_meta.exists():
        print(f"Loading cached augmented shots → {augmented_path}", flush=True)
        shots_df = pd.read_parquet(augmented_path)
        home_id_by_game = {int(k): int(v) for k, v in json.loads(home_meta.read_text()).items()}
        print(f"  {len(shots_df):,} shots loaded (abs_secs cached, no PBP re-fetch).", flush=True)
        # ALWAYS (re)score the xg column with the current xg_model.pkl so the RAPM
        # pipeline reflects the latest (e.g. isotonic-calibrated) model — never trust
        # a stale cached `xg` column.
        print(f"  Scoring shots with current xG model → {XG_MODEL_PATH}", flush=True)
        model = joblib.load(XG_MODEL_PATH)
        shots_df["xg"] = model.predict_proba(shots_df[FEATURE_COLS].astype("float32"))[:, 1]
        print(f"  Mean xG per shot: {shots_df['xg'].mean():.4f} "
              f"(goal rate {shots_df['is_goal'].mean():.4f})", flush=True)
    else:
        shots_df = score_shots()
        shots_df, home_id_by_game = augment_shots_with_abs_secs(shots_df)
        shots_df.to_parquet(augmented_path, index=False)
        home_meta.write_text(json.dumps({str(k): v for k, v in home_id_by_game.items()}))

    # Stage 2: shifts — per-season cache (each season fetched only if missing).
    print(f"\nLoading / fetching 15 seasons of shift data (per-season cache) …", flush=True)
    shifts_df = load_or_fetch_shifts(workers=args.workers)
    if shifts_df.empty:
        print("ERROR: no shift data available.", flush=True)
        sys.exit(1)
    shifts_df.to_parquet(SHIFTS_PATH, index=False)
    print(f"  {len(shifts_df):,} shift records — combined cache → {SHIFTS_PATH}", flush=True)

    pid_to_pos = build_position_lookup()
    goalie_gsax = build_goalie_gsax()

    segments_df = build_all_segments(shifts_df, shots_df, home_id_by_game, pid_to_pos)
    if segments_df.empty:
        print("ERROR: no segments built.", flush=True)
        sys.exit(1)

    pid_to_col, col_to_pid, qualifying_toi = build_player_index(
        segments_df, shifts_df, pid_to_pos
    )
    if not pid_to_col:
        print(f"ERROR: no players cleared {MIN_EV_MINUTES} EV min threshold.", flush=True)
        sys.exit(1)

    X, y_xgf, y_xga, weights = build_design_matrix(segments_df, pid_to_col, goalie_gsax)

    # Single regression on target=team_xGF/60. With the doubled-row + doubled-column
    # encoding, this single regression identifies BOTH offensive and defensive
    # coefficients per player independently:
    #   - First P coefs = off RAPM (positive = better attacker, boosts xGF when on offense)
    #   - Last P coefs = "allowed-while-defending" (positive = bad defender, allows xGF
    #     when on defense). We negate to produce def RAPM where positive = better defender.
    # Running separate regressions on xGF vs xGA targets (as initially spec'd) collapses
    # under this symmetric encoding because the def-regression solution is the
    # off-regression solution with off/def columns swapped — so off=-def exactly.
    alphas = list(ALPHAS)
    model = fit_ridge(X, y_xgf, weights, alphas=alphas, label="combined (target = team_xGF/60)")
    # Boundary check — if the optimal alpha pins at either grid edge, widen the
    # grid in that direction and re-fit (immediately, without asking).
    while model.alpha_ == alphas[0] or model.alpha_ == alphas[-1]:
        if model.alpha_ == alphas[-1]:
            print(f"  alpha pinned at UPPER boundary {alphas[-1]} — widening grid up.", flush=True)
            top = alphas[-1]
            alphas = alphas + [top * 2, top * 5, top * 10]
        else:
            print(f"  alpha pinned at LOWER boundary {alphas[0]} — widening grid down.", flush=True)
            lo = alphas[0]
            alphas = [lo / 10, lo / 5, lo / 2] + alphas
        model = fit_ridge(X, y_xgf, weights, alphas=alphas, label="combined (widened grid)")
    print(f"  Optimal alpha {model.alpha_} is interior to grid "
          f"[{alphas[0]} … {alphas[-1]}].", flush=True)
    alphas_used = alphas
    # Use the same model object's coefs for both off (first P) and def (last P, negated)
    off_model = model
    def_model = model

    results = assemble_results(off_model, def_model, col_to_pid, qualifying_toi,
                                shifts_df, model.alpha_, model.alpha_)

    print(f"\nSaving → {RAPM_CSV_PATH}", flush=True)
    results.to_csv(RAPM_CSV_PATH, index=False)

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seasons": SEASONS,
        "n_segments": int(len(segments_df)),
        "n_design_rows": int(X.shape[0]),
        "n_players": int(len(pid_to_col)),
        "min_segment_secs": MIN_SEGMENT_SECS,
        "min_ev_minutes": MIN_EV_MINUTES,
        "alphas_tested": alphas_used,
        "off_alpha_selected": float(off_model.alpha_),
        "def_alpha_selected": float(def_model.alpha_),
        "encoding": "duplicated-row (one row per perspective per segment)",
        "score_state_weighting": "0.5 weight when |score_diff| > 3",
        "season_weights": SEASON_WEIGHTS,
        "year_weighting": "season recency applied as RidgeCV sample_weight (x score-state)",
        "goalie_adjustment": (f"on-ice xGA reduced by goalie career GSAX/60, "
                              f"capped +/-{GOALIE_ADJ_CAP}/60"),
    }
    RAPM_META_PATH.write_text(json.dumps(meta, indent=2))

    # Print top/bottom
    print(f"\nTotal players: {len(results)}", flush=True)
    print(f"Optimal alphas: off={off_model.alpha_}, def={def_model.alpha_}", flush=True)

    print("\nTop 10 by total RAPM:")
    for _, r in results.head(10).iterrows():
        print(f"  {r['player_name']:25s} {r['team']:4s} "
              f"total={r['total_rapm']:+.4f}  off={r['offensive_rapm']:+.4f}  def={r['defensive_rapm']:+.4f}  TOI={r['toi_minutes']:.0f}min")

    print("\nBottom 10 by total RAPM:")
    for _, r in results.tail(10).iterrows():
        print(f"  {r['player_name']:25s} {r['team']:4s} "
              f"total={r['total_rapm']:+.4f}  off={r['offensive_rapm']:+.4f}  def={r['defensive_rapm']:+.4f}  TOI={r['toi_minutes']:.0f}min")

    print("\nTop 5 by offensive RAPM:")
    for _, r in results.sort_values("offensive_rapm", ascending=False).head(5).iterrows():
        print(f"  {r['player_name']:25s} {r['team']:4s} off={r['offensive_rapm']:+.4f}")

    print("\nTop 5 by defensive RAPM (positive = best, suppresses xGA):")
    for _, r in results.sort_values("defensive_rapm", ascending=False).head(5).iterrows():
        print(f"  {r['player_name']:25s} {r['team']:4s} def={r['defensive_rapm']:+.4f}")


# ---------------------------------------------------------------------------
# Helper: augment shots with abs_secs by re-fetching PBP (parallel, cached)
# ---------------------------------------------------------------------------
def augment_shots_with_abs_secs(shots_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Phase 1's shots_dataset doesn't have abs_secs. Re-fetch PBP for every game
    in the dataset to recover both:
      - abs_secs per shot (period * 1200 + time_in_period)
      - home_team_id per game (used downstream by segment builder)
    Single pass over PBP."""
    print(f"\nRe-fetching PBP for {shots_df['game_id'].nunique():,} games to recover abs_secs and home IDs…", flush=True)
    games = list(shots_df.groupby("game_id"))

    def _process(item):
        gid, game_shots = item
        try:
            r = requests.get(f"{NHL_API}/gamecenter/{gid}/play-by-play",
                             headers=HEADERS, timeout=15)
            if r.status_code != 200:
                return None
            d = r.json()
            plays = d.get("plays") or []
            home_id = (d.get("homeTeam") or {}).get("id")

            # Walk plays, count shots in same order Phase 1 used (goal/shot-on-goal/missed-shot,
            # has coords, not empty-net). Phase 1's order corresponds 1:1 with our re-walk
            # ONLY if we apply the same filters. We can't perfectly replicate the empty-net
            # filter without situationCode, so use a lenient match: pair by index assuming
            # counts roughly match.
            shot_events = []
            for p in plays:
                t = p.get("typeDescKey")
                if t not in {"goal", "shot-on-goal", "missed-shot"}:
                    continue
                details = p.get("details") or {}
                if details.get("xCoord") is None or details.get("yCoord") is None:
                    continue
                # Skip empty-net (matches Phase 1)
                sit_code = p.get("situationCode") or ""
                shooter_team = details.get("eventOwnerTeamId")
                if shooter_team is None:
                    continue
                shooter_is_home = (shooter_team == home_id)
                if sit_code and len(sit_code) == 4:
                    away_g, _, _, home_g = sit_code[0], sit_code[1], sit_code[2], sit_code[3]
                    empty_net = (shooter_is_home and away_g == "0") or (not shooter_is_home and home_g == "0")
                    if empty_net:
                        continue
                period = (p.get("periodDescriptor") or {}).get("number") or 1
                time_str = p.get("timeInPeriod") or "00:00"
                m, s = time_str.split(":") if ":" in time_str else ("0", "0")
                abs_secs = (period - 1) * 1200 + int(m) * 60 + int(s)
                shot_events.append(abs_secs)

            # Pair by ORDER with Phase 1 shots for this game
            game_shots_sorted = game_shots.copy()
            n = len(game_shots_sorted)
            if len(shot_events) >= n:
                game_shots_sorted["abs_secs"] = shot_events[:n]
            else:
                # Fewer reconstructed shots than expected — pad missing with -1
                padded = shot_events + [-1] * (n - len(shot_events))
                game_shots_sorted["abs_secs"] = padded
            return gid, game_shots_sorted, home_id
        except Exception:
            return None

    out_pieces = []
    home_id_by_game = {}
    completed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_process, item): item[0] for item in games}
        for fut in as_completed(futures):
            res = fut.result()
            completed += 1
            if res:
                gid, augmented_shots, home_id = res
                out_pieces.append(augmented_shots)
                if home_id:
                    home_id_by_game[gid] = home_id
            if completed % 250 == 0:
                rate = completed / max(time.time() - t0, 0.1)
                print(f"  {completed}/{len(games)} games re-parsed — {rate:.1f}/s", flush=True)

    if not out_pieces:
        print("WARNING: no shots augmented; abs_secs missing.", flush=True)
        shots_df["abs_secs"] = -1
        return shots_df, {}

    augmented_df = pd.concat(out_pieces, ignore_index=False).sort_index()
    print(f"  Augmented {len(augmented_df):,} shots with abs_secs.", flush=True)
    print(f"  Resolved {len(home_id_by_game):,} home team IDs.", flush=True)
    return augmented_df, home_id_by_game


if __name__ == "__main__":
    main()
