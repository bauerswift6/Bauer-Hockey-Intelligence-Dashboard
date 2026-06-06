"""
Composite Player Rating System — Phase 2.5.

Combines three independently z-scored components into a single player-strength
metric that will feed the game-simulation model (replacing raw RAPM as the
primary metric):

  Component 1 — RAPM                 weight 0.35   (total_rapm, 0 if not in RAPM set)
  Component 2 — Individual xG impact weight 0.40   (ixG/60 + iHDCF/60, 5v5)
  Component 3 — Relative on-ice xGF% weight 0.25   (on-ice xGF% minus off-ice team xGF%)

Each component is z-scored to mean=0/std=1 (stats taken over the qualified
≥300-EV-minute pool) before the weighted sum.

Run from project root:
    python3 model/build_composite.py

Uses model/rapm_results.csv (Phase 2), the MoneyPuck season CSVs, and a
play-by-play re-fetch for the PP-finishing sub-metric (Fix 2, cached per
season). Output: model/composite_ratings.csv
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
CACHE_DIR = MODEL_DIR / "season_cache"
RAPM_CSV = MODEL_DIR / "rapm_results.csv"
SHOTS_CSV = MODEL_DIR / "shots_dataset.parquet"
AUGMENTED_CSV = MODEL_DIR / "shots_augmented.parquet"
XG_MODEL_PATH = MODEL_DIR / "xg_model.pkl"
OUT_CSV = MODEL_DIR / "composite_ratings.csv"
OUT_META = MODEL_DIR / "composite_meta.json"
# Name to rename the existing OUT_CSV to before a --save overwrite. Override at
# module level (e.g. from build_composite_sim.py) when running a parallel build.
PRIOR_ARCHIVE_NAME = "composite_ratings_v3_archived.csv"
NHL_API = "https://api-web.nhle.com/v1"

# Phase 1 xG model feature columns (for scoring shots → per-player finishing)
FEATURE_COLS = [
    "x_norm", "y_norm", "distance_from_net", "angle_from_center", "shot_type_id",
    "is_rush", "is_rebound", "time_since_last_shot", "period", "score_state", "is_home",
]

# 2025-refresh — Components 2-6 use the 4 most recent seasons (2022-23 → 2025-26).
# RAPM (Component 1) uses the full 16-season model. MoneyPuck codes the season by
# its START year, so "2022" = 2022-23 … "2025" = 2025-26.
COMPONENT_SEASONS = ["2022", "2023", "2024", "2025"]
# Game-ID season start-years for the 4-season window (PP-shot re-fetch, finishing).
COMPONENT_GAME_YEARS = [2022, 2023, 2024, 2025]
MP_URL = ("https://moneypuck.com/moneypuck/playerData/seasonSummary/"
          "{season}/regular/{kind}.csv")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# Year-weighting within the 4-season window (2025-refresh; keyed by MoneyPuck
# start-year code). All four seasons are full 82-game seasons.
SEASON_WEIGHTS = {
    "2025": 1.00,
    "2024": 0.82,
    "2023": 0.67,
    "2022": 0.55,
}


def game_id_season_weight(game_id: int) -> float:
    """Year-weight for a shot, keyed by the game ID's season start year.
    Returns 0.0 for any season outside the 4-season component window."""
    return SEASON_WEIGHTS.get(str(int(game_id) // 1_000_000), 0.0)

# v4 — 6 components incl. special teams. Sum = 1.00.
# Changes from v3:
#   - RAPM ↑0.0125 (0.1875→0.20), Individual ↑0.1175 (0.2625→0.38, with iGSAX),
#     Relative xGF% ↑0.05 (0.15→0.20), Playmaking ↑0.05 (0.15→0.20, with points/60
#     sub-metric), PP ↓0.03 (0.15→0.12), PK unchanged.
WEIGHTS = {
    "rapm": 0.20, "individual": 0.33, "relative_xgf": 0.17,
    "playmaking": 0.18, "power_play": 0.07, "penalty_kill": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values())}"
# Replacement-level baseline gate (≥500 EV min over the 4-season window). Used ONLY
# for the WAR-equivalent rescale; the per-component qualified pool stays at ≥200 EV min.
MIN_EV_MINUTES = 200   # Components 2 & 3 + overall qualified pool
MIN_REPL_EV_MINUTES = 500  # pool used to define replacement level + WAR sd
MIN_ST_MINUTES = 60    # minimum PP / PK minutes to receive a special-teams component
# Soft cap on blended-z components (C2, C4, C5, C6) — v4 raised to ±3.0 to give
# uncapped elites (Draisaitl/Matthews iGSAX, Draisaitl/MacKinnon playmaking) room
# to score above the previous ±2.5 ceiling.
COMPONENT_CAP = 3.0
# Component 2 sub-metric weights — v4 replaces SH%-above-expected with iGSAX/60
# (total goals above expected per 60), which captures both volume AND conversion.
C2_SUBWEIGHTS = {"ixg_60": 0.40, "ihdcf_60": 0.30, "igsax_60": 0.30}
# Component 4 sub-metric weights — v4 adds an individual-points/60 sub-metric
# (global z) alongside the position-blended primary-assists/60.
C4_SUBWEIGHTS = {"primary_assists_60": 0.70, "points_60": 0.30}
# Blended z-score split for position-aware components (C2 individual, C4 playmaking)
BLEND_GLOBAL = 0.40
BLEND_POSREL = 0.60
# WAR rescale: one unit = 0.5 wins. composite_war = (raw - replacement)/std * 0.5
WAR_UNIT = 0.5


def get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def zscore(series: pd.Series, ref: pd.Series) -> pd.Series:
    """Z-score `series` using mean/std computed from `ref` (the qualified pool)."""
    mu = ref.mean()
    sd = ref.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def compute_igsax_by_player() -> tuple[dict, dict]:
    """v4 sub-metric — per-player Individual Goals Scored Above Expected (iGSAX).

    Loads the Phase 1 shot dataset (15 seasons), scores every shot with the
    calibrated xG model, restricts to the 4-season component window, then
    year-weights and aggregates per shooter:
        iGSAX_total = Σ w·is_goal  −  Σ w·xg   (across ALL of the player's shots)

    The shots dataset already excludes empty-net attempts but is otherwise
    all-strengths; iGSAX therefore captures finishing across EV + PP + PK in a
    single number, consistent with the spec's pointer at shots_augmented.parquet.
    Returns ({pid → iGSAX_total}, {pid → n_shots}).
    """
    shots = pd.read_parquet(SHOTS_CSV)
    if "shooter_id" not in shots.columns:
        raise RuntimeError("shots_dataset.parquet has no shooter_id — re-run train_xg.py")
    shots = shots[(shots["game_id"] // 1_000_000).isin(COMPONENT_GAME_YEARS)].copy()
    model = joblib.load(XG_MODEL_PATH)
    shots["xg"] = model.predict_proba(shots[FEATURE_COLS].astype("float32"))[:, 1]
    shots["sw"] = shots["game_id"].map(game_id_season_weight)
    shots = shots[shots["shooter_id"] > 0]

    shots["w_goal"] = shots["sw"] * shots["is_goal"]
    shots["w_xg"] = shots["sw"] * shots["xg"]
    agg = shots.groupby("shooter_id").agg(
        n=("is_goal", "size"),
        w_goal=("w_goal", "sum"),
        w_xg=("w_xg", "sum"),
    )
    agg["igsax_total"] = agg["w_goal"] - agg["w_xg"]
    print(f"  Computed year-weighted iGSAX (goals above expected) for {len(agg):,} "
          f"shooters from {len(shots):,} Phase 1 shots in the 4-season window.",
          flush=True)
    return agg["igsax_total"].to_dict(), agg["n"].to_dict()


# ---------------------------------------------------------------------------
# Fix 2 — PP goals above expected (sub-metric B of Component 5)
# ---------------------------------------------------------------------------
SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot"}


def _fetch_pbp(game_id: int, max_retries: int = 2) -> dict | None:
    url = f"{NHL_API}/gamecenter/{game_id}/play-by-play"
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt < max_retries:
                time.sleep(0.4 * (attempt + 1))
                continue
            return None
    return None


def _season_game_ids(year: int) -> list[int]:
    ids = [int(f"{year}02{n:04d}") for n in range(1, 1313)]
    for rnd in range(1, 5):
        for m in range(1, {1: 8, 2: 4, 3: 2, 4: 1}[rnd] + 1):
            for g in range(1, 8):
                ids.append(int(f"{year}030{rnd}{m}{g}"))
    return ids


def _extract_pp_shots(pbp: dict | None) -> list[tuple]:
    """Return (abs_secs, shooter_id, is_goal) for 5-on-4 shots taken by the
    team on the power play (both goalies in net, shooter's team has 5 skaters,
    opponent 4) — matches MoneyPuck's 5on4 situation."""
    if not pbp:
        return []
    home_id = (pbp.get("homeTeam") or {}).get("id")
    out = []
    for p in pbp.get("plays") or []:
        if p.get("typeDescKey") not in SHOT_TYPES:
            continue
        sit = p.get("situationCode") or ""
        if len(sit) != 4:
            continue
        details = p.get("details") or {}
        team = details.get("eventOwnerTeamId")
        if team is None:
            continue
        away_g, away_sk, home_sk, home_g = sit[0], sit[1], sit[2], sit[3]
        if away_g != "1" or home_g != "1":
            continue   # both goalies must be in net (excludes empty-net)
        shooter_is_home = (team == home_id)
        if shooter_is_home:
            is_5on4 = (home_sk == "5" and away_sk == "4")
        else:
            is_5on4 = (away_sk == "5" and home_sk == "4")
        if not is_5on4:
            continue
        sid = details.get("shootingPlayerId") or details.get("scoringPlayerId")
        if not sid:
            continue
        period = (p.get("periodDescriptor") or {}).get("number") or 1
        ts = p.get("timeInPeriod") or "00:00"
        try:
            mm, ss = ts.split(":")
            abs_secs = (period - 1) * 1200 + int(mm) * 60 + int(ss)
        except Exception:
            continue
        out.append((abs_secs, int(sid), 1 if p.get("typeDescKey") == "goal" else 0))
    return out


def fetch_pp_shots() -> pd.DataFrame:
    """Re-fetch play-by-play for the 4-season window and extract 5-on-4 shots.
    Cached per season at model/season_cache/pp_shots_<season>.parquet — each
    season is fetched only if its parquet is missing."""
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for year in COMPONENT_GAME_YEARS:
        season = f"{year}{year + 1}"
        path = CACHE_DIR / f"pp_shots_{season}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            print(f"    [{season}] cached — {len(df):,} PP shots (skipping fetch).", flush=True)
            frames.append(df)
            continue
        game_ids = _season_game_ids(year)
        print(f"    [{season}] re-fetching play-by-play for {len(game_ids)} games …", flush=True)
        rows = []
        t0 = time.time()

        def _proc(gid):
            return gid, _extract_pp_shots(_fetch_pbp(gid))

        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_proc, g): g for g in game_ids}
            for fut in as_completed(futs):
                gid, shots = fut.result()
                for abs_secs, sid, is_goal in shots:
                    rows.append({"game_id": gid, "abs_secs": abs_secs,
                                 "shooter_id": sid, "is_goal": is_goal})
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)
        print(f"    [{season}] {len(df):,} PP shots in {time.time()-t0:.0f}s → {path.name}",
              flush=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compute_pp_finishing() -> dict:
    """Fix 2 sub-metric B — individual PP goals above expected, year-weighted,
    over the 4-season window. Each 5-on-4 shot's xG is taken from
    shots_augmented.parquet (the Phase 1 model's score) by matching on
    (game_id, abs_secs, shooter_id). Returns {player_id -> goals_above_expected}."""
    print("  Building PP goals-above-expected (Fix 2 sub-metric B) …", flush=True)
    pp = fetch_pp_shots()
    aug = pd.read_parquet(AUGMENTED_CSV)
    aug = aug[(aug["game_id"] // 1_000_000).isin(COMPONENT_GAME_YEARS)]
    xg_idx: dict = {}
    for gid, secs, sid, xg in zip(aug["game_id"], aug["abs_secs"],
                                  aug["shooter_id"], aug["xg"]):
        xg_idx.setdefault((int(gid), int(secs), int(sid)), []).append(float(xg))
    acc: dict = {}   # pid -> [weighted goals, weighted xg]
    matched = unmatched = 0
    for gid, secs, sid, is_goal in zip(pp["game_id"], pp["abs_secs"],
                                       pp["shooter_id"], pp["is_goal"]):
        lst = xg_idx.get((int(gid), int(secs), int(sid)))
        if not lst:
            unmatched += 1
            continue
        xg = lst.pop(0)
        matched += 1
        w = game_id_season_weight(gid)
        a = acc.setdefault(int(sid), [0.0, 0.0])
        a[0] += w * float(is_goal)
        a[1] += w * xg
    print(f"    {len(pp):,} PP shots — matched {matched:,} to shots_augmented xG "
          f"({unmatched:,} unmatched: no coords, skipped).", flush=True)
    return {pid: g - x for pid, (g, x) in acc.items()}


def refresh_current_teams(player_ids: list[int], workers: int = 16,
                          max_retries: int = 1) -> dict[int, str]:
    """Pull live `currentTeamAbbrev` from NHL API `/v1/player/{id}/landing` for
    every player_id in parallel. Returns {player_id → team_abbrev}. Players whose
    landing page is unreachable (retired, edge-case IDs) are simply omitted; the
    caller falls back to the MoneyPuck most-recent-season label."""
    url_t = f"{NHL_API}/player/{{}}/landing"
    out: dict[int, str] = {}

    def _one(pid: int) -> tuple[int, str | None]:
        for attempt in range(max_retries + 1):
            try:
                r = requests.get(url_t.format(pid), headers=HEADERS, timeout=8)
                if r.status_code == 200:
                    abbrev = (r.json() or {}).get("currentTeamAbbrev")
                    if abbrev:
                        return pid, str(abbrev)
                    return pid, None
                if r.status_code == 404:
                    return pid, None
            except Exception:
                if attempt < max_retries:
                    time.sleep(0.3)
                    continue
        return pid, None

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, p): p for p in player_ids}
        for fut in as_completed(futs):
            pid, ab = fut.result()
            if ab:
                out[pid] = ab
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - t0, 0.1)
                print(f"    [{done:,}/{len(player_ids):,}] live team labels — "
                      f"{rate:.1f}/s", flush=True)
    print(f"    Done — {len(out):,}/{len(player_ids):,} resolved in "
          f"{time.time()-t0:.0f}s.", flush=True)
    return out


def main(save: bool = False):
    print(f"Building composite player ratings (v4) … {'[SAVE]' if save else '[dry-run]'}", flush=True)

    # --- Load RAPM (Phase 2) ---
    if not RAPM_CSV.exists():
        print(f"ERROR: {RAPM_CSV} not found. Run Phase 2 (model/train_rapm.py) first.", flush=True)
        sys.exit(1)
    rapm = pd.read_csv(RAPM_CSV)
    rapm_by_id = dict(zip(rapm["player_id"].astype(int), rapm["total_rapm"].astype(float)))
    print(f"  Loaded RAPM for {len(rapm_by_id):,} players.", flush=True)

    # --- v4 — per-player iGSAX (goals above expected) from Phase 1 xG, 4-season window ---
    igsax_by_id, igsax_shot_count = compute_igsax_by_player()

    # --- Per-player PP goals above expected (Fix 2 sub-metric B) ---
    pp_finishing_by_id = compute_pp_finishing()

    # --- Aggregate the 4-season MoneyPuck skater + team data (year-weighted) ---
    # Fix 1: Components 2-6 use only 2021-22 → 2024-25. Counting stats are summed
    # across seasons with each scaled by its year-weight; per-60 rates then divide
    # weighted stat by weighted icetime, so recent/higher-volume seasons dominate.
    # Relative xGF% is computed per season (needs that season's team context) then
    # year-weighted averaged. Identity (name/team/position) is taken from the most
    # recent season.
    from collections import defaultdict
    print(f"  Aggregating {len(COMPONENT_SEASONS)} MoneyPuck seasons "
          f"(2021-22 … 2024-25) …", flush=True)

    acc = defaultdict(lambda: {
        "all_ixg": 0.0, "all_ihdcf": 0.0, "all_pa": 0.0, "all_goals": 0.0,
        "all_w_ice": 0.0, "all_gp_w": 0.0,
        "pp_onf": 0.0, "pp_w_ice": 0.0, "pp_raw_ice": 0.0,
        "pk_ona": 0.0, "pk_w_ice": 0.0, "pk_raw_ice": 0.0,
        "rel_num": 0.0, "rel_den": 0.0,   # legacy xGF% (diagnostic only)
        # EH-diagnostic R1 — rate-based offensive + defensive sub-metrics at 5v5.
        # Track year-weighted player on-ice xGF/xGA + 5v5 icetime, AND year-weighted
        # team-total xGF/xGA + 5v5 iceTime per team-stint, so we can compute
        # off-ice team rates as (team_total - player_on_ice) / (team_ice - player_ice).
        "p5_on_f": 0.0, "p5_on_a": 0.0, "p5_w_ice": 0.0,
        "tm_xgf": 0.0, "tm_xga": 0.0, "tm_w_ice": 0.0,
        # EH-diagnostic R2 — short-handed individual offense (4on5 G/A/SA).
        "pk_g": 0.0, "pk_pa": 0.0, "pk_sa": 0.0,
        "ev_raw_ice": 0.0,
    })
    meta_recent = {}   # pid -> (season_idx, name, team, position)

    for idx, code in enumerate(COMPONENT_SEASONS):
        w = SEASON_WEIGHTS[code]
        try:
            skaters = get_csv(MP_URL.format(season=code, kind="skaters"))
            teams = get_csv(MP_URL.format(season=code, kind="teams"))
        except Exception as e:
            print(f"  WARNING: MoneyPuck {code} failed ({e}) — season skipped.", flush=True)
            continue

        t5 = teams[teams["situation"] == "5on5"].copy()
        for c in ["xGoalsFor", "xGoalsAgainst", "iceTime"]:
            t5[c] = pd.to_numeric(t5[c], errors="coerce").fillna(0)
        team_xgf = dict(zip(t5["team"], t5["xGoalsFor"]))
        team_xga = dict(zip(t5["team"], t5["xGoalsAgainst"]))
        team_ice = dict(zip(t5["team"], t5["iceTime"]))   # team 5v5 iceTime in seconds

        def _rows(situation, cols):
            sub = skaters[skaters["situation"] == situation].copy()
            for c in cols:
                sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0)
            return sub

        s_all = _rows("all", ["I_F_xGoals", "I_F_highDangerShots", "I_F_primaryAssists",
                              "I_F_goals", "icetime", "games_played"])
        s5 = _rows("5on5", ["OnIce_F_xGoals", "OnIce_A_xGoals", "icetime"])
        s_pp = _rows("5on4", ["OnIce_F_xGoals", "icetime"])
        s_pk = _rows("4on5", ["OnIce_A_xGoals", "icetime",
                              "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists"])

        # all-situations individual offense (Components 2 & 4)
        for _, r in s_all.iterrows():
            try:
                pid = int(r["playerId"])
            except (TypeError, ValueError):
                continue
            a = acc[pid]
            a["all_ixg"] += w * float(r["I_F_xGoals"])
            a["all_ihdcf"] += w * float(r["I_F_highDangerShots"])
            a["all_pa"] += w * float(r["I_F_primaryAssists"])
            a["all_goals"] += w * float(r["I_F_goals"])
            a["all_w_ice"] += w * float(r["icetime"])
            # EH-diagnostic R4 — year-weighted games played (for expected_gp_share)
            a["all_gp_w"] += w * float(r.get("games_played", 0))
            meta_recent[pid] = (idx, r.get("name", ""), r.get("team", ""), r.get("position", ""))

        # 5v5 on-ice → relative xGF% (legacy) + rate-based offensive/defensive
        # sub-metrics (EH-diagnostic R1) + qualification TOI
        for _, r in s5.iterrows():
            try:
                pid = int(r["playerId"])
            except (TypeError, ValueError):
                continue
            ice = float(r["icetime"])
            if ice <= 0:
                continue
            a = acc[pid]
            a["ev_raw_ice"] += ice
            on_f = float(r["OnIce_F_xGoals"])
            on_a = float(r["OnIce_A_xGoals"])
            tf = team_xgf.get(r["team"], 0.0)
            ta = team_xga.get(r["team"], 0.0)
            ti = team_ice.get(r["team"], 0.0)   # team 5v5 iceTime in seconds
            # Legacy proportion-based aggregation (kept for diagnostic comparison).
            off_f = max(tf - on_f, 0.0)
            off_a = max(ta - on_a, 0.0)
            on_pct = (on_f / (on_f + on_a) * 100.0) if (on_f + on_a) > 0 else 0.0
            off_pct = (off_f / (off_f + off_a) * 100.0) if (off_f + off_a) > 0 else 0.0
            a["rel_num"] += w * ice * (on_pct - off_pct)
            a["rel_den"] += w * ice
            # EH-diagnostic R1 — year-weighted player on-ice xGF/xGA + ice, AND
            # year-weighted team-stint xGF/xGA + team iceTime so we can later
            # derive per-60 off-ice team rates.
            a["p5_on_f"] += w * on_f
            a["p5_on_a"] += w * on_a
            a["p5_w_ice"] += w * ice
            a["tm_xgf"] += w * tf
            a["tm_xga"] += w * ta
            a["tm_w_ice"] += w * ti

        for _, r in s_pp.iterrows():
            try:
                pid = int(r["playerId"])
            except (TypeError, ValueError):
                continue
            a = acc[pid]
            a["pp_onf"] += w * float(r["OnIce_F_xGoals"])
            a["pp_w_ice"] += w * float(r["icetime"])
            a["pp_raw_ice"] += float(r["icetime"])

        for _, r in s_pk.iterrows():
            try:
                pid = int(r["playerId"])
            except (TypeError, ValueError):
                continue
            a = acc[pid]
            a["pk_ona"] += w * float(r["OnIce_A_xGoals"])
            a["pk_w_ice"] += w * float(r["icetime"])
            a["pk_raw_ice"] += float(r["icetime"])
            # EH-diagnostic R2 — short-handed individual G/A/SA (year-weighted)
            a["pk_g"] += w * float(r.get("I_F_goals", 0))
            a["pk_pa"] += w * float(r.get("I_F_primaryAssists", 0))
            a["pk_sa"] += w * float(r.get("I_F_secondaryAssists", 0))
        print(f"    {code}: {len(s_all):,} skaters (weight {w:.4f})", flush=True)

    # --- Build per-player rows from the aggregated accumulators ---
    rows = []
    for pid, a in acc.items():
        if pid not in meta_recent:
            continue
        _, name, team, position = meta_recent[pid]

        # Components 2 & 4: all-situations per-60 rates (weighted sum / weighted icetime).
        # iGSAX/60 uses the same denominator so all C2 sub-metrics share a TOI base.
        # points/60 (C4) is (goals + primary assists) / 60 all-situations.
        all_hours = a["all_w_ice"] / 3600.0
        if all_hours > 0:
            ixg_60 = a["all_ixg"] / all_hours
            ihdcf_60 = a["all_ihdcf"] / all_hours
            primary_assists_60 = a["all_pa"] / all_hours
            igsax_60 = float(igsax_by_id.get(pid, 0.0)) / all_hours
            points_60 = (a["all_goals"] + a["all_pa"]) / all_hours
        else:
            ixg_60 = ihdcf_60 = primary_assists_60 = igsax_60 = points_60 = 0.0

        # Component 3: legacy year-weighted average of per-season relative on-ice xGF%
        # (kept on the dataframe for diagnostic comparison; no longer feeds the model).
        rel_xgf = (a["rel_num"] / a["rel_den"]) if a["rel_den"] > 0 else 0.0

        # EH-diagnostic R1 — rate-based offensive + defensive sub-metrics at 5v5.
        # player on-ice xGF/60 minus team xGF/60 when player is off the ice; same
        # construction for xGA. Both are year-weighted by construction (the
        # accumulators were already year-weighted).
        if a["p5_w_ice"] > 0 and a["tm_w_ice"] > a["p5_w_ice"]:
            player_xgf_60 = a["p5_on_f"] / (a["p5_w_ice"] / 3600.0)
            player_xga_60 = a["p5_on_a"] / (a["p5_w_ice"] / 3600.0)
            off_ice_team_hours = (a["tm_w_ice"] - a["p5_w_ice"]) / 3600.0
            team_off_xgf_60 = (a["tm_xgf"] - a["p5_on_f"]) / off_ice_team_hours
            team_off_xga_60 = (a["tm_xga"] - a["p5_on_a"]) / off_ice_team_hours
            rel_xgf_off_60 = player_xgf_60 - team_off_xgf_60     # positive = good offense
            rel_xga_def_60 = -(player_xga_60 - team_off_xga_60)  # inverted: positive = good defense
        else:
            rel_xgf_off_60 = rel_xga_def_60 = 0.0

        # Component 5 — power play: weighted on-ice xGF/60 at 5on4, min 60 raw PP min.
        pp_min = a["pp_raw_ice"] / 60.0
        if pp_min >= MIN_ST_MINUTES and a["pp_w_ice"] > 0:
            pp_xgf_60 = a["pp_onf"] / (a["pp_w_ice"] / 3600.0)
            pp_qualified = True
        else:
            pp_xgf_60, pp_qualified = 0.0, False

        # Component 6 — penalty kill: weighted on-ice xGA/60 at 4on5 + EH-diagnostic
        # R2 SH-points/60 sub-metric (year-weighted; only meaningful when pk_qualified).
        pk_min = a["pk_raw_ice"] / 60.0
        if pk_min >= MIN_ST_MINUTES and a["pk_w_ice"] > 0:
            pk_xga_60 = a["pk_ona"] / (a["pk_w_ice"] / 3600.0)
            sh_points_60 = (a["pk_g"] + a["pk_pa"] + a["pk_sa"]) / (a["pk_w_ice"] / 3600.0)
            pk_qualified = True
        else:
            pk_xga_60, sh_points_60, pk_qualified = 0.0, 0.0, False

        # EH-diagnostic R4 — expected_gp_share = year-weighted average GP / 82, with
        # the FULL window denominator (sum of season weights) so seasons not played
        # count as 0 GP (conservative availability estimate). For 3-season window with
        # weights {1.00, 0.60, 0.25}, denominator = 1.85.
        gp_w_denom = sum(SEASON_WEIGHTS.values()) * 82.0
        expected_gp_share = (a["all_gp_w"] / gp_w_denom) if gp_w_denom > 0 else 0.0
        expected_gp_share = min(expected_gp_share, 1.0)   # cap at 1.0 to be safe

        rows.append({
            "player_id": pid,
            "player_name": name,
            "team": team,
            "position": position,
            "toi_minutes": round(a["ev_raw_ice"] / 60.0, 1),  # raw total 5v5 minutes
            "rapm_raw": rapm_by_id.get(pid, 0.0),   # 0 if not in RAPM dataset
            "ixg_60": ixg_60,
            "ihdcf_60": ihdcf_60,
            # v4 C2 sub-metric 3 — Individual Goals Scored Above Expected per 60.
            "igsax_60": igsax_60,
            # v4 keep raw iGSAX total for the diagnostic report (Draisaitl/Matthews).
            "igsax_total": float(igsax_by_id.get(pid, 0.0)),
            "primary_assists_60": primary_assists_60,
            # v4 C4 sub-metric 2 — individual points (G+A1) per 60, global z-scored.
            "points_60": points_60,
            "rel_xgf_raw": rel_xgf,
            # EH-diagnostic R1 — rate-based offensive + defensive sub-metrics.
            "rel_xgf_off_60": rel_xgf_off_60,
            "rel_xga_def_60": rel_xga_def_60,
            "pp_xgf_60": pp_xgf_60,
            # Fix 2 sub-metric B: PP goals above expected (year-weighted, 4-season).
            "pp_goals_above_exp": float(pp_finishing_by_id.get(pid, 0.0)),
            "pp_qualified": pp_qualified,
            "pk_xga_60": pk_xga_60,
            # EH-diagnostic R2 — short-handed individual points/60 (sub-B of PK).
            "sh_points_60": sh_points_60,
            "pk_qualified": pk_qualified,
            # EH-diagnostic R4 — availability column for Phase 4 to multiply against.
            "expected_gp_share": round(expected_gp_share, 4),
        })

    df = pd.DataFrame(rows)
    df["sample_size_flag"] = np.where(df["toi_minutes"] >= MIN_EV_MINUTES, "ok", "small_sample")
    n_qual = int((df["sample_size_flag"] == "ok").sum())
    print(f"  {len(df):,} skaters total, {n_qual:,} with ≥ {MIN_EV_MINUTES} EV min "
          f"(qualified pool for z-score stats).", flush=True)

    qual = df[df["sample_size_flag"] == "ok"]
    qual_mask = (df["sample_size_flag"] == "ok")
    is_d = (df["position"] == "D")
    pos_stats = {}  # for reporting

    # --- Blended z-score (Fix 1): 40% global + 60% position-relative ---
    # Global z corrects nothing for position; pure position-relative over-corrects
    # (lets an elite-among-D defenseman rank #1 overall). The 40/60 blend preserves the
    # forward-bias correction while keeping defensemen behind elite forwards on raw
    # offensive output.
    def blended_z(metric: str, mask: pd.Series | None = None) -> pd.Series:
        """40% global + 60% position-relative z-score. `mask` selects the reference
        pool (defaults to the ≥300 EV-min qualified pool; special-teams components
        pass their own ≥60-ST-min mask)."""
        ref_mask = qual_mask if mask is None else mask
        # Global z-score (vs. the reference pool)
        gref = df.loc[ref_mask, metric]
        gmu, gsd = gref.mean(), gref.std(ddof=0)
        z_global = (df[metric] - gmu) / gsd if gsd and not np.isnan(gsd) else pd.Series(0.0, index=df.index)
        pos_stats[(metric, "GLOBAL")] = (float(gmu), float(gsd), int(len(gref)))

        # Position-relative z-score (vs. own position group)
        z_pos = pd.Series(0.0, index=df.index)
        for label, grp in (("F", ~is_d), ("D", is_d)):
            ref = df.loc[grp & ref_mask, metric]
            mu, sd = ref.mean(), ref.std(ddof=0)
            pos_stats[(metric, label)] = (float(mu), float(sd), int(len(ref)))
            if sd and not np.isnan(sd):
                z_pos.loc[grp] = (df.loc[grp, metric] - mu) / sd

        return BLEND_GLOBAL * z_global + BLEND_POSREL * z_pos

    # Pre-compute the rate-based 5v5 sub-metric z-scores (EH-diagnostic R1).
    # Both are 40/60 blended (global + position-relative) for consistency. Used in
    # BOTH Component 3 (split off/def halves) AND Component 2 for defensemen.
    z_rel_xgf_off = blended_z("rel_xgf_off_60")     # offensive: positive = good
    z_rel_xga_def = blended_z("rel_xga_def_60")     # defensive: positive = good (already inverted)

    # Component 2 — individual shot generation. Forwards: ixG/60 (0.40) +
    # iHDCF/60 (0.30) + iGSAX/60 (0.30). EH-diagnostic R5 — defensemen use
    # 25/25/25/25 with the 4th slot being the rate-based on-ice xGA defensive
    # sub-metric, so shot-suppression D get position-appropriate credit inside
    # the Individual component instead of a structurally low score.
    c2_ixg = blended_z("ixg_60")
    c2_ihdcf = blended_z("ihdcf_60")
    c2_igsax = blended_z("igsax_60")
    individual_raw_F = (
        C2_SUBWEIGHTS["ixg_60"] * c2_ixg
        + C2_SUBWEIGHTS["ihdcf_60"] * c2_ihdcf
        + C2_SUBWEIGHTS["igsax_60"] * c2_igsax
    )
    # D-only: 25% each of the three offensive sub-metrics + 25% of defensive z.
    individual_raw_D = (
        0.25 * c2_ixg + 0.25 * c2_ihdcf + 0.25 * c2_igsax + 0.25 * z_rel_xga_def
    )
    individual_raw = individual_raw_F.where(~is_d, individual_raw_D)
    # Component 4 — playmaking (v4): 70/30 blend of
    #   sub-A = primary assists/60 (40/60 blended z, position-relative as before)
    #   sub-B = points/60 (goals + primary assists), GLOBAL z (spec: "z-scored globally")
    # Sub-B gives MacKinnon-type pure producers direct credit for their scoring,
    # which the assists-only metric under-credits.
    playmaking_subA = blended_z("primary_assists_60")
    # Global z of points/60 over the qualified pool (no position split per spec)
    ref_pts = df.loc[qual_mask, "points_60"]
    pts_mu, pts_sd = ref_pts.mean(), ref_pts.std(ddof=0)
    if pts_sd and not np.isnan(pts_sd):
        playmaking_subB = (df["points_60"] - pts_mu) / pts_sd
    else:
        playmaking_subB = pd.Series(0.0, index=df.index)
    pos_stats[("points_60", "GLOBAL")] = (float(pts_mu), float(pts_sd), int(len(ref_pts)))
    playmaking_raw = (
        C4_SUBWEIGHTS["primary_assists_60"] * playmaking_subA
        + C4_SUBWEIGHTS["points_60"] * playmaking_subB
    )
    # Keep the two sub-metric z-scores on the dataframe for the validation report.
    df["playmaking_subA_z"] = playmaking_subA.round(4)
    df["playmaking_subB_z"] = playmaking_subB.round(4)

    # Component 5 — power play (Fix 2): 50/50 blend of two sub-metrics, each
    # z-scored separately with the same blended position-relative approach over
    # the ≥60-PP-min pool.
    #   Sub-metric A = PP on-ice xGF/60 at 5on4 (unit shot-quality, unchanged).
    #   Sub-metric B = individual PP goals above expected (finishing skill).
    pp_mask = df["pp_qualified"] & qual_mask
    pp_z_a = blended_z("pp_xgf_60", mask=pp_mask)
    pp_z_b = blended_z("pp_goals_above_exp", mask=pp_mask)
    pp_blend = 0.5 * pp_z_a + 0.5 * pp_z_b
    # Keep the sub-metric z-scores for the validation report (gated to 0 below thresh).
    df["pp_subA_z"] = np.where(df["pp_qualified"], pp_z_a, 0.0)
    df["pp_subB_z"] = np.where(df["pp_qualified"], pp_z_b, 0.0)
    # Component 6 — penalty kill. EH-diagnostic R2 — 70/30 blend of:
    #   sub-A = on-ice xGA/60 at 4on5, inverted (lower allowed = better)
    #   sub-B = individual SH points/60 (G + A1 + A2), z-scored over PK-qualified pool
    # Both z-scores are blended within the ≥60-PK-min sub-pool.
    pk_mask = df["pk_qualified"] & qual_mask
    pk_subA = -blended_z("pk_xga_60", mask=pk_mask)
    pk_subB = blended_z("sh_points_60", mask=pk_mask)
    pk_raw = 0.70 * pk_subA + 0.30 * pk_subB
    df["pk_subA_z"] = np.where(df["pk_qualified"], pk_subA, 0.0)
    df["pk_subB_z"] = np.where(df["pk_qualified"], pk_subB, 0.0)

    # --- Final component normalization (global z-score over qualified pool) ---
    # C1/C3 are global z-scores of raw stats; C2/C4 re-z-score their blended raw
    # values to a clean mean-0/std-1 scale (uniform rescale — the blend is already baked in).
    qref = lambda s: s[qual_mask]
    df["rapm_component"] = zscore(df["rapm_raw"], qref(df["rapm_raw"]))
    df["individual_component"] = zscore(individual_raw, qref(individual_raw))
    # Component 3 (EH-diagnostic R1) — 50/50 blend of the rate-based offensive
    # and defensive sub-metric z-scores, then re-z-scored over the qualified pool.
    relxg_raw = 0.50 * z_rel_xgf_off + 0.50 * z_rel_xga_def
    df["relative_xgf_component"] = zscore(relxg_raw, qref(relxg_raw))
    # Diagnostic columns — keep the off/def halves separately on the dataframe.
    df["rel_xgf_off_z"] = z_rel_xgf_off.round(4)
    df["rel_xga_def_z"] = z_rel_xga_def.round(4)
    df["playmaking_component"] = zscore(playmaking_raw, qref(playmaking_raw))
    # Fix 3 — Components 2 & 3 require ≥200 EV min (4-season); below that → z 0.
    ev_qual = df["toi_minutes"] >= MIN_EV_MINUTES
    df["individual_component"] = np.where(ev_qual, df["individual_component"], 0.0)
    df["relative_xgf_component"] = np.where(ev_qual, df["relative_xgf_component"], 0.0)
    # Component 5 (Fix 2): final z = average of the two sub-metric z-scores; players
    # below the ≥60-PP-min threshold get exactly 0.
    df["power_play_component"] = np.where(df["pp_qualified"], pp_blend, 0.0)
    # Special-teams PK component: z-score over its qualified sub-pool only, then players
    # below the ST minute threshold get exactly 0 (neither help nor hurt their composite).
    pk_z = zscore(pk_raw, pk_raw[pk_mask])
    df["penalty_kill_component"] = np.where(df["pk_qualified"], pk_z, 0.0)

    # --- Soft cap: clip the four blended-z components to ±2.5 ---
    for c in ["individual_component", "playmaking_component",
              "power_play_component", "penalty_kill_component"]:
        df[c] = df[c].clip(-COMPONENT_CAP, COMPONENT_CAP)

    # --- Weighted composite (6 components, weights sum to 1.0) ---
    composite_raw = (
        WEIGHTS["rapm"] * df["rapm_component"]
        + WEIGHTS["individual"] * df["individual_component"]
        + WEIGHTS["relative_xgf"] * df["relative_xgf_component"]
        + WEIGHTS["playmaking"] * df["playmaking_component"]
        + WEIGHTS["power_play"] * df["power_play_component"]
        + WEIGHTS["penalty_kill"] * df["penalty_kill_component"]
    )
    df["composite_rating"] = composite_raw.round(4)

    # --- v4 WAR-equivalent rescale ---
    # Replacement level = 20th percentile composite_rating among players with
    # ≥500 EV minutes over the 4-season window. Subtract it so a replacement-
    # level player scores ≈ 0; then divide by the std of (raw − replacement)
    # over the same ≥500-EV-min pool and multiply by 0.5 so one unit ≈ ½ a win.
    repl_pool_mask = df["toi_minutes"] >= MIN_REPL_EV_MINUTES
    if repl_pool_mask.sum() < 50:
        raise RuntimeError(f"Replacement pool too small ({int(repl_pool_mask.sum())}) "
                           f"— relax MIN_REPL_EV_MINUTES.")
    replacement_level = float(composite_raw[repl_pool_mask].quantile(0.20))
    adjusted = composite_raw - replacement_level
    war_std = float(adjusted[repl_pool_mask].std(ddof=0)) or 1.0
    df["composite_war"] = (adjusted / war_std * WAR_UNIT).round(4)
    repl_meta = {
        "replacement_level_raw": round(replacement_level, 4),
        "war_std_raw": round(war_std, 4),
        "n_repl_pool": int(repl_pool_mask.sum()),
        "min_repl_ev_minutes": MIN_REPL_EV_MINUTES,
        "war_unit_wins": WAR_UNIT,
    }
    print(f"\n  Replacement level (20th pct of composite_rating in n={repl_meta['n_repl_pool']} "
          f"≥{MIN_REPL_EV_MINUTES}-EV-min pool) = {replacement_level:+.4f}", flush=True)
    print(f"  Adjusted-composite std over that pool = {war_std:.4f} "
          f"→ composite_war scale {WAR_UNIT}/unit", flush=True)

    for c in ["rapm_component", "individual_component", "relative_xgf_component",
              "playmaking_component", "power_play_component", "penalty_kill_component"]:
        df[c] = df[c].round(4)

    # Rank the QUALIFIED pool only — small-sample players have explosive per-60
    # rates (a 20-minute call-up with 2 high-danger shots scores a +10 z-score)
    # and must not contaminate the leaderboard. They stay in the CSV (flagged),
    # sorted after the qualified players. v4 sorts by composite_war (monotonic
    # transform of composite_rating, so the ordering is identical).
    qualified_df = (df[df["sample_size_flag"] == "ok"]
                    .sort_values("composite_war", ascending=False)
                    .reset_index(drop=True))
    small_df = (df[df["sample_size_flag"] == "small_sample"]
                .sort_values("composite_war", ascending=False)
                .reset_index(drop=True))
    df = pd.concat([qualified_df, small_df], ignore_index=True)

    # 2025-refresh — overwrite the MoneyPuck-most-recent-season team label with
    # the LIVE current team from the NHL API /v1/player/{id}/landing endpoint,
    # so the CSV reflects 2025-26 roster moves (e.g. Q. Hughes → MIN).
    print("\nFetching live current-team labels from NHL API …", flush=True)
    live_teams = refresh_current_teams(df["player_id"].astype(int).tolist())
    pre_team = df["team"].copy()
    df["team"] = df["player_id"].astype(int).map(live_teams).fillna(df["team"])
    n_changed = int((pre_team != df["team"]).sum())
    print(f"  Updated current team for {len(live_teams):,}/{len(df):,} players  "
          f"({n_changed:,} labels changed from MoneyPuck most-recent-season).", flush=True)

    out = df[["player_id", "player_name", "team", "position",
              "rapm_component", "individual_component", "relative_xgf_component",
              "playmaking_component", "power_play_component", "penalty_kill_component",
              "composite_rating", "composite_war", "expected_gp_share",
              "rel_xgf_off_z", "rel_xga_def_z",
              "toi_minutes", "sample_size_flag"]]

    qual_final = df[df["sample_size_flag"] == "ok"]
    meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": ("v4 (2025-refresh) — iGSAX in C2, points/60 sub-metric in C4, "
                    "reweighted components, WAR-equivalent rescale, NHL-API current-team "
                    "labels, 4-season window 2022-23→2025-26, 16-season RAPM"),
        "weights": WEIGHTS,
        "min_ev_minutes": MIN_EV_MINUTES,
        "min_repl_ev_minutes": MIN_REPL_EV_MINUTES,
        "min_st_minutes": MIN_ST_MINUTES,
        "rapm_window": "full 16-season model (2010-11 → 2025-26), 2025-refresh",
        "component_2to6_seasons": COMPONENT_SEASONS,
        "season_weights": SEASON_WEIGHTS,
        "n_players": int(len(df)),
        "n_qualified": n_qual,
        "n_pp_qualified": int(df["pp_qualified"].sum()),
        "n_pk_qualified": int(df["pk_qualified"].sum()),
        "components": {
            "rapm": "16-season RAPM total (year-weighted, goalie-adjusted), global z, weight 0.20",
            "individual": ("Forwards: ixG/60 (0.40) + iHDCF/60 (0.30) + iGSAX/60 (0.30). "
                           "Defensemen (EH-diagnostic R5): 0.25 each of ixG/60, iHDCF/60, "
                           "iGSAX/60, and on-ice xGA/60 (inverted, position-relative). "
                           "All blended z (40 global / 60 pos-rel); weight 0.33"),
            "relative_xgf": ("EH-diagnostic R1 — 50/50 split: 50% offensive (on-ice xGF/60 "
                             "vs off-ice team xGF/60, blended z) + 50% defensive (on-ice "
                             "xGA/60 vs off-ice team xGA/60, inverted, blended z). Weight 0.17"),
            "playmaking": ("70% primary-assists/60 (40/60 blended z) + 30% points/60 "
                           "global z, 3-season window, weight 0.18"),
            "power_play": ("50/50 blend of PP on-ice xGF/60 (5on4) and individual PP goals "
                           "above expected, 3-season window, blended z; weight 0.07; min 60 PP min"),
            "penalty_kill": ("EH-diagnostic R2 — 70/30 blend: 70% on-ice xGA/60 at 4on5 "
                             "inverted + 30% individual SH points/60 (G+A1+A2). Both blended z "
                             "over PK-qualified pool. Weight 0.05; min 60 PK min"),
        },
        "expected_gp_share": ("EH-diagnostic R4 — year-weighted average of games_played/82 "
                              "over the 3-season window (denominator = sum of season weights × "
                              "82 = 151.7); seasons not played count as 0 GP. For Phase 4 to "
                              "multiply against composite_war for availability-adjusted team strength."),
        "fixes_applied": {
            "fix1": "Components 2-6 aggregate only 2021-22 → 2024-25 (year-weighted)",
            "fix2": "Component 5 = 50/50 blend of on-ice PP xGF/60 + individual PP goals above expected",
            "fix3": "min 200 EV min for C2/C3, min 60 PP/PK min for C5/C6 (z=0 below)",
            "v4_improvement1": ("C2 replaces SH%-above-expected with iGSAX/60 (total "
                                "goals above expected per 60), captures finishing volume + skill"),
            "v4_improvement2": ("Reweighted: RAPM 0.20, Individual 0.33, RelxG% 0.17, "
                                "Playmaking 0.18 (with points/60 sub-metric), PP 0.07, PK 0.05"),
            "v4_improvement3": ("WAR-equivalent rescale: composite_war = (raw − 20th-pct of "
                                "≥500-EV-min pool) / std(adjusted) × 0.5 wins"),
        },
        "blend": {"global": BLEND_GLOBAL, "position_relative": BLEND_POSREL},
        "component2_subweights": C2_SUBWEIGHTS,
        "component4_subweights": C4_SUBWEIGHTS,
        "soft_cap": {"applies_to": ["individual", "playmaking", "power_play", "penalty_kill"],
                     "cap": COMPONENT_CAP},
        "position_group_stats": {
            f"{m}_{lbl}": {"mean": round(v[0], 4), "std": round(v[1], 4), "n": v[2]}
            for (m, lbl), v in pos_stats.items()
        },
        "war_rescale": repl_meta,
        "composite_mean_qualified": round(float(qual_final["composite_rating"].mean()), 4),
        "composite_std_qualified": round(float(qual_final["composite_rating"].std(ddof=0)), 4),
        "composite_war_mean_qualified": round(float(qual_final["composite_war"].mean()), 4),
        "composite_war_std_qualified": round(float(qual_final["composite_war"].std(ddof=0)), 4),
    }

    # --- Position-group + global stats (verification of blended z-scoring) ---
    print("\n" + "=" * 78, flush=True)
    print("POSITION-GROUP + GLOBAL STATS (all-situations rates) — blended z verification", flush=True)
    print("=" * 78, flush=True)
    for metric in ("ixg_60", "ihdcf_60", "igsax_60", "primary_assists_60", "points_60"):
        for lbl in ("GLOBAL", "F", "D"):
            if (metric, lbl) not in pos_stats:
                continue
            mu, sd, n = pos_stats[(metric, lbl)]
            print(f"  {metric:20s} {lbl:7s}:  mean={mu:.4f}  std={sd:.4f}  n={n}", flush=True)

    def _line(rank, r):
        return (f"{rank:>3} {r['player_name']:<21}{r['team']:>4}{r['position']:>4}"
                f"{r['composite_war']:>8.3f}{r['composite_rating']:>+8.3f}"
                f"{r['rapm_component']:>+6.2f}"
                f"{r['individual_component']:>+6.2f}{r['relative_xgf_component']:>+6.2f}"
                f"{r['playmaking_component']:>+6.2f}{r['power_play_component']:>+6.2f}"
                f"{r['penalty_kill_component']:>+6.2f}")

    hdr = (f"{'#':>3} {'Player':<21}{'Tm':>4}{'Pos':>4}{'WAR':>8}{'Comp':>8}"
           f"{'RAPM':>6}{'Indiv':>6}{'RelxG':>6}{'Play':>6}{'PP':>6}{'PK':>6}")

    # --- Validation report ---
    print("\n" + "=" * 96, flush=True)
    print("TOP 20 BY composite_war  (composite_rating shown alongside; 6 component z-scores)", flush=True)
    print("=" * 96, flush=True)
    print(hdr, flush=True)
    for i, r in out.head(20).iterrows():
        print(_line(i + 1, r), flush=True)

    # Bottom 15 of the QUALIFIED pool — Sanity Check 4 (replacement-level players low)
    n_qual_rows = int((out["sample_size_flag"] == "ok").sum())
    print("\n" + "=" * 84, flush=True)
    print(f"BOTTOM 15 of QUALIFIED pool (ranks {n_qual_rows-14}-{n_qual_rows} of {n_qual_rows})", flush=True)
    print("=" * 84, flush=True)
    print(hdr, flush=True)
    for i, r in out[out["sample_size_flag"] == "ok"].head(n_qual_rows).tail(15).iterrows():
        print(_line(i + 1, r), flush=True)

    print("\n" + "=" * 96, flush=True)
    print("REFERENCE PLAYERS  (WAR / comp / RAPM / Indiv / RelxG / Play / PP / PK)", flush=True)
    print("=" * 96, flush=True)
    ref_players = ["Connor McDavid", "Auston Matthews", "Leon Draisaitl", "Nathan MacKinnon",
                   "Cale Makar", "Adam Fox", "Patrice Bergeron", "Victor Hedman",
                   "Sidney Crosby", "Mikko Rantanen", "Nikita Kucherov",
                   "Alex Ovechkin", "Steven Stamkos", "Joe Thornton"]
    for name in ref_players:
        sub = out[out["player_name"] == name]
        if sub.empty:
            print(f"  {name:<22} NOT FOUND in dataset", flush=True)
            continue
        row = sub.iloc[0]
        rank = int(sub.index[0]) + 1
        print(f"  {name:<20} #{rank:<4} WAR={row['composite_war']:+.3f}  "
              f"comp={row['composite_rating']:+.3f}  "
              f"[RAPM={row['rapm_component']:+.2f}  Indiv={row['individual_component']:+.2f}  "
              f"RelxG={row['relative_xgf_component']:+.2f}  Play={row['playmaking_component']:+.2f}  "
              f"PP={row['power_play_component']:+.2f}  PK={row['penalty_kill_component']:+.2f}]",
              flush=True)

    # --- v4 iGSAX diagnostic — Draisaitl + Matthews must be clearly positive ---
    print("\n" + "=" * 96, flush=True)
    print("iGSAX DIAGNOSTIC (Improvement 1) — Draisaitl & Matthews must show high iGSAX/60", flush=True)
    print("=" * 96, flush=True)
    for name in ("Leon Draisaitl", "Auston Matthews", "Connor McDavid",
                 "Nathan MacKinnon", "David Pastrnak", "Sam Reinhart"):
        s = df[df["player_name"] == name]
        if s.empty:
            print(f"  {name:<22} not found in dataset", flush=True)
            continue
        r = s.iloc[0]
        print(f"  {name:<20}  iGSAX/60={r['igsax_60']:+.4f}  "
              f"iGSAX_total(y-w)={r['igsax_total']:+.2f}  "
              f"ixG/60={r['ixg_60']:.4f}  C2_z={r['individual_component']:+.3f}",
              flush=True)

    # --- v4 benchmark check ---
    print("\n" + "=" * 96, flush=True)
    print("BENCHMARK CHECK (v4 spec)", flush=True)
    print("=" * 96, flush=True)
    bench = [("Connor McDavid", "top 3"),
             ("Leon Draisaitl", "top 15"),
             ("Nathan MacKinnon", "top 15"),
             ("Cale Makar", "top 25")]
    bench_pass = True
    for name, gate in bench:
        s = out[out["player_name"] == name]
        if s.empty:
            print(f"  ✗ {name:<22} NOT FOUND  (need {gate})", flush=True)
            bench_pass = False
            continue
        rank = int(s.index[0]) + 1
        gate_n = int(gate.split()[-1])
        ok = rank <= gate_n
        if not ok:
            bench_pass = False
        print(f"  {'✓' if ok else '✗'} {name:<22} #{rank:<4}  (need {gate})", flush=True)
    print(f"  → BENCHMARKS {'PASS' if bench_pass else 'FAIL'}", flush=True)

    # --- Power-play component: sub-metric A / B breakdown (Fix 2 verification) ---
    print("\n" + "=" * 84, flush=True)
    print("POWER PLAY COMPONENT (Fix 2) — top 5 + Draisaitl + Ovechkin", flush=True)
    print("  sub-A = on-ice PP xGF/60 z   sub-B = individual PP goals-above-expected z", flush=True)
    print("=" * 84, flush=True)
    ppq = df[df["sample_size_flag"] == "ok"].sort_values(
        "power_play_component", ascending=False).reset_index(drop=True)
    for i, r in ppq.head(5).iterrows():
        print(f"  {i+1:>3} {r['player_name']:<22}{r['team']:>4}  "
              f"PP z={r['power_play_component']:+.3f}  "
              f"(sub-A={r['pp_subA_z']:+.3f}  sub-B={r['pp_subB_z']:+.3f})", flush=True)
    for name in ("Leon Draisaitl", "Alex Ovechkin"):
        s = ppq[ppq["player_name"] == name]
        if not s.empty:
            r = s.iloc[0]
            print(f"  {name} PP-component rank #{int(s.index[0]) + 1}: "
                  f"PP z={r['power_play_component']:+.3f}  "
                  f"(sub-A={r['pp_subA_z']:+.3f}  sub-B={r['pp_subB_z']:+.3f})", flush=True)

    # --- Save (only when --save passed) — archive the v3 CSV first ---
    if save:
        if not bench_pass:
            print("\n⚠ NOTE: v4 benchmark check failed (e.g. MacKinnon outside top 15). "
                  "User explicitly approved save anyway — proceeding.", flush=True)
        if OUT_CSV.exists():
            archive = MODEL_DIR / PRIOR_ARCHIVE_NAME
            OUT_CSV.rename(archive)
            print(f"\nArchived previous v3 → {archive}", flush=True)
        out.to_csv(OUT_CSV, index=False)
        OUT_META.write_text(json.dumps(meta, indent=2))
        print(f"Saved → {OUT_CSV}", flush=True)
        print(f"Saved → {OUT_META}", flush=True)
    else:
        dry = MODEL_DIR / "composite_ratings_dryrun.csv"
        out.to_csv(dry, index=False)
        print(f"\n[dry-run] Full ranked table written → {dry} (not the live CSV).", flush=True)
        print("[dry-run] Re-run with --save to persist results to composite_ratings.csv.", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Write composite_ratings.csv + meta. Default: dry-run (report only).")
    args = parser.parse_args()
    main(save=args.save)
