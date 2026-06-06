"""
xG model training pipeline (Phase 1) — v3, 15-season window.

Pulls 15 seasons of NHL play-by-play (2010-11 → 2024-25), extracts shot
features, and trains an XGBoost classifier (isotonic-calibrated) to predict
goal vs. no-goal.

Per-season caching: each season's shots are cached to
model/season_cache/shots_<SEASON>.parquet. On every run each season is
checked individually — if its parquet already exists it is loaded and the
fetch is skipped. Only missing seasons are pulled from the NHL API.

Run from the project root:
    python3 model/train_xg.py            # fetch missing seasons, then train
    python3 model/train_xg.py --fetch-only   # fetch missing seasons, no train

Outputs:
    model/shots_dataset.parquet    — all 15 seasons of shots (model schema)
    model/shots_augmented.parquet  — same + abs_secs + xg (consumed by RAPM)
    model/home_id_by_game.json     — {game_id: home_team_id} for all games
    model/xg_model.pkl, model/xg_model_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# XGBoost imported lazily — install requires libomp on macOS
import joblib

NHL_API = "https://api-web.nhle.com/v1"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# 16 seasons — 2010-11 through 2025-26 (regular season + playoffs each).
SEASONS = [
    "20102011", "20112012", "20122013", "20132014", "20142015",
    "20152016", "20162017", "20172018", "20182019", "20192020",
    "20202021", "20212022", "20222023", "20232024", "20242025",
    "20252026",
]

# Year-weighting (2025-refresh): decay-slot anchored at 2025-26 = 1.00. The decay
# sequence is shifted one slot older vs. the previous v4 anchor; COVID/lockout
# games-played fractions stay attached to their actual calendar seasons
# (2020-21 56g, 2019-20 ~69.5g, 2012-13 48g). New floor weight (0.04) added for
# the oldest slot since the window is now 16 seasons.
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
DEFAULT_WEIGHT = 0.05


def season_str_from_game_id(game_id: int) -> str:
    """Game IDs are {startYear}02{NNNN} → return '{startYear}{startYear+1}'."""
    start = int(game_id) // 1_000_000
    return f"{start}{start + 1}"

# Shot event types we keep (goals, on-net saves, missed) — exclude blocked
SHOT_EVENT_TYPES = {"goal", "shot-on-goal", "missed-shot"}

# Shot type encoding (NHL standard `shotType` values)
SHOT_TYPE_MAP = {
    "wrist": 1,
    "snap": 2,
    "slap": 3,
    "backhand": 4,
    "tip-in": 5,
    "deflected": 6,
    "wrap-around": 7,
    "poke": 8,
    "bat": 9,
    "cradle": 10,
    "between-legs": 11,
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
CACHE_DIR = MODEL_DIR / "season_cache"
DATA_CACHE_PATH = MODEL_DIR / "shots_dataset.parquet"
AUGMENTED_PATH = MODEL_DIR / "shots_augmented.parquet"
HOME_ID_PATH = MODEL_DIR / "home_id_by_game.json"
MODEL_PATH = MODEL_DIR / "xg_model.pkl"
META_PATH = MODEL_DIR / "xg_model_meta.json"

# Columns persisted to the per-season cache (model schema + abs_secs).
SHOT_CACHE_COLS = [
    "x_norm", "y_norm", "distance_from_net", "angle_from_center", "shot_type_id",
    "is_rush", "is_rebound", "time_since_last_shot", "period", "score_state",
    "is_home", "is_goal", "shooter_id", "on_goal", "game_id", "abs_secs",
]

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


# ---------------------------------------------------------------------------
# Game ID enumeration
# ---------------------------------------------------------------------------
def regular_season_game_ids(season: str) -> list[int]:
    """Regular-season game IDs for a season: {year}02{0001..1312}.
    32 teams × 82 / 2 = 1312 is the modern maximum; older 30/31-team seasons
    and lockout seasons simply 404 on the unused tail (handled as no-data)."""
    year = season[:4]
    return [int(f"{year}02{n:04d}") for n in range(1, 1313)]


def playoff_game_ids(season: str) -> list[int]:
    """Playoff game IDs for a season: {year}030{round}{matchup}{game}."""
    year = season[:4]
    ids = []
    for rnd in range(1, 5):
        n_matchups = {1: 8, 2: 4, 3: 2, 4: 1}[rnd]
        for m in range(1, n_matchups + 1):
            for g in range(1, 8):
                ids.append(int(f"{year}030{rnd}{m}{g}"))
    return ids


def season_game_ids(season: str) -> list[int]:
    return regular_season_game_ids(season) + playoff_game_ids(season)


# ---------------------------------------------------------------------------
# Play-by-play fetching
# ---------------------------------------------------------------------------
def fetch_pbp(game_id: int, max_retries: int = 2) -> dict | None:
    """GET /v1/gamecenter/{id}/play-by-play. Return None on 404 or persistent failure."""
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


def parse_time_in_period(s: str) -> int:
    """Parse 'MM:SS' time-in-period string to seconds. Defensive against bad input."""
    if not s or ":" not in s:
        return 0
    try:
        m, sec = s.split(":")
        return int(m) * 60 + int(sec)
    except Exception:
        return 0


def absolute_seconds(period: int, time_in_period: str) -> int:
    """Return absolute game seconds: (period - 1) × 1200 + (MM:SS in period)."""
    return (max(period, 1) - 1) * 1200 + parse_time_in_period(time_in_period)


# ---------------------------------------------------------------------------
# Shot extraction with running-state features
# ---------------------------------------------------------------------------
def situation_code_indicates_empty_net(sit_code: str, shooter_is_home: bool) -> bool:
    """
    NHL situationCode is a 4-char string: '[awayGoalie][awaySkaters][homeSkaters][homeGoalie]'.
    1 = goalie present, 0 = pulled. Empty-net for the shooter = the OPPOSING goalie is pulled.
    """
    if not sit_code or len(sit_code) != 4:
        return False
    away_goalie, _, _, home_goalie = sit_code[0], sit_code[1], sit_code[2], sit_code[3]
    if shooter_is_home:
        return away_goalie == "0"
    return home_goalie == "0"


def extract_shots_from_game(pbp: dict) -> list[dict]:
    """Walk a game's plays in order, extract shots with running-state features."""
    if not pbp:
        return []
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    plays = pbp.get("plays") or []

    home_score = 0
    away_score = 0
    last_event_abs_secs = None
    last_event_zone = None
    last_event_type = None
    last_event_team_id = None
    last_shot_abs_secs = None

    out = []
    for p in plays:
        ev_type = p.get("typeDescKey")
        period = (p.get("periodDescriptor") or {}).get("number") or 1
        time_str = p.get("timeInPeriod") or "00:00"
        abs_secs = absolute_seconds(period, time_str)
        details = p.get("details") or {}
        event_team_id = details.get("eventOwnerTeamId")
        zone = details.get("zoneCode")

        if ev_type in SHOT_EVENT_TYPES:
            shooter_team_id = event_team_id
            if shooter_team_id is None:
                last_event_abs_secs = abs_secs
                last_event_zone = zone
                last_event_type = ev_type
                last_event_team_id = event_team_id
                if ev_type == "shot-on-goal" or ev_type == "goal":
                    last_shot_abs_secs = abs_secs
                continue

            shooter_is_home = (shooter_team_id == home_id)

            sit_code = p.get("situationCode") or ""
            if situation_code_indicates_empty_net(sit_code, shooter_is_home):
                last_event_abs_secs = abs_secs
                last_event_zone = zone
                last_event_type = ev_type
                last_event_team_id = event_team_id
                last_shot_abs_secs = abs_secs
                continue

            x = details.get("xCoord")
            y = details.get("yCoord")
            if x is None or y is None:
                last_event_abs_secs = abs_secs
                last_event_zone = zone
                last_event_type = ev_type
                last_event_team_id = event_team_id
                last_shot_abs_secs = abs_secs
                continue

            if x < 0:
                x_norm = -x
                y_norm = -y
            else:
                x_norm = x
                y_norm = y

            distance = math.hypot(89 - x_norm, y_norm)
            denom = (89 - x_norm)
            if denom == 0:
                angle = math.pi / 2 if y_norm != 0 else 0
            else:
                angle = math.atan2(abs(y_norm), denom)

            shot_type = (details.get("shotType") or "").lower()
            shot_type_id = SHOT_TYPE_MAP.get(shot_type, 0)

            is_rush = 0
            if last_event_abs_secs is not None and last_event_zone is not None:
                gap = abs_secs - last_event_abs_secs
                if gap <= 4:
                    if last_event_team_id == shooter_team_id:
                        prior_zone_for_shooter = last_event_zone
                    else:
                        prior_zone_for_shooter = {"O": "D", "D": "O", "N": "N"}.get(last_event_zone, last_event_zone)
                    if prior_zone_for_shooter in ("N", "D"):
                        is_rush = 1

            is_rebound = 0
            if last_event_abs_secs is not None and last_event_type in SHOT_EVENT_TYPES:
                gap = abs_secs - last_event_abs_secs
                if gap <= 3 and last_event_team_id == shooter_team_id:
                    is_rebound = 1

            tsl = (abs_secs - last_shot_abs_secs) if last_shot_abs_secs is not None else 9999

            if shooter_is_home:
                score_state = home_score - away_score
            else:
                score_state = away_score - home_score
            score_state = max(-3, min(3, score_state))

            shooter_id = details.get("shootingPlayerId") or details.get("scoringPlayerId")

            out.append({
                "x_norm": float(x_norm),
                "y_norm": float(y_norm),
                "distance_from_net": float(distance),
                "angle_from_center": float(angle),
                "shot_type_id": int(shot_type_id),
                "is_rush": int(is_rush),
                "is_rebound": int(is_rebound),
                "time_since_last_shot": float(min(tsl, 600)),
                "period": int(period),
                "score_state": int(score_state),
                "is_home": int(shooter_is_home),
                "is_goal": 1 if ev_type == "goal" else 0,
                "shooter_id": int(shooter_id) if shooter_id else 0,
                "on_goal": 1 if ev_type in ("goal", "shot-on-goal") else 0,
                "abs_secs": int(abs_secs),
            })

            if ev_type == "goal":
                if shooter_is_home:
                    home_score += 1
                else:
                    away_score += 1

            last_shot_abs_secs = abs_secs

        last_event_abs_secs = abs_secs
        last_event_zone = zone
        last_event_type = ev_type
        last_event_team_id = event_team_id

    return out


# ---------------------------------------------------------------------------
# Per-season pull
# ---------------------------------------------------------------------------
def fetch_season(season: str, workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """Pull every game of one season. Returns (shots_df, {game_id: home_id})."""
    game_ids = season_game_ids(season)
    print(f"  [{season}] pulling up to {len(game_ids)} games with {workers} workers …", flush=True)

    all_rows = []
    home_ids = {}
    completed = 0
    skipped = 0
    start = time.time()

    def _process(gid):
        pbp = fetch_pbp(gid)
        if not pbp:
            return gid, None, None
        home_id = (pbp.get("homeTeam") or {}).get("id")
        return gid, extract_shots_from_game(pbp), home_id

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process, gid): gid for gid in game_ids}
        for fut in as_completed(futures):
            try:
                gid, shots, home_id = fut.result()
            except Exception:
                gid, shots, home_id = None, None, None
            completed += 1
            if not shots:
                skipped += 1
            else:
                for s in shots:
                    s["game_id"] = gid
                all_rows.extend(shots)
                if home_id:
                    home_ids[gid] = home_id
            if completed % 500 == 0:
                elapsed = time.time() - start
                rate = completed / max(elapsed, 0.1)
                eta = (len(game_ids) - completed) / max(rate, 0.1)
                print(f"    [{season}] {completed}/{len(game_ids)} games — "
                      f"{len(all_rows):,} shots — {rate:.1f} g/s — ETA {eta:.0f}s", flush=True)

    elapsed = time.time() - start
    n_games = len(game_ids) - skipped
    print(f"  [{season}] done — {n_games} games with data, {len(all_rows):,} shots "
          f"in {elapsed:.0f}s", flush=True)
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df[SHOT_CACHE_COLS]
    return df, home_ids


def load_or_fetch_all_seasons(workers: int = 8) -> tuple[pd.DataFrame, dict]:
    """For each of the 15 seasons: load the cached per-season parquet if it
    exists, otherwise fetch it from the NHL API and cache it. Returns the
    concatenated shot DataFrame and the merged {game_id: home_id} dict."""
    CACHE_DIR.mkdir(exist_ok=True)
    season_frames = []
    home_id_by_game = {}

    for season in SEASONS:
        shots_path = CACHE_DIR / f"shots_{season}.parquet"
        home_path = CACHE_DIR / f"home_ids_{season}.json"
        if shots_path.exists():
            df = pd.read_parquet(shots_path)
            print(f"  [{season}] cached — {len(df):,} shots (skipping fetch).", flush=True)
            if home_path.exists():
                hids = {int(k): int(v) for k, v in json.loads(home_path.read_text()).items()}
                home_id_by_game.update(hids)
        else:
            df, hids = fetch_season(season, workers=workers)
            if df.empty:
                print(f"  [{season}] WARNING: no shots pulled — skipping season.", flush=True)
                continue
            df.to_parquet(shots_path, index=False)
            home_path.write_text(json.dumps({str(k): int(v) for k, v in hids.items()}))
            home_id_by_game.update({int(k): int(v) for k, v in hids.items()})
            print(f"  [{season}] cached → {shots_path.name}", flush=True)
        season_frames.append(df)

    combined = pd.concat(season_frames, ignore_index=True)
    return combined, home_id_by_game


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame) -> dict:
    print("\nTraining XGBoost classifier with isotonic calibration …", flush=True)
    print(f"Total shots: {len(df):,}, goal rate: {df['is_goal'].mean()*100:.2f}%", flush=True)

    X = df[FEATURE_COLS].astype("float32")
    y = df["is_goal"].astype("int32")

    # Year-weighting: shots from recent seasons get higher training sample weight.
    season_of = df["game_id"].map(season_str_from_game_id)
    season_w = season_of.map(lambda s: SEASON_WEIGHTS.get(s, DEFAULT_WEIGHT)).astype("float32")
    print("  Year-weighting (sample_weight) by season:", flush=True)
    for s in SEASONS:
        n = int((season_of == s).sum())
        print(f"    {s}: weight {SEASON_WEIGHTS[s]:.4f}  ({n:,} shots)", flush=True)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, season_w, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_tr):,}  Test: {len(X_te):,}", flush=True)

    import xgboost as xgb
    from sklearn.calibration import CalibratedClassifierCV

    pos = int(y_tr.sum())
    neg = int(len(y_tr) - pos)
    spw = neg / max(pos, 1)

    def _make_base():
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.07,
            min_child_weight=10,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    base_clf = _make_base()
    t0 = time.time()
    base_clf.fit(X_tr, y_tr, sample_weight=w_tr)
    base_proba = base_clf.predict_proba(X_te)[:, 1]
    base_auc = float(roc_auc_score(y_te, base_proba))
    base_ll = float(log_loss(y_te, base_proba))
    base_mean = float(base_proba.mean())
    print(f"  Base (uncalibrated): AUC={base_auc:.4f}  logloss={base_ll:.4f}  "
          f"mean prob={base_mean:.4f} (inflated by scale_pos_weight)", flush=True)

    calibrated = CalibratedClassifierCV(estimator=_make_base(), method="isotonic", cv=5)
    calibrated.fit(X_tr, y_tr, sample_weight=w_tr)
    train_time = time.time() - t0

    cal_proba = calibrated.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, cal_proba))
    ll = float(log_loss(y_te, cal_proba))

    full_proba = calibrated.predict_proba(X)[:, 1]
    mean_pred = float(full_proba.mean())
    goal_rate = float(y.mean())

    importances = dict(zip(FEATURE_COLS, [float(v) for v in base_clf.feature_importances_]))
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "v4 (2025-refresh; 16-season window, 2010-11 → 2025-26)",
        "seasons": SEASONS,
        "season_weights": SEASON_WEIGHTS,
        "year_weighting": "applied as XGBoost sample_weight (recent seasons weighted higher)",
        "n_shots_total": int(len(df)),
        "n_shots_train": int(len(X_tr)),
        "n_shots_test": int(len(X_te)),
        "goal_rate": goal_rate,
        "auc_roc": auc,
        "log_loss": ll,
        "base_auc_roc_uncalibrated": base_auc,
        "base_log_loss_uncalibrated": base_ll,
        "base_mean_predicted_prob_uncalibrated": base_mean,
        "calibration_method": "isotonic",
        "calibration_cv_folds": 5,
        "mean_predicted_probability": mean_pred,
        "scale_pos_weight": spw,
        "train_time_seconds": train_time,
        "features": FEATURE_COLS,
        "feature_importances": importances,
        "shot_event_types_kept": sorted(SHOT_EVENT_TYPES),
        "empty_net_excluded": True,
        "side_normalized_to_x_pos": True,
    }

    print(f"\nResults (isotonic-calibrated model):", flush=True)
    print(f"  AUC-ROC : {auc:.4f}  (uncalibrated base: {base_auc:.4f})", flush=True)
    print(f"  Log loss: {ll:.4f}  (uncalibrated base: {base_ll:.4f})", flush=True)
    print(f"  Mean predicted probability (all {len(df):,} shots): {mean_pred:.4f}", flush=True)
    print(f"  True goal rate:                                    {goal_rate:.4f}", flush=True)
    diff_pct = abs(mean_pred - goal_rate) / goal_rate * 100
    status = "OK — calibrated" if diff_pct < 5 else f"WARNING — off by {diff_pct:.1f}%"
    print(f"  Calibration check: {status}", flush=True)
    print(f"  Train time: {train_time:.1f}s", flush=True)
    print(f"\nFeature importances (top 5):", flush=True)
    for k, v in list(importances.items())[:5]:
        print(f"  {k:25s} {v:.4f}", flush=True)

    print(f"\nSaving calibrated model → {MODEL_PATH}", flush=True)
    joblib.dump(calibrated, MODEL_PATH)
    print(f"Saving meta             → {META_PATH}", flush=True)
    META_PATH.write_text(json.dumps(meta, indent=2))

    return meta, calibrated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch-only", action="store_true",
                        help="Fetch any missing seasons into the per-season cache, then exit.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel HTTP workers for the data pull.")
    args = parser.parse_args()

    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading / fetching 15 seasons of shot data (per-season cache) …", flush=True)
    t0 = time.time()
    df, home_id_by_game = load_or_fetch_all_seasons(workers=args.workers)
    print(f"\nTotal: {len(df):,} shots across {df['game_id'].nunique():,} games "
          f"({len(SEASONS)} seasons) — {time.time()-t0:.0f}s.", flush=True)

    # Persist the combined model-schema dataset.
    df[[c for c in SHOT_CACHE_COLS if c != "abs_secs"]].to_parquet(DATA_CACHE_PATH, index=False)
    print(f"Cached combined dataset → {DATA_CACHE_PATH}", flush=True)

    if args.fetch_only:
        print("--fetch-only: seasons cached, skipping training.", flush=True)
        return

    meta, model = train(df)

    # Build shots_augmented.parquet (abs_secs + xg) for the RAPM pipeline so it
    # never needs to re-fetch play-by-play.
    print(f"\nScoring all shots → {AUGMENTED_PATH}", flush=True)
    aug = df.copy()
    aug["xg"] = model.predict_proba(aug[FEATURE_COLS].astype("float32"))[:, 1]
    aug.to_parquet(AUGMENTED_PATH, index=False)
    HOME_ID_PATH.write_text(json.dumps({str(k): int(v) for k, v in home_id_by_game.items()}))
    print(f"  shots_augmented: {len(aug):,} shots (abs_secs + xg).", flush=True)
    print(f"  home_id_by_game: {len(home_id_by_game):,} games → {HOME_ID_PATH}", flush=True)


if __name__ == "__main__":
    main()
