"""
Phase 5.1 calibration — empirical ALPHA sweep.

ALPHA is the team-strength elasticity in the Phase 4 game engine:
    xGF_H = LEAGUE_GPG + ALPHA * (ts[H] - ts[A]) + HOME_ICE

The original 0.50 compressed team goal totals relative to the actual 25-26
season (Edmonton: 282 actual vs ~252 simulated). Players on strong offenses
underprojected by 10-15 points across the board as a result.

This script sweeps ALPHA from 0.45 to 0.80 in 0.05 steps, simulating the
actual 1,312-game 25-26 schedule N times under each value with current
team_strength values, and computes the average absolute error in season
GF per team vs the actual MoneyPuck 25-26 totals. The minimum-MAE ALPHA
is the empirically calibrated value.

Run from project root:
    python3 model/calibrate_alpha.py
"""
from __future__ import annotations
import json
import random
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

import simulate as sim_engine
import simulate_season as ss

ALPHAS = [round(0.20 + 0.05 * i, 2) for i in range(13)]   # 0.20 … 0.80
N_SIMS = 250
MP_TEAMS_URL = ("https://moneypuck.com/moneypuck/playerData/seasonSummary/"
                "2025/regular/teams.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUT_JSON = MODEL_DIR / "alpha_sweep_results.json"


def fetch_actual_team_gf() -> dict[str, float]:
    """Pull 25-26 actual team GF per game played from MoneyPuck (situation=all)."""
    r = requests.get(MP_TEAMS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[df["situation"] == "all"].copy()
    df["goalsFor"] = pd.to_numeric(df["goalsFor"], errors="coerce").fillna(0)
    df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce").fillna(0)
    out = {}
    for r in df.itertuples(index=False):
        # Normalise to 82-game equivalent in case GP != 82 anywhere.
        gp = float(r.games_played) or 82.0
        out[str(r.team)] = float(r.goalsFor) * 82.0 / gp
    return out


def run_season_once(schedule, ts, rng) -> dict[str, dict]:
    """One full RS pass. Returns {team: {gf, ga, gp}} dict."""
    rec = {t: {"gf": 0, "ga": 0, "gp": 0}
           for t in sim_engine.TEAM_TO_DIVISION}
    for (home, away) in schedule:
        out = sim_engine.simulate_game(home, away, ts, rng)
        rec[home]["gf"] += out["home_score"]
        rec[home]["ga"] += out["away_score"]
        rec[home]["gp"] += 1
        rec[away]["gf"] += out["away_score"]
        rec[away]["ga"] += out["home_score"]
        rec[away]["gp"] += 1
    return rec


def avg_gf_for_alpha(alpha: float, schedule, ts, n_sims: int,
                      seed: int = 42) -> dict[str, float]:
    """Set ALPHA, run n_sims full RS seasons, return per-team avg GF/season."""
    sim_engine.ALPHA = alpha
    rng = random.Random(seed)
    totals = {t: 0.0 for t in sim_engine.TEAM_TO_DIVISION}
    for _ in range(n_sims):
        rec = run_season_once(schedule, ts, rng)
        for t, r in rec.items():
            totals[t] += r["gf"]
    return {t: v / n_sims for t, v in totals.items()}


def main():
    t0 = time.time()
    print("Loading inputs …", flush=True)
    ts = sim_engine.load_team_strengths()
    actual = fetch_actual_team_gf()
    sched_raw = ss.fetch_schedule()
    schedule = [(g["home"], g["away"]) for g in sched_raw]
    print(f"  team_strength: {len(ts)} teams")
    print(f"  actual GF:     {len(actual)} teams")
    print(f"  schedule:      {len(schedule)} games")

    original_alpha = sim_engine.ALPHA
    results = []
    for alpha in ALPHAS:
        ta = time.time()
        sim_gf = avg_gf_for_alpha(alpha, schedule, ts, N_SIMS, seed=42)
        # Compare across teams with actual data
        diffs = []
        per_team = []
        for team, sim_v in sim_gf.items():
            act_v = actual.get(team)
            if act_v is None:
                continue
            diffs.append(abs(sim_v - act_v))
            per_team.append({
                "team": team,
                "sim_gf": round(sim_v, 1),
                "actual_gf": round(act_v, 1),
                "delta": round(sim_v - act_v, 1),
            })
        mae = float(np.mean(diffs)) if diffs else float("nan")
        rmse = float(np.sqrt(np.mean(np.square(diffs)))) if diffs else float("nan")
        max_team_gf = max(sim_gf.values())
        max_team = max(sim_gf, key=sim_gf.get)
        elapsed = time.time() - ta
        print(f"  ALPHA={alpha:.2f}   MAE={mae:5.2f}   RMSE={rmse:5.2f}   "
              f"max_team_GF={max_team_gf:.1f} ({max_team})   "
              f"[{elapsed:.1f}s]", flush=True)
        results.append({
            "alpha": alpha,
            "mae_team_gf": mae,
            "rmse_team_gf": rmse,
            "max_team_gf": max_team_gf,
            "max_team": max_team,
            "per_team": per_team,
        })

    # Restore original ALPHA
    sim_engine.ALPHA = original_alpha

    best = min(results, key=lambda r: r["mae_team_gf"])
    print("\n" + "=" * 60)
    print(f"BEST ALPHA: {best['alpha']:.2f}   MAE={best['mae_team_gf']:.2f}  "
          f"RMSE={best['rmse_team_gf']:.2f}")
    print("=" * 60)

    OUT_JSON.write_text(json.dumps({
        "n_sims": N_SIMS,
        "alphas_swept": ALPHAS,
        "best_alpha": best["alpha"],
        "best_mae": best["mae_team_gf"],
        "best_rmse": best["rmse_team_gf"],
        "results": results,
        "actual_team_gf": actual,
    }, indent=2))
    print(f"Saved sweep results → {OUT_JSON}")
    print(f"Total compute: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
