"""
Phase 5.1 — ALPHA empirical re-calibration (v2, 2026-06-05).

The original ALPHA sweep (model/calibrate_alpha.py) optimised team-GF MAE
only, and was run BEFORE both:
  - Fix 1 (TOI-based depth-chart sort widened the team_strength spread)
  - Assist-rate retune (which lifted A1+A2 across the board)

The post-fixes PPG diagnostic shows a persistent top-vs-bottom team
asymmetry (top-10 strongest teams' players at −0.035 PPG, bottom-10 at
−0.007 PPG) that suggests ALPHA = 0.45 is now slightly low — top-team
GF is over-compressed.

This script sweeps ALPHA 0.40-0.65 in 0.025 steps. For each candidate
it runs the full 32-team × 250-sim per-player calibration (~45s per
step), computes:
  - team GF MAE vs actual 25-26 MoneyPuck GF
  - league-wide mean Δ PPG
  - top-10 strongest teams' mean Δ PPG
  - bottom-10 weakest teams' mean Δ PPG

Selection objective:
  Minimise GF MAE, subject to:
    - |top10 mean Δ PPG − bottom10 mean Δ PPG| ≤ 0.015
    - |league mean Δ PPG| ≤ 0.015

The actual baseline is built once and reused. The team_strength table is
unchanged across the sweep (it doesn't depend on ALPHA).

Run from project root:
    python3 model/calibrate_alpha_v2.py
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
import build_team_strength as bts
import calibration_check as cc

ALPHAS = [round(0.40 + 0.025 * i, 3) for i in range(11)]   # 0.400 … 0.650
N_SIMS = 250
MIN_GP_FILTER = 41
SIM_GP = 82

OUT_JSON = MODEL_DIR / "alpha_sweep_v2_results.json"

MP_TEAMS_URL = ("https://moneypuck.com/moneypuck/playerData/seasonSummary/"
                "2025/regular/teams.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_actual_team_gf() -> dict[str, float]:
    """82-game-normalised actual team GF from MoneyPuck 25-26."""
    r = requests.get(MP_TEAMS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df = df[df["situation"] == "all"].copy()
    df["goalsFor"]     = pd.to_numeric(df["goalsFor"], errors="coerce").fillna(0)
    df["games_played"] = pd.to_numeric(df["games_played"], errors="coerce").fillna(0)
    out = {}
    for r in df.itertuples(index=False):
        gp = float(r.games_played) or 82.0
        out[str(r.team)] = float(r.goalsFor) * 82.0 / gp
    return out


def run_sweep_step(alpha: float, comp_lookup, toi_lookup, ts_lookup,
                    actual_team_gf, actual_baseline, team_strength_df):
    """One ALPHA value: run 32-team × 250-sim simulation, compute metrics."""
    # Patch ALPHA in the engine
    sim_engine.ALPHA = alpha
    t0 = time.time()

    sim_team_gf = {}
    sim_rows = []
    for team in cc.TEAMS:
        lineup = cc._build_team_lineup(team, comp_lookup, toi_lookup)
        team_ts = ts_lookup.get(team, 0.0)
        res = ss.monte_carlo_custom_season(team, lineup, team_ts,
                                            n_sims=N_SIMS, seed=42)
        rs = res["regular_season"]
        sim_team_gf[team] = rs["avg_gf"]
        for p in res["player_stats"]:
            if p["player_id"] is None:
                continue
            sim_rows.append({
                "team":         team,
                "player_id":    int(p["player_id"]),
                "position":     p["position"],
                "slot_kind":    p["slot_kind"],
                "slot_idx":     p["slot_idx"],
                "sim_points":   p["avg_points"],
                "sim_team_gf":  rs["avg_gf"],
            })

    sim_df = pd.DataFrame(sim_rows)
    merged = sim_df.merge(actual_baseline, on="player_id", how="left")

    # PPG
    merged["actual_ppg"] = merged["actual_points"] / merged["actual_gp_weighted_avg"].replace(0, np.nan)
    merged["sim_ppg"]    = merged["sim_points"] / SIM_GP
    merged["delta_ppg"]  = merged["sim_ppg"] - merged["actual_ppg"]
    # Qualified pool matches PPG diagnostic
    qual = merged[(merged["actual_gp_total"] >= MIN_GP_FILTER)
                   & merged["actual_points"].notna()].copy()

    # Team-GF MAE
    gf_errs = []
    for team, sim_gf in sim_team_gf.items():
        if team in actual_team_gf:
            gf_errs.append(abs(sim_gf - actual_team_gf[team]))
    mae_gf = float(np.mean(gf_errs))
    rmse_gf = float(np.sqrt(np.mean(np.square(gf_errs))))
    max_team_gf = max(sim_team_gf.values())
    max_team    = max(sim_team_gf, key=sim_team_gf.get)

    # PPG aggregates
    top10 = set(team_strength_df.head(10)["team"])
    bot10 = set(team_strength_df.tail(10)["team"])
    league_mean_ppg = float(qual["delta_ppg"].mean())
    top10_mean_ppg  = float(qual[qual["team"].isin(top10)]["delta_ppg"].mean())
    bot10_mean_ppg  = float(qual[qual["team"].isin(bot10)]["delta_ppg"].mean())
    spread = abs(top10_mean_ppg - bot10_mean_ppg)

    elapsed = time.time() - t0
    print(f"  ALPHA={alpha:.3f}   GF_MAE={mae_gf:5.2f}   "
          f"max_team_GF={max_team_gf:.0f} ({max_team})   "
          f"league_ΔPPG={league_mean_ppg:+.4f}   "
          f"top10_ΔPPG={top10_mean_ppg:+.4f}   "
          f"bot10_ΔPPG={bot10_mean_ppg:+.4f}   "
          f"spread={spread:.4f}   "
          f"[{elapsed:.1f}s]", flush=True)

    return {
        "alpha": alpha,
        "gf_mae":          mae_gf,
        "gf_rmse":         rmse_gf,
        "max_team_gf":     max_team_gf,
        "max_team":        max_team,
        "league_mean_delta_ppg": league_mean_ppg,
        "top10_mean_delta_ppg":  top10_mean_ppg,
        "bot10_mean_delta_ppg":  bot10_mean_ppg,
        "top_bot_spread":  spread,
        "elapsed_s":       elapsed,
    }


def main():
    t_start = time.time()
    print("Loading inputs …", flush=True)
    comp_lookup = bts.load_composite_lookup()
    toi_lookup  = bts.load_toi_lookup()
    ts_df = pd.read_csv(MODEL_DIR / "team_strength.csv") \
                .sort_values("team_strength", ascending=False) \
                .reset_index(drop=True)
    ts_lookup = dict(zip(ts_df["team"], ts_df["team_strength"]))
    actual_team_gf = fetch_actual_team_gf()
    print(f"  team_strength: {len(ts_lookup)} teams")
    print(f"  actual GF:     {len(actual_team_gf)} teams")

    # Build actual baseline ONCE (network-heavy; doesn't depend on ALPHA)
    print("\nBuilding actual baseline (one-time) …", flush=True)
    actual_baseline = cc.build_actual_baseline()
    actual_baseline = actual_baseline.drop(columns=["player_name"], errors="ignore")
    print(f"  actual baseline: {len(actual_baseline):,} players\n")

    original_alpha = sim_engine.ALPHA
    print(f"Starting sweep across {len(ALPHAS)} ALPHA values "
          f"(prior live ALPHA = {original_alpha})…\n", flush=True)

    results = []
    for alpha in ALPHAS:
        r = run_sweep_step(alpha, comp_lookup, toi_lookup, ts_lookup,
                            actual_team_gf, actual_baseline, ts_df)
        results.append(r)

    sim_engine.ALPHA = original_alpha   # restore

    # Selection
    # Filter to those meeting the two PPG constraints
    feasible = [r for r in results
                if r["top_bot_spread"] <= 0.015
                and abs(r["league_mean_delta_ppg"]) <= 0.015]
    best = min(feasible, key=lambda r: r["gf_mae"]) if feasible else \
           min(results, key=lambda r: r["gf_mae"])

    print("\n" + "=" * 80)
    print("SWEEP TABLE")
    print("=" * 80)
    print(f"  {'ALPHA':>6}  {'GF_MAE':>7}  {'max_GF':>7}  "
          f"{'lgΔPPG':>9}  {'top10':>9}  {'bot10':>9}  {'spread':>7}  feasible")
    for r in results:
        feas = "✓" if (r["top_bot_spread"] <= 0.015
                       and abs(r["league_mean_delta_ppg"]) <= 0.015) else ""
        marker = " ←★" if r is best else ""
        print(f"  {r['alpha']:>6.3f}  {r['gf_mae']:>7.2f}  "
              f"{r['max_team_gf']:>7.0f}  {r['league_mean_delta_ppg']:>+9.4f}  "
              f"{r['top10_mean_delta_ppg']:>+9.4f}  {r['bot10_mean_delta_ppg']:>+9.4f}  "
              f"{r['top_bot_spread']:>7.4f}   {feas}{marker}")

    print(f"\nFEASIBLE candidates (meeting both PPG constraints): {len(feasible)}")
    print(f"WINNER: ALPHA = {best['alpha']:.3f}")
    print(f"  GF MAE             {best['gf_mae']:.2f}")
    print(f"  League mean ΔPPG   {best['league_mean_delta_ppg']:+.4f}")
    print(f"  Top-10 mean ΔPPG   {best['top10_mean_delta_ppg']:+.4f}")
    print(f"  Bottom-10 mean ΔPPG {best['bot10_mean_delta_ppg']:+.4f}")
    print(f"  Max team GF        {best['max_team_gf']:.0f} ({best['max_team']})")

    OUT_JSON.write_text(json.dumps({
        "n_sims": N_SIMS,
        "alphas_swept": ALPHAS,
        "winner": best,
        "constraints": {
            "max_top_bot_spread_ppg": 0.015,
            "max_abs_league_mean_ppg": 0.015,
        },
        "results": results,
    }, indent=2))
    print(f"\nSaved sweep results → {OUT_JSON}")
    print(f"Total compute: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
