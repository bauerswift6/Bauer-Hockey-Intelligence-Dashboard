"""
Phase 4 — Monte Carlo driver.

Runs two simulations and writes the results to disk:

  A) PLAYOFFS-ONLY FORECAST (current state, 2026-05-21)
     The 2025-26 regular season is complete (all 32 teams played 82 games).
     Round 1 and Round 2 are done. Round 3 (conference finals) is in progress:
       - VGK vs COL (VGK leads 1-0 going into game we just saw)
       - MTL vs CAR (0-0)
     We pull the live R3 series state from the NHL playoff carousel, lock the
     bracket (R1/R2 winners), and Monte-Carlo the remaining series + Cup Final.
     Output: probabilities of reaching each remaining round / winning the Cup
     for the four alive teams.

  B) 25-26 FULL-SEASON BACKTEST
     Treats the start of October 2025 as "current state": every team has
     wins=0/losses=0/otl=0/gf=0/ga=0, and the entire 1,312-game regular-season
     schedule is "remaining". Each sim picks a Cup champion from scratch via:
        sim all 1,312 RS games → standings → playoff seeding → 4 rounds.
     Output: projected playoff%, division%, conf-final%, cup%. Compare to
     actual final standings (which are known) for sanity.

Both runs use the same `team_strength.csv` (built from `composite_ratings_sim.csv`
which is already in 25-26 final-state — so the backtest is "hindsight talent"
projecting forward, intentionally conservative for validation, not predictive).

Run from project root:
    python3 model/run_season_simulation.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
NHL_API = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SEASON_25_26 = "20252026"

sys.path.insert(0, str(MODEL_DIR))
from simulate import (
    DIVISIONS, TEAM_TO_DIVISION, EASTERN, WESTERN,
    load_team_strengths, compute_seeding,
    simulate_remaining_schedule, _simulate_bracket_once,
    monte_carlo_full_season, monte_carlo_playoffs_only,
    _points_from_record,
)

ALL_TEAMS = list(TEAM_TO_DIVISION.keys())
OUT_CSV = MODEL_DIR / "season_simulation_results.csv"
PLAYOFF_OUT = MODEL_DIR / "playoff_projection_now.csv"


# ---------------------------------------------------------------------------
# Live data fetchers
# ---------------------------------------------------------------------------
def fetch_current_standings() -> dict[str, dict]:
    """Final 25-26 regular-season standings (all teams complete)."""
    r = requests.get(f"{NHL_API}/standings/now", headers=HEADERS, timeout=15)
    r.raise_for_status()
    d = r.json()
    out = {}
    for s in d["standings"]:
        abbrev = s["teamAbbrev"]["default"] if isinstance(s["teamAbbrev"], dict) else s["teamAbbrev"]
        out[abbrev] = {
            "wins": int(s.get("wins", 0)),
            "losses": int(s.get("losses", 0)),
            "otl": int(s.get("otLosses", 0)),
            "gf": int(s.get("goalFor", 0)),
            "ga": int(s.get("goalAgainst", 0)),
            "gp": int(s.get("gamesPlayed", 0)),
        }
    return out


def fetch_full_25_26_schedule() -> list[dict]:
    """Pull every 25-26 regular-season game from the NHL schedule endpoints.
    For the backtest we treat all 1,312 games as 'remaining'."""
    # Iterate week-by-week through Oct 2025 → Apr 2026 and collect home/away.
    games = []
    seen = set()
    date = "2025-10-01"
    end = "2026-04-30"
    while date <= end:
        r = requests.get(f"{NHL_API}/schedule/{date}", headers=HEADERS, timeout=15)
        if r.status_code != 200:
            break
        d = r.json()
        for day in d.get("gameWeek", []):
            for g in day.get("games", []):
                if g.get("gameType") != 2:  # 2 = regular season
                    continue
                gid = g.get("id")
                if gid in seen:
                    continue
                seen.add(gid)
                games.append({
                    "game_id": gid,
                    "home": g["homeTeam"]["abbrev"],
                    "away": g["awayTeam"]["abbrev"],
                })
        # advance to the next week
        next_start = d.get("nextStartDate") or _add_days(date, 7)
        if next_start == date:
            break
        date = next_start
    return games


def _add_days(date_str: str, n: int) -> str:
    import datetime
    return (datetime.date.fromisoformat(date_str)
            + datetime.timedelta(days=n)).isoformat()


def fetch_playoff_state() -> tuple[dict, dict]:
    """Pull the current 25-26 playoff bracket and any in-progress series.

    Returns (bracket, alive_state) where:
      - `bracket` is shaped like compute_seeding() output: {east: [(h,a), …], west: […]}
        Round 1 already decided, so we only seed Round 3 properly (we use a
        playoffs-only Monte Carlo that takes R1/R2 as given).
      - `alive_state` is {series_key: {a_wins, b_wins}} for the active CFs.
    """
    r = requests.get(f"{NHL_API}/playoff-series/carousel/{SEASON_25_26}",
                     headers=HEADERS, timeout=15)
    r.raise_for_status()
    d = r.json()
    rounds = d.get("rounds", [])
    cf_state = {}
    cf_pairings = {"east": [], "west": []}
    for rnd in rounds:
        if rnd["roundNumber"] != 3:
            continue
        for s in rnd["series"]:
            top = s["topSeed"]["abbrev"]
            bot = s["bottomSeed"]["abbrev"]
            tw = int(s["topSeed"].get("wins", 0))
            bw = int(s["bottomSeed"].get("wins", 0))
            conf = "east" if top in EASTERN else "west"
            cf_pairings[conf].append((top, bot))
            cf_state[(top, bot)] = {"a_wins": tw, "b_wins": bw}
    return cf_pairings, cf_state


# ---------------------------------------------------------------------------
# (A) PLAYOFFS-ONLY FORECAST
# ---------------------------------------------------------------------------
def run_playoffs_only(ts: dict[str, float], n_sims: int = 10000) -> pd.DataFrame:
    """Fold the live CF state into a small Monte Carlo over the remaining
    games. Round 1 and Round 2 are complete; only R3 + R4 remain."""
    cf_pairings, cf_state = fetch_playoff_state()
    print(f"  CF pairings: east={cf_pairings['east']}, west={cf_pairings['west']}")
    print(f"  CF current state: {cf_state}")

    import random
    rng = random.Random(42)
    counts = {t: {"conf_final": 0, "final": 0, "cup": 0}
              for t in TEAM_TO_DIVISION}
    from simulate import simulate_series

    for t0_a, t0_b in cf_pairings["east"] + cf_pairings["west"]:
        counts[t0_a]["conf_final"] = n_sims
        counts[t0_b]["conf_final"] = n_sims

    for _ in range(n_sims):
        conf_winners = {}
        for conf in ("east", "west"):
            (a, b) = cf_pairings[conf][0]
            init = cf_state.get((a, b), {"a_wins": 0, "b_wins": 0})
            # Higher seed by team_strength gets HOME ice as a proxy for
            # higher playoff seed (we don't have actual seeding info here).
            higher = a if ts.get(a, 0) >= ts.get(b, 0) else b
            res = simulate_series(a, b, ts, higher_seed=higher,
                                  a_wins=init["a_wins"], b_wins=init["b_wins"],
                                  rng=rng)
            conf_winners[conf] = res["winner"]
            counts[res["winner"]]["final"] += 1
        # Cup Final
        e, w = conf_winners["east"], conf_winners["west"]
        higher = e if ts.get(e, 0) >= ts.get(w, 0) else w
        cup = simulate_series(e, w, ts, higher_seed=higher, rng=rng)
        counts[cup["winner"]]["cup"] += 1

    rows = []
    for t, c in counts.items():
        if c["conf_final"] == 0 and c["final"] == 0 and c["cup"] == 0:
            continue
        rows.append({
            "team": t,
            "in_round": "R3" if c["conf_final"] > 0 else "eliminated",
            "conf_final_pct": round(c["conf_final"] / n_sims, 4),
            "final_pct": round(c["final"] / n_sims, 4),
            "cup_pct": round(c["cup"] / n_sims, 4),
        })
    return pd.DataFrame(rows).sort_values("cup_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# (B) 25-26 FULL-SEASON BACKTEST
# ---------------------------------------------------------------------------
def run_full_season_backtest(ts: dict[str, float],
                              schedule: list[dict],
                              n_sims: int = 10000) -> pd.DataFrame:
    """Treat October 2025 as 'current state' — zero records, all 1,312 games
    remaining — and sim out the full season + playoffs n_sims times."""
    start_standings = {t: {"wins": 0, "losses": 0, "otl": 0, "gf": 0, "ga": 0, "gp": 0}
                       for t in TEAM_TO_DIVISION}
    return monte_carlo_full_season(schedule, start_standings, ts, n_sims=n_sims)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase 4 — Monte Carlo driver")
    print("=" * 70)
    ts = load_team_strengths()
    print(f"Loaded team_strength for {len(ts)} teams.")

    # --- A. Playoffs-only forecast (live state) ---
    print("\n[A] Playoffs-only forecast (live 25-26 CFs in progress, 10k sims) …")
    t0 = time.time()
    po_df = run_playoffs_only(ts, n_sims=10000)
    po_df.to_csv(PLAYOFF_OUT, index=False)
    print(f"  done in {time.time()-t0:.1f}s → {PLAYOFF_OUT.name}")
    print("\n  Cup probabilities (4 alive teams):")
    for _, r in po_df.iterrows():
        print(f"    {r['team']:>4}  conf-final {r['conf_final_pct']*100:>5.1f}%  "
              f"final {r['final_pct']*100:>5.1f}%  cup {r['cup_pct']*100:>5.1f}%")

    # --- B. 25-26 full-season backtest (10k sims) ---
    print("\n[B] 25-26 full-season backtest (treats Oct 2025 as 'current state', 10k sims) …")
    print("  Pulling full 25-26 schedule from NHL API …")
    t0 = time.time()
    schedule = fetch_full_25_26_schedule()
    print(f"  Pulled {len(schedule):,} regular-season games in {time.time()-t0:.1f}s")
    print("  Running 10k Monte Carlo (sim 1,312 games × 10k = 13.1M game-sims) …")
    t0 = time.time()
    bt_df = run_full_season_backtest(ts, schedule, n_sims=10000)
    bt_df.to_csv(OUT_CSV, index=False)
    print(f"  done in {time.time()-t0:.1f}s → {OUT_CSV.name}")
    print("\n  Top 15 by projected Cup% (25-26 backtest):")
    print(bt_df.head(15)[["team","division","conference","avg_points",
                          "playoffs_pct","conf_final_pct","cup_pct"]].to_string(index=False))

    # --- Sanity vs actual final 25-26 standings ---
    print("\n  Backtest vs actual 25-26 final standings (top 8 alive playoff teams) …")
    actual = fetch_current_standings()
    actual_top = sorted(actual.items(),
                        key=lambda kv: _points_from_record(kv[1]), reverse=True)[:8]
    print(f"    Actual top 8 by points: {[t for t,_ in actual_top]}")
    bt_top_playoff = bt_df.sort_values("playoffs_pct", ascending=False).head(16)
    print(f"    Backtest top 16 by playoff%: {bt_top_playoff['team'].tolist()}")


if __name__ == "__main__":
    main()
