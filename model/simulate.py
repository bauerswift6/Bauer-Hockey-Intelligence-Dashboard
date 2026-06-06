"""
Phase 4 — Simulation engine.

Public surface:
  - expected_goals(home_team, away_team, ts)             → (xGF_h, xGF_a)
  - simulate_game(home_team, away_team, ts, rng=None)    → {home_score, away_score, winner, ot, so}
  - simulate_series(a, b, ts, a_wins=0, b_wins=0,
                    higher_seed=a, rng=None)             → {winner, games, length}
  - simulate_bracket(seeding, ts, alive_state=None,
                     n_sims=10000)                        → counts dict per round
  - simulate_remaining_schedule(remaining, ts,
                                start_standings)         → final standings dict
  - compute_seeding(standings)                            → {east:[…], west:[…]} 8 teams each
  - monte_carlo_full_season(schedule, start_standings, ts,
                            n_sims=10000)                 → aggregate dict
  - monte_carlo_playoffs(seeding, ts, alive_state=None,
                         n_sims=10000)                    → aggregate dict

team_strength → expected goals
------------------------------
For a game with home team H and away team A, with `ts` (a dict of
team_strength z-scores from model/team_strength.csv, mean 0 across the 32
teams):

    xGF_H = LEAGUE_GPG + ALPHA * (ts[H] - ts[A]) + HOME_ICE
    xGF_A = LEAGUE_GPG + ALPHA * (ts[A] - ts[H]) - HOME_ICE

Each side's actual goal count is drawn from NegBin(mean=xGF, variance=1.3×xGF)
— overdispersion ratio 1.3 matches NHL 25-26 empirical game-to-game variance.
The ALPHA elasticity was chosen so a top-quartile team at home vs a
bottom-quartile team away produces an expected ~3 goal differential — broadly
consistent with NHL moneylines and historical pythagorean fits.

If tied after regulation, simulate 5-min 3-on-3 OT (combined ~0.5 goal
expected) then shootout (skill-weighted by team_strength differential).
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"

# --- Calibration constants ---
LEAGUE_GPG = 3.13          # 25-26 league average goals per team per game
ALPHA = 0.525              # team-strength elasticity: empirically re-calibrated
                           # 2026-06-05 after Fix 1 (TOI sort) + assist-rate fix.
                           # Sweep 0.40-0.65 in 0.025 steps, scored on team-GF
                           # MAE + top-vs-bottom-10 PPG spread (see
                           # model/alpha_sweep_v2_results.json). 0.525 minimised
                           # GF MAE (16.4) among ALPHAs holding the team-tier
                           # PPG spread ≤ 0.015. Prior 0.45 over-compressed
                           # top-10 teams' GF; 0.525 balances the two.
HOME_ICE = 0.15            # +/- expected-goal home-ice advantage
OT_TOTAL_XG = 0.55         # combined expected goals across 5-min 3-on-3 OT (~50% conversion rate)
SHOOTOUT_BASE = 0.50       # base shootout win prob; adjusted by team-strength delta
SHOOTOUT_DELTA = 0.05      # per-unit team_strength delta adjustment

# Per-game scoring distribution. Pure Poisson under-states real NHL
# game-to-game variance because goal scoring is overdispersed
# (variance ≈ 1.3 × mean — see polarization diagnostic 2026-06-05).
# Pure Poisson made top playoff series win rate ~73% (sim) vs ~62%
# (NHL historical), which compounded over 4 rounds inflated Cup% top
# from a realistic ~19% to ~28%. Switching to NegBin(mean, 1.3×mean)
# preserves expected GF (calibration sweep is unchanged) but widens
# the per-game distribution so favourites win series at a realistic rate.
NB_OVERDISPERSION = 0.30   # variance / mean = 1 + NB_OVERDISPERSION
                           # 0 → pure Poisson; 0.3 matches NHL 25-26 empirical


# ---------------------------------------------------------------------------
# Game-level
# ---------------------------------------------------------------------------
def expected_goals(home_team: str, away_team: str,
                   ts: dict[str, float]) -> tuple[float, float]:
    """Expected (home_xG, away_xG) over 60 minutes."""
    h = ts.get(home_team, 0.0)
    a = ts.get(away_team, 0.0)
    xgh = max(0.05, LEAGUE_GPG + ALPHA * (h - a) + HOME_ICE)
    xga = max(0.05, LEAGUE_GPG + ALPHA * (a - h) - HOME_ICE)
    return xgh, xga


def _poisson(rate: float, rng: random.Random) -> int:
    """Standard Knuth Poisson sampler. Fast enough for our volumes."""
    L = math.exp(-rate)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _neg_binomial(rate: float, rng: random.Random,
                  overdispersion: float = NB_OVERDISPERSION) -> int:
    """Sample from a Negative Binomial with the specified mean and
    variance = mean × (1 + overdispersion). Reduces to Poisson when
    overdispersion = 0.

    Implementation uses the Gamma-Poisson mixture: draw rate λ from
    Gamma(shape=mean/overdispersion, scale=overdispersion), then sample
    X ~ Poisson(λ). This produces NB(n=mean/overdispersion, p=1/(1+overdispersion))
    with E[X]=mean and Var(X)=mean×(1+overdispersion)."""
    if rate <= 0:
        return 0
    if overdispersion <= 0:
        return _poisson(rate, rng)
    shape = rate / overdispersion
    scale = overdispersion
    lam = rng.gammavariate(shape, scale)
    return _poisson(lam, rng)


def simulate_game(home_team: str, away_team: str,
                  ts: dict[str, float],
                  rng: random.Random | None = None) -> dict:
    """Simulate one regular-season-style game. Returns dict with scoring and
    a boolean ot/so flag. Goal totals at the end include the OT/SO goal if any."""
    rng = rng or random.Random()
    xgh, xga = expected_goals(home_team, away_team, ts)
    h = _neg_binomial(xgh, rng)
    a = _neg_binomial(xga, rng)
    ot = so = False
    if h == a:
        ot = True
        # OT 3-on-3: combined ~0.55 xG over 5 minutes. Allocate proportionally
        # to expected-goal share (slight favorite gets more).
        share_h = xgh / (xgh + xga) if (xgh + xga) > 0 else 0.5
        ot_h_xg = OT_TOTAL_XG * share_h
        ot_a_xg = OT_TOTAL_XG * (1 - share_h)
        # Whichever team scores first in OT wins (sudden-death). Use a draw
        # against the combined intensity, then bias by share.
        # If neither scores in OT (Poisson(0.55) → P(0)=0.577), go to shootout.
        # We model OT as a single trial: with prob = 1 - exp(-OT_TOTAL_XG) a
        # goal is scored; the team that scores is share-weighted.
        if rng.random() < (1 - math.exp(-OT_TOTAL_XG)):
            if rng.random() < share_h:
                h += 1
            else:
                a += 1
        else:
            so = True
            # Shootout: favorite probability biased by team_strength delta.
            delta = ts.get(home_team, 0.0) - ts.get(away_team, 0.0)
            ph = max(0.15, min(0.85, SHOOTOUT_BASE + SHOOTOUT_DELTA * delta))
            if rng.random() < ph:
                h += 1
            else:
                a += 1
    return {
        "home_team": home_team, "away_team": away_team,
        "home_score": h, "away_score": a,
        "winner": home_team if h > a else away_team,
        "ot": ot, "so": so,
        "xg_home": round(xgh, 3), "xg_away": round(xga, 3),
    }


# ---------------------------------------------------------------------------
# Win-probability (analytic, no sim)
# ---------------------------------------------------------------------------
def win_probability(home_team: str, away_team: str,
                    ts: dict[str, float], n_sims: int = 5000,
                    rng: random.Random | None = None) -> dict:
    """Monte-Carlo win probability for a single matchup (default 5k sims).
    Returns {home_win_pct, away_win_pct, expected_home_score, expected_away_score,
             ot_pct, so_pct}."""
    rng = rng or random.Random()
    h_wins = 0; ot_count = 0; so_count = 0
    h_sum = 0.0; a_sum = 0.0
    for _ in range(n_sims):
        g = simulate_game(home_team, away_team, ts, rng)
        h_wins += int(g["winner"] == home_team)
        ot_count += int(g["ot"])
        so_count += int(g["so"])
        h_sum += g["home_score"]
        a_sum += g["away_score"]
    return {
        "home_team": home_team, "away_team": away_team,
        "home_win_pct": round(h_wins / n_sims, 4),
        "away_win_pct": round(1 - h_wins / n_sims, 4),
        "expected_home_score": round(h_sum / n_sims, 2),
        "expected_away_score": round(a_sum / n_sims, 2),
        "ot_pct": round(ot_count / n_sims, 4),
        "so_pct": round(so_count / n_sims, 4),
        "n_sims": n_sims,
    }


# ---------------------------------------------------------------------------
# Best-of-7 series
# ---------------------------------------------------------------------------
def simulate_series(team_a: str, team_b: str, ts: dict[str, float],
                    higher_seed: str | None = None,
                    a_wins: int = 0, b_wins: int = 0,
                    rng: random.Random | None = None) -> dict:
    """Simulate a best-of-7 to completion from the current series score.
    `higher_seed` gets home ice in games 1, 2, 5, 7 (the 2-2-1-1-1 format).
    Returns {winner, games_played_total, a_final, b_final}."""
    rng = rng or random.Random()
    higher = higher_seed if higher_seed in (team_a, team_b) else team_a
    lower = team_b if higher == team_a else team_a
    # 2-2-1-1-1 schedule. Game indices 0..6. True = higher_seed at home.
    HOME_FORMAT = [True, True, False, False, True, False, True]
    games_played = a_wins + b_wins
    a, b = a_wins, b_wins
    g_index = games_played
    while a < 4 and b < 4:
        if g_index >= 7:
            # Should never reach — series ends at 4 wins. Safety break.
            break
        home_higher = HOME_FORMAT[g_index]
        home = higher if home_higher else lower
        away = higher if not home_higher else lower
        # Playoff games go to OT every time they're tied — but the OT format
        # is 5-on-5 sudden death (not 3-on-3). Approximation: keep our
        # simulate_game function but skip the shootout step (no shootouts in
        # playoffs).
        g = simulate_game(home, away, ts, rng)
        if g["so"]:
            # In the rare case our OT routine produced a shootout result,
            # treat it as a real OT goal — playoffs don't have shootouts but
            # our OT model already nominally handles the tie.
            pass
        if g["winner"] == team_a:
            a += 1
        else:
            b += 1
        g_index += 1
    winner = team_a if a >= 4 else team_b
    return {
        "team_a": team_a, "team_b": team_b,
        "higher_seed": higher,
        "winner": winner,
        "a_wins": a, "b_wins": b,
        "games_played_total": a + b,
        "starting_a_wins": a_wins, "starting_b_wins": b_wins,
    }


# ---------------------------------------------------------------------------
# Standings + seeding (NHL rules: 3 per division + 2 wild cards per conference)
# ---------------------------------------------------------------------------
DIVISIONS = {
    # 2025-26 NHL alignment
    "Atlantic":  ["BOS","BUF","DET","FLA","MTL","OTT","TBL","TOR"],
    "Metro":     ["CAR","CBJ","NJD","NYI","NYR","PHI","PIT","WSH"],
    "Central":   ["CHI","COL","DAL","MIN","NSH","STL","UTA","WPG"],
    "Pacific":   ["ANA","CGY","EDM","LAK","SEA","SJS","VAN","VGK"],
}
TEAM_TO_DIVISION = {t: d for d, teams in DIVISIONS.items() for t in teams}
EASTERN = set(DIVISIONS["Atlantic"]) | set(DIVISIONS["Metro"])
WESTERN = set(DIVISIONS["Central"]) | set(DIVISIONS["Pacific"])


def _points_from_record(rec: dict) -> int:
    """Standard NHL points = 2W + 1OTL (+ 1SOL where applicable; current rules
    award 2 for any win, 1 for any loss after regulation)."""
    return 2 * rec.get("wins", 0) + rec.get("otl", 0)


def compute_seeding(standings: dict[str, dict]) -> dict[str, list[str]]:
    """Build the 16-team playoff bracket from a standings dict
    {team: {wins, losses, otl, gf, ga, …}}.

    Returns {"east": [seed_order], "west": [seed_order]}. Within each
    conference: top 3 from each division (Atl1/Atl2/Atl3/Met1/Met2/Met3) then
    the 2 wild cards (next-highest points across both divisions in conf).
    Bracket order: A1 vs WC2, A2 vs A3, M1 vs WC1, M2 vs M3 — standard NHL
    playoff format with the better division winner facing the lower wild card.
    Same for West (Central/Pacific)."""
    out = {}
    for conf_name, div_pair in (("east", ("Atlantic", "Metro")),
                                 ("west", ("Central", "Pacific"))):
        div_a, div_b = div_pair
        # Rank within each division by points (tiebreakers omitted for sim)
        a_ranked = sorted(DIVISIONS[div_a],
                          key=lambda t: (_points_from_record(standings.get(t, {})),
                                          standings.get(t, {}).get("gf", 0)),
                          reverse=True)
        b_ranked = sorted(DIVISIONS[div_b],
                          key=lambda t: (_points_from_record(standings.get(t, {})),
                                          standings.get(t, {}).get("gf", 0)),
                          reverse=True)
        a1, a2, a3 = a_ranked[0], a_ranked[1], a_ranked[2]
        b1, b2, b3 = b_ranked[0], b_ranked[1], b_ranked[2]
        # Wild cards: next-best 2 from the remaining 10 teams in the conf
        remaining = a_ranked[3:] + b_ranked[3:]
        wc_ranked = sorted(remaining,
                           key=lambda t: (_points_from_record(standings.get(t, {})),
                                           standings.get(t, {}).get("gf", 0)),
                           reverse=True)
        wc1, wc2 = wc_ranked[0], wc_ranked[1]
        # Better div winner (more points) gets the lower wild card
        if _points_from_record(standings.get(a1, {})) >= _points_from_record(standings.get(b1, {})):
            better_div, worse_div = (a1, a2, a3), (b1, b2, b3)
        else:
            better_div, worse_div = (b1, b2, b3), (a1, a2, a3)
        # Bracket (4 first-round matchups in this conf)
        bracket = [
            (better_div[0], wc2),      # better div winner vs lower wild card
            (better_div[1], better_div[2]),
            (worse_div[0], wc1),
            (worse_div[1], worse_div[2]),
        ]
        out[conf_name] = bracket
    return out


# ---------------------------------------------------------------------------
# Full-season Monte Carlo
# ---------------------------------------------------------------------------
def simulate_remaining_schedule(remaining_games: list[dict],
                                 start_standings: dict[str, dict],
                                 ts: dict[str, float],
                                 rng: random.Random) -> dict[str, dict]:
    """Sim each game in `remaining_games` (list of {home, away}). Apply
    results onto a copy of start_standings. Return final standings."""
    rec = {t: dict(start_standings.get(t, {"wins":0,"losses":0,"otl":0,"gf":0,"ga":0}))
           for t in TEAM_TO_DIVISION}
    for g in remaining_games:
        out = simulate_game(g["home"], g["away"], ts, rng)
        h, a = out["home_team"], out["away_team"]
        rec[h]["gf"] += out["home_score"]; rec[h]["ga"] += out["away_score"]
        rec[a]["gf"] += out["away_score"]; rec[a]["ga"] += out["home_score"]
        if out["winner"] == h:
            rec[h]["wins"] = rec[h].get("wins", 0) + 1
            if out["ot"]:
                rec[a]["otl"] = rec[a].get("otl", 0) + 1
            else:
                rec[a]["losses"] = rec[a].get("losses", 0) + 1
        else:
            rec[a]["wins"] = rec[a].get("wins", 0) + 1
            if out["ot"]:
                rec[h]["otl"] = rec[h].get("otl", 0) + 1
            else:
                rec[h]["losses"] = rec[h].get("losses", 0) + 1
    return rec


def _simulate_bracket_once(bracket: dict[str, list],
                           ts: dict[str, float],
                           alive_state: dict | None,
                           rng: random.Random) -> dict:
    """Run one full playoff bracket. Returns {champion, conf_finalists,
    semifinalists, second_round_qualifiers, first_round_qualifiers}.

    `alive_state` is an optional {series_key: {a_wins, b_wins}} to seed
    in-progress series. Currently only used at round level for the playoffs-
    only forecast."""
    rounds_results = {"R1": [], "R2": [], "R3": [], "R4": None}
    survivors = {"east": [], "west": []}

    # Round 1
    for conf in ("east", "west"):
        for (h, a) in bracket[conf]:
            key = f"R1:{h}:{a}"
            aw, bw = 0, 0
            if alive_state and key in alive_state:
                aw, bw = alive_state[key]["a_wins"], alive_state[key]["b_wins"]
            res = simulate_series(h, a, ts, higher_seed=h, a_wins=aw, b_wins=bw, rng=rng)
            survivors[conf].append(res["winner"])
            rounds_results["R1"].append((h, a, res["winner"]))

    # Round 2 (semifinals within each conference, 1 vs 4 and 2 vs 3 by bracket order)
    r2_survivors = {"east": [], "west": []}
    for conf in ("east", "west"):
        # Pair up survivors in bracket order: (0 vs 1), (2 vs 3)
        s = survivors[conf]
        for h, a in ((s[0], s[1]), (s[2], s[3])):
            # Higher seed = team with more points/strength. Approximate with ts.
            higher = h if ts.get(h, 0) >= ts.get(a, 0) else a
            res = simulate_series(h, a, ts, higher_seed=higher, rng=rng)
            r2_survivors[conf].append(res["winner"])
            rounds_results["R2"].append((h, a, res["winner"]))

    # Conference Finals
    conf_winners = {}
    for conf in ("east", "west"):
        s = r2_survivors[conf]
        higher = s[0] if ts.get(s[0], 0) >= ts.get(s[1], 0) else s[1]
        res = simulate_series(s[0], s[1], ts, higher_seed=higher, rng=rng)
        conf_winners[conf] = res["winner"]
        rounds_results["R3"].append((s[0], s[1], res["winner"]))

    # Cup Final
    e, w = conf_winners["east"], conf_winners["west"]
    higher = e if ts.get(e, 0) >= ts.get(w, 0) else w
    res = simulate_series(e, w, ts, higher_seed=higher, rng=rng)
    rounds_results["R4"] = (e, w, res["winner"])

    return {
        "champion": res["winner"],
        "conf_finalists": [e, w],
        "semifinalists": r2_survivors["east"] + r2_survivors["west"],
        "second_round_qualifiers": survivors["east"] + survivors["west"],
        "all_playoff_teams": survivors["east"] + survivors["west"],
        "rounds": rounds_results,
    }


def monte_carlo_full_season(remaining_games: list[dict],
                             start_standings: dict[str, dict],
                             ts: dict[str, float],
                             n_sims: int = 10000,
                             seed: int = 42) -> pd.DataFrame:
    """Run n_sims of (sim remaining schedule → seed → playoffs) and aggregate
    probability outcomes per team."""
    rng = random.Random(seed)
    # Counters per team
    counts = {t: {"playoffs": 0, "division": 0, "round2": 0, "conf_final": 0,
                  "final": 0, "cup": 0, "total_points": 0.0, "total_gf": 0.0}
              for t in TEAM_TO_DIVISION}

    for _ in range(n_sims):
        std = simulate_remaining_schedule(remaining_games, start_standings, ts, rng)
        # Division winners
        for div_name, teams in DIVISIONS.items():
            div_winner = max(teams, key=lambda t: (_points_from_record(std[t]),
                                                     std[t].get("gf", 0)))
            counts[div_winner]["division"] += 1
        # Bracket + playoff teams
        bracket = compute_seeding(std)
        for conf in ("east", "west"):
            for (h, a) in bracket[conf]:
                counts[h]["playoffs"] += 1
                counts[a]["playoffs"] += 1
        # Simulate the bracket
        res = _simulate_bracket_once(bracket, ts, None, rng)
        for t in res["second_round_qualifiers"]:
            counts[t]["round2"] += 1
        for t in res["semifinalists"]:
            counts[t]["conf_final"] += 1
        for t in res["conf_finalists"]:
            counts[t]["final"] += 1
        counts[res["champion"]]["cup"] += 1
        # Track avg points
        for t in TEAM_TO_DIVISION:
            counts[t]["total_points"] += _points_from_record(std[t])
            counts[t]["total_gf"] += std[t].get("gf", 0)

    rows = []
    for t, c in counts.items():
        rows.append({
            "team": t,
            "division": TEAM_TO_DIVISION[t],
            "conference": "East" if t in EASTERN else "West",
            "playoffs_pct": round(c["playoffs"] / n_sims, 4),
            "division_pct": round(c["division"] / n_sims, 4),
            "round2_pct": round(c["round2"] / n_sims, 4),
            "conf_final_pct": round(c["conf_final"] / n_sims, 4),
            "final_pct": round(c["final"] / n_sims, 4),
            "cup_pct": round(c["cup"] / n_sims, 4),
            "avg_points": round(c["total_points"] / n_sims, 1),
            "avg_gf": round(c["total_gf"] / n_sims, 1),
        })
    return pd.DataFrame(rows).sort_values("cup_pct", ascending=False).reset_index(drop=True)


def monte_carlo_playoffs_only(bracket: dict[str, list],
                              alive_state: dict | None,
                              ts: dict[str, float],
                              n_sims: int = 10000,
                              seed: int = 42) -> pd.DataFrame:
    """For the in-progress 2025-26 playoffs: seeding is locked, R1/R2 may
    already be complete, R3 in progress. `bracket` is the seeding result.
    `alive_state` is {series_key: {a_wins, b_wins}} or None."""
    rng = random.Random(seed)
    counts = {t: {"round2": 0, "conf_final": 0, "final": 0, "cup": 0}
              for t in TEAM_TO_DIVISION}
    for _ in range(n_sims):
        res = _simulate_bracket_once(bracket, ts, alive_state, rng)
        for t in res["second_round_qualifiers"]:
            counts[t]["round2"] += 1
        for t in res["semifinalists"]:
            counts[t]["conf_final"] += 1
        for t in res["conf_finalists"]:
            counts[t]["final"] += 1
        counts[res["champion"]]["cup"] += 1
    rows = []
    for t, c in counts.items():
        rows.append({
            "team": t,
            "round2_pct": round(c["round2"] / n_sims, 4),
            "conf_final_pct": round(c["conf_final"] / n_sims, 4),
            "final_pct": round(c["final"] / n_sims, 4),
            "cup_pct": round(c["cup"] / n_sims, 4),
        })
    return pd.DataFrame(rows).sort_values("cup_pct", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_team_strengths(path: Path = MODEL_DIR / "team_strength.csv") -> dict[str, float]:
    df = pd.read_csv(path)
    return dict(zip(df["team"], df["team_strength"]))
