"""
Phase 5.1 — Season simulator.

Runs the full 25-26 schedule (1,312 RS games + the playoff bracket) some
number of times for a USER-CUSTOMISED roster on ONE team while all other 31
teams use their default `team_strength`. Per-player projected stats for the
custom team are computed from a four-layer distribution model:

  Layer 1 — Base historical shares (3-season year-weighted, 5v5+5v4)
            → goal_share, p_assist_share, s_assist_share per player.
  Layer 2 — Ice-time scaling. Each lineup slot has a canonical minutes value
            (top line F = 22, 2nd = 18, …). Player's share scales by
            (slot_min / historical_avg_TOI).
  Layer 3 — Linemate-quality multiplier. For a forward, the avg composite_war
            of the other 2 line-mates; for a defenseman, the partner's
            composite_war. Multiplier = 1 + 0.15 × (linemate_war - baseline),
            clipped to [0.70, 1.40]. Applied to goal and primary-assist
            shares only (not secondary).
  Layer 4 — Talent floor for thin samples (<1,500 EV min over the 3-year
            window). Blends 60% composite-derived shares with 40% historical
            so rookies and breakouts don't crater. Composite-derived shares
            use a linear regression fit on the well-sampled pool.

After per-player shares are computed, they're normalised within the roster so
the team's actual Poisson goal total is fully distributed across the 18
skaters every game. Assists follow with NHL-realistic 10%/25%/65% unassisted/
single/double-assist mix.

Per-team game outcomes (W/L/OTL/GF/GA) follow the existing Phase 4 engine
(`simulate.simulate_game`); we just track custom-team player stats in the
hot loop. After the 82-game custom-team RS, the rest of the bracket is
simulated to derive playoff-round probabilities.

Public surface:
  monte_carlo_custom_season(custom_team, custom_lineup, custom_ts,
                             n_sims, default_ts=None, also_run_default=False)
      → dict (regular season block + player_stats block + playoff block)

Typical compute budget at n_sims=250: 8–15 seconds; the default-comparison
variant doubles it.
"""
from __future__ import annotations
import json
import math
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
SHARES_CSV = MODEL_DIR / "player_offense_shares.csv"
SHARES_META = MODEL_DIR / "player_offense_shares_meta.json"
COMPOSITE_SIM_CSV = MODEL_DIR / "composite_ratings_sim.csv"
TEAM_STRENGTH_CSV = MODEL_DIR / "team_strength.csv"
SCHEDULE_CACHE = MODEL_DIR / "schedule_25_26.json"
NHL_API = "https://api-web.nhle.com/v1"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# Phase 4 engine — game simulation and bracket
import sys as _sys
if str(MODEL_DIR) not in _sys.path:
    _sys.path.insert(0, str(MODEL_DIR))
import simulate as sim_engine

# Layer 2 — canonical slot minutes
FWD_SLOT_MINUTES = [22, 22, 22, 18, 18, 18, 14, 14, 14, 10, 10, 10]
DEF_SLOT_MINUTES = [24, 24, 20, 20, 16, 16]

# Layer 3 — linemate-quality multiplier
LINEMATE_K = 0.15
LINEMATE_CAP_LO = 0.70
LINEMATE_CAP_HI = 1.40

# Layer 4 — thin-sample blend
LAYER4_HIST_WEIGHT = 0.40   # 40% historical
LAYER4_TALENT_WEIGHT = 0.60 # 60% composite-derived

# Assist distribution — empirical NHL 25-26 rates (Phase 5.1 calibration,
# 2026-06-05). Source: 32-team aggregate from MoneyPuck 2025-26 skater CSV
# (8,084 goals; 7,571 A1; 6,080 A2 → A1/goal=0.937, A2/goal=0.752).
# Prior tuning (0.10/0.25/0.65) under-projected A1 by 4% and A2 by 14%,
# systematically suppressing pure playmakers and offensive D in PPG terms.
UNASSISTED_RATE = 0.064
SINGLE_ASSIST_RATE = 0.184
# implied double-assist rate = 0.752


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_shares() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(SHARES_CSV)
    meta = json.loads(SHARES_META.read_text()) if SHARES_META.exists() else {}
    return df, meta


def load_composite() -> pd.DataFrame:
    return pd.read_csv(COMPOSITE_SIM_CSV)


def load_team_strengths() -> dict[str, float]:
    df = pd.read_csv(TEAM_STRENGTH_CSV)
    return dict(zip(df["team"], df["team_strength"]))


def fetch_schedule() -> list[dict]:
    """Pull the full 25-26 RS schedule (1,312 games). Cached to JSON for
    subsequent calls — schedule is fixed once the season has played out."""
    if SCHEDULE_CACHE.exists():
        return json.loads(SCHEDULE_CACHE.read_text())
    games, seen = [], set()
    date = "2025-10-01"
    end = "2026-04-30"
    while date <= end:
        r = requests.get(f"{NHL_API}/schedule/{date}", headers=HEADERS, timeout=15)
        if r.status_code != 200:
            break
        d = r.json()
        for day in d.get("gameWeek", []):
            for g in day.get("games", []):
                if g.get("gameType") != 2:
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
        next_start = d.get("nextStartDate")
        if not next_start or next_start == date:
            import datetime
            date = (datetime.date.fromisoformat(date)
                    + datetime.timedelta(days=7)).isoformat()
        else:
            date = next_start
    SCHEDULE_CACHE.write_text(json.dumps(games))
    return games


# ---------------------------------------------------------------------------
# Four-layer player distribution for a single roster
# ---------------------------------------------------------------------------
def compute_baseline_linemate_war(comp_df: pd.DataFrame) -> float:
    """League median composite_war among rated players. ~0.0 by construction
    of the WAR rescale but recompute defensively."""
    return float(comp_df["composite_war"].median())


def _layer4_shares(player_id: int, position: str,
                   shares_row: pd.Series | None,
                   comp_row: pd.Series | None,
                   league_avg: dict, regressions: dict) -> dict:
    """Return the Layer-4-blended (or raw historical) share dict for one
    player.

    `shares_row` may be None for a player who has zero MoneyPuck history
    (extreme edge case — rookies who debuted only after our window).
    `comp_row` may be None for unrated players.
    """
    pos = position[0] if position else "C"
    if pos == "G":
        return {"goal_share": 0.0, "p_assist_share": 0.0, "s_assist_share": 0.0,
                "avg_toi_per_game": 0.0, "ev_min_3yr": 0.0,
                "thin_sample": True, "rated": False}

    avg_toi = float(getattr(shares_row, "avg_toi_per_game", 0.0)) if shares_row is not None else 0.0
    ev_min  = float(getattr(shares_row, "ev_minutes_3yr",   0.0)) if shares_row is not None else 0.0
    hist_g  = float(getattr(shares_row, "goal_share",       0.0)) if shares_row is not None else 0.0
    hist_a1 = float(getattr(shares_row, "p_assist_share",   0.0)) if shares_row is not None else 0.0
    hist_a2 = float(getattr(shares_row, "s_assist_share",   0.0)) if shares_row is not None else 0.0

    thin = (shares_row is None) or (ev_min < 1500.0)
    rated = comp_row is not None

    if not thin:
        return {
            "goal_share": hist_g, "p_assist_share": hist_a1,
            "s_assist_share": hist_a2,
            "avg_toi_per_game": avg_toi, "ev_min_3yr": ev_min,
            "thin_sample": False, "rated": rated,
        }

    # Layer 4 — blend with composite-derived shares.
    pos_avg = league_avg.get(pos, league_avg.get("C", {})) or {}
    if rated:
        ind_z = float(getattr(comp_row, "individual_component", 0.0))
        play_z = float(getattr(comp_row, "playmaking_component", 0.0))
        reg_g = regressions.get("goal_share", {})
        reg_a = regressions.get("p_assist_share", {})
        talent_g  = max(0.0, reg_g.get("slope", 0.0) * ind_z + reg_g.get("intercept", 0.0))
        talent_a1 = max(0.0, reg_a.get("slope", 0.0) * play_z + reg_a.get("intercept", 0.0))
    else:
        talent_g  = pos_avg.get("goal_share", 0.05)
        talent_a1 = pos_avg.get("p_assist_share", 0.05)
    talent_a2 = pos_avg.get("s_assist_share", 0.05)

    g  = LAYER4_HIST_WEIGHT * hist_g  + LAYER4_TALENT_WEIGHT * talent_g
    a1 = LAYER4_HIST_WEIGHT * hist_a1 + LAYER4_TALENT_WEIGHT * talent_a1
    a2 = LAYER4_HIST_WEIGHT * hist_a2 + LAYER4_TALENT_WEIGHT * talent_a2
    return {
        "goal_share": g, "p_assist_share": a1, "s_assist_share": a2,
        "avg_toi_per_game": avg_toi if avg_toi > 0 else 14.0,
        "ev_min_3yr": ev_min,
        "thin_sample": True, "rated": rated,
    }


def build_player_distribution(custom_lineup: dict, shares_df: pd.DataFrame,
                              comp_df: pd.DataFrame, meta: dict,
                              baseline_war: float) -> dict:
    """Apply all 4 layers to derive each skater's normalised share.

    `custom_lineup` is shaped like the simulator UI:
        {"forwards": [{"player_id":…, "position":"C",…}, …(12 slots)…],
         "defense":  [(6 slots)…],
         "goalies":  [{…(starter)…}, {…(backup)…}]}

    Each forwards entry is interpreted as a SLOT (slot index 0..11). A
    `None`/empty slot is treated as a replacement-level skater with 0 share.

    Returns:
      {
        "team_norm_g":  array of 18 shares (sum = 1.0)
        "team_norm_a1": …
        "team_norm_a2": …
        "skaters": [ {player_id, name, position, slot_kind, slot_idx,
                       slot_minutes, line_index, share_g, share_a1, share_a2,
                       rated, thin_sample, hist_avg_toi, ev_min_3yr,
                       linemate_war, linemate_mult}, …(18 rows)…]
      }
    """
    league_avg = meta.get("league_avg_shares_by_position", {})
    regressions = meta.get("share_regressions", {})

    shares_by_pid = {int(r.player_id): r for r in shares_df.itertuples(index=False)}
    comp_by_pid = {int(r.player_id): r for r in comp_df.itertuples(index=False)}

    fwd = custom_lineup.get("forwards", []) or []
    dfn = custom_lineup.get("defense", []) or []

    # Helper to fetch a player's composite_war (0.0 if unrated/empty slot)
    def _war(pid):
        if pid is None: return 0.0
        c = comp_by_pid.get(int(pid))
        return float(c.composite_war) if c is not None else 0.0

    # Pre-compute linemate WARs per slot
    # Forwards: for each line (3 slots), the multiplier sees the OTHER 2 slots
    # Defense: each slot sees its pair partner
    skaters = []
    for slot_idx, p in enumerate(fwd[:12]):
        if p is None or not p.get("player_id"):
            skaters.append({
                "player_id": None, "name": "(empty)", "position": "C",
                "slot_kind": "F", "slot_idx": slot_idx,
                "slot_minutes": FWD_SLOT_MINUTES[slot_idx],
                "line_index": slot_idx // 3, "is_filled": False,
            })
            continue
        skaters.append({
            "player_id": int(p["player_id"]),
            "name": p.get("first_name", "") + " " + p.get("last_name", ""),
            "position": (p.get("position") or "C"),
            "slot_kind": "F", "slot_idx": slot_idx,
            "slot_minutes": FWD_SLOT_MINUTES[slot_idx],
            "line_index": slot_idx // 3, "is_filled": True,
        })
    for slot_idx, p in enumerate(dfn[:6]):
        if p is None or not p.get("player_id"):
            skaters.append({
                "player_id": None, "name": "(empty)", "position": "D",
                "slot_kind": "D", "slot_idx": slot_idx,
                "slot_minutes": DEF_SLOT_MINUTES[slot_idx],
                "line_index": slot_idx // 2, "is_filled": False,
            })
            continue
        skaters.append({
            "player_id": int(p["player_id"]),
            "name": p.get("first_name", "") + " " + p.get("last_name", ""),
            "position": "D",
            "slot_kind": "D", "slot_idx": slot_idx,
            "slot_minutes": DEF_SLOT_MINUTES[slot_idx],
            "line_index": slot_idx // 2, "is_filled": True,
        })

    # For each slot, identify linemates (line peers) and compute the mult.
    for s in skaters:
        if not s["is_filled"]:
            s["linemate_war"] = 0.0
            s["linemate_mult"] = 1.0
            continue
        if s["slot_kind"] == "F":
            line_start = s["line_index"] * 3
            peers = [skaters[line_start + j] for j in range(3)
                     if (line_start + j) != s["slot_idx"]
                        and skaters[line_start + j]["is_filled"]]
        else:
            # defense — the pair partner only
            pair_start = 12 + s["line_index"] * 2
            partner_idx = pair_start + (1 - (s["slot_idx"] % 2))
            # skaters list has indices: 0..11 = F slots, 12..17 = D slots.
            # But we stored slot_idx as 0..5 for D within their bucket — the
            # `skaters` array index for D[i] is 12+i.
            partner_slot = s["slot_idx"] ^ 1   # 0↔1, 2↔3, 4↔5
            if 0 <= partner_slot <= 5 and partner_slot < len(dfn):
                p_idx_in_skaters = 12 + partner_slot
                if 0 <= p_idx_in_skaters < len(skaters) and skaters[p_idx_in_skaters]["is_filled"]:
                    peers = [skaters[p_idx_in_skaters]]
                else:
                    peers = []
            else:
                peers = []
        if peers:
            linemate_war = sum(_war(p["player_id"]) for p in peers) / len(peers)
        else:
            linemate_war = 0.0
        mult = 1.0 + LINEMATE_K * (linemate_war - baseline_war)
        mult = max(LINEMATE_CAP_LO, min(LINEMATE_CAP_HI, mult))
        s["linemate_war"] = round(linemate_war, 3)
        s["linemate_mult"] = round(mult, 4)

    # Now apply Layers 1+2+4 per player, then layer 3 to G/A1.
    for s in skaters:
        if not s["is_filled"]:
            s["share_g"] = s["share_a1"] = s["share_a2"] = 0.0
            s["rated"] = False; s["thin_sample"] = False
            s["hist_avg_toi"] = 0.0; s["ev_min_3yr"] = 0.0
            continue
        pid = s["player_id"]
        srow = shares_by_pid.get(pid)
        crow = comp_by_pid.get(pid)
        L1 = _layer4_shares(pid, s["position"], srow, crow, league_avg, regressions)
        s["rated"] = L1["rated"]
        s["thin_sample"] = L1["thin_sample"]
        s["hist_avg_toi"] = L1["avg_toi_per_game"]
        s["ev_min_3yr"] = L1["ev_min_3yr"]
        # Layer 2 — slot-minutes / historical-TOI ratio. Floor historical at
        # 8 min to prevent division explosions for ghost-rostered minimum-TOI
        # players, cap the multiplier at 2.5x for safety.
        hist_toi = max(L1["avg_toi_per_game"], 8.0)
        ratio = s["slot_minutes"] / hist_toi
        ratio = max(0.40, min(2.5, ratio))
        s["layer2_ratio"] = round(ratio, 3)
        # Layers 1 + 2 + 3 (for G and A1; A2 skips L3 per spec)
        s["share_g"]  = L1["goal_share"]      * ratio * s["linemate_mult"]
        s["share_a1"] = L1["p_assist_share"]  * ratio * s["linemate_mult"]
        s["share_a2"] = L1["s_assist_share"]  * ratio

    # Normalise within the team so the Poisson goal total is fully distributed.
    arr_g  = np.array([s["share_g"]  for s in skaters], dtype=np.float64)
    arr_a1 = np.array([s["share_a1"] for s in skaters], dtype=np.float64)
    arr_a2 = np.array([s["share_a2"] for s in skaters], dtype=np.float64)
    if arr_g.sum() > 0:  arr_g  = arr_g  / arr_g.sum()
    if arr_a1.sum() > 0: arr_a1 = arr_a1 / arr_a1.sum()
    if arr_a2.sum() > 0: arr_a2 = arr_a2 / arr_a2.sum()
    for i, s in enumerate(skaters):
        s["norm_share_g"]  = float(arr_g[i])
        s["norm_share_a1"] = float(arr_a1[i])
        s["norm_share_a2"] = float(arr_a2[i])

    return {"team_norm_g": arr_g, "team_norm_a1": arr_a1,
            "team_norm_a2": arr_a2, "skaters": skaters}


# ---------------------------------------------------------------------------
# One-sim hot loop
# ---------------------------------------------------------------------------
def simulate_one_season(custom_team: str,
                         dist: dict,
                         schedule_arr: list[tuple[str, str]],
                         ts: dict[str, float],
                         rng: random.Random,
                         np_rng: np.random.Generator) -> dict:
    """Run all 1,312 RS games. Return:
       { 'standings': { team: {wins,losses,otl,gf,ga} },
         'player_goals':  np.array(18),
         'player_a1':     np.array(18),
         'player_a2':     np.array(18),
         'custom_gp':     int (number of games custom team played in this sim) }
    The schedule is fixed (every team plays 82) so custom_gp == 82 for every
    sim — kept for explicitness."""
    n_skaters = len(dist["skaters"])
    p_goals = np.zeros(n_skaters, dtype=np.int32)
    p_a1    = np.zeros(n_skaters, dtype=np.int32)
    p_a2    = np.zeros(n_skaters, dtype=np.int32)
    standings = {t: {"wins": 0, "losses": 0, "otl": 0, "gf": 0, "ga": 0, "gp": 0}
                 for t in sim_engine.TEAM_TO_DIVISION}

    g_share  = dist["team_norm_g"]
    a1_share = dist["team_norm_a1"]
    a2_share = dist["team_norm_a2"]
    idx_arr = np.arange(n_skaters)
    has_any_g  = float(g_share.sum())  > 0
    has_any_a1 = float(a1_share.sum()) > 0
    has_any_a2 = float(a2_share.sum()) > 0

    custom_gp = 0
    for (home, away) in schedule_arr:
        out = sim_engine.simulate_game(home, away, ts, rng)
        h_g = out["home_score"]; a_g = out["away_score"]
        winner = out["winner"]; ot = out["ot"]
        # Standings
        standings[home]["gf"] += h_g; standings[home]["ga"] += a_g
        standings[away]["gf"] += a_g; standings[away]["ga"] += h_g
        standings[home]["gp"] += 1;   standings[away]["gp"] += 1
        if winner == home:
            standings[home]["wins"] += 1
            if ot: standings[away]["otl"] += 1
            else:  standings[away]["losses"] += 1
        else:
            standings[away]["wins"] += 1
            if ot: standings[home]["otl"] += 1
            else:  standings[home]["losses"] += 1

        # If custom team scored in this game, distribute their goals.
        if home == custom_team or away == custom_team:
            custom_gp += 1
            custom_goals = h_g if home == custom_team else a_g
            if custom_goals > 0 and has_any_g:
                # Sample with replacement — a player can score multiple times.
                scorers = np_rng.choice(idx_arr, size=int(custom_goals),
                                        replace=True, p=g_share)
                for sc in scorers:
                    p_goals[sc] += 1
                    # Determine assists for this goal
                    u = np_rng.random()
                    if u < UNASSISTED_RATE:
                        continue
                    elif u < UNASSISTED_RATE + SINGLE_ASSIST_RATE:
                        n_assists = 1
                    else:
                        n_assists = 2
                    # A1 sampled with the scorer excluded
                    if has_any_a1:
                        a1_pool = a1_share.copy(); a1_pool[sc] = 0
                        s_total = a1_pool.sum()
                        if s_total > 0:
                            a1_pool /= s_total
                            a1_idx = np_rng.choice(idx_arr, p=a1_pool)
                            p_a1[a1_idx] += 1
                        else:
                            a1_idx = -1
                    else:
                        a1_idx = -1
                    if n_assists >= 2 and has_any_a2 and a1_idx >= 0:
                        a2_pool = a2_share.copy(); a2_pool[sc] = 0; a2_pool[a1_idx] = 0
                        s_total = a2_pool.sum()
                        if s_total > 0:
                            a2_pool /= s_total
                            a2_idx = np_rng.choice(idx_arr, p=a2_pool)
                            p_a2[a2_idx] += 1
    return {
        "standings": standings,
        "player_goals": p_goals,
        "player_a1":    p_a1,
        "player_a2":    p_a2,
        "custom_gp": custom_gp,
    }


# ---------------------------------------------------------------------------
# Playoff tracking — which round did custom team reach?
# ---------------------------------------------------------------------------
def _custom_playoff_round(custom_team: str, standings: dict,
                          ts: dict[str, float],
                          rng: random.Random) -> str:
    """Returns one of:
       'miss' | 'r1_out' | 'r2_out' | 'cf_out' | 'final_out' | 'champion'
    """
    bracket = sim_engine.compute_seeding(standings)
    # First-round survival: search for custom_team in any series
    surviving = {"east": [], "west": []}
    custom_round = None  # 'r1','r2','cf','final' as we go

    found_r1 = False
    for conf in ("east", "west"):
        for (h, a) in bracket[conf]:
            if h == custom_team or a == custom_team:
                found_r1 = True
                higher = h
                res = sim_engine.simulate_series(h, a, ts, higher_seed=higher, rng=rng)
                if res["winner"] == custom_team:
                    custom_round = "r1"
                    # determine which conference they're in
                    custom_conf = conf
                    surviving[conf].append(res["winner"])
                else:
                    return "r1_out"
            else:
                higher = h
                res = sim_engine.simulate_series(h, a, ts, higher_seed=higher, rng=rng)
                surviving[conf].append(res["winner"])
    if not found_r1:
        return "miss"

    # Round 2 (each conf has 4 winners now in bracket order)
    r2_survivors = {"east": [], "west": []}
    for conf in ("east", "west"):
        s = surviving[conf]
        for h, a in ((s[0], s[1]), (s[2], s[3])):
            higher = h if ts.get(h, 0) >= ts.get(a, 0) else a
            res = sim_engine.simulate_series(h, a, ts, higher_seed=higher, rng=rng)
            if (h == custom_team or a == custom_team):
                if res["winner"] == custom_team:
                    custom_round = "r2"
                    r2_survivors[conf].append(res["winner"])
                else:
                    return "r2_out"
            else:
                r2_survivors[conf].append(res["winner"])

    # Conference finals
    conf_winners = {}
    for conf in ("east", "west"):
        s = r2_survivors[conf]
        if len(s) < 2: continue
        higher = s[0] if ts.get(s[0], 0) >= ts.get(s[1], 0) else s[1]
        res = sim_engine.simulate_series(s[0], s[1], ts, higher_seed=higher, rng=rng)
        if (s[0] == custom_team or s[1] == custom_team):
            if res["winner"] == custom_team:
                custom_round = "cf"
                conf_winners[conf] = res["winner"]
            else:
                return "cf_out"
        else:
            conf_winners[conf] = res["winner"]

    # Cup final
    e = conf_winners.get("east")
    w = conf_winners.get("west")
    if not e or not w: return custom_round or "miss"
    higher = e if ts.get(e, 0) >= ts.get(w, 0) else w
    res = sim_engine.simulate_series(e, w, ts, higher_seed=higher, rng=rng)
    if e == custom_team or w == custom_team:
        return "champion" if res["winner"] == custom_team else "final_out"
    return custom_round or "miss"


# ---------------------------------------------------------------------------
# Monte Carlo top-level
# ---------------------------------------------------------------------------
def _aggregate_player_table(dist: dict, total_goals: np.ndarray,
                            total_a1: np.ndarray, total_a2: np.ndarray,
                            custom_total_gp: int, n_sims: int) -> list[dict]:
    skaters = dist["skaters"]
    rows = []
    for i, s in enumerate(skaters):
        avg_g  = float(total_goals[i]) / n_sims
        avg_a1 = float(total_a1[i])    / n_sims
        avg_a2 = float(total_a2[i])    / n_sims
        avg_p  = avg_g + avg_a1 + avg_a2
        rows.append({
            "player_id": s["player_id"],
            "name": s["name"].strip(),
            "position": s["position"],
            "slot_kind": s["slot_kind"],
            "slot_idx": s["slot_idx"],
            "line_index": s["line_index"],
            "slot_minutes": s["slot_minutes"],
            "expected_toi": s["slot_minutes"],
            "expected_gp": 82,
            "avg_goals":     round(avg_g, 1),
            "avg_a1":        round(avg_a1, 1),
            "avg_a2":        round(avg_a2, 1),
            "avg_points":    round(avg_p, 1),
            "thin_sample": s["thin_sample"],
            "rated": s["rated"],
            "linemate_war": s["linemate_war"],
            "linemate_mult": s["linemate_mult"],
            "share_g_norm":  round(s["norm_share_g"], 4),
            "share_a1_norm": round(s["norm_share_a1"], 4),
            "share_a2_norm": round(s["norm_share_a2"], 4),
        })
    return rows


def _aggregate_team(custom_team: str, all_standings: list[dict],
                    all_playoff_outcomes: list[str], n_sims: int) -> dict:
    # Averages for the custom team
    wins = np.array([s[custom_team]["wins"] for s in all_standings])
    losses = np.array([s[custom_team]["losses"] for s in all_standings])
    otl = np.array([s[custom_team]["otl"] for s in all_standings])
    points = 2 * wins + otl
    gf = np.array([s[custom_team]["gf"] for s in all_standings])
    ga = np.array([s[custom_team]["ga"] for s in all_standings])

    # Division finish distribution
    from collections import Counter
    div = sim_engine.TEAM_TO_DIVISION[custom_team]
    div_teams = sim_engine.DIVISIONS[div]
    finish_counts = Counter()
    for st in all_standings:
        ranked = sorted(div_teams,
                        key=lambda t: (sim_engine._points_from_record(st[t]),
                                        st[t].get("gf", 0)),
                        reverse=True)
        finish_counts[ranked.index(custom_team) + 1] += 1
    finish_dist = {f"finish_{rank}": round(finish_counts.get(rank, 0) / n_sims, 4)
                   for rank in range(1, len(div_teams) + 1)}

    # Playoff outcome distribution
    out_counts = Counter(all_playoff_outcomes)
    playoff_pct = sum(out_counts[k] for k in
                       ("r1_out", "r2_out", "cf_out", "final_out", "champion")) / n_sims

    return {
        "team": custom_team,
        "n_sims": n_sims,
        "avg_wins":  round(float(wins.mean()), 1),
        "avg_losses": round(float(losses.mean()), 1),
        "avg_otl": round(float(otl.mean()), 1),
        "avg_points": round(float(points.mean()), 1),
        "avg_gf": round(float(gf.mean()), 1),
        "avg_ga": round(float(ga.mean()), 1),
        "playoff_pct": round(playoff_pct, 4),
        "division_finish": finish_dist,
        "outcome_distribution": {
            "miss":       round(out_counts.get("miss", 0)       / n_sims, 4),
            "r1_out":     round(out_counts.get("r1_out", 0)     / n_sims, 4),
            "r2_out":     round(out_counts.get("r2_out", 0)     / n_sims, 4),
            "cf_out":     round(out_counts.get("cf_out", 0)     / n_sims, 4),
            "final_out":  round(out_counts.get("final_out", 0)  / n_sims, 4),
            "champion":   round(out_counts.get("champion", 0)   / n_sims, 4),
        },
    }


def monte_carlo_custom_season(custom_team: str,
                              custom_lineup: dict,
                              custom_team_strength: float,
                              n_sims: int = 250,
                              seed: int | None = None) -> dict:
    """Top-level entry point.

    Args:
      custom_team: 3-letter abbrev (e.g., 'EDM')
      custom_lineup: {"forwards":[…12 slots…], "defense":[…6…], "goalies":[…]}
      custom_team_strength: the recomputed `team_strength` for the lineup
                            (use the value returned by /api/team-strength-custom)
      n_sims: how many full-season simulations to run
      seed: optional RNG seed for reproducibility

    Returns dict with `regular_season`, `player_stats`, `playoff_outcome`
    blocks (plus timing). The caller can run this twice (once with default
    lineup, once custom) and diff for the "Compare to default" toggle.
    """
    if seed is None:
        seed = random.randint(0, 10**9)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    t0 = time.time()
    shares_df, meta = load_shares()
    comp_df = load_composite()
    ts = load_team_strengths()
    # Override the custom team's strength with the user's lineup-adjusted value
    ts = dict(ts)
    ts[custom_team] = custom_team_strength
    baseline_war = compute_baseline_linemate_war(comp_df)

    schedule_arr = [(g["home"], g["away"]) for g in fetch_schedule()]

    dist = build_player_distribution(custom_lineup, shares_df, comp_df, meta,
                                      baseline_war)

    n_skaters = len(dist["skaters"])
    total_goals = np.zeros(n_skaters, dtype=np.int64)
    total_a1    = np.zeros(n_skaters, dtype=np.int64)
    total_a2    = np.zeros(n_skaters, dtype=np.int64)
    all_standings = []
    all_playoff_outcomes = []
    custom_total_gp = 0

    for _ in range(n_sims):
        season = simulate_one_season(custom_team, dist, schedule_arr, ts,
                                      rng, np_rng)
        all_standings.append(season["standings"])
        total_goals += season["player_goals"]
        total_a1    += season["player_a1"]
        total_a2    += season["player_a2"]
        custom_total_gp += season["custom_gp"]
        out = _custom_playoff_round(custom_team, season["standings"], ts, rng)
        all_playoff_outcomes.append(out)

    elapsed = time.time() - t0
    team_block = _aggregate_team(custom_team, all_standings,
                                  all_playoff_outcomes, n_sims)
    player_rows = _aggregate_player_table(dist, total_goals, total_a1, total_a2,
                                          custom_total_gp, n_sims)
    player_rows.sort(key=lambda r: r["avg_points"], reverse=True)

    return {
        "team": custom_team,
        "team_strength_used": round(custom_team_strength, 4),
        "n_sims": n_sims,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "baseline_linemate_war": round(baseline_war, 4),
        "regular_season": {
            "avg_wins":   team_block["avg_wins"],
            "avg_losses": team_block["avg_losses"],
            "avg_otl":    team_block["avg_otl"],
            "avg_points": team_block["avg_points"],
            "avg_gf":     team_block["avg_gf"],
            "avg_ga":     team_block["avg_ga"],
            "playoff_pct": team_block["playoff_pct"],
            "division_finish": team_block["division_finish"],
        },
        "player_stats": player_rows,
        "playoff_outcome": team_block["outcome_distribution"],
    }
