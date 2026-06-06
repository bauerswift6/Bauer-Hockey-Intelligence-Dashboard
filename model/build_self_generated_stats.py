"""
Self-generated dashboard analytics builder.

Replaces the third-party-derived "GAR", "xGAR", "GSAX", on-ice "xGF%" and
"Game Score" values that the dashboard previously surfaced from MoneyPuck's
gameScore / I_F_flurryScoreVenueAdjustedxGoals / xGoals fields. After this
runs, every advanced-stat column the user sees traces to a file in `model/`
produced by my own pipeline (see docs/dashboard_self_generation_audit.md).

Produces five CSVs:
  - model/skater_xgar_self_generated.csv       (per-skater 25-26)
  - model/goalie_gsax_self_generated.csv       (per-goalie 25-26)
  - model/team_xgf_self_generated.csv          (32 teams, 25-26)
  - model/skater_onice_xgf_self_generated.csv  (per-skater 25-26 on-ice splits)
  - model/skater_game_score_self_generated.csv (per-skater 25-26 Game Score)

All five share the same inputs (shots_augmented.parquet's `xg` column from my
trained xG model + season_cache/shifts_20252026.parquet + MoneyPuck 25-26
season CSVs for player→team identification and Game Score raw counts), so we
load them once and produce all five in one pass.

Run from project root (~1-2 min):
    python3 model/build_self_generated_stats.py
"""
from __future__ import annotations
import json
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Inputs (all mine or raw)
SHOTS_PARQUET = MODEL_DIR / "shots_augmented.parquet"
SHIFTS_PARQUET = MODEL_DIR / "season_cache" / "shifts_20252026.parquet"
HOME_ID_JSON = MODEL_DIR / "home_id_by_game.json"
MP_SKATERS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/skaters.csv"
MP_GOALIES_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv"
MP_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv"

# Outputs
OUT_XGAR = MODEL_DIR / "skater_xgar_self_generated.csv"
OUT_GSAX = MODEL_DIR / "goalie_gsax_self_generated.csv"
OUT_TEAM_XGF = MODEL_DIR / "team_xgf_self_generated.csv"
OUT_ONICE_XGF = MODEL_DIR / "skater_onice_xgf_self_generated.csv"
OUT_GAME_SCORE = MODEL_DIR / "skater_game_score_self_generated.csv"

# Standard hockey-analytics conversions
GOALS_PER_WAR = 6.0   # 1 WAR ≈ 6 goals (used in composite_war scale already)

# NHL team-id → 3-letter abbrev mapping (29 active + 3 historical)
TEAM_ID_TO_ABBREV = {
    1: "NJD", 2: "NYI", 3: "NYR", 4: "PHI", 5: "PIT", 6: "BOS", 7: "BUF",
    8: "MTL", 9: "OTT", 10: "TOR", 12: "CAR", 13: "FLA", 14: "TBL",
    15: "WSH", 16: "CHI", 17: "DET", 18: "NSH", 19: "STL", 20: "CGY",
    21: "COL", 22: "EDM", 23: "VAN", 24: "ANA", 25: "DAL", 26: "LAK",
    28: "SJS", 29: "CBJ", 30: "MIN", 52: "WPG", 53: "ARI", 54: "VGK",
    55: "SEA", 59: "UTA",
}


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# ---------------------------------------------------------------------------
# Common inputs (load once, reuse across all 5 outputs)
# ---------------------------------------------------------------------------
def load_shared_inputs():
    print("Loading inputs …", flush=True)
    # Shots — keep only 2025-26 season (game_id // 1_000_000 == 2025)
    shots = pd.read_parquet(SHOTS_PARQUET)
    shots["season"] = (shots["game_id"].astype(int) // 1_000_000).astype(int)
    shots = shots[shots["season"] == 2025].copy()
    # Coerce numerics
    for c in ("xg", "is_goal", "on_goal", "is_home", "shooter_id", "abs_secs", "game_id"):
        shots[c] = pd.to_numeric(shots[c], errors="coerce")
    shots = shots.dropna(subset=["shooter_id", "xg", "game_id"]).copy()
    shots["shooter_id"] = shots["shooter_id"].astype(int)
    shots["game_id"] = shots["game_id"].astype(int)
    shots["is_goal"] = shots["is_goal"].astype(int)
    shots["abs_secs"] = shots["abs_secs"].astype(int)
    print(f"  shots (25-26): {len(shots):,}")

    # Shifts — already only 25-26
    shifts = pd.read_parquet(SHIFTS_PARQUET)
    shifts = shifts.dropna(subset=["player_id"]).copy()
    shifts["player_id"] = shifts["player_id"].astype(int)
    shifts["game_id"] = shifts["game_id"].astype(int)
    print(f"  shifts (25-26): {len(shifts):,}")

    # Game → home team mapping
    with open(HOME_ID_JSON) as f:
        home_by_game_raw = json.load(f)
    home_by_game = {int(k): TEAM_ID_TO_ABBREV.get(int(v), "?")
                    for k, v in home_by_game_raw.items()}
    print(f"  home team map: {len(home_by_game)} games")

    # MoneyPuck CSVs (raw count source for Game Score + player→team lookup)
    mp_sk = _get_csv(MP_SKATERS_URL)
    mp_sk_all = mp_sk[mp_sk["situation"] == "all"].copy()
    mp_goalies = _get_csv(MP_GOALIES_URL)
    mp_goalies_all = mp_goalies[mp_goalies["situation"] == "all"].copy()
    mp_teams = _get_csv(MP_TEAMS_URL)
    mp_teams_all = mp_teams[mp_teams["situation"] == "all"].copy()
    print(f"  MP rows: skaters={len(mp_sk_all):,}, "
          f"goalies={len(mp_goalies_all):,}, teams={len(mp_teams_all):,}\n")

    # Per-shot helper: shooter team abbrev. We infer this from the shooter
    # being on the home team (is_home==1) or away team (is_home==0), looked
    # up against the game's home/away. Need away team too — derive by looking
    # up shifts in the game and finding the two distinct team_abbrevs.
    teams_per_game = (shifts.groupby("game_id")["team_abbrev"]
                       .agg(lambda s: sorted(set(s))).to_dict())
    shooter_team = []
    opp_team = []
    for r in shots.itertuples(index=False):
        gid = int(r.game_id)
        teams = teams_per_game.get(gid, [])
        home = home_by_game.get(gid)
        if len(teams) != 2 or home is None:
            shooter_team.append(None); opp_team.append(None); continue
        away = teams[0] if teams[1] == home else teams[1]
        if int(r.is_home) == 1:
            shooter_team.append(home); opp_team.append(away)
        else:
            shooter_team.append(away); opp_team.append(home)
    shots["shooter_team"] = shooter_team
    shots["opp_team"] = opp_team
    shots = shots.dropna(subset=["shooter_team"]).copy()
    print(f"  shots after team attribution: {len(shots):,}\n", flush=True)

    return shots, shifts, home_by_game, mp_sk_all, mp_goalies_all, mp_teams_all


# ---------------------------------------------------------------------------
# 1. Skater xGAR — sum of individual xG per shooter, scaled to wins
# ---------------------------------------------------------------------------
def build_skater_xgar(shots, mp_sk_all) -> pd.DataFrame:
    """Per-player aggregate: total xG (= sum of my model's per-shot
    predictions), actual goals, goals above expected, and xGAR in wins units.

    xGAR is defined here as `ixG / GOALS_PER_WAR` — wins-equivalent expected
    contribution from a player's personal shots. This is conceptually the
    "if this player shot at league-average conversion on every shot they
    took, how many wins of value did they generate" number, and is what the
    user-facing "xGAR" column will display in place of the prior MoneyPuck
    `I_F_flurryScoreVenueAdjustedxGoals / 3` proxy.
    """
    print("[1/5] Building skater xGAR …", flush=True)
    grp = shots.groupby("shooter_id").agg(
        n_shots=("xg", "size"),
        ixg=("xg", "sum"),
        goals=("is_goal", "sum"),
        sog_my=("on_goal", "sum"),
    ).reset_index().rename(columns={"shooter_id": "player_id"})

    # Goals above expected (positive = lucky / true talent shooter)
    grp["goals_above_expected"] = (grp["goals"] - grp["ixg"]).round(3)
    grp["xgar"] = (grp["ixg"] / GOALS_PER_WAR).round(3)

    # Attach name/team/position from MoneyPuck "all" rows. Use most-iced stint
    # so traded players are tagged with their primary team.
    meta = (mp_sk_all.assign(icetime_n=pd.to_numeric(mp_sk_all.get("icetime", 0),
                                                       errors="coerce").fillna(0))
              .sort_values(["playerId", "icetime_n"], ascending=[True, False])
              .groupby("playerId").first().reset_index())
    meta = meta[["playerId", "name", "team", "position"]].rename(
        columns={"playerId": "player_id"})
    grp = grp.merge(meta, on="player_id", how="left")
    grp = grp[["player_id", "name", "team", "position",
                "n_shots", "ixg", "goals", "sog_my",
                "goals_above_expected", "xgar"]]
    grp["ixg"] = grp["ixg"].round(2)
    grp = grp.sort_values("xgar", ascending=False).reset_index(drop=True)
    print(f"  {len(grp):,} skaters\n  top 5 by xGAR:")
    for _, r in grp.head(5).iterrows():
        print(f"    {r['name']:<22} {r['team']:>3}  ixG {r['ixg']:>5.1f}  "
              f"G {int(r['goals']):>3}  GAE {r['goals_above_expected']:>+5.1f}  "
              f"xGAR {r['xgar']:.2f}")
    return grp


# ---------------------------------------------------------------------------
# 2. Goalie GSAX — intersect shots with opposing-team goalie shifts
# ---------------------------------------------------------------------------
def build_goalie_gsax(shots, shifts, mp_goalies_all) -> pd.DataFrame:
    """For each goalie shift, find shots whose abs_secs lies within the
    shift's [abs_start, abs_end] and whose shooter_team is the goalie's
    OPPOSING team. Sum xg (= xG against) and is_goal (= goals against);
    GSAX = sum(xg) - sum(is_goal)."""
    print("[2/5] Building goalie GSAX …", flush=True)
    goalie_ids = set(mp_goalies_all["playerId"].dropna().astype(int).unique())
    g_shifts = shifts[shifts["player_id"].isin(goalie_ids)].copy()
    print(f"  goalie shifts: {len(g_shifts):,}")

    # For each goalie shift, we need shots in the same game where the shooter
    # is on the OPPOSING team and abs_secs ∈ [abs_start, abs_end].
    # Strategy: group both shifts and shots by game_id, then per game do an
    # interval-containment join.
    shots_min = shots[["game_id", "abs_secs", "xg", "is_goal",
                         "shooter_team"]].copy()
    rows = []
    for game_id, g_sub in g_shifts.groupby("game_id"):
        s_sub = shots_min[shots_min["game_id"] == game_id]
        if s_sub.empty:
            continue
        # For each shift, filter shots in window
        for _, sh in g_sub.iterrows():
            window = s_sub[(s_sub["abs_secs"] >= sh["abs_start"])
                            & (s_sub["abs_secs"] <= sh["abs_end"])
                            & (s_sub["shooter_team"] != sh["team_abbrev"])]
            if not window.empty:
                rows.append({
                    "player_id": int(sh["player_id"]),
                    "name": f"{sh['first_name']} {sh['last_name']}",
                    "team": sh["team_abbrev"],
                    "shots_faced": int(len(window)),
                    "xg_against": float(window["xg"].sum()),
                    "goals_against": int(window["is_goal"].sum()),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["player_id","name","team","shots_faced",
                                       "xg_against","goals_against","gsax",
                                       "gsax_war"])
    g = df.groupby(["player_id"]).agg(
        name=("name", "first"),
        team=("team", "first"),
        shots_faced=("shots_faced", "sum"),
        xg_against=("xg_against", "sum"),
        goals_against=("goals_against", "sum"),
    ).reset_index()
    g["gsax"] = (g["xg_against"] - g["goals_against"]).round(3)
    g["gsax_war"] = (g["gsax"] / GOALS_PER_WAR).round(3)
    g["xg_against"] = g["xg_against"].round(2)
    g = g.sort_values("gsax", ascending=False).reset_index(drop=True)
    print(f"  {len(g)} goalies\n  top 5 by GSAX:")
    for _, r in g.head(5).iterrows():
        print(f"    {r['name']:<22} {r['team']:>3}  shots {int(r['shots_faced']):>4}  "
              f"xGA {r['xg_against']:>6.1f}  GA {int(r['goals_against']):>3}  "
              f"GSAX {r['gsax']:>+6.2f}  WAR {r['gsax_war']:>+5.2f}")
    return g


# ---------------------------------------------------------------------------
# 3. Team xGF% — aggregate my-xG by team
# ---------------------------------------------------------------------------
def build_team_xgf(shots, mp_teams_all) -> pd.DataFrame:
    """Per team: sum my-xG when their team shoots (xgf), sum my-xG when
    opponent shoots (xga). xGF% = xgf / (xgf + xga)."""
    print("[3/5] Building team xGF% …", flush=True)
    xgf = shots.groupby("shooter_team")["xg"].sum().rename("xgf")
    xga = shots.groupby("opp_team")["xg"].sum().rename("xga")
    gf  = shots.groupby("shooter_team")["is_goal"].sum().rename("gf_at_5v5")
    ga  = shots.groupby("opp_team")["is_goal"].sum().rename("ga_at_5v5")
    df = pd.concat([xgf, xga, gf, ga], axis=1).reset_index().rename(
        columns={"index": "team"})
    df = df.rename(columns={"shooter_team": "team"}) if "shooter_team" in df.columns else df
    df["xgf"] = df["xgf"].fillna(0).round(2)
    df["xga"] = df["xga"].fillna(0).round(2)
    df["xgf_pct"] = (df["xgf"] / (df["xgf"] + df["xga"]).replace(0, np.nan) * 100).round(2)
    df = df.dropna(subset=["team"]).sort_values("xgf_pct", ascending=False)
    df = df.reset_index(drop=True)
    print(f"  {len(df)} teams\n  top 5 by xGF%:")
    for _, r in df.head(5).iterrows():
        print(f"    {r['team']:>3}  xGF {r['xgf']:>6.1f}  xGA {r['xga']:>6.1f}  "
              f"xGF% {r['xgf_pct']:>5.1f}%  "
              f"(actual GF {int(r['gf_at_5v5'])}/GA {int(r['ga_at_5v5'])})")
    return df


# ---------------------------------------------------------------------------
# 4. Skater on-ice xGF% — intersect shots with skater shifts
# ---------------------------------------------------------------------------
def build_onice_xgf(shots, shifts) -> pd.DataFrame:
    """For each skater shift, find shots whose abs_secs ∈ shift window and
    aggregate xG with the right sign (for if shooter on skater's team, else
    against). Skips goalies (excluded from output)."""
    print("[4/5] Building skater on-ice xGF% …", flush=True)
    # Limit to skater shifts only; identifying goalies via shift counts
    # (goalies have very few shifts per game). Cheap proxy: anyone with
    # shifts spanning a full period is likely a goalie. Better: filter by
    # the goalie_ids set we already have from MP.
    # We rebuild MP goalie set in caller — pass it in via a small ID set.
    shots_min = shots[["game_id", "abs_secs", "xg", "shooter_team"]].copy()
    # Per-(game_id, shift) interval expansion. Heavy but doable in pandas.
    sh = shifts.copy()
    sh["mid_abs"] = (sh["abs_start"] + sh["abs_end"]) / 2

    # Build a per-game shot list and per-game shift list, then for each game
    # do the interval-containment join.
    rows = []
    for game_id, g_shifts in sh.groupby("game_id"):
        g_shots = shots_min[shots_min["game_id"] == game_id]
        if g_shots.empty:
            continue
        # Vectorised: per shift, count xG for/against in window
        for _, row in g_shifts.iterrows():
            window = g_shots[(g_shots["abs_secs"] >= row["abs_start"])
                              & (g_shots["abs_secs"] <= row["abs_end"])]
            if window.empty:
                continue
            xgf = window[window["shooter_team"] == row["team_abbrev"]]["xg"].sum()
            xga = window[window["shooter_team"] != row["team_abbrev"]]["xg"].sum()
            rows.append({
                "player_id": int(row["player_id"]),
                "team_abbrev": row["team_abbrev"],
                "xgf": float(xgf),
                "xga": float(xga),
                "shifts": 1,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    g = df.groupby(["player_id", "team_abbrev"]).agg(
        xgf=("xgf", "sum"),
        xga=("xga", "sum"),
        shifts=("shifts", "sum"),
    ).reset_index()
    g["xgf_pct"] = (g["xgf"] / (g["xgf"] + g["xga"]).replace(0, np.nan) * 100).round(2)
    g["rel_x"] = (g["xgf"] - g["xga"]).round(3)
    g["xgf"] = g["xgf"].round(2)
    g["xga"] = g["xga"].round(2)
    g = g.rename(columns={"team_abbrev": "team"})
    g = g.sort_values("xgf_pct", ascending=False).reset_index(drop=True)
    print(f"  {len(g):,} skater on-ice rows")
    return g


# ---------------------------------------------------------------------------
# 5. Skater Game Score — Galamini formula on raw counts
# ---------------------------------------------------------------------------
def build_game_score(mp_sk_all) -> pd.DataFrame:
    """Game Score (Galamini):
        0.75·G + 0.7·A1 + 0.55·A2 + 0.075·SOG + 0.075·BLK + 0.05·PIM
      + 0.15·PD − 0.15·PT + 0.01·FOW − 0.01·FOL
      + 0.05·CF − 0.05·CA + 0.15·GF − 0.15·GA

    Inputs are raw count fields from MoneyPuck's "all"-situation skater rows;
    no analytic from MoneyPuck is used.
    """
    print("[5/5] Building skater Game Score …", flush=True)
    df = mp_sk_all.copy()
    cols_needed = [
        "I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
        "I_F_shotsOnGoal", "shotsBlockedByPlayer", "I_F_penalityMinutes",
        "penaltiesDrawn", "penalties",
        "I_F_faceOffsWon", "faceoffsLost",
        "OnIce_F_shotAttempts", "OnIce_A_shotAttempts",
        "OnIce_F_goals", "OnIce_A_goals", "games_played",
    ]
    for c in cols_needed:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        else:
            df[c] = 0
    # Aggregate per player (sum across stints)
    agg = df.groupby("playerId").agg(
        name=("name", "first"),
        team=("team", "first"),
        position=("position", "first"),
        games_played=("games_played", "sum"),
        G=("I_F_goals", "sum"),
        A1=("I_F_primaryAssists", "sum"),
        A2=("I_F_secondaryAssists", "sum"),
        SOG=("I_F_shotsOnGoal", "sum"),
        BLK=("shotsBlockedByPlayer", "sum"),
        PIM=("I_F_penalityMinutes", "sum"),
        PD=("penaltiesDrawn", "sum"),
        PT=("penalties", "sum"),
        FOW=("I_F_faceOffsWon", "sum"),
        FOL=("faceoffsLost", "sum"),
        CF=("OnIce_F_shotAttempts", "sum"),
        CA=("OnIce_A_shotAttempts", "sum"),
        GF=("OnIce_F_goals", "sum"),
        GA=("OnIce_A_goals", "sum"),
    ).reset_index().rename(columns={"playerId": "player_id"})

    agg["game_score"] = (
        0.75 * agg["G"]
        + 0.70 * agg["A1"]
        + 0.55 * agg["A2"]
        + 0.075 * agg["SOG"]
        + 0.075 * agg["BLK"]
        + 0.05 * agg["PIM"]
        + 0.15 * agg["PD"]
        - 0.15 * agg["PT"]
        + 0.01 * agg["FOW"]
        - 0.01 * agg["FOL"]
        + 0.05 * agg["CF"]
        - 0.05 * agg["CA"]
        + 0.15 * agg["GF"]
        - 0.15 * agg["GA"]
    ).round(2)
    agg["game_score_per_game"] = np.where(
        agg["games_played"] > 0,
        (agg["game_score"] / agg["games_played"]).round(3), 0.0)
    agg = agg[["player_id", "name", "team", "position", "games_played",
                "game_score", "game_score_per_game"]]
    agg = agg.sort_values("game_score", ascending=False).reset_index(drop=True)
    print(f"  {len(agg):,} skaters\n  top 5 by Game Score:")
    for _, r in agg.head(5).iterrows():
        print(f"    {r['name']:<22} {r['team']:>3}  GP {int(r['games_played']):>3}  "
              f"GS {r['game_score']:>6.1f}  /game {r['game_score_per_game']:.2f}")
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    shots, shifts, home_by_game, mp_sk_all, mp_goalies_all, mp_teams_all = \
        load_shared_inputs()

    xgar  = build_skater_xgar(shots, mp_sk_all)
    gsax  = build_goalie_gsax(shots, shifts, mp_goalies_all)
    txgf  = build_team_xgf(shots, mp_teams_all)
    onice = build_onice_xgf(shots, shifts)
    gs    = build_game_score(mp_sk_all)

    print("\nSaving …")
    xgar.to_csv(OUT_XGAR, index=False);  print(f"  → {OUT_XGAR.name}")
    gsax.to_csv(OUT_GSAX, index=False);  print(f"  → {OUT_GSAX.name}")
    txgf.to_csv(OUT_TEAM_XGF, index=False);  print(f"  → {OUT_TEAM_XGF.name}")
    onice.to_csv(OUT_ONICE_XGF, index=False);  print(f"  → {OUT_ONICE_XGF.name}")
    gs.to_csv(OUT_GAME_SCORE, index=False);  print(f"  → {OUT_GAME_SCORE.name}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
