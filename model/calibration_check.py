"""
Phase 5.1 — Comprehensive calibration check.

Diagnostic script ONLY. Does NOT modify the simulator or any model code.

Builds a per-player calibration dataset by:
  1. Running 32 default-roster season simulations (250 sims each) and
     collecting per-player simulated stats (G / A1 / A2 / P).
  2. Pulling actual season stats from the same MoneyPuck CSVs the
     simulator trains on (2023-24, 2024-25, 2025-26) and computing
     year-weighted per-season production using the same 1.00/0.35/0.10
     weights now in `player_offense_shares.csv`.
  3. Computing per-player deltas (simulated minus actual) for G/A1/A2/P
     and the delta-as-percentage-of-actual.
  4. Aggregating by tier, position, team strength, archetype.
  5. Identifying the 20 largest negative/positive outliers.
  6. Tracing the pipeline for McDavid / Draisaitl / MacKinnon.

Outputs:
  model/phase_5_1_calibration_check.csv  — per-player calibration row
  STDOUT report                           — patterns + outliers + traces
                                           + root-cause hypotheses

Run from project root:
    python3 model/calibration_check.py
"""
from __future__ import annotations
import json
import sys
import time
from io import StringIO
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

import simulate_season as ss
import build_team_strength as bts

OUT_CSV = MODEL_DIR / "phase_5_1_calibration_check.csv"
N_SIMS = 250
MP_URL = ("https://moneypuck.com/moneypuck/playerData/seasonSummary/"
          "{season}/regular/{kind}.csv")
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Matches the live build_player_offense_shares.py
SEASONS = ["2023", "2024", "2025"]
SEASON_WEIGHTS = {"2025": 1.00, "2024": 0.35, "2023": 0.10}
MIN_GP_OVER_WINDOW = 41

TEAMS = bts.TEAMS


# ---------------------------------------------------------------------------
# Phase 1 — Simulated dataset (32 teams × 250 sims)
# ---------------------------------------------------------------------------
def _build_team_lineup(team: str, comp_lookup: dict,
                       toi_lookup: dict | None = None) -> dict:
    """Default lineup for a team: top-12 F + top-6 D + top-2 G by historical
    avg_toi_per_game, identical to what the dashboard Lineup Editor shows on
    first load."""
    roster = bts.fetch_roster(team)
    fwd = sorted(bts._attach_war([p for p in roster if p["roster_role"] == "F"],
                                   comp_lookup, toi_lookup),
                 key=lambda p: p["avg_toi_per_game"], reverse=True)
    dfn = sorted(bts._attach_war([p for p in roster if p["roster_role"] == "D"],
                                   comp_lookup, toi_lookup),
                 key=lambda p: p["avg_toi_per_game"], reverse=True)
    glx = [p for p in roster if p["roster_role"] == "G"][:2]
    return {
        "forwards": fwd[:12] + [None] * max(0, 12 - len(fwd)),
        "defense":  dfn[:6]  + [None] * max(0, 6 - len(dfn)),
        "goalies":  glx,
    }


def run_all_team_simulations(n_sims: int = 250) -> pd.DataFrame:
    """For each of the 32 teams, run a default-roster season Monte Carlo
    and collect per-player simulated stats. Returns a flat per-player
    DataFrame."""
    print(f"Phase 1 — running {len(TEAMS)} team simulations × {n_sims} sims …",
          flush=True)
    comp_lookup = bts.load_composite_lookup()
    toi_lookup = bts.load_toi_lookup()
    ts_df = pd.read_csv(MODEL_DIR / "team_strength.csv")
    ts_lookup = dict(zip(ts_df["team"], ts_df["team_strength"]))

    rows = []
    sim_team_gf = {}    # team → avg sim GF/season (for diagnostic Step 7)
    sim_team_ga = {}
    sim_team_wins = {}
    for i, team in enumerate(TEAMS, 1):
        t0 = time.time()
        lineup = _build_team_lineup(team, comp_lookup, toi_lookup)
        ts = ts_lookup.get(team, 0.0)
        res = ss.monte_carlo_custom_season(team, lineup, ts,
                                            n_sims=n_sims, seed=42)
        rs = res["regular_season"]
        sim_team_gf[team] = rs["avg_gf"]
        sim_team_ga[team] = rs["avg_ga"]
        sim_team_wins[team] = rs["avg_wins"]
        for p in res["player_stats"]:
            if p["player_id"] is None:
                continue
            rows.append({
                "team": team,
                "player_id": int(p["player_id"]),
                "player_name": p["name"].strip(),
                "position": p["position"],
                "slot_kind": p["slot_kind"],
                "slot_idx": p["slot_idx"],
                "line_index": p["line_index"],
                "slot_minutes": p["slot_minutes"],
                "thin_sample": p["thin_sample"],
                "rated": p["rated"],
                "linemate_war": p["linemate_war"],
                "linemate_mult": p["linemate_mult"],
                "share_g_norm":  p["share_g_norm"],
                "share_a1_norm": p["share_a1_norm"],
                "share_a2_norm": p["share_a2_norm"],
                "sim_goals":    p["avg_goals"],
                "sim_a1":       p["avg_a1"],
                "sim_a2":       p["avg_a2"],
                "sim_points":   p["avg_points"],
                "sim_team_gf":  rs["avg_gf"],
                "sim_team_ga":  rs["avg_ga"],
            })
        print(f"  [{i:>2}/{len(TEAMS)}]  {team}  done in {time.time()-t0:.1f}s",
              flush=True)
    df = pd.DataFrame(rows)
    print(f"\n  Collected {len(df):,} (team, player) simulated rows.\n",
          flush=True)
    return df, sim_team_gf, sim_team_ga, sim_team_wins


# ---------------------------------------------------------------------------
# Phase 2 — Actual baseline (3-season weighted average from MoneyPuck)
# ---------------------------------------------------------------------------
def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def build_actual_baseline() -> pd.DataFrame:
    """For every player who appears in MoneyPuck's 23-24 / 24-25 / 25-26
    CSVs, compute weighted-average per-season production using the same
    1.00 / 0.35 / 0.10 weights as the simulator. Empty-net excluded; we
    use the "all" situation for goal/assist totals (matches how the
    simulator's xG model treats goals).

    Returns a player-indexed table with columns:
      player_id, player_name, team_last, position,
      actual_goals, actual_a1, actual_a2, actual_points,
      actual_gp_total, actual_seasons_present, actual_gp_weighted_avg
    """
    print("Phase 2 — pulling 3 MoneyPuck seasons + computing actual baselines …",
          flush=True)
    # Player-level accumulators (year-weighted sums and total-weight tracker)
    pacc: dict = {}    # pid → {g_w, a1_w, a2_w, gp_w, total_w_present}
    pmeta: dict = {}   # pid → (idx, name, team, position)
    for idx, code in enumerate(SEASONS):
        w = SEASON_WEIGHTS[code]
        try:
            sk = _get_csv(MP_URL.format(season=code, kind="skaters"))
        except Exception as e:
            print(f"  skipped season {code}: {e}", flush=True)
            continue
        sall = sk[sk["situation"] == "all"].copy()
        for c in ("I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                  "games_played", "icetime"):
            sall[c] = pd.to_numeric(sall[c], errors="coerce").fillna(0)
        # Per-player sum within this season (handles traded players whose
        # totals are split across multiple team-stint rows)
        merged = (sall.groupby("playerId")
                  [["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                    "games_played", "icetime"]].sum().reset_index())
        # Most recent team + name + position from the "all"-situation rows
        # ordered by icetime descending (i.e., the team where they got the
        # most ice).
        meta_rows = sall.sort_values(["playerId", "icetime"],
                                       ascending=[True, False])
        meta_first = meta_rows.groupby("playerId").first().reset_index()
        for r in merged.itertuples(index=False):
            try:
                pid = int(r.playerId)
            except (TypeError, ValueError):
                continue
            a = pacc.setdefault(pid, {"g_w": 0.0, "a1_w": 0.0, "a2_w": 0.0,
                                        "gp_w": 0.0, "total_w_present": 0.0,
                                        "gp_raw": 0.0})
            a["g_w"]  += w * float(r.I_F_goals)
            a["a1_w"] += w * float(r.I_F_primaryAssists)
            a["a2_w"] += w * float(r.I_F_secondaryAssists)
            a["gp_w"] += w * float(r.games_played)
            a["gp_raw"] += float(r.games_played)
            a["total_w_present"] += w
            meta_r = meta_first[meta_first["playerId"] == int(r.playerId)]
            if not meta_r.empty:
                pmeta[pid] = (idx,
                               meta_r.iloc[0].get("name", ""),
                               meta_r.iloc[0].get("team", ""),
                               meta_r.iloc[0].get("position", ""))
        print(f"  pulled {code}  (weight {w}, {len(merged):,} players)",
              flush=True)

    rows = []
    for pid, a in pacc.items():
        m = pmeta.get(pid, (None, "", "", ""))
        _, name, team, pos = m
        # Weighted-avg per-season production:
        # For a player present in all 3 seasons, total_w_present = 1.45
        # For a sophomore (2 seasons), total_w_present = 1.35
        # For a rookie, 1.00
        if a["total_w_present"] <= 0:
            continue
        per_season_g  = a["g_w"]  / a["total_w_present"]
        per_season_a1 = a["a1_w"] / a["total_w_present"]
        per_season_a2 = a["a2_w"] / a["total_w_present"]
        per_season_gp = a["gp_w"] / a["total_w_present"]
        rows.append({
            "player_id": pid, "player_name": name,
            "team_actual_last": team, "position_actual": pos,
            "actual_goals":  round(per_season_g, 2),
            "actual_a1":     round(per_season_a1, 2),
            "actual_a2":     round(per_season_a2, 2),
            "actual_points": round(per_season_g + per_season_a1
                                    + per_season_a2, 2),
            "actual_gp_weighted_avg": round(per_season_gp, 1),
            "actual_gp_total": int(a["gp_raw"]),
            "actual_seasons_present": int(round(a["total_w_present"]
                                                  / max(1, SEASON_WEIGHTS["2025"]))),
        })
    df = pd.DataFrame(rows)
    print(f"\n  Built actual baseline for {len(df):,} players "
          f"(before the 41-GP minimum filter).\n", flush=True)
    return df


# ---------------------------------------------------------------------------
# Phase 3 — Compute deltas
# ---------------------------------------------------------------------------
def compute_deltas(sim_df: pd.DataFrame, actual_df: pd.DataFrame
                   ) -> pd.DataFrame:
    """Join sim and actual on player_id; compute deltas. Players present in
    only one of the two sets get NaN for the absent side.

    Adds:
      delta_goals, delta_a1, delta_a2, delta_points
      pct_delta_points (None when actual_points is near zero)
    """
    # Drop duplicate name/position from actual to avoid pandas _x/_y suffixes
    actual_df = actual_df.drop(columns=["player_name"], errors="ignore")
    merged = sim_df.merge(actual_df, on="player_id", how="left")
    merged["delta_goals"]  = merged["sim_goals"]  - merged["actual_goals"]
    merged["delta_a1"]     = merged["sim_a1"]     - merged["actual_a1"]
    merged["delta_a2"]     = merged["sim_a2"]     - merged["actual_a2"]
    merged["delta_points"] = merged["sim_points"] - merged["actual_points"]
    merged["pct_delta_points"] = np.where(
        merged["actual_points"] > 5,
        (merged["sim_points"] - merged["actual_points"]) / merged["actual_points"],
        np.nan)
    return merged


# ---------------------------------------------------------------------------
# Phase 4 — Aggregate analysis
# ---------------------------------------------------------------------------
def aggregate_analysis(df: pd.DataFrame, sim_team_gf, sim_team_ga,
                       n_sims: int) -> dict:
    """Produce all the cuts the user wants. Operates only on qualified
    rows (≥41 GP across 3-yr window AND has actual baseline)."""
    qual = df[(df["actual_gp_total"] >= MIN_GP_OVER_WINDOW)
               & (df["actual_points"].notna())].copy()
    print(f"Phase 4 — qualified pool: {len(qual):,} (team, player) rows  "
          f"(of {len(df):,} simulated rows; filtered by ≥41 GP + has actual).",
          flush=True)

    out = {}
    out["n_qualified"] = int(len(qual))

    # League-wide
    out["league"] = {
        "mean_delta_goals":  float(qual["delta_goals"].mean()),
        "median_delta_goals": float(qual["delta_goals"].median()),
        "mean_delta_a1":     float(qual["delta_a1"].mean()),
        "median_delta_a1":   float(qual["delta_a1"].median()),
        "mean_delta_a2":     float(qual["delta_a2"].mean()),
        "median_delta_a2":   float(qual["delta_a2"].median()),
        "mean_delta_points": float(qual["delta_points"].mean()),
        "median_delta_points": float(qual["delta_points"].median()),
        "std_delta_points":   float(qual["delta_points"].std()),
        "pct_higher_sim":     float((qual["delta_points"] > 0).mean()),
        "pct_lower_sim":      float((qual["delta_points"] < 0).mean()),
    }

    # Total league goals (simulated vs actual). Total league goals per
    # SIMULATED SEASON = sum of sim_team_gf across all 32 teams divided by 2
    # (each goal counted once for the scoring team and once for "GA"
    # on the other side, but we sum GF only).
    total_sim_team_goals = sum(sim_team_gf.values())
    # Actual league goals per "weighted-average season": sum of each player's
    # year-weighted-per-season goals. This is the natural comparison since
    # the actual baseline is per-season.
    total_actual_player_goals = qual["actual_goals"].sum()
    total_sim_player_goals    = qual["sim_goals"].sum()
    out["league"]["total_sim_team_goals_per_season"] = float(total_sim_team_goals)
    out["league"]["total_sim_player_goals_per_season"] = float(total_sim_player_goals)
    out["league"]["total_actual_player_goals_per_season"] = float(total_actual_player_goals)
    out["league"]["sim_minus_actual_total_goals"] = float(
        total_sim_player_goals - total_actual_player_goals)
    out["league"]["sim_team_goals_avg_per_team"] = float(total_sim_team_goals / 32)

    # By tier (top 30 / 31-100 / 101-300 / 301+) — ranked by ACTUAL points desc
    rank_by_actual = qual.sort_values("actual_points", ascending=False).reset_index(drop=True)
    rank_by_actual["actual_rank"] = rank_by_actual.index + 1
    tiers = [("elite (top 30)",       1,   30),
             ("top-6 (31-100)",       31,  100),
             ("middle (101-300)",     101, 300),
             ("bottom (301+)",        301, 99999)]
    out["by_tier"] = {}
    for label, lo, hi in tiers:
        sub = rank_by_actual[(rank_by_actual["actual_rank"] >= lo)
                              & (rank_by_actual["actual_rank"] <= hi)]
        if sub.empty: continue
        out["by_tier"][label] = {
            "n": int(len(sub)),
            "mean_delta_goals":  float(sub["delta_goals"].mean()),
            "median_delta_goals": float(sub["delta_goals"].median()),
            "mean_delta_points": float(sub["delta_points"].mean()),
            "median_delta_points": float(sub["delta_points"].median()),
            "mean_pct_delta_points": float(sub["pct_delta_points"].mean()),
        }

    # By position (F/D split using the simulator's position assignment)
    qual["pos_group"] = qual["position"].str[0].map(
        lambda c: "D" if c == "D" else ("G" if c == "G" else "F"))
    out["by_position"] = {}
    for grp in ("F", "D"):
        sub = qual[qual["pos_group"] == grp]
        out["by_position"][grp] = {
            "n": int(len(sub)),
            "mean_delta_goals":  float(sub["delta_goals"].mean()),
            "mean_delta_points": float(sub["delta_points"].mean()),
            "median_delta_points": float(sub["delta_points"].median()),
            "mean_pct_delta_points": float(sub["pct_delta_points"].mean()),
        }
    # And refine F into C/L/R using exact position code
    for c, lbl in (("C", "F_centers"), ("L", "F_left_wings"), ("R", "F_right_wings")):
        sub = qual[qual["position"].str.startswith(c)]
        out["by_position"][lbl] = {
            "n": int(len(sub)),
            "mean_delta_points": float(sub["delta_points"].mean()),
            "median_delta_points": float(sub["delta_points"].median()),
        }

    # By team strength tier
    ts_df = pd.read_csv(MODEL_DIR / "team_strength.csv").sort_values(
        "team_strength", ascending=False).reset_index(drop=True)
    top10_teams = set(ts_df.head(10)["team"])
    bot10_teams = set(ts_df.tail(10)["team"])
    out["by_team_strength"] = {}
    for label, members in (("top 10 strongest", top10_teams),
                            ("middle 12",        set(ts_df["team"]) - top10_teams - bot10_teams),
                            ("bottom 10 weakest", bot10_teams)):
        sub = qual[qual["team"].isin(members)]
        out["by_team_strength"][label] = {
            "n_teams": len(members),
            "n_players": int(len(sub)),
            "mean_delta_points": float(sub["delta_points"].mean()),
            "median_delta_points": float(sub["delta_points"].median()),
            "mean_pct_delta_points": float(sub["pct_delta_points"].mean()),
        }

    # Archetypes — using player_offense_shares
    shares = pd.read_csv(MODEL_DIR / "player_offense_shares.csv")
    shares_indexed = shares.set_index("player_id")
    # Heavy shooters: top 30 by goal_share
    top_shooters = set(shares.sort_values("goal_share", ascending=False)
                       .head(30)["player_id"])
    # Pure playmakers: top 30 by p_assist_share / max(goal_share, .01) ratio
    shares["pm_ratio"] = shares["p_assist_share"] / shares["goal_share"].clip(lower=0.01)
    top_playmakers = set(shares[shares["position"].isin(["C","L","R"])]
                          .sort_values("pm_ratio", ascending=False)
                          .head(30)["player_id"])
    # Offensive D: top 30 D by (goal_share + p_assist_share)
    d_off = shares[shares["position"] == "D"].copy()
    d_off["off_total"] = d_off["goal_share"] + d_off["p_assist_share"]
    top_off_d = set(d_off.sort_values("off_total", ascending=False).head(30)["player_id"])
    # Defensive D: bottom-half D by off_total, top by ev_minutes (so true minute-eaters)
    low_off_d = d_off.sort_values("off_total").head(80)
    def_d = set(low_off_d.sort_values("ev_minutes_3yr", ascending=False).head(30)["player_id"])

    out["by_archetype"] = {}
    for label, members in (("heavy shooters (top 30 by goal_share)", top_shooters),
                            ("pure playmakers (top 30 by A1/G ratio)", top_playmakers),
                            ("offensive defensemen (top 30)", top_off_d),
                            ("defensive defensemen (top 30 by EV min, low off)", def_d)):
        sub = qual[qual["player_id"].isin(members)]
        if sub.empty: continue
        out["by_archetype"][label] = {
            "n": int(len(sub)),
            "mean_delta_goals":  float(sub["delta_goals"].mean()),
            "mean_delta_a1":     float(sub["delta_a1"].mean()),
            "mean_delta_a2":     float(sub["delta_a2"].mean()),
            "mean_delta_points": float(sub["delta_points"].mean()),
            "median_delta_points": float(sub["delta_points"].median()),
        }

    return out, rank_by_actual


# ---------------------------------------------------------------------------
# Phase 5 — Outliers
# ---------------------------------------------------------------------------
def find_outliers(df: pd.DataFrame, n: int = 20) -> dict:
    qual = df[(df["actual_gp_total"] >= MIN_GP_OVER_WINDOW)
               & (df["actual_points"].notna())].copy()
    most_under = qual.sort_values("delta_points").head(n)
    most_over  = qual.sort_values("delta_points", ascending=False).head(n)
    return {
        "most_under": most_under[["player_name", "team", "position",
                                    "actual_points", "sim_points",
                                    "delta_points", "pct_delta_points",
                                    "actual_gp_total", "slot_kind",
                                    "slot_idx", "thin_sample"]].to_dict("records"),
        "most_over":  most_over[["player_name", "team", "position",
                                    "actual_points", "sim_points",
                                    "delta_points", "pct_delta_points",
                                    "actual_gp_total", "slot_kind",
                                    "slot_idx", "thin_sample"]].to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Phase 6 — Pipeline trace for the 3 elites
# ---------------------------------------------------------------------------
def trace_pipeline(player_names: list[str], df: pd.DataFrame,
                   sim_team_gf: dict) -> list[dict]:
    """For each player, gather every step of the calculation chain."""
    shares = pd.read_csv(MODEL_DIR / "player_offense_shares.csv")
    traces = []
    for nm in player_names:
        row = df[df["player_name"].str.contains(nm, na=False, regex=False)]
        if row.empty:
            traces.append({"player": nm, "error": "not in calibration dataset"})
            continue
        r = row.iloc[0]
        sh = shares[shares["player_id"] == r["player_id"]]
        sh = sh.iloc[0] if not sh.empty else None
        team_g = sim_team_gf.get(r["team"], 0)
        # Theoretical points under perfect-conditions assumption:
        # G_th = team_g * normalized_g_share
        # A1_th = (team_g - G_th) * normalized_a1_share * 0.90 (assist-rate)
        # A2_th = (team_g - G_th) * normalized_a2_share * 0.65
        g_th  = team_g * float(r["share_g_norm"])
        a1_th = (team_g - g_th) * float(r["share_a1_norm"]) * 0.90
        a2_th = (team_g - g_th) * float(r["share_a2_norm"]) * 0.65
        traces.append({
            "player": nm,
            "team": r["team"],
            "actual": {"G": r["actual_goals"], "A1": r["actual_a1"],
                        "A2": r["actual_a2"], "P": r["actual_points"]},
            "raw_shares": {
                "hist_goal_share":     float(sh["goal_share"])     if sh is not None else None,
                "hist_p_assist_share": float(sh["p_assist_share"]) if sh is not None else None,
                "hist_s_assist_share": float(sh["s_assist_share"]) if sh is not None else None,
                "hist_avg_TOI":        float(sh["avg_toi_per_game"]) if sh is not None else None,
                "ev_minutes_3yr":      float(sh["ev_minutes_3yr"])  if sh is not None else None,
                "thin_sample":         bool(sh["thin_sample"])      if sh is not None else None,
            },
            "slot": {
                "kind": r["slot_kind"], "slot_idx": int(r["slot_idx"]),
                "slot_minutes": int(r["slot_minutes"]),
            },
            "layer_3_linemate": {
                "linemate_war": float(r["linemate_war"]),
                "linemate_mult": float(r["linemate_mult"]),
            },
            "normalized_shares": {
                "G":  float(r["share_g_norm"]),
                "A1": float(r["share_a1_norm"]),
                "A2": float(r["share_a2_norm"]),
            },
            "team_sim_GF_per_season": float(team_g),
            "theoretical_under_perfect_conditions": {
                "G":  round(g_th,  1),
                "A1": round(a1_th, 1),
                "A2": round(a2_th, 1),
                "P":  round(g_th + a1_th + a2_th, 1),
            },
            "simulator_actually_produces": {
                "G":  float(r["sim_goals"]),
                "A1": float(r["sim_a1"]),
                "A2": float(r["sim_a2"]),
                "P":  float(r["sim_points"]),
            },
            "gap": {
                "sim_vs_actual": round(float(r["delta_points"]), 1),
                "theoretical_vs_actual": round(
                    (g_th + a1_th + a2_th) - r["actual_points"], 1),
            },
        })
    return traces


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t_start = time.time()

    sim_df, sim_team_gf, sim_team_ga, sim_team_wins = run_all_team_simulations(
        n_sims=N_SIMS)
    actual_df = build_actual_baseline()

    merged = compute_deltas(sim_df, actual_df)

    # Save raw per-player table
    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved per-player calibration table → {OUT_CSV}", flush=True)

    aggs, rank_by_actual = aggregate_analysis(merged, sim_team_gf, sim_team_ga,
                                                n_sims=N_SIMS)
    outliers = find_outliers(merged)
    traces = trace_pipeline(["Connor McDavid", "Leon Draisaitl",
                              "Nathan MacKinnon"], merged, sim_team_gf)

    # Dump the full report payload as JSON next to the CSV for downstream use.
    report_path = MODEL_DIR / "phase_5_1_calibration_report.json"
    report_path.write_text(json.dumps({
        "n_sims": N_SIMS,
        "weights": SEASON_WEIGHTS,
        "min_gp_filter": MIN_GP_OVER_WINDOW,
        "aggregates": aggs,
        "outliers": outliers,
        "pipeline_traces": traces,
        "sim_team_gf": sim_team_gf,
        "sim_team_wins": sim_team_wins,
    }, indent=2))
    print(f"Saved JSON report payload → {report_path}", flush=True)

    # --- Print the human-readable report ---
    print("\n" + "=" * 78)
    print("PHASE 5.1 CALIBRATION CHECK — diagnostic report")
    print("=" * 78)
    print(f"Total compute time: {(time.time()-t_start)/60:.1f} minutes")
    print(f"\nQualified pool: {aggs['n_qualified']:,} (team, player) rows "
          f"(≥{MIN_GP_OVER_WINDOW} GP across 3-yr window).")

    print("\n--- LEAGUE-WIDE BIAS ---")
    L = aggs["league"]
    print(f"  mean Δ points     {L['mean_delta_points']:+.2f}   "
          f"median {L['median_delta_points']:+.2f}   std {L['std_delta_points']:.2f}")
    print(f"  mean Δ goals      {L['mean_delta_goals']:+.2f}   median {L['median_delta_goals']:+.2f}")
    print(f"  mean Δ A1         {L['mean_delta_a1']:+.2f}   median {L['median_delta_a1']:+.2f}")
    print(f"  mean Δ A2         {L['mean_delta_a2']:+.2f}   median {L['median_delta_a2']:+.2f}")
    print(f"  % players where sim > actual: {L['pct_higher_sim']*100:.1f}%")
    print(f"  % players where sim < actual: {L['pct_lower_sim']*100:.1f}%")
    print(f"\n  total simulated player goals/season:  {L['total_sim_player_goals_per_season']:,.0f}")
    print(f"  total actual player goals/season:     {L['total_actual_player_goals_per_season']:,.0f}")
    print(f"  Δ league-wide:                        {L['sim_minus_actual_total_goals']:+,.0f}")
    print(f"  avg sim-team goals-for/season:        {L['sim_team_goals_avg_per_team']:.1f}")

    print("\n--- BY TIER (ranked by ACTUAL points) ---")
    for tier, m in aggs["by_tier"].items():
        print(f"  {tier:<22}  n={m['n']:>4}   "
              f"mean Δ pts {m['mean_delta_points']:+5.1f}  "
              f"median {m['median_delta_points']:+5.1f}  "
              f"mean Δ% {m['mean_pct_delta_points']*100:+5.1f}%   "
              f"mean Δ G {m['mean_delta_goals']:+4.1f}")

    print("\n--- BY POSITION ---")
    for grp, m in aggs["by_position"].items():
        if "n" not in m: continue
        print(f"  {grp:<14}  n={m['n']:>4}   "
              f"mean Δ pts {m['mean_delta_points']:+5.1f}  "
              f"median {m['median_delta_points']:+5.1f}  "
              f"mean Δ% {(m.get('mean_pct_delta_points') or 0)*100:+5.1f}%")

    print("\n--- BY TEAM STRENGTH TIER ---")
    for label, m in aggs["by_team_strength"].items():
        print(f"  {label:<22}  n_teams={m['n_teams']:>2}  n_players={m['n_players']:>4}   "
              f"mean Δ pts {m['mean_delta_points']:+5.1f}  "
              f"mean Δ% {m['mean_pct_delta_points']*100:+5.1f}%")

    print("\n--- BY ARCHETYPE ---")
    for label, m in aggs["by_archetype"].items():
        print(f"  {label:<46}  n={m['n']:>3}")
        print(f"    Δ G {m['mean_delta_goals']:+5.1f}  Δ A1 {m['mean_delta_a1']:+5.1f}  "
              f"Δ A2 {m['mean_delta_a2']:+5.1f}  Δ pts {m['mean_delta_points']:+5.1f}")

    print("\n--- TOP 20 MOST UNDERPROJECTED (sim < actual) ---")
    print(f"  {'Player':<22}{'Team':>5}{'Pos':>4}{'Slot':>5}"
          f"{'Actual':>8}{'Sim':>7}{'Δ':>7}{'Δ%':>8}{'GP':>5}")
    for r in outliers["most_under"]:
        slot = f"{r['slot_kind']}{r['slot_idx']}"
        pct = (r['pct_delta_points'] * 100) if r['pct_delta_points'] is not None else 0
        print(f"  {r['player_name'][:22]:<22}{r['team']:>5}{r['position']:>4}{slot:>5}"
              f"{r['actual_points']:>8.1f}{r['sim_points']:>7.1f}"
              f"{r['delta_points']:>+7.1f}{pct:>+7.1f}%{r['actual_gp_total']:>5}")

    print("\n--- TOP 20 MOST OVERPROJECTED (sim > actual) ---")
    print(f"  {'Player':<22}{'Team':>5}{'Pos':>4}{'Slot':>5}"
          f"{'Actual':>8}{'Sim':>7}{'Δ':>7}{'Δ%':>8}{'GP':>5}")
    for r in outliers["most_over"]:
        slot = f"{r['slot_kind']}{r['slot_idx']}"
        pct = (r['pct_delta_points'] * 100) if r['pct_delta_points'] is not None else 0
        print(f"  {r['player_name'][:22]:<22}{r['team']:>5}{r['position']:>4}{slot:>5}"
              f"{r['actual_points']:>8.1f}{r['sim_points']:>7.1f}"
              f"{r['delta_points']:>+7.1f}{pct:>+7.1f}%{r['actual_gp_total']:>5}")

    print("\n--- PIPELINE TRACES (McDavid / Draisaitl / MacKinnon) ---")
    for t in traces:
        print(f"\n  • {t['player']} ({t.get('team','?')})")
        print(f"    Actual / season (year-weighted): "
              f"G {t['actual']['G']}  A1 {t['actual']['A1']}  "
              f"A2 {t['actual']['A2']}  →  P {t['actual']['P']}")
        rs = t["raw_shares"]
        print(f"    Raw historical shares:  G {rs['hist_goal_share']*100:.2f}%  "
              f"A1 {rs['hist_p_assist_share']*100:.2f}%  A2 {rs['hist_s_assist_share']*100:.2f}%")
        print(f"    Hist avg TOI {rs['hist_avg_TOI']:.1f}  EV_min_3yr {rs['ev_minutes_3yr']:.0f}  "
              f"thin_sample {rs['thin_sample']}")
        print(f"    Slot: {t['slot']['kind']}{t['slot']['slot_idx']}  "
              f"minutes {t['slot']['slot_minutes']}  "
              f"Layer-2 ratio = {t['slot']['slot_minutes']}/{rs['hist_avg_TOI']:.1f} = "
              f"{t['slot']['slot_minutes']/max(rs['hist_avg_TOI'],8):.3f}")
        l3 = t["layer_3_linemate"]
        print(f"    Layer 3 linemate WAR {l3['linemate_war']:+.2f}  "
              f"multiplier {l3['linemate_mult']:.3f}")
        ns = t["normalized_shares"]
        print(f"    Normalized (within-roster) shares:  G {ns['G']*100:.2f}%  "
              f"A1 {ns['A1']*100:.2f}%  A2 {ns['A2']*100:.2f}%")
        print(f"    Team sim GF/season  {t['team_sim_GF_per_season']:.1f}")
        th = t["theoretical_under_perfect_conditions"]
        sm = t["simulator_actually_produces"]
        print(f"    Theoretical points  G {th['G']}  A1 {th['A1']}  A2 {th['A2']}  →  P {th['P']}")
        print(f"    Simulator output    G {sm['G']}  A1 {sm['A1']}  A2 {sm['A2']}  →  P {sm['P']}")
        print(f"    GAP — sim vs actual         {t['gap']['sim_vs_actual']:+.1f}")
        print(f"    GAP — theoretical vs actual {t['gap']['theoretical_vs_actual']:+.1f}")

    print("\n" + "=" * 78)
    print(f"Done in {(time.time()-t_start)/60:.1f} minutes total.")


if __name__ == "__main__":
    main()
