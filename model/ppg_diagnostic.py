"""
Phase 5.1 — Per-game-rate (PPG) calibration diagnostic.

Diagnostic ONLY. No code or model changes.

Total-points comparison conflates rate (talent per game) with durability
(games played). The Barkov calibration miss is the textbook example: his
goal_share is divided by year-weighted GP that includes his ACL-shortened
25-26 season, depressing his per-game share even though the simulator
correctly slots him on the first line.

This script joins the existing Phase 5.1 calibration check with the
player_offense_shares CSV and composite_ratings_sim CSV, computes
sim PPG vs actual PPG, breaks the signed errors down by tier / position /
team-strength / archetype, surfaces the 30 worst over- and under-projections,
and traces 6 named players' calculation chains.

Inputs (read only):
  - model/phase_5_1_calibration_check.csv  (latest sim + actual baselines)
  - model/player_offense_shares.csv        (shares + avg_toi + gp_w)
  - model/composite_ratings_sim.csv        (composite_war)
  - model/team_strength.csv                (team strength tiers)

Outputs:
  - model/phase_5_1_ppg_comparison.csv     (per-player PPG table)
  - STDOUT report

Run from project root:
    python3 model/ppg_diagnostic.py
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
CALIB_CSV   = MODEL_DIR / "phase_5_1_calibration_check.csv"
SHARES_CSV  = MODEL_DIR / "player_offense_shares.csv"
COMP_CSV    = MODEL_DIR / "composite_ratings_sim.csv"
TS_CSV      = MODEL_DIR / "team_strength.csv"
OUT_CSV     = MODEL_DIR / "phase_5_1_ppg_comparison.csv"

# Simulator schedule: every team plays 82
SIM_GP = 82
MIN_GP_FILTER = 41   # matches the calibration check qualified-pool filter


# ---------------------------------------------------------------------------
# Load + join
# ---------------------------------------------------------------------------
def load_and_join() -> pd.DataFrame:
    calib = pd.read_csv(CALIB_CSV)
    shares = pd.read_csv(SHARES_CSV)
    comp = pd.read_csv(COMP_CSV)

    sh = shares[["player_id", "goal_share", "p_assist_share", "s_assist_share",
                 "avg_toi_per_game", "ev_minutes_3yr", "gp_w", "g_w", "a1_w",
                 "a2_w", "thin_sample"]].rename(columns={
                     "goal_share": "hist_goal_share",
                     "p_assist_share": "hist_a1_share",
                     "s_assist_share": "hist_a2_share",
                     "thin_sample": "thin_sample_shares",
                 })
    cp = comp[["player_id", "composite_war", "expected_gp_share"]]
    df = calib.merge(sh, on="player_id", how="left")
    df = df.merge(cp, on="player_id", how="left")

    # PPG calculations
    df["actual_ppg"] = df["actual_points"] / df["actual_gp_weighted_avg"].replace(0, np.nan)
    df["sim_ppg"]    = df["sim_points"] / SIM_GP
    df["delta_ppg"]  = df["sim_ppg"] - df["actual_ppg"]
    df["pct_delta_ppg"] = np.where(
        df["actual_ppg"] > 0.05,
        df["delta_ppg"] / df["actual_ppg"], np.nan)

    # Filter to qualified pool (matches calibration check methodology)
    df["qualified"] = (df["actual_gp_total"] >= MIN_GP_FILTER) & df["actual_points"].notna()
    return df


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------
def league_block(qual: pd.DataFrame) -> dict:
    d = qual["delta_ppg"]
    return {
        "n": int(len(qual)),
        "mean_delta_ppg":   float(d.mean()),
        "median_delta_ppg": float(d.median()),
        "std_delta_ppg":    float(d.std()),
        "pct_sim_higher":   float((d > 0).mean()),
        "pct_sim_lower":    float((d < 0).mean()),
        "mean_actual_ppg":  float(qual["actual_ppg"].mean()),
        "mean_sim_ppg":     float(qual["sim_ppg"].mean()),
    }


def tier_block(qual: pd.DataFrame) -> dict:
    """Same 4-tier scheme as the calibration check (ranked by ACTUAL total pts)."""
    rank = qual.sort_values("actual_points", ascending=False).reset_index(drop=True)
    rank["actual_rank"] = rank.index + 1
    bins = [("elite (top 30)",       1,   30),
            ("top-6 (31-100)",       31,  100),
            ("middle (101-300)",     101, 300),
            ("bottom (301+)",        301, 99999)]
    out = {}
    for lbl, lo, hi in bins:
        sub = rank[(rank["actual_rank"] >= lo) & (rank["actual_rank"] <= hi)]
        if sub.empty: continue
        out[lbl] = {
            "n": int(len(sub)),
            "mean_actual_ppg":  float(sub["actual_ppg"].mean()),
            "mean_sim_ppg":     float(sub["sim_ppg"].mean()),
            "mean_delta_ppg":   float(sub["delta_ppg"].mean()),
            "median_delta_ppg": float(sub["delta_ppg"].median()),
            "mean_pct_delta":   float(sub["pct_delta_ppg"].mean()),
        }
    return out


def position_block(qual: pd.DataFrame) -> dict:
    out = {}
    for code, lbl in (("C", "centers"), ("L", "left wings"), ("R", "right wings"),
                       ("D", "defensemen")):
        sub = qual[qual["position"].str.startswith(code, na=False)]
        if sub.empty: continue
        out[lbl] = {
            "n": int(len(sub)),
            "mean_actual_ppg":  float(sub["actual_ppg"].mean()),
            "mean_sim_ppg":     float(sub["sim_ppg"].mean()),
            "mean_delta_ppg":   float(sub["delta_ppg"].mean()),
            "mean_pct_delta":   float(sub["pct_delta_ppg"].mean()),
        }
    return out


def team_strength_block(qual: pd.DataFrame) -> dict:
    ts = pd.read_csv(TS_CSV).sort_values("team_strength", ascending=False)
    top10 = set(ts.head(10)["team"])
    bot10 = set(ts.tail(10)["team"])
    mid12 = set(ts["team"]) - top10 - bot10
    out = {}
    for lbl, members in (("top 10 strongest", top10),
                          ("middle 12",        mid12),
                          ("bottom 10 weakest", bot10)):
        sub = qual[qual["team"].isin(members)]
        out[lbl] = {
            "n_teams":  len(members),
            "n_players": int(len(sub)),
            "mean_actual_ppg": float(sub["actual_ppg"].mean()),
            "mean_sim_ppg":    float(sub["sim_ppg"].mean()),
            "mean_delta_ppg":  float(sub["delta_ppg"].mean()),
            "mean_pct_delta":  float(sub["pct_delta_ppg"].mean()),
        }
    return out


def archetype_block(qual: pd.DataFrame) -> dict:
    """Reuse the calibration check's archetype membership rules."""
    shares = pd.read_csv(SHARES_CSV)
    heavy = set(shares.sort_values("goal_share", ascending=False).head(30)["player_id"])
    shares["pm_ratio"] = shares["p_assist_share"] / shares["goal_share"].clip(lower=0.01)
    pure_pm = set(shares[shares["position"].isin(["C","L","R"])]
                  .sort_values("pm_ratio", ascending=False).head(30)["player_id"])
    d_off = shares[shares["position"] == "D"].copy()
    d_off["off_total"] = d_off["goal_share"] + d_off["p_assist_share"]
    off_d = set(d_off.sort_values("off_total", ascending=False).head(30)["player_id"])
    low_off = d_off.sort_values("off_total").head(80)
    def_d = set(low_off.sort_values("ev_minutes_3yr", ascending=False).head(30)["player_id"])
    # Two-way forwards: high TOI but middle goal/assist share — proxy via
    # centers with avg_toi >= 19 outside the top-30 heavy/playmaker groups.
    two_way = set(shares[(shares["position"] == "C")
                          & (shares["avg_toi_per_game"] >= 19)
                          & (~shares["player_id"].isin(heavy))
                          & (~shares["player_id"].isin(pure_pm))]["player_id"])

    out = {}
    for lbl, members in (("heavy shooters (top 30 by goal_share)", heavy),
                          ("pure playmakers (top 30 by A1/G ratio)", pure_pm),
                          ("two-way forwards (C, TOI>=19, not in heavy/pm)", two_way),
                          ("offensive defensemen (top 30)", off_d),
                          ("defensive defensemen (top 30 by EV min, low off)", def_d)):
        sub = qual[qual["player_id"].isin(members)]
        if sub.empty: continue
        out[lbl] = {
            "n": int(len(sub)),
            "mean_actual_ppg": float(sub["actual_ppg"].mean()),
            "mean_sim_ppg":    float(sub["sim_ppg"].mean()),
            "mean_delta_ppg":  float(sub["delta_ppg"].mean()),
            "mean_pct_delta":  float(sub["pct_delta_ppg"].mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Outliers + traces
# ---------------------------------------------------------------------------
def outliers(qual: pd.DataFrame, n: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["player_name", "team", "position", "slot_kind", "slot_idx",
            "slot_minutes", "actual_gp_weighted_avg", "actual_ppg",
            "sim_ppg", "delta_ppg", "pct_delta_ppg",
            "avg_toi_per_game", "composite_war", "thin_sample_shares",
            "linemate_mult"]
    under = qual.sort_values("delta_ppg").head(n)[cols]
    over  = qual.sort_values("delta_ppg", ascending=False).head(n)[cols]
    return under, over


def trace_one(df: pd.DataFrame, name_substr: str) -> dict | None:
    """Find by name substring; if multiple, prefer the one with highest actual_ppg."""
    sub = df[df["player_name"].str.contains(name_substr, na=False, regex=False)]
    if sub.empty: return None
    r = sub.sort_values("actual_ppg", ascending=False).iloc[0]
    # Diagnose the source of the gap
    # 1) Share-builder dilution: actual_gp_weighted_avg << 82 indicates the
    #    shares CSV under-counted his GP. If his avg_toi_per_game is high
    #    (>=18) but his gp_w in shares CSV is low, the share is depressed.
    gp_w  = float(r.get("gp_w", 0) or 0)
    avg_toi = float(r.get("avg_toi_per_game", 0) or 0)
    full_gp_w = 1.0 * 82 + 0.35 * 82 + 0.10 * 82   # 118.9
    dilution_ratio = gp_w / full_gp_w
    slot_min = int(r["slot_minutes"])
    toi_vs_slot = (slot_min / avg_toi) if avg_toi > 0 else float("nan")
    diag = []
    if dilution_ratio < 0.55 and avg_toi >= 17:
        diag.append(f"share-builder dilution: gp_w={gp_w:.1f} = {dilution_ratio*100:.0f}% of full → goal_share depressed")
    if slot_min < 18 and avg_toi >= 18:
        diag.append(f"slot mismatch: avg_toi {avg_toi:.1f} but slot {slot_min} min")
    if r["sim_team_gf"] < 230:
        diag.append(f"team GF low: sim {r['sim_team_gf']:.0f}")
    if not diag:
        diag.append("no single dominant cause; PPG gap likely from share-model regression")
    return {
        "player": str(r["player_name"]),
        "team": str(r["team"]),
        "position": str(r["position"]),
        "slot": f"{r['slot_kind']}{int(r['slot_idx'])} ({slot_min} min)",
        "actual_total_pts": float(r["actual_points"]),
        "actual_gp_weighted_avg": float(r["actual_gp_weighted_avg"]),
        "actual_ppg": float(r["actual_ppg"]),
        "sim_total_pts": float(r["sim_points"]),
        "sim_gp": SIM_GP,
        "sim_ppg": float(r["sim_ppg"]),
        "delta_ppg": float(r["delta_ppg"]),
        "pct_delta_ppg": float(r["pct_delta_ppg"]) if pd.notna(r["pct_delta_ppg"]) else None,
        "hist_goal_share":   float(r.get("hist_goal_share", 0) or 0),
        "hist_a1_share":     float(r.get("hist_a1_share", 0) or 0),
        "hist_a2_share":     float(r.get("hist_a2_share", 0) or 0),
        "avg_toi_per_game":  avg_toi,
        "gp_w_shares_csv":   gp_w,
        "dilution_ratio":    dilution_ratio,
        "ev_min_3yr":        float(r.get("ev_minutes_3yr", 0) or 0),
        "thin_sample":       bool(r["thin_sample_shares"]) if pd.notna(r["thin_sample_shares"]) else None,
        "composite_war":     float(r["composite_war"]) if pd.notna(r["composite_war"]) else None,
        "linemate_mult":     float(r["linemate_mult"]),
        "share_g_norm":      float(r["share_g_norm"]),
        "share_a1_norm":     float(r["share_a1_norm"]),
        "share_a2_norm":     float(r["share_a2_norm"]),
        "sim_team_gf":       float(r["sim_team_gf"]),
        "toi_vs_slot_ratio": toi_vs_slot,
        "diagnosis":         diag,
    }


# ---------------------------------------------------------------------------
# Barkov-style screen — high TOI, low simulated minutes OR depressed gp_w
# ---------------------------------------------------------------------------
def barkov_screen(df: pd.DataFrame) -> dict:
    """Two screens.

    A) "Lineup-placement bug" candidates — high historical TOI but slotted
       below 15 min in simulator (likely 4th line / 3rd-pair). After Fix 1
       this should be a short list, but the user wants explicit confirmation.

    B) "Share-builder GP-dilution bug" candidates (the actual Barkov pattern)
       — high avg_toi_per_game when healthy, but gp_w in shares CSV is heavily
       depressed because of missed seasons. Identified by:
            avg_toi_per_game >= 17  AND  gp_w / 118.9 <= 0.55
       (118.9 = full-time weighted GP across the 3-year window). The 0.55
       cutoff means they were essentially absent for at least one full season
       and partially absent in another. Barkov is the prototype.
    """
    full_gp_w = 1.0 * 82 + 0.35 * 82 + 0.10 * 82
    df = df.copy()
    df["dilution_ratio"] = df["gp_w"] / full_gp_w

    a = df[(df["avg_toi_per_game"] >= 18)
            & (df["slot_minutes"] < 15)
            & df["actual_points"].notna()].copy()
    b = df[(df["avg_toi_per_game"] >= 17)
            & (df["dilution_ratio"] <= 0.55)
            & df["actual_points"].notna()].copy()
    return {
        "placement_candidates": a.sort_values("avg_toi_per_game", ascending=False)[
            ["player_name","team","position","slot_kind","slot_idx","slot_minutes",
             "avg_toi_per_game","actual_ppg","sim_ppg","delta_ppg",
             "dilution_ratio","gp_w"]],
        "share_dilution_candidates": b.sort_values("delta_ppg")[
            ["player_name","team","position","slot_kind","slot_idx","slot_minutes",
             "avg_toi_per_game","gp_w","dilution_ratio","actual_ppg","sim_ppg",
             "delta_ppg","pct_delta_ppg"]],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = load_and_join()
    qual = df[df["qualified"]].copy()

    # Save the wide CSV (qualified rows only — matches "qualified pool" in report)
    csv_cols = ["player_id", "player_name", "team", "position",
                "slot_kind", "slot_idx", "slot_minutes",
                "actual_gp_total", "actual_gp_weighted_avg", "actual_points",
                "actual_ppg",
                "sim_points", "sim_gp" if False else None,
                "sim_ppg", "delta_ppg", "pct_delta_ppg",
                "avg_toi_per_game", "ev_minutes_3yr", "gp_w",
                "hist_goal_share", "hist_a1_share", "hist_a2_share",
                "thin_sample_shares", "composite_war",
                "linemate_war", "linemate_mult",
                "share_g_norm", "share_a1_norm", "share_a2_norm",
                "sim_team_gf"]
    csv_cols = [c for c in csv_cols if c is not None]
    out = qual.copy()
    out["sim_gp"] = SIM_GP
    out = out[[c for c in csv_cols if c in out.columns] +
              (["sim_gp"] if "sim_gp" not in csv_cols else [])]
    # Re-order so sim_gp appears between sim_points and sim_ppg
    final_cols = ["player_id","player_name","team","position",
                   "slot_kind","slot_idx","slot_minutes",
                   "actual_gp_total","actual_gp_weighted_avg","actual_points",
                   "actual_ppg",
                   "sim_points","sim_gp","sim_ppg",
                   "delta_ppg","pct_delta_ppg",
                   "avg_toi_per_game","ev_minutes_3yr","gp_w",
                   "hist_goal_share","hist_a1_share","hist_a2_share",
                   "thin_sample_shares","composite_war",
                   "linemate_war","linemate_mult",
                   "share_g_norm","share_a1_norm","share_a2_norm",
                   "sim_team_gf"]
    out = out[[c for c in final_cols if c in out.columns]]
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved per-player PPG comparison → {OUT_CSV}\n", flush=True)

    # Run aggregates
    lg = league_block(qual)
    tiers = tier_block(qual)
    poss = position_block(qual)
    tss = team_strength_block(qual)
    arch = archetype_block(qual)
    under, over = outliers(qual, n=30)
    screens = barkov_screen(df)

    # Traces
    trace_targets = ["Connor McDavid", "Aleksander Barkov",
                      "Macklin Celebrini", "Auston Matthews",
                      "Cale Makar", "Nathan MacKinnon"]
    traces = [trace_one(df, t) for t in trace_targets]

    # ---- Report ----
    print("=" * 80)
    print("PHASE 5.1 — PPG (POINTS-PER-GAME) CALIBRATION DIAGNOSTIC")
    print("=" * 80)
    print(f"\nQualified pool: {lg['n']:,} (team, player) rows (≥{MIN_GP_FILTER} GP total).")
    print("Methodology: actual PPG = year-weighted points / year-weighted GP")
    print("             (weights match shares CSV: 2025=1.00, 2024=0.35, 2023=0.10)")
    print("             sim PPG = simulator points / 82")

    print("\n--- LEAGUE-WIDE PPG BIAS ---")
    print(f"  mean Δ PPG     {lg['mean_delta_ppg']:+.3f}   median {lg['median_delta_ppg']:+.3f}   std {lg['std_delta_ppg']:.3f}")
    print(f"  mean actual PPG {lg['mean_actual_ppg']:.3f}   mean sim PPG {lg['mean_sim_ppg']:.3f}")
    print(f"  % sim > actual {lg['pct_sim_higher']*100:.1f}%   % sim < actual {lg['pct_sim_lower']*100:.1f}%")

    print("\n--- BY TIER (ranked by ACTUAL total points) ---")
    for lbl, m in tiers.items():
        print(f"  {lbl:<22}  n={m['n']:>4}   "
              f"actual {m['mean_actual_ppg']:.3f}   "
              f"sim {m['mean_sim_ppg']:.3f}   "
              f"Δ {m['mean_delta_ppg']:+.3f}   "
              f"Δ% {m['mean_pct_delta']*100:+5.1f}%")

    print("\n--- BY POSITION ---")
    for lbl, m in poss.items():
        print(f"  {lbl:<14}  n={m['n']:>4}   "
              f"actual {m['mean_actual_ppg']:.3f}   "
              f"sim {m['mean_sim_ppg']:.3f}   "
              f"Δ {m['mean_delta_ppg']:+.3f}   "
              f"Δ% {m['mean_pct_delta']*100:+5.1f}%")

    print("\n--- BY TEAM STRENGTH TIER ---")
    for lbl, m in tss.items():
        print(f"  {lbl:<22}  n_teams={m['n_teams']:>2}  n_players={m['n_players']:>4}   "
              f"actual {m['mean_actual_ppg']:.3f}   "
              f"sim {m['mean_sim_ppg']:.3f}   "
              f"Δ {m['mean_delta_ppg']:+.3f}   Δ% {m['mean_pct_delta']*100:+5.1f}%")

    print("\n--- BY ARCHETYPE ---")
    for lbl, m in arch.items():
        print(f"  {lbl:<48}  n={m['n']:>3}")
        print(f"    actual {m['mean_actual_ppg']:.3f}   sim {m['mean_sim_ppg']:.3f}   "
              f"Δ {m['mean_delta_ppg']:+.3f}   Δ% {m['mean_pct_delta']*100:+5.1f}%")

    print("\n--- 30 LARGEST UNDER-PROJECTIONS BY PPG (sim PPG << actual PPG) ---")
    print(f"  {'Player':<22}{'Tm':>4}{'Pos':>4}{'Slot':>5}"
          f"{'actGP':>7}{'aPPG':>7}{'sPPG':>7}{'ΔPPG':>7}{'Δ%':>7}"
          f"{'TOI':>6}{'WAR':>6}{'thin':>5}")
    for _, r in under.iterrows():
        slot = f"{r['slot_kind']}{int(r['slot_idx'])}"
        pct = (r['pct_delta_ppg']*100) if pd.notna(r['pct_delta_ppg']) else 0
        thin = "Y" if r['thin_sample_shares'] else "N"
        toi = r['avg_toi_per_game'] or 0
        war = r['composite_war'] or 0
        agp = r['actual_gp_weighted_avg'] or 0
        print(f"  {str(r['player_name'])[:22]:<22}{str(r['team']):>4}{str(r['position']):>4}{slot:>5}"
              f"{agp:>7.1f}{r['actual_ppg']:>7.3f}{r['sim_ppg']:>7.3f}"
              f"{r['delta_ppg']:>+7.3f}{pct:>+6.1f}%{toi:>6.1f}{war:>+6.2f}{thin:>5}")

    print("\n--- 30 LARGEST OVER-PROJECTIONS BY PPG (sim PPG >> actual PPG) ---")
    print(f"  {'Player':<22}{'Tm':>4}{'Pos':>4}{'Slot':>5}"
          f"{'actGP':>7}{'aPPG':>7}{'sPPG':>7}{'ΔPPG':>7}{'Δ%':>7}"
          f"{'TOI':>6}{'WAR':>6}{'thin':>5}")
    for _, r in over.iterrows():
        slot = f"{r['slot_kind']}{int(r['slot_idx'])}"
        pct = (r['pct_delta_ppg']*100) if pd.notna(r['pct_delta_ppg']) else 0
        thin = "Y" if r['thin_sample_shares'] else "N"
        toi = r['avg_toi_per_game'] or 0
        war = r['composite_war'] or 0
        agp = r['actual_gp_weighted_avg'] or 0
        print(f"  {str(r['player_name'])[:22]:<22}{str(r['team']):>4}{str(r['position']):>4}{slot:>5}"
              f"{agp:>7.1f}{r['actual_ppg']:>7.3f}{r['sim_ppg']:>7.3f}"
              f"{r['delta_ppg']:>+7.3f}{pct:>+6.1f}%{toi:>6.1f}{war:>+6.2f}{thin:>5}")

    print("\n--- BARKOV-STYLE SCREEN A: TOI≥18 but slotted <15 min (lineup-placement) ---")
    a = screens["placement_candidates"]
    if a.empty:
        print("  (none — Fix 1 correctly handled all high-TOI players)")
    else:
        print(f"  {'Player':<22}{'Tm':>4}{'Pos':>4}{'Slot':>5}{'slotMin':>8}"
              f"{'TOI':>6}{'aPPG':>7}{'sPPG':>7}{'ΔPPG':>7}")
        for _, r in a.iterrows():
            slot = f"{r['slot_kind']}{int(r['slot_idx'])}"
            print(f"  {str(r['player_name'])[:22]:<22}{str(r['team']):>4}{str(r['position']):>4}"
                  f"{slot:>5}{int(r['slot_minutes']):>8}"
                  f"{r['avg_toi_per_game']:>6.1f}"
                  f"{r['actual_ppg']:>7.3f}{r['sim_ppg']:>7.3f}{r['delta_ppg']:>+7.3f}")

    print("\n--- BARKOV-STYLE SCREEN B: TOI≥17, weighted-GP < 55% of full (share dilution) ---")
    b = screens["share_dilution_candidates"]
    if b.empty:
        print("  (none flagged)")
    else:
        print(f"  {'Player':<22}{'Tm':>4}{'Pos':>4}{'Slot':>5}{'gp_w':>6}{'dilu':>6}"
              f"{'TOI':>6}{'aPPG':>7}{'sPPG':>7}{'ΔPPG':>7}{'Δ%':>7}")
        for _, r in b.iterrows():
            slot = f"{r['slot_kind']}{int(r['slot_idx'])}"
            pct = (r['pct_delta_ppg']*100) if pd.notna(r['pct_delta_ppg']) else 0
            print(f"  {str(r['player_name'])[:22]:<22}{str(r['team']):>4}{str(r['position']):>4}{slot:>5}"
                  f"{r['gp_w']:>6.1f}{r['dilution_ratio']*100:>5.0f}%"
                  f"{r['avg_toi_per_game']:>6.1f}"
                  f"{r['actual_ppg']:>7.3f}{r['sim_ppg']:>7.3f}{r['delta_ppg']:>+7.3f}{pct:>+6.1f}%")

    print("\n--- PIPELINE TRACES (6 named players) ---")
    for t in traces:
        if t is None:
            print("\n  (player not found)")
            continue
        print(f"\n  • {t['player']} ({t['team']}, {t['position']})  slot {t['slot']}")
        print(f"    Actual:  {t['actual_total_pts']:.1f} pts in {t['actual_gp_weighted_avg']:.1f} GP "
              f"= {t['actual_ppg']:.3f} PPG")
        print(f"    Sim:     {t['sim_total_pts']:.1f} pts in {t['sim_gp']} GP = {t['sim_ppg']:.3f} PPG")
        print(f"    Δ PPG:   {t['delta_ppg']:+.3f}   ({(t['pct_delta_ppg'] or 0)*100:+.1f}%)")
        print(f"    Inputs:  hist goal_share {t['hist_goal_share']*100:.2f}%, "
              f"A1_share {t['hist_a1_share']*100:.2f}%, A2_share {t['hist_a2_share']*100:.2f}%")
        print(f"             avg_TOI {t['avg_toi_per_game']:.1f}  "
              f"gp_w {t['gp_w_shares_csv']:.1f} ({t['dilution_ratio']*100:.0f}% of full 118.9)")
        print(f"             ev_min_3yr {t['ev_min_3yr']:.0f}  thin_sample={t['thin_sample']}  "
              f"composite_war {t['composite_war']:+.2f}")
        print(f"    Slot:    {t['slot']}, TOI/slot ratio {t['toi_vs_slot_ratio']:.3f}, "
              f"linemate_mult {t['linemate_mult']:.3f}")
        print(f"             normalised shares  G {t['share_g_norm']*100:.2f}%  "
              f"A1 {t['share_a1_norm']*100:.2f}%  A2 {t['share_a2_norm']*100:.2f}%")
        print(f"    Team sim GF: {t['sim_team_gf']:.0f}/season")
        print(f"    Diagnosis: {'; '.join(t['diagnosis'])}")

    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    # Compute key takeaways
    qm = qual["delta_ppg"].mean()
    high_share_dilution = len(screens["share_dilution_candidates"])
    placement_misses = len(screens["placement_candidates"])

    print(f"\n  League-wide sim PPG vs actual PPG bias: {qm:+.3f} PPG/player "
          f"({(qm/qual['actual_ppg'].mean())*100:+.1f}% of mean actual PPG).")
    if qm > 0.02:
        print("    → Sim is biased HIGH on PPG (over-projecting per-game rate).")
    elif qm < -0.02:
        print("    → Sim is biased LOW on PPG (under-projecting per-game rate).")
    else:
        print("    → Sim PPG bias is near zero in aggregate.")
    print(f"\n  Sub-group asymmetry (mean Δ PPG):")
    for lbl, m in tiers.items():
        print(f"    {lbl:<22} {m['mean_delta_ppg']:+.3f}")
    print(f"\n  Share-dilution (Barkov-style) candidates flagged: {high_share_dilution}")
    print(f"  Lineup-placement-mismatch candidates flagged:     {placement_misses}")

    print(f"\n  Detail saved to {OUT_CSV.name} and the per-player rows above.")


if __name__ == "__main__":
    main()
