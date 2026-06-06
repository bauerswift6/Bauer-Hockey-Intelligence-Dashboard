"""
Phase 5.1 — Build per-player offensive shares cache.

Layer 1 of the four-layer season-simulator distribution model.

For every player that appears in MoneyPuck's 2023-24 / 2024-25 / 2025-26
skater CSVs we compute:

  - goal_share        = year-weighted goals      / team's year-weighted goals
  - p_assist_share    = year-weighted A1         / team's year-weighted A1
  - s_assist_share    = year-weighted A2         / team's year-weighted A2
  - avg_toi_per_game  = year-weighted icetime (all situations) / GP
  - ev_minutes_3yr    = raw 5v5 icetime sum (used to detect thin samples)

All three share metrics combine "5on5" + "5on4" situations (regulation
even-strength + power-play offence). Empty-net goals and short-handed
production are excluded — too noisy at the share level.

Year weights match the simulation composite: 2025-26 = 1.00, 2024-25 = 0.60,
2023-24 = 0.25. A player traded mid-season is summed across team-stints in
that season; the most recent team is used as the player's "team" label.

Also written into the file (as side-band rows below the per-player table):
  - league average shares by position (used in Layer 4 fallback)
  - linear-regression slope/intercept fitting `historical_share` to
    `individual_component` z-score (used for thin-sample Layer 4 share
    estimation)

Outputs:
  model/player_offense_shares.csv

Run from project root:
    python3 model/build_player_offense_shares.py
"""
from __future__ import annotations
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
COMPOSITE_SIM_CSV = MODEL_DIR / "composite_ratings_sim.csv"
OUT_CSV = MODEL_DIR / "player_offense_shares.csv"

# 3-season window. The share weights are intentionally STEEPER than the
# simulation composite's 1.00/0.60/0.25 because shares answer a different
# question — "what will this player produce based on their most recent form?"
# — vs. the composite which asks "what is this player's stable underlying
# talent?". A steeper recency curve here keeps the season projections aligned
# with each player's actual recent output (e.g. McDavid's 130+ point season,
# Celebrini's 115-point sophomore campaign) instead of regressing them
# halfway back to a two-season-ago baseline.
SEASONS = ["2023", "2024", "2025"]   # MoneyPuck start-year codes
SEASON_WEIGHTS = {"2025": 1.00, "2024": 0.35, "2023": 0.10}
FULL_WINDOW_W = sum(SEASON_WEIGHTS.values())   # 1.45

# Thin-sample threshold (Layer 4 trigger). EV minutes only.
THIN_SAMPLE_MIN_EV = 1500.0

# Share GP-dilution fix (Phase 5.1, 2026-06-04). Players who missed a full
# season (Barkov, Tkachuk, Klingberg, etc.) saw their shares depressed because
# their year-weighted numerators were divided by the full 3-season weight in
# the team denominator while their own weighted contribution was 0 for the
# missing season(s). Fix: each player's numerator is rescaled by
# (FULL_WINDOW_W / seasons_present_w) so the share denominator and numerator
# compare on the same per-season-equivalent basis. Players who played all
# three seasons see no change.
#
# Safety threshold: only apply the rescale to players whose weighted GP in
# their playing seasons is at least this many games. Players below the
# threshold (debuts of <30 games, etc.) keep the raw-share calculation so a
# tiny sample doesn't inflate the projection.
MIN_GP_FOR_RESCALE = 30.0

MP_URL = ("https://moneypuck.com/moneypuck/playerData/seasonSummary/"
          "{season}/regular/{kind}.csv")
HEADERS = {"User-Agent":
           "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}


def _get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def aggregate_three_seasons() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (player_agg, team_agg). Both indexed by their natural key."""
    # Accumulators
    pacc: dict = {}     # pid → dict of year-weighted sums
    tacc: dict = {}     # team_abbrev → year-weighted sums
    pmeta: dict = {}    # pid → (season_idx, name, team, position)

    for idx, code in enumerate(SEASONS):
        w = SEASON_WEIGHTS[code]
        print(f"  Pulling MoneyPuck {code} (weight {w}) …", flush=True)
        try:
            sk = _get_csv(MP_URL.format(season=code, kind="skaters"))
        except Exception as e:
            print(f"    skipped — {e}", flush=True)
            continue

        s55 = _num(sk[sk["situation"] == "5on5"].copy(),
                   ["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                    "icetime"])
        s54 = _num(sk[sk["situation"] == "5on4"].copy(),
                   ["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                    "icetime"])
        sall = _num(sk[sk["situation"] == "all"].copy(),
                    ["I_F_goals", "I_F_primaryAssists", "I_F_secondaryAssists",
                     "icetime", "games_played"])

        # --- per-player aggregation (5v5 + 5v4 combined for shares) ---
        merged = (s55.groupby("playerId").sum(numeric_only=True)
                  .add(s54.groupby("playerId").sum(numeric_only=True),
                       fill_value=0)
                  .reset_index())

        for r in merged.itertuples(index=False):
            try:
                pid = int(r.playerId)
            except (TypeError, ValueError):
                continue
            a = pacc.setdefault(pid, {
                "g": 0.0, "a1": 0.0, "a2": 0.0,
                "ev_seconds_raw": 0.0,
                "all_seconds_w": 0.0, "gp_w": 0.0,
                "seasons_present_w": 0.0,
                "gp_raw_present": 0.0,
            })
            a["g"]  += w * float(r.I_F_goals)
            a["a1"] += w * float(r.I_F_primaryAssists)
            a["a2"] += w * float(r.I_F_secondaryAssists)
            # raw (unweighted) EV minutes to identify thin samples
        # add raw 5v5 ice per player without year-weight (sample-size sense)
        ev_by_pid = (s55.groupby("playerId")["icetime"].sum() / 60.0).to_dict()
        for pid, mins in ev_by_pid.items():
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            pacc.setdefault(pid, {
                "g": 0.0, "a1": 0.0, "a2": 0.0,
                "ev_seconds_raw": 0.0,
                "all_seconds_w": 0.0, "gp_w": 0.0,
                "seasons_present_w": 0.0,
                "gp_raw_present": 0.0,
            })
            pacc[pid]["ev_seconds_raw"] += mins * 60.0

        # All-situations icetime (for avg TOI/game) + games_played + meta.
        # We aggregate per-player within this season so traded players (split
        # across multiple team-stint rows) contribute one "present" credit.
        sall_by_pid = (sall.groupby("playerId")
                       [["icetime", "games_played"]].sum().reset_index())
        for r in sall_by_pid.itertuples(index=False):
            try:
                pid = int(r.playerId)
            except (TypeError, ValueError):
                continue
            a = pacc.setdefault(pid, {
                "g": 0.0, "a1": 0.0, "a2": 0.0,
                "ev_seconds_raw": 0.0,
                "all_seconds_w": 0.0, "gp_w": 0.0,
                "seasons_present_w": 0.0,
                "gp_raw_present": 0.0,
            })
            gp_this_season = float(r.games_played)
            a["all_seconds_w"] += w * float(r.icetime)
            a["gp_w"] += w * gp_this_season
            if gp_this_season > 0:
                a["seasons_present_w"] += w
                a["gp_raw_present"] += gp_this_season
        # Meta (team / name / position) — use the season-level data we just rolled up
        meta_first = sall.sort_values(["playerId", "icetime"],
                                       ascending=[True, False]).groupby("playerId").first()
        for pid, row in meta_first.iterrows():
            try:
                pid = int(pid)
            except (TypeError, ValueError):
                continue
            pmeta[pid] = (idx, row.get("name", ""), row.get("team", ""),
                          row.get("position", ""))

        # --- per-team aggregation (5v5 + 5v4 combined for the share denom) ---
        # Sum players within team to get team-side numerator. (We could use a
        # teams CSV, but for goals/assists the player sum gives the same answer
        # and avoids team CSV column-name surprises.)
        for situ_df in (s55, s54):
            for r in situ_df.itertuples(index=False):
                t = getattr(r, "team", "") or ""
                if not t:
                    continue
                ta = tacc.setdefault(t, {"g": 0.0, "a1": 0.0, "a2": 0.0})
                ta["g"]  += w * float(r.I_F_goals)
                ta["a1"] += w * float(r.I_F_primaryAssists)
                ta["a2"] += w * float(r.I_F_secondaryAssists)

    # --- materialise to DataFrames ---
    prows = []
    for pid, a in pacc.items():
        meta = pmeta.get(pid, (None, "", "", ""))
        idx, name, team, pos = meta
        gp_w = a["gp_w"]
        avg_toi = (a["all_seconds_w"] / gp_w / 60.0) if gp_w > 0 else 0.0
        prows.append({
            "player_id": pid,
            "player_name": name,
            "team": team,
            "position": pos,
            "g_w": round(a["g"], 4),
            "a1_w": round(a["a1"], 4),
            "a2_w": round(a["a2"], 4),
            "avg_toi_per_game": round(avg_toi, 3),
            "ev_minutes_3yr": round(a["ev_seconds_raw"] / 60.0, 1),
            "gp_w": round(gp_w, 1),
            "seasons_present_w": round(a.get("seasons_present_w", 0.0), 4),
            "gp_raw_present": round(a.get("gp_raw_present", 0.0), 1),
        })
    p_df = pd.DataFrame(prows)
    trows = [{"team": t, "g_w": v["g"], "a1_w": v["a1"], "a2_w": v["a2"]}
             for t, v in tacc.items()]
    t_df = pd.DataFrame(trows)
    return p_df, t_df


def compute_shares(p_df: pd.DataFrame, t_df: pd.DataFrame) -> pd.DataFrame:
    """Attach goal_share / p_assist_share / s_assist_share columns.

    GP-dilution correction (Phase 5.1 fix, 2026-06-04). The team denominators
    `tm["g_w"]` etc. are summed across the full 3-season window (FULL_WINDOW_W
    = 1.45). A player who missed a season has their per-season-equivalent
    production diluted because their numerator weights only the seasons they
    played while the denominator weights all three. We rescale each
    qualifying player's numerator by (FULL_WINDOW_W / seasons_present_w) so
    both sides compare on the same scale.

    The rescale is gated on `gp_raw_present >= MIN_GP_FOR_RESCALE` to keep
    tiny samples (cup-of-coffee debuts, short call-ups) from being projected
    upward off noise. Players below the threshold keep the original
    dilution-prone calculation and rely on the Layer-4 thin-sample blend at
    simulator time.
    """
    tmap = t_df.set_index("team").to_dict(orient="index")
    out = p_df.copy()
    out["goal_share"]        = 0.0
    out["p_assist_share"]    = 0.0
    out["s_assist_share"]    = 0.0
    out["share_rescale_applied"] = False
    out["share_rescale_factor"]  = 1.0
    for i, r in out.iterrows():
        tm = tmap.get(r["team"])
        if not tm:
            continue
        sp = float(r.get("seasons_present_w", 0.0) or 0.0)
        gp_present = float(r.get("gp_raw_present", 0.0) or 0.0)
        if sp > 0 and sp < FULL_WINDOW_W and gp_present >= MIN_GP_FOR_RESCALE:
            factor = FULL_WINDOW_W / sp
            out.at[i, "share_rescale_applied"] = True
            out.at[i, "share_rescale_factor"]  = round(factor, 4)
        else:
            factor = 1.0
        g_w  = r["g_w"]  * factor
        a1_w = r["a1_w"] * factor
        a2_w = r["a2_w"] * factor
        out.at[i, "goal_share"]     = g_w  / tm["g_w"]  if tm["g_w"]  > 0 else 0.0
        out.at[i, "p_assist_share"] = a1_w / tm["a1_w"] if tm["a1_w"] > 0 else 0.0
        out.at[i, "s_assist_share"] = a2_w / tm["a2_w"] if tm["a2_w"] > 0 else 0.0
    return out


def league_averages_by_position(out_df: pd.DataFrame, thin_min: float) -> dict:
    """Average share per position among NON-thin-sample players. Used as the
    fallback baseline for Layer 4 secondary-assist projection."""
    well = out_df[out_df["ev_minutes_3yr"] >= thin_min].copy()
    grp = well.groupby("position")[["goal_share", "p_assist_share", "s_assist_share"]].mean()
    return {pos: row.to_dict() for pos, row in grp.iterrows()}


def fit_share_regressions(out_df: pd.DataFrame, comp: pd.DataFrame,
                           thin_min: float) -> dict:
    """Linear regression of historical share ↔ composite component z-score,
    computed on the non-thin-sample pool. Used for Layer 4 thin-sample share
    estimation. Returns:
      {
        "goal_share":     {"slope": …, "intercept": …, "n": …, "r2": …},
        "p_assist_share": {…},
      }
    """
    merged = out_df.merge(
        comp[["player_id", "individual_component", "playmaking_component"]],
        on="player_id", how="left")
    well = merged[(merged["ev_minutes_3yr"] >= thin_min)
                  & merged["individual_component"].notna()]

    def _fit(x_col, y_col):
        x = well[x_col].astype(float).values
        y = well[y_col].astype(float).values
        if len(x) < 50:
            return {"slope": 0.0, "intercept": float(np.mean(y) if len(y) else 0.0),
                    "n": int(len(x)), "r2": 0.0}
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"slope": float(slope), "intercept": float(intercept),
                "n": int(len(x)), "r2": round(r2, 4)}

    return {
        "goal_share":     _fit("individual_component", "goal_share"),
        "p_assist_share": _fit("playmaking_component", "p_assist_share"),
    }


def main():
    print("Phase 5.1 — building player_offense_shares.csv …", flush=True)
    print(f"  Window: {SEASONS}  weights: {SEASON_WEIGHTS}", flush=True)

    p_df, t_df = aggregate_three_seasons()
    print(f"\n  {len(p_df):,} player rows aggregated across {len(t_df)} teams",
          flush=True)
    out = compute_shares(p_df, t_df)

    # Mark thin-sample players
    out["thin_sample"] = out["ev_minutes_3yr"] < THIN_SAMPLE_MIN_EV
    n_thin = int(out["thin_sample"].sum())
    n_well = int((~out["thin_sample"]).sum())
    print(f"  Thin-sample players (<{THIN_SAMPLE_MIN_EV:.0f} EV min): {n_thin:,}", flush=True)
    print(f"  Well-sampled players: {n_well:,}", flush=True)

    # League position averages + regression coefficients for Layer 4
    if COMPOSITE_SIM_CSV.exists():
        comp = pd.read_csv(COMPOSITE_SIM_CSV)
    else:
        print(f"  WARNING: {COMPOSITE_SIM_CSV} not found — Layer 4 regression "
              f"will be skipped.", flush=True)
        comp = pd.DataFrame(columns=["player_id", "individual_component",
                                      "playmaking_component"])

    league_avg = league_averages_by_position(out, THIN_SAMPLE_MIN_EV)
    regressions = fit_share_regressions(out, comp, THIN_SAMPLE_MIN_EV) \
                  if not comp.empty else {}

    # Persist the per-player table
    out_cols = ["player_id", "player_name", "team", "position",
                "goal_share", "p_assist_share", "s_assist_share",
                "avg_toi_per_game", "ev_minutes_3yr", "gp_w",
                "g_w", "a1_w", "a2_w", "thin_sample",
                "seasons_present_w", "gp_raw_present",
                "share_rescale_applied", "share_rescale_factor"]
    out_table = out[out_cols].sort_values("goal_share", ascending=False)
    out_table.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}", flush=True)

    # Side-band metadata as JSON blob next to the CSV for the simulator to load
    meta_path = MODEL_DIR / "player_offense_shares_meta.json"
    meta = {
        "seasons": SEASONS,
        "season_weights": SEASON_WEIGHTS,
        "thin_sample_min_ev": THIN_SAMPLE_MIN_EV,
        "n_players": int(len(out_table)),
        "n_thin_sample": n_thin,
        "league_avg_shares_by_position": league_avg,
        "share_regressions": regressions,
        "min_gp_for_rescale": MIN_GP_FOR_RESCALE,
        "notes": [
            "Shares combine 5on5 + 5on4 only (no SH, no empty-net).",
            ("Year-weighted using shares weights: 2025-26=1.00, "
             "2024-25=0.35, 2023-24=0.10."),
            "Regressions fit on non-thin-sample players only.",
            ("GP-dilution fix (2026-06-04): players who missed full seasons "
             "(gp_raw_present >= 30) have their share numerators rescaled by "
             "FULL_WINDOW_W / seasons_present_w so per-season-equivalent shares "
             "match the team denominator's 3-season scale."),
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved → {meta_path}", flush=True)

    # Print sanity preview
    print("\nTop 10 by goal_share:")
    print(out_table.head(10)[["player_name", "team", "position",
                                "goal_share", "p_assist_share",
                                "avg_toi_per_game", "ev_minutes_3yr"]]
          .to_string(index=False))
    print("\nLeague average shares by position (non-thin-sample pool):")
    for pos, vals in league_avg.items():
        print(f"  {pos}: goal={vals['goal_share']:.4f}  "
              f"p_a={vals['p_assist_share']:.4f}  s_a={vals['s_assist_share']:.4f}")
    if regressions:
        print("\nLayer-4 regression coefficients:")
        for k, v in regressions.items():
            print(f"  {k}: slope={v['slope']:.5f}  intercept={v['intercept']:.5f}  "
                  f"n={v['n']}  R²={v['r2']}")


if __name__ == "__main__":
    main()
