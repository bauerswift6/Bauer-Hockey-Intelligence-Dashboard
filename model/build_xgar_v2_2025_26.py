"""
build_xgar_v2_2025_26.py  (xGAR stopgap, recomputed on v2 ixG, SELF-GENERATED TOI)
---------------------------------------------------------------------------------
Recomputes the displayed xGAR using the IDENTICAL stopgap formula from
build_self_generated_stats.py::build_skater_xgar, but with:
  * individual xG from the v2 model (model/skater_xg_2025_26.csv), and
  * TOI, team, and position all SELF-GENERATED (no MoneyPuck in any output).

build_self_generated_stats.py is READ ONLY (imported only for the Step-2 MoneyPuck
comparison, which never feeds an output).

Formula (verbatim from the stopgap):
  xGAR = ixG - replacement_ixG_per_60[pos_group] * (TOI_all_seconds / 3600)
Replacement pool = forwards ranked outside their team's top 13 by TOI and
defensemen outside their team's top 7, pooled league-wide by position group;
replacement_ixG_per_60 = Σ pool ixG / (Σ pool TOI / 3600). All situations.

SELF-GENERATED inputs (the fix):
  * TOI_all_seconds: per-skater all-situations on-ice time from the validated
    stint file (model/rapm_stints_2025_26.parquet) = Σ stint duration over every
    stint where the skater is an on-ice skater (home or away). This feeds BOTH the
    TOI term and the replacement-pool ranking.
  * team / position / name: from model/skater_xg_2025_26.csv, which derives team as
    the skater's primary team (most all-situations on-ice stint TOI), position from
    model/player_positions.csv (NHL API cache), and name from the shift file.
  Multi-team rule: a traded skater is assigned to the ONE team where he logged the
  most all-situations on-ice stint TOI, and is ranked within that team only (one
  pool membership per player). This matches skater_xg_2025_26.csv's `team`.

Output: model/xgar_v2_2025_26.csv (same columns as skater_xgar_self_generated.csv).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SKATER_XG_V2 = os.path.join(HERE, "skater_xg_2025_26.csv")
STINTS = os.path.join(HERE, "rapm_stints_2025_26.parquet")
ENRICHED = os.path.join(HERE, "shots_enriched_2025_26.parquet")
OUT = os.path.join(HERE, "xgar_v2_2025_26.csv")


def self_toi_from_stints() -> pd.Series:
    """All-situations on-ice TOI (seconds) per skater from the validated stint file."""
    st = pd.read_parquet(STINTS, columns=["duration_secs", "home_skater_ids", "away_skater_ids"])
    from collections import defaultdict
    toi = defaultdict(float)
    for d, hs, aw in zip(st.duration_secs.to_numpy(), st.home_skater_ids, st.away_skater_ids):
        for p in hs:
            toi[int(p)] += d
        for p in aw:
            toi[int(p)] += d
    return pd.Series(toi, name="toi_sec")


def sog_from_enriched() -> pd.DataFrame:
    en = pd.read_parquet(ENRICHED, columns=["period", "shooter_id", "on_goal"])
    en = en[en.period != 5]
    return (en.groupby("shooter_id").on_goal.sum().rename("sog_my").reset_index()
              .rename(columns={"shooter_id": "player_id"}))


def compute_xgar(agg: pd.DataFrame):
    """agg: player_id, name, team, position, n_shots, ixg, goals, sog_my, toi_sec.
    Returns (output_df, baseline dict). No external sources."""
    grp = agg.copy()
    grp["goals_above_expected"] = (grp["goals"] - grp["ixg"]).round(3)
    grp["_grp"] = np.where(grp["position"].isin(["C", "L", "R"]), "F",
                           np.where(grp["position"] == "D", "D", None))
    grp["toi_sec"] = pd.to_numeric(grp["toi_sec"], errors="coerce").fillna(0.0)
    ranked = grp.dropna(subset=["_grp", "team"]).copy()
    ranked = ranked[ranked["toi_sec"] > 0]
    ranked["team_rank"] = (ranked.groupby(["team", "_grp"])["toi_sec"]
                           .rank(ascending=False, method="first"))
    repl = ranked[((ranked["_grp"] == "F") & (ranked["team_rank"] > 13)) |
                  ((ranked["_grp"] == "D") & (ranked["team_rank"] > 7))]
    REPL_IXG_60 = {}
    for _g in ["F", "D"]:
        _pool = repl[repl["_grp"] == _g]
        _hours = _pool["toi_sec"].sum() / 3600.0
        REPL_IXG_60[_g] = (_pool["ixg"].sum() / _hours) if _hours > 0 else 0.0
    _baseline = grp["_grp"].map(REPL_IXG_60).fillna(REPL_IXG_60["F"])
    grp["xgar"] = (grp["ixg"] - _baseline * (grp["toi_sec"] / 3600.0)).round(3)

    grp = grp[["player_id", "name", "team", "position",
               "n_shots", "ixg", "goals", "sog_my",
               "goals_above_expected", "xgar"]]
    grp["ixg"] = grp["ixg"].round(2)
    grp = grp.sort_values("xgar", ascending=False).reset_index(drop=True)
    return grp, REPL_IXG_60


def build_self_agg() -> pd.DataFrame:
    sx = pd.read_csv(SKATER_XG_V2)[["player_id", "name", "team", "position",
                                    "n_shots", "goals", "ixg"]]
    sog = sog_from_enriched()
    toi = self_toi_from_stints().reset_index().rename(columns={"index": "player_id"})
    a = sx.merge(sog, on="player_id", how="left").merge(toi, on="player_id", how="left")
    a["sog_my"] = a["sog_my"].fillna(0).astype(int)
    a["toi_sec"] = a["toi_sec"].fillna(0.0)
    return a


def main():
    agg = build_self_agg()
    out, base = compute_xgar(agg)
    out.to_csv(OUT, index=False)
    print(f"WROTE {OUT} ({len(out)} skaters)")
    print(f"SELF-TOI baseline ixG/60: F={base['F']:.4f} D={base['D']:.4f}")
    return out, base, agg


if __name__ == "__main__":
    main()
