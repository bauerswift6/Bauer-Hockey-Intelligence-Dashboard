"""
build_display_tables_2025_26.py  (Part A3)
------------------------------------------
Display-ready tables derived from the final rebuild outputs. READ-ONLY w.r.t. all
source files (never modifies them). Deterministic. New outputs only:

  model/gsax_2025_26_display.csv          — GSAX recentered to league total 0
  model/qoc_qot_5v5_2025_26_display.csv   — QoC/QoT with within-position percentiles
  model/rapm_display_2025_26.csv          — 5v5 RAPM joined with special-teams RAPM
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def gsax_display():
    g = pd.read_csv(os.path.join(HERE, "gsax_2025_26.csv"))
    # recenter: scale xGA by league GA/xGA (all situations) so Σ GSAX = 0
    factor = g["all_ga"].sum() / g["all_xga"].sum()
    out = g.copy()
    for sit in ["all", "5v5", "pk"]:
        xga = g[f"{sit}_xga"]; ga = g[f"{sit}_ga"]; sh = g[f"{sit}_shots"]
        toi_col = {"all": "toi_all_min", "5v5": "toi_5v5_min", "pk": "toi_pk_min"}[sit]
        toi_min = g[toi_col]
        # raw copies
        out[f"{sit}_xga_raw"] = xga
        out[f"{sit}_gsax_raw"] = g[f"{sit}_gsax"]
        out[f"{sit}_xsv_raw"] = g[f"{sit}_xsv"]
        out[f"{sit}_gsax_100_raw"] = g[f"{sit}_gsax_100"]
        out[f"{sit}_gsax_60_raw"] = g[f"{sit}_gsax_60"]
        # scaled
        sx = xga * factor
        gsax = sx - ga
        out[f"{sit}_xga"] = sx.round(6)
        out[f"{sit}_gsax"] = gsax.round(6)
        out[f"{sit}_xsv"] = np.where(sh > 0, 1 - sx / sh, np.nan).round(5)
        out[f"{sit}_gsax_100"] = np.where(sh > 0, gsax / sh * 100, np.nan).round(4)
        out[f"{sit}_gsax_60"] = np.where(toi_min > 0, gsax / (toi_min / 60), np.nan).round(4)
    out.to_csv(os.path.join(HERE, "gsax_2025_26_display.csv"), index=False)
    return factor, float(out["all_gsax"].sum())


def qoc_qot_display():
    q = pd.read_csv(os.path.join(HERE, "qoc_qot_5v5_2025_26.csv"))
    q = q.drop(columns=["qot_toi"])            # not displayed; qoc_toi kept but not shown
    q["grp"] = q.position.map(lambda p: "D" if p == "D" else ("F" if p in ("C", "L", "R") else None))
    q["small_sample"] = q.toi_min_5v5 < 200
    cols = ["qot_total", "qot_offense", "qot_defense", "qoc_total", "qoc_offense", "qoc_defense"]
    for c in cols:
        q[f"{c}_pctile"] = np.nan
    qual = q[(~q.small_sample) & q.grp.notna()]
    for grp in ["F", "D"]:
        sub = qual[qual.grp == grp]
        for c in cols:
            pct = sub[c].rank(pct=True) * 100
            q.loc[pct.index, f"{c}_pctile"] = pct.round(1)
    n_f = int((qual.grp == "F").sum()); n_d = int((qual.grp == "D").sum())
    n_small = int(q.small_sample.sum())
    q.to_csv(os.path.join(HERE, "qoc_qot_5v5_2025_26_display.csv"), index=False)
    return n_f, n_d, n_small


def rapm_display():
    r5 = pd.read_csv(os.path.join(HERE, "rapm_5v5_2025_26.csv"))
    rst = pd.read_csv(os.path.join(HERE, "rapm_st_2025_26.csv"))
    out = r5[["player_id", "name", "position", "teams", "toi_min_5v5",
              "offense", "defense", "total", "total_impact"]].copy()
    out = out.merge(rst[["player_id", "pp_toi_min", "pp_offense", "pk_toi_min", "pk_defense"]],
                    on="player_id", how="left")
    # rank by total among >=500 5v5 min (null below)
    elig = out[out.toi_min_5v5 >= 500].copy()
    elig["rank_total"] = elig["total"].rank(ascending=False, method="min").astype(int)
    out = out.merge(elig[["player_id", "rank_total"]], on="player_id", how="left")
    out["rank_total"] = out["rank_total"].astype("Int64")
    out["pk_low_confidence"] = True
    out.to_csv(os.path.join(HERE, "rapm_display_2025_26.csv"), index=False)
    return len(out), int(out.rank_total.notna().sum())


def main():
    factor, league_total = gsax_display()
    print(f"GSAX display: recenter factor={factor:.6f}, all-sit league GSAX={league_total:.2e}")
    n_f, n_d, n_small = qoc_qot_display()
    print(f"QoC/QoT display: percentile pop F={n_f} D={n_d}; small_sample(<200min)={n_small}")
    n, n_ranked = rapm_display()
    print(f"RAPM display: {n} skaters, {n_ranked} ranked (>=500 5v5 min)")


if __name__ == "__main__":
    main()
