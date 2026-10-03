"""
build_xg_aggregates_2025_26.py  (Part A1)
-----------------------------------------
Per-skater and per-team v2 xG aggregates for 2025-26, from the validated stint
file (rapm_stints_2025_26.parquet, per-stint home_xg/away_xg = Σ xg_v2) and the
per-shot v2 scores (shots_xg_2025_26_v2.parquet). Deterministic; READ-ONLY w.r.t.
all source files. New outputs only:

  model/skater_xg_2025_26.csv
  model/team_xg_2025_26.csv

Individual (by all / 5v5 / PP / PK, shooter perspective via skaters_for/against):
  n_shots (unblocked attempts), goals, ixg (Σ xg_v2), ixg_60 (ixg per 60 of the
  matching-strength on-ice TOI from the stint file).
On-ice (all / 5v5), from stints:
  onice_xgf, onice_xga, onice_xgf_pct, onice_xgf_60, onice_xga_60.

Column names reuse the OLD files where meaning matches (`ixg`, `goals`, `n_shots`,
`xgf`, `xga`, `xgf_pct`, `team`); all strength/60 splits are new columns.
"""
from __future__ import annotations
import os
from collections import defaultdict
import numpy as np
import pandas as pd

import train_rapm_5v5 as T
import build_rapm_stints as B

HERE = T.HERE
OUT_SK = os.path.join(HERE, "skater_xg_2025_26.csv")
OUT_TM = os.path.join(HERE, "team_xg_2025_26.csv")


def load_shots():
    """enriched non-SO shots with xg_v2 attached (occ join; 21 impossible -> 0)."""
    en = pd.read_parquet(os.path.join(HERE, "shots_enriched_2025_26.parquet"))
    en = en[en.period != 5].reset_index(drop=True)
    key = ["game_id", "abs_secs", "shooter_id"]
    en["_occ"] = en.groupby(key).cumcount()
    v2 = pd.read_parquet(os.path.join(HERE, "shots_xg_2025_26_v2.parquet")).copy()
    v2["_occ"] = v2.groupby(key).cumcount()
    m = en.merge(v2[key + ["_occ", "xg_v2"]], on=key + ["_occ"], how="left")
    m["xg_v2"] = m["xg_v2"].fillna(0.0)
    return m


def strength_bucket(sf, sa):
    if sf == 5 and sa == 5:
        return "5v5"
    if sf > sa:
        return "pp"
    if sf < sa:
        return "pk"
    return "other"   # 4v4, 3v3, etc. -> only in 'all'


def build():
    stints = pd.read_parquet(os.path.join(HERE, "rapm_stints_2025_26.parquet"))
    shots = load_shots()
    names = T.player_names()
    teamabbr = T.team_abbrevs()
    gteams = T.game_teams("2025_26")
    pos = pd.read_csv(os.path.join(HERE, "player_positions.csv"))
    pos_map = dict(zip(pos.player_id.astype(int), pos.position))

    # ---------------- individual shooting ----------------
    sf = shots.skaters_for.to_numpy(); sa = shots.skaters_against.to_numpy()
    bucket = np.array([strength_bucket(a, b) for a, b in zip(sf, sa)])
    shots = shots.assign(bk=bucket)
    ind = defaultdict(lambda: defaultdict(float))
    for sit, mask in [("all", np.ones(len(shots), bool)),
                      ("5v5", bucket == "5v5"), ("pp", bucket == "pp"), ("pk", bucket == "pk")]:
        d = shots[mask]
        gp = d.groupby("shooter_id").agg(n=("is_goal", "size"), g=("is_goal", "sum"),
                                         x=("xg_v2", "sum"))
        for pid, r in gp.iterrows():
            ind[int(pid)][f"n_{sit}"] = int(r.n)
            ind[int(pid)][f"goals_{sit}"] = int(r.g)
            ind[int(pid)][f"ixg_{sit}"] = float(r.x)

    # ---------------- on-ice TOI (by strength) + on-ice xGF/xGA ----------------
    toi = defaultdict(lambda: defaultdict(float))     # pid -> {all,5v5,pp,pk}
    onice = defaultdict(lambda: defaultdict(float))   # pid -> {xgf,xga,xgf5,xga5}
    team_xg = defaultdict(lambda: defaultdict(float)) # team_id -> {xgf,xga,toi,xgf5,xga5,toi5}
    player_team_toi = defaultdict(lambda: defaultdict(float))
    for r in stints.itertuples(index=False):
        d = r.duration_secs
        hs, as_ = r.home_skaters, r.away_skaters
        is5 = (hs == 5 and as_ == 5)
        ht, at = gteams.get(int(r.game_id), (-1, -1))
        # home side
        for pid in r.home_skater_ids:
            pid = int(pid)
            toi[pid]["all"] += d
            onice[pid]["xgf"] += r.home_xg; onice[pid]["xga"] += r.away_xg
            player_team_toi[pid][ht] += d
            if is5:
                toi[pid]["5v5"] += d
                onice[pid]["xgf5"] += r.home_xg; onice[pid]["xga5"] += r.away_xg
            elif hs > as_:
                toi[pid]["pp"] += d
            elif hs < as_:
                toi[pid]["pk"] += d
        for pid in r.away_skater_ids:
            pid = int(pid)
            toi[pid]["all"] += d
            onice[pid]["xgf"] += r.away_xg; onice[pid]["xga"] += r.home_xg
            player_team_toi[pid][at] += d
            if is5:
                toi[pid]["5v5"] += d
                onice[pid]["xgf5"] += r.away_xg; onice[pid]["xga5"] += r.home_xg
            elif as_ > hs:
                toi[pid]["pp"] += d
            elif as_ < hs:
                toi[pid]["pk"] += d
        # team
        team_xg[ht]["xgf"] += r.home_xg; team_xg[ht]["xga"] += r.away_xg; team_xg[ht]["toi"] += d
        team_xg[at]["xgf"] += r.away_xg; team_xg[at]["xga"] += r.home_xg; team_xg[at]["toi"] += d
        if is5:
            team_xg[ht]["xgf5"] += r.home_xg; team_xg[ht]["xga5"] += r.away_xg; team_xg[ht]["toi5"] += d
            team_xg[at]["xgf5"] += r.away_xg; team_xg[at]["xga5"] += r.home_xg; team_xg[at]["toi5"] += d

    # primary team per player (by total on-ice TOI)
    prim = {}
    for pid, teams in player_team_toi.items():
        prim[pid] = max(teams, key=teams.get)

    # ---------------- assemble skater table ----------------
    all_pids = sorted(set(ind) | set(toi))
    rows = []
    for pid in all_pids:
        i = ind.get(pid, {}); to = toi.get(pid, {}); on = onice.get(pid, {})
        def rate(x, secs):
            return (x / (secs / 3600.0)) if secs and secs > 0 else 0.0
        xgf = on.get("xgf", 0.0); xga = on.get("xga", 0.0)
        xgf5 = on.get("xgf5", 0.0); xga5 = on.get("xga5", 0.0)
        rows.append({
            "player_id": pid, "name": names.get(pid, str(pid)),
            "team": teamabbr.get(int(prim.get(pid, -1)), "?"),
            "position": pos_map.get(pid) if isinstance(pos_map.get(pid), str) else None,
            # all-situations individual
            "n_shots": int(i.get("n_all", 0)), "goals": int(i.get("goals_all", 0)),
            "ixg": round(i.get("ixg_all", 0.0), 6),
            "ixg_60": round(rate(i.get("ixg_all", 0.0), to.get("all", 0)), 4),
            # 5v5 individual
            "n_shots_5v5": int(i.get("n_5v5", 0)), "goals_5v5": int(i.get("goals_5v5", 0)),
            "ixg_5v5": round(i.get("ixg_5v5", 0.0), 4),
            "ixg_60_5v5": round(rate(i.get("ixg_5v5", 0.0), to.get("5v5", 0)), 4),
            # PP / PK individual
            "n_shots_pp": int(i.get("n_pp", 0)), "goals_pp": int(i.get("goals_pp", 0)),
            "ixg_pp": round(i.get("ixg_pp", 0.0), 4),
            "ixg_60_pp": round(rate(i.get("ixg_pp", 0.0), to.get("pp", 0)), 4),
            "n_shots_pk": int(i.get("n_pk", 0)), "goals_pk": int(i.get("goals_pk", 0)),
            "ixg_pk": round(i.get("ixg_pk", 0.0), 4),
            "ixg_60_pk": round(rate(i.get("ixg_pk", 0.0), to.get("pk", 0)), 4),
            # on-ice all
            "xgf": round(xgf, 6), "xga": round(xga, 6),
            "xgf_pct": round(100 * xgf / (xgf + xga), 2) if (xgf + xga) > 0 else None,
            "onice_xgf_60": round(rate(xgf, to.get("all", 0)), 4),
            "onice_xga_60": round(rate(xga, to.get("all", 0)), 4),
            # on-ice 5v5
            "xgf_5v5": round(xgf5, 4), "xga_5v5": round(xga5, 4),
            "xgf_pct_5v5": round(100 * xgf5 / (xgf5 + xga5), 2) if (xgf5 + xga5) > 0 else None,
            "onice_xgf_60_5v5": round(rate(xgf5, to.get("5v5", 0)), 4),
            "onice_xga_60_5v5": round(rate(xga5, to.get("5v5", 0)), 4),
            # TOI (minutes)
            "toi_all_min": round(to.get("all", 0) / 60, 1),
            "toi_5v5_min": round(to.get("5v5", 0) / 60, 1),
        })
    sk = pd.DataFrame(rows).sort_values("player_id").reset_index(drop=True)
    sk.to_csv(OUT_SK, index=False)

    # ---------------- team table ----------------
    trows = []
    for tid, v in team_xg.items():
        if tid < 0:
            continue
        xgf, xga, to = v["xgf"], v["xga"], v["toi"]
        xgf5, xga5, to5 = v["xgf5"], v["xga5"], v["toi5"]
        trows.append({
            "team": teamabbr.get(int(tid), str(tid)),
            "xgf": round(xgf, 6), "xga": round(xga, 6),
            "xgf_pct": round(100 * xgf / (xgf + xga), 2) if (xgf + xga) > 0 else None,
            "xgf_60": round(xgf / (to / 3600), 3) if to > 0 else None,
            "xga_60": round(xga / (to / 3600), 3) if to > 0 else None,
            "xgf_5v5": round(xgf5, 3), "xga_5v5": round(xga5, 3),
            "xgf_pct_5v5": round(100 * xgf5 / (xgf5 + xga5), 2) if (xgf5 + xga5) > 0 else None,
            "xgf_60_5v5": round(xgf5 / (to5 / 3600), 3) if to5 > 0 else None,
            "xga_60_5v5": round(xga5 / (to5 / 3600), 3) if to5 > 0 else None,
        })
    tm = pd.DataFrame(trows).sort_values("xgf_pct", ascending=False).reset_index(drop=True)
    tm.to_csv(OUT_TM, index=False)
    return sk, tm, shots


if __name__ == "__main__":
    sk, tm, _ = build()
    print("WROTE", OUT_SK, len(sk), "skaters;", OUT_TM, len(tm), "teams")
