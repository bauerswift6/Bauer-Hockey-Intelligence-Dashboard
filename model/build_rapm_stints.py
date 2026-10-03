"""
build_rapm_stints.py
--------------------
Season-parameterized version of build_rapm_stints_2025_26.py. The stint
construction, shot assignment, score differential, and edge-case logic are
REUSED verbatim by importing the functions from build_rapm_stints_2025_26
(build_stints, attach_shot_stats, attach_score_diff) and the enrichment goalie
identification (derive_goalie_set). Only the file paths / game-id range and the
shift-source list (season cache + optional supplement) are parameterized.

Equivalence: running --season 2025_26 reproduces model/rapm_stints_2025_26.parquet
exactly (proven by build_rapm_stints.py's __main__ equivalence path / the report).

Inputs are READ ONLY. Output: model/rapm_stints_<season>.parquet (or --out).
Regular season only; period 5 (shootout) excluded everywhere.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

import build_shots_enriched as bse
import build_rapm_stints_2025_26 as B0

HERE = os.path.dirname(os.path.abspath(__file__))
SHOOTOUT_PERIOD = 5

SEASONS = {
    "2025_26": dict(
        gid=(2025020001, 2025021312),
        shifts=["season_cache/shifts_20252026.parquet"],
        enriched="shots_enriched_2025_26.parquet",
        xgv2="shots_xg_2025_26_v2.parquet",
        out="rapm_stints_2025_26.parquet"),
    "2024_25": dict(
        gid=(2024020001, 2024021312),
        shifts=["season_cache/shifts_20242025.parquet",
                "shifts_supplement_2024_25.parquet"],
        enriched="shots_enriched_2024_25.parquet",
        xgv2="shots_xg_2024_25_v2.parquet",
        out="rapm_stints_2024_25.parquet"),
    "2023_24": dict(
        gid=(2023020001, 2023021312),
        shifts=["season_cache/shifts_20232024.parquet"],
        enriched="shots_enriched_2023_24.parquet",
        xgv2="shots_xg_2023_24_v2.parquet",
        out="rapm_stints_2023_24.parquet"),
}


def load_shifts(cfg) -> pd.DataFrame:
    lo, hi = cfg["gid"]
    frames = []
    for rel in cfg["shifts"]:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):                      # supplement is optional
            frames.append(pd.read_parquet(p))
    sh = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    sh = sh[(sh.game_id >= lo) & (sh.game_id <= hi)].copy()
    sh = sh[sh.period != SHOOTOUT_PERIOD].copy()
    sh["dur"] = sh.abs_end - sh.abs_start
    return sh


def load_shots_with_xg(cfg) -> pd.DataFrame:
    """Non-shootout enriched shots joined to xg_v2 without fan-out.

    Identical occurrence-index alignment to build_rapm_stints_2025_26; xg_v2 is a
    subset of non-shootout shots so unmatched rows keep their attempts/goals and
    contribute 0 to xG.
    """
    lo, hi = cfg["gid"]
    en = pd.read_parquet(os.path.join(HERE, cfg["enriched"]))
    en = en[(en.game_id >= lo) & (en.game_id <= hi)].copy()
    en = en[en.period != SHOOTOUT_PERIOD].copy().reset_index(drop=True)
    key = ["game_id", "abs_secs", "shooter_id"]
    en["_occ"] = en.groupby(key).cumcount()

    xg = pd.read_parquet(os.path.join(HERE, cfg["xgv2"])).copy()
    xg["_occ"] = xg.groupby(key).cumcount()

    m = en.merge(xg[key + ["_occ", "xg_v2"]], on=key + ["_occ"], how="left",
                 indicator=True)
    m["has_xg"] = (m["_merge"] == "both").to_numpy()
    m["xg_v2_filled"] = m["xg_v2"].fillna(0.0)
    return m


def build(season: str, out_path: str | None = None):
    cfg = SEASONS[season]
    shifts = load_shifts(cfg)
    goalie_set = bse.derive_goalie_set(shifts)

    shots = load_shots_with_xg(cfg)
    ht = shots.assign(home=np.where(shots.is_home == 1,
                                    shots.shooting_team_id,
                                    shots.defending_team_id))
    home_team = ht.groupby("game_id")["home"].first().astype(int).to_dict()

    # ---- identical construction/assembly to build_rapm_stints_2025_26.main ----
    records, self_overlap_rows = B0.build_stints(shifts, goalie_set, home_team)
    stints = pd.DataFrame(records)

    stints["home_skaters"] = stints.h_sk.map(len).astype(np.int16)
    stints["away_skaters"] = stints.a_sk.map(len).astype(np.int16)
    stints["home_skater_ids"] = stints.h_sk.map(lambda s: sorted(s))
    stints["away_skater_ids"] = stints.a_sk.map(lambda s: sorted(s))
    stints["home_goalie_id"] = pd.array([x if x is not None else pd.NA
                                         for x in stints.h_g], dtype="Int64")
    stints["away_goalie_id"] = pd.array([x if x is not None else pd.NA
                                         for x in stints.a_g], dtype="Int64")
    stints["home_empty_net"] = stints.h_g.map(lambda x: x is None)
    stints["away_empty_net"] = stints.a_g.map(lambda x: x is None)
    stints["duration_secs"] = (stints.end_secs - stints.start_secs).astype(np.int32)
    stints["flag_two_goalies"] = stints.two_goalies
    stints["flag_empty_side"] = stints.empty_side

    stints, unmatched_shots, imp_df, shot_sid = B0.attach_shot_stats(
        stints, shots, home_team)
    stints = B0.attach_score_diff(stints, shots, home_team)

    cols = ["game_id", "period", "start_secs", "end_secs", "duration_secs",
            "home_skaters", "away_skaters", "home_skater_ids", "away_skater_ids",
            "home_goalie_id", "away_goalie_id", "home_empty_net", "away_empty_net",
            "score_diff_home",
            "home_attempts", "away_attempts", "home_goals", "away_goals",
            "home_xg", "away_xg",
            "flag_two_goalies", "flag_empty_side"]
    out = stints[cols].sort_values(["game_id", "period", "start_secs"]).reset_index(drop=True)

    dest = out_path or os.path.join(HERE, cfg["out"])
    out.to_parquet(dest, index=False)
    meta = dict(self_overlap_rows=self_overlap_rows, unmatched_shots=unmatched_shots,
                imp_df=imp_df, home_team=home_team, goalie_set=goalie_set)
    print("SEASON", season, "-> STINTS", len(out), "DEST", dest)
    print("SELF_OVERLAP_ROWS", self_overlap_rows, "UNMATCHED_SHOTS", unmatched_shots,
          "IMPOSSIBLE_STRENGTH_SHOTS", len(imp_df))
    return out, meta, shots, shifts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, choices=list(SEASONS.keys()))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build(a.season, a.out)
