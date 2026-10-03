"""
build_shots_enriched.py
-----------------------
Season-parameterized version of build_shots_enriched_2025_26.py. Same logic
(byte-for-byte equivalent output on 2025-26 — proven by the Step-1 equivalence
test), generalized to any regular season via --season.

The audit (multi_season_audit_report.md) found 4 constants to parameterize:
SHIFTS_IN, OUT, GID_LO, GID_HI. GOALIE_MEAN_SECS (180) is unchanged and is
re-validated per season against NHL API positionCode outside this script.

For 2024-25, the 57 games whose JSON shift-chart endpoint is empty
(2024021235-2024021291) are recovered separately into
model/shifts_supplement_2024_25.parquet (same schema) via the train_rapm HTML
TOI fallback; this script reads the season cache PLUS that supplement when the
supplement file exists. Other seasons read only their season cache. Inputs are
READ ONLY.

Overlap rule, goalie identification, de-dup, edge-case flags: identical to the
2025-26 script (see that module's docstring).
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS_IN = os.path.join(HERE, "shots_augmented.parquet")
GOALIE_MEAN_SECS = 180                     # sits in the 157<->207 gap (2025-26); re-validated per season

# season key -> (game_id range, shift source files [season cache + optional supplement], output)
SEASONS = {
    "2025_26": dict(gid=(2025020001, 2025021312),
                    shifts=["season_cache/shifts_20252026.parquet"],
                    out="shots_enriched_2025_26.parquet"),
    "2024_25": dict(gid=(2024020001, 2024021312),
                    shifts=["season_cache/shifts_20242025.parquet",
                            "shifts_supplement_2024_25.parquet"],
                    out="shots_enriched_2024_25.parquet"),
    "2023_24": dict(gid=(2023020001, 2023021312),
                    shifts=["season_cache/shifts_20232024.parquet"],
                    out="shots_enriched_2023_24.parquet"),
    "2022_23": dict(gid=(2022020001, 2022021312),
                    shifts=["season_cache/shifts_20222023.parquet"],
                    out="shots_enriched_2022_23.parquet"),
}


def load_filtered(cfg):
    lo, hi = cfg["gid"]
    shots = pd.read_parquet(SHOTS_IN)
    shots = shots[(shots.game_id >= lo) & (shots.game_id <= hi)].copy()
    shots = shots.reset_index(drop=True)
    shots["_row"] = np.arange(len(shots))

    frames = []
    for rel in cfg["shifts"]:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):                       # supplement is optional
            frames.append(pd.read_parquet(p))
    shifts = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    shifts = shifts[(shifts.game_id >= lo) & (shifts.game_id <= hi)].copy()
    shifts["dur"] = shifts.abs_end - shifts.abs_start
    return shots, shifts


def derive_goalie_set(shifts: pd.DataFrame) -> set[int]:
    gp = shifts.groupby(["game_id", "player_id"])["dur"].mean().reset_index()
    return set(gp.loc[gp["dur"] > GOALIE_MEAN_SECS, "player_id"].astype(int))


def build(cfg):
    shots, shifts = load_filtered(cfg)
    goalie_set = derive_goalie_set(shifts)
    shifts["is_goalie"] = shifts.player_id.isin(goalie_set)

    n = len(shots)
    skaters_for = np.zeros(n, dtype=np.int16)
    skaters_against = np.zeros(n, dtype=np.int16)
    goalie_id = np.full(n, -1, dtype=np.int64)
    goalie_for_id = np.full(n, -1, dtype=np.int64)
    shooting_team = np.full(n, -1, dtype=np.int64)
    defending_team = np.full(n, -1, dtype=np.int64)
    on_ice_for = [None] * n
    on_ice_against = [None] * n
    empty_net_for = np.zeros(n, dtype=bool)
    empty_net_against = np.zeros(n, dtype=bool)
    f_boundary = np.zeros(n, dtype=bool)
    f_no_for = np.zeros(n, dtype=bool)
    f_no_against = np.zeros(n, dtype=bool)
    f_two_goalies = np.zeros(n, dtype=bool)
    f_shooter_office = np.zeros(n, dtype=bool)
    f_skaters_gt6 = np.zeros(n, dtype=bool)

    shots_by_game = {g: d for g, d in shots.groupby("game_id")}
    shifts_by_game = {g: d for g, d in shifts.groupby("game_id")}

    for g, sg in shots_by_game.items():
        hg = shifts_by_game.get(g)
        if hg is None:
            continue
        h_team = hg.team_id.to_numpy()
        h_pid = hg.player_id.to_numpy()
        h_start = hg.abs_start.to_numpy()
        h_end = hg.abs_end.to_numpy()
        h_goalie = hg.is_goalie.to_numpy()
        h_dur = hg.dur.to_numpy()
        teams = np.unique(h_team)
        pid2team = dict(zip(h_pid, h_team))

        for _, shot in sg.iterrows():
            r = int(shot["_row"])
            t = int(shot.abs_secs)
            shooter = int(shot.shooter_id)

            mask = (h_start < t) & (t <= h_end)
            if np.any(h_start == t) or np.any(h_end == t):
                f_boundary[r] = True

            s_team = pid2team.get(shooter)
            if s_team is None:
                f_shooter_office[r] = True
                continue
            shooting_team[r] = s_team
            d_team = teams[teams != s_team]
            d_team = int(d_team[0]) if len(d_team) else -1
            defending_team[r] = d_team

            idx = np.where(mask)[0]
            for_idx = idx[h_team[idx] == s_team]
            ag_idx = idx[h_team[idx] == d_team]

            if len(for_idx) == 0:
                f_no_for[r] = True
            if len(ag_idx) == 0:
                f_no_against[r] = True

            if shooter not in set(h_pid[for_idx].tolist()):
                f_shooter_office[r] = True

            for_sk_idx = for_idx[~h_goalie[for_idx]]
            ag_sk_idx = ag_idx[~h_goalie[ag_idx]]
            for_g_idx = for_idx[h_goalie[for_idx]]
            ag_g_idx = ag_idx[h_goalie[ag_idx]]

            for_sk_ids = pd.unique(h_pid[for_sk_idx])
            ag_sk_ids = pd.unique(h_pid[ag_sk_idx])
            sf = len(for_sk_ids)
            sa = len(ag_sk_ids)
            skaters_for[r] = sf
            skaters_against[r] = sa
            if sf > 6 or sa > 6:
                f_skaters_gt6[r] = True

            on_ice_for[r] = for_sk_ids.astype(np.int64).tolist()
            on_ice_against[r] = ag_sk_ids.astype(np.int64).tolist()

            ag_g_ids = pd.unique(h_pid[ag_g_idx])
            if len(ag_g_ids) == 0:
                empty_net_against[r] = True
            else:
                if len(ag_g_ids) > 1:
                    f_two_goalies[r] = True
                    best = ag_g_idx[np.argmax(h_dur[ag_g_idx])]
                    goalie_id[r] = int(h_pid[best])
                else:
                    goalie_id[r] = int(ag_g_ids[0])

            for_g_ids = pd.unique(h_pid[for_g_idx])
            if len(for_g_ids) == 0:
                empty_net_for[r] = True
            else:
                if len(for_g_ids) > 1:
                    best_f = for_g_idx[np.argmax(h_dur[for_g_idx])]
                    goalie_for_id[r] = int(h_pid[best_f])
                else:
                    goalie_for_id[r] = int(for_g_ids[0])

    strength_state = np.array([f"{a}v{b}" for a, b in zip(skaters_for, skaters_against)])

    out = shots.drop(columns=["_row"]).copy()
    out["shooting_team_id"] = shooting_team
    out["defending_team_id"] = defending_team
    out["skaters_for"] = skaters_for
    out["skaters_against"] = skaters_against
    out["strength_state"] = strength_state
    out["empty_net_for"] = empty_net_for
    out["empty_net_against"] = empty_net_against
    out["goalie_id"] = pd.Series(goalie_id, index=out.index).astype("Int64").where(goalie_id >= 0, pd.NA)
    out["goalie_for_id"] = pd.Series(goalie_for_id, index=out.index).astype("Int64").where(goalie_for_id >= 0, pd.NA)
    out["on_ice_for"] = on_ice_for
    out["on_ice_against"] = on_ice_against
    out["flag_shift_boundary"] = f_boundary
    out["flag_no_shifts_for"] = f_no_for
    out["flag_no_shifts_against"] = f_no_against
    out["flag_two_goalies_against"] = f_two_goalies
    out["flag_shooter_not_onice"] = f_shooter_office
    out["flag_skaters_gt6"] = f_skaters_gt6

    out.to_parquet(os.path.join(HERE, cfg["out"]), index=False)

    print("SEASON_OUT", cfg["out"])
    print("INPUT_SHOTS", len(shots))
    print("OUTPUT_ROWS", len(out))
    print("GOALIE_SET_SIZE", len(goalie_set))
    print("FLAG_boundary", int(f_boundary.sum()))
    print("FLAG_no_shifts_for", int(f_no_for.sum()))
    print("FLAG_no_shifts_against", int(f_no_against.sum()))
    print("FLAG_two_goalies_against", int(f_two_goalies.sum()))
    print("FLAG_shooter_not_onice", int(f_shooter_office.sum()))
    print("FLAG_skaters_gt6", int(f_skaters_gt6.sum()))
    print("EMPTY_NET_against", int(empty_net_against.sum()))
    print("EMPTY_NET_for", int(empty_net_for.sum()))
    print("GOALIE_ID_null", int(out.goalie_id.isna().sum()))
    vc = pd.Series(strength_state).value_counts()
    print("STRENGTH_STATE_TOP")
    print(vc.head(15).to_string())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, choices=list(SEASONS.keys()))
    a = ap.parse_args()
    build(SEASONS[a.season])
