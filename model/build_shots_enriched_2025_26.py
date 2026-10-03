"""
build_shots_enriched_2025_26.py
--------------------------------
Self-generated enrichment step for the single-season analytics layer.

Reconstructs the three fields that shots_augmented.parquet does NOT store
(per data_audit_report.md) by time-overlapping each shot event against the
shift intervals in shifts_20252026.parquet:

  1. STRENGTH STATE  -- skater counts per team at the shot instant (5v5, 5v4, ...)
  2. GOALIE_ID       -- goalie on-ice for the team being shot AGAINST
  3. ON-ICE ARRAYS   -- skater player_ids on-ice for the shooting and defending teams

Scope: regular season 2025-26 ONLY  (game_id 2025020001 .. 2025021312).
Inputs are READ ONLY. Output: model/shots_enriched_2025_26.parquet.

Overlap rule: a player is on-ice for a shot at time t iff
    abs_start < t <= abs_end          (EXCLUSIVE start, inclusive end)
A shot is a live-play event: it happens during the shift that was already in
progress, not the line arriving at the change. So at a line-change boundary
(outgoing shift end == t == incoming shift start) we credit the OUTGOING line
and drop the incoming one. This (a) keeps the shooter on the ice for their own
shot -- a pure [start, end) rule silently dropped 5,871 shooters whose shift
was logged as ending on the shot second -- and (b) still prevents double-
counting the changing line (a pure [start, end] rule inflated 12,803 shots to
>5 skaters). Under this hybrid rule shooter-not-on-ice falls to 11 (true source
quirks) and the only remaining >5-skater FOR cases are legitimate pulled-goalie
(6 skaters, no goalie) situations.

Goalie identification: there is no position field in the shift data, so a
season-level goalie set is derived first -- any player whose MEAN shift
duration exceeds GOALIE_MEAN_SECS in ANY game is a goalie for ALL games.
This is validated against the clean statistical gap between skaters
(max mean shift ~157s) and goalies (min mean shift ~207s) and reproduces
MoneyPuck's 98-goalie roster exactly. Season-level (vs per-game) is required
so brief goalie appearances (e.g. a 6-second relief stint) are not
misclassified as skaters.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS_IN = os.path.join(HERE, "shots_augmented.parquet")
SHIFTS_IN = os.path.join(HERE, "season_cache", "shifts_20252026.parquet")
OUT = os.path.join(HERE, "shots_enriched_2025_26.parquet")

GID_LO, GID_HI = 2025020001, 2025021312   # regular season 2025-26
GOALIE_MEAN_SECS = 180                     # sits in the 157<->207 gap


def load_filtered():
    shots = pd.read_parquet(SHOTS_IN)
    shots = shots[(shots.game_id >= GID_LO) & (shots.game_id <= GID_HI)].copy()
    shots = shots.reset_index(drop=True)
    shots["_row"] = np.arange(len(shots))

    shifts = pd.read_parquet(SHIFTS_IN)
    shifts = shifts[(shifts.game_id >= GID_LO) & (shifts.game_id <= GID_HI)].copy()
    shifts["dur"] = shifts.abs_end - shifts.abs_start
    return shots, shifts


def derive_goalie_set(shifts: pd.DataFrame) -> set[int]:
    gp = shifts.groupby(["game_id", "player_id"])["dur"].mean().reset_index()
    return set(gp.loc[gp["dur"] > GOALIE_MEAN_SECS, "player_id"].astype(int))


def build():
    shots, shifts = load_filtered()
    goalie_set = derive_goalie_set(shifts)
    shifts["is_goalie"] = shifts.player_id.isin(goalie_set)

    n = len(shots)
    # output columns
    skaters_for = np.zeros(n, dtype=np.int16)
    skaters_against = np.zeros(n, dtype=np.int16)
    goalie_id = np.full(n, -1, dtype=np.int64)          # -1 -> null at write time
    goalie_for_id = np.full(n, -1, dtype=np.int64)
    shooting_team = np.full(n, -1, dtype=np.int64)
    defending_team = np.full(n, -1, dtype=np.int64)
    on_ice_for = [None] * n        # skater ids, shooting team
    on_ice_against = [None] * n    # skater ids, defending team
    empty_net_for = np.zeros(n, dtype=bool)
    empty_net_against = np.zeros(n, dtype=bool)
    # edge-case flags
    f_boundary = np.zeros(n, dtype=bool)       # shot lands exactly on a shift boundary
    f_no_for = np.zeros(n, dtype=bool)         # zero shifts found for shooting team
    f_no_against = np.zeros(n, dtype=bool)     # zero shifts found for defending team
    f_two_goalies = np.zeros(n, dtype=bool)    # >1 goalie on-ice for defending team
    f_shooter_office = np.zeros(n, dtype=bool) # shooter not in on-ice set at t
    f_skaters_gt6 = np.zeros(n, dtype=bool)    # >6 distinct skaters a side (residual raw anomaly)

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
        # per-game player -> team map (team is constant within a game)
        pid2team = dict(zip(h_pid, h_team))

        for _, shot in sg.iterrows():
            r = int(shot["_row"])
            t = int(shot.abs_secs)
            shooter = int(shot.shooter_id)

            mask = (h_start < t) & (t <= h_end)   # exclusive start, inclusive end
            # boundary flag: t coincides with any start or end in this game
            if np.any(h_start == t) or np.any(h_end == t):
                f_boundary[r] = True

            # shooting team from the game-level player->team map
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

            # shooter should be among on-ice for-team players
            if shooter not in set(h_pid[for_idx].tolist()):
                f_shooter_office[r] = True

            for_sk_idx = for_idx[~h_goalie[for_idx]]
            ag_sk_idx = ag_idx[~h_goalie[ag_idx]]
            for_g_idx = for_idx[h_goalie[for_idx]]
            ag_g_idx = ag_idx[h_goalie[ag_idx]]

            # De-duplicate skaters by player_id: 0.01% of raw shift rows are
            # self-overlapping (a player with two rows both covering t), which
            # would otherwise inflate the skater count and produce impossible
            # states like 7v6 / 8v7. Count DISTINCT players on-ice.
            for_sk_ids = pd.unique(h_pid[for_sk_idx])
            ag_sk_ids = pd.unique(h_pid[ag_sk_idx])
            sf = len(for_sk_ids)
            sa = len(ag_sk_ids)
            skaters_for[r] = sf
            skaters_against[r] = sa
            if sf > 6 or sa > 6:
                # >6 distinct skaters is physically impossible (6 = pulled
                # goalie); after de-dup these are residual raw-shift anomalies.
                f_skaters_gt6[r] = True

            on_ice_for[r] = for_sk_ids.astype(np.int64).tolist()
            on_ice_against[r] = ag_sk_ids.astype(np.int64).tolist()

            # goalie for the DEFENDING team (the one being shot against)
            ag_g_ids = pd.unique(h_pid[ag_g_idx])
            if len(ag_g_ids) == 0:
                empty_net_against[r] = True            # goalie_id stays -1 (null)
            else:
                if len(ag_g_ids) > 1:
                    f_two_goalies[r] = True
                    # resolve: goalie whose shift has the longest duration at t
                    best = ag_g_idx[np.argmax(h_dur[ag_g_idx])]
                    goalie_id[r] = int(h_pid[best])
                else:
                    goalie_id[r] = int(ag_g_ids[0])

            # goalie for the SHOOTING team (for empty-net-for detection)
            for_g_ids = pd.unique(h_pid[for_g_idx])
            if len(for_g_ids) == 0:
                empty_net_for[r] = True
            else:
                if len(for_g_ids) > 1:
                    best_f = for_g_idx[np.argmax(h_dur[for_g_idx])]
                    goalie_for_id[r] = int(h_pid[best_f])
                else:
                    goalie_for_id[r] = int(for_g_ids[0])

    # strength label from the shooter's perspective, e.g. "5v4"
    strength_state = np.array([f"{a}v{b}" for a, b in zip(skaters_for, skaters_against)])

    out = shots.drop(columns=["_row"]).copy()
    out["shooting_team_id"] = shooting_team
    out["defending_team_id"] = defending_team
    out["skaters_for"] = skaters_for
    out["skaters_against"] = skaters_against
    out["strength_state"] = strength_state
    out["empty_net_for"] = empty_net_for
    out["empty_net_against"] = empty_net_against
    # nullable Int64 so downstream joins on goalie_id stay integer-clean
    out["goalie_id"] = pd.Series(goalie_id, index=out.index).astype("Int64").where(goalie_id >= 0, pd.NA)          # facing goalie
    out["goalie_for_id"] = pd.Series(goalie_for_id, index=out.index).astype("Int64").where(goalie_for_id >= 0, pd.NA)
    out["on_ice_for"] = on_ice_for
    out["on_ice_against"] = on_ice_against
    # edge-case flags
    out["flag_shift_boundary"] = f_boundary
    out["flag_no_shifts_for"] = f_no_for
    out["flag_no_shifts_against"] = f_no_against
    out["flag_two_goalies_against"] = f_two_goalies
    out["flag_shooter_not_onice"] = f_shooter_office
    out["flag_skaters_gt6"] = f_skaters_gt6

    out.to_parquet(OUT, index=False)

    # ---- reconciliation summary (printed; captured into the audit note) ----
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
    build()
