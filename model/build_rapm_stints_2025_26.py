"""
build_rapm_stints_2025_26.py
----------------------------
Build the stint-level dataset that RAPM will be fit on, for the 2025-26
REGULAR SEASON only (game_id 2025020001 .. 2025021312). Deterministic; no
randomness. All inputs are READ ONLY. Output is written to NEW files only:

  model/rapm_stints_2025_26.parquet   (this script)
  rapm_stints_report.md               (written by report_rapm_stints.py)

A STINT is a maximal stretch of play within a single game and period during
which the exact set of skaters and goalies on the ice for BOTH teams stays
constant.

------------------------------------------------------------------------------
Consistency with the enrichment step (model/build_shots_enriched.py / _2025_26)
------------------------------------------------------------------------------
The enrichment step decides who is on ice for a shot at instant t with the rule

    abs_start < t <= abs_end            (exclusive start, inclusive end)

and counts DISTINCT player_ids per side (collapsing self-overlapping raw shift
rows). We reuse BOTH decisions so the two datasets agree on who was on the ice
at every moment:

* We import `derive_goalie_set` from build_shots_enriched so the goalie roster
  is byte-identical to enrichment (season-level mean-shift rule, 98 goalies).
* On-ice membership for a stint is defined so that the enrichment shot rule
  `start < t <= end` maps a shot at t into exactly the stint whose on-ice set
  equals enrichment's on-ice set at t. Concretely, over the sorted breakpoints
  B = unique(abs_start) U unique(abs_end) of a (game, period), the elementary
  interval (B[k], B[k+1]] has on-ice set

      { shift : abs_start <= B[k]  AND  abs_end >= B[k+1] }

  because within (B[k], B[k+1]] there is no breakpoint, so `abs_start < t` holds
  for all such t iff abs_start <= B[k], and `t <= abs_end` holds iff
  abs_end >= B[k+1]. A shot at t is assigned to the stint with
  `start_secs < t <= end_secs` -- the identical boundary rule -- so it lands in
  the stint where its shooter was credited on ice by enrichment. On-ice players
  are counted as DISTINCT player_ids (pd.unique), exactly as enrichment does,
  which collapses the 146 self-overlapping raw shift rows.

Elementary intervals that are adjacent AND share the identical (home skaters,
away skaters, home goalie, away goalie) set are MERGED into one maximal stint
(this is what makes a stint "maximal" and also absorbs same-player overlap
artifacts, which split a breakpoint without changing the distinct on-ice set).

Because abs times are game-absolute and every regular-season shift lies wholly
within one period band (P1 [0,1200], P2 [1200,2400], P3 [2400,3600],
OT/P4 [3600,3900]), building per (game, period) makes period-spanning
impossible by construction; we verify the count is zero.

Scope: regular season only. Period 5 (shootout) carries no ice time and is
excluded everywhere. No playoff game (2025030xxx) is ever read.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# Reuse the enrichment goalie identification verbatim (identical goalie set).
import build_shots_enriched as bse

HERE = os.path.dirname(os.path.abspath(__file__))
SHIFTS_IN = os.path.join(HERE, "season_cache", "shifts_20252026.parquet")
ENRICHED_IN = os.path.join(HERE, "shots_enriched_2025_26.parquet")
XGV2_IN = os.path.join(HERE, "shots_xg_2025_26_v2.parquet")
OUT = os.path.join(HERE, "rapm_stints_2025_26.parquet")

GID_LO, GID_HI = 2025020001, 2025021312   # regular season 2025-26
SHOOTOUT_PERIOD = 5


# ----------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------
def load_shifts() -> pd.DataFrame:
    sh = pd.read_parquet(SHIFTS_IN)
    sh = sh[(sh.game_id >= GID_LO) & (sh.game_id <= GID_HI)].copy()
    sh = sh[sh.period != SHOOTOUT_PERIOD].copy()            # no shootout ice time
    sh["dur"] = sh.abs_end - sh.abs_start
    return sh


def load_shots_with_xg() -> pd.DataFrame:
    """Non-shootout enriched shots joined to xg_v2 without fan-out.

    The (game_id, abs_secs, shooter_id) key is non-unique on genuine rapid-fire
    same-second pairs (verified real events -- NOT dropped). A naive merge would
    fan out. We align positionally by within-key occurrence index (cumcount),
    which is exact for stint aggregation: same-second duplicates share a second,
    hence share a stint, so the stint's xG sum is invariant to which duplicate
    maps to which xg_v2 value. xg_v2 is a subset of the non-shootout shots
    (minus 21 impossible-strength rows dropped from xG training), so the left
    join leaves exactly those 21 with NA xg_v2 -> counted as attempts/goals,
    contributing 0 to xG sums.
    """
    en = pd.read_parquet(ENRICHED_IN)
    en = en[en.period != SHOOTOUT_PERIOD].copy().reset_index(drop=True)
    key = ["game_id", "abs_secs", "shooter_id"]
    en["_occ"] = en.groupby(key).cumcount()

    xg = pd.read_parquet(XGV2_IN).copy()
    xg["_occ"] = xg.groupby(key).cumcount()

    m = en.merge(xg[key + ["_occ", "xg_v2"]], on=key + ["_occ"], how="left",
                 indicator=True)
    m["has_xg"] = (m["_merge"] == "both").to_numpy()
    m["xg_v2_filled"] = m["xg_v2"].fillna(0.0)
    return m


# ----------------------------------------------------------------------------
# Stint construction (per game, per period)
# ----------------------------------------------------------------------------
def build_stints(shifts: pd.DataFrame, goalie_set: set, home_team: dict):
    shifts = shifts.copy()
    shifts["is_goalie"] = shifts.player_id.isin(goalie_set)

    records = []          # one dict per maximal stint
    self_overlap_rows = 0 # raw shift rows that self-overlap another row (same player)

    for (g, p), grp in shifts.groupby(["game_id", "period"], sort=True):
        start = grp.abs_start.to_numpy()
        end = grp.abs_end.to_numpy()
        pid = grp.player_id.to_numpy()
        team = grp.team_id.to_numpy()
        isg = grp.is_goalie.to_numpy()
        dur = grp.dur.to_numpy()

        h_team = home_team.get(int(g))
        # count self-overlapping rows (same player, overlapping intervals)
        self_overlap_rows += _count_self_overlaps(pid, start, end)

        bp = np.unique(np.concatenate([start, end]))
        if len(bp) < 2:
            continue

        # Elementary intervals (B[k], B[k+1]]; on-ice = start<=B[k] & end>=B[k+1]
        left = bp[:-1]
        right = bp[1:]
        # membership matrix (n_shifts, n_intervals)
        member = (start[:, None] <= left[None, :]) & (end[:, None] >= right[None, :])

        elem = []   # (h_sk frozenset, a_sk frozenset, h_g, a_g, start, end, flags)
        for k in range(len(left)):
            on = member[:, k]
            if not on.any():
                # coverage gap: emit as an (empty) stint to preserve clock time
                elem.append(_make_interval(frozenset(), frozenset(), None, None,
                                           int(left[k]), int(right[k]),
                                           empty_side=True))
                continue
            o_pid = pid[on]
            o_team = team[on]
            o_isg = isg[on]
            o_dur = dur[on]

            is_home_side = (o_team == h_team)
            elem.append(_resolve_side(
                o_pid, o_isg, o_dur, is_home_side,
                int(left[k]), int(right[k])))

        # Merge adjacent elementary intervals with identical on-ice sets
        merged = _merge_adjacent(elem)
        for st in merged:
            st["game_id"] = int(g)
            st["period"] = int(p)
            records.append(st)

    return records, self_overlap_rows


def _count_self_overlaps(pid, start, end) -> int:
    """Number of raw shift rows that overlap another row of the SAME player."""
    n = 0
    order = np.argsort(pid, kind="mergesort")
    pid_s, st_s, en_s = pid[order], start[order], end[order]
    i = 0
    while i < len(pid_s):
        j = i
        while j < len(pid_s) and pid_s[j] == pid_s[i]:
            j += 1
        # rows [i, j) share a player; count those overlapping any sibling
        if j - i > 1:
            a = st_s[i:j]
            b = en_s[i:j]
            o = np.argsort(a, kind="mergesort")
            a, b = a[o], b[o]
            for r in range(len(a)):
                # overlaps a sibling if any other interval intersects (a[r], b[r])
                inter = (a < b[r]) & (b > a[r])
                inter[r] = False
                if inter.any():
                    n += 1
        i = j
    return n


def _resolve_side(o_pid, o_isg, o_dur, is_home_side, s, e):
    def side(mask):
        p_all = o_pid[mask]
        g_mask = o_isg[mask]
        sk = pd.unique(p_all[~g_mask])                 # distinct skaters (enrichment rule)
        g_ids = pd.unique(p_all[g_mask])
        d = o_dur[mask][g_mask]
        if len(g_ids) == 0:
            g = None
            two_g = False
        elif len(g_ids) == 1:
            g = int(g_ids[0])
            two_g = False
        else:
            # resolve to the goalie with the longest overlapping shift (enrichment rule)
            gp = p_all[g_mask]
            g = int(gp[np.argmax(d)])
            two_g = True
        return frozenset(int(x) for x in sk), g, two_g

    h_sk, h_g, h2 = side(is_home_side)
    a_sk, a_g, a2 = side(~is_home_side)
    return _make_interval(h_sk, a_sk, h_g, a_g, s, e,
                          two_goalies=(h2 or a2))


def _make_interval(h_sk, a_sk, h_g, a_g, s, e, empty_side=False, two_goalies=False):
    return {
        "h_sk": h_sk, "a_sk": a_sk, "h_g": h_g, "a_g": a_g,
        "start_secs": s, "end_secs": e,
        "empty_side": empty_side, "two_goalies": two_goalies,
    }


def _merge_adjacent(elem):
    if not elem:
        return []
    out = [dict(elem[0])]
    for cur in elem[1:]:
        prev = out[-1]
        same = (cur["h_sk"] == prev["h_sk"] and cur["a_sk"] == prev["a_sk"]
                and cur["h_g"] == prev["h_g"] and cur["a_g"] == prev["a_g"]
                and cur["end_secs"] > prev["start_secs"])  # contiguous
        if same and cur["start_secs"] == prev["end_secs"]:
            prev["end_secs"] = cur["end_secs"]
            prev["empty_side"] = prev["empty_side"] or cur["empty_side"]
            prev["two_goalies"] = prev["two_goalies"] or cur["two_goalies"]
        else:
            out.append(dict(cur))
    return out


# ----------------------------------------------------------------------------
# Shot aggregation + score differential
# ----------------------------------------------------------------------------
def attach_shot_stats(stints: pd.DataFrame, shots: pd.DataFrame, home_team: dict):
    """Assign each non-shootout shot to its stint and sum per-team stats.

    Shot -> stint rule: start_secs < t <= end_secs (identical to enrichment).
    Per team (home/away, by shooting_team_id vs game home team): unblocked shot
    attempts, goals, xG (xg_v2; 0 for the 21 impossible-strength rows).
    """
    n = len(stints)
    hs_att = np.zeros(n, np.int32); as_att = np.zeros(n, np.int32)
    hs_g = np.zeros(n, np.int32);   as_g = np.zeros(n, np.int32)
    hs_xg = np.zeros(n, np.float64); as_xg = np.zeros(n, np.float64)

    # index of stints by (game, period), each sorted by start
    stints = stints.reset_index(drop=True)
    stints["_sid"] = np.arange(n)
    grp_idx = {k: v.sort_values("start_secs")
               for k, v in stints.groupby(["game_id", "period"])}

    unmatched_shots = 0
    imp_records = []   # where the 21 impossible-strength (no xg) shots land
    shot_sid = np.full(len(shots), -1, dtype=np.int64)

    shots = shots.reset_index(drop=True)
    for (g, p), sg in shots.groupby(["game_id", "period"]):
        st = grp_idx.get((int(g), int(p)))
        if st is None:
            unmatched_shots += len(sg)
            continue
        ends = st.end_secs.to_numpy()
        starts = st.start_secs.to_numpy()
        sids = st._sid.to_numpy()
        t = sg.abs_secs.to_numpy()
        pos = np.searchsorted(ends, t, side="left")   # first end >= t
        for local, (ridx, srow) in enumerate(sg.iterrows()):
            k = pos[local]
            if k >= len(ends) or not (starts[k] < t[local] <= ends[k]):
                unmatched_shots += 1
                continue
            sid = int(sids[k])
            shot_sid[ridx] = sid
            is_home_shot = (srow.shooting_team_id == home_team.get(int(g)))
            goal = int(srow.is_goal)
            xgv = float(srow.xg_v2_filled)
            if is_home_shot:
                hs_att[sid] += 1; hs_g[sid] += goal; hs_xg[sid] += xgv
            else:
                as_att[sid] += 1; as_g[sid] += goal; as_xg[sid] += xgv
            if not srow.has_xg:
                imp_records.append({
                    "game_id": int(g), "period": int(p), "sid": sid,
                    "abs_secs": int(srow.abs_secs),
                    "enriched_strength": srow.strength_state,
                    "is_goal": goal, "is_home_shot": bool(is_home_shot),
                })

    stints["home_attempts"] = hs_att
    stints["away_attempts"] = as_att
    stints["home_goals"] = hs_g
    stints["away_goals"] = as_g
    stints["home_xg"] = hs_xg
    stints["away_xg"] = as_xg
    return stints, unmatched_shots, pd.DataFrame(imp_records), shot_sid


def attach_score_diff(stints: pd.DataFrame, shots: pd.DataFrame, home_team: dict):
    """Score differential (home perspective) at the START of each stint.

    Goals with abs_secs <= start_secs have already occurred in prior stints.
    """
    goals = shots[shots.is_goal == 1][["game_id", "abs_secs", "shooting_team_id"]].copy()
    home_g = {}; away_g = {}
    for g, gg in goals.groupby("game_id"):
        ht = home_team.get(int(g))
        hs = np.sort(gg.loc[gg.shooting_team_id == ht, "abs_secs"].to_numpy())
        as_ = np.sort(gg.loc[gg.shooting_team_id != ht, "abs_secs"].to_numpy())
        home_g[int(g)] = hs
        away_g[int(g)] = as_

    diff = np.zeros(len(stints), dtype=np.int32)
    for i, row in enumerate(stints.itertuples(index=False)):
        g = int(row.game_id); s = int(row.start_secs)
        hg = home_g.get(g, np.empty(0)); ag = away_g.get(g, np.empty(0))
        diff[i] = np.searchsorted(hg, s, side="right") - np.searchsorted(ag, s, side="right")
    stints["score_diff_home"] = diff
    return stints


# ----------------------------------------------------------------------------
# Assemble output
# ----------------------------------------------------------------------------
def main():
    shifts = load_shifts()
    goalie_set = bse.derive_goalie_set(shifts)

    shots = load_shots_with_xg()
    # game -> home team id (from enriched is_home; verified consistent)
    ht = shots.assign(home=np.where(shots.is_home == 1,
                                    shots.shooting_team_id,
                                    shots.defending_team_id))
    home_team = ht.groupby("game_id")["home"].first().astype(int).to_dict()

    records, self_overlap_rows = build_stints(shifts, goalie_set, home_team)
    stints = pd.DataFrame(records)

    # numeric skater counts + goalie fields
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

    stints, unmatched_shots, imp_df, shot_sid = attach_shot_stats(
        stints, shots, home_team)
    stints = attach_score_diff(stints, shots, home_team)

    cols = ["game_id", "period", "start_secs", "end_secs", "duration_secs",
            "home_skaters", "away_skaters", "home_skater_ids", "away_skater_ids",
            "home_goalie_id", "away_goalie_id", "home_empty_net", "away_empty_net",
            "score_diff_home",
            "home_attempts", "away_attempts", "home_goals", "away_goals",
            "home_xg", "away_xg",
            "flag_two_goalies", "flag_empty_side"]
    out = stints[cols].sort_values(["game_id", "period", "start_secs"]).reset_index(drop=True)
    out.to_parquet(OUT, index=False)

    # persist artifacts the report needs (in-memory hand-off if run together)
    meta = {
        "self_overlap_rows": self_overlap_rows,
        "unmatched_shots": unmatched_shots,
        "imp_df": imp_df,
        "shot_sid": shot_sid,
        "home_team": home_team,
        "goalie_set": goalie_set,
    }
    print("STINTS", len(out))
    print("SELF_OVERLAP_ROWS", self_overlap_rows)
    print("UNMATCHED_SHOTS", unmatched_shots)
    print("IMPOSSIBLE_STRENGTH_SHOTS", len(imp_df))
    return out, meta, shots, shifts


if __name__ == "__main__":
    main()
