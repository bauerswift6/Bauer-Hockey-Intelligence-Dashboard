"""
validate_rapm_stints.py --season <s>
------------------------------------
Runs the same validation suite as the 2025-26 stint report against any season's
model/rapm_stints_<season>.parquet: clock-time tiling per game, player stint TOI
vs de-duplicated shift TOI, shot/goal/xG reconciliation, and edge-case counts.

Gate (per the RAPM build spec): STOP if >5 games fail clock tiling OR >1% of
players fall outside the 2s TOI tolerance.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

import build_rapm_stints as B
from report_rapm_stints import union_toi, explode_players, fmt_toi


def validate(season: str):
    cfg = B.SEASONS[season]
    import os
    out = pd.read_parquet(os.path.join(B.HERE, cfg["out"]))
    shifts = B.load_shifts(cfg)
    shots = B.load_shots_with_xg(cfg)

    print(f"=== VALIDATION {season} : {len(out):,} stints, {out.game_id.nunique()} games ===")

    # 1. totals reconciliation
    tot_att = int(out.home_attempts.sum() + out.away_attempts.sum())
    tot_goals = int(out.home_goals.sum() + out.away_goals.sum())
    tot_xg = float(out.home_xg.sum() + out.away_xg.sum())
    exp_att, exp_goals, exp_xg = len(shots), int(shots.is_goal.sum()), float(shots.xg_v2_filled.sum())
    print(f"attempts {tot_att} vs {exp_att}  {'OK' if tot_att==exp_att else 'FAIL'}")
    print(f"goals    {tot_goals} vs {exp_goals}  {'OK' if tot_goals==exp_goals else 'FAIL'}")
    print(f"xG       {tot_xg:.2f} vs {exp_xg:.2f}  {'OK' if abs(tot_xg-exp_xg)<0.01 else 'FAIL'}")

    # 2. clock tiling
    per = out.groupby(["game_id", "period"]).agg(
        smin=("start_secs", "min"), emax=("end_secs", "max"),
        dsum=("duration_secs", "sum")).reset_index()
    per["span"] = per.emax - per.smin
    per["gap"] = per.span - per.dsum
    exp_start = {1: 0, 2: 1200, 3: 2400, 4: 3600}
    exp_end = {1: 1200, 2: 2400, 3: 3600}
    reg = per[per.period.isin([1, 2, 3])]
    bad_start = reg[reg.smin != reg.period.map(exp_start)]
    bad_end = reg[reg.emax != reg.period.map(exp_end)]
    bad_gap = per[per.gap != 0]
    bad_games = set(bad_start.game_id) | set(bad_end.game_id) | set(bad_gap.game_id)
    n_ot = int((per.period == 4).sum())
    print(f"clock: bad_start={len(bad_start)} bad_end={len(bad_end)} bad_gap={len(bad_gap)} "
          f"=> games failing tiling={len(bad_games)} | OT games={n_ot}")

    # 3. player TOI
    long = explode_players(out)
    stint_toi = long.groupby("player_id", as_index=False).dur.sum().rename(columns={"dur": "stint_toi"})
    shift_toi = union_toi(shifts).rename(columns={"toi": "shift_toi"})
    cmp = stint_toi.merge(shift_toi, on="player_id", how="outer").fillna(0)
    cmp["diff"] = (cmp.stint_toi - cmp.shift_toi).abs()
    outside = cmp[cmp["diff"] > 2]
    pct_out = 100 * len(outside) / len(cmp)
    print(f"player TOI: {len(cmp)} players, outside 2s tolerance={len(outside)} ({pct_out:.2f}%)")
    if len(outside):
        nm = shifts.drop_duplicates("player_id").set_index("player_id")
        for r in cmp.sort_values("diff", ascending=False).head(10).itertuples(index=False):
            try:
                x = nm.loc[int(r.player_id)]; name = f"{x.first_name} {x.last_name}"
            except Exception:
                name = "?"
            print(f"   {int(r.player_id)} {name}: stint {fmt_toi(r.stint_toi)} shift {fmt_toi(r.shift_toi)} diff {int(r.diff)}s")

    # 4. edge cases
    gt6 = out[(out.home_skaters > 6) | (out.away_skaters > 6)]
    out_range = out[(out.home_skaters < 3) | (out.home_skaters > 6) | (out.away_skaters < 3) | (out.away_skaters > 6)]
    sub1 = int((out.duration_secs < 1).sum())
    print(f"edge: sub-1s={sub1} | >6-a-side={len(gt6)} | skaters outside 3-6={len(out_range)} | "
          f"two-goalie stints={int(out.flag_two_goalies.sum())} | empty-side={int(out.flag_empty_side.sum())}")

    # 5v5 summary
    v5 = out[(out.home_skaters == 5) & (out.away_skaters == 5)
             & out.home_goalie_id.notna() & out.away_goalie_id.notna()]
    print(f"5v5 (both goalies): {len(v5):,} stints, {fmt_toi(v5.duration_secs.sum())} total")

    # gate
    ok = (tot_att == exp_att and tot_goals == exp_goals and abs(tot_xg - exp_xg) < 0.01
          and len(bad_games) <= 5 and pct_out <= 1.0)
    print(f"\nGATE: {'PASS' if ok else 'FAIL'} (games failing tiling {len(bad_games)}<=5, "
          f"players outside TOI {pct_out:.2f}%<=1%)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    a = ap.parse_args()
    validate(a.season)
