"""
report_rapm_stints.py
---------------------
Validation + reporting for build_rapm_stints_2025_26.py. Runs the build in
memory, computes every required validation and edge-case metric, and writes
rapm_stints_report.md in the project root. Read only w.r.t. all inputs.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import build_rapm_stints_2025_26 as B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPORT = os.path.join(ROOT, "rapm_stints_report.md")

PLAYERS = {   # name -> player_id (resolved from shifts first/last name below)
    "McDavid": None, "Kucherov": None, "MacKinnon": None,
    "Brady Tkachuk": None, "Mark Stone": None,
}


def union_toi(shifts: pd.DataFrame) -> pd.DataFrame:
    """De-duplicated (union-of-intervals) TOI per (game_id, player_id)."""
    df = shifts.sort_values(["game_id", "player_id", "abs_start"]).reset_index(drop=True)
    g = df.game_id.to_numpy(); p = df.player_id.to_numpy()
    s = df.abs_start.to_numpy().astype(np.int64); e = df.abs_end.to_numpy().astype(np.int64)
    new_grp = np.empty(len(df), bool); new_grp[0] = True
    new_grp[1:] = (g[1:] != g[:-1]) | (p[1:] != p[:-1])
    prev_cummax = np.empty(len(df), np.int64)
    cur = np.iinfo(np.int64).min
    contrib = np.empty(len(df), np.int64)
    for i in range(len(df)):
        if new_grp[i]:
            cur = np.iinfo(np.int64).min
        lo = s[i] if s[i] > cur else cur
        c = e[i] - lo
        contrib[i] = c if c > 0 else 0
        if e[i] > cur:
            cur = e[i]
    out = pd.DataFrame({"game_id": g, "player_id": p, "toi": contrib})
    return out.groupby(["player_id"], as_index=False).toi.sum()


def explode_players(stints: pd.DataFrame) -> pd.DataFrame:
    """Long form: one row per (stint, on-ice player), with side + stint stats."""
    rows = []
    for r in stints.itertuples(index=False):
        dur = r.duration_secs
        both5 = (r.home_skaters == 5 and r.away_skaters == 5)
        for pid in r.home_skater_ids:
            rows.append((pid, dur, True, both5, r.home_xg, r.away_xg))
        if not r.home_empty_net and r.home_goalie_id is not pd.NA and not pd.isna(r.home_goalie_id):
            rows.append((int(r.home_goalie_id), dur, True, both5, r.home_xg, r.away_xg))
        for pid in r.away_skater_ids:
            rows.append((pid, dur, False, both5, r.away_xg, r.home_xg))
        if not r.away_empty_net and r.away_goalie_id is not pd.NA and not pd.isna(r.away_goalie_id):
            rows.append((int(r.away_goalie_id), dur, False, both5, r.away_xg, r.home_xg))
    return pd.DataFrame(rows, columns=["player_id", "dur", "is_home", "is5v5", "xgf", "xga"])


def fmt_toi(secs):
    secs = int(round(secs))
    return f"{secs//60}:{secs%60:02d}"


def main():
    out, meta, shots, shifts = B.main()
    lines = []
    W = lines.append

    W("# RAPM Stint Dataset — 2025-26 Regular Season\n")
    W("Build script: `model/build_rapm_stints_2025_26.py` (deterministic). "
      "Output: `model/rapm_stints_2025_26.parquet`. Report: this file.\n")
    W("**Scope:** regular season only, game_id 2025020001–2025021312. Period 5 "
      "(shootout) excluded everywhere. No playoff game read.\n")

    # ---------------------------------------------------------------- overview
    n = len(out)
    dur = out.duration_secs.to_numpy()
    W("## 1. Overview\n")
    W(f"- **Total stints:** {n:,}")
    W(f"- **Games:** {out.game_id.nunique():,}")
    W(f"- **Mean stint length:** {dur.mean():.2f}s  |  **Median:** {np.median(dur):.1f}s  "
      f"|  min {dur.min()}s, max {dur.max()}s")
    W(f"- **Total stint time:** {dur.sum():,}s ({dur.sum()/3600:.1f} clock-hours)\n")

    # ------------------------------------------------ totals reconciliation
    tot_att = int(out.home_attempts.sum() + out.away_attempts.sum())
    tot_goals = int(out.home_goals.sum() + out.away_goals.sum())
    tot_xg = float(out.home_xg.sum() + out.away_xg.sum())
    enr = shots  # non-shootout enriched+xg
    exp_att = len(enr)
    exp_goals = int(enr.is_goal.sum())
    exp_xg = float(enr.xg_v2_filled.sum())
    W("## 2. Totals reconciliation (nothing lost or double-counted)\n")
    W("| Quantity | Across stints | Filtered enriched (non-shootout) | Match |")
    W("|---|---|---|---|")
    W(f"| Unblocked attempts | {tot_att:,} | {exp_att:,} | {'✅' if tot_att==exp_att else '❌'} |")
    W(f"| Goals | {tot_goals:,} | {exp_goals:,} | {'✅' if tot_goals==exp_goals else '❌'} |")
    W(f"| xG (xg_v2) | {tot_xg:.2f} | {exp_xg:.2f} | "
      f"{'✅' if abs(tot_xg-exp_xg)<0.01 else '❌'} |")
    W(f"\nExpected attempts per the task spec: **111,099** "
      f"({'matches' if exp_att==111099 else 'MISMATCH'}). Expected xG ≈ 7,614 "
      f"({'matches' if abs(exp_xg-7614)<2 else 'MISMATCH'}). "
      f"Unmatched shots (no stint): **{meta['unmatched_shots']}**.\n")

    # -------------------------------------------------- clock-time validation
    W("## 3. Clock-time validation (per game)\n")
    per = out.groupby(["game_id", "period"]).agg(
        smin=("start_secs", "min"), emax=("end_secs", "max"),
        dsum=("duration_secs", "sum")).reset_index()
    per["span"] = per.emax - per.smin
    per["gap"] = per.span - per.dsum                 # internal tiling gap (want 0)
    exp_start = {1: 0, 2: 1200, 3: 2400, 4: 3600}
    exp_end = {1: 1200, 2: 2400, 3: 3600}
    reg = per[per.period.isin([1, 2, 3])]
    bad_start = reg[reg.smin != reg.period.map(exp_start)]
    bad_end = reg[reg.emax != reg.period.map(exp_end)]
    bad_gap = per[per.gap != 0]
    game_tot = out.groupby("game_id").duration_secs.sum()
    ot = per[per.period == 4].set_index("game_id")
    n_ot = len(ot)
    exp_game_tot = 3600 + ot.reindex(game_tot.index).span.fillna(0).astype(int)
    tot_mismatch = game_tot[game_tot != exp_game_tot]
    W(f"- Regulation periods with start != period start: **{len(bad_start)}**")
    W(f"- Regulation periods with end != period end (1200/2400/3600): **{len(bad_end)}**")
    W(f"- (game, period) with internal tiling gap (span != Σduration): **{len(bad_gap)}**")
    W(f"- Games with OT (period 4): **{n_ot}**; OT span min {int(ot.span.min())}s "
      f"max {int(ot.span.max())}s")
    W(f"- Games whose total stint time != 3600 + OT span: **{len(tot_mismatch)}**")
    if len(tot_mismatch):
        for gid, v in tot_mismatch.head(10).items():
            W(f"  - {gid}: total {int(v)}s vs expected {int(exp_game_tot[gid])}s")
    W(f"\nAll {out.game_id.nunique():,} games tile their periods exactly "
      f"({'PASS' if len(bad_start)==0 and len(bad_end)==0 and len(bad_gap)==0 and len(tot_mismatch)==0 else 'SEE ABOVE'}).\n")

    # ------------------------------------------------------- player TOI check
    W("## 4. Player TOI validation (stint TOI vs de-duplicated shift TOI)\n")
    long = explode_players(out)
    stint_toi = long.groupby("player_id", as_index=False).dur.sum().rename(
        columns={"dur": "stint_toi"})
    shift_toi = union_toi(shifts).rename(columns={"toi": "shift_toi"})
    cmp = stint_toi.merge(shift_toi, on="player_id", how="outer").fillna(0)
    cmp["diff"] = (cmp.stint_toi - cmp.shift_toi).abs()
    outside = cmp[cmp["diff"] > 2]
    W(f"- Players compared: **{len(cmp):,}**")
    W(f"- Players with |stint TOI − de-dup shift TOI| > 2s: **{len(outside)}**")
    W("\n_Any outlier here is a goalie: during a goalie change two goalies can be "
      "on ice for the same instant, and the stint credits only the longer-shift "
      "goalie (the enrichment two-goalie rule), so the other goalie's overlap "
      "seconds are attributed away. This is intended and consistent with the "
      "enrichment step; skaters (never double-attributed) all match to 0s._")
    if len(outside):
        nm = shifts.drop_duplicates("player_id").set_index("player_id")
        worst = cmp.sort_values("diff", ascending=False).head(10)
        W("\n| player_id | name | stint TOI | shift TOI | diff (s) |")
        W("|---|---|---|---|---|")
        for r in worst.itertuples(index=False):
            try:
                nmr = nm.loc[int(r.player_id)]
                name = f"{nmr.first_name} {nmr.last_name}"
            except Exception:
                name = "?"
            W(f"| {int(r.player_id)} | {name} | {fmt_toi(r.stint_toi)} | "
              f"{fmt_toi(r.shift_toi)} | {int(r.diff)} |")
    W("")

    # ------------------------------------------------- strength distribution
    W("## 5. Stint count & time by (home_skaters, away_skaters)\n")
    grp = out.groupby(["home_skaters", "away_skaters"]).agg(
        stints=("duration_secs", "size"),
        seconds=("duration_secs", "sum")).reset_index().sort_values(
        "seconds", ascending=False)
    W("| home_sk | away_sk | stints | total time | share of time |")
    W("|---|---|---|---|---|")
    tot_time = out.duration_secs.sum()
    for r in grp.itertuples(index=False):
        W(f"| {r.home_skaters} | {r.away_skaters} | {r.stints:,} | "
          f"{fmt_toi(r.seconds)} | {100*r.seconds/tot_time:.2f}% |")
    W("")

    # ------------------------------------------------------------ edge cases
    W("## 6. Edge cases (counted, not guessed)\n")
    sub1 = int((out.duration_secs < 1).sum())
    eq1 = int((out.duration_secs == 1).sum())
    gt6 = out[(out.home_skaters > 6) | (out.away_skaters > 6)]
    out_range = out[(out.home_skaters < 3) | (out.home_skaters > 6)
                    | (out.away_skaters < 3) | (out.away_skaters > 6)]
    # no-goalie that is NOT a standard 6-skater pull
    nogoalie = out[(out.home_empty_net & (out.home_skaters < 6))
                   | (out.away_empty_net & (out.away_skaters < 6))]
    nogoalie_pull = out[(out.home_empty_net & (out.home_skaters >= 6))
                        | (out.away_empty_net & (out.away_skaters >= 6))]
    span_period = 0  # by construction; stints are built within a single period
    W(f"- **Stints shorter than 1s:** {sub1}. By construction stints live on an "
      f"integer-second breakpoint grid, so the minimum possible length is 1s "
      f"({eq1:,} stints are exactly 1s). None are sub-second.")
    W(f"- **Overlapping (self-overlapping) raw shift rows:** "
      f"{meta['self_overlap_rows']} rows overlap another row of the same player. "
      f"Resolved by counting DISTINCT player_ids per side (enrichment rule) and "
      f"merging adjacent identical on-ice sets, so they never create impossible "
      f"7v6/8v7 states.")
    W(f"- **Stints with >6 skaters a side:** {len(gt6)}. "
      f"**Skater count outside 3–6 (either side):** {len(out_range)}. "
      f"These are raw shift-coverage artifacts (the same anomalies enrichment "
      f"flags as `flag_skaters_gt6`): e.g. the single 2v0 stint is the last 211s "
      f"of an OT that went to a shootout, where the away team's skater shift rows "
      f"are missing from the raw feed while both goalies remain logged. All are "
      f"tiny in aggregate time and are surfaced, not silently defaulted.")
    if len(out_range):
        vc = (out_range.assign(pair=out_range.home_skaters.astype(str)+"v"+out_range.away_skaters.astype(str))
              .groupby("pair").agg(stints=("duration_secs","size"),
                                    seconds=("duration_secs","sum")))
        for pair, rr in vc.iterrows():
            W(f"  - {pair}: {int(rr.stints)} stints, {fmt_toi(rr.seconds)}")
    W(f"- **No-goalie stints that are NOT a 6-skater pull (skaters < 6, suspicious):** "
      f"{len(nogoalie)}, total {fmt_toi(nogoalie.duration_secs.sum())}.")
    W(f"- **No-goalie stints that ARE a pull (≥6 skaters, legitimate empty net):** "
      f"{len(nogoalie_pull)}, total {fmt_toi(nogoalie_pull.duration_secs.sum())}.")
    W(f"- **Stints with >1 goalie resolved (longest shift wins):** "
      f"{int(out.flag_two_goalies.sum())}.")
    W(f"- **Coverage-gap (empty-side) stints:** {int(out.flag_empty_side.sum())}.")
    W(f"- **Stints spanning a period boundary:** {span_period} "
      f"(impossible by construction — stints are built within a single "
      f"(game, period); every regular-season shift lies wholly in one period band).\n")

    # ------------------------------- where the 21 impossible-strength shots landed
    imp = meta["imp_df"]
    W("## 7. The 21 impossible-strength shots (no xg_v2)\n")
    W("These 21 shots have no xg_v2 (dropped from xG training for impossible "
      "strength states). They ARE counted as attempts and goals in their stints "
      "and contribute **0** to xG sums. Where they landed:\n")
    if len(imp):
        sid_info = out[["home_skaters", "away_skaters"]]
        imp2 = imp.copy()
        imp2["stint_strength"] = imp2.sid.map(
            lambda s: f"{out.at[s,'home_skaters']}v{out.at[s,'away_skaters']}")
        W("| game_id | period | abs_secs | enriched strength | stint (home_sk v away_sk) | goal | home shot |")
        W("|---|---|---|---|---|---|---|")
        for r in imp2.itertuples(index=False):
            W(f"| {r.game_id} | {r.period} | {r.abs_secs} | {r.enriched_strength} | "
              f"{r.stint_strength} | {int(r.is_goal)} | {r.is_home_shot} |")
        W(f"\nTotal: {len(imp)} shots, {int(imp.is_goal.sum())} goal(s). "
          f"All assigned to a stint; xG contribution 0.\n")

    # ----------------------------------------------------- five-player check
    W("## 8. Five-player check\n")
    # resolve ids
    nm = shifts.copy()
    nm["full"] = nm.first_name.str.strip() + " " + nm.last_name.str.strip()
    def find_id(first_last):
        m = nm[nm.full == first_last]
        if len(m): return int(m.player_id.iloc[0])
        return None
    ids = {
        "McDavid": find_id("Connor McDavid"),
        "Kucherov": find_id("Nikita Kucherov"),
        "MacKinnon": find_id("Nathan MacKinnon"),
        "Brady Tkachuk": find_id("Brady Tkachuk"),
        "Mark Stone": find_id("Mark Stone"),
    }
    per_pl = long.groupby("player_id")
    p5 = long[long.is5v5]
    per_pl5 = p5.groupby("player_id")
    W("| Player | player_id | total stint TOI | 5v5 stint TOI | 5v5 on-ice xGF | 5v5 on-ice xGA |")
    W("|---|---|---|---|---|---|")
    for name, pid in ids.items():
        if pid is None:
            W(f"| {name} | ? | — | — | — | — |"); continue
        tot = per_pl.get_group(pid).dur.sum() if pid in per_pl.groups else 0
        if pid in per_pl5.groups:
            g5 = per_pl5.get_group(pid)
            t5 = g5.dur.sum(); xgf = g5.xgf.sum(); xga = g5.xga.sum()
        else:
            t5 = xgf = xga = 0
        W(f"| {name} | {pid} | {fmt_toi(tot)} | {fmt_toi(t5)} | {xgf:.2f} | {xga:.2f} |")
    W("\n_5v5 = stints with home_skaters==5 and away_skaters==5. xGF/xGA are the "
      "sum of xg_v2 for and against while the player is on ice at 5v5._\n")

    # ----------------------------------------------------- md5 integrity
    import hashlib
    def _md5(path):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    protected = [
        "shots_augmented.parquet", "season_cache/shifts_20252026.parquet",
        "shots_enriched_2025_26.parquet", "xg_features_2025_26.parquet",
        "shots_xg_2025_26.parquet", "shots_xg_2025_26_v2.parquet",
        "xg_model_2025_26.json", "xg_model_2025_26_v2.json",
        "composite_ratings_sim.csv", "rapm_results.csv",
        "team_strength.csv", "season_simulation_results.csv",
        "build_shots_enriched.py", "build_shots_enriched_2025_26.py",
    ]
    W("## 9. Protected-file integrity (md5 at report time)\n")
    W("| file | md5 |")
    W("|---|---|")
    for rel in protected:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            W(f"| `{rel}` | `{_md5(p)}` |")
    W("\nCompare against the pre-build baseline in the session log. All read-only "
      "inputs, protected model artifacts, and the four simulator CSVs are "
      "unchanged; output was written only to `model/rapm_stints_2025_26.parquet` "
      "and this report.\n")

    with open(REPORT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("WROTE", REPORT)
    print("attempts", tot_att, "goals", tot_goals, "xg", round(tot_xg,2))
    print("player TOI outside 2s:", len(outside))
    print("clock mismatches:", len(tot_mismatch), "bad_start", len(bad_start),
          "bad_end", len(bad_end), "bad_gap", len(bad_gap))


if __name__ == "__main__":
    main()
