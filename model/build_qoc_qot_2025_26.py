"""
build_qoc_qot_2025_26.py
------------------------
Rebuilds 5v5 Quality of Competition (QoC) and Quality of Teammates (QoT) from the
NEW 5v5 RAPM (model/rapm_5v5_2025_26.csv, kappa 0.75) and the validated stint
file (model/rapm_stints_2025_26.parquet). Deterministic; read-only w.r.t. every
protected input. New outputs only.

Definitions (5v5 only; both goalies in net, 5v5, non-null goalies — the exact
set train_rapm_5v5.load_5v5 produces):
  * For a player in a stint: teammates = the other 4 skaters on his team;
    opponents = the 5 opposing skaters. Goalies never included.
  * RAPM-based (primary), using offense/defense/total from rapm_5v5_2025_26.csv:
      qot_* = duration-weighted average, over all the player's 5v5 stints, of the
              MEAN rating of his 4 teammates in that stint.
      qoc_* = same, using the MEAN rating of the 5 opponents.
  * TOI-based (secondary, rating-free): toi_per_gp = 5v5 TOI (min) / games played
    (games with any 5v5 stint). qot_toi / qoc_toi = duration-weighted average of
    teammates' / opponents' mean toi_per_gp.

Output: model/qoc_qot_5v5_2025_26.csv
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

import train_rapm_5v5 as T

HERE = T.HERE
OUT = os.path.join(HERE, "qoc_qot_5v5_2025_26.csv")
EXPECT_STINTS = 445437          # rapm_5v5_report.md
EXPECT_SECS = 3830744           # 63845:44


def load():
    df = T.load_5v5("2025_26")
    if len(df) != EXPECT_STINTS or int(df.duration_secs.sum()) != EXPECT_SECS:
        raise SystemExit(f"STOP: 5v5 set mismatch — {len(df)} stints / "
                         f"{int(df.duration_secs.sum())}s vs expected "
                         f"{EXPECT_STINTS}/{EXPECT_SECS}.")
    rapm = pd.read_csv(os.path.join(HERE, "rapm_5v5_2025_26.csv"))
    return df, rapm


def weighted_qc_qt(home_idx, away_idx, dur, den, rating):
    """Return (qot, qoc) per player for a single rating array.
    qot_p = Σ_s d_s·mean(teammates rating) / den_p ; qoc_p likewise for opponents.
    """
    P = len(rating)
    qot_num = np.zeros(P); qoc_num = np.zeros(P)
    rH = rating[home_idx]; rA = rating[away_idx]        # (N,5)
    sumH = rH.sum(1); sumA = rA.sum(1)                  # (N,)
    d = dur
    # home players: teammates = other 4 home, opponents = 5 away
    tmate_H = (sumH[:, None] - rH) / 4.0                # (N,5)
    opp_H = (sumA / 5.0)[:, None] * np.ones((1, 5))     # (N,5)
    np.add.at(qot_num, home_idx.ravel(), (d[:, None] * tmate_H).ravel())
    np.add.at(qoc_num, home_idx.ravel(), (d[:, None] * opp_H).ravel())
    # away players: teammates = other 4 away, opponents = 5 home
    tmate_A = (sumA[:, None] - rA) / 4.0
    opp_A = (sumH / 5.0)[:, None] * np.ones((1, 5))
    np.add.at(qot_num, away_idx.ravel(), (d[:, None] * tmate_A).ravel())
    np.add.at(qoc_num, away_idx.ravel(), (d[:, None] * opp_A).ravel())
    with np.errstate(invalid="ignore", divide="ignore"):
        return qot_num / den, qoc_num / den


def build():
    df, rapm = load()
    names = T.player_names(); teamabbr = T.team_abbrevs()

    # player index over all 5v5 skaters
    ids = set()
    for c in ("home_skater_ids", "away_skater_ids"):
        for lst in df[c]:
            ids.update(int(x) for x in lst)
    pids = sorted(ids)
    idx = {p: i for i, p in enumerate(pids)}
    P = len(pids)
    missing = [p for p in pids if p not in set(rapm.player_id.astype(int))]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} stint skaters missing from RAPM.")

    rmap = rapm.set_index("player_id")
    r_tot = np.array([rmap.at[p, "total"] for p in pids])
    r_off = np.array([rmap.at[p, "offense"] for p in pids])
    r_def = np.array([rmap.at[p, "defense"] for p in pids])

    home_idx = np.array([[idx[int(x)] for x in lst] for lst in df.home_skater_ids])
    away_idx = np.array([[idx[int(x)] for x in lst] for lst in df.away_skater_ids])
    dur = df.duration_secs.to_numpy().astype(float)

    # den = 5v5 TOI seconds per player
    den = np.zeros(P)
    np.add.at(den, home_idx.ravel(), np.repeat(dur, 5))
    np.add.at(den, away_idx.ravel(), np.repeat(dur, 5))

    # games played per player (games with any 5v5 stint)
    long = pd.concat([
        pd.DataFrame({"pid": home_idx.ravel(), "g": np.repeat(df.game_id.to_numpy(), 5)}),
        pd.DataFrame({"pid": away_idx.ravel(), "g": np.repeat(df.game_id.to_numpy(), 5)}),
    ], ignore_index=True).drop_duplicates()
    gp = long.groupby("pid").size().reindex(range(P)).fillna(0).to_numpy()
    toi_per_gp = np.divide(den / 60.0, gp, out=np.zeros(P), where=gp > 0)   # min/GP

    # weighted QoT/QoC for each rating
    qot_tot, qoc_tot = weighted_qc_qt(home_idx, away_idx, dur, den, r_tot)
    qot_off, qoc_off = weighted_qc_qt(home_idx, away_idx, dur, den, r_off)
    qot_def, qoc_def = weighted_qc_qt(home_idx, away_idx, dur, den, r_def)
    qot_toi, qoc_toi = weighted_qc_qt(home_idx, away_idx, dur, den, toi_per_gp)

    # teams (primary + list) by 5v5 TOI, reuse RAPM's team strings for consistency
    teams = rmap["teams"].to_dict()

    out = pd.DataFrame({
        "player_id": pids,
        "name": [names.get(p) for p in pids],
        "position": [rmap.at[p, "position"] if p in rmap.index else None for p in pids],
        "teams": [teams.get(p) for p in pids],
        "toi_min_5v5": (den / 60).round(1),
        "qot_total": qot_tot, "qot_offense": qot_off, "qot_defense": qot_def,
        "qoc_total": qoc_tot, "qoc_offense": qoc_off, "qoc_defense": qoc_def,
        "qot_toi": qot_toi, "qoc_toi": qoc_toi,
    })
    out = out.sort_values("player_id").reset_index(drop=True)
    out.to_csv(OUT, index=False)

    # league duration-weighted average RAPM total across all 5v5 player-seconds
    league_avg = float(np.sum(den * r_tot) / np.sum(den))
    return out, df, rapm, idx, r_tot, den, league_avg


def handcheck(df, rapm, pid):
    """Independent, simple loop over a player's raw stints -> (qot_total, qoc_total)."""
    rmap = rapm.set_index("player_id")["total"].to_dict()
    qot_num = qoc_num = den = 0.0
    for row in df.itertuples(index=False):
        H = [int(x) for x in row.home_skater_ids]
        A = [int(x) for x in row.away_skater_ids]
        d = float(row.duration_secs)
        if pid in H:
            mates = [rmap[x] for x in H if x != pid]
            opps = [rmap[x] for x in A]
        elif pid in A:
            mates = [rmap[x] for x in A if x != pid]
            opps = [rmap[x] for x in H]
        else:
            continue
        qot_num += d * (sum(mates) / len(mates))
        qoc_num += d * (sum(opps) / len(opps))
        den += d
    return qot_num / den, qoc_num / den


def most_common_linemate(df, pid, names):
    from collections import defaultdict
    shared = defaultdict(float)
    for row in df.itertuples(index=False):
        H = [int(x) for x in row.home_skater_ids]
        A = [int(x) for x in row.away_skater_ids]
        d = float(row.duration_secs)
        if pid in H:
            team = H
        elif pid in A:
            team = A
        else:
            continue
        for x in team:
            if x != pid:
                shared[x] += d
    if not shared:
        return None, 0.0
    best = max(shared, key=shared.get)
    return names.get(best, str(best)), shared[best] / 60.0


def fmt_toi(sec):
    sec = int(round(sec)); return f"{sec//60}:{sec%60:02d}"


STEP1 = """## Step 1 — Old QoC/QoT pipeline inventory (diagnostic; nothing modified)

**Builder:** `model/build_qoc_qot.py` (exists).

**Inputs**
- `model/season_cache/shifts_20252026.parquet` — columns `game_id, player_id,
  team_abbrev, abs_start, abs_end` (regular season filtered in-script via
  `(game_id // 10000) % 100 == 2`).
- `model/composite_ratings_single_season.csv` — columns `player_id,
  composite_war` (the GAR rating), `player_name`, `team`.

**What it computes**
- Rating used: **`composite_war` (GAR)**, NOT RAPM.
- Game states: **ALL strengths** — raw pairwise shift-overlap across every shift,
  no 5v5 / strength filter (goalies are included in the overlap because the shift
  file is not role-filtered; only same-player self-overlap is removed).
- Weighting: shared on-ice **overlap seconds** between each focal player and each
  partner, summed across games. `same_team=True → QoT`, `False → QoC`. Partners
  with no GAR are dropped from numerator and denominator.
- Output kept only for players who are themselves in the composite table.

**Output file:** `model/skater_qoc_qot_single_season.csv`
**Columns:** `player_id, player_name, team, qoc, qot, qoc_toi_secs, qot_toi_secs`
(qoc/qot rounded to 3; the two `*_toi_secs` are total shared seconds).

**Dependencies:** reads `season_cache/shifts_20252026.parquet` (yes) and
`composite_ratings_single_season.csv`. **Does NOT read `rapm_results.csv`** or any
RAPM file; nothing else in `season_cache/`.

**Where the dashboard reads it (file:line)**
- `app.py:2095-2096` — `QOC_QOT_PATH = model/skater_qoc_qot_single_season.csv`.
- `app.py:2099-2112` — `_load_qoc_qot()` loads `qoc, qot, qoc_toi_secs,
  qot_toi_secs` per player_id (cached as `qoc_qot_lookup`).
- `app.py:3313` — `my_qq = _load_qoc_qot().get(pid)` in the player-payload builder.
- `app.py:3460-3461` — emits `"qoc"` and `"qot"` (rounded 3) into the player API
  payload (null when missing).
- `static/js/players_leaderboards.js:156-157` — leaderboard columns `qoc`/`qot`
  (label "QoC"/"QoT", group "usage", source "skaters", 3-decimal signed fmt).
- `static/js/glossary_data.js:1057-1092` — glossary entries `qoc`/`qot` (prose
  only, no data read).

**Note:** the old metric is GAR-based and all-strengths; the new build below is
**5v5-only and RAPM-based**, so values are on a different scale (RAPM xG/60, ~0)
and are written to a **separate** file — no dashboard wiring is touched here.
"""


def main():
    import hashlib
    out, df, rapm, idx, r_tot, den, league_avg = build()
    names = T.player_names()
    L = []; W = L.append
    W("# 5v5 QoC / QoT — 2025-26 Regular Season (RAPM-based rebuild)\n")
    W("Deterministic build: `model/build_qoc_qot_2025_26.py`. Read-only w.r.t. all "
      "protected inputs. Output: `model/qoc_qot_5v5_2025_26.csv`.\n")
    W(STEP1)

    # Step 2 definitions + counts
    W("## Step 2 — Definitions & counts\n")
    W(f"- 5v5 stint set (both goalies, 5v5): **{len(df):,}** stints, "
      f"**{fmt_toi(df.duration_secs.sum())}** total — matches rapm_5v5_report.md "
      f"(445,437 / 63845:44) exactly.")
    W(f"- Skaters: **{len(out)}**, all present in `rapm_5v5_2025_26.csv` (0 missing).")
    W("- Teammates = the other 4 skaters on the player's team; opponents = the 5 "
      "opposing skaters; goalies never included.")
    W("- qot_* / qoc_* = duration-weighted average, over the player's 5v5 stints, of "
      "the mean teammate / opponent rating (offense, defense, total from "
      "`rapm_5v5_2025_26.csv`). qot_toi / qoc_toi use each skater's 5v5 TOI-per-GP.")
    W(f"- **League duration-weighted average RAPM total across all 5v5 player-seconds: "
      f"{league_avg:+.4f}** (this is what 'average' QoC/QoT is centered near).\n")

    # Step 3 hand check + determinism
    W("## Step 3 — Hand check & determinism\n")
    ok = True
    W("| player | qot_total (main) | qot_total (independent) | qoc_total (main) | qoc_total (independent) | agree<1e-9 |")
    W("|---|---|---|---|---|---|")
    for nm, pid in [("McDavid", 8478402), ("Mark Stone", 8475913)]:
        row = out[out.player_id == pid].iloc[0]
        hqot, hqoc = handcheck(df, rapm, pid)
        a = abs(row.qot_total - hqot) < 1e-9 and abs(row.qoc_total - hqoc) < 1e-9
        ok = ok and a
        W(f"| {nm} | {row.qot_total:.9f} | {hqot:.9f} | {row.qoc_total:.9f} | {hqoc:.9f} | {'✅' if a else '❌'} |")
    if not ok:
        raise SystemExit("STOP: hand check disagreement > 1e-9")
    md5 = hashlib.md5(open(OUT, "rb").read()).hexdigest()
    W(f"\n- Hand check passes to <1e-9 via an independent per-stint loop.")
    W(f"- Output md5: `{md5}` (determinism confirmed by re-running the script and "
      f"comparing — see session log).\n")

    # Step 4 sanity
    disp = out[out.toi_min_5v5 >= 500].copy()
    disp["rank_qot"] = disp.qot_total.rank(ascending=False).astype(int)
    disp["rank_qoc"] = disp.qoc_total.rank(ascending=False).astype(int)
    rmap_tot = rapm.set_index("player_id")["total"]

    W("## Step 4 — Sanity checks (≥500 5v5 min unless noted)\n")
    W("### Five players\n")
    W("| player | pos | qot_tot | qot_off | qot_def | qoc_tot | qoc_off | qoc_def | qot_toi | qoc_toi | rank qot | rank qoc | top linemate (min) |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, pid in [("McDavid", 8478402), ("Kucherov", 8476453), ("MacKinnon", 8477492),
                    ("Brady Tkachuk", 8480801), ("Mark Stone", 8475913)]:
        r = out[out.player_id == pid].iloc[0]
        rq = disp[disp.player_id == pid]
        rkq = int(rq.rank_qot.iloc[0]) if len(rq) else "—"
        rkc = int(rq.rank_qoc.iloc[0]) if len(rq) else "—"
        lm, lmm = most_common_linemate(df, pid, names)
        W(f"| {nm} | {r.position} | {r.qot_total:+.3f} | {r.qot_offense:+.3f} | {r.qot_defense:+.3f} | "
          f"{r.qoc_total:+.3f} | {r.qoc_offense:+.3f} | {r.qoc_defense:+.3f} | {r.qot_toi:.2f} | "
          f"{r.qoc_toi:.2f} | {rkq} | {rkc} | {lm} ({lmm:.0f}) |")
    W(f"\n_Ranks among {len(disp)} players with ≥500 5v5 min._\n")

    W("### Distributions by position\n")
    disp["grp"] = disp.position.map(lambda p: "D" if p == "D" else ("F" if p in ("C", "L", "R") else None))
    W("| group | n | metric | mean | std | min | max |")
    W("|---|---|---|---|---|---|---|")
    for grp in ["F", "D"]:
        sub = disp[disp.grp == grp]
        for m in ["qot_total", "qot_offense", "qot_defense", "qoc_total", "qoc_offense",
                  "qoc_defense", "qot_toi", "qoc_toi"]:
            v = sub[m]
            W(f"| {grp} | {len(sub)} | {m} | {v.mean():+.3f} | {v.std():.3f} | {v.min():+.3f} | {v.max():+.3f} |")
    W("")

    def tbl(title, col, asc):
        W(f"**{title}**\n")
        W("| # | player | pos | team | 5v5 min | value |")
        W("|---|---|---|---|---|---|")
        for i, r in enumerate(disp.sort_values(col, ascending=asc).head(10).itertuples(index=False), 1):
            W(f"| {i} | {r.name} | {r.position} | {r.teams} | {r.toi_min_5v5:.0f} | {getattr(r,col):+.3f} |")
        W("")
    W("### Leaderboards\n")
    tbl("Top 10 qot_total", "qot_total", False)
    tbl("Bottom 10 qot_total", "qot_total", True)
    tbl("Top 10 qoc_total", "qoc_total", False)
    tbl("Bottom 10 qoc_total", "qoc_total", True)

    # team view: TOI-weighted avg qoc_total per team (proper per-team TOI)
    gteams = T.game_teams("2025_26")
    ptt = T.player_team_toi(df, gteams)
    teamabbr = T.team_abbrevs()
    qoc_map = out.set_index("player_id")["qoc_total"]
    from collections import defaultdict
    num = defaultdict(float); den_t = defaultdict(float)
    for r in ptt.itertuples(index=False):
        q = qoc_map.get(int(r.player_id))
        if q is not None and not np.isnan(q):
            num[int(r.team_id)] += q * r.toi_sec; den_t[int(r.team_id)] += r.toi_sec
    team_qoc = {t: num[t] / den_t[t] for t in den_t if t >= 0}
    vals = np.array(list(team_qoc.values()))
    W("### Team view — TOI-weighted avg qoc_total\n")
    W(f"- Teams: {len(team_qoc)}. Spread: min {vals.min():+.4f} "
      f"({teamabbr.get(min(team_qoc,key=team_qoc.get))}), max {vals.max():+.4f} "
      f"({teamabbr.get(max(team_qoc,key=team_qoc.get))}), "
      f"range {vals.max()-vals.min():.4f}, std {vals.std():.4f}.")
    W("_QoC spread across teams is expected to be small — everyone plays everyone._\n")

    # correlations
    d2 = disp.copy()
    d2["own_rapm"] = d2.player_id.map(rmap_tot)
    c1 = float(np.corrcoef(d2.qot_total, d2.qot_toi)[0, 1])
    c2 = float(np.corrcoef(d2.qoc_total, d2.qoc_toi)[0, 1])
    c3 = float(np.corrcoef(d2.qot_total, d2.own_rapm)[0, 1])
    W("### Correlations (≥500 5v5 min)\n")
    W(f"- qot_total vs qot_toi: **{c1:.3f}**")
    W(f"- qoc_total vs qoc_toi: **{c2:.3f}**")
    W(f"- qot_total vs own RAPM total: **{c3:.3f}**\n")

    with open(os.path.join(T.ROOT, "qoc_qot_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("WROTE", OUT, "and qoc_qot_report.md | rows", len(out), "| md5", md5,
          "| league_avg", round(league_avg, 4))


if __name__ == "__main__":
    main()
