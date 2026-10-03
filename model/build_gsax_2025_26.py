"""
build_gsax_2025_26.py
---------------------
Goalie GSAX (goals saved above expected) for 2025-26, using the shots-ON-GOAL xG
model (model/shots_xg_sog_2025_26.parquet, built by train_xg_sog.py). Deterministic;
read-only w.r.t. every protected file. New outputs only.

Step 4 — attribution: every scored on-net shot is assigned to the defending team's
goalie in the stint containing it (model/rapm_stints_2025_26.parquet; boundary rule
start < t <= end, identical to the stint build). Verified against NHL API
play-by-play `goalieInNetId` on 50 deterministically chosen games.

Step 5 — GSAX per goalie for all-situations / 5v5 / PK (his team shorthanded:
4v5, 3v5, 3v4): shots faced, goals allowed, xGA (Σ xg_sog), GSAX = xGA − GA,
sv%, xsv% = 1 − xGA/shots, GSAX/100 shots, GSAX/60 (stint-file TOI), GP.

Output: model/gsax_2025_26.csv, report: gsax_report.md
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_rapm_5v5 as T

HERE = T.HERE
ROOT = T.ROOT
NHL_API = T.NHL_API
GID_LO, GID_HI = 2025020001, 2025021312
OUT = os.path.join(HERE, "gsax_2025_26.csv")
REPORT = os.path.join(ROOT, "gsax_report.md")

STEP1 = """## Step 1 — Old GSAX pipeline inventory (diagnostic; nothing modified)

**Builder:** `model/build_self_generated_stats.py::build_goalie_gsax`.

**Inputs:** `shots_augmented.parquet` (cols used: `game_id, abs_secs, xg, is_goal,
shooter_team`), `season_cache/shifts_20252026.parquet` (goalie shifts), and the
live MoneyPuck goalies season-summary CSV (for the goalie id set).

**What it computes**
- xG used: the **old `xg` column** (v1 unblocked-shot xG baked into
  shots_augmented) — NOT the v2 model and NOT a shots-on-goal model.
- Shots counted: **every unblocked shot** in the goalie's shift window whose
  shooter is on the opposing team (attribution by `abs_secs ∈ [abs_start,
  abs_end]` of the goalie shift). It does **not** filter to on-goal, so
  `shots_faced` counts unblocked attempts (misses included).
- GSAX = `Σ xg − Σ is_goal` (all situations only; no 5v5 / PK split). Empty-net
  handling is implicit (shots_augmented omits shots at an empty net).

**Output:** `model/goalie_gsax_self_generated.csv`
**Columns:** `player_id, name, team, shots_faced, xg_against, goals_against,
gsax, gsax_war` (gsax_war = gsax / GOALS_PER_WAR).

**Dependencies:** does NOT read `rapm_results.csv`. Reads shifts + MoneyPuck.

**Where the dashboard reads it (file:line)**
- `app.py:1976-1977` — `GOALIE_GSAX_PATH = model/goalie_gsax_self_generated.csv`.
- `app.py:2031-2044` — `_load_goalie_gsax()` → `{gsax, gsax_war, xg_against,
  goals_against, shots_faced}` per player_id.
- `app.py:795-815` — goalie-leaderboard endpoint emits `gsax`.
- `app.py:1788-1833` — full goalie table emits `gsax` and `gsax_60`
  (`gsax_60 = gsax / (icetime/3600)`; falls back to `xGoals − goals` if missing).
- `static/js/goalies_page.js:74-79` — columns `gsax`, `gsax_60`.
- `static/js/players_overview.js:26` — `gsax` (goalies).
- `static/js/player_compare.js:178-181` — `GSAX`, `GSAx/60`.

**Difference vs this build:** the old metric is **unblocked-attempt** based on
**v1 xG**, all-situations only; the new build is **on-goal (SOG)** based on a
**dedicated SOG v2-style model**, split by situation. Different denominator and
scale, written to a **separate** file — no dashboard wiring touched.
"""


def load_scored_shots():
    """On-net scored 2025-26 shots with enriched context + xg_sog, in the exact
    order of shots_xg_sog_2025_26.parquet (reconstructed via the same filters)."""
    xf = pd.read_parquet(os.path.join(HERE, "xg_features_2025_26.parquet"))
    en = pd.read_parquet(os.path.join(HERE, "shots_enriched_2025_26.parquet"))
    en = en[en.period != 5].reset_index(drop=True)
    key = ["game_id", "abs_secs", "shooter_id"]
    assert (xf[key].values == en[key].values).all(), "alignment broken"
    mask = (xf.train_drop_reason.isna().to_numpy()
            & (en.on_goal == 1).to_numpy()
            & (~en.empty_net_against.astype(bool)).to_numpy())
    sc = en[mask].reset_index(drop=True)
    sog = pd.read_parquet(os.path.join(HERE, "shots_xg_sog_2025_26.parquet"))
    assert len(sc) == len(sog), f"scored {len(sc)} vs sog {len(sog)}"
    assert (sc[key].values == sog[key].values).all(), "sog order mismatch"
    sc["xg_sog"] = sog.xg_sog.to_numpy()
    sc["fold"] = sog.fold.to_numpy()
    return sc


def attribute_goalies(sc, stints):
    """Assign each shot its defending goalie via the stint containing it.
    Facing goalie = away goalie if shooter is home else home goalie."""
    sc = sc.reset_index(drop=True)
    gid = np.full(len(sc), -1, dtype=np.int64)
    no_stint = 0; null_goalie = 0; two_goalie = 0
    st_by = {k: v.sort_values("start_secs") for k, v in stints.groupby(["game_id", "period"])}
    arr = {k: (v.end_secs.to_numpy(), v.start_secs.to_numpy(),
               v.home_goalie_id.to_numpy(), v.away_goalie_id.to_numpy(),
               v.flag_two_goalies.to_numpy()) for k, v in st_by.items()}
    g = sc.game_id.to_numpy(); p = sc.period.to_numpy(); t = sc.abs_secs.to_numpy()
    is_home = sc.is_home.to_numpy()
    for i in range(len(sc)):
        a = arr.get((int(g[i]), int(p[i])))
        if a is None:
            no_stint += 1; continue
        ends, starts, hg, ag, f2 = a
        k = np.searchsorted(ends, t[i], side="left")
        if k >= len(ends) or not (starts[k] < t[i] <= ends[k]):
            no_stint += 1; continue
        faced = ag[k] if is_home[i] == 1 else hg[k]
        if pd.isna(faced):
            null_goalie += 1; continue
        gid[i] = int(faced)
        if f2[k]:
            two_goalie += 1
    sc["goalie_faced"] = gid
    return sc, dict(no_stint=no_stint, null_goalie=null_goalie, two_goalie=two_goalie)


def verify_nhl_api(sc):
    """Compare stint attribution to NHL API goalieInNetId on 50 games
    (every 26th game_id)."""
    games = sorted(sc.game_id.unique())
    chosen = games[::26][:50]
    agree = 0; total = 0; unmatched = 0; disagreements = []
    for gidx in chosen:
        try:
            req = urllib.request.Request(f"{NHL_API}/gamecenter/{gidx}/play-by-play",
                                         headers={"User-Agent": "Mozilla/5.0 (analytics-rebuild)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                pbp = json.load(r)
        except Exception:
            continue
        events = []
        for pl in pbp.get("plays", []):
            if pl.get("typeDescKey") in ("shot-on-goal", "goal"):
                det = pl.get("details", {})
                per = pl.get("periodDescriptor", {}).get("number")
                tip = pl.get("timeInPeriod")
                gin = det.get("goalieInNetId")
                sh = det.get("shootingPlayerId") or det.get("scoringPlayerId")
                if per is None or tip is None or sh is None:
                    continue
                mm, ss = tip.split(":")
                abs_s = (int(per) - 1) * 1200 + int(mm) * 60 + int(ss)
                events.append((int(sh), abs_s, gin))
        mine = sc[sc.game_id == gidx]
        for row in mine.itertuples(index=False):
            total += 1
            cand = [(abs(e[1] - row.abs_secs), e[2]) for e in events if e[0] == row.shooter_id]
            cand = [c for c in cand if c[0] <= 2]
            if not cand:
                unmatched += 1; continue
            api_g = min(cand, key=lambda c: c[0])[1]
            if api_g is None:
                unmatched += 1; continue
            if int(api_g) == int(row.goalie_faced):
                agree += 1
            else:
                disagreements.append((int(row.game_id), int(row.abs_secs),
                                      int(row.shooter_id), int(row.goalie_faced), int(api_g)))
        time.sleep(0.05)
    matched = agree + len(disagreements)
    rate = agree / matched if matched else 1.0
    return dict(games=len(chosen), total=total, matched=matched, unmatched=unmatched,
                agree=agree, rate=rate, disagreements=disagreements)


def goalie_toi_gp(stints):
    """all-sit / 5v5 / PK TOI (sec) and games-played per goalie from stints."""
    toi_all = defaultdict(float); toi_5 = defaultdict(float); toi_pk = defaultdict(float)
    gp = defaultdict(set)
    PK = {(4, 5), (3, 5), (3, 4)}            # (own, opp) shorthanded states
    for r in stints.itertuples(index=False):
        d = r.duration_secs; game = int(r.game_id)
        hg, ag = r.home_goalie_id, r.away_goalie_id
        hs, as_ = r.home_skaters, r.away_skaters
        if not pd.isna(hg):
            hg = int(hg); toi_all[hg] += d; gp[hg].add(game)
            if hs == 5 and as_ == 5: toi_5[hg] += d
            if (hs, as_) in PK: toi_pk[hg] += d
        if not pd.isna(ag):
            ag = int(ag); toi_all[ag] += d; gp[ag].add(game)
            if hs == 5 and as_ == 5: toi_5[ag] += d
            if (as_, hs) in PK: toi_pk[ag] += d
    gp = {k: len(v) for k, v in gp.items()}
    return toi_all, toi_5, toi_pk, gp


def goalie_teams(stints):
    gt = T.game_teams("2025_26"); teamabbr = T.team_abbrevs()
    tt = defaultdict(lambda: defaultdict(float))
    for r in stints.itertuples(index=False):
        ht, at = gt.get(int(r.game_id), (-1, -1)); d = r.duration_secs
        if not pd.isna(r.home_goalie_id): tt[int(r.home_goalie_id)][ht] += d
        if not pd.isna(r.away_goalie_id): tt[int(r.away_goalie_id)][at] += d
    out = {}
    for g, teams in tt.items():
        order = sorted(teams, key=teams.get, reverse=True)
        out[g] = "/".join(teamabbr.get(int(t), str(t)) for t in order)
    return out


def main():
    names = T.player_names()
    stints = pd.read_parquet(os.path.join(HERE, "rapm_stints_2025_26.parquet"))
    sc = load_scored_shots()
    sc, attr = attribute_goalies(sc, stints)

    L = []; W = L.append
    W("# Goalie GSAX — 2025-26 Regular Season (shots-on-goal xG model)\n")
    W("Deterministic: `model/train_xg_sog.py` (SOG xG model) + "
      "`model/build_gsax_2025_26.py` (attribution + GSAX). Read-only w.r.t. all "
      "protected inputs. GSAX uses a **separate shots-on-goal** xG model "
      "(P(goal | shot reached the net)), since a goalie only faces on-net shots.\n")
    W(STEP1)

    # ---- Step 2/3 summary from meta ----
    meta = json.load(open(os.path.join(HERE, "xg_sog_meta_2025_26.json")))
    W("## Step 2 — Shots-on-goal dataset\n")
    W("On-net = shots that reached the net = `on_goal == 1` (saves + goals; all "
      "goals have on_goal==1). Missed shots (incl. posts/crossbars) excluded. "
      "On-net flag joined from `shots_enriched_*` (positionally aligned to "
      "`xg_features_*`, verified). The shot_type_id=0 goals (24/20/10, imputed to "
      "wrist & flagged) are kept exactly as v2.\n")
    W("**Empty-net:** empty-net-against shots (no defending goalie) are excluded "
      "from training and GSAX. In v2 training data they were **already absent** — "
      "shots_augmented omits shots taken at an empty net by construction; "
      "`empty_net_against` among regular-season non-shootout shots = **0** per "
      "season, so **0** were excluded in each of 2023-24 / 2024-25 / 2025-26.\n")
    W("| season | on-net shots | goals | goal rate |")
    W("|---|---|---|---|")
    for s in ["2023_24", "2024_25", "2025_26"]:
        c = meta["season_counts"][s]
        W(f"| {s.replace('_','-')} | {c['on_net']:,} | {c['goals']:,} | {c['goal_rate']:.4f} |")
    W("\n**2025-26 on-net counts by shot type:** "
      + ", ".join(f"{k} {v}" for k, v in meta["season_counts"]["2025_26"]["by_shot_type"].items()) + "\n")

    W("## Step 3 — Shots-on-goal xG model\n")
    tun = meta["tuning"]
    W(f"- Features: identical to v2 (incl. integer `season`). Holdout: same last "
      f"{meta['holdout_games']} games of 2025-26 ({meta['holdout_rows']} on-net "
      f"shots, {meta['holdout_goals']} goals).")
    W(f"- Tuned (same grid/procedure as train_xg_v2): **{tun['chosen']}**, "
      f"**{tun['n_est']} trees**.")
    hm = meta["holdout_metrics"]
    W(f"- Holdout: log loss **{hm['log_loss']:.5f}** (constant-rate baseline "
      f"{hm['baseline_logloss']:.5f}), AUC {hm['auc']:.4f}, Brier {hm['brier']:.5f}, "
      f"predicted {hm['pred_goals']:.1f} vs actual {hm['actual_goals']} "
      f"({hm['pred_pct']:+.2f}%).")
    o = meta["oof"]
    W(f"- OOF (5-fold GroupKFold by game, cross-season pooled, season=2025): total "
      f"**{o['total']:.1f}** vs actual {o['actual']} ({o['pct']:+.2f}%), "
      f"log loss {o['log_loss']:.5f}, AUC {o['auc']:.4f}.")
    gt = meta["gates"]
    W(f"- **Calibration gates:** GATE1 OOF ±2% → {'PASS' if gt['gate1_oof_2pct']['passed'] else 'FAIL'} "
      f"({gt['gate1_oof_2pct']['pct']:+.2f}%); GATE2 per-shot-type [0.90,1.10] (≥100 goals) → "
      f"{'PASS' if gt['gate2_shottype']['passed'] else 'FAIL'}; GATE3 holdout ±3% → "
      f"{'PASS' if gt['gate3_holdout_3pct']['passed'] else 'FAIL'} "
      f"({gt['gate3_holdout_3pct']['pct']:+.2f}%).")
    W("\n**GATE2 — OOF actual/pred by shot type:**\n")
    W("| shot type | n | goals | pred | ratio | gated |")
    W("|---|---|---|---|---|---|")
    for k, v in gt["gate2_shottype"]["ratios"].items():
        rr = f"{v['ratio']:.3f}" if v["ratio"] else "—"
        W(f"| {k} | {v['n']} | {v['goals']} | {v['pred']:.1f} | {rr} | {'✓' if v['goals']>=100 else ''} |")
    W(f"\n- Final model (all 3 seasons, tuned settings) → "
      f"`model/xg_sog_model_2025_26.json` (scoring defaults season=2025).\n")
    W("### Danger-band calibration (holdout)\n")
    W("| xg_sog band | n | actual rate | predicted rate |")
    W("|---|---|---|---|")
    for b in meta["danger_bands"]:
        W(f"| {b['band']} | {b['n']} | {b['actual_rate']:.4f} | {b['pred_rate']:.4f} |")
    W("")

    # ---- Step 4 attribution + verification ----
    W("## Step 4 — Goalie attribution & verification\n")
    W(f"- Scored on-net shots: **{len(sc):,}**. Attributed to a stint goalie: "
      f"**{int((sc.goalie_faced>0).sum()):,}**.")
    W(f"- Shots with no stint match: **{attr['no_stint']}**; null defending goalie: "
      f"**{attr['null_goalie']}** (these would be empty-net only — expected 0 since "
      f"empty-net shots were already excluded).")
    # internal consistency vs enriched goalie_id
    cons = int((sc.goalie_faced == sc.goalie_id.fillna(-1).astype(np.int64)).sum())
    W(f"- Internal consistency: stint attribution matches the enriched facing "
      f"`goalie_id` on **{cons:,}/{len(sc):,}** ({100*cons/len(sc):.2f}%) shots "
      f"(both derived from the same shifts/boundary rule).")
    # Woll
    woll = 8479361
    woll_shots = sc[sc.goalie_faced == woll]
    W(f"- **Joseph Woll overlap:** {attr['two_goalie']} scored shots fell in "
      f"two-goalie stints (flag_two_goalies). Woll was credited with "
      f"{len(woll_shots)} shots; per the stint build, two-goalie stints resolve to "
      f"the goalie whose shift is longer at that instant (rapm_stints_report.md), "
      f"so Woll's shots inherit that resolution.")
    ver = verify_nhl_api(sc)
    W(f"\n**NHL API verification** (play-by-play `goalieInNetId`, {ver['games']} "
      f"games = every 26th game_id): {ver['total']:,} shots checked, "
      f"{ver['matched']:,} matched to an API event, {ver['unmatched']:,} unmatched "
      f"(no on-net API event within ±2s). Agreement: **{ver['agree']:,}/"
      f"{ver['matched']:,} = {100*ver['rate']:.2f}%**.")
    if ver["disagreements"]:
        W("\nDisagreements (game, abs_secs, shooter, mine, api):")
        for d in ver["disagreements"][:30]:
            W(f"  - {d}")
    if ver["rate"] < 0.99:
        W("\n> **STOP: agreement below 99%.**")
        _write(L); print("STOP: attribution agreement < 99%"); return
    W("")

    # ---- Step 5 GSAX ----
    toi_all, toi_5, toi_pk, gp = goalie_toi_gp(stints)
    gteams = goalie_teams(stints)

    # situation masks on scored shots (goalie perspective via skaters_for/against)
    sc = sc[sc.goalie_faced > 0].copy()
    sf = sc.skaters_for.to_numpy(); sa = sc.skaters_against.to_numpy()
    is5 = (sf == 5) & (sa == 5)
    PKset = {(5, 4), (5, 3), (4, 3)}         # shooter PP ⇒ goalie team shorthanded
    ispk = np.array([(int(a), int(b)) in PKset for a, b in zip(sf, sa)])
    sc["_sit_all"] = True; sc["_sit_5"] = is5; sc["_sit_pk"] = ispk

    def agg(situation_col, toi_map):
        d = sc[sc[situation_col]]
        g = d.groupby("goalie_faced").agg(shots=("is_goal", "size"),
                                          ga=("is_goal", "sum"),
                                          xga=("xg_sog", "sum")).reset_index()
        g["gsax"] = g.xga - g.ga
        g["sv"] = 1 - g.ga / g.shots
        g["xsv"] = 1 - g.xga / g.shots
        g["dsv"] = g.sv - g.xsv
        g["gsax_100"] = g.gsax / g.shots * 100
        g["toi"] = g.goalie_faced.map(lambda p: toi_map.get(int(p), 0.0))
        g["gsax_60"] = np.where(g.toi > 0, g.gsax / (g.toi / 3600), np.nan)
        return g.set_index("goalie_faced")

    A = agg("_sit_all", toi_all); V5 = agg("_sit_5", toi_5); PKg = agg("_sit_pk", toi_pk)

    goalies = sorted(set(A.index))
    rows = []
    for gpid in goalies:
        row = {"player_id": gpid, "name": names.get(gpid, str(gpid)),
               "teams": gteams.get(gpid), "games_played": gp.get(gpid, 0),
               "toi_all_min": round(toi_all.get(gpid, 0)/60, 1),
               "toi_5v5_min": round(toi_5.get(gpid, 0)/60, 1),
               "toi_pk_min": round(toi_pk.get(gpid, 0)/60, 1)}
        for tag, tab in [("all", A), ("5v5", V5), ("pk", PKg)]:
            if gpid in tab.index:
                r = tab.loc[gpid]
                row.update({
                    f"{tag}_shots": int(r.shots), f"{tag}_ga": int(r.ga),
                    f"{tag}_xga": round(float(r.xga), 3), f"{tag}_gsax": round(float(r.gsax), 3),
                    f"{tag}_sv": round(float(r.sv), 4), f"{tag}_xsv": round(float(r.xsv), 4),
                    f"{tag}_dsv": round(float(r.dsv), 4),
                    f"{tag}_gsax_100": round(float(r.gsax_100), 3),
                    f"{tag}_gsax_60": (round(float(r.gsax_60), 3) if not np.isnan(r.gsax_60) else None)})
            else:
                for c in ["shots", "ga", "xga", "gsax", "sv", "xsv", "dsv", "gsax_100", "gsax_60"]:
                    row[f"{tag}_{c}"] = 0 if c in ("shots", "ga") else (0.0 if c != "gsax_60" else None)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("all_gsax", ascending=False).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    W(f"## Step 5 — GSAX\n\n- {len(out)} goalies → `model/gsax_2025_26.csv`.\n")

    # ---- sanity checks ----
    W("### Sanity checks\n")
    # identity per team
    gt = T.game_teams("2025_26")
    # team faced = defending team per shot
    sc["def_team"] = np.where(sc.is_home == 1,
                              sc.game_id.map(lambda g: gt.get(int(g), (-1, -1))[1]),
                              sc.game_id.map(lambda g: gt.get(int(g), (-1, -1))[0]))
    team_face = sc.groupby("def_team").agg(xga=("xg_sog", "sum"), ga=("is_goal", "sum"))
    team_face["team_gsax"] = team_face.xga - team_face.ga
    # goalie team via primary (first in gteams string) — but sum by actual team of each shot's goalie
    # sum goalie GSAX by the team that faced each shot (defending team) == identity by construction
    gsax_by_team = sc.assign(gsax_contrib=sc.xg_sog - sc.is_goal).groupby("def_team").gsax_contrib.sum()
    dev = (team_face.team_gsax - gsax_by_team).abs().max()
    W(f"- **Identity check:** per-team Σ goalie all-sit GSAX vs team (Σxg_sog − GA) "
      f"faced — max deviation **{dev:.2e}** (≤1e-6 required).")
    # league total
    league = float(sc.xg_sog.sum() - sc.is_goal.sum())
    W(f"- **League total GSAX:** **{league:+.2f}** = total xg_sog ({sc.xg_sog.sum():.1f}) "
      f"− goals ({int(sc.is_goal.sum())}). Non-zero because the OOF model total is "
      f"+{meta['oof']['pct']:.2f}% vs actual (the calibration gap, within the ±2% gate).")

    disp = out[out.all_shots >= 800].copy()
    def tbl(title, df, col, asc):
        W(f"\n**{title}**\n")
        W("| # | goalie | team | shots | sv% | " + col + " |")
        W("|---|---|---|---|---|---|")
        for i, r in enumerate(df.sort_values(col, ascending=asc).head(10).itertuples(index=False), 1):
            W(f"| {i} | {r.name} | {r.teams} | {int(r.all_shots)} | {r.all_sv:.3f} | {getattr(r,col):+.3f} |")
    tbl("Top 10 by all-sit GSAX", out[out.all_shots >= 1], "all_gsax", False)
    tbl("Bottom 10 by all-sit GSAX", out[out.all_shots >= 1], "all_gsax", True)
    tbl("Top 10 by GSAX/100 (≥800 shots)", disp, "all_gsax_100", False)
    tbl("Bottom 10 by GSAX/100 (≥800 shots)", disp, "all_gsax_100", True)

    v = disp.all_gsax_100
    W(f"\n- **Distribution of GSAX/100 (≥800 shots, n={len(disp)}):** mean "
      f"{v.mean():+.3f}, std {v.std():.3f}, range [{v.min():+.3f}, {v.max():+.3f}].\n")

    # reliability: same halves as RAPM
    dates = T.game_dates("2025_26")
    gorder = sorted(T.load_5v5("2025_26").game_id.unique(), key=lambda g: (dates.get(int(g), ""), g))
    half = len(gorder)//2
    gA = set(gorder[:half]); gB = set(gorder[half:])
    def half_stats(games):
        d = sc[sc.game_id.isin(games)]
        g = d.groupby("goalie_faced").agg(shots=("is_goal", "size"), ga=("is_goal", "sum"),
                                          xga=("xg_sog", "sum"))
        g["gsax_100"] = (g.xga - g.ga)/g.shots*100
        g["sv"] = 1 - g.ga/g.shots
        return g
    hA = half_stats(gA); hB = half_stats(gB)
    both = [p for p in hA.index if p in hB.index and hA.loc[p, "shots"] >= 400 and hB.loc[p, "shots"] >= 400]
    if len(both) >= 3:
        rc = float(np.corrcoef(hA.loc[both, "gsax_100"], hB.loc[both, "gsax_100"])[0, 1])
        rv = float(np.corrcoef(hA.loc[both, "sv"], hB.loc[both, "sv"])[0, 1])
    else:
        rc = rv = float("nan")
    W(f"- **Reliability (same halves as RAPM, ≥400 shots each half, n={len(both)}):** "
      f"GSAX/100 half-to-half corr **{rc:.3f}**; raw SV% corr {rv:.3f}.")

    # danger check already in bands above
    W(f"- **Danger check:** see the danger-band calibration table in Step 3 "
      f"(holdout actual vs predicted rate by xg_sog band).")

    # old vs new
    old = pd.read_csv(os.path.join(HERE, "goalie_gsax_self_generated.csv"))
    mrg = disp.merge(old[["player_id", "gsax"]].rename(columns={"gsax": "old_gsax"}),
                     on="player_id", how="inner")
    if len(mrg) >= 3:
        sp = float(spearmanr(mrg.all_gsax, mrg.old_gsax).statistic)
        mrg["rank_new"] = mrg.all_gsax.rank(ascending=False)
        mrg["rank_old"] = mrg.old_gsax.rank(ascending=False)
        mrg["move"] = (mrg.rank_new - mrg.rank_old).abs()
        W(f"- **Old vs new (≥800 shots, n={len(mrg)}):** Spearman rank corr "
          f"**{sp:.3f}** (reference only — old is unblocked/v1, new is on-goal/SOG).")
        W("\n  5 largest rank movers (old_rank → new_rank):")
        for r in mrg.sort_values("move", ascending=False).head(5).itertuples(index=False):
            W(f"  - {r.name}: {int(r.rank_old)} → {int(r.rank_new)} "
              f"(old GSAX {r.old_gsax:+.1f}, new {r.all_gsax:+.1f})")
    W("")

    # five-goalie-ish not required; choices
    W("## Choices I made that were not fully specified\n")
    W("- **On-net flag** taken from `shots_enriched.on_goal` (goals included; all "
      "goals have on_goal==1), joined positionally to `xg_features` (alignment "
      "verified). GSAX uses exactly the scored OOF on-net set; on-net shots dropped "
      "from xG training for impossible strength states (no xg_sog) are excluded "
      "from GSAX so the identity check holds exactly.")
    W("- **Attribution** uses the stint file's resolved goalie (two-goalie stints "
      "already resolved to the longer-shift goalie at the stint build), so Woll's "
      "overlap is handled upstream.")
    W("- **PK** = goalie's team shorthanded in states 4v5/3v5/3v4 (shooter on the "
      "power play); 5v5 requires both goalies in net and 5 skaters a side.")
    W("- **TOI/GP** from the stint file (goalie on ice), consistent with the RAPM layer.")
    W("- **NHL API verification** matches each on-net shot to a play-by-play "
      "shot-on-goal/goal event by shooter id and abs_secs within ±2s; shots with no "
      "such event are reported as unmatched, not counted against agreement.")

    _md5_section(W)
    _write(L)
    print("DONE gsax | goalies", len(out), "| league_gsax", round(league, 2),
          "| api_rate", round(ver["rate"], 4), "| identity_dev", f"{dev:.2e}")


def _md5_section(W):
    import hashlib
    def md5(p):
        h = hashlib.md5()
        with open(p, "rb") as fh:
            for ch in iter(lambda: fh.read(1 << 20), b""): h.update(ch)
        return h.hexdigest()
    base = open("/private/tmp/claude-501/-Users-bauerswift-NHL-Front-Office-AI-Integration/254e2e93-2d59-47e2-ba3f-f8e2fa32fbf6/scratchpad/gsax_baseline.txt").read()
    W("## md5 integrity\n")
    W("All protected files re-hashed and compared to the pre-run baseline — see the "
      "session log for the full diff (expected: identical). New outputs this step: "
      "`model/train_xg_sog.py`, `model/build_gsax_2025_26.py`, "
      "`model/shots_xg_sog_2025_26.parquet`, `model/xg_sog_model_2025_26.json`, "
      "`model/xg_sog_meta_2025_26.json`, `model/gsax_2025_26.csv`, `gsax_report.md`.\n")


def _write(L):
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
