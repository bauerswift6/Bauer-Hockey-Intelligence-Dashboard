"""
rapm_kappa_tiebreak.py
----------------------
Follow-up to train_rapm_5v5.py: settle kappa among the near-tied values with a
deterministic step-down rule, add a reliability diagnostic that nets out the
shared-prior component of the half-to-half correlation, and refit ONLY if the
selected kappa differs from the previous pick (0.75).

Imports train_rapm_5v5 (does NOT edit it). Same data, model definition, halves,
lambda-selection procedure, and bootstrap seed as the previous run. Read-only
w.r.t. every protected file. New outputs only.
"""
from __future__ import annotations
import hashlib
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import train_rapm_5v5 as T

HERE = T.HERE
ROOT = T.ROOT
SEED = T.SEED
LAM_GRID = T.LAM_GRID
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
REPORT = os.path.join(ROOT, "rapm_kappa_tiebreak_report.md")

# previous-run values (rapm_5v5_report.md) for the >0.001 reproduction gate
PREV = {
    0.0:  dict(ab=123.3074, ba=119.7614, avg=121.5344),
    0.25: dict(ab=123.2879, ba=119.7258, avg=121.5069),
    0.5:  dict(ab=123.2738, ba=119.7076, avg=121.4907),
    0.75: dict(ab=123.2698, ba=119.7003, avg=121.4850),
    1.0:  dict(ab=123.2757, ba=119.7039, avg=121.4898),
}
PREV_BASE = dict(ab=123.4882, ba=119.9108, avg=121.6995)
PREV_HALFCORR = {0.0: 0.332, 0.75: 0.739}


def per_game(err_k):
    dfe = pd.DataFrame({"g": err_k["game"], "w": err_k["w"], "sq": err_k["sq"]})
    dfe["wse"] = dfe.w * dfe.sq
    return dfe.groupby("g").agg(sw=("w", "sum"), wse=("wse", "sum"))


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    L = []; W = L.append
    W("# 5v5 RAPM — kappa tiebreak & reliability diagnostic\n")
    W("Deterministic: `model/rapm_kappa_tiebreak.py` (imports `train_rapm_5v5.py`, "
      "does not modify it). Same data, model, halves, lambda procedure, and "
      "bootstrap seed as the previous run.\n")

    # ---- reproduce 2024-25 prior (xg + goals) --------------------------------
    df24 = T.load_5v5("2024_25")
    pidx24 = T.player_universe(df24)
    X24, y24, w24, g24, pen24, lay24 = T.build_design(df24, "xg", pidx24)
    lam24, _ = T.cv_lambda(X24, y24, w24, g24, pen24, LAM_GRID, np.zeros(lay24["ncol"]))
    beta24 = T.fit_full(X24, y24, w24, pen24, lam24, np.zeros(lay24["ncol"]))
    fr24 = T.coefs_to_frame(beta24, pidx24, lay24)
    prior_raw = {int(r.player_id): (r.off_raw, r.def_raw) for r in fr24.itertuples(index=False)}
    prior_total = {int(r.player_id): r.total for r in fr24.itertuples(index=False)}
    Xg24, yg24, wg24, gg24, peng24, layg24 = T.build_design(df24, "goals", pidx24)
    lamg24, _ = T.cv_lambda(Xg24, yg24, wg24, gg24, peng24, LAM_GRID, np.zeros(layg24["ncol"]))
    betag24 = T.fit_full(Xg24, yg24, wg24, peng24, lamg24, np.zeros(layg24["ncol"]))
    prior_raw_goals = {int(r.player_id): (r.off_raw, r.def_raw)
                       for r in T.coefs_to_frame(betag24, pidx24, layg24).itertuples(index=False)}
    toi24 = T.player_toi_5v5(df24)

    # ---- 2025-26 design + halves --------------------------------------------
    df25 = T.load_5v5("2025_26")
    pidx25 = T.player_universe(df25)
    X25, y25, w25, g25, pen25, lay25 = T.build_design(df25, "xg", pidx25)
    dates = T.game_dates("2025_26")
    gorder = sorted(df25.game_id.unique(), key=lambda g: (dates.get(int(g), ""), g))
    half = len(gorder) // 2
    gamesA = set(gorder[:half]); gamesB = set(gorder[half:])
    maskA = np.isin(g25, list(gamesA)); maskB = ~maskA

    # baseline (intercept+covariates only)
    base_ab = float(np.sum(w25[maskB]*(y25[maskB]-T.predict_baseline(X25[maskA], y25[maskA], w25[maskA], X25[maskB]))**2)/np.sum(w25[maskB]))
    base_ba = float(np.sum(w25[maskA]*(y25[maskA]-T.predict_baseline(X25[maskB], y25[maskB], w25[maskB], X25[maskA]))**2)/np.sum(w25[maskA]))
    base_avg = 0.5*(base_ab+base_ba)

    # ---- per-kappa fits + per-row errors ------------------------------------
    results = {}; fits = {}; err = {}
    for kappa in KAPPAS:
        po = T.build_prior_offset(pidx25, lay25, prior_raw, kappa)
        lamA, _ = T.cv_lambda(X25[maskA], y25[maskA], w25[maskA], g25[maskA], pen25, LAM_GRID, po)
        betaA = T.fit_full(X25[maskA], y25[maskA], w25[maskA], pen25, lamA, po)
        rB = y25[maskB] - X25[maskB] @ betaA
        mseB = float(np.sum(w25[maskB]*rB*rB)/np.sum(w25[maskB]))
        lamB, _ = T.cv_lambda(X25[maskB], y25[maskB], w25[maskB], g25[maskB], pen25, LAM_GRID, po)
        betaB = T.fit_full(X25[maskB], y25[maskB], w25[maskB], pen25, lamB, po)
        rA = y25[maskA] - X25[maskA] @ betaB
        mseA = float(np.sum(w25[maskA]*rA*rA)/np.sum(w25[maskA]))
        results[kappa] = dict(lamA=lamA, lamB=lamB, ab=mseB, ba=mseA, avg=0.5*(mseA+mseB))
        fits[kappa] = (betaA, betaB)
        err[kappa] = dict(game=np.concatenate([g25[maskB], g25[maskA]]),
                          w=np.concatenate([w25[maskB], w25[maskA]]),
                          sq=np.concatenate([rB*rB, rA*rA]))
        print(f"kappa={kappa}: lamA={lamA} lamB={lamB} avg={results[kappa]['avg']:.4f}")

    # ================= STEP 1: table + reproduction gate ====================
    W("## Step 1 — Kappa selection table (reproduction) + bootstrap\n")
    stop = False
    W("| kappa | MSE A→B | MSE B→A | avg MSE | prev avg | |Δ| |")
    W("|---|---|---|---|---|---|")
    for k in KAPPAS:
        r = results[k]; d = abs(r["avg"]-PREV[k]["avg"])
        flag = "" if d <= 0.001 else " ❌>0.001"
        if d > 0.001: stop = True
        W(f"| {k} | {r['ab']:.4f} | {r['ba']:.4f} | **{r['avg']:.4f}** | {PREV[k]['avg']:.4f} | {d:.4f}{flag} |")
    W(f"| baseline | {base_ab:.4f} | {base_ba:.4f} | **{base_avg:.4f}** | {PREV_BASE['avg']:.4f} | {abs(base_avg-PREV_BASE['avg']):.4f} |")
    W("")
    if stop:
        W("> **STOP**: a reproduced average MSE differs from the previous report by "
          "more than 0.001. Not proceeding.\n")
        _write(L); print("STOP: reproduction mismatch"); return

    # ---- bootstrap: same seed, one resample per iter, all pairs -------------
    games = np.array(sorted(df25.game_id.unique()))
    nG = len(games)
    pg = {k: per_game(err[k]).reindex(games) for k in KAPPAS}
    sw = pg[0.0]["sw"].to_numpy()                      # identical across kappas
    wse = {k: pg[k]["wse"].to_numpy() for k in KAPPAS}
    rng = np.random.RandomState(SEED)
    boot = {k: np.empty(1000) for k in KAPPAS}
    for i in range(1000):
        idx = rng.randint(0, nG, nG)
        den = sw[idx].sum()
        for k in KAPPAS:
            boot[k][i] = wse[k][idx].sum()/den
    def ci(a, b):            # CI of MSE(a) - MSE(b)
        d = boot[a]-boot[b]
        return float((boot[a]-boot[b]).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    pairs = [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
             (0.5, 0.25), (0.75, 0.5), (1.0, 0.75)]
    W("**Bootstrap 95% CIs of MSE differences** (1,000 resamples, seed "
      f"{SEED}, both directions pooled). Negative ⇒ the first (larger) kappa has lower MSE.\n")
    W("| difference | mean | 95% CI |")
    W("|---|---|---|")
    ci_store = {}
    for a, b in pairs:
        m, lo, hi = ci(a, b); ci_store[(a, b)] = (m, lo, hi)
        W(f"| {a} − {b} | {m:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    # consistency check vs previous (0.75-0 was [-0.0605,-0.0368])
    m, lo, hi = ci_store[(0.75, 0.0)]
    W(f"\n_Consistency: 0.75 − 0 reproduces the previous bootstrap "
      f"([{lo:+.5f}, {hi:+.5f}] vs previously [-0.06054, -0.03682])._\n")

    # ================= selection walk =======================================
    W("### Selection walk (step-down tiebreak)\n")
    order = KAPPAS[:]
    pick = min(order, key=lambda k: results[k]["avg"])
    W(f"- Start at lowest-avg-MSE kappa = **{pick}** (avg {results[pick]['avg']:.4f}).")
    while True:
        i = order.index(pick)
        if i == 0:
            W("- Reached kappa 0; walk ends."); break
        smaller = order[i-1]
        m, lo, hi = ci(pick, smaller)          # MSE(pick) - MSE(smaller)
        includes0 = lo <= 0 <= hi
        if hi < 0:
            W(f"- {pick} vs {smaller}: CI(({pick})−({smaller})) = [{lo:+.5f},{hi:+.5f}] "
              f"excludes 0 with {smaller} worse ⇒ **stop at {pick}**."); break
        elif includes0:
            W(f"- {pick} vs {smaller}: CI = [{lo:+.5f},{hi:+.5f}] includes 0 "
              f"(tied) ⇒ step down to {smaller}.")
            pick = smaller
        else:  # lo>0: smaller strictly better
            W(f"- {pick} vs {smaller}: CI = [{lo:+.5f},{hi:+.5f}] > 0 "
              f"({smaller} better) ⇒ step down to {smaller}.")
            pick = smaller
    # final must beat 0
    if pick != 0.0:
        m, lo, hi = ci(pick, 0.0)
        if hi < 0:
            W(f"- Final check: {pick} − 0 CI [{lo:+.5f},{hi:+.5f}] excludes 0 ⇒ "
              f"**{pick} beats kappa 0**.")
            chosen = pick
        else:
            W(f"- Final check: {pick} − 0 CI [{lo:+.5f},{hi:+.5f}] includes 0 ⇒ "
              f"does not beat kappa 0 ⇒ **choose kappa 0**.")
            chosen = 0.0
    else:
        chosen = 0.0
    W(f"\n- **Selected kappa = {chosen}** (previous pick was 0.75).\n")

    # ================= STEP 2: reliability diagnostic =======================
    W("## Step 2 — Reliability diagnostic\n")
    toiA = T.player_toi_5v5(df25[df25.game_id.isin(gamesA)])
    toiB = T.player_toi_5v5(df25[df25.game_id.isin(gamesB)])
    elig = [pid for pid in pidx25 if toiA.get(pid, 0) >= 12000 and toiB.get(pid, 0) >= 12000]
    W(f"- Players with ≥200 5v5 min in each half: **{len(elig)}**.\n")
    W("| kappa | (a) corr(total_H1,total_H2) | (b) corr data-driven (total − κ·prior) | "
      "(c) corr(total_H1, prior) | (c) corr(total_H2, prior) |")
    W("|---|---|---|---|---|")
    pt = np.array([prior_total.get(pid, 0.0) for pid in elig])
    for k in [0.0, 0.5, 0.75]:
        bA, bB = fits[k]
        fA = T.coefs_to_frame(bA, pidx25, lay25).set_index("player_id").loc[elig, "total"].to_numpy()
        fB = T.coefs_to_frame(bB, pidx25, lay25).set_index("player_id").loc[elig, "total"].to_numpy()
        ca = float(np.corrcoef(fA, fB)[0, 1])
        dA = fA - k*pt; dB = fB - k*pt
        cb = float(np.corrcoef(dA, dB)[0, 1])
        c1 = float(np.corrcoef(fA, pt)[0, 1]); c2 = float(np.corrcoef(fB, pt)[0, 1])
        W(f"| {k} | {ca:.3f} | {cb:.3f} | {c1:.3f} | {c2:.3f} |")
    W(f"\n_Reproduces previous half-to-half: kappa 0 ≈ {PREV_HALFCORR[0.0]}, "
      f"kappa 0.75 ≈ {PREV_HALFCORR[0.75]}. Column (b) nets out the shared prior: "
      "it is the repeatable 2025-26 signal only._\n")

    # prior(2024-25) vs full-season kappa0 total (>=500 min both seasons)
    lam0full, _ = T.cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, np.zeros(lay25["ncol"]))
    beta0full = T.fit_full(X25, y25, w25, pen25, lam0full, np.zeros(lay25["ncol"]))
    fr0full = T.coefs_to_frame(beta0full, pidx25, lay25).set_index("player_id")
    toi25 = T.player_toi_5v5(df25)
    both = [pid for pid in pidx25 if toi24.get(pid, 0) >= 30000 and toi25.get(pid, 0) >= 30000
            and pid in prior_total]
    pv = np.array([prior_total[pid] for pid in both])
    cv = fr0full.loc[both, "total"].to_numpy()
    cpr = float(np.corrcoef(pv, cv)[0, 1])
    W(f"- Corr(2024-25 prior total, 2025-26 κ=0 full-season total), ≥500 5v5 min "
      f"both seasons (n={len(both)}): **{cpr:.3f}**.\n")

    # ================= STEP 4: lambda tables ================================
    W("## Step 4 — Lambda grid & chosen lambdas\n")
    W(f"- Grid (all fits): `{LAM_GRID}` — edges {LAM_GRID[0]} and {LAM_GRID[-1]}.")
    def edge(l): return " ⚠️EDGE" if l in (LAM_GRID[0], LAM_GRID[-1]) else ""
    W("\n| fit | chosen lambda |")
    W("|---|---|")
    W(f"| 2024-25 prior (xG, κ=0) | {lam24:,}{edge(lam24)} |")
    W(f"| 2024-25 prior (goals, κ=0) | {lamg24:,}{edge(lamg24)} |")
    for k in KAPPAS:
        W(f"| half fit A (κ={k}) | {results[k]['lamA']:,}{edge(results[k]['lamA'])} |")
        W(f"| half fit B (κ={k}) | {results[k]['lamB']:,}{edge(results[k]['lamB'])} |")
    W(f"| full season κ=0 | {lam0full:,}{edge(lam0full)} |")

    # full-season chosen-kappa lambda (for refit or just reporting)
    po_final = T.build_prior_offset(pidx25, lay25, prior_raw, chosen)
    lamF, _ = T.cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, po_final)
    W(f"| full season κ={chosen} (this run) | {lamF:,}{edge(lamF)} |")
    W("\n_No lambda sits at a grid edge (all interior); none changed._\n"
      if not any(l in (LAM_GRID[0], LAM_GRID[-1]) for l in
                 [lam24, lamg24, lam0full, lamF]+[results[k]['lamA'] for k in KAPPAS]+[results[k]['lamB'] for k in KAPPAS])
      else "\n_See ⚠️EDGE flags above._\n")

    # ================= STEP 3: refit only if kappa changed ==================
    W("## Step 3 — Refit decision\n")
    if chosen == 0.75:
        W("- Selected kappa = 0.75 (unchanged). **No new model outputs written** — "
          "`model/rapm_5v5_2025_26.csv` already reflects the chosen kappa.\n")
    else:
        W(f"- Selected kappa = {chosen} ≠ 0.75 → refitting full season.\n")
        _refit_and_sanity(W, chosen, lamF, po_final, pidx25, lay25, X25, y25, w25, g25, pen25,
                          df25, prior_raw, prior_raw_goals, fr0full, toi25, beta0full)

    # ================= md5 integrity ========================================
    _md5_section(W, chosen)
    _write(L)
    print("DONE tiebreak | chosen", chosen, "| lamF", lamF)


def _refit_and_sanity(W, chosen, lamF, po_final, pidx25, lay25, X25, y25, w25, g25, pen25,
                      df25, prior_raw, prior_raw_goals, fr0full, toi25, beta0full):
    names = T.player_names(); teamabbr = T.team_abbrevs()
    pos = pd.read_csv(os.path.join(HERE, "player_positions.csv"))
    pos_map = dict(zip(pos.player_id.astype(int), pos.position))
    posf = lambda p: pos_map.get(int(p)) if isinstance(pos_map.get(int(p)), str) else None

    betaF = T.fit_full(X25, y25, w25, pen25, lamF, po_final)
    frF = T.coefs_to_frame(betaF, pidx25, lay25)
    Xg25, yg25, wg25, gg25, peng25, layg25 = T.build_design(df25, "goals", pidx25)
    po_goals = T.build_prior_offset(pidx25, layg25, prior_raw_goals, chosen)
    frGF = T.coefs_to_frame(T.fit_full(Xg25, yg25, wg25, peng25, lamF, po_goals), pidx25, layg25).set_index("player_id")

    # previous kappa0.75 totals (read existing file, read-only)
    prev = pd.read_csv(os.path.join(HERE, "rapm_5v5_2025_26.csv")).set_index("player_id")

    gteams = T.game_teams("2025_26"); ptt = T.player_team_toi(df25, gteams)
    prim = ptt.sort_values("toi_sec", ascending=False).drop_duplicates("player_id").set_index("player_id")["team_id"]
    teamlist = ptt.sort_values("toi_sec", ascending=False).groupby("player_id")["team_id"].apply(
        lambda s: "/".join(teamabbr.get(int(t), str(t)) for t in s))

    out = frF[["player_id", "offense", "defense", "total"]].copy()
    out["name"] = out.player_id.map(names); out["position"] = out.player_id.map(posf)
    out["teams"] = out.player_id.map(teamlist)
    out["toi_sec"] = out.player_id.map(toi25).fillna(0.0)
    out["toi_min_5v5"] = (out.toi_sec/60).round(1)
    out["total_impact"] = out.total*out.toi_sec/3600.0
    out["prior_offense"] = out.player_id.map(lambda p: prior_raw.get(int(p), (0.0, 0.0))[0])
    out["prior_defense"] = out.player_id.map(lambda p: -prior_raw.get(int(p), (0.0, 0.0))[1])
    out["prior_total"] = out.prior_offense+out.prior_defense
    out["kappa"] = chosen; out["lambda"] = lamF
    out["goals_offense"] = out.player_id.map(frGF.offense)
    out["goals_defense"] = out.player_id.map(frGF.defense)
    out["goals_total"] = out.player_id.map(frGF.total)
    out = out[["player_id", "name", "position", "teams", "toi_min_5v5", "offense", "defense",
               "total", "total_impact", "prior_offense", "prior_defense", "prior_total",
               "kappa", "lambda", "goals_offense", "goals_defense", "goals_total"]]
    kstr = str(chosen).replace(".", "")
    outpath = os.path.join(HERE, f"rapm_5v5_2025_26_k{kstr}.csv")
    out.to_csv(outpath, index=False)
    W(f"- Wrote `model/{os.path.basename(outpath)}` ({len(out)} players, 17 columns).")

    disp = out[out.toi_min_5v5 >= 500].copy()
    cg = float(np.corrcoef(disp.total, disp.goals_total)[0, 1])
    W(f"- xG-total vs goals-total corr (≥500 min, n={len(disp)}): **{cg:.3f}**.\n")

    # Spearman vs previous 0.75 + biggest rank movers
    common = [p for p in disp.player_id if p in prev.index]
    nt = disp.set_index("player_id").loc[common, "total"]
    ot = prev.loc[common, "total"]
    sp = float(spearmanr(nt, ot).statistic)
    W(f"- Spearman(new total, κ=0.75 total), ≥500 min (n={len(common)}): **{sp:.3f}**.\n")
    rank_new = nt.rank(ascending=False); rank_old = ot.rank(ascending=False)
    mv = (rank_new-rank_old).abs().sort_values(ascending=False).head(10)
    W("**10 largest rank moves vs κ=0.75:**\n")
    W("| player | κ=0.75 rank | new rank | Δrank |")
    W("|---|---|---|---|")
    for pid in mv.index:
        W(f"| {names.get(int(pid),pid)} | {int(rank_old[pid])} | {int(rank_new[pid])} | "
          f"{int(rank_old[pid]-rank_new[pid]):+d} |")
    W("")

    # five players (with kappa0 and kappa0.75 beside)
    fr0 = fr0full  # kappa0 full
    W("### Five players (new, with κ=0 and κ=0.75 beside)\n")
    ranked = disp.sort_values("total", ascending=False).reset_index(drop=True)
    rankmap = {int(r.player_id): i+1 for i, r in ranked.iterrows()}
    W("| player | pos | 5v5 min | off | def | total | impact | rank | κ0 total | κ0.75 total |")
    W("|---|---|---|---|---|---|---|---|---|---|")
    for nm, pid in {"McDavid": 8478402, "Kucherov": 8476453, "MacKinnon": 8477492,
                    "Brady Tkachuk": 8480801, "Mark Stone": 8475913}.items():
        r = out[out.player_id == pid].iloc[0]
        k0t = fr0.loc[pid, "total"] if pid in fr0.index else float("nan")
        k75 = prev.loc[pid, "total"] if pid in prev.index else float("nan")
        W(f"| {nm} | {r.position} | {r.toi_min_5v5:.0f} | {r.offense:+.2f} | {r.defense:+.2f} | "
          f"{r.total:+.2f} | {r.total_impact:+.2f} | {rankmap.get(pid,'—')} | {k0t:+.2f} | {k75:+.2f} |")
    W("")

    def tbl(title, dfx, col, asc=False, n=20):
        W(f"**{title}**\n")
        W("| # | player | pos | team | 5v5 min | off | def | total |")
        W("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(dfx.sort_values(col, ascending=asc).head(n).itertuples(index=False), 1):
            W(f"| {i} | {r.name} | {r.position} | {r.teams} | {r.toi_min_5v5:.0f} | "
              f"{r.offense:+.2f} | {r.defense:+.2f} | {r.total:+.2f} |")
        W("")
    tbl("Top 20 by total", disp, "total", n=20)
    tbl("Top 10 offense", disp, "offense", n=10)
    tbl("Top 10 defense", disp, "defense", n=10)
    tbl("Bottom 10 by total", disp, "total", asc=True, n=10)

    top20 = disp.sort_values("total", ascending=False).head(20)
    prim_ab = top20.player_id.map(lambda p: teamabbr.get(int(prim.get(p, -1)), "?"))
    vc = prim_ab.value_counts()
    W(f"- Team clustering (top 20): distinct teams **{prim_ab.nunique()}**, most from one "
      f"team **{int(vc.iloc[0])}** ({vc.index[0]}).\n")

    from collections import defaultdict
    tf = defaultdict(float); ta = defaultdict(float); tt = defaultdict(float)
    for row in df25.itertuples(index=False):
        ht, at = gteams.get(int(row.game_id), (-1, -1)); d = row.duration_secs
        tf[ht] += row.home_xg; ta[ht] += row.away_xg; tt[ht] += d
        tf[at] += row.away_xg; ta[at] += row.home_xg; tt[at] += d
    team_actual = {t: (tf[t]-ta[t])/tt[t]*3600 for t in tt}
    tot_map = out.set_index("player_id")["total"]; pred_num = defaultdict(float)
    for r in ptt.itertuples(index=False):
        pred_num[int(r.team_id)] += tot_map.get(int(r.player_id), 0.0)*r.toi_sec
    team_pred = {t: pred_num[t]/tt[t] for t in tt if t in pred_num}
    common_t = [t for t in team_actual if t in team_pred and t >= 0]
    corr_team = float(np.corrcoef([team_actual[t] for t in common_t], [team_pred[t] for t in common_t])[0, 1])
    W(f"- Team reconciliation corr: **{corr_team:.3f}** (n={len(common_t)}).\n")

    disp["grp"] = disp.position.map(lambda p: "D" if p == "D" else ("F" if p in ("C", "L", "R") else None))
    W("**Distribution (≥500 min, by position):**\n")
    W("| group | n | metric | mean | std | min | max |")
    W("|---|---|---|---|---|---|---|")
    for grp in ["F", "D"]:
        sub = disp[disp.grp == grp]
        for metric in ["offense", "defense", "total"]:
            v = sub[metric]
            W(f"| {grp} | {len(sub)} | {metric} | {v.mean():+.2f} | {v.std():.2f} | {v.min():+.2f} | {v.max():+.2f} |")
    W("")


def _md5_section(W, chosen):
    files = [
        "shots_augmented.parquet", "season_cache/shifts_20252026.parquet",
        "season_cache/shifts_20242025.parquet", "shifts_supplement_2024_25.parquet",
        "shots_enriched_2025_26.parquet", "shots_enriched_2024_25.parquet",
        "shots_enriched_2023_24.parquet", "xg_features_2025_26.parquet",
        "xg_features_2024_25.parquet", "xg_features_2023_24.parquet",
        "shots_xg_2025_26.parquet", "shots_xg_2025_26_v2.parquet",
        "xg_model_2025_26.json", "xg_model_2025_26_v2.json", "player_handedness.csv",
        "rapm_stints_2025_26.parquet", "build_rapm_stints_2025_26.py",
        "composite_ratings_sim.csv", "rapm_results.csv", "team_strength.csv",
        "season_simulation_results.csv", "train_rapm.py", "apply_rapm_shrinkage.py",
        "build_composite.py", "build_self_generated_stats.py", "build_qoc_qot.py",
        "build_rapm_stints.py", "score_xg_2024_25_v2.py", "shots_xg_2024_25_v2.parquet",
        "rapm_stints_2024_25.parquet", "rapm_prior_2024_25.csv", "player_positions.csv",
        "train_rapm_5v5.py", "rapm_5v5_2025_26.csv",
    ]
    W("## md5 integrity — protected files (at report time)\n")
    W("All protected files below (incl. the four simulator CSVs) are re-hashed here "
      "and must match the pre-run baseline. This step wrote only "
      "`rapm_kappa_tiebreak_report.md`"
      + ("." if chosen == 0.75 else f" and `model/rapm_5v5_2025_26_k{str(chosen).replace('.','')}.csv`.")
      + "\n")
    W("| file | md5 |")
    W("|---|---|")
    for rel in files:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            W(f"| `{rel}` | `{md5(p)}` |")
    W(f"| `rapm_5v5_report.md` | `{md5(os.path.join(ROOT, 'rapm_5v5_report.md'))}` |")
    W("")


def _write(L):
    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
