"""
train_rapm_special_teams.py
---------------------------
Special-teams RAPM on xG for the 2025-26 regular season. One JOINT model per
stint: estimates POWER-PLAY OFFENSE for the skaters of the team with the man
advantage and PENALTY-KILL DEFENSE for the skaters on the shorthanded team.

Same stint files, same ridge-with-prior machinery (imported from
train_rapm_5v5.py, which is NOT modified), same first/second-half split and same
bootstrap seed and step-down kappa rule as the 5v5 fit. Read-only w.r.t. all
protected files. New outputs only.

Model (identical for every fit):
  * Data: both goalies in net, states 5v4 / 5v3 / 4v3 (4v3 incl. OT). PP team =
    more skaters. Goalie-pulled / null-goalie / counts outside 3-6 excluded.
  * Rows: one per stint, PP team attacking. Target = PP xG/60 = pp_xg/dur*3600.
    Weight = duration seconds. Shorthanded chances are not modeled.
  * Columns: per skater a PP-offense col (+1 for PP-team skaters) and a PK-defense
    col (+1 for PK-team skaters). Unpenalized covariates: intercept, home-PP,
    state dummies (5v3, 4v3; 5v4 ref), score buckets from the PP team's
    perspective (lead2+/lead1/tied[ref]/trail1/trail2+).
  * Output signs: pp_offense = PP xGF/60 above avg (positive good);
    pk_defense = -(raw PK coef) so preventing PP xGA reads positive. No total.
  * Ridge-with-prior exactly as 5v5; players with no 2024-25 ST time in a role
    get prior 0 for that column (falls out of the normal equations).
  * Lambda by 5-fold GroupKFold-by-game weighted MSE, grid = 5v5 grid extended
    one step each way.
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

import train_rapm_5v5 as T
import build_rapm_stints as B

HERE = T.HERE
ROOT = T.ROOT
SEED = T.SEED
NHL_API = T.NHL_API
REPORT = os.path.join(ROOT, "rapm_special_teams_report.md")

COVARS_ST = ["intercept", "home_pp", "state_5v3", "state_4v3",
             "lead2", "lead1", "trail1", "trail2"]
NCOV = len(COVARS_ST)                                   # 8
LAM_GRID = [500] + T.LAM_GRID + [512000]               # one step wider each way
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]
STATES = {(5, 4): "5v4", (5, 3): "5v3", (4, 3): "4v3"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_st(season: str) -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(HERE, B.SEASONS[season]["out"]))
    df = df[df.home_goalie_id.notna() & df.away_goalie_id.notna() & (df.duration_secs > 0)].copy()
    hi = np.maximum(df.home_skaters, df.away_skaters)
    lo = np.minimum(df.home_skaters, df.away_skaters)
    key = list(zip(hi, lo))
    df["state"] = [STATES.get(k) for k in key]
    df = df[df.state.notna()].reset_index(drop=True)
    pp_is_home = (df.home_skaters > df.away_skaters).to_numpy()
    df["pp_is_home"] = pp_is_home
    df["pp_skater_ids"] = np.where(pp_is_home, df.home_skater_ids, df.away_skater_ids)
    df["pk_skater_ids"] = np.where(pp_is_home, df.away_skater_ids, df.home_skater_ids)
    df["pp_xg"] = np.where(pp_is_home, df.home_xg, df.away_xg)
    df["pp_goals"] = np.where(pp_is_home, df.home_goals, df.away_goals)
    df["pp_attempts"] = np.where(pp_is_home, df.home_attempts, df.away_attempts)
    df["sh_xg"] = np.where(pp_is_home, df.away_xg, df.home_xg)
    df["sh_goals"] = np.where(pp_is_home, df.away_goals, df.home_goals)
    df["sh_attempts"] = np.where(pp_is_home, df.away_attempts, df.home_attempts)
    # score from PP team perspective
    df["score_pp"] = np.where(pp_is_home, df.score_diff_home, -df.score_diff_home)
    df["duration"] = df.duration_secs.astype(float)
    return df


def player_universe(df: pd.DataFrame) -> dict:
    ids = set()
    for c in ("pp_skater_ids", "pk_skater_ids"):
        for lst in df[c]:
            ids.update(int(x) for x in lst)
    return {pid: i for i, pid in enumerate(sorted(ids))}


def build_design(df: pd.DataFrame, target: str, player_index: dict):
    N = len(df); P = len(player_index)
    OFF0, DEF0 = NCOV, NCOV + P
    ncol = NCOV + 2 * P
    _map = pd.Series(player_index)
    dur = df.duration.to_numpy()
    fval = (df.pp_xg if target == "xg" else df.pp_goals).to_numpy().astype(float)
    y = fval / dur * 3600.0
    w = dur
    groups = df.game_id.to_numpy()

    rows, cols, vals = [], [], []
    r = np.arange(N)
    rows.append(r); cols.append(np.zeros(N, int)); vals.append(np.ones(N))           # intercept
    hp = np.where(df.pp_is_home.to_numpy())[0]
    rows.append(hp); cols.append(np.ones(len(hp), int)); vals.append(np.ones(len(hp)))  # home_pp
    state = df.state.to_numpy()
    for val, ci in [("5v3", 2), ("4v3", 3)]:
        sel = np.where(state == val)[0]
        rows.append(sel); cols.append(np.full(len(sel), ci)); vals.append(np.ones(len(sel)))
    sp = df.score_pp.to_numpy()
    b = np.full(N, -1)
    b[sp >= 2] = 4; b[sp == 1] = 5; b[sp == -1] = 6; b[sp <= -2] = 7
    sel = b >= 0
    rows.append(r[sel]); cols.append(b[sel]); vals.append(np.ones(sel.sum()))

    def add(id_lists, base):
        counts = np.array([len(x) for x in id_lists])
        flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in id_lists])
        rr = np.repeat(np.arange(N), counts)
        cc = base + _map.reindex(flat).to_numpy().astype(np.int64)
        rows.append(rr); cols.append(cc); vals.append(np.ones(len(flat)))
    add(df.pp_skater_ids.tolist(), OFF0)
    add(df.pk_skater_ids.tolist(), DEF0)

    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    X = sparse.coo_matrix((V, (R, C)), shape=(N, ncol)).tocsr()
    pen = np.zeros(ncol); pen[NCOV:] = 1.0
    layout = dict(OFF0=OFF0, DEF0=DEF0, P=P, ncol=ncol)
    return X, y, w, groups, pen, layout


def st_coefs(beta, player_index, layout):
    fr = T.coefs_to_frame(beta, player_index, layout)      # offense/defense/off_raw/def_raw
    return fr.rename(columns={"offense": "pp_offense", "defense": "pk_defense"})[
        ["player_id", "pp_offense", "pk_defense", "off_raw", "def_raw"]]


def predict_baseline(Xtr, ytr, wtr, Xva):
    A, b = T._normal_eq(Xtr[:, :NCOV], ytr, wtr)
    beta = np.linalg.solve(A + 1e-6 * np.eye(NCOV), b)
    return Xva[:, :NCOV] @ beta


def pp_pk_toi(df):
    pp, pk = defaultdict(float), defaultdict(float)
    for ids, d in zip(df.pp_skater_ids, df.duration):
        for p in ids: pp[int(p)] += d
    for ids, d in zip(df.pk_skater_ids, df.duration):
        for p in ids: pk[int(p)] += d
    return pd.Series(pp, dtype=float), pd.Series(pk, dtype=float)


def player_team_role_toi(df, gteams):
    pp, pk = defaultdict(float), defaultdict(float)
    for row in df.itertuples(index=False):
        ht, at = gteams.get(int(row.game_id), (-1, -1))
        pp_team = ht if row.pp_is_home else at
        pk_team = at if row.pp_is_home else ht
        for p in row.pp_skater_ids: pp[(int(p), pp_team)] += row.duration
        for p in row.pk_skater_ids: pk[(int(p), pk_team)] += row.duration
    return pp, pk


def get_positions(player_ids):
    """Read the protected cache (model/player_positions.csv) read-only; fetch any
    missing players into a SEPARATE supplement file so the protected cache is
    never modified. Returns (pos_map, n_added)."""
    cache = pd.read_csv(os.path.join(HERE, "player_positions.csv"))
    cache["player_id"] = cache.player_id.astype(int)
    supp_path = os.path.join(HERE, "player_positions_st_supplement.csv")
    if os.path.exists(supp_path):
        supp = pd.read_csv(supp_path); supp["player_id"] = supp.player_id.astype(int)
    else:
        supp = pd.DataFrame(columns=["player_id", "position", "full_name"])
    have = set(cache.player_id) | set(supp.player_id)
    need = [int(p) for p in player_ids if int(p) not in have]
    new = []
    for pid in need:
        pos = None; nm = None
        for _ in range(3):
            try:
                req = urllib.request.Request(f"{NHL_API}/player/{pid}/landing",
                                             headers={"User-Agent": "Mozilla/5.0 (analytics-rebuild)"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    d = json.load(r)
                pos = d.get("position"); nm = f"{d['firstName']['default']} {d['lastName']['default']}"
                break
            except Exception:
                time.sleep(1.0)
        new.append({"player_id": pid, "position": pos, "full_name": nm})
    if new:
        supp = pd.concat([supp, pd.DataFrame(new)], ignore_index=True)
        supp.to_csv(supp_path, index=False)
    pos_map = dict(zip(cache.player_id, cache.position))
    pos_map.update(dict(zip(supp.player_id, supp.position)))
    return pos_map, len(need)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(sec):
    sec = int(round(sec)); return f"{sec//60}:{sec%60:02d}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    L = []; W = L.append
    W("# Special-Teams RAPM on xG — 2025-26 Regular Season\n")
    W("Deterministic: `model/train_rapm_special_teams.py` (imports "
      "`train_rapm_5v5.py`; does not modify it). Joint per-stint model: PP offense "
      "for the man-advantage skaters, PK defense for the shorthanded skaters. "
      "Same stint files, machinery, halves, bootstrap seed, and step-down kappa "
      "rule as 5v5. Read-only w.r.t. all protected files.\n")

    names = T.player_names(); teamabbr = T.team_abbrevs()

    # ===== STEP 1: data =====
    W("## Step 1 — Special-teams data\n")
    df = {s: load_st(s) for s in ("2025_26", "2024_25")}
    for s in ("2025_26", "2024_25"):
        d = df[s]
        W(f"**{s.replace('_','-')}** — {len(d):,} ST stints "
          f"({fmt(d.duration.sum())} total).\n")
        W("| state | stints | minutes | PP attempts | PP goals | PP xG | SH attempts | SH goals | SH xG |")
        W("|---|---|---|---|---|---|---|---|---|")
        for st in ["5v4", "5v3", "4v3"]:
            g = d[d.state == st]
            W(f"| {st} | {len(g):,} | {fmt(g.duration.sum())} | {int(g.pp_attempts.sum())} | "
              f"{int(g.pp_goals.sum())} | {g.pp_xg.sum():.1f} | {int(g.sh_attempts.sum())} | "
              f"{int(g.sh_goals.sum())} | {g.sh_xg.sum():.1f} |")
        W(f"| **all** | {len(d):,} | {fmt(d.duration.sum())} | {int(d.pp_attempts.sum())} | "
          f"{int(d.pp_goals.sum())} | {d.pp_xg.sum():.1f} | {int(d.sh_attempts.sum())} | "
          f"{int(d.sh_goals.sum())} | {d.sh_xg.sum():.1f} |")
        W("\n_SH (shorthanded) columns are reference only — not modeled._\n")

    d24, d25 = df["2024_25"], df["2025_26"]

    # ===== STEP 3: prior (kappa 0 on 2024-25) =====
    W("## Step 3 — 2024-25 prior (plain ridge, kappa=0)\n")
    pidx24 = player_universe(d24)
    X24, y24, w24, g24, pen24, lay24 = build_design(d24, "xg", pidx24)
    lam24, _ = T.cv_lambda(X24, y24, w24, g24, pen24, LAM_GRID, np.zeros(lay24["ncol"]))
    beta24 = T.fit_full(X24, y24, w24, pen24, lam24, np.zeros(lay24["ncol"]))
    fr24 = st_coefs(beta24, pidx24, lay24)
    prior_raw = {int(r.player_id): (r.off_raw, r.def_raw) for r in fr24.itertuples(index=False)}
    pp24, pk24 = pp_pk_toi(d24)
    # goals prior
    Xg24, yg24, wg24, gg24, peng24, layg24 = build_design(d24, "goals", pidx24)
    lamg24, _ = T.cv_lambda(Xg24, yg24, wg24, gg24, peng24, LAM_GRID, np.zeros(layg24["ncol"]))
    betag24 = T.fit_full(Xg24, yg24, wg24, peng24, lamg24, np.zeros(layg24["ncol"]))
    prior_raw_goals = {int(r.player_id): (r.off_raw, r.def_raw)
                       for r in st_coefs(betag24, pidx24, layg24).itertuples(index=False)}

    prior_out = fr24[["player_id", "pp_offense", "pk_defense"]].copy()
    prior_out["name"] = prior_out.player_id.map(names)
    allids_prior = set(pidx24)
    pos_map, _ = get_positions(sorted(allids_prior | set(player_universe(d25))))
    posf = lambda p: pos_map.get(int(p)) if isinstance(pos_map.get(int(p)), str) else None
    prior_out["position"] = prior_out.player_id.map(posf)
    prior_out["pp_toi_min"] = (prior_out.player_id.map(pp24).fillna(0) / 60).round(1)
    prior_out["pk_toi_min"] = (prior_out.player_id.map(pk24).fillna(0) / 60).round(1)
    prior_out = prior_out[["player_id", "name", "position", "pp_toi_min", "pk_toi_min",
                           "pp_offense", "pk_defense"]]
    prior_out.to_csv(os.path.join(HERE, "rapm_st_prior_2024_25.csv"), index=False)
    W(f"- 2024-25 ST stints: {len(d24):,}, players: {len(pidx24)}. "
      f"Chosen lambda (xG prior): **{lam24:,}** (goals prior: {lamg24:,}).")
    W(f"- Saved `model/rapm_st_prior_2024_25.csv`.\n")
    toppp = prior_out[prior_out.pp_toi_min >= 100].sort_values("pp_offense", ascending=False).head(10)
    W("**2024-25 top 10 PP offense (≥100 PP min) — sanity only:**\n")
    W("| # | player | pos | PP min | pp_offense |")
    W("|---|---|---|---|---|")
    for i, r in enumerate(toppp.itertuples(index=False), 1):
        W(f"| {i} | {r.name} | {r.position} | {r.pp_toi_min:.0f} | {r.pp_offense:+.2f} |")
    toppk = prior_out[prior_out.pk_toi_min >= 100].sort_values("pk_defense", ascending=False).head(10)
    W("\n**2024-25 top 10 PK defense (≥100 PK min) — sanity only:**\n")
    W("| # | player | pos | PK min | pk_defense |")
    W("|---|---|---|---|---|")
    for i, r in enumerate(toppk.itertuples(index=False), 1):
        W(f"| {i} | {r.name} | {r.position} | {r.pk_toi_min:.0f} | {r.pk_defense:+.2f} |")
    W("")

    # ===== STEP 4: OOS kappa test =====
    W("## Step 4 — Out-of-sample kappa test\n")
    pidx25 = player_universe(d25)
    X25, y25, w25, g25, pen25, lay25 = build_design(d25, "xg", pidx25)
    # SAME halves as the 5v5 test
    dates = T.game_dates("2025_26")
    g5 = T.load_5v5("2025_26").game_id.unique()
    gorder = sorted(g5, key=lambda g: (dates.get(int(g), ""), g))
    half = len(gorder) // 2
    gamesA = set(gorder[:half]); gamesB = set(gorder[half:])
    maskA = np.isin(g25, list(gamesA)); maskB = ~maskA
    W(f"- Same split as 5v5: first half {len(gamesA)} games, second half {len(gamesB)}. "
      f"ST rows: half A {int(maskA.sum()):,}, half B {int(maskB.sum()):,}.")

    base_ab = float(np.sum(w25[maskB]*(y25[maskB]-predict_baseline(X25[maskA], y25[maskA], w25[maskA], X25[maskB]))**2)/np.sum(w25[maskB]))
    base_ba = float(np.sum(w25[maskA]*(y25[maskA]-predict_baseline(X25[maskB], y25[maskB], w25[maskB], X25[maskA]))**2)/np.sum(w25[maskA]))
    W(f"- Baseline (intercept+covariates only): A→B {base_ab:.3f}, B→A {base_ba:.3f}, "
      f"avg **{0.5*(base_ab+base_ba):.3f}**.\n")

    results, fits, err = {}, {}, {}
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
        print(f"kappa={kappa}: lamA={lamA} lamB={lamB} avg={results[kappa]['avg']:.4f}", flush=True)

    W("| kappa | lambda A | lambda B | MSE A→B | MSE B→A | avg MSE |")
    W("|---|---|---|---|---|---|")
    for k in KAPPAS:
        r = results[k]
        W(f"| {k} | {r['lamA']:,} | {r['lamB']:,} | {r['ab']:.3f} | {r['ba']:.3f} | **{r['avg']:.4f}** |")
    W("")

    # bootstrap (same seed, one resample/iter, pooled both directions)
    games = np.array(sorted(d25.game_id.unique())); nG = len(games)
    def per_game(e):
        dfe = pd.DataFrame({"g": e["game"], "w": e["w"], "sq": e["sq"]})
        dfe["wse"] = dfe.w*dfe.sq
        return dfe.groupby("g").agg(sw=("w", "sum"), wse=("wse", "sum")).reindex(games)
    pg = {k: per_game(err[k]) for k in KAPPAS}
    sw = pg[0.0]["sw"].to_numpy()
    wse = {k: pg[k]["wse"].to_numpy() for k in KAPPAS}
    rng = np.random.RandomState(SEED)
    boot = {k: np.empty(1000) for k in KAPPAS}
    for i in range(1000):
        idx = rng.randint(0, nG, nG); den = sw[idx].sum()
        for k in KAPPAS:
            boot[k][i] = wse[k][idx].sum()/den
    def ci(a, b):
        d = boot[a]-boot[b]
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    pairs = [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0),
             (0.5, 0.25), (0.75, 0.5), (1.0, 0.75)]
    W(f"**Bootstrap 95% CIs** (1,000 resamples, seed {SEED}, both directions pooled; "
      f"{nG} ST games). Negative ⇒ the larger kappa has lower MSE.\n")
    W("| difference | mean | 95% CI |")
    W("|---|---|---|")
    ci_store = {}
    for a, b in pairs:
        m, lo, hi = ci(a, b); ci_store[(a, b)] = (m, lo, hi)
        W(f"| {a} − {b} | {m:+.5f} | [{lo:+.5f}, {hi:+.5f}] |")
    W("")

    # selection walk
    W("### Selection walk\n")
    order = KAPPAS[:]
    pick = min(order, key=lambda k: results[k]["avg"])
    W(f"- Start at lowest-avg-MSE kappa = **{pick}** (avg {results[pick]['avg']:.4f}).")
    while True:
        i = order.index(pick)
        if i == 0:
            W("- Reached kappa 0."); break
        smaller = order[i-1]
        m, lo, hi = ci(pick, smaller)
        if hi < 0:
            W(f"- {pick} vs {smaller}: CI [{lo:+.5f},{hi:+.5f}] excludes 0 "
              f"({smaller} worse) ⇒ **stop at {pick}**."); break
        elif lo <= 0 <= hi:
            W(f"- {pick} vs {smaller}: CI [{lo:+.5f},{hi:+.5f}] includes 0 ⇒ step down to {smaller}.")
            pick = smaller
        else:
            W(f"- {pick} vs {smaller}: CI [{lo:+.5f},{hi:+.5f}] > 0 ({smaller} better) ⇒ step down to {smaller}.")
            pick = smaller
    if pick != 0.0:
        m, lo, hi = ci(pick, 0.0)
        if hi < 0:
            W(f"- Final check: {pick} − 0 CI [{lo:+.5f},{hi:+.5f}] excludes 0 ⇒ **{pick} beats kappa 0**.")
            chosen = pick
        else:
            W(f"- Final check: {pick} − 0 CI [{lo:+.5f},{hi:+.5f}] includes 0 ⇒ **choose kappa 0**.")
            chosen = 0.0
    else:
        chosen = 0.0
    W(f"\n- **Selected kappa = {chosen}.**\n")

    # reliability
    W("### Reliability decomposition\n")
    ppA, pkA = pp_pk_toi(d25[d25.game_id.isin(gamesA)])
    ppB, pkB = pp_pk_toi(d25[d25.game_id.isin(gamesB)])
    prior_pp = {int(r.player_id): r.pp_offense for r in fr24.itertuples(index=False)}
    prior_pk = {int(r.player_id): r.pk_defense for r in fr24.itertuples(index=False)}

    def reliability(role):
        toiA, toiB = (ppA, ppB) if role == "pp" else (pkA, pkB)
        col = "pp_offense" if role == "pp" else "pk_defense"
        prior_map = prior_pp if role == "pp" else prior_pk
        elig = [p for p in pidx25 if toiA.get(p, 0) >= 3000 and toiB.get(p, 0) >= 3000]  # 50 min
        pr = np.array([prior_map.get(p, 0.0) for p in elig])
        rows = []
        for k in [0.0, chosen] if chosen != 0.0 else [0.0]:
            bA, bB = fits[k]
            vA = st_coefs(bA, pidx25, lay25).set_index("player_id").loc[elig, col].to_numpy()
            vB = st_coefs(bB, pidx25, lay25).set_index("player_id").loc[elig, col].to_numpy()
            a = float(np.corrcoef(vA, vB)[0, 1])
            bb = float(np.corrcoef(vA-k*pr, vB-k*pr)[0, 1])
            c1 = float(np.corrcoef(vA, pr)[0, 1]); c2 = float(np.corrcoef(vB, pr)[0, 1])
            rows.append((k, len(elig), a, bb, c1, c2))
        return rows

    W("| role | kappa | n | (a) half-half | (b) data-driven (−κ·prior) | (c) H1 vs prior | (c) H2 vs prior |")
    W("|---|---|---|---|---|---|---|")
    for role, lab in [("pp", "pp_offense"), ("pk", "pk_defense")]:
        for (k, n, a, bb, c1, c2) in reliability(role):
            W(f"| {lab} | {k} | {n} | {a:.3f} | {bb:.3f} | {c1:.3f} | {c2:.3f} |")
    W("\n_Role minimum: 50 min in each half. Column (b) nets out the shared prior._\n")

    # prior vs full-season kappa0 (>=100 min both seasons)
    lam0full, _ = T.cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, np.zeros(lay25["ncol"]))
    beta0full = T.fit_full(X25, y25, w25, pen25, lam0full, np.zeros(lay25["ncol"]))
    fr0 = st_coefs(beta0full, pidx25, lay25).set_index("player_id")
    pp25, pk25 = pp_pk_toi(d25)
    for role, col, pmap, t24, t25 in [("pp_offense", "pp_offense", prior_pp, pp24, pp25),
                                      ("pk_defense", "pk_defense", prior_pk, pk24, pk25)]:
        both = [p for p in pidx25 if t24.get(p, 0) >= 6000 and t25.get(p, 0) >= 6000 and p in pmap]
        pv = np.array([pmap[p] for p in both]); cv = fr0.loc[both, col].to_numpy()
        c = float(np.corrcoef(pv, cv)[0, 1])
        W(f"- Corr(2024-25 prior {role}, 2025-26 κ=0 full-season {role}), ≥100 min both "
          f"seasons (n={len(both)}): **{c:.3f}**.")
    W("")

    # ===== STEP 5: final fit + outputs =====
    W("## Step 5 — Final fit & outputs\n")
    po_final = T.build_prior_offset(pidx25, lay25, prior_raw, chosen)
    lamF, _ = T.cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, po_final)
    betaF = T.fit_full(X25, y25, w25, pen25, lamF, po_final)
    frF = st_coefs(betaF, pidx25, lay25)
    # goals version
    Xg25, yg25, wg25, gg25, peng25, layg25 = build_design(d25, "goals", pidx25)
    po_goals = T.build_prior_offset(pidx25, layg25, prior_raw_goals, chosen)
    frGF = st_coefs(T.fit_full(Xg25, yg25, wg25, peng25, lamF, po_goals), pidx25, layg25).set_index("player_id")

    pos_map, n_added = get_positions(sorted(pidx25))
    W(f"- Positions: read `model/player_positions.csv` (read-only); **{n_added}** missing "
      f"players fetched into `model/player_positions_st_supplement.csv` (protected cache "
      f"untouched).")
    W(f"- Chosen kappa **{chosen}**, full-season lambda **{lamF:,}** "
      f"(κ=0 full-season lambda {lam0full:,}; goals version lambda {lamF:,}).")

    gteams = T.game_teams("2025_26")
    pp_tt, pk_tt = player_team_role_toi(d25, gteams)
    # primary team by combined ST toi
    comb = defaultdict(float)
    for (p, t), s in list(pp_tt.items()) + list(pk_tt.items()):
        comb[(p, t)] += s
    comb_df = pd.DataFrame([(p, t, s) for (p, t), s in comb.items()], columns=["player_id", "team_id", "toi"])
    prim = comb_df.sort_values("toi", ascending=False).drop_duplicates("player_id").set_index("player_id")["team_id"]
    teamlist = comb_df.sort_values("toi", ascending=False).groupby("player_id")["team_id"].apply(
        lambda s: "/".join(teamabbr.get(int(t), str(t)) for t in s))

    out = frF[["player_id", "pp_offense", "pk_defense"]].copy()
    out["name"] = out.player_id.map(names); out["position"] = out.player_id.map(lambda p: pos_map.get(int(p)) if isinstance(pos_map.get(int(p)), str) else None)
    out["teams"] = out.player_id.map(teamlist)
    out["pp_toi_min"] = (out.player_id.map(pp25).fillna(0)/60).round(1)
    out["pk_toi_min"] = (out.player_id.map(pk25).fillna(0)/60).round(1)
    out["pp_offense_impact"] = out.pp_offense * out.player_id.map(pp25).fillna(0)/3600.0
    out["pk_defense_impact"] = out.pk_defense * out.player_id.map(pk25).fillna(0)/3600.0
    out["prior_pp_offense"] = out.player_id.map(lambda p: prior_raw.get(int(p), (0.0, 0.0))[0])
    out["prior_pk_defense"] = out.player_id.map(lambda p: -prior_raw.get(int(p), (0.0, 0.0))[1])
    out["kappa"] = chosen; out["lambda"] = lamF
    out["goals_pp_offense"] = out.player_id.map(frGF.pp_offense)
    out["goals_pk_defense"] = out.player_id.map(frGF.pk_defense)
    out = out[["player_id", "name", "position", "teams", "pp_toi_min", "pp_offense",
               "pp_offense_impact", "pk_toi_min", "pk_defense", "pk_defense_impact",
               "prior_pp_offense", "prior_pk_defense", "kappa", "lambda",
               "goals_pp_offense", "goals_pk_defense"]]
    out.to_csv(os.path.join(HERE, "rapm_st_2025_26.csv"), index=False)
    W(f"- Saved `model/rapm_st_2025_26.csv` ({len(out)} players).")

    pp_disp = out[out.pp_toi_min >= 100].copy()
    pk_disp = out[out.pk_toi_min >= 100].copy()
    cgp = float(np.corrcoef(pp_disp.pp_offense, pp_disp.goals_pp_offense)[0, 1])
    cgk = float(np.corrcoef(pk_disp.pk_defense, pk_disp.goals_pk_defense)[0, 1])
    W(f"- xG vs goals corr: pp_offense (≥100 PP min, n={len(pp_disp)}) **{cgp:.3f}**; "
      f"pk_defense (≥100 PK min, n={len(pk_disp)}) **{cgk:.3f}**.\n")

    # ---- sanity: five players ----
    W("### Sanity — five players (κ=0 beside)\n")
    fr0pp = fr0["pp_offense"]; fr0pk = fr0["pk_defense"]
    pp_rank = {int(r.player_id): i+1 for i, r in pp_disp.sort_values("pp_offense", ascending=False).reset_index(drop=True).iterrows()}
    pk_rank = {int(r.player_id): i+1 for i, r in pk_disp.sort_values("pk_defense", ascending=False).reset_index(drop=True).iterrows()}
    W("| player | pos | PP min | pp_off | pp_impact | PP rank | κ0 pp_off | PK min | pk_def | pk_impact | PK rank | κ0 pk_def |")
    W("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, pid in {"McDavid": 8478402, "Kucherov": 8476453, "MacKinnon": 8477492,
                    "Brady Tkachuk": 8480801, "Mark Stone": 8475913}.items():
        r = out[out.player_id == pid].iloc[0]
        ppr = pp_rank.get(pid, "below min"); pkr = pk_rank.get(pid, "below min")
        k0pp = fr0pp.get(pid, float("nan")); k0pk = fr0pk.get(pid, float("nan"))
        W(f"| {nm} | {r.position} | {r.pp_toi_min:.0f} | {r.pp_offense:+.2f} | {r.pp_offense_impact:+.2f} | "
          f"{ppr} | {k0pp:+.2f} | {r.pk_toi_min:.0f} | {r.pk_defense:+.2f} | {r.pk_defense_impact:+.2f} | "
          f"{pkr} | {k0pk:+.2f} |")
    W("")

    def tbl(title, dfx, col, asc, n):
        W(f"**{title}**\n")
        W("| # | player | pos | team | role min | value |")
        W("|---|---|---|---|---|---|")
        mincol = "pp_toi_min" if col == "pp_offense" else "pk_toi_min"
        for i, r in enumerate(dfx.sort_values(col, ascending=asc).head(n).itertuples(index=False), 1):
            W(f"| {i} | {r.name} | {r.position} | {r.teams} | {getattr(r,mincol):.0f} | {getattr(r,col):+.2f} |")
        W("")
    W("### Sanity — leaderboards\n")
    tbl("Top 20 pp_offense (≥100 PP min)", pp_disp, "pp_offense", False, 20)
    tbl("Bottom 10 pp_offense (≥100 PP min)", pp_disp, "pp_offense", True, 10)
    tbl("Top 20 pk_defense (≥100 PK min)", pk_disp, "pk_defense", False, 20)
    tbl("Bottom 10 pk_defense (≥100 PK min)", pk_disp, "pk_defense", True, 10)

    # team clustering
    def clustering(dfx, col):
        top = dfx.sort_values(col, ascending=False).head(20)
        ab = top.player_id.map(lambda p: teamabbr.get(int(prim.get(p, -1)), "?"))
        vc = ab.value_counts()
        return ab.nunique(), int(vc.iloc[0]), vc.index[0]
    nd_pp, mx_pp, tm_pp = clustering(pp_disp, "pp_offense")
    nd_pk, mx_pk, tm_pk = clustering(pk_disp, "pk_defense")
    W("### Sanity — team clustering (top 20)\n")
    W(f"- pp_offense: distinct teams **{nd_pp}**, most from one team **{mx_pp}** ({tm_pp}).")
    W(f"- pk_defense: distinct teams **{nd_pk}**, most from one team **{mx_pk}** ({tm_pk}).\n")

    # team reconciliation
    team_ppxgf = defaultdict(float); team_pptoi = defaultdict(float)
    team_pkxga = defaultdict(float); team_pktoi = defaultdict(float)
    for row in d25.itertuples(index=False):
        ht, at = gteams.get(int(row.game_id), (-1, -1))
        ppt = ht if row.pp_is_home else at
        pkt = at if row.pp_is_home else ht
        team_ppxgf[ppt] += row.pp_xg; team_pptoi[ppt] += row.duration
        team_pkxga[pkt] += row.pp_xg; team_pktoi[pkt] += row.duration
    league_pp = sum(team_ppxgf.values())/sum(team_pptoi.values())*3600
    league_pk = sum(team_pkxga.values())/sum(team_pktoi.values())*3600
    ppoff_map = out.set_index("player_id")["pp_offense"]; pkdef_map = out.set_index("player_id")["pk_defense"]
    pred_pp = defaultdict(float); pred_pk = defaultdict(float)
    for (p, t), s in pp_tt.items(): pred_pp[t] += ppoff_map.get(int(p), 0.0)*s
    for (p, t), s in pk_tt.items(): pred_pk[t] += pkdef_map.get(int(p), 0.0)*s
    tpp = [t for t in team_pptoi if t >= 0 and t in pred_pp]
    act_pp = np.array([team_ppxgf[t]/team_pptoi[t]*3600 - league_pp for t in tpp])
    prd_pp = np.array([pred_pp[t]/team_pptoi[t] for t in tpp])
    tpk = [t for t in team_pktoi if t >= 0 and t in pred_pk]
    act_pk = np.array([-(team_pkxga[t]/team_pktoi[t]*3600 - league_pk) for t in tpk])  # sign flipped: higher=better
    prd_pk = np.array([pred_pk[t]/team_pktoi[t] for t in tpk])
    cpp = float(np.corrcoef(act_pp, prd_pp)[0, 1]); cpk = float(np.corrcoef(act_pk, prd_pk)[0, 1])
    W("### Sanity — team reconciliation\n")
    W(f"- pp_offense vs team PP xGF/60 − league ({len(tpp)} teams): **{cpp:.3f}**.")
    W(f"- pk_defense vs team PK xGA/60 (sign-flipped) ({len(tpk)} teams): **{cpk:.3f}**.\n")

    # distributions
    W("### Sanity — distributions (by position, role minimums)\n")
    W("| role | group | n | mean | std | min | max |")
    W("|---|---|---|---|---|---|---|")
    for role, dfx, col in [("pp_offense", pp_disp, "pp_offense"), ("pk_defense", pk_disp, "pk_defense")]:
        dfx = dfx.copy()
        dfx["grp"] = dfx.position.map(lambda p: "D" if p == "D" else ("F" if p in ("C", "L", "R") else None))
        for grp in ["F", "D"]:
            v = dfx[dfx.grp == grp][col]
            W(f"| {role} | {grp} | {len(v)} | {v.mean():+.2f} | {v.std():.2f} | {v.min():+.2f} | {v.max():+.2f} |")
    W("")

    # ===== Step 4 lambda table + edges =====
    W("## Lambda grid & chosen lambdas\n")
    W(f"- Grid: `{LAM_GRID}` (5v5 grid extended one step each way). Edges: {LAM_GRID[0]}, {LAM_GRID[-1]}.")
    def edge(l): return " ⚠️EDGE" if l in (LAM_GRID[0], LAM_GRID[-1]) else ""
    W("\n| fit | lambda |")
    W("|---|---|")
    W(f"| 2024-25 prior (xG) | {lam24:,}{edge(lam24)} |")
    W(f"| 2024-25 prior (goals) | {lamg24:,}{edge(lamg24)} |")
    for k in KAPPAS:
        W(f"| half A (κ={k}) | {results[k]['lamA']:,}{edge(results[k]['lamA'])} |")
        W(f"| half B (κ={k}) | {results[k]['lamB']:,}{edge(results[k]['lamB'])} |")
    W(f"| full season κ=0 | {lam0full:,}{edge(lam0full)} |")
    W(f"| full season κ={chosen} (final) | {lamF:,}{edge(lamF)} |")
    all_l = [lam24, lamg24, lam0full, lamF] + [results[k]['lamA'] for k in KAPPAS] + [results[k]['lamB'] for k in KAPPAS]
    W("\n" + ("_No lambda at a grid edge._" if not any(l in (LAM_GRID[0], LAM_GRID[-1]) for l in all_l)
             else "_See ⚠️EDGE flags._") + "\n")

    # ===== md5 =====
    _md5_section(W, n_added)

    # ===== choices =====
    W("## Choices I made that were not fully specified\n")
    W("- **Lambda grid** = 5v5 grid extended one step each way: "
      f"`{LAM_GRID}`. Chosen by GroupKFold-by-game weighted MSE.")
    W("- **player_positions.csv is in the protected do-not-modify list**, yet Step 5 "
      "asks to append missing players. I reconciled by treating the cache as "
      "read-only and writing any newly fetched players to a separate "
      "`model/player_positions_st_supplement.csv` (same UA header/retry as 5v5), "
      "combined in memory. Count added is reported. Unknown positions never defaulted.")
    W("- **Score buckets** capped at ±2 from the PP team's perspective, tied = reference.")
    W("- **Goals version** uses the chosen kappa and the final xG lambda, with a "
      "goals-based 2024-25 ST prior built identically (used only when kappa>0).")
    W("- **Team reconciliation** predicted value = Σ(player role-rating × player-team "
      "role TOI) / team role TOI; actual PP = team PP xGF/60 − league; actual PK = "
      "−(team PK xGA/60 − league) so positive is better, matching pk_defense's sign.")
    W("- **Primary team / team list** ranked by combined (PP+PK) special-teams TOI.")

    with open(REPORT, "w") as f:
        f.write("\n".join(L) + "\n")
    print("DONE special teams | chosen", chosen, "| lamF", lamF, "| added", n_added)


def _md5_section(W, n_added):
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
        "train_rapm_5v5.py", "rapm_kappa_tiebreak.py", "rapm_5v5_2025_26.csv",
    ]
    W("## md5 integrity — protected files (at report time)\n")
    W(f"All protected files re-hashed; must match the pre-run baseline (incl. "
      f"`player_positions.csv`, untouched — {n_added} new players went to the "
      f"supplement). New outputs this step: `model/rapm_st_prior_2024_25.csv`, "
      f"`model/rapm_st_2025_26.csv`, `model/train_rapm_special_teams.py`, "
      f"`rapm_special_teams_report.md`"
      + (", `model/player_positions_st_supplement.csv`" if n_added else "") + ".\n")
    W("| file | md5 |")
    W("|---|---|")
    for rel in files:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            W(f"| `{rel}` | `{md5(p)}` |")
    for rel in ["rapm_5v5_report.md", "rapm_kappa_tiebreak_report.md"]:
        W(f"| `{rel}` | `{md5(os.path.join(ROOT, rel))}` |")
    W("")


if __name__ == "__main__":
    main()
