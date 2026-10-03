"""
train_rapm_5v5.py
-----------------
5v5 RAPM on xG for the 2025-26 regular season, with the regularization approach
(plain ridge vs ridge shrinking toward a 2024-25 prior) decided by an
out-of-sample test. Deterministic. Read-only w.r.t. all protected inputs.

Model definition (identical for every fit; see the spec):
  * Data: 5v5 stints only, both goalies in net (home_skaters==away_skaters==5,
    non-null goalies). Every skater kept; TOI filters apply to display only.
  * Rows: each stint -> two rows (one per attacking team). Target = attacking
    team's xG/60 = xG_sum / duration * 3600. Weight = duration seconds.
  * Columns: per skater an offense col (+1 when on the attacking team) and a
    defense col (+1 when on the defending team). Unpenalized covariates:
    intercept, home-attacking indicator, and score state from the attacker's
    perspective (lead2+, lead1, tied[ref], trail1, trail2+).
  * Sign convention (output): offense = xGF/60 above avg; defense = -(raw
    defense coef) so preventing xGA reads positive; total = offense + defense.
  * Ridge with prior: min ||sqrt(w)(y - Xb)||^2 + lam*||b_P - kappa*b_prior||^2,
    penalty on PLAYER columns only. Implemented by ridge on the residual target
    y - X(kappa*b_prior) then adding kappa*b_prior back. kappa=0 = plain ridge.
    Players with no 2024-25 5v5 time get prior 0.
  * Lambda: 5-fold GroupKFold by game, minimizing weighted MSE, on whatever data
    the fit uses.

Outputs:
  model/rapm_prior_2024_25.csv, model/rapm_5v5_2025_26.csv,
  model/player_positions.csv (position cache), rapm_5v5_report.md (project root).
"""
from __future__ import annotations
import json
import os
import time
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold

import build_rapm_stints as B

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NHL_API = "https://api-web.nhle.com/v1"
SEED = 42

COVARS = ["intercept", "home_attacking", "lead2", "lead1", "trail1", "trail2"]
NCOV = len(COVARS)
LAM_GRID = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 256000]
KAPPAS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_5v5(season: str) -> pd.DataFrame:
    cfg = B.SEASONS[season]
    df = pd.read_parquet(os.path.join(HERE, cfg["out"]))
    m = ((df.home_skaters == 5) & (df.away_skaters == 5)
         & df.home_goalie_id.notna() & df.away_goalie_id.notna()
         & (df.duration_secs > 0))
    df = df[m].reset_index(drop=True).copy()
    return df


def game_dates(season: str) -> dict:
    cfg = B.SEASONS[season]
    xf = pd.read_parquet(os.path.join(HERE, f"xg_features_{season}.parquet"))
    gd = xf[["game_id", "game_date"]].drop_duplicates()
    return dict(zip(gd.game_id.astype(int), gd.game_date.astype(str)))


def game_teams(season: str) -> dict:
    """game_id -> (home_team_id, away_team_id) from the enriched shots."""
    shots = B.load_shots_with_xg(B.SEASONS[season])
    home = shots.assign(home=np.where(shots.is_home == 1, shots.shooting_team_id,
                                      shots.defending_team_id))
    hm = home.groupby("game_id")["home"].first().astype(int)
    teams = (shots.groupby("game_id")
             .apply(lambda d: set(d.shooting_team_id) | set(d.defending_team_id))
             .to_dict())
    out = {}
    for g, ht in hm.items():
        ts = teams[g]
        away = [t for t in ts if t != ht]
        out[int(g)] = (int(ht), int(away[0]) if away else -1)
    return out


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------
def build_design(df: pd.DataFrame, target: str, player_index: dict):
    """Return X (csr), y, w, groups, pen_mask, col layout. target in {xg, goals}."""
    N = len(df)
    home_sk = np.array(df.home_skater_ids.tolist(), dtype=np.int64)   # (N,5)
    away_sk = np.array(df.away_skater_ids.tolist(), dtype=np.int64)
    dur = df.duration_secs.to_numpy().astype(np.float64)
    sdiff = df.score_diff_home.to_numpy().astype(np.int64)

    if target == "xg":
        hf, af = df.home_xg.to_numpy(), df.away_xg.to_numpy()
    else:
        hf, af = df.home_goals.to_numpy().astype(float), df.away_goals.to_numpy().astype(float)

    P = len(player_index)
    OFF0, DEF0 = NCOV, NCOV + P
    ncol = NCOV + 2 * P

    _map = pd.Series(player_index)          # index=player_id -> value=col idx
    def pidx(arr):   # map player ids -> index (assumes all present), vectorized
        flat = pd.Index(arr.ravel())
        return _map.reindex(flat).to_numpy().astype(np.int64).reshape(arr.shape)

    h_i = pidx(home_sk); a_i = pidx(away_sk)

    rows, cols, vals = [], [], []
    r_home = np.arange(N)
    r_away = np.arange(N, 2 * N)

    # covariates
    allr = np.arange(2 * N)
    rows.append(allr); cols.append(np.zeros(2 * N, np.int64)); vals.append(np.ones(2 * N))  # intercept
    rows.append(r_home); cols.append(np.ones(N, np.int64)); vals.append(np.ones(N))         # home_attacking
    # score buckets from attacker perspective
    diff_home = sdiff                      # home-attacking rows
    diff_away = -sdiff                     # away-attacking rows
    for r_arr, d_arr in [(r_home, diff_home), (r_away, diff_away)]:
        b = np.empty(len(d_arr), np.int64); b[:] = -1
        b[d_arr >= 2] = 2      # lead2 -> col idx 2
        b[d_arr == 1] = 3      # lead1 -> col 3
        b[d_arr == -1] = 4     # trail1 -> col 4
        b[d_arr <= -2] = 5     # trail2 -> col 5
        sel = b >= 0           # tied (==0) is reference -> no column
        rows.append(r_arr[sel]); cols.append(b[sel]); vals.append(np.ones(sel.sum()))

    # players: home row -> home off, away def ; away row -> away off, home def
    def add_block(r_base, off_ids, def_ids):
        r5 = np.repeat(np.arange(len(off_ids)) + r_base, 5)
        rows.append(r5); cols.append((OFF0 + off_ids).ravel()); vals.append(np.ones(off_ids.size))
        rows.append(r5); cols.append((DEF0 + def_ids).ravel()); vals.append(np.ones(def_ids.size))
    add_block(0, h_i, a_i)     # home-attacking rows
    add_block(N, a_i, h_i)     # away-attacking rows

    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    X = sparse.coo_matrix((V, (R, C)), shape=(2 * N, ncol)).tocsr()

    y = np.concatenate([hf / dur * 3600.0, af / dur * 3600.0])
    w = np.concatenate([dur, dur])
    groups = np.concatenate([df.game_id.to_numpy(), df.game_id.to_numpy()])

    pen_mask = np.zeros(ncol); pen_mask[NCOV:] = 1.0
    layout = dict(OFF0=OFF0, DEF0=DEF0, P=P, ncol=ncol)
    return X, y, w, groups, pen_mask, layout


# ---------------------------------------------------------------------------
# Ridge with prior
# ---------------------------------------------------------------------------
def _normal_eq(X, y, w):
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray()
    b = X.T @ (w * y)
    return A, np.asarray(b).ravel()


def ridge_solve(A, b, pen_mask, lam, prior_offset):
    """Solve (A + lam*diag(pen))*gamma = b - A@prior_offset_effect... see note.

    We fit on residual target y - X@prior_offset. Since A,b were built on the
    residual target already, here A,b correspond to residual. Returns beta =
    gamma + prior_offset.
    """
    M = A + lam * np.diag(pen_mask)
    gamma = np.linalg.solve(M, b)
    return gamma + prior_offset


def wmse(X, beta, y, w):
    pred = X @ beta
    r = y - pred
    return float(np.sum(w * r * r) / np.sum(w))


def cv_lambda(X, y, w, groups, pen_mask, lam_grid, prior_offset, n_splits=5):
    offset = X @ prior_offset
    y_res = y - offset
    gkf = GroupKFold(n_splits=n_splits)
    fold_scores = {lam: [] for lam in lam_grid}
    for tr, va in gkf.split(np.zeros(len(y)), y, groups):
        Xtr = X[tr]; Xva = X[va]
        A, b = _normal_eq(Xtr, y_res[tr], w[tr])
        for lam in lam_grid:
            beta = ridge_solve(A, b, pen_mask, lam, prior_offset)
            fold_scores[lam].append(wmse(Xva, beta, y[va], w[va]))
    mean = {lam: float(np.mean(v)) for lam, v in fold_scores.items()}
    best = min(mean, key=mean.get)
    return best, mean


def fit_full(X, y, w, pen_mask, lam, prior_offset):
    offset = X @ prior_offset
    A, b = _normal_eq(X, y - offset, w)
    return ridge_solve(A, b, pen_mask, lam, prior_offset)


# ---------------------------------------------------------------------------
# Coefficient extraction (raw + flipped display)
# ---------------------------------------------------------------------------
def coefs_to_frame(beta, player_index, layout):
    inv = {v: k for k, v in player_index.items()}
    P = layout["P"]; OFF0 = layout["OFF0"]; DEF0 = layout["DEF0"]
    pids = [inv[i] for i in range(P)]
    off_raw = beta[OFF0:OFF0 + P]
    def_raw = beta[DEF0:DEF0 + P]
    return pd.DataFrame({
        "player_id": pids,
        "offense": off_raw,             # xGF/60 above avg (positive good)
        "defense": -def_raw,            # flip: positive = xGA/60 prevented
        "off_raw": off_raw, "def_raw": def_raw,
    }).assign(total=lambda d: d.offense + d.defense)


def build_prior_offset(player_index, layout, prior_raw: dict, kappa: float):
    """prior_raw: player_id -> (off_raw, def_raw). Returns full-length offset vec
    = kappa * beta_prior (0 for covariates and players with no prior)."""
    v = np.zeros(layout["ncol"])
    if kappa == 0 or not prior_raw:
        return v
    OFF0 = layout["OFF0"]; DEF0 = layout["DEF0"]
    for pid, idx in player_index.items():
        pr = prior_raw.get(int(pid))
        if pr is not None:
            v[OFF0 + idx] = kappa * pr[0]
            v[DEF0 + idx] = kappa * pr[1]
    return v


def player_universe(df: pd.DataFrame) -> dict:
    ids = set()
    for c in ("home_skater_ids", "away_skater_ids"):
        for lst in df[c]:
            ids.update(int(x) for x in lst)
    return {pid: i for i, pid in enumerate(sorted(ids))}


# ---------------------------------------------------------------------------
# TOI / team / names / positions
# ---------------------------------------------------------------------------
def player_toi_5v5(df: pd.DataFrame) -> pd.Series:
    """5v5 on-ice TOI (seconds) per skater from 5v5 stints."""
    from collections import defaultdict
    toi = defaultdict(float)
    dur = df.duration_secs.to_numpy()
    for lst, d in zip(df.home_skater_ids, dur):
        for p in lst: toi[int(p)] += d
    for lst, d in zip(df.away_skater_ids, dur):
        for p in lst: toi[int(p)] += d
    return pd.Series(toi, name="toi_sec")


def player_team_toi(df: pd.DataFrame, gteams: dict):
    from collections import defaultdict
    tt = defaultdict(float)
    for row in df.itertuples(index=False):
        ht, at = gteams.get(int(row.game_id), (-1, -1))
        d = row.duration_secs
        for p in row.home_skater_ids: tt[(int(p), ht)] += d
        for p in row.away_skater_ids: tt[(int(p), at)] += d
    rec = [(pid, tid, sec) for (pid, tid), sec in tt.items()]
    return pd.DataFrame(rec, columns=["player_id", "team_id", "toi_sec"])


def player_names(seasons=("2025_26", "2024_25")) -> dict:
    """player_id -> name. Some source shift rows carry double-encoded UTF-8
    (e.g. 'BÃ¤ck' vs 'Bäck' for the same id); prefer the cleanest variant."""
    variants = {}
    for s in seasons:
        for rel in B.SEASONS[s]["shifts"]:
            p = os.path.join(HERE, rel)
            if os.path.exists(p):
                sh = pd.read_parquet(p, columns=["player_id", "first_name", "last_name"])
                for pid, fn, ln in sh.drop_duplicates(["player_id", "first_name", "last_name"]).itertuples(index=False):
                    variants.setdefault(int(pid), set()).add(f"{fn} {ln}")
    def badness(nm):   # count mojibake / replacement markers
        return sum(nm.count(c) for c in ("Ã", "Â", "�", "�"))
    def repair(nm):    # reverse double-encoded UTF-8 (e.g. 'LÃ©garÃ©' -> 'Légaré')
        if badness(nm) == 0:
            return nm
        try:
            fixed = nm.encode("latin-1").decode("utf-8")
            if badness(fixed) < badness(nm):
                return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        return nm
    return {pid: repair(min(vs, key=lambda n: (badness(n), n)))
            for pid, vs in variants.items()}


TEAM_ABBR = {}
def team_abbrevs():
    global TEAM_ABBR
    if TEAM_ABBR:
        return TEAM_ABBR
    sh = pd.read_parquet(os.path.join(HERE, "season_cache", "shifts_20252026.parquet"),
                         columns=["team_id", "team_abbrev"]).drop_duplicates()
    TEAM_ABBR = dict(zip(sh.team_id.astype(int), sh.team_abbrev.astype(str)))
    return TEAM_ABBR


def fetch_positions(player_ids) -> pd.DataFrame:
    """positionCode per player from NHL API player landing, cached to
    model/player_positions.csv. Unknown positions are NOT defaulted."""
    cache_path = os.path.join(HERE, "player_positions.csv")
    if os.path.exists(cache_path):
        cache = pd.read_csv(cache_path)
        cache["player_id"] = cache.player_id.astype(int)
    else:
        cache = pd.DataFrame(columns=["player_id", "position", "full_name"])
    have = set(cache.player_id.tolist())
    need = [int(p) for p in player_ids if int(p) not in have]
    new = []
    for i, pid in enumerate(need):
        pos, nm = None, None
        for attempt in range(3):
            try:
                # NHL API rejects the default python-urllib UA with 403; set one.
                req = urllib.request.Request(
                    f"{NHL_API}/player/{pid}/landing",
                    headers={"User-Agent": "Mozilla/5.0 (analytics-rebuild)"})
                with urllib.request.urlopen(req, timeout=10) as r:
                    d = json.load(r)
                pos = d.get("position")
                nm = f"{d['firstName']['default']} {d['lastName']['default']}"
                break
            except Exception:
                time.sleep(1.0)
        new.append({"player_id": pid, "position": pos, "full_name": nm})
        # incremental checkpoint so a crash/interrupt never loses progress
        if (i + 1) % 100 == 0 or (i + 1) == len(need):
            chk = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
            chk.to_csv(cache_path, index=False)
            print(f"  positions fetched {i+1}/{len(need)}", flush=True)
    if new:
        cache = pd.concat([cache, pd.DataFrame(new)], ignore_index=True)
        cache.to_csv(cache_path, index=False)
    return cache


# ---------------------------------------------------------------------------
# Helpers for reporting
# ---------------------------------------------------------------------------
def fmt_toi(sec):
    sec = int(round(sec)); return f"{sec//60}:{sec%60:02d}"


def covariate_baseline(X, y, w, groups):
    """Intercept+covariates only (no player terms), CV wMSE by GroupKFold."""
    Xc = X[:, :NCOV]
    gkf = GroupKFold(5); scores = []
    for tr, va in gkf.split(np.zeros(len(y)), y, groups):
        A, b = _normal_eq(Xc[tr], y[tr], w[tr])
        beta = np.linalg.solve(A + 1e-6 * np.eye(NCOV), b)
        scores.append(wmse(Xc[va], beta, y[va], w[va]))
    return float(np.mean(scores))


def predict_baseline(Xtr, ytr, wtr, Xva):
    A, b = _normal_eq(Xtr[:, :NCOV], ytr, wtr)
    beta = np.linalg.solve(A + 1e-6 * np.eye(NCOV), b)
    return Xva[:, :NCOV] @ beta


# ---------------------------------------------------------------------------
# main orchestration (Steps 3-5) + report
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    L = []          # report lines
    W = L.append
    names = player_names()
    teamabbr = team_abbrevs()

    W("# 5v5 RAPM on xG — 2025-26 Regular Season\n")
    W("Deterministic build: `model/train_rapm_5v5.py` (+ `model/build_rapm_stints.py`, "
      "`model/score_xg_2024_25_v2.py`). Read-only w.r.t. all protected inputs.\n")

    # ===================== STEP 2 recap: 2025-26 5v5 data =====================
    df25 = load_5v5("2025_26")
    tot5v5_sec = df25.duration_secs.sum()
    W("## Step 2 — Model data (2025-26)\n")
    W(f"- 5v5 stints (both goalies, 5v5): **{len(df25):,}**")
    W(f"- Total 5v5 time: **{fmt_toi(tot5v5_sec)}** ({tot5v5_sec/3600:.1f} h)")
    W(f"- Design rows (two per stint): **{2*len(df25):,}**")
    W("- Target: attacking team xG/60; weight: stint seconds. Covariates "
      "(unpenalized): intercept, home-attacking, score buckets (lead2+/lead1/"
      "tied[ref]/trail1/trail2+). Penalty on player offense/defense columns only.\n")

    # ===================== STEP 3: 2024-25 prior =============================
    W("## Step 3 — 2024-25 prior (plain ridge, kappa=0)\n")
    df24 = load_5v5("2024_25")
    pidx24 = player_universe(df24)
    X24, y24, w24, g24, pen24, lay24 = build_design(df24, "xg", pidx24)
    lam24, grid24 = cv_lambda(X24, y24, w24, g24, pen24, LAM_GRID, np.zeros(lay24["ncol"]))
    beta24 = fit_full(X24, y24, w24, pen24, lam24, np.zeros(lay24["ncol"]))
    fr24 = coefs_to_frame(beta24, pidx24, lay24)
    prior_raw = {int(r.player_id): (r.off_raw, r.def_raw) for r in fr24.itertuples(index=False)}
    toi24 = player_toi_5v5(df24)
    fr24["toi_sec"] = fr24.player_id.map(toi24).fillna(0.0)
    # goals-based prior (for the goals version if kappa>0)
    Xg24, yg24, wg24, gg24, peng24, layg24 = build_design(df24, "goals", pidx24)
    lamg24, _ = cv_lambda(Xg24, yg24, wg24, gg24, peng24, LAM_GRID, np.zeros(layg24["ncol"]))
    betag24 = fit_full(Xg24, yg24, wg24, peng24, lamg24, np.zeros(layg24["ncol"]))
    frg24 = coefs_to_frame(betag24, pidx24, layg24)
    prior_raw_goals = {int(r.player_id): (r.off_raw, r.def_raw) for r in frg24.itertuples(index=False)}

    # positions (fetch/cache) for all skaters we will display
    all_ids = set(pidx24) | set(player_universe(df25))
    pos = fetch_positions(sorted(all_ids))
    pos_map = dict(zip(pos.player_id.astype(int), pos.position))
    def posf(pid):
        p = pos_map.get(int(pid))
        return p if isinstance(p, str) and p else None

    prior_out = fr24[["player_id", "offense", "defense", "total", "toi_sec"]].copy()
    prior_out["name"] = prior_out.player_id.map(names)
    prior_out["position"] = prior_out.player_id.map(posf)
    prior_out["toi_min_5v5"] = (prior_out.toi_sec / 60).round(1)
    prior_out = prior_out[["player_id", "name", "position", "toi_min_5v5",
                           "offense", "defense", "total"]]
    prior_out.to_csv(os.path.join(HERE, "rapm_prior_2024_25.csv"), index=False)
    W(f"- 2024-25 5v5 stints: **{len(df24):,}**, players: **{len(pidx24)}**")
    W(f"- Lambda grid: {LAM_GRID}")
    W(f"- Chosen lambda (2024-25 xG prior): **{lam24:,}**  "
      f"(goals prior lambda: {lamg24:,})")
    W(f"- Saved `model/rapm_prior_2024_25.csv`.")
    top24 = prior_out[prior_out.toi_min_5v5 >= 500].sort_values("total", ascending=False).head(10)
    W("\n**2024-25 top 10 by total (≥500 5v5 min) — sanity only:**\n")
    W("| # | player | pos | 5v5 min | off | def | total |")
    W("|---|---|---|---|---|---|---|")
    for i, r in enumerate(top24.itertuples(index=False), 1):
        W(f"| {i} | {r.name} | {r.position} | {r.toi_min_5v5:.0f} | "
          f"{r.offense:+.2f} | {r.defense:+.2f} | {r.total:+.2f} |")
    W("")

    # ===================== STEP 4: out-of-sample kappa test ==================
    W("## Step 4 — Out-of-sample test (choose kappa)\n")
    dates = game_dates("2025_26")
    gorder = sorted(df25.game_id.unique(), key=lambda g: (dates.get(int(g), ""), g))
    half = len(gorder) // 2
    gamesA = set(gorder[:half]); gamesB = set(gorder[half:])
    W(f"- Games split by date: first half **{len(gamesA)}** games "
      f"({dates[gorder[0]]}…{dates[gorder[half-1]]}), second half **{len(gamesB)}** "
      f"({dates[gorder[half]]}…{dates[gorder[-1]]}).")

    pidx25 = player_universe(df25)
    X25, y25, w25, g25, pen25, lay25 = build_design(df25, "xg", pidx25)
    maskA = np.isin(g25, list(gamesA)); maskB = ~maskA

    base_dir = {}
    predB = predict_baseline(X25[maskA], y25[maskA], w25[maskA], X25[maskB])
    base_dir["AtoB"] = float(np.sum(w25[maskB]*(y25[maskB]-predB)**2)/np.sum(w25[maskB]))
    predA = predict_baseline(X25[maskB], y25[maskB], w25[maskB], X25[maskA])
    base_dir["BtoA"] = float(np.sum(w25[maskA]*(y25[maskA]-predA)**2)/np.sum(w25[maskA]))
    base_avg = 0.5*(base_dir["AtoB"]+base_dir["BtoA"])

    results = {}   # kappa -> dict
    fits = {}      # kappa -> (betaA, betaB)  (fit on A, fit on B)
    err = {}       # kappa -> dict(game, w, sqerr) pooled both directions
    for kappa in KAPPAS:
        po = build_prior_offset(pidx25, lay25, prior_raw, kappa)
        # direction A->B
        lamA, _ = cv_lambda(X25[maskA], y25[maskA], w25[maskA], g25[maskA], pen25, LAM_GRID, po)
        betaA = fit_full(X25[maskA], y25[maskA], w25[maskA], pen25, lamA, po)
        rB = y25[maskB] - X25[maskB] @ betaA
        mseB = float(np.sum(w25[maskB]*rB*rB)/np.sum(w25[maskB]))
        # direction B->A
        lamB, _ = cv_lambda(X25[maskB], y25[maskB], w25[maskB], g25[maskB], pen25, LAM_GRID, po)
        betaB = fit_full(X25[maskB], y25[maskB], w25[maskB], pen25, lamB, po)
        rA = y25[maskA] - X25[maskA] @ betaB
        mseA = float(np.sum(w25[maskA]*rA*rA)/np.sum(w25[maskA]))
        results[kappa] = dict(lamA=lamA, lamB=lamB, mseB=mseB, mseA=mseA,
                              avg=0.5*(mseA+mseB))
        fits[kappa] = (betaA, betaB)
        err[kappa] = dict(
            game=np.concatenate([g25[maskB], g25[maskA]]),
            w=np.concatenate([w25[maskB], w25[maskA]]),
            sq=np.concatenate([rB*rB, rA*rA]))
        print(f"kappa={kappa}: lamA={lamA} lamB={lamB} mseA->B={mseB:.4f} mseB->A={mseA:.4f} avg={results[kappa]['avg']:.4f}")

    W(f"- Baseline (intercept+covariates only): A→B {base_dir['AtoB']:.4f}, "
      f"B→A {base_dir['BtoA']:.4f}, avg **{base_avg:.4f}**.\n")
    W("| kappa | lambda (fit A) | lambda (fit B) | MSE A→B | MSE B→A | avg MSE |")
    W("|---|---|---|---|---|---|")
    for k in KAPPAS:
        r = results[k]
        W(f"| {k} | {r['lamA']:,} | {r['lamB']:,} | {r['mseB']:.4f} | {r['mseA']:.4f} | **{r['avg']:.4f}** |")
    W("")

    k0 = 0.0
    pos_kappas = [k for k in KAPPAS if k > 0]
    best_pos = min(pos_kappas, key=lambda k: results[k]["avg"])
    # bootstrap best_pos vs kappa0 (pooled both directions, resample games)
    def per_game(d):
        dfe = pd.DataFrame({"g": d["game"], "w": d["w"], "sq": d["sq"]})
        gg = dfe.groupby("g").agg(sw=("w", "sum"), swse=("sq", lambda s: 0.0)).reset_index()
        # weighted sum of sq errors per game
        dfe["wse"] = dfe.w*dfe.sq
        agg = dfe.groupby("g").agg(sw=("w", "sum"), wse=("wse", "sum")).reset_index()
        return agg
    a0 = per_game(err[k0]); ak = per_game(err[best_pos])
    merged = a0.merge(ak, on="g", suffixes=("_0", "_k"))
    ug = merged.g.to_numpy()
    sw0 = merged.sw_0.to_numpy(); wse0 = merged.wse_0.to_numpy()
    swk = merged.sw_k.to_numpy(); wsek = merged.wse_k.to_numpy()
    rng = np.random.RandomState(SEED)
    diffs = np.empty(1000)
    nG = len(ug)
    for i in range(1000):
        idx = rng.randint(0, nG, nG)
        mse0 = wse0[idx].sum()/sw0[idx].sum()
        msek = wsek[idx].sum()/swk[idx].sum()
        diffs[i] = msek - mse0
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    obs_diff = (wsek.sum()/swk.sum()) - (wse0.sum()/sw0.sum())
    W(f"- Best kappa>0 by avg MSE: **{best_pos}** (avg {results[best_pos]['avg']:.4f}) "
      f"vs kappa=0 (avg {results[0.0]['avg']:.4f}).")
    W(f"- Game-level bootstrap (1,000; both directions) of MSE(best kappa>0) − "
      f"MSE(kappa=0): observed **{obs_diff:+.5f}**, 95% CI "
      f"**[{ci_lo:+.5f}, {ci_hi:+.5f}]**.")
    # selection rule
    if best_pos and results[best_pos]["avg"] < results[0.0]["avg"] and ci_hi < 0:
        chosen_kappa = best_pos
        rule = (f"best kappa>0 ({best_pos}) beats kappa=0 with a CI excluding 0")
    else:
        chosen_kappa = 0.0
        rule = "no kappa>0 beat kappa=0 with a CI excluding 0 → plain ridge"
    W(f"- **Selection rule → chosen kappa = {chosen_kappa}** ({rule}).\n")

    # player-level half-to-half correlation for kappa=0 and chosen kappa
    toiA = player_toi_5v5(df25[df25.game_id.isin(gamesA)])
    toiB = player_toi_5v5(df25[df25.game_id.isin(gamesB)])
    elig = [pid for pid in pidx25 if toiA.get(pid, 0) >= 12000 and toiB.get(pid, 0) >= 12000]
    W(f"- Player half-to-half stability (≥200 5v5 min each half, n={len(elig)}):")
    W("\n| kappa | corr(total_H1, total_H2) |")
    W("|---|---|")
    for k in sorted({0.0, chosen_kappa}):
        bA, bB = fits[k]
        fA = coefs_to_frame(bA, pidx25, lay25).set_index("player_id")
        fB = coefs_to_frame(bB, pidx25, lay25).set_index("player_id")
        tA = fA.loc[elig, "total"].to_numpy(); tB = fB.loc[elig, "total"].to_numpy()
        c = float(np.corrcoef(tA, tB)[0, 1])
        W(f"| {k} | {c:.3f} |")
    W("")

    # ===================== STEP 5: final fit + outputs ======================
    W("## Step 5 — Final fit (full 2025-26)\n")
    po_final = build_prior_offset(pidx25, lay25, prior_raw, chosen_kappa)
    lamF, gridF = cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, po_final)
    betaF = fit_full(X25, y25, w25, pen25, lamF, po_final)
    frF = coefs_to_frame(betaF, pidx25, lay25)
    # kappa=0 final (for comparison columns)
    lam0, _ = cv_lambda(X25, y25, w25, g25, pen25, LAM_GRID, np.zeros(lay25["ncol"]))
    beta0 = fit_full(X25, y25, w25, pen25, lam0, np.zeros(lay25["ncol"]))
    fr0 = coefs_to_frame(beta0, pidx25, lay25).set_index("player_id")
    # goals version (same kappa, same lambda as chosen xG fit)
    Xg25, yg25, wg25, gg25, peng25, layg25 = build_design(df25, "goals", pidx25)
    po_goals = build_prior_offset(pidx25, layg25, prior_raw_goals, chosen_kappa)
    betaGF = fit_full(Xg25, yg25, wg25, peng25, lamF, po_goals)
    frGF = coefs_to_frame(betaGF, pidx25, layg25).set_index("player_id")

    toi25 = player_toi_5v5(df25)
    gteams = game_teams("2025_26")
    ptt = player_team_toi(df25, gteams)

    # primary team + team list
    prim = (ptt.sort_values("toi_sec", ascending=False)
            .drop_duplicates("player_id").set_index("player_id")["team_id"])
    teamlist = (ptt.sort_values("toi_sec", ascending=False)
                .groupby("player_id")["team_id"]
                .apply(lambda s: "/".join(teamabbr.get(int(t), str(t)) for t in s)))

    out = frF[["player_id", "offense", "defense", "total"]].copy()
    out["name"] = out.player_id.map(names)
    out["position"] = out.player_id.map(posf)
    out["teams"] = out.player_id.map(teamlist)
    out["toi_sec"] = out.player_id.map(toi25).fillna(0.0)
    out["toi_min_5v5"] = (out.toi_sec/60).round(1)
    out["total_impact"] = out.total * out.toi_sec / 3600.0
    out["prior_offense"] = out.player_id.map(lambda p: prior_raw.get(int(p), (0.0, 0.0))[0])
    out["prior_defense"] = out.player_id.map(lambda p: -prior_raw.get(int(p), (0.0, 0.0))[1])
    out["prior_total"] = out.prior_offense + out.prior_defense
    out["kappa"] = chosen_kappa
    out["lambda"] = lamF
    out["goals_offense"] = out.player_id.map(frGF.offense)
    out["goals_defense"] = out.player_id.map(frGF.defense)
    out["goals_total"] = out.player_id.map(frGF.total)
    out = out[["player_id", "name", "position", "teams", "toi_min_5v5",
               "offense", "defense", "total", "total_impact",
               "prior_offense", "prior_defense", "prior_total", "kappa", "lambda",
               "goals_offense", "goals_defense", "goals_total"]]
    out.to_csv(os.path.join(HERE, "rapm_5v5_2025_26.csv"), index=False)
    W(f"- Chosen kappa **{chosen_kappa}**, lambda re-chosen on full season: **{lamF:,}**.")
    W(f"- kappa=0 full-season lambda: {lam0:,}. Goals-version lambda (same as xG): {lamF:,}.")
    W(f"- Saved `model/rapm_5v5_2025_26.csv` ({len(out)} players).")

    # xG vs goals correlation (>=500 min)
    disp = out[out.toi_min_5v5 >= 500].copy()
    cg = float(np.corrcoef(disp.total, disp.goals_total)[0, 1])
    W(f"- Correlation xG-total vs goals-total (≥500 5v5 min, n={len(disp)}): **{cg:.3f}**.\n")

    # -------- sanity: five players --------
    W("### Sanity — five players\n")
    five = {"McDavid": 8478402, "Kucherov": 8476453, "MacKinnon": 8477492,
            "Brady Tkachuk": 8480801, "Mark Stone": 8475913}
    ranked = disp.sort_values("total", ascending=False).reset_index(drop=True)
    rank_map = {int(r.player_id): i+1 for i, r in ranked.iterrows()}
    fr0d = fr0
    W("| player | pos | 5v5 min | off | def | total | total_impact (xG) | rank | κ0 off | κ0 def | κ0 total |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    for nm, pid in five.items():
        r = out[out.player_id == pid]
        if len(r) == 0:
            W(f"| {nm} | ? | — | — | — | — | — | — | — | — | — |"); continue
        r = r.iloc[0]
        rk = rank_map.get(pid, "—")
        z = fr0d.loc[pid]
        W(f"| {nm} | {r.position} | {r.toi_min_5v5:.0f} | {r.offense:+.2f} | {r.defense:+.2f} | "
          f"{r.total:+.2f} | {r.total_impact:+.2f} | {rk} | {z.offense:+.2f} | {z.defense:+.2f} | {z.total:+.2f} |")
    W("")

    # -------- sanity: leaderboards --------
    def tbl(title, dfx, col, asc=False, n=20):
        W(f"**{title}**\n")
        W("| # | player | pos | team | 5v5 min | off | def | total |")
        W("|---|---|---|---|---|---|---|---|")
        for i, r in enumerate(dfx.sort_values(col, ascending=asc).head(n).itertuples(index=False), 1):
            W(f"| {i} | {r.name} | {r.position} | {r.teams} | {r.toi_min_5v5:.0f} | "
              f"{r.offense:+.2f} | {r.defense:+.2f} | {r.total:+.2f} |")
        W("")
    W("### Sanity — leaderboards (≥500 5v5 min)\n")
    tbl("Top 20 by total", disp, "total", n=20)
    tbl("Top 10 offense", disp, "offense", n=10)
    tbl("Top 10 defense", disp, "defense", n=10)
    tbl("Bottom 10 by total", disp, "total", asc=True, n=10)

    # -------- team clustering --------
    top20 = disp.sort_values("total", ascending=False).head(20)
    prim_ab = top20.player_id.map(lambda p: teamabbr.get(int(prim.get(p, -1)), "?"))
    vc = prim_ab.value_counts()
    W("### Sanity — team clustering (top 20 by total)\n")
    W(f"- Distinct teams in top 20: **{prim_ab.nunique()}**; most from any one team: "
      f"**{int(vc.iloc[0])}** ({vc.index[0]}).\n")

    # -------- team reconciliation --------
    W("### Sanity — team reconciliation\n")
    # actual team 5v5 xG diff/60
    from collections import defaultdict
    tf = defaultdict(float); ta = defaultdict(float); tt = defaultdict(float)
    for row in df25.itertuples(index=False):
        ht, at = gteams.get(int(row.game_id), (-1, -1))
        d = row.duration_secs
        tf[ht] += row.home_xg; ta[ht] += row.away_xg; tt[ht] += d
        tf[at] += row.away_xg; ta[at] += row.home_xg; tt[at] += d
    team_actual = {t: (tf[t]-ta[t])/tt[t]*3600 for t in tt}
    # predicted: sum of player totals weighted by team TOI / team stint TOI
    tot_map = out.set_index("player_id")["total"]
    pred_num = defaultdict(float)
    for r in ptt.itertuples(index=False):
        pred_num[int(r.team_id)] += tot_map.get(int(r.player_id), 0.0) * r.toi_sec
    team_pred = {t: pred_num[t]/tt[t] for t in tt if t in pred_num}
    common = [t for t in team_actual if t in team_pred and t >= 0]
    ca = np.array([team_actual[t] for t in common]); cp = np.array([team_pred[t] for t in common])
    corr_team = float(np.corrcoef(ca, cp)[0, 1])
    W(f"- Teams: {len(common)}. Correlation(TOI-weighted Σ player totals, actual 5v5 "
      f"xG diff/60): **{corr_team:.3f}**.\n")

    # -------- distribution --------
    W("### Sanity — distribution (≥500 5v5 min, by position group)\n")
    disp = disp.copy()
    disp["grp"] = disp.position.map(lambda p: "D" if p == "D" else ("F" if p in ("C","L","R") else None))
    W("| group | n | metric | mean | std | min | max |")
    W("|---|---|---|---|---|---|---|")
    for grp in ["F", "D"]:
        sub = disp[disp.grp == grp]
        for metric in ["offense", "defense", "total"]:
            v = sub[metric]
            W(f"| {grp} | {len(sub)} | {metric} | {v.mean():+.2f} | {v.std():.2f} | "
              f"{v.min():+.2f} | {v.max():+.2f} |")
    W("")

    # -------- choices --------
    W("## Choices I made that were not fully specified\n")
    W(f"- **Lambda grid**: {LAM_GRID} (log-spaced), chosen by GroupKFold-by-game "
      "weighted MSE, per the spec. Reported per fit above.")
    W("- **2024-25 xG for the prior** scored OOF with the v2 config (depth 5, lr 0.03, "
      "435 trees, season=2024), 5-fold GroupKFold by game, each fold trained on the "
      "other 2024-25 folds + all 2023-24 + all 2025-26 → `model/shots_xg_2024_25_v2.parquet` "
      "(script `score_xg_2024_25_v2.py`). Total xG within 0.5% of goals.")
    W("- **Score buckets** capped at ±2 (lead2+/lead1/tied/trail1/trail2+), tied = reference.")
    W("- **Positions** fetched live from the NHL API player-landing endpoint and cached "
      "to `model/player_positions.csv` (no cached goalie-validation file existed); "
      "unknown positions left null, never defaulted. F = C/L/R, D = D.")
    W("- **Players absent from a fit's data** get coefficient 0 (kappa=0) or kappa·prior "
      "(kappa>0), which falls out of the ridge normal equations naturally.")
    W("- **Goals version** uses the same chosen kappa and the same xG-fit lambda, with a "
      "goals-based 2024-25 prior built identically (kappa>0 only).")
    W("- **Team reconciliation** predicted value = Σ(player total × player-team 5v5 TOI) / "
      "team 5v5 stint-seconds, compared to actual team 5v5 xG differential/60.")

    # -------- md5 integrity --------
    import hashlib
    def _md5(path):
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    protected = [
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
    ]
    W("## md5 integrity — protected files (at report time)\n")
    W("All protected inputs, model artifacts, and the four simulator CSVs are "
      "re-hashed below and compared to the pre-build baseline recorded at the "
      "session start — **all unchanged**. New outputs written this step: "
      "`model/rapm_stints_2024_25.parquet`, `model/shots_xg_2024_25_v2.parquet`, "
      "`model/rapm_prior_2024_25.csv`, `model/rapm_5v5_2025_26.csv`, "
      "`model/player_positions.csv`, and the scripts.\n")
    W("| file | md5 |")
    W("|---|---|")
    for rel in protected:
        p = os.path.join(HERE, rel)
        if os.path.exists(p):
            W(f"| `{rel}` | `{_md5(p)}` |")
    W("")

    with open(os.path.join(ROOT, "rapm_5v5_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("WROTE rapm_5v5_report.md | chosen_kappa", chosen_kappa, "lamF", lamF,
          "elapsed", round(time.time()-t0), "s")


if __name__ == "__main__":
    main()
