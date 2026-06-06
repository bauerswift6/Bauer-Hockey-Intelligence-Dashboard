"""
Experimental Bayesian (prior-informed) RAPM — Enhancement 4.

A PARALLEL model to the ridge RAPM in train_rapm.py. It does NOT replace it and
is NOT (yet) wired into the composite rating. Its sole purpose is to measure how
much the standard ridge's collinearity tax (co-stars splitting credit) is relieved
when each player is shrunk toward an individual prior instead of toward zero.

Method — prior-informed ridge
-----------------------------
Standard ridge minimises  ||y - Xb||^2 + a||b||^2  → shrinks every coefficient
toward 0. When two stars share almost all their ice time, ridge cannot tell them
apart and pulls both toward 0 (the collinearity tax).

Prior-informed ridge instead minimises  ||y - Xb||^2 + a||b - b0||^2 — it shrinks
each coefficient toward a player-specific prior b0. Substituting b' = b - b0 gives
a STANDARD ridge on the centered target:

    minimise ||(y - X b0) - X b'||^2 + a||b'||^2 ,   then  b = b' + b0

i.e. "centre each player's shifts using their prior before the ridge regression"
(subtract the prior contribution of everyone on the ice from each shift's target),
fit ordinary ridge, then add the prior back.

Player priors (scaled to goal units)
-------------------------------------
  offensive prior = (player ixG/60)      - (league-average ixG/60)
  defensive prior = -((player on-ice xGA/60) - (league-average on-ice xGA/60))
                    (negated so positive = suppresses xGA = good defender)

Both are computed year-weighted over the same 15-season window as the RAPM, from
the in-pipeline shot + segment data. They are then linearly rescaled so their
spread matches the standard ridge RAPM's coefficient spread — that is the
"scale to goal units" step (the ridge RAPM coefficients ARE goal-rate units).

Run from project root:
    python3 model/train_rapm_bayesian.py

Output: model/rapm_bayesian_results.csv , model/rapm_bayesian_meta.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

# Reuse the entire ridge-RAPM pipeline — segment build, design matrix, etc.
from train_rapm import (
    ALPHAS,
    FEATURE_COLS,
    MODEL_DIR,
    SEASON_WEIGHTS,
    DEFAULT_WEIGHT,
    XG_MODEL_PATH,
    build_all_segments,
    build_design_matrix,
    build_goalie_gsax,
    build_player_index,
    build_position_lookup,
    load_or_fetch_shifts,
    season_str_from_game_id,
)

AUGMENTED_PATH = MODEL_DIR / "shots_augmented.parquet"
HOME_META_PATH = MODEL_DIR / "home_id_by_game.json"
SHIFTS_PATH = MODEL_DIR / "shifts_dataset.parquet"
RIDGE_CSV_PATH = MODEL_DIR / "rapm_results.csv"
OUT_CSV_PATH = MODEL_DIR / "rapm_bayesian_results.csv"
OUT_META_PATH = MODEL_DIR / "rapm_bayesian_meta.json"


def season_weight_for_game(game_id: int) -> float:
    return SEASON_WEIGHTS.get(season_str_from_game_id(int(game_id)), DEFAULT_WEIGHT)


# ---------------------------------------------------------------------------
# Player priors
# ---------------------------------------------------------------------------
def compute_priors(shots_df: pd.DataFrame, segments_df: pd.DataFrame,
                   pid_to_col: dict, ridge_csv: pd.DataFrame) -> dict:
    """Return {pid -> (off_prior, def_prior, ixg60, xga60)} for every qualified
    player. Priors are year-weighted deviations from the league average, then
    rescaled so their std matches the standard ridge RAPM coefficient std."""
    print("\nComputing player priors (year-weighted ixG/60 and on-ice xGA/60) …", flush=True)

    # --- year-weighted individual xG per shooter ---
    sw = shots_df["game_id"].map(season_weight_for_game).astype(float)
    ixg_w = (shots_df["xg"].astype(float) * sw).groupby(shots_df["shooter_id"]).sum()
    ixg_w = ixg_w.to_dict()

    # --- year-weighted on-ice xGA and TOI per player, from segments ---
    toi_w: dict = {}   # year-weighted seconds
    xga_w: dict = {}   # year-weighted xG allowed while on ice
    for r in segments_df.itertuples(index=False):
        w = season_weight_for_game(r.game_id)
        wdur = w * r.dur_sec
        for pid in r.home_skaters:        # home players: allowed = home_xga
            toi_w[pid] = toi_w.get(pid, 0.0) + wdur
            xga_w[pid] = xga_w.get(pid, 0.0) + w * r.home_xga
        for pid in r.away_skaters:        # away players: allowed = home_xgf
            toi_w[pid] = toi_w.get(pid, 0.0) + wdur
            xga_w[pid] = xga_w.get(pid, 0.0) + w * r.home_xgf

    # --- per-60 rates for the qualified pool ---
    ixg60, xga60 = {}, {}
    for pid in pid_to_col:
        hours = toi_w.get(pid, 0.0) / 3600.0
        if hours <= 0:
            ixg60[pid] = xga60[pid] = 0.0
            continue
        ixg60[pid] = ixg_w.get(pid, 0.0) / hours
        xga60[pid] = xga_w.get(pid, 0.0) / hours

    lg_ixg = float(np.mean(list(ixg60.values())))
    lg_xga = float(np.mean(list(xga60.values())))
    print(f"  League-average ixG/60 = {lg_ixg:.4f},  on-ice xGA/60 = {lg_xga:.4f} "
          f"(over {len(pid_to_col):,} qualified players).", flush=True)

    # Raw priors: deviation from league average (defensive negated → +ve = good).
    off_raw = {pid: ixg60[pid] - lg_ixg for pid in pid_to_col}
    def_raw = {pid: -(xga60[pid] - lg_xga) for pid in pid_to_col}

    # --- scale to goal units: match the standard ridge RAPM coefficient spread ---
    ridge_off_std = float(ridge_csv["offensive_rapm"].std(ddof=0))
    ridge_def_std = float(ridge_csv["defensive_rapm"].std(ddof=0))
    off_raw_std = float(np.std(list(off_raw.values()))) or 1.0
    def_raw_std = float(np.std(list(def_raw.values()))) or 1.0
    off_scale = ridge_off_std / off_raw_std
    def_scale = ridge_def_std / def_raw_std
    print(f"  Prior scaling — offensive ×{off_scale:.4f}  (raw std {off_raw_std:.4f} "
          f"→ ridge std {ridge_off_std:.4f})", flush=True)
    print(f"  Prior scaling — defensive ×{def_scale:.4f}  (raw std {def_raw_std:.4f} "
          f"→ ridge std {ridge_def_std:.4f})", flush=True)

    priors = {}
    for pid in pid_to_col:
        priors[pid] = (
            off_raw[pid] * off_scale,
            def_raw[pid] * def_scale,
            ixg60[pid],
            xga60[pid],
        )
    return priors, {"league_ixg60": lg_ixg, "league_xga60": lg_xga,
                    "off_scale": off_scale, "def_scale": def_scale,
                    "ridge_off_std": ridge_off_std, "ridge_def_std": ridge_def_std}


# ---------------------------------------------------------------------------
# Prior-informed ridge fit
# ---------------------------------------------------------------------------
def fit_prior_informed_ridge(X, y, weights, beta0, alphas, label="bayesian"):
    """Centre the target by the prior contribution (X @ beta0), fit a standard
    RidgeCV on the residual, then add beta0 back. Widens the alpha grid if the
    optimum pins to a grid edge."""
    y_centered = y - X.dot(beta0)
    alphas = list(alphas)
    print(f"\nFitting prior-informed RidgeCV ({label}) — target centred by prior …",
          flush=True)
    t0 = time.time()
    model = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
    model.fit(X, y_centered, sample_weight=weights)
    while model.alpha_ == alphas[0] or model.alpha_ == alphas[-1]:
        if model.alpha_ == alphas[-1]:
            print(f"  alpha pinned at UPPER boundary {alphas[-1]} — widening up.", flush=True)
            top = alphas[-1]
            alphas = alphas + [top * 2, top * 5, top * 10]
        else:
            print(f"  alpha pinned at LOWER boundary {alphas[0]} — widening down.", flush=True)
            lo = alphas[0]
            alphas = [lo / 10, lo / 5, lo / 2] + alphas
        model = RidgeCV(alphas=alphas, fit_intercept=True, cv=5)
        model.fit(X, y_centered, sample_weight=weights)
    print(f"  done in {time.time()-t0:.1f}s — alpha selected: {model.alpha_} "
          f"(interior to [{alphas[0]} … {alphas[-1]}])", flush=True)
    # b = b' + b0 — add the prior back to the shrunk residual coefficients.
    final_coef = model.coef_ + beta0
    return final_coef, model.alpha_, alphas


def main():
    print("Experimental Bayesian (prior-informed) RAPM — Enhancement 4", flush=True)
    for p in (AUGMENTED_PATH, HOME_META_PATH, XG_MODEL_PATH, RIDGE_CSV_PATH):
        if not p.exists():
            print(f"ERROR: required input missing → {p}", flush=True)
            sys.exit(1)

    # --- Stage 1: shots (cached augmented parquet) + fresh xG scoring ---
    print(f"\nLoading augmented shots → {AUGMENTED_PATH}", flush=True)
    shots_df = pd.read_parquet(AUGMENTED_PATH)
    home_id_by_game = {int(k): int(v)
                       for k, v in json.loads(HOME_META_PATH.read_text()).items()}
    model = joblib.load(XG_MODEL_PATH)
    shots_df["xg"] = model.predict_proba(shots_df[FEATURE_COLS].astype("float32"))[:, 1]
    print(f"  {len(shots_df):,} shots — mean xG {shots_df['xg'].mean():.4f} "
          f"(goal rate {shots_df['is_goal'].mean():.4f})", flush=True)

    # --- Stage 2: shifts (per-season cache) ---
    print("\nLoading 15 seasons of shift data (per-season cache) …", flush=True)
    shifts_df = load_or_fetch_shifts()
    if shifts_df.empty:
        print("ERROR: no shift data available.", flush=True)
        sys.exit(1)

    # --- Stage 3: segments, player index, design matrix (identical to ridge RAPM) ---
    pid_to_pos = build_position_lookup()
    goalie_gsax = build_goalie_gsax()
    segments_df = build_all_segments(shifts_df, shots_df, home_id_by_game, pid_to_pos)
    if segments_df.empty:
        print("ERROR: no segments built.", flush=True)
        sys.exit(1)
    pid_to_col, col_to_pid, qualifying_toi = build_player_index(
        segments_df, shifts_df, pid_to_pos)
    if not pid_to_col:
        print("ERROR: no players cleared the EV-minute threshold.", flush=True)
        sys.exit(1)
    X, y_xgf, y_xga, weights = build_design_matrix(segments_df, pid_to_col, goalie_gsax)

    # --- Stage 4: player priors ---
    ridge_csv = pd.read_csv(RIDGE_CSV_PATH)
    priors, prior_meta = compute_priors(shots_df, segments_df, pid_to_col, ridge_csv)

    # --- Stage 5: build the prior vector beta0 (length 2P) ---
    # Columns 0..P-1 are offensive; P..2P-1 are defensive. The combined regression's
    # off RAPM = coef[c], def RAPM = -coef[c+P]. To shrink def RAPM toward def_prior
    # the defensive-column target must be -def_prior.
    P = len(pid_to_col)
    beta0 = np.zeros(2 * P, dtype=np.float64)
    for pid, c in pid_to_col.items():
        off_prior, def_prior, _, _ = priors[pid]
        beta0[c] = off_prior
        beta0[c + P] = -def_prior
    print(f"\nPrior vector beta0 built — {P} offensive + {P} defensive entries.", flush=True)

    # --- Stage 6: prior-informed ridge ---
    final_coef, alpha, alphas_used = fit_prior_informed_ridge(
        X, y_xgf, weights, beta0, ALPHAS, label="combined target = team xGF/60")

    # --- Stage 7: assemble results ---
    name_team = (shifts_df.sort_values("game_id")
                 .groupby("player_id")
                 .agg(first_name=("first_name", "last"),
                      last_name=("last_name", "last"),
                      team=("team_abbrev", "last"),
                      shift_count=("game_id", "size"))
                 .to_dict(orient="index"))
    rows = []
    for c, pid in col_to_pid.items():
        off_rapm = float(final_coef[c])
        def_rapm = -float(final_coef[c + P])
        off_prior, def_prior, ixg60, xga60 = priors[pid]
        meta = name_team.get(pid, {})
        rows.append({
            "player_id": pid,
            "player_name": f"{meta.get('first_name','')} {meta.get('last_name','')}".strip(),
            "team": meta.get("team", ""),
            "season": "20102025",
            "total_rapm_bayes": round(off_rapm + def_rapm, 4),
            "offensive_rapm_bayes": round(off_rapm, 4),
            "defensive_rapm_bayes": round(def_rapm, 4),
            "off_prior": round(off_prior, 4),
            "def_prior": round(def_prior, 4),
            "ixg_60": round(ixg60, 4),
            "onice_xga_60": round(xga60, 4),
            "toi_minutes": round(qualifying_toi.get(pid, 0) / 60.0, 1),
            "shift_count": int(meta.get("shift_count", 0)),
        })
    out = (pd.DataFrame(rows)
           .sort_values("total_rapm_bayes", ascending=False)
           .reset_index(drop=True))
    out["alpha"] = alpha

    out.to_csv(OUT_CSV_PATH, index=False)
    print(f"\nSaved → {OUT_CSV_PATH}", flush=True)

    meta = {
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": "experimental Bayesian (prior-informed) RAPM — parallel to ridge RAPM",
        "method": ("prior-informed ridge: minimise ||y-Xb||^2 + a||b-b0||^2 by "
                   "centering the target with X@b0, fitting standard ridge, then b=b'+b0"),
        "n_segments": int(len(segments_df)),
        "n_players": int(P),
        "alpha_selected": float(alpha),
        "alphas_tested": alphas_used,
        "priors": {
            "offensive": "year-weighted (player ixG/60 - league-average ixG/60)",
            "defensive": "year-weighted -((player on-ice xGA/60) - league-average)",
            **{k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in prior_meta.items()},
        },
        "note": ("Experimental. Not wired into the composite rating. Does NOT replace "
                 "model/rapm_results.csv (the live ridge RAPM)."),
    }
    OUT_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved → {OUT_META_PATH}", flush=True)

    # --- Report: Bayesian top 20 alongside the ridge RAPM ---
    ridge = ridge_csv.copy().sort_values("total_rapm", ascending=False).reset_index(drop=True)
    ridge_rank = {n: i + 1 for i, n in enumerate(ridge["player_name"])}
    ridge_row = {r["player_name"]: r for _, r in ridge.iterrows()}

    print("\n" + "=" * 92, flush=True)
    print("BAYESIAN RAPM — TOP 20  (ridge rank shown for comparison)", flush=True)
    print("=" * 92, flush=True)
    print(f"{'#':>3} {'Player':<24}{'Tm':>4}{'BayesTot':>10}{'BayesOff':>10}"
          f"{'BayesDef':>10}{'RidgeTot':>10}{'RidgeRk':>9}", flush=True)
    for i, r in out.head(20).iterrows():
        rr = ridge_row.get(r["player_name"])
        rt = f"{rr['total_rapm']:+.4f}" if rr is not None else "  n/a"
        rk = ridge_rank.get(r["player_name"], "n/a")
        print(f"{i+1:>3} {r['player_name']:<24}{r['team']:>4}"
              f"{r['total_rapm_bayes']:>+10.4f}{r['offensive_rapm_bayes']:>+10.4f}"
              f"{r['defensive_rapm_bayes']:>+10.4f}{rt:>10}{str(rk):>9}", flush=True)

    print("\n" + "=" * 92, flush=True)
    print("RIDGE RAPM — TOP 20  (Bayesian rank shown for comparison)", flush=True)
    print("=" * 92, flush=True)
    bayes_rank = {n: i + 1 for i, n in enumerate(out["player_name"])}
    print(f"{'#':>3} {'Player':<24}{'Tm':>4}{'RidgeTot':>10}{'RidgeOff':>10}"
          f"{'RidgeDef':>10}{'BayesRk':>9}", flush=True)
    for i, r in ridge.head(20).iterrows():
        print(f"{i+1:>3} {r['player_name']:<24}{r['team']:>4}"
              f"{r['total_rapm']:>+10.4f}{r['offensive_rapm']:>+10.4f}"
              f"{r['defensive_rapm']:>+10.4f}{str(bayes_rank.get(r['player_name'],'n/a')):>9}",
              flush=True)

    print("\n" + "=" * 92, flush=True)
    print("COLLINEARITY CHECK — MacKinnon / Draisaitl / Makar in each model", flush=True)
    print("=" * 92, flush=True)
    for name in ("Nathan MacKinnon", "Leon Draisaitl", "Cale Makar",
                 "Connor McDavid", "Mikko Rantanen"):
        b = out[out["player_name"] == name]
        if b.empty:
            print(f"  {name:<22} not found in Bayesian set", flush=True)
            continue
        br = b.iloc[0]
        b_rank = int(b.index[0]) + 1
        rr = ridge_row.get(name)
        if rr is not None:
            print(f"  {name:<20}  Bayesian #{b_rank:<4} total={br['total_rapm_bayes']:+.4f}"
                  f"  (off={br['offensive_rapm_bayes']:+.4f} def={br['defensive_rapm_bayes']:+.4f})"
                  f"   |   Ridge #{ridge_rank.get(name,'n/a'):<4} total={rr['total_rapm']:+.4f}"
                  f"  (prior off={br['off_prior']:+.4f} def={br['def_prior']:+.4f})", flush=True)
        else:
            print(f"  {name:<20}  Bayesian #{b_rank} — not in ridge set", flush=True)


if __name__ == "__main__":
    main()
