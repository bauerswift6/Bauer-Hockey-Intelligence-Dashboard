"""
train_xg_sog.py
---------------
Shots-ON-GOAL xG model for goalie GSAX: P(goal | the shot reached the net).
A goalie only faces on-net shots, so GSAX needs a model trained on on-net shots
only (SOG = saves + goals; missed shots incl. posts/crossbars excluded).

Built with the EXACT v2 approach (imports train_xg_v2.py; does not modify it):
same FEATURES (incl. integer `season`), same tuning procedure and grid, same
fixed holdout (last 263 games of 2025-26 by date), same cross-season pooling.
Deterministic (random_state=42).

On-net identification: xg_features_*.parquet has no event-type column, but it is
row-for-row positionally aligned with shots_enriched_*.parquet (non-shootout)
[verified], whose `on_goal` flag marks shots that reached the net (all goals have
on_goal==1). Empty-net-against shots (no defending goalie) are excluded; in the
regular-season non-shootout data they number 0 per season (shots_augmented omits
shots taken at an empty net by construction).

Outputs (new files only):
  model/shots_xg_sog_2025_26.parquet   (game_id, abs_secs, shooter_id, xg_sog, fold)
  model/xg_sog_model_2025_26.json      (final booster; scoring defaults season=2025)
  model/xg_sog_meta_2025_26.json       (metrics/gates for the report)
"""
from __future__ import annotations
import itertools
import json
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

import train_xg_v2 as V2

HERE = V2.HERE
SEED = 42
FEATURES = V2.FEATURES
LAB = V2.LAB
OUT_PARQUET = os.path.join(HERE, "shots_xg_sog_2025_26.parquet")
OUT_MODEL = os.path.join(HERE, "xg_sog_model_2025_26.json")
OUT_META = os.path.join(HERE, "xg_sog_meta_2025_26.json")


def load_sog(season: str) -> pd.DataFrame:
    """v2 feature load, restricted to ON-NET, NON-empty-net shots."""
    xf = pd.read_parquet(os.path.join(HERE, f"xg_features_{season}.parquet"))
    en = pd.read_parquet(os.path.join(HERE, f"shots_enriched_{season}.parquet"))
    en = en[en.period != 5].reset_index(drop=True)
    key = ["game_id", "abs_secs", "shooter_id"]
    assert (xf[key].reset_index(drop=True).values == en[key].values).all(), \
        f"{season}: xg_features not positionally aligned to enriched"
    xf = xf.copy()
    xf["on_goal"] = en.on_goal.to_numpy()
    xf["empty_net_against"] = en.empty_net_against.to_numpy()
    xf = xf[xf.train_drop_reason.isna()].reset_index(drop=True)
    xf["season"] = V2.SEASON_YEAR[season]
    sog = xf[(xf.on_goal == 1) & (~xf.empty_net_against.astype(bool))].reset_index(drop=True)
    return sog


def main():
    data = {s: load_sog(s) for s in V2.POOL_SEASONS}
    all25 = data["2025_26"]
    meta = {"season_counts": {}}
    for s in V2.POOL_SEASONS:
        d = data[s]
        by_type = {LAB.get(int(t), int(t)): int((d.shot_type_id == t).sum())
                   for t in sorted(d.shot_type_id.unique())}
        meta["season_counts"][s] = dict(
            on_net=len(d), goals=int(d.is_goal.sum()),
            goal_rate=float(d.is_goal.mean()), by_shot_type=by_type)
        print(f"{s}: on-net={len(d)} goals={int(d.is_goal.sum())} rate={d.is_goal.mean():.4f}")

    # fixed holdout: last 263 games of 2025-26 by date (same as v2)
    gd = all25[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
    holdout_games = set(gd.iloc[int(len(gd) * 0.8):].game_id)
    hold = all25[all25.game_id.isin(holdout_games)].reset_index(drop=True)
    yh = hold.is_goal.values
    pool = pd.concat([data["2023_24"], data["2024_25"],
                      all25[~all25.game_id.isin(holdout_games)]], ignore_index=True)
    meta["holdout_games"] = len(holdout_games)
    meta["holdout_rows"] = len(hold)
    meta["holdout_goals"] = int(yh.sum())
    print(f"pool={len(pool)} holdout={len(hold)} holdout_goals={int(yh.sum())}")

    # ---- tuning: same grid/procedure as train_xg_v2 ----
    grid = [dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, subsample=ss,
                 colsample_bytree=1.0)
            for md, lr, mcw, ss in itertools.product([4, 5, 6], [0.03, 0.05], [5, 10, 20], [0.8, 1.0])]
    rng = np.random.RandomState(SEED)
    if len(grid) > 14:
        grid = [grid[i] for i in sorted(rng.choice(len(grid), 14, replace=False))]
    tune = []
    for params in grid:
        ll, sd, nest = V2.cv_logloss(pool, params)
        tune.append(dict(params=params, cv_logloss=ll, n_est=nest))
        print(f"  tune {params} cv={ll:.5f} n_est~{nest}")
    tune.sort(key=lambda r: r["cv_logloss"])
    best = tune[0]; BP = best["params"]; NEST = best["n_est"]
    meta["tuning"] = dict(grid=tune, chosen=BP, n_est=NEST)
    print("BEST", BP, "NEST", NEST)

    # ---- holdout eval ----
    m = V2.mk(NEST, BP); m.fit(V2.prep(pool)[FEATURES], pool.is_goal.values)
    p_h = np.clip(m.predict_proba(V2.prep(hold)[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    base_rate = float(pool.is_goal.mean())
    hold_metrics = dict(
        log_loss=float(log_loss(yh, p_h)), auc=float(roc_auc_score(yh, p_h)),
        brier=float(brier_score_loss(yh, p_h)),
        pred_goals=float(p_h.sum()), actual_goals=int(yh.sum()),
        baseline_logloss=float(log_loss(yh, np.full_like(p_h, base_rate))),
        base_rate=base_rate)
    hold_metrics["pred_pct"] = (hold_metrics["pred_goals"] - hold_metrics["actual_goals"]) / hold_metrics["actual_goals"] * 100
    meta["holdout_metrics"] = hold_metrics
    print(f"HOLDOUT ll={hold_metrics['log_loss']:.5f} auc={hold_metrics['auc']:.4f} "
          f"pred={hold_metrics['pred_goals']:.1f} actual={hold_metrics['actual_goals']} "
          f"({hold_metrics['pred_pct']:+.2f}%) baseline_ll={hold_metrics['baseline_logloss']:.5f}")

    # danger-band calibration on holdout
    bands = [(-1, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
    band_rows = []
    for lo, hi in bands:
        ii = (p_h >= lo) & (p_h < hi) if lo != -1 else (p_h < hi)
        if ii.sum() > 0:
            band_rows.append(dict(band=f"[{max(lo,0):.2f},{hi:.2f})", n=int(ii.sum()),
                                  actual_rate=float(yh[ii].mean()), pred_rate=float(p_h[ii].mean())))
    meta["danger_bands"] = band_rows

    # ---- OOF over 2025-26 on-net ----
    prior = pd.concat([data["2023_24"], data["2024_25"]], ignore_index=True)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(all25)); fold = np.full(len(all25), -1)
    for k, (tr, va) in enumerate(gkf.split(all25[FEATURES], all25.is_goal.values, all25.game_id.values)):
        tp = pd.concat([all25.iloc[tr], prior], ignore_index=True)
        mm = V2.mk(NEST, BP); mm.fit(V2.prep(tp)[FEATURES], tp.is_goal.values)
        oof[va] = mm.predict_proba(V2.prep(all25.iloc[va])[FEATURES])[:, 1]
        fold[va] = k
    total_oof = float(oof.sum()); actual = int(all25.is_goal.sum())
    oof_pct = (total_oof - actual) / actual * 100
    meta["oof"] = dict(total=total_oof, actual=actual, pct=oof_pct,
                       log_loss=float(log_loss(all25.is_goal.values, np.clip(oof, 1e-6, 1 - 1e-6))),
                       auc=float(roc_auc_score(all25.is_goal.values, oof)))
    print(f"OOF total={total_oof:.1f} actual={actual} ({oof_pct:+.2f}%)")

    # ---- gates ----
    g1 = abs(oof_pct) <= 2.0
    st_ratio = {}
    for t in sorted(all25.shot_type_id.unique().tolist()):
        ii = all25.shot_type_id.values == t
        a = int(all25.is_goal.values[ii].sum()); pr = float(oof[ii].sum())
        st_ratio[LAB.get(t, t)] = dict(n=int(ii.sum()), goals=a, pred=pr,
                                       ratio=(a / pr if pr > 0 else None))
    g2_fail = [k for k, v in st_ratio.items()
               if v["goals"] >= 100 and v["ratio"] is not None and not (0.90 <= v["ratio"] <= 1.10)]
    g2 = len(g2_fail) == 0
    g3 = abs(hold_metrics["pred_pct"]) <= 3.0
    meta["gates"] = dict(gate1_oof_2pct=dict(passed=bool(g1), pct=oof_pct),
                         gate2_shottype=dict(passed=bool(g2), failures=g2_fail, ratios=st_ratio),
                         gate3_holdout_3pct=dict(passed=bool(g3), pct=hold_metrics["pred_pct"]))
    all_pass = g1 and g2 and g3
    meta["all_gates_pass"] = bool(all_pass)
    print(f"GATE1 OOF {oof_pct:+.2f}% -> {'PASS' if g1 else 'FAIL'}")
    print(f"GATE2 shot-type -> {'PASS' if g2 else 'FAIL'} fails={g2_fail}")
    print(f"GATE3 holdout {hold_metrics['pred_pct']:+.2f}% -> {'PASS' if g3 else 'FAIL'}")

    if not all_pass:
        json.dump(meta, open(OUT_META, "w"), indent=2, default=float)
        print(">>> A GATE FAILED — writing meta, NO model/parquet. STOP.")
        return

    # ---- outputs ----
    out = all25[["game_id", "abs_secs", "shooter_id"]].copy()
    out["xg_sog"] = oof.round(6); out["fold"] = fold
    out.to_parquet(OUT_PARQUET, index=False)

    fpool = pd.concat([data["2023_24"], data["2024_25"], all25], ignore_index=True)
    fm = V2.mk(NEST, BP); fm.fit(V2.prep(fpool)[FEATURES], fpool.is_goal.values)
    fm.get_booster().save_model(OUT_MODEL)
    json.dump(meta, open(OUT_META, "w"), indent=2, default=float)
    print("WROTE", OUT_PARQUET, OUT_MODEL, OUT_META)


def score_sog(df: pd.DataFrame, season: int = 2025) -> np.ndarray:
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(OUT_MODEL)
    x = df.copy(); x["season"] = season
    x["shot_type_id"] = x["shot_type_id"].astype("category")
    return model.predict_proba(x[FEATURES])[:, 1]


if __name__ == "__main__":
    main()
