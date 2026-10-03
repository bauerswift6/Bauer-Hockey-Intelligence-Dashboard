"""
train_xg_multiseason.py
-----------------------
v2 xG: test whether adding prior seasons (2022-23, 2023-24, 2024-25) to the
2025-26 training pool improves the xG model, select the best configuration on
the fixed 2025-26 holdout, and produce an out-of-fold v2 xG for every 2025-26
shot. Deterministic (random_state=42 throughout). v1 files are never modified;
all outputs carry a _v2 suffix.

Fixed holdout = the same last 263 games of 2025-26 by date (22,089 rows) used by
v1. Training pool per config = first 80% of 2025-26 (by game date) + whichever
prior seasons the config includes (prior seasons used in full).

Run: python3 model/train_xg_multiseason.py
Artifacts: xg_model_2025_26_v2.json, shots_xg_2025_26_v2.parquet, and a JSON
results dump the report is built from.
"""
from __future__ import annotations
import json, os, itertools
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

SEED = 42
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/private/tmp/claude-501/-Users-bauerswift-NHL-Front-Office-AI-Integration/702051fb-ab76-4f35-9b0d-24052e0f7296/scratchpad"

V1_PARAMS = dict(max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=1.0)
FEATURES = ["distance_from_net", "angle_from_center", "x_norm", "abs_y_norm", "shot_type_id",
            "is_rebound", "time_since_last_shot", "is_rush", "score_state", "period",
            "time_in_period", "is_home", "skaters_for", "skaters_against", "off_wing"]
SEASON_FILE = {"2025_26": "xg_features_2025_26.parquet", "2024_25": "xg_features_2024_25.parquet",
               "2023_24": "xg_features_2023_24.parquet", "2022_23": "xg_features_2022_23.parquet"}
RECENCY = {"2025_26": 1.0, "2024_25": 0.8, "2023_24": 0.6, "2022_23": 0.4}
SEASON_SETS = {"A": ["2025_26"], "B": ["2025_26", "2024_25"],
               "C": ["2025_26", "2024_25", "2023_24"], "D": ["2025_26", "2024_25", "2023_24", "2022_23"]}


def load_season(s):
    d = pd.read_parquet(os.path.join(HERE, SEASON_FILE[s]))
    d = d[d.train_drop_reason.isna()].reset_index(drop=True).copy()
    d["season"] = s
    return d


def merge_shot_types(df):
    df = df.copy()
    df["shot_type_id"] = df["shot_type_id"].replace(2, 1)   # snap -> wrist
    return df


def prep(df):
    df = df.copy()
    df["shot_type_id"] = df["shot_type_id"].astype("category")
    return df


def mk(n_estimators, params):
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             n_estimators=n_estimators, tree_method="hist", enable_categorical=True,
                             random_state=SEED, n_jobs=8, **params)


def holdout_split(all25):
    gd = all25[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
    cut = int(len(gd) * 0.8)
    holdout_games = set(gd.iloc[cut:].game_id)
    return holdout_games


def build_pool(cfg, seasons_data, holdout_games):
    """Training pool: 2025-26 non-holdout games + prior seasons in full."""
    frames = []
    for s in cfg["seasons"]:
        d = seasons_data[s]
        if s == "2025_26":
            d = d[~d.game_id.isin(holdout_games)]
        frames.append(d)
    pool = pd.concat(frames, ignore_index=True)
    if cfg["shot_types"] == "merged":
        pool = merge_shot_types(pool)
    if cfg["weight"] == "recency":
        w = pool["season"].map(RECENCY).values
    else:
        w = np.ones(len(pool))
    return pool, w


def fit_early_stop(pool, w, params, n_estimators=2000):
    """Fit with a single game-grouped 85/15 validation split for early stopping.
    Early stopping optimizes UNWEIGHTED validation log loss (matches the unweighted
    holdout metric)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    tr, va = next(gss.split(pool, pool.is_goal.values, pool.game_id.values))
    m = mk(n_estimators, params)
    m.set_params(early_stopping_rounds=50)
    m.fit(prep(pool.iloc[tr])[FEATURES], pool.is_goal.values[tr],
          sample_weight=w[tr],
          eval_set=[(prep(pool.iloc[va])[FEATURES], pool.is_goal.values[va])], verbose=False)
    return m, m.best_iteration + 1


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(logloss=float(log_loss(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), pred_goals=float(p.sum()))


def main():
    seasons_data = {s: load_season(s) for s in SEASON_FILE}
    all25 = seasons_data["2025_26"]
    # v1 OOF xG aligns POSITIONALLY to xg_features_2025_26 (same 111,078 rows, same
    # order) — a key-merge would fan out on the same-second rapid-fire pairs, so
    # attach by position instead. Verified: keys + xg_old match positionally.
    _v1sx = pd.read_parquet(os.path.join(HERE, "shots_xg_2025_26.parquet"))
    assert len(_v1sx) == len(all25), "v1 shots_xg length mismatch"
    assert (all25[["game_id", "abs_secs", "shooter_id"]].values == _v1sx[["game_id", "abs_secs", "shooter_id"]].values).all()
    all25 = all25.copy(); all25["xg_v1"] = _v1sx.xg_new.values
    seasons_data["2025_26"] = all25
    holdout_games = holdout_split(all25)
    hold = all25[all25.game_id.isin(holdout_games)].reset_index(drop=True)
    yh = hold.is_goal.values
    hold_old = np.clip(hold.xg_old.values, 1e-6, 1 - 1e-6)
    print(f"holdout: {len(hold)} rows / {hold.game_id.nunique()} games / {int(yh.sum())} goals")

    # ---------- STEP 1: 14 configs ----------
    configs = []
    for st in ["asis", "merged"]:
        for setk in ["A", "B", "C", "D"]:
            weights = ["unweighted"] if setk == "A" else ["unweighted", "recency"]
            for wk in weights:
                configs.append(dict(name=f"{setk}-{wk}-{st}", seasons=SEASON_SETS[setk],
                                    set=setk, weight=wk, shot_types=st))
    results = []
    hold_pred = {}
    for cfg in configs:
        pool, w = build_pool(cfg, seasons_data, holdout_games)
        m, ntrees = fit_early_stop(pool, w, V1_PARAMS)
        hp = m.predict_proba(prep(merge_shot_types(hold) if cfg["shot_types"] == "merged" else hold)[FEATURES])[:, 1]
        hold_pred[cfg["name"]] = np.clip(hp, 1e-6, 1 - 1e-6)
        mm = metrics(yh, hp)
        mm.update(name=cfg["name"], set=cfg["set"], weight=cfg["weight"], shot_types=cfg["shot_types"],
                  n_trees=int(ntrees), pool_rows=int(len(pool)))
        results.append(mm)
        print(f"  {cfg['name']:22s} logloss={mm['logloss']:.5f} auc={mm['auc']:.4f} "
              f"brier={mm['brier']:.5f} predG={mm['pred_goals']:.0f} trees={ntrees} pool={len(pool)}")
    results.sort(key=lambda r: r["logloss"])
    A_name = "A-unweighted-asis"

    # ---------- STEP 2: bootstrap top-3 vs A ----------
    def per_shot_ll(p):
        return -(yh * np.log(p) + (1 - yh) * np.log(1 - p))
    ll_A = per_shot_ll(hold_pred[A_name])
    game_ids = hold.game_id.values
    uniq_games = np.array(sorted(set(game_ids)))
    game_to_idx = {g: np.where(game_ids == g)[0] for g in uniq_games}
    rng = np.random.RandomState(SEED)
    B = 1000
    boot_game_samples = [rng.choice(uniq_games, size=len(uniq_games), replace=True) for _ in range(B)]

    def boot_diff(ll_cfg):
        diffs = []
        for gs in boot_game_samples:
            idx = np.concatenate([game_to_idx[g] for g in gs])
            diffs.append(ll_cfg[idx].mean() - ll_A[idx].mean())
        d = np.array(diffs)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(d.mean())

    top3 = [r for r in results if r["name"] != A_name][:3]
    bootstrap = {}
    for r in top3:
        lo, hi, mean = boot_diff(per_shot_ll(hold_pred[r["name"]]))
        sig = hi < 0            # entire 95% CI below 0 => significantly better (lower logloss) than A
        bootstrap[r["name"]] = dict(ci_lo=lo, ci_hi=hi, mean=mean, sig_better=bool(sig),
                                    logloss=r["logloss"])
        print(f"  bootstrap {r['name']}: dLogloss mean={mean:+.5f} 95%CI[{lo:+.5f},{hi:+.5f}] sig_better={sig}")

    # selection rule
    A_ll = next(r for r in results if r["name"] == A_name)["logloss"]
    sig_configs = [r for r in top3 if bootstrap[r["name"]]["sig_better"]]
    winner = None
    if sig_configs:
        winner = min(sig_configs, key=lambda r: r["logloss"])
        # simplicity preference if others within winner's bootstrap noise handled in report
    dump = dict(step1=results, A_name=A_name, A_logloss=A_ll, bootstrap=bootstrap,
                winner=(winner["name"] if winner else None))
    json.dump(dump, open(f"{SC}/v2_step12.json", "w"), indent=2)
    print("\nWINNER:", winner["name"] if winner else "NONE beats A significantly -> STOP (no v2)")
    if winner is None:
        return

    # ---------- STEP 3: re-tune winner ----------
    wcfg = next(c for c in configs if c["name"] == winner["name"])
    pool, w = build_pool(wcfg, seasons_data, holdout_games)
    poolP = prep(pool)
    grid = list(itertools.product([4, 5, 6, 7], [0.03, 0.05], [5, 10, 20], [0.8, 1.0], [1.0]))
    # keep grid modest: sample deterministically to <=16
    rng2 = np.random.RandomState(SEED)
    if len(grid) > 16:
        sel = rng2.choice(len(grid), size=16, replace=False)
        grid = [grid[i] for i in sorted(sel)]
    gkf = GroupKFold(n_splits=5)
    gridres = []
    for (md, lr, mcw, ss, cs) in grid:
        params = dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, subsample=ss, colsample_bytree=cs)
        lls, its = [], []
        for tr, va in gkf.split(poolP[FEATURES], pool.is_goal.values, pool.game_id.values):
            m = mk(2000, params); m.set_params(early_stopping_rounds=50)
            m.fit(poolP.iloc[tr][FEATURES], pool.is_goal.values[tr], sample_weight=w[tr],
                  eval_set=[(poolP.iloc[va][FEATURES], pool.is_goal.values[va])], verbose=False)
            p = m.predict_proba(poolP.iloc[va][FEATURES])[:, 1]
            lls.append(log_loss(pool.is_goal.values[va], np.clip(p, 1e-6, 1 - 1e-6)))
            its.append(m.best_iteration + 1)
        gridres.append(dict(params=params, cv_logloss=float(np.mean(lls)), cv_std=float(np.std(lls)),
                            n_est=int(np.mean(its))))
        print(f"  tune {params} cv={np.mean(lls):.5f} n_est~{int(np.mean(its))}")
    gridres.sort(key=lambda r: r["cv_logloss"])
    best = gridres[0]
    BP = best["params"]; NEST = best["n_est"]
    print("BEST tuned:", BP, "n_est", NEST)

    # full holdout eval with tuned params (train on winner pool, fixed n_est)
    mfull = mk(NEST, BP); mfull.fit(poolP[FEATURES], pool.is_goal.values, sample_weight=w)
    holdX = prep(merge_shot_types(hold) if wcfg["shot_types"] == "merged" else hold)
    p_v2 = np.clip(mfull.predict_proba(holdX[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    p_v1 = np.clip(hold.xg_v1.values, 1e-6, 1 - 1e-6)   # positional alignment
    step3 = dict(grid=gridres, tuned=BP, n_est=NEST,
                 holdout=dict(v2=metrics(yh, p_v2), v1=metrics(yh, p_v1), old=metrics(yh, hold_old),
                              baseline_logloss=float(log_loss(yh, np.full_like(p_v2, pool.is_goal.mean())))))
    # calibration deciles / strength / shot type
    def deciles(p):
        df = pd.DataFrame({"p": p, "y": yh}); df["d"] = pd.qcut(df.p, 10, labels=False, duplicates="drop")
        return df.groupby("d").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean")).reset_index().to_dict("records")
    def sgroup(sf, sa):
        if sf == 6: return "extra_attacker"
        if sf == 5 and sa == 5: return "5v5"
        if sf > sa: return "PP"
        if sf < sa: return "PK"
        return "3v3/4v4"
    hg = [sgroup(a, b) for a, b in zip(hold.skaters_for, hold.skaters_against)]
    strcal = {}
    for g in set(hg):
        ii = np.array([x == g for x in hg])
        strcal[g] = dict(n=int(ii.sum()), act=int(yh[ii].sum()), v2=float(p_v2[ii].sum()),
                         v1=float(p_v1[ii].sum()), old=float(hold_old[ii].sum()))
    LAB = {1: "wrist", 2: "snap", 3: "slap", 4: "backhand", 5: "tip-in", 6: "deflected",
           7: "wrap-around", 8: "poke", 9: "bat", 10: "cradle", 11: "between-legs"}
    st_used = (merge_shot_types(hold) if wcfg["shot_types"] == "merged" else hold).shot_type_id.values
    stcal = {}
    for st in sorted(set(st_used.tolist())):
        ii = st_used == st
        lab = "wrist+snap" if (wcfg["shot_types"] == "merged" and st == 1) else LAB.get(st, str(st))
        stcal[lab] = dict(n=int(ii.sum()), act=int(yh[ii].sum()), v2=float(p_v2[ii].sum()),
                          v1=float(p_v1[ii].sum()), old=float(hold_old[ii].sum()))
    step3["cal_deciles_v2"] = deciles(p_v2)
    step3["cal_deciles_v1"] = deciles(p_v1)
    step3["cal_strength"] = strcal
    step3["cal_shottype"] = stcal

    # ---------- STEP 4: OOF v2 over 2025-26 + final model ----------
    train25 = all25.copy()
    if wcfg["shot_types"] == "merged":
        prior = [merge_shot_types(seasons_data[s]) for s in wcfg["seasons"] if s != "2025_26"]
    else:
        prior = [seasons_data[s] for s in wcfg["seasons"] if s != "2025_26"]
    prior_df = pd.concat(prior, ignore_index=True) if prior else None
    gkf5 = GroupKFold(n_splits=5)
    oof = np.zeros(len(train25)); fold = np.full(len(train25), -1)
    base25 = merge_shot_types(train25) if wcfg["shot_types"] == "merged" else train25
    for k, (tr, va) in enumerate(gkf5.split(base25[FEATURES], base25.is_goal.values, base25.game_id.values)):
        parts = [base25.iloc[tr]]
        if prior_df is not None:
            parts.append(prior_df)
        tp = pd.concat(parts, ignore_index=True)
        if wcfg["weight"] == "recency":
            wtp = tp["season"].map(RECENCY).values
        else:
            wtp = np.ones(len(tp))
        m = mk(NEST, BP); m.fit(prep(tp)[FEATURES], tp.is_goal.values, sample_weight=wtp)
        oof[va] = m.predict_proba(prep(base25.iloc[va])[FEATURES])[:, 1]
        fold[va] = k
    total_oof = float(oof.sum()); actual = int(train25.is_goal.sum())
    pct = (total_oof - actual) / actual * 100
    print(f"\nSTEP4 OOF v2 total xG={total_oof:.1f} vs actual={actual} diff={pct:+.2f}%")
    step4 = dict(total_oof=total_oof, actual=actual, pct=pct,
                 oof_logloss=float(log_loss(train25.is_goal.values, np.clip(oof, 1e-6, 1-1e-6))),
                 oof_auc=float(roc_auc_score(train25.is_goal.values, oof)))
    if abs(pct) > 2.0:
        step4["STOP"] = True
        json.dump(dict(step3=step3, step4=step4), open(f"{SC}/v2_step345.json", "w"), indent=2, default=float)
        print("STOP: OOF calibration off by >2% -> not writing outputs")
        return

    # write v2 shots parquet
    out = train25[["game_id", "abs_secs", "shooter_id"]].copy()
    out["xg_v2"] = oof.round(6); out["fold"] = fold
    out["xg_v1"] = train25.xg_v1.values          # positional
    out["xg_old"] = train25.xg_old.values
    out.to_parquet(os.path.join(HERE, "shots_xg_2025_26_v2.parquet"), index=False)

    # final model on full pool (all prior seasons + ALL of 2025-26)
    finalparts = [base25]
    if prior_df is not None: finalparts.append(prior_df)
    fpool = pd.concat(finalparts, ignore_index=True)
    wf = fpool["season"].map(RECENCY).values if wcfg["weight"] == "recency" else np.ones(len(fpool))
    fm = mk(NEST, BP); fm.fit(prep(fpool)[FEATURES], fpool.is_goal.values, sample_weight=wf)
    fm.get_booster().save_model(os.path.join(HERE, "xg_model_2025_26_v2.json"))
    print("saved xg_model_2025_26_v2.json + shots_xg_2025_26_v2.parquet")

    # ---------- STEP 5: player check (all v1/v2/old aligned POSITIONALLY) ----------
    keyed = train25[["shooter_id", "is_goal", "xg_old"]].copy().reset_index(drop=True)
    keyed["xg_v2"] = oof
    keyed["xg_v1"] = train25.xg_v1.values
    keyed["is5v5"] = (train25.skaters_for.values == 5) & (train25.skaters_against.values == 5)
    players = {"McDavid": 8478402, "Kucherov": 8476453, "MacKinnon": 8477492,
               "Brady Tkachuk": 8480801, "Mark Stone": 8475913}
    pcheck = {}
    for nm, pid in players.items():
        s = keyed[keyed.shooter_id == pid]; s5 = s[s.is5v5]
        pcheck[nm] = dict(goals=int(s.is_goal.sum()), ixg_v1=float(s.xg_v1.sum()), ixg_v2=float(s.xg_v2.sum()),
                          ixg_old=float(s.xg_old.sum()),
                          goals5=int(s5.is_goal.sum()), ixg_v1_5=float(s5.xg_v1.sum()),
                          ixg_v2_5=float(s5.xg_v2.sum()), ixg_old_5=float(s5.xg_old.sum()))
    agg = keyed.groupby("shooter_id").agg(ixg_v2=("xg_v2", "sum"), ixg_v1=("xg_v1", "sum"),
                                          goals=("is_goal", "sum"), shots=("is_goal", "size")).reset_index()
    top20 = agg.sort_values("ixg_v2", ascending=False).head(20).to_dict("records")
    big = agg[agg.shots >= 50].copy()
    big["abs_change"] = (big.ixg_v2 - big.ixg_v1).abs()
    big["pct_change"] = (big.ixg_v2 - big.ixg_v1) / big.ixg_v1 * 100
    biggest = big.sort_values("abs_change", ascending=False).head(10).to_dict("records")
    step5 = dict(players=pcheck, top20=top20, biggest_change=biggest, merged=(wcfg["shot_types"] == "merged"))

    json.dump(dict(step3=step3, step4=step4, step5=step5, winner=wcfg),
              open(f"{SC}/v2_step345.json", "w"), indent=2, default=float)
    print("DONE")


if __name__ == "__main__":
    main()
