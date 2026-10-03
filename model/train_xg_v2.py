"""
train_xg_v2.py
--------------
v2 xG with a SEASON feature. Configuration C (2023-24 + 2024-25 + 2025-26,
unweighted, 12 shot types) plus one new feature: `season` = integer start year
(2023, 2024, 2025). Hypothesis (confirmed in Step 1): some shot types converted
differently across seasons (tip-in strongest), so a pooled model without season
mis-levels 2025-26; the season feature lets the model score 2025-26 at 2025-26
rates.

Deterministic (random_state=42). v1 files are never modified. All v2 outputs
carry a _v2 suffix and are written ONLY if every gate passes.

Fixed holdout = same last 263 games of 2025-26 by date (22,089 rows). Training
pool = first 80% of 2025-26 + all of 2023-24 + all of 2024-25.

Public scoring API: score_shots(df, season=2025) — see bottom.
"""
from __future__ import annotations
import json, os, itertools
import numpy as np, pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

SEED = 42
np.random.seed(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))
SC = "/private/tmp/claude-501/-Users-bauerswift-NHL-Front-Office-AI-Integration/702051fb-ab76-4f35-9b0d-24052e0f7296/scratchpad"

BASE_FEATURES = ["distance_from_net", "angle_from_center", "x_norm", "abs_y_norm", "shot_type_id",
                 "is_rebound", "time_since_last_shot", "is_rush", "score_state", "period",
                 "time_in_period", "is_home", "skaters_for", "skaters_against", "off_wing"]
FEATURES = BASE_FEATURES + ["season"]
V1_PARAMS = dict(max_depth=4, learning_rate=0.05, min_child_weight=5, subsample=0.8, colsample_bytree=1.0)
SEASON_YEAR = {"2023_24": 2023, "2024_25": 2024, "2025_26": 2025}
POOL_SEASONS = ["2023_24", "2024_25", "2025_26"]   # configuration C
LAB = {1: "wrist", 2: "snap", 3: "slap", 4: "backhand", 5: "tip-in", 6: "deflected",
       7: "wrap-around", 8: "poke", 9: "bat", 10: "cradle", 11: "between-legs"}


def load(s):
    d = pd.read_parquet(os.path.join(HERE, f"xg_features_{s}.parquet"))
    d = d[d.train_drop_reason.isna()].reset_index(drop=True).copy()
    d["season"] = SEASON_YEAR[s]
    return d


def prep(df):
    df = df.copy(); df["shot_type_id"] = df["shot_type_id"].astype("category"); return df


def mk(n_estimators, params):
    return xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                             n_estimators=n_estimators, tree_method="hist", enable_categorical=True,
                             random_state=SEED, n_jobs=8, **params)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(logloss=float(log_loss(y, p)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), pred_goals=float(p.sum()))


def cv_logloss(pool, params, n_estimators=2000):
    gkf = GroupKFold(n_splits=5)
    P = prep(pool); y = pool.is_goal.values; g = pool.game_id.values
    lls, its = [], []
    for tr, va in gkf.split(P[FEATURES], y, g):
        m = mk(n_estimators, params); m.set_params(early_stopping_rounds=50)
        m.fit(P.iloc[tr][FEATURES], y[tr], eval_set=[(P.iloc[va][FEATURES], y[va])], verbose=False)
        p = m.predict_proba(P.iloc[va][FEATURES])[:, 1]
        lls.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6))); its.append(m.best_iteration + 1)
    return float(np.mean(lls)), float(np.std(lls)), int(np.mean(its))


def main():
    data = {s: load(s) for s in POOL_SEASONS}
    all25 = data["2025_26"]
    # positional v1 OOF alignment (keys non-unique due to rapid-fire pairs)
    v1sx = pd.read_parquet(os.path.join(HERE, "shots_xg_2025_26.parquet"))
    assert (all25[["game_id", "abs_secs", "shooter_id"]].values == v1sx[["game_id", "abs_secs", "shooter_id"]].values).all()
    all25 = all25.copy(); all25["xg_v1"] = v1sx.xg_new.values; data["2025_26"] = all25

    gd = all25[["game_id", "game_date"]].drop_duplicates().sort_values(["game_date", "game_id"])
    holdout_games = set(gd.iloc[int(len(gd) * 0.8):].game_id)
    hold = all25[all25.game_id.isin(holdout_games)].reset_index(drop=True)
    yh = hold.is_goal.values
    hold_old = np.clip(hold.xg_old.values, 1e-6, 1 - 1e-6)
    p_v1 = np.clip(hold.xg_v1.values, 1e-6, 1 - 1e-6)

    pool = pd.concat([data["2023_24"], data["2024_25"], all25[~all25.game_id.isin(holdout_games)]], ignore_index=True)
    print(f"pool rows={len(pool)} | holdout rows={len(hold)} goals={int(yh.sum())}")

    results = {}

    # -------- STEP 2: tune --------
    grid = [dict(max_depth=md, learning_rate=lr, min_child_weight=mcw, subsample=ss, colsample_bytree=1.0)
            for md, lr, mcw, ss in itertools.product([4, 5, 6], [0.03, 0.05], [5, 10, 20], [0.8, 1.0])]
    rng = np.random.RandomState(SEED)
    if len(grid) > 14:
        grid = [grid[i] for i in sorted(rng.choice(len(grid), 14, replace=False))]
    tune = []
    for params in grid:
        ll, sd, nest = cv_logloss(pool, params)
        tune.append(dict(params=params, cv_logloss=ll, cv_std=sd, n_est=nest))
        print(f"  tune {params} cv={ll:.5f} n_est~{nest}")
    v1ll, v1sd, v1nest = cv_logloss(pool, V1_PARAMS)
    tune.sort(key=lambda r: r["cv_logloss"])
    best = tune[0]; BP = best["params"]; NEST = best["n_est"]
    print(f"  [ref] v1 params cv={v1ll:.5f} n_est~{v1nest}")
    print("BEST:", BP, "n_est", NEST)
    results["step2_tune"] = dict(grid=tune, v1_params_cv=dict(cv_logloss=v1ll, cv_std=v1sd, n_est=v1nest),
                                 chosen=BP, n_est=NEST)

    # fit on pool for holdout eval + season diagnostics
    poolP = prep(pool)
    m = mk(NEST, BP); m.fit(poolP[FEATURES], pool.is_goal.values)
    imp = m.get_booster().get_score(importance_type="gain")
    fmap = {f"f{i}": FEATURES[i] for i in range(len(FEATURES))}
    imp = {fmap.get(k, k): v for k, v in imp.items()}
    ranked = sorted(imp.items(), key=lambda x: -x[1])
    season_rank = [i for i, (k, _) in enumerate(ranked, 1) if k == "season"]
    results["step2_importance"] = dict(ranked=[(k, float(v)) for k, v in ranked],
                                       season_rank=(season_rank[0] if season_rank else None))
    print("season importance rank:", season_rank[0] if season_rank else None, "of", len(ranked))

    # season what-if: score holdout 3 ways, mean predicted rate by shot type
    whatif = {}
    for yr in (2023, 2024, 2025):
        hh = hold.copy(); hh["season"] = yr
        pp = m.predict_proba(prep(hh)[FEATURES])[:, 1]
        by = pd.DataFrame({"st": hold.shot_type_id.values, "p": pp}).groupby("st").p.mean()
        whatif[yr] = {LAB.get(int(k), int(k)): float(v) for k, v in by.items()}
    results["step2_season_whatif"] = whatif
    print("season what-if (mean pred rate slap/tip-in):",
          {yr: {k: round(whatif[yr][k], 4) for k in ("slap", "tip-in")} for yr in whatif})

    # -------- STEP 3: holdout eval (this vs v1 vs C-no-season vs old) --------
    p_v2 = np.clip(m.predict_proba(prep(hold)[FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    # C without season, tuned params from v2 report (depth5/lr0.03), retrained here
    Cparams = dict(max_depth=5, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=1.0)
    poolC = pd.concat([data["2023_24"], data["2024_25"], all25[~all25.game_id.isin(holdout_games)]], ignore_index=True)
    mC = mk(472, Cparams); mC.fit(prep(poolC)[BASE_FEATURES], poolC.is_goal.values)
    p_Cns = np.clip(mC.predict_proba(prep(hold)[BASE_FEATURES])[:, 1], 1e-6, 1 - 1e-6)

    hold_m = dict(v2_season=metrics(yh, p_v2), v1=metrics(yh, p_v1), C_no_season=metrics(yh, p_Cns),
                  old_leaked=metrics(yh, hold_old),
                  baseline_logloss=float(log_loss(yh, np.full_like(p_v2, pool.is_goal.mean()))),
                  actual_goals=int(yh.sum()))

    def deciles(p):
        df = pd.DataFrame({"p": p, "y": yh}); df["d"] = pd.qcut(df.p, 10, labels=False, duplicates="drop")
        return df.groupby("d").agg(n=("y", "size"), pred=("p", "mean"), act=("y", "mean")).reset_index().to_dict("records")

    def sgroup(sf, sa):
        if sf == 6: return "extra_attacker"
        if sf == 5 and sa == 5: return "5v5"
        if sf > sa: return "PP"
        if sf < sa: return "PK"
        return "3v3/4v4"
    hg = np.array([sgroup(a, b) for a, b in zip(hold.skaters_for, hold.skaters_against)])
    strcal = {}
    for gname in sorted(set(hg)):
        ii = hg == gname
        strcal[gname] = dict(n=int(ii.sum()), act=int(yh[ii].sum()), v2=float(p_v2[ii].sum()),
                             v1=float(p_v1[ii].sum()), Cns=float(p_Cns[ii].sum()), old=float(hold_old[ii].sum()))
    stcal = {}
    for st in sorted(hold.shot_type_id.unique().tolist()):
        ii = hold.shot_type_id.values == st
        stcal[LAB.get(st, st)] = dict(n=int(ii.sum()), act=int(yh[ii].sum()), v2=float(p_v2[ii].sum()),
                                      v1=float(p_v1[ii].sum()), Cns=float(p_Cns[ii].sum()), old=float(hold_old[ii].sum()))

    # bootstrap logloss diff vs v1
    def per_ll(p): return -(yh * np.log(p) + (1 - yh) * np.log(1 - p))
    ll2, ll1 = per_ll(p_v2), per_ll(p_v1)
    gids = hold.game_id.values; ug = np.array(sorted(set(gids)))
    g2i = {g: np.where(gids == g)[0] for g in ug}
    rng2 = np.random.RandomState(SEED)
    diffs = []
    for _ in range(1000):
        gs = rng2.choice(ug, len(ug), replace=True)
        idx = np.concatenate([g2i[g] for g in gs])
        diffs.append(ll2[idx].mean() - ll1[idx].mean())
    diffs = np.array(diffs)
    boot = dict(mean=float(diffs.mean()), ci_lo=float(np.percentile(diffs, 2.5)),
                ci_hi=float(np.percentile(diffs, 97.5)))
    results["step3"] = dict(metrics=hold_m, deciles_v2=deciles(p_v2), strength=strcal, shottype=stcal,
                            bootstrap_vs_v1=boot)
    print(f"HOLDOUT v2={hold_m['v2_season']['logloss']:.5f} v1={hold_m['v1']['logloss']:.5f} "
          f"C_no_season={hold_m['C_no_season']['logloss']:.5f} old={hold_m['old_leaked']['logloss']:.5f}")
    print(f"bootstrap dLL vs v1: mean={boot['mean']:+.5f} CI[{boot['ci_lo']:+.5f},{boot['ci_hi']:+.5f}]")

    # -------- STEP 4: OOF + gates --------
    prior = pd.concat([data["2023_24"], data["2024_25"]], ignore_index=True)
    gkf5 = GroupKFold(n_splits=5)
    oof = np.zeros(len(all25)); fold = np.full(len(all25), -1)
    for k, (tr, va) in enumerate(gkf5.split(all25[FEATURES], all25.is_goal.values, all25.game_id.values)):
        tp = pd.concat([all25.iloc[tr], prior], ignore_index=True)
        mm = mk(NEST, BP); mm.fit(prep(tp)[FEATURES], tp.is_goal.values)
        oof[va] = mm.predict_proba(prep(all25.iloc[va])[FEATURES])[:, 1]; fold[va] = k
    total_oof = float(oof.sum()); actual = int(all25.is_goal.sum())
    pct = (total_oof - actual) / actual * 100

    # GATE 1
    gate1 = abs(pct) <= 2.0
    # GATE 2: per shot type ratio (all reported; gate on >=150 goals)
    st_ratio = {}
    for st in sorted(all25.shot_type_id.unique().tolist()):
        ii = all25.shot_type_id.values == st
        a = int(all25.is_goal.values[ii].sum()); pr = float(oof[ii].sum())
        st_ratio[LAB.get(st, st)] = dict(n=int(ii.sum()), goals=a, pred=pr, ratio=(a / pr if pr > 0 else None))
    gate2_fail = [k for k, v in st_ratio.items() if v["goals"] >= 150 and v["ratio"] is not None and not (0.90 <= v["ratio"] <= 1.10)]
    gate2 = len(gate2_fail) == 0
    # GATE 3: holdout logloss sig better than v1
    gate3 = boot["ci_hi"] < 0
    results["step4"] = dict(total_oof=total_oof, actual=actual, pct=pct,
                            oof_logloss=float(log_loss(all25.is_goal.values, np.clip(oof, 1e-6, 1 - 1e-6))),
                            oof_auc=float(roc_auc_score(all25.is_goal.values, oof)),
                            gate1_total_2pct=dict(passed=bool(gate1), pct=pct),
                            gate2_shottype=dict(passed=bool(gate2), failures=gate2_fail, ratios=st_ratio),
                            gate3_sig_vs_v1=dict(passed=bool(gate3), ci=[boot["ci_lo"], boot["ci_hi"]]))
    print(f"\nGATE1 total OOF {total_oof:.1f} vs {actual} ({pct:+.2f}%) -> {'PASS' if gate1 else 'FAIL'}")
    print("GATE2 shot-type ratios (>=150 goals):")
    for k, v in st_ratio.items():
        mark = " (gated)" if v["goals"] >= 150 else ""
        print(f"   {k:12s} n={v['n']:6d} goals={v['goals']:5d} pred={v['pred']:7.1f} ratio={v['ratio'] if v['ratio'] else float('nan'):.3f}{mark}")
    print(f"GATE2 -> {'PASS' if gate2 else 'FAIL'} failures={gate2_fail}")
    print(f"GATE3 sig vs v1 -> {'PASS' if gate3 else 'FAIL'} (CI hi={boot['ci_hi']:+.5f})")

    all_pass = gate1 and gate2 and gate3
    results["all_gates_pass"] = bool(all_pass)

    if not all_pass:
        json.dump(results, open(f"{SC}/vseason_results.json", "w"), indent=2, default=float)
        print("\n>>> A GATE FAILED -> writing NO v2 files. STOP.")
        return

    # -------- STEP 5: final model + outputs + player check --------
    out = all25[["game_id", "abs_secs", "shooter_id"]].copy()
    out["xg_v2"] = oof.round(6); out["fold"] = fold
    out["xg_v1"] = all25.xg_v1.values; out["xg_old"] = all25.xg_old.values
    out.to_parquet(os.path.join(HERE, "shots_xg_2025_26_v2.parquet"), index=False)

    fpool = pd.concat([data["2023_24"], data["2024_25"], all25], ignore_index=True)
    fm = mk(NEST, BP); fm.fit(prep(fpool)[FEATURES], fpool.is_goal.values)
    fm.get_booster().save_model(os.path.join(HERE, "xg_model_2025_26_v2.json"))

    keyed = all25[["shooter_id", "is_goal", "shot_type_id"]].copy()
    keyed["xg_v2"] = oof; keyed["xg_v1"] = all25.xg_v1.values; keyed["xg_old"] = all25.xg_old.values
    keyed["is5v5"] = (all25.skaters_for.values == 5) & (all25.skaters_against.values == 5)
    players = {"McDavid": 8478402, "Kucherov": 8476453, "MacKinnon": 8477492,
               "Brady Tkachuk": 8480801, "Mark Stone": 8475913}
    pcheck = {}
    for nm, pid in players.items():
        s = keyed[keyed.shooter_id == pid]; s5 = s[s.is5v5]
        slaptip = s.shot_type_id.isin([3, 5]).mean()
        pcheck[nm] = dict(goals=int(s.is_goal.sum()), ixg_v1=float(s.xg_v1.sum()), ixg_v2=float(s.xg_v2.sum()),
                          ixg_old=float(s.xg_old.sum()), goals5=int(s5.is_goal.sum()),
                          ixg_v1_5=float(s5.xg_v1.sum()), ixg_v2_5=float(s5.xg_v2.sum()),
                          ixg_old_5=float(s5.xg_old.sum()), slap_tip_share=float(slaptip))
    agg = keyed.groupby("shooter_id").agg(ixg_v2=("xg_v2", "sum"), ixg_v1=("xg_v1", "sum"),
                                          goals=("is_goal", "sum"), shots=("is_goal", "size"),
                                          slap_tip=("shot_type_id", lambda s: s.isin([3, 5]).mean())).reset_index()
    top20 = agg.sort_values("ixg_v2", ascending=False).head(20).to_dict("records")
    big = agg[agg.shots >= 50].copy()
    big["abs_change"] = (big.ixg_v2 - big.ixg_v1).abs()
    big["pct_change"] = (big.ixg_v2 - big.ixg_v1) / big.ixg_v1 * 100
    biggest = big.sort_values("abs_change", ascending=False).head(10).to_dict("records")
    results["step5"] = dict(players=pcheck, top20=top20, biggest_change=biggest)
    json.dump(results, open(f"{SC}/vseason_results.json", "w"), indent=2, default=float)
    print("\nALL GATES PASSED -> wrote xg_model_2025_26_v2.json + shots_xg_2025_26_v2.parquet")


# ---------------------------------------------------------------------------
# Public scoring API for future games. season defaults to 2025 (most recent
# trained season) so new games are scored at 2025-26 levels until retrained.
# ---------------------------------------------------------------------------
def score_shots(df: pd.DataFrame, season: int = 2025) -> np.ndarray:
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(os.path.join(HERE, "xg_model_2025_26_v2.json"))
    x = df.copy()
    x["season"] = season
    x["shot_type_id"] = x["shot_type_id"].astype("category")
    return model.predict_proba(x[FEATURES])[:, 1]


if __name__ == "__main__":
    main()
