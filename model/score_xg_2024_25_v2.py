"""
score_xg_2024_25_v2.py
----------------------
Step 1b of the 5v5 RAPM build: score every 2024-25 shot OUT OF FOLD with the
v2 xG configuration, to feed the 2024-25 RAPM prior.

Reuses train_xg_v2.py verbatim (imported, NOT modified): the same FEATURES, the
same tuned hyperparameters (max_depth=5, lr=0.03, min_child_weight=5,
subsample=0.8, colsample=1.0), and n_estimators=435 (the tree count of the final
v2 model, model/xg_model_2025_26_v2.json). The `season` feature carries each
row's true start year (2024-25 rows -> 2024).

Procedure: 5-fold GroupKFold by game WITHIN 2024-25. Each fold's model is trained
on the other 2024-25 folds PLUS all of 2023-24 and all of 2025-26, then predicts
the held-out 2024-25 fold. Deterministic (random_state=42, GroupKFold unshuffled).

Output: model/shots_xg_2024_25_v2.parquet  (game_id, abs_secs, shooter_id, xg_v2, fold)
Gate: |total xG - actual goals| must be <= 2% (else report & STOP; no file kept
if it would mislead -- we still write but flag).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import train_xg_v2 as V2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "shots_xg_2024_25_v2.parquet")

BP = dict(max_depth=5, learning_rate=0.03, min_child_weight=5,
          subsample=0.8, colsample_bytree=1.0)   # v2 tuned params
NEST = 435                                        # v2 final model tree count


def main():
    d23 = V2.load("2023_24")
    d24 = V2.load("2024_25")
    d25 = V2.load("2025_26")
    prior_pool = pd.concat([d23, d25], ignore_index=True)   # cross-season context
    print(f"2024-25 kept rows={len(d24)} goals={int(d24.is_goal.sum())} | "
          f"context 2023-24={len(d23)} 2025-26={len(d25)}")

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(d24))
    fold = np.full(len(d24), -1, dtype=int)
    for k, (tr, va) in enumerate(gkf.split(d24[V2.FEATURES], d24.is_goal.values,
                                           d24.game_id.values)):
        tp = pd.concat([d24.iloc[tr], prior_pool], ignore_index=True)
        m = V2.mk(NEST, BP)
        m.fit(V2.prep(tp)[V2.FEATURES], tp.is_goal.values)
        oof[va] = m.predict_proba(V2.prep(d24.iloc[va])[V2.FEATURES])[:, 1]
        fold[va] = k
        print(f"  fold {k}: train={len(tp)} val={len(va)} "
              f"val_xg={oof[va].sum():.1f} val_goals={int(d24.is_goal.values[va].sum())}")

    total_xg = float(oof.sum())
    actual = int(d24.is_goal.sum())
    pct = (total_xg - actual) / actual * 100
    print(f"\nTOTAL xG={total_xg:.1f} vs actual goals={actual}  ({pct:+.2f}%)")

    out = d24[["game_id", "abs_secs", "shooter_id"]].copy()
    out["xg_v2"] = oof.round(6)
    out["fold"] = fold
    out.to_parquet(OUT, index=False)
    print("WROTE", OUT, "rows", len(out))

    if abs(pct) > 2.0:
        print(f">>> GATE FAILED: |{pct:.2f}%| > 2% -- STOP and report.")
    else:
        print(f">>> GATE PASSED: |{pct:.2f}%| <= 2%.")
    return total_xg, actual, pct


if __name__ == "__main__":
    main()
