"""
build_xg_features_2025_26.py
----------------------------
Deterministic feature builder for the from-scratch 2025-26 regular-season xG model.
Reads model/shots_enriched_2025_26.parquet (READ ONLY) and writes the cleaned
modeling table to model/xg_features_2025_26.parquet.

All prep logic from Part 1 + items A/B lives here so Part 2 (and any future run)
reads a single reproducible file rather than an in-memory session.

Pipeline
  1. Scope filter: game_id in [2025020001, 2025021312]; drop period-5 / shootout
     rows (abs_secs == 4800, goalie_id NULL).
  2. NO duplicate drop.  The 10 same-second shooter/coordinate "pairs" were each
     verified against NHL API play-by-play (item A) to be TWO DISTINCT events:
     two distinct eventIds with consecutive sortOrders, and the official SOG
     counter increments by 1 between them (or one is shot-on-goal + the other a
     missed-shot).  They are genuine rapid-fire sequences (shot + immediate
     rebound), NOT duplications.  Do NOT reintroduce a duplicate drop here.
  3. Shot type: category 0 = "unknown" (NHL feed logged no shotType).  All 24
     category-0 rows are goals; item B verified via the API that 0 of the 24 goal
     events carry a shotType in the feed, so none are backfillable.  They are
     imputed to the modeling-set mode (wrist = 1) and flagged shot_type_imputed.
     (API backfill was attempted in item B and yielded 0 resolvable; imputation is
     hard-coded here to keep this script deterministic / network-free.)
  4. Handedness join from model/player_handedness.csv (cached in Part 1).
  5. off_wing using the empirically verified convention y_norm > 0 = shooter's LEFT
     (item 4: geometric trace of raw NHL coords + PP one-timer pattern).
  6. Garbage strength states flagged (NOT silently removed): rows kept in the
     output with train_drop_reason set, so Part 2 filters them out explicitly.

Output columns: shot keys (game_id, abs_secs, shooter_id), game_date, the 15
model features, is_goal (target), xg_old (prior model, benchmark only — NOT a
feature), shot_type_imputed, and train_drop_reason (NULL = used for training).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ENRICHED = os.path.join(HERE, "shots_enriched_2025_26.parquet")
HAND = os.path.join(HERE, "player_handedness.csv")
DATES = os.path.join(HERE, "game_dates_2025_26.csv")
OUT = os.path.join(HERE, "xg_features_2025_26.parquet")

GID_LO, GID_HI = 2025020001, 2025021312

SHOT_TYPE_LABELS = {0: "unknown", 1: "wrist", 2: "snap", 3: "slap", 4: "backhand",
                    5: "tip-in", 6: "deflected", 7: "wrap-around", 8: "poke",
                    9: "bat", 10: "cradle", 11: "between-legs"}

FEATURE_COLS = [
    "distance_from_net", "angle_from_center", "x_norm", "abs_y_norm",
    "shot_type_id", "is_rebound", "time_since_last_shot", "is_rush",
    "score_state", "period", "time_in_period", "is_home",
    "skaters_for", "skaters_against", "off_wing",
]


def is_garbage_strength(sf: int, sa: int) -> bool:
    """Reconstruction artifacts to exclude from training (item 5)."""
    if not (3 <= sf <= 6 and 3 <= sa <= 6):
        return True                      # count outside 3-6
    if abs(sf - sa) > 2:
        return True                      # >2 skater differential impossible
    if sf == 6 and sa == 6:
        return True                      # both goalies pulled — impossible
    if sa == 6:
        return True                      # defending 6 skaters => empty-net-against,
                                         # which does not occur in this set (spurious)
    return False


def build() -> pd.DataFrame:
    E = pd.read_parquet(ENRICHED)

    # 1. scope filter
    f = E[(E.game_id >= GID_LO) & (E.game_id <= GID_HI)].copy()
    f = f[~((f.period == 5) & (f.abs_secs == 4800))].copy()
    n_scope = len(f)

    # 2. NO duplicate drop (see module docstring / item A).

    # 3. shot type imputation (category 0 -> mode; flag imputed)
    mode_id = int(f.loc[f.shot_type_id != 0, "shot_type_id"].mode().iloc[0])  # = 1 (wrist)
    f["shot_type_imputed"] = f.shot_type_id == 0
    f["shot_type_id"] = f.shot_type_id.replace(0, mode_id).astype(int)

    # 4. handedness join
    hand = pd.read_csv(HAND)
    f = f.merge(hand, left_on="shooter_id", right_on="player_id", how="left")

    # 5. off_wing: y_norm > 0 = shooter LEFT  ->  off-wing = L-shot on right / R-shot on left
    off = np.where(f.shoots.isna(), np.nan,
            np.where((f.shoots == "L") & (f.y_norm < 0), 1.0,
            np.where((f.shoots == "R") & (f.y_norm > 0), 1.0, 0.0)))
    f["off_wing"] = off

    # derived features
    f["abs_y_norm"] = f.y_norm.abs()
    f["time_in_period"] = f.abs_secs - (f.period - 1) * 1200
    f["score_state"] = f.score_state.clip(-3, 3)

    # game dates (for the chronological holdout in Part 2)
    dates = pd.read_csv(DATES)
    f = f.merge(dates, on="game_id", how="left")

    # 6. flag garbage strength states (kept in output, excluded from training)
    garb = f.apply(lambda r: is_garbage_strength(int(r.skaters_for), int(r.skaters_against)), axis=1)
    f["train_drop_reason"] = np.where(garb, "garbage_strength_state", pd.NA)

    f["skaters_for"] = f.skaters_for.astype(int)
    f["skaters_against"] = f.skaters_against.astype(int)

    keep_cols = (["game_id", "abs_secs", "shooter_id", "game_date"]
                 + FEATURE_COLS + ["is_goal", "shot_type_imputed", "train_drop_reason"])
    out = f[keep_cols].copy()
    out = out.rename(columns={})  # keep names
    out["xg_old"] = f["xg"].values   # prior model output — benchmark only, NOT a feature

    out.to_parquet(OUT, index=False)

    # ---- reconciliation print ----
    train = out[out.train_drop_reason.isna()]
    print("SCOPE_ROWS", n_scope)
    print("OUTPUT_ROWS", len(out))
    print("TRAIN_ROWS", len(train))
    print("TRAIN_GOALS", int(train.is_goal.sum()))
    print("TRAIN_GOAL_RATE", round(float(train.is_goal.mean()), 5))
    print("DROPPED_GARBAGE", int((~out.train_drop_reason.isna()).sum()),
          "goals", int(out.loc[~out.train_drop_reason.isna(), "is_goal"].sum()))
    print("SHOT_TYPE_IMPUTED", int(out.shot_type_imputed.sum()),
          "(all ->", SHOT_TYPE_LABELS[mode_id], "=", mode_id, ")")
    print("OFF_WING_NULL", int(out.off_wing.isna().sum()))
    print("DUP_ROWS_DROPPED 0 (item A: verified two real events per pair)")
    return out


if __name__ == "__main__":
    build()
