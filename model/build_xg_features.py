"""
build_xg_features.py
--------------------
Season-parameterized version of build_xg_features_2025_26.py. Same logic
(byte-for-byte equivalent output on 2025-26 — proven by the Step-1 equivalence
test), generalized to any regular season via --season.

Parameterized: ENRICHED, DATES, OUT, GID range. HAND (player_handedness.csv) is
shared across seasons. All rules (no duplicate drop, shot_type-0 imputation to
mode, off_wing via y_norm>0 = shooter-left, garbage-strength flagging, feature
list, exclusions) are identical to the 2025-26 builder.
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAND = os.path.join(HERE, "player_handedness.csv")

SHOT_TYPE_LABELS = {0: "unknown", 1: "wrist", 2: "snap", 3: "slap", 4: "backhand",
                    5: "tip-in", 6: "deflected", 7: "wrap-around", 8: "poke",
                    9: "bat", 10: "cradle", 11: "between-legs"}

FEATURE_COLS = [
    "distance_from_net", "angle_from_center", "x_norm", "abs_y_norm",
    "shot_type_id", "is_rebound", "time_since_last_shot", "is_rush",
    "score_state", "period", "time_in_period", "is_home",
    "skaters_for", "skaters_against", "off_wing",
]

SEASONS = {
    "2025_26": dict(gid=(2025020001, 2025021312), enriched="shots_enriched_2025_26.parquet",
                    dates="game_dates_2025_26.csv", out="xg_features_2025_26.parquet"),
    "2024_25": dict(gid=(2024020001, 2024021312), enriched="shots_enriched_2024_25.parquet",
                    dates="game_dates_2024_25.csv", out="xg_features_2024_25.parquet"),
    "2023_24": dict(gid=(2023020001, 2023021312), enriched="shots_enriched_2023_24.parquet",
                    dates="game_dates_2023_24.csv", out="xg_features_2023_24.parquet"),
    "2022_23": dict(gid=(2022020001, 2022021312), enriched="shots_enriched_2022_23.parquet",
                    dates="game_dates_2022_23.csv", out="xg_features_2022_23.parquet"),
}


def is_garbage_strength(sf: int, sa: int) -> bool:
    if not (3 <= sf <= 6 and 3 <= sa <= 6):
        return True
    if abs(sf - sa) > 2:
        return True
    if sf == 6 and sa == 6:
        return True
    if sa == 6:
        return True
    return False


def build(cfg):
    lo, hi = cfg["gid"]
    E = pd.read_parquet(os.path.join(HERE, cfg["enriched"]))

    f = E[(E.game_id >= lo) & (E.game_id <= hi)].copy()
    f = f[~((f.period == 5) & (f.abs_secs == 4800))].copy()
    n_scope = len(f)

    mode_id = int(f.loc[f.shot_type_id != 0, "shot_type_id"].mode().iloc[0])
    f["shot_type_imputed"] = f.shot_type_id == 0
    f["shot_type_id"] = f.shot_type_id.replace(0, mode_id).astype(int)

    hand = pd.read_csv(HAND)
    f = f.merge(hand, left_on="shooter_id", right_on="player_id", how="left")

    off = np.where(f.shoots.isna(), np.nan,
            np.where((f.shoots == "L") & (f.y_norm < 0), 1.0,
            np.where((f.shoots == "R") & (f.y_norm > 0), 1.0, 0.0)))
    f["off_wing"] = off

    f["abs_y_norm"] = f.y_norm.abs()
    f["time_in_period"] = f.abs_secs - (f.period - 1) * 1200
    f["score_state"] = f.score_state.clip(-3, 3)

    dates = pd.read_csv(os.path.join(HERE, cfg["dates"]))
    f = f.merge(dates, on="game_id", how="left")

    garb = f.apply(lambda r: is_garbage_strength(int(r.skaters_for), int(r.skaters_against)), axis=1)
    f["train_drop_reason"] = np.where(garb, "garbage_strength_state", pd.NA)

    f["skaters_for"] = f.skaters_for.astype(int)
    f["skaters_against"] = f.skaters_against.astype(int)

    keep_cols = (["game_id", "abs_secs", "shooter_id", "game_date"]
                 + FEATURE_COLS + ["is_goal", "shot_type_imputed", "train_drop_reason"])
    out = f[keep_cols].copy()
    out["xg_old"] = f["xg"].values

    out.to_parquet(os.path.join(HERE, cfg["out"]), index=False)

    train = out[out.train_drop_reason.isna()]
    print("SEASON_OUT", cfg["out"])
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
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, choices=list(SEASONS.keys()))
    a = ap.parse_args()
    build(SEASONS[a.season])
