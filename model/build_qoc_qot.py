"""
Quality of Competition (QoC) + Quality of Teammates (QoT) — single-season 25-26.

For each NHL skater, computes the average GAR (composite_war from
composite_ratings_single_season.csv) of:
  - the opponents they shared ice time with, weighted by shared seconds → QoC
  - the teammates they shared ice time with, weighted by shared seconds → QoT

Methodology
-----------
1. Load 25-26 shifts (season_cache/shifts_20252026.parquet).
2. Per game, compute the pairwise shift-overlap matrix:
       overlap[i,j] = max(0, min(end_i, end_j) − max(start_i, start_j))
   Discard self-overlap (same shift) and any overlap between two shifts of
   the same player (a player can have multiple shifts per game).
3. Sum overlap seconds by (focal_player, partner_player, same_team_flag)
   across all games.
4. For each focal player, weighted-average partner GAR:
       QoT = Σ(secs_with_teammate × GAR_teammate) / Σ(secs_with_teammate)
       QoC = Σ(secs_with_opponent × GAR_opponent) / Σ(secs_with_opponent)
   Partners without a GAR (rookies / sub-150-EV-min players) are excluded
   from both numerator and denominator so unrated bottom-line scrubs don't
   blur the signal.

Both metrics come out on the same scale as GAR (composite_war units, where
the qualified pool runs roughly −1.5 to +2.5). A QoT of +1.2 reads as
"this player's typical teammates were a +1.2-GAR player."

Run from project root (~1-2 min):
    python3 model/build_qoc_qot.py

Output: model/skater_qoc_qot_single_season.csv
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
SHIFTS_PATH = MODEL_DIR / "season_cache" / "shifts_20252026.parquet"
COMPOSITE_PATH = MODEL_DIR / "composite_ratings_single_season.csv"
OUT_CSV = MODEL_DIR / "skater_qoc_qot_single_season.csv"


def build_overlap_table(shifts: pd.DataFrame) -> pd.DataFrame:
    """For every game, compute pairwise shift-overlap seconds.
    Returns a long DataFrame: (focal_pid, partner_pid, same_team, shared_secs).
    Each (focal, partner, same_team) tuple may appear multiple times (one per
    game) — caller aggregates."""
    print(f"Computing pairwise overlaps across "
          f"{shifts['game_id'].nunique()} games …", flush=True)
    t0 = time.time()
    frames = []
    for game_id, gdf in shifts.groupby("game_id", sort=False):
        starts = gdf["abs_start"].to_numpy()
        ends   = gdf["abs_end"].to_numpy()
        pids   = gdf["player_id"].to_numpy()
        teams  = gdf["team_abbrev"].to_numpy()

        # Pairwise overlap matrix (n × n)
        s_max = np.maximum(starts[:, None], starts[None, :])
        e_min = np.minimum(ends[:, None], ends[None, :])
        overlap = np.maximum(0, e_min - s_max)
        np.fill_diagonal(overlap, 0)

        ii, jj = np.where(overlap > 0)
        if ii.size == 0:
            continue
        # Drop same-player pairs (multiple shifts for the same player)
        keep = pids[ii] != pids[jj]
        ii, jj = ii[keep], jj[keep]
        if ii.size == 0:
            continue
        secs = overlap[ii, jj]
        same_team = teams[ii] == teams[jj]
        frames.append(pd.DataFrame({
            "focal_pid":   pids[ii],
            "partner_pid": pids[jj],
            "same_team":   same_team,
            "shared_secs": secs,
        }))
    out = pd.concat(frames, ignore_index=True)
    print(f"  raw pairwise rows: {len(out):,}  "
          f"({time.time()-t0:.1f}s)", flush=True)
    # Aggregate across games — sum shared_secs per (focal, partner, same_team)
    out = (out.groupby(["focal_pid", "partner_pid", "same_team"],
                        as_index=False)["shared_secs"].sum())
    print(f"  aggregated unique pair-rows: {len(out):,}", flush=True)
    return out


def compute_qoc_qot(pair_secs: pd.DataFrame,
                    gar_by_pid: dict[int, float]) -> pd.DataFrame:
    """Per focal player, compute QoT (same_team weighted avg GAR) and
    QoC (opposing-team weighted avg GAR)."""
    print("Computing weighted averages …", flush=True)
    # Attach partner GAR; drop rows where partner has no GAR
    pair_secs = pair_secs.copy()
    pair_secs["partner_gar"] = pair_secs["partner_pid"].map(gar_by_pid)
    pair_secs = pair_secs[pair_secs["partner_gar"].notna()].copy()
    # Weighted sums
    pair_secs["weighted"] = pair_secs["shared_secs"] * pair_secs["partner_gar"]
    grouped = (pair_secs.groupby(["focal_pid", "same_team"])
               .agg(weight=("shared_secs", "sum"),
                    weighted_sum=("weighted", "sum"))
               .reset_index())
    grouped["weighted_avg"] = grouped["weighted_sum"] / grouped["weight"]
    # Pivot: same_team True → QoT, False → QoC
    pivot = grouped.pivot(index="focal_pid", columns="same_team",
                          values="weighted_avg").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={True: "qot", False: "qoc",
                                    "focal_pid": "player_id"})
    # Also surface total shared-TOI counts so downstream can see sample size
    weights = (grouped.pivot(index="focal_pid", columns="same_team",
                              values="weight")
                .reset_index()
                .rename(columns={True: "qot_toi_secs", False: "qoc_toi_secs",
                                  "focal_pid": "player_id"}))
    weights.columns.name = None
    out = pivot.merge(weights, on="player_id", how="left")
    return out


def main():
    t_start = time.time()
    if not SHIFTS_PATH.exists():
        print(f"ERROR: {SHIFTS_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    if not COMPOSITE_PATH.exists():
        print(f"ERROR: {COMPOSITE_PATH} not found. Run "
              f"build_composite_single_season.py first.", file=sys.stderr)
        sys.exit(1)

    shifts = pd.read_parquet(SHIFTS_PATH)
    pre_rs = len(shifts)
    shifts = shifts[((shifts["game_id"] // 10_000) % 100) == 2].copy()
    print(f"Loaded {pre_rs:,} 25-26 shifts; "
          f"{len(shifts):,} after regular-season filter "
          f"({pre_rs - len(shifts):,} playoff dropped).")

    composite = pd.read_csv(COMPOSITE_PATH)
    gar_by_pid = dict(zip(composite["player_id"].astype(int),
                          composite["composite_war"].astype(float)))
    name_by_pid = dict(zip(composite["player_id"].astype(int),
                            composite["player_name"]))
    team_by_pid = dict(zip(composite["player_id"].astype(int),
                            composite["team"]))
    print(f"Loaded {len(gar_by_pid):,} skater GAR values from "
          f"{COMPOSITE_PATH.name}.\n")

    pair_secs = build_overlap_table(shifts)
    qoc_qot = compute_qoc_qot(pair_secs, gar_by_pid)

    # Decorate with name + team for the CSV
    qoc_qot["player_name"] = qoc_qot["player_id"].map(name_by_pid)
    qoc_qot["team"] = qoc_qot["player_id"].map(team_by_pid)
    qoc_qot["qoc"] = qoc_qot["qoc"].round(3)
    qoc_qot["qot"] = qoc_qot["qot"].round(3)
    qoc_qot["qoc_toi_secs"] = qoc_qot["qoc_toi_secs"].fillna(0).astype(int)
    qoc_qot["qot_toi_secs"] = qoc_qot["qot_toi_secs"].fillna(0).astype(int)

    # Only keep players who themselves are in the composite table — they're
    # the qualified pool the dashboard shows.
    qoc_qot = qoc_qot[qoc_qot["player_id"].isin(gar_by_pid.keys())]
    qoc_qot = qoc_qot[["player_id", "player_name", "team",
                        "qoc", "qot", "qoc_toi_secs", "qot_toi_secs"]]
    qoc_qot = qoc_qot.sort_values("qot", ascending=False).reset_index(drop=True)
    qoc_qot.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {len(qoc_qot):,} rows → {OUT_CSV}")

    print("\nTop 10 QoT (highest average teammate GAR):")
    for _, r in qoc_qot.head(10).iterrows():
        print(f"  {r['player_name']:<22} {r['team']:>3}  QoT={r['qot']:+.3f}  "
              f"QoC={r['qoc']:+.3f}  toi_with_teammates={r['qot_toi_secs']/60:.0f}m")

    print("\nBottom 10 QoT:")
    for _, r in qoc_qot.tail(10).iterrows():
        print(f"  {r['player_name']:<22} {r['team']:>3}  QoT={r['qot']:+.3f}  "
              f"QoC={r['qoc']:+.3f}  toi_with_teammates={r['qot_toi_secs']/60:.0f}m")

    qoc_qot_by_qoc = qoc_qot.sort_values("qoc", ascending=False)
    print("\nTop 10 QoC (faced toughest competition):")
    for _, r in qoc_qot_by_qoc.head(10).iterrows():
        print(f"  {r['player_name']:<22} {r['team']:>3}  QoC={r['qoc']:+.3f}  "
              f"QoT={r['qot']:+.3f}")

    print(f"\nDone in {(time.time()-t_start)/60:.1f} min.")


if __name__ == "__main__":
    main()
