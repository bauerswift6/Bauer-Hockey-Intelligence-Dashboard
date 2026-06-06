"""
Simulation-focused composite — parallel artifact to model/composite_ratings.csv.

Phase 2.7. Built for forward-looking simulation use (Phase 4 will consume this
file for team-strength aggregation). It does NOT touch the historical
15/4-season composite (`composite_ratings.csv` / `composite_meta.json`) — that
remains the trend / historical view.

Differences vs. the live v4 composite (build_composite.py):
  - Components 2-6 aggregate a TIGHTER 3-season window: 2023-24, 2024-25, 2025-26
    with season weights {2025-26: 1.00, 2024-25: 0.80, 2023-24: 0.50}.
  - Component 1 (RAPM) still uses the full 16-season `rapm_results.csv`
    coefficients — long-window RAPM is needed to disentangle shared ice time.
  - Qualifying threshold lowered to 300 EV min (was 200 in v4) — high enough to
    filter one-month wonders, low enough that breakout sophomores like Lane
    Hutson, Macklin Celebrini, and Cutter Gauthier qualify.
  - WAR-rescale replacement pool is the SAME 300-EV-min qualified pool (single
    tier; v4 used a separate 500-EV-min tier for the WAR baseline).
  - Everything else identical to v4: same six-component structure, same weights
    (RAPM 0.20, Indiv 0.33, RelxG% 0.17, Play 0.18, PP 0.07, PK 0.05), same C2
    sub-weights (ixG 0.40 / iHDCF 0.30 / iGSAX 0.30), same C4 sub-weights
    (P1A/60 0.70 / pts/60 0.30), same ±3.0 soft cap, same 40/60 blended z, same
    NHL-API current-team join.

Run from project root:
    python3 model/build_composite_sim.py            # dry-run
    python3 model/build_composite_sim.py --save     # promote to composite_ratings_sim.csv

Outputs:
    model/composite_ratings_sim.csv
    model/composite_meta_sim.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Import the v4 builder, then monkey-patch module-level config so its main()
# uses the sim window/thresholds. Functions inside build_composite.py read these
# constants from module globals at call time, so the patches take effect.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_composite as bc

# --- Sim config ---
bc.COMPONENT_SEASONS = ["2023", "2024", "2025"]
bc.COMPONENT_GAME_YEARS = [2023, 2024, 2025]
bc.SEASON_WEIGHTS = {
    "2025": 1.00,
    "2024": 0.60,
    "2023": 0.25,
}
bc.MIN_EV_MINUTES = 300         # qualifying threshold (was 200 in v4)
bc.MIN_REPL_EV_MINUTES = 300    # WAR replacement pool = qualified pool (single tier)
# Output paths — separate parallel artifacts
bc.OUT_CSV = bc.MODEL_DIR / "composite_ratings_sim.csv"
bc.OUT_META = bc.MODEL_DIR / "composite_meta_sim.json"
# Archive name for second + subsequent --save runs (preserves one level of history)
bc.PRIOR_ARCHIVE_NAME = "composite_ratings_sim_prev.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Write composite_ratings_sim.csv + meta. Default: dry-run.")
    args = parser.parse_args()
    print(f"Building SIM composite (3-season window 2023-24 → 2025-26, "
          f"≥{bc.MIN_EV_MINUTES} EV min, v4 weights/methodology). "
          f"{'[SAVE]' if args.save else '[dry-run]'}", flush=True)
    bc.main(save=args.save)
