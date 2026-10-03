"""
Single-season composite — parallel artifact for the Players dashboard.

Same six-component blend as build_composite_sim.py (RAPM, individual xG
impact, relative xGF%, playmaking, PP, PK), but:
  - Components 2-6 aggregate 2025-26 only (no year-decay)
  - Component 1 reads model/rapm_single_season.csv (instead of the
    multi-season rapm_results.csv)
  - Lower qualifying floor: 150 EV min (vs 300 for the simulation composite)

The simulation composite (model/composite_ratings_sim.csv) is untouched —
the simulator still uses multi-season talent estimates per design.

Run from project root (~30-60 sec):
    python3 model/build_composite_single_season.py --save

Outputs:
    model/composite_ratings_single_season.csv
    model/composite_meta_single_season.json
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_composite as bc

# --- Single-season config ---
bc.COMPONENT_SEASONS = ["2025"]
bc.COMPONENT_GAME_YEARS = [2025]
bc.SEASON_WEIGHTS = {"2025": 1.00}
bc.MIN_EV_MINUTES = 150       # qualifying threshold (one-season pool)
bc.MIN_REPL_EV_MINUTES = 150  # WAR replacement pool = qualified pool
# Component 1 uses the single-season RAPM instead of the multi-season one
bc.RAPM_CSV = bc.MODEL_DIR / "rapm_single_season.csv"
# Output paths
bc.OUT_CSV = bc.MODEL_DIR / "composite_ratings_single_season.csv"
bc.OUT_META = bc.MODEL_DIR / "composite_meta_single_season.json"
bc.PRIOR_ARCHIVE_NAME = "composite_ratings_single_season_prev.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Write composite_ratings_single_season.csv + meta. "
                             "Default: dry-run.")
    args = parser.parse_args()
    print(f"Building SINGLE-SEASON composite (25-26 only, ≥{bc.MIN_EV_MINUTES} "
          f"EV min). {'[SAVE]' if args.save else '[dry-run]'}", flush=True)
    bc.main(save=args.save)
