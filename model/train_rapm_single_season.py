"""
Single-season RAPM — wrapper around model/train_rapm.py for the
2026-06-07 dashboard refactor.

The Players section's RAPM column should answer "how did this player perform
in 2025-26" — not "what is this player's 16-season weighted talent estimate."
This script monkey-patches the multi-season trainer to use only 2025-26:

  - Single season window: ["20252026"]
  - Single-season weight 1.00 (no year-decay)
  - Lower qualifying threshold: 150 EV minutes (one season ~82 GP, so 150 min
    = ~25 GP of meaningful 5v5 ice time)
  - Output paths point at rapm_single_season.csv / rapm_single_season_meta.json

The pipeline is otherwise identical (same xG model, same segment construction,
same ridge regression with the same alpha grid, same duplicated-row encoding).

The original 16-season `rapm_results.csv` is unchanged — that artifact is
still required for the simulator's multi-season composite via
build_composite_sim.py.

Run from project root (~3-5 min compute):
    python3 model/train_rapm_single_season.py
    python3 model/apply_rapm_shrinkage.py   # adds shrunk_*/ms_* columns
                                            # consumed by /api/rapm-leaders
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_rapm as tr

# --- Single-season config ---
tr.SEASONS = ["20252026"]
tr.SEASON_WEIGHTS = {"20252026": 1.00}
tr.DEFAULT_WEIGHT = 1.00
# Lower the EV-minute floor: with one 82-game season, 500 min would exclude
# everyone but the heaviest-usage players. 150 min ≈ 25 GP of meaningful 5v5.
tr.MIN_EV_MINUTES = 150
# Goalie GSAX is used to remove goalie quality from skater on-ice xGA. Keep
# the multi-season goalie career baseline as-is — switching to single season
# would let one hot/cold goalie season dominate the adjustment.

# Output paths — separate parallel artifacts
tr.RAPM_CSV_PATH = tr.MODEL_DIR / "rapm_single_season.csv"
tr.RAPM_META_PATH = tr.MODEL_DIR / "rapm_single_season_meta.json"


if __name__ == "__main__":
    print(f"Training SINGLE-SEASON RAPM (25-26 only, "
          f"≥{tr.MIN_EV_MINUTES} EV min). Output → "
          f"{tr.RAPM_CSV_PATH.name}\n", flush=True)
    # train_rapm.main() reads sys.argv; we pass no args (defaults).
    sys.argv = [sys.argv[0]]
    tr.main()
