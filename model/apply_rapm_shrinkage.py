"""
Apply Bayesian shrinkage to the single-season RAPM CSV so the Players-page
leaderboard can display a noise-controlled headline value alongside the raw
single-season and the multi-season career numbers.

Why this exists
---------------
Single-season ridge RAPM (n_segments ≈ 122k for a single 1,312-game season)
has noticeable noise from collinearity with linemates and small per-player
samples. Consensus elites (McDavid, MacKinnon, Makar, Kucherov) routinely
land outside the top 50 in raw single-season output even though their
underlying play is unchanged. Raising the TOI floor doesn't fix this —
collinearity persists well above 750 EV minutes. The standard public-model
treatment is empirical-Bayes shrinkage toward a longer-horizon prior.

Formula (applied independently to total / offensive / defensive RAPM)
---------------------------------------------------------------------
    w           = TOI_single_season / (TOI_single_season + K)
    shrunk_X    = w * SS_X + (1 - w) * MS_X

with K = 1000 EV minutes. Higher TOI ⇒ higher w ⇒ less shrinkage; very
low-TOI players get pulled most of the way back to the multi-season
prior. K = 1000 was picked from a sweep at K ∈ {500, 1000, 2000} —
diagnostic showed K = 1000 promotes McDavid to #2 and Kucherov into the
top-100 without admitting a 313-minute Matt Tkachuk-style sample into
the top 5 (which K = 2000 does).

Fallback for missing prior
--------------------------
Players not present in rapm_results.csv (true rookies, returning-from-
injury veterans whose career didn't overlap the 16-season window) get
shrunk_X = SS_X unchanged. We do NOT shrink toward 0 — that would
penalise rookies for having no track record, which is the opposite of
the desired behaviour. The CSV exposes ms_total_rapm = NaN for these
players so the frontend can show "—" instead of pretending we have a
prior we don't.

Run after model/train_rapm_single_season.py:
    python3 model/apply_rapm_shrinkage.py

The single-season CSV is extended in place with these new columns:
    shrunk_total_rapm, shrunk_off_rapm, shrunk_def_rapm
    ms_total_rapm,     ms_off_rapm,     ms_def_rapm,    ms_toi_minutes
    shrink_weight_ss   (= w, the single-season weight; for transparency)
    shrink_K           (constant K used; for audit)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent
SS_CSV = MODEL_DIR / "rapm_single_season.csv"
MS_CSV = MODEL_DIR / "rapm_results.csv"

K = 1000  # shrinkage constant (EV minutes)


def main() -> None:
    if not SS_CSV.exists():
        sys.exit(f"ERROR: {SS_CSV} not found. Run train_rapm_single_season.py first.")
    if not MS_CSV.exists():
        sys.exit(f"ERROR: {MS_CSV} not found. Multi-season prior unavailable.")

    ss = pd.read_csv(SS_CSV)
    ms = pd.read_csv(MS_CSV)

    print(f"single-season rows: {len(ss):,}")
    print(f"multi-season rows:  {len(ms):,}")

    ms_lookup = (
        ms.set_index("player_id")[
            ["total_rapm", "offensive_rapm", "defensive_rapm", "toi_minutes"]
        ]
        .rename(
            columns={
                "total_rapm": "ms_total_rapm",
                "offensive_rapm": "ms_off_rapm",
                "defensive_rapm": "ms_def_rapm",
                "toi_minutes": "ms_toi_minutes",
            }
        )
    )

    # Drop any prior shrinkage columns from a previous run so we always
    # write fresh values.
    drop_cols = [
        "shrunk_total_rapm", "shrunk_off_rapm", "shrunk_def_rapm",
        "ms_total_rapm", "ms_off_rapm", "ms_def_rapm", "ms_toi_minutes",
        "shrink_weight_ss", "shrink_K",
    ]
    ss = ss.drop(columns=[c for c in drop_cols if c in ss.columns])

    merged = ss.merge(ms_lookup, left_on="player_id", right_index=True, how="left")

    n_with_prior = merged["ms_total_rapm"].notna().sum()
    n_no_prior = len(merged) - n_with_prior
    print(f"players with multi-season prior: {n_with_prior:,}")
    print(f"players without prior (pass-through):  {n_no_prior:,}")

    # Single-season weight w = TOI / (TOI + K).
    w = merged["toi_minutes"] / (merged["toi_minutes"] + K)
    merged["shrink_weight_ss"] = w.round(4)
    merged["shrink_K"] = K

    # Apply per-component shrinkage with NaN-safe fallback. The vectorised
    # form is `w*SS + (1-w)*MS` when MS exists, else SS unchanged.
    for ss_col, ms_col, out_col in (
        ("total_rapm",     "ms_total_rapm", "shrunk_total_rapm"),
        ("offensive_rapm", "ms_off_rapm",   "shrunk_off_rapm"),
        ("defensive_rapm", "ms_def_rapm",   "shrunk_def_rapm"),
    ):
        has_prior = merged[ms_col].notna()
        shrunk = merged[ss_col].copy()
        shrunk.loc[has_prior] = (
            w.loc[has_prior] * merged.loc[has_prior, ss_col]
            + (1 - w.loc[has_prior]) * merged.loc[has_prior, ms_col]
        )
        merged[out_col] = shrunk.round(4)

    merged.to_csv(SS_CSV, index=False)
    print(f"\nwritten → {SS_CSV.name}")
    print(f"new columns: shrunk_total_rapm, shrunk_off_rapm, shrunk_def_rapm,")
    print(f"             ms_total_rapm, ms_off_rapm, ms_def_rapm, ms_toi_minutes,")
    print(f"             shrink_weight_ss, shrink_K (= {K})")

    # Quick sanity print: top 10 by new shrunk_total_rapm
    print(f"\nTop 10 by shrunk_total_rapm (K = {K}):")
    top = merged.sort_values("shrunk_total_rapm", ascending=False).head(10)
    for i, r in enumerate(top.itertuples(index=False), 1):
        ms_disp = (
            f"{r.ms_total_rapm:+.3f}" if pd.notna(r.ms_total_rapm) else "  n/a"
        )
        print(
            f"  {i:>2}. {r.player_name:<24} {r.team:<4}  "
            f"TOI {r.toi_minutes:>5.0f}  "
            f"w {r.shrink_weight_ss:>.2f}  "
            f"SS {r.total_rapm:>+6.3f}  "
            f"MS {ms_disp:>7}  "
            f"shrunk {r.shrunk_total_rapm:>+6.3f}"
        )


if __name__ == "__main__":
    main()
