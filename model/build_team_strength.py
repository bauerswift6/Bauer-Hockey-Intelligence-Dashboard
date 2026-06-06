"""
Phase 4 — Team strength aggregation.

Consumes:
  - model/composite_ratings_sim.csv  (per-player composite_war + expected_gp_share)
  - NHL API /v1/roster/{team}/{season}  (canonical 25-26 active rosters)
  - MoneyPuck 2025-26 goalies CSV     (top-2 goalies, GSAA, GP for weighting)

Produces:
  - model/team_strength.csv with columns: team, team_strength,
    top6_forward_strength, bottom6_forward_strength,
    top_pair_d_strength, bottom_pair_d_strength, goalie_strength,
    n_forwards, n_defense, n_goalies, top_goalie, top_goalie_gp,
    backup_goalie, backup_goalie_gp.

Player → role:
  Forwards sorted by historical avg_toi_per_game (descending). Top 12:
    top 3      → 22 min       \
    next 3     → 18 min        > top-6 strength (weighted avg by minutes)
    next 3     → 14 min       \
    bottom 3   → 10 min        > bottom-6 strength
  Defensemen sorted same way. Top 6:
    top 2      → 24 min       \
    next 2     → 20 min        > top-pair strength
    bottom 2   → 16 min        > bottom-pair strength
  Unrated rostered players → composite_war = 0.0, expected_gp_share = 0.5.

Goalie strength:
  Top 2 goalies by 25-26 GP. GSAA = xGoals − goals (MoneyPuck "all" situation).
  Converted to WAR by dividing by 6 goals/win. Weighted by GP share between
  the two.

team_strength blend (sums to 1.0; weights derived from ice-time shares with a
30% allocation to goalies):
  team_strength = 0.270 × top6_F + 0.162 × bot6_F + 0.197 × topPairD
                + 0.072 × botPairD + 0.300 × goalie_war

Run from project root:
    python3 model/build_team_strength.py
"""
from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
COMPOSITE_CSV = MODEL_DIR / "composite_ratings_sim.csv"
SHARES_CSV = MODEL_DIR / "player_offense_shares.csv"
OUT_CSV = MODEL_DIR / "team_strength.csv"
NHL_API = "https://api-web.nhle.com/v1"
MP_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/{kind}.csv"
SEASON = "20252026"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}

# All 32 NHL team abbreviations
TEAMS = [
    "ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK",
    "MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL",
    "TOR","UTA","VAN","VGK","WPG","WSH",
]

# Ice-time tier assignments (minutes per game)
FWD_TIERS = [22, 22, 22, 18, 18, 18, 14, 14, 14, 10, 10, 10]   # 12 forwards
DEF_TIERS = [24, 24, 20, 20, 16, 16]                            # 6 defensemen

# Team-strength blend weights (sum = 1.000)
W_TOP6_F     = 0.270
W_BOT6_F     = 0.162
W_TOP_PAIR_D = 0.197
W_BOT_PAIR_D = 0.072
W_GOALIE     = 0.300

# Conversion: 1 WAR ≈ 6 goals (standard hockey-analytics conversion)
GOALS_PER_WAR = 6.0
# Default availability when a rostered player has no composite rating
DEFAULT_GP_SHARE = 0.50

# Cap on the goalie blended_war BEFORE z-score conversion. Elite individual
# goaltending (Shesterkin +21 GSAA → +2.026 raw; Thompson +25 GSAA → +3.353)
# was contributing uncapped lift to team_strength via the 30% goalie weight,
# masking weak team defense (NYR were rank 18 by TS but rank 29 actual).
# Applied in `_goalie_strength()` directly so an elite individual goalie
# contributes the same as a "very good" goalie at the cap. Z-normalization
# then proceeds across the capped distribution. See
# model/team_strength_vs_actual.csv and STATUS.md polarization fixes notes.
GOALIE_WAR_CAP = 1.5

# Empirical compression on the final team_strength magnitudes. The
# composite_war multi-year baseline polarizes top and bottom teams beyond
# their single-season form; rank correlation with actual 25-26 points was
# already 0.736 (in the public-model range), but the magnitudes stretched
# enough that ALPHA × (TS_top - TS_avg) gave top teams unrealistically
# strong per-game edges. This is a methodologically-defensible
# regularization that preserves rank order exactly while shrinking the
# magnitude spread by (1 - TEAM_STRENGTH_COMPRESSION) = 15%.
TEAM_STRENGTH_COMPRESSION = 0.85

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


def load_composite_lookup() -> dict[int, dict]:
    """{player_id: {composite_war, expected_gp_share, position, player_name, team}}"""
    df = pd.read_csv(COMPOSITE_CSV)
    out = {}
    for r in df.itertuples(index=False):
        out[int(r.player_id)] = {
            "composite_war": float(r.composite_war),
            "expected_gp_share": float(r.expected_gp_share),
            "position": str(r.position),
            "player_name": str(r.player_name),
            "team": str(r.team),
        }
    print(f"  Loaded {len(out):,} composite_war rows from {COMPOSITE_CSV.name}",
          flush=True)
    return out


def load_toi_lookup() -> dict[int, float]:
    """{player_id: avg_toi_per_game} from player_offense_shares.csv. Used to
    sort the active roster the way coaches actually do — by historical ice
    time per game. Players missing here are treated as 0 TOI and sink to the
    bottom of the depth chart (rookies/AHL call-ups)."""
    if not SHARES_CSV.exists():
        print(f"  WARNING: {SHARES_CSV.name} not found — TOI sort will fall "
              f"back to 0.0 for every player", flush=True)
        return {}
    df = pd.read_csv(SHARES_CSV, usecols=["player_id", "avg_toi_per_game"])
    out = {int(r.player_id): float(r.avg_toi_per_game)
           for r in df.itertuples(index=False)}
    print(f"  Loaded {len(out):,} avg_toi_per_game rows from {SHARES_CSV.name}",
          flush=True)
    return out


def fetch_roster(team_abbrev: str) -> list[dict]:
    """Return list of {player_id, first_name, last_name, position} for the
    active 2025-26 roster of one team."""
    url = f"{NHL_API}/roster/{team_abbrev}/{SEASON}"
    r = requests.get(url, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return []
    d = r.json()
    players = []
    for bucket in ("forwards", "defensemen", "goalies"):
        for p in d.get(bucket, []):
            players.append({
                "player_id": int(p["id"]),
                "first_name": (p.get("firstName") or {}).get("default", ""),
                "last_name":  (p.get("lastName")  or {}).get("default", ""),
                "position":   p.get("positionCode") or
                              ("G" if bucket == "goalies"
                               else "D" if bucket == "defensemen" else "F"),
                "roster_role": "G" if bucket == "goalies"
                               else "D" if bucket == "defensemen" else "F",
            })
    return players


def load_goalies_csv() -> pd.DataFrame:
    """MoneyPuck 2025-26 goalies, situation=all, with GSAA and GP."""
    g = get_csv(MP_URL.format(kind="goalies"))
    g = g[g["situation"] == "all"].copy()
    for c in ("xGoals", "goals", "games_played", "icetime"):
        g[c] = pd.to_numeric(g[c], errors="coerce").fillna(0)
    g["gsaa"] = g["xGoals"] - g["goals"]
    g["goalie_war"] = g["gsaa"] / GOALS_PER_WAR
    g = g[["playerId", "name", "team", "games_played", "icetime",
           "xGoals", "goals", "gsaa", "goalie_war"]]
    g["playerId"] = g["playerId"].astype(int)
    print(f"  Loaded {len(g):,} goalie-season rows (25-26 MoneyPuck)", flush=True)
    return g


# ---------------------------------------------------------------------------
# Team strength
# ---------------------------------------------------------------------------
def _split_by_position(roster: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return (forwards, defensemen, goalies) sub-lists. Uses roster_role first,
    falls back to positionCode."""
    fwd = [p for p in roster if p["roster_role"] == "F"]
    dfn = [p for p in roster if p["roster_role"] == "D"]
    glx = [p for p in roster if p["roster_role"] == "G"]
    return fwd, dfn, glx


def _attach_war(roster_players: list[dict],
                comp_lookup: dict[int, dict],
                toi_lookup: dict[int, float] | None = None) -> list[dict]:
    """For each rostered player, attach composite_war + expected_gp_share +
    avg_toi_per_game. Unrated players get (war=0.0, gp_share=DEFAULT_GP_SHARE).
    Players with no TOI history get 0.0 (sinks them to bottom of depth chart)."""
    toi_lookup = toi_lookup or {}
    out = []
    for p in roster_players:
        c = comp_lookup.get(p["player_id"])
        if c is None:
            war, gp = 0.0, DEFAULT_GP_SHARE
            rated = False
        else:
            war, gp = c["composite_war"], c["expected_gp_share"]
            rated = True
        avg_toi = float(toi_lookup.get(p["player_id"], 0.0))
        out.append({
            **p,
            "composite_war": war,
            "expected_gp_share": gp,
            "rated": rated,
            "effective_war": war * gp,   # availability-adjusted talent
            "avg_toi_per_game": avg_toi, # primary sort key for depth chart
        })
    return out


def _ice_time_weighted_avg(players: list[dict], minutes: list[float]) -> float:
    """Σ(war × ice_minutes) / Σ(ice_minutes) — weighted by ice time only.
    The availability adjustment is already baked into composite_war × gp_share
    via the effective_war column the caller passes in."""
    if not players or not minutes or len(players) != len(minutes):
        return 0.0
    num = sum(p["effective_war"] * m for p, m in zip(players, minutes))
    den = sum(minutes)
    return num / den if den > 0 else 0.0


def _goalie_strength(goalie_rows: list[dict],
                     goalies_csv: pd.DataFrame) -> tuple[float, dict]:
    """Return (goalie_war_blend, detail_dict). Picks top 2 by 25-26 GP from
    MoneyPuck (filtered to this team's rostered goalies). If only one is
    rated, uses that one alone; if none, returns 0.0."""
    rostered_ids = {g["player_id"] for g in goalie_rows}
    g = goalies_csv[goalies_csv["playerId"].isin(rostered_ids)].copy()
    g = g.sort_values("games_played", ascending=False).head(2)
    if g.empty:
        return 0.0, {"top_goalie": "", "top_goalie_gp": 0,
                     "backup_goalie": "", "backup_goalie_gp": 0,
                     "top_goalie_war": 0.0, "backup_goalie_war": 0.0}
    total_gp = float(g["games_played"].sum())
    if total_gp <= 0:
        return 0.0, {"top_goalie": "", "top_goalie_gp": 0,
                     "backup_goalie": "", "backup_goalie_gp": 0,
                     "top_goalie_war": 0.0, "backup_goalie_war": 0.0}
    blended_war = float((g["goalie_war"] * g["games_played"]).sum() / total_gp)
    # Cap the blended_war BEFORE z-norm. Elite individual goaltending
    # contributes the same as a "very good" goalie at +1.5 WAR; below-
    # replacement tandems get a -1.5 floor. See GOALIE_WAR_CAP comment.
    blended_war = max(-GOALIE_WAR_CAP, min(GOALIE_WAR_CAP, blended_war))
    rows = g.to_dict(orient="records")
    return blended_war, {
        "top_goalie": rows[0]["name"],
        "top_goalie_gp": int(rows[0]["games_played"]),
        "top_goalie_war": round(float(rows[0]["goalie_war"]), 3),
        "backup_goalie": rows[1]["name"] if len(rows) > 1 else "",
        "backup_goalie_gp": int(rows[1]["games_played"]) if len(rows) > 1 else 0,
        "backup_goalie_war": round(float(rows[1]["goalie_war"]), 3) if len(rows) > 1 else 0.0,
    }


def compute_team_strength(team_abbrev: str,
                          comp_lookup: dict[int, dict],
                          goalies_csv: pd.DataFrame,
                          roster_override: list[dict] | None = None,
                          toi_lookup: dict[int, float] | None = None,
                          ) -> dict:
    """Single team's strength + component breakdown. If `roster_override` is
    supplied, use that instead of pulling the NHL API roster (used by the
    lineup-editor endpoint)."""
    roster = roster_override if roster_override is not None else fetch_roster(team_abbrev)
    if not roster:
        return {"team": team_abbrev, "team_strength": 0.0,
                "top6_forward_strength": 0.0, "bottom6_forward_strength": 0.0,
                "top_pair_d_strength": 0.0, "bottom_pair_d_strength": 0.0,
                "goalie_strength": 0.0, "n_forwards": 0, "n_defense": 0,
                "n_goalies": 0, "top_goalie": "", "top_goalie_gp": 0,
                "backup_goalie": "", "backup_goalie_gp": 0, "error": "no roster"}

    fwds_raw, dfns_raw, glxs_raw = _split_by_position(roster)
    fwds = _attach_war(fwds_raw, comp_lookup, toi_lookup)
    dfns = _attach_war(dfns_raw, comp_lookup, toi_lookup)
    # Sort by historical avg ice time (calibration fix — top of the depth
    # chart first). Coaches deploy players by TOI; using effective_war here
    # pushed injury-prone stars down the lineup card.
    fwds.sort(key=lambda p: p["avg_toi_per_game"], reverse=True)
    dfns.sort(key=lambda p: p["avg_toi_per_game"], reverse=True)

    # Pad / truncate to 12 forwards + 6 defensemen.
    fwd_lineup = fwds[:12] + [None] * max(0, 12 - len(fwds))
    def_lineup = dfns[:6] + [None] * max(0, 6 - len(dfns))

    def _avg_unit(lineup_slice, minute_slice):
        present = [(p, m) for p, m in zip(lineup_slice, minute_slice) if p is not None]
        if not present:
            return 0.0
        return _ice_time_weighted_avg([p for p, _ in present],
                                       [m for _, m in present])

    top6_F = _avg_unit(fwd_lineup[0:6], FWD_TIERS[0:6])
    bot6_F = _avg_unit(fwd_lineup[6:12], FWD_TIERS[6:12])
    top_pair_D = _avg_unit(def_lineup[0:4], DEF_TIERS[0:4])
    bot_pair_D = _avg_unit(def_lineup[4:6], DEF_TIERS[4:6])

    goalie_war, goalie_detail = _goalie_strength(glxs_raw, goalies_csv)

    team_strength = (W_TOP6_F * top6_F
                     + W_BOT6_F * bot6_F
                     + W_TOP_PAIR_D * top_pair_D
                     + W_BOT_PAIR_D * bot_pair_D
                     + W_GOALIE * goalie_war)

    return {
        "team": team_abbrev,
        "team_strength": round(team_strength, 4),
        "top6_forward_strength": round(top6_F, 4),
        "bottom6_forward_strength": round(bot6_F, 4),
        "top_pair_d_strength": round(top_pair_D, 4),
        "bottom_pair_d_strength": round(bot_pair_D, 4),
        "goalie_strength": round(goalie_war, 4),
        "n_forwards": len(fwds),
        "n_defense": len(dfns),
        "n_goalies": len(glxs_raw),
        **goalie_detail,
    }


def _z_normalize_components(df: pd.DataFrame) -> pd.DataFrame:
    """The five raw component values (top6_F, bot6_F, topPairD, botPairD,
    goalie_war) live on very different scales — skater values are aggregated
    composite_war (~0.5–1.5 range), goalie WAR is GSAA/6 (~-3 to +5). To make
    them comparable in the weighted blend, z-score each column across the
    32-team population, then recompute team_strength as a weighted sum of
    z-scores. Final team_strength is a unitless rating centered ~0 across the
    league."""
    z = df.copy()
    cols = ["top6_forward_strength", "bottom6_forward_strength",
            "top_pair_d_strength", "bottom_pair_d_strength", "goalie_strength"]
    for c in cols:
        mu = z[c].mean()
        sd = z[c].std(ddof=0)
        z[c + "_z"] = (z[c] - mu) / sd if sd > 0 else 0.0
    # Goalie is already capped on raw blended_war upstream in
    # `_goalie_strength()` (GOALIE_WAR_CAP), so no additional z-score cap
    # is needed here.
    z["team_strength"] = (
        W_TOP6_F     * z["top6_forward_strength_z"]
        + W_BOT6_F     * z["bottom6_forward_strength_z"]
        + W_TOP_PAIR_D * z["top_pair_d_strength_z"]
        + W_BOT_PAIR_D * z["bottom_pair_d_strength_z"]
        + W_GOALIE     * z["goalie_strength_z"]
    )
    # Magnitude compression (TEAM_STRENGTH_COMPRESSION). Preserves rank
    # order exactly; shrinks spread between top and bottom teams by 15%.
    z["team_strength"] = (z["team_strength"] * TEAM_STRENGTH_COMPRESSION).round(4)
    return z


def build_all_teams(workers: int = 8) -> pd.DataFrame:
    """Compute team strength for all 32 teams in parallel, then z-normalize
    each component across the league before blending."""
    print(f"Building team strengths for {len(TEAMS)} teams …", flush=True)
    comp_lookup = load_composite_lookup()
    toi_lookup = load_toi_lookup()
    goalies_csv = load_goalies_csv()
    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(compute_team_strength, t, comp_lookup, goalies_csv,
                          None, toi_lookup): t
                for t in TEAMS}
        for fut in as_completed(futs):
            row = fut.result()
            rows.append(row)
    df = pd.DataFrame(rows)
    df = _z_normalize_components(df)
    df = df.sort_values("team_strength", ascending=False).reset_index(drop=True)
    print(f"  Built strength for {len(df)} teams in {time.time()-t0:.1f}s", flush=True)
    return df


def main():
    df = build_all_teams()
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved → {OUT_CSV}")
    print("\nTop 10 by team_strength:")
    print(df.head(10)[["team", "team_strength", "top6_forward_strength",
                       "top_pair_d_strength", "goalie_strength"]].to_string(index=False))
    print("\nBottom 10 by team_strength:")
    print(df.tail(10)[["team", "team_strength", "top6_forward_strength",
                       "top_pair_d_strength", "goalie_strength"]].to_string(index=False))


if __name__ == "__main__":
    main()
