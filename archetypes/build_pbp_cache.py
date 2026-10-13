"""
NHL play-by-play event cache builder for the Player Archetypes module.

Self-contained data step. Reads nothing from the existing pipeline and writes
ONLY under archetypes/data/. Does not import or trigger any existing builder.

Source feed (confirmed via a diagnostic fetch of game 2025020001):
    https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Field names confirmed in the live feed:
  - venue.default (str), gameDate (YYYY-MM-DD), homeTeam.id, awayTeam.id
  - rosterSpots[]: teamId, playerId, firstName.default, lastName.default, positionCode
  - plays[]: eventId, periodDescriptor{number, periodType}, timeInPeriod "MM:SS",
             situationCode ("AGAS HSHG" 4 chars), typeDescKey, details{...}
  - situationCode "1551" => away_goalie=1, away_skaters=5, home_skaters=5, home_goalie=1
  - goal:         details.scoringPlayerId, assist1PlayerId, assist2PlayerId, goalieInNetId
  - shot-on-goal: details.shootingPlayerId, goalieInNetId
  - missed-shot:  details.shootingPlayerId
  - blocked-shot: details.shootingPlayerId, blockingPlayerId  (eventOwnerTeamId = SHOOTING team)
  - hit:          details.hittingPlayerId, hitteePlayerId
  - takeaway/giveaway: details.playerId
  - penalty:      details.committedByPlayerId (or servedByPlayerId), drawnByPlayerId,
                  descKey, duration, typeCode
  - faceoff:      details.winningPlayerId, losingPlayerId
  - all above:    details.xCoord, yCoord, zoneCode, eventOwnerTeamId

abs_secs convention matches model/shots_enriched: (period-1)*1200 + mm*60 + ss.
Shootout (periodType == "SO") events are excluded — they are not live play and
would pollute shot/goal reconciliation (shootout goals are not real-goal totals).

Resumable: each game is cached under archetypes/data/raw/{season}/ as
{game_id}.parquet (events) + {game_id}.roster.parquet (rosters). A game that
returns no plays gets a {game_id}.empty marker so it is not re-fetched. On
rerun, any game with a cached file (or marker) is skipped.

Run from project root:
    python3 archetypes/build_pbp_cache.py                 # all 3 seasons
    python3 archetypes/build_pbp_cache.py --seasons 2025  # one season
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RAW = DATA / "raw"

PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
HEADERS = {"User-Agent": "Mozilla/5.0 (archetypes-pbp-cache)"}

SEASONS = {2023: "2023_24", 2024: "2024_25", 2025: "2025_26"}
GAMES_PER_SEASON = 1312

REQUEST_DELAY = 0.35      # polite gap between requests (seconds)
MAX_RETRIES = 4
BACKOFF_BASE = 1.5

SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot"}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_game(game_id: int) -> dict | None:
    """GET the play-by-play for one game. Returns parsed JSON, or None if the
    game does not exist / has no data after retries."""
    url = PBP_URL.format(gid=game_id)
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            # 429/5xx -> backoff and retry
            time.sleep(BACKOFF_BASE ** (attempt + 1))
        except Exception:
            time.sleep(BACKOFF_BASE ** (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def _abs_secs(period: int, tip: str) -> int | None:
    try:
        mm, ss = tip.split(":")
        return (int(period) - 1) * 1200 + int(mm) * 60 + int(ss)
    except Exception:
        return None


def _strength_state(situation: str | None, owner_is_home: bool | None):
    """Return (strength_state, home/away skater+goalie digits). strength_state is
    from the event owner's perspective ('5v4' == owner on the power play)."""
    if not situation or len(situation) != 4 or not situation.isdigit():
        return None
    ag, a_sk, h_sk, hg = (int(situation[0]), int(situation[1]),
                          int(situation[2]), int(situation[3]))
    if owner_is_home is None:
        return None
    if owner_is_home:
        return f"{h_sk}v{a_sk}"
    return f"{a_sk}v{h_sk}"


def _role_cols(type_key: str, d: dict, owner=None, pid2team: dict | None = None) -> dict:
    """Map an event's details into the role columns. Absent roles stay None.

    For blocked-shot, the NHL feed emits a spurious ~9% subset whose
    blockingPlayerId belongs to the SHOOTING (event-owner) team — impossible for
    a real block (verified: filtering to blocker_team != event_owner_team makes
    per-player totals match NHL official exactly). We only credit blocker_id when
    the blocking player is on the defending team; otherwise it is left null."""
    out = {k: None for k in (
        "shooter_id", "scorer_id", "assist1_id", "assist2_id", "blocker_id",
        "hitter_id", "hittee_id", "player_id", "committed_by_id", "drawn_by_id",
        "penalty_type", "penalty_minutes", "faceoff_winner_id", "faceoff_loser_id",
    )}
    if type_key == "goal":
        out["scorer_id"] = d.get("scoringPlayerId")
        out["shooter_id"] = d.get("scoringPlayerId")
        out["assist1_id"] = d.get("assist1PlayerId")
        out["assist2_id"] = d.get("assist2PlayerId")
    elif type_key in ("shot-on-goal", "missed-shot"):
        out["shooter_id"] = d.get("shootingPlayerId")
    elif type_key == "blocked-shot":
        out["shooter_id"] = d.get("shootingPlayerId")
        blk = d.get("blockingPlayerId")
        if blk is not None and pid2team is not None and owner is not None \
                and pid2team.get(blk) == owner:
            blk = None  # spurious same-team "block" — drop the credit
        out["blocker_id"] = blk
    elif type_key == "hit":
        out["hitter_id"] = d.get("hittingPlayerId")
        out["hittee_id"] = d.get("hitteePlayerId")
    elif type_key in ("takeaway", "giveaway"):
        out["player_id"] = d.get("playerId")
    elif type_key == "penalty":
        # committedBy is absent for bench/too-many-men minors; fall back to servedBy
        out["committed_by_id"] = d.get("committedByPlayerId") or d.get("servedByPlayerId")
        out["drawn_by_id"] = d.get("drawnByPlayerId")
        out["penalty_type"] = d.get("descKey")
        out["penalty_minutes"] = d.get("duration")
    elif type_key == "faceoff":
        out["faceoff_winner_id"] = d.get("winningPlayerId")
        out["faceoff_loser_id"] = d.get("losingPlayerId")
    return out


def parse_game(pbp: dict, season_key: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    """Return (events_df, rosters_df, game_meta) or None if the game has no plays."""
    plays = pbp.get("plays") or []
    if not plays:
        return None
    game_id = pbp.get("id")
    home_id = (pbp.get("homeTeam") or {}).get("id")
    away_id = (pbp.get("awayTeam") or {}).get("id")
    venue = (pbp.get("venue") or {}).get("default")
    date = pbp.get("gameDate")

    pid2team = {r.get("playerId"): r.get("teamId") for r in (pbp.get("rosterSpots") or [])}

    rows = []
    for p in plays:
        pd_ = p.get("periodDescriptor") or {}
        if pd_.get("periodType") == "SO":
            continue  # exclude shootout
        period = pd_.get("number")
        tip = p.get("timeInPeriod") or "00:00"
        d = p.get("details") or {}
        owner = d.get("eventOwnerTeamId")
        owner_is_home = None if owner is None else (owner == home_id)
        type_key = p.get("typeDescKey")
        row = {
            "game_id": game_id,
            "season": season_key,
            "event_id": p.get("eventId"),
            "period": period,
            "time_in_period": tip,
            "abs_secs": _abs_secs(period, tip),
            "type": type_key,
            "situation_code": p.get("situationCode"),
            "strength_state": _strength_state(p.get("situationCode"), owner_is_home),
            "event_owner_team_id": owner,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "x": d.get("xCoord"),
            "y": d.get("yCoord"),
            "zone_code": d.get("zoneCode"),
        }
        row.update(_role_cols(type_key, d, owner=owner, pid2team=pid2team))
        rows.append(row)

    events = pd.DataFrame(rows)

    rspots = pbp.get("rosterSpots") or []
    rosters = pd.DataFrame([{
        "game_id": game_id,
        "player_id": r.get("playerId"),
        "team_id": r.get("teamId"),
        "position_code": r.get("positionCode"),
        "full_name": f"{(r.get('firstName') or {}).get('default', '')} "
                     f"{(r.get('lastName') or {}).get('default', '')}".strip(),
    } for r in rspots])

    meta = {"game_id": game_id, "date": date, "home_team_id": home_id,
            "away_team_id": away_id, "venue": venue}
    return events, rosters, meta


# ---------------------------------------------------------------------------
# Per-game cache (resumable unit)
# ---------------------------------------------------------------------------
def _paths(season: int, game_id: int):
    base = RAW / str(season)
    return (base / f"{game_id}.parquet",
            base / f"{game_id}.roster.parquet",
            base / f"{game_id}.empty")


def cache_one(season: int, game_id: int) -> str:
    """Fetch + parse + persist one game. Returns status: 'cached','empty','skip'."""
    ev_path, ros_path, empty_path = _paths(season, game_id)
    if ev_path.exists() or empty_path.exists():
        return "skip"
    pbp = fetch_game(game_id)
    parsed = parse_game(pbp, SEASONS[season]) if pbp else None
    if parsed is None:
        empty_path.parent.mkdir(parents=True, exist_ok=True)
        empty_path.touch()
        return "empty"
    events, rosters, meta = parsed
    # stash game meta on the events frame so season assembly needs no re-fetch
    for k in ("date", "venue"):
        events[k] = meta[k]
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(ev_path, index=False)
    rosters.to_parquet(ros_path, index=False)
    return "cached"


# ---------------------------------------------------------------------------
# Season assembly from the per-game cache
# ---------------------------------------------------------------------------
def assemble_season(season: int):
    season_key = SEASONS[season]
    base = RAW / str(season)
    ev_files = sorted(base.glob("*.parquet"))
    ev_files = [f for f in ev_files if not f.name.endswith(".roster.parquet")]
    if not ev_files:
        print(f"  [{season_key}] no cached games to assemble"); return

    ev_list, games = [], []
    for f in ev_files:
        e = pd.read_parquet(f)
        ev_list.append(e)
        g = e.iloc[0]
        games.append({"game_id": int(g["game_id"]), "date": g["date"],
                      "home_team_id": int(g["home_team_id"]),
                      "away_team_id": int(g["away_team_id"]), "venue": g["venue"]})
    events = pd.concat(ev_list, ignore_index=True)
    # drop the two meta helper cols from the event table (kept in games_ table)
    events_out = events.drop(columns=["date", "venue"])

    ros_files = sorted(base.glob("*.roster.parquet"))
    rosters = pd.concat([pd.read_parquet(f) for f in ros_files], ignore_index=True)

    # Idempotent safeguard: null any blocked-shot blocker_id whose player is on the
    # event-owner (shooting) team. Harmless if parse_game already dropped it; also
    # corrects per-game files written before the blocker fix without a re-fetch.
    teamlk = rosters.set_index(["game_id", "player_id"])["team_id"].to_dict()
    is_blk = (events_out["type"] == "blocked-shot") & events_out["blocker_id"].notna()
    if is_blk.any():
        bt = events_out.loc[is_blk].apply(
            lambda r: teamlk.get((r["game_id"], r["blocker_id"])), axis=1)
        spurious = is_blk.copy()
        spurious.loc[is_blk] = (bt == events_out.loc[is_blk, "event_owner_team_id"]).values
        events_out.loc[spurious, "blocker_id"] = None

    games_df = pd.DataFrame(games).drop_duplicates("game_id").sort_values("game_id")

    ev_out = DATA / f"pbp_events_{season_key}.parquet"
    ros_out = DATA / f"game_rosters_{season_key}.parquet"
    gm_out = DATA / f"games_{season_key}.parquet"
    events_out.to_parquet(ev_out, index=False)
    rosters.to_parquet(ros_out, index=False)
    games_df.to_parquet(gm_out, index=False)
    print(f"  [{season_key}] events={len(events_out):,} rosters={len(rosters):,} "
          f"games={len(games_df):,}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=list(SEASONS),
                    help="start years, e.g. 2023 2024 2025")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    for season in args.seasons:
        if season not in SEASONS:
            print(f"skip unknown season {season}"); continue
        if not args.assemble_only:
            print(f"=== fetching {SEASONS[season]} ===", flush=True)
            counts = {"cached": 0, "empty": 0, "skip": 0}
            for n in range(1, GAMES_PER_SEASON + 1):
                gid = int(f"{season}02{n:04d}")
                status = cache_one(season, gid)
                counts[status] += 1
                if status == "cached":
                    time.sleep(REQUEST_DELAY)
                if n % 100 == 0:
                    print(f"  {SEASONS[season]} {n}/{GAMES_PER_SEASON} "
                          f"cached={counts['cached']} empty={counts['empty']} "
                          f"skip={counts['skip']}", flush=True)
            print(f"  {SEASONS[season]} done: {counts}", flush=True)
        assemble_season(season)


if __name__ == "__main__":
    main()
