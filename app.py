import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import unicodedata
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request

from news_classifier import is_real_move, rank_big_news

app = Flask(__name__)
# Live-reload during development: re-read templates on every request,
# never cache static JS/CSS in the browser. So edit → save → refresh works.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("nhl_brief")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NHL_API = "https://api-web.nhle.com/v1"
MP_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular"
NST_BASE = "https://www.naturalstattrick.com"   # only used by the (blocked) line-combinations attempt

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

BLOCKED_SENTINEL = {
    "blocked": True,
    "message": "Full access requires an Evolving Hockey subscription",
}

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
_cache: dict = {}
CACHE_TTL = 21600  # 6 hours


def _is_likely_failure_response(result) -> bool:
    """Heuristic: a dict that's empty, or whose top-level keys are all empty
    lists / None / False / error-flagged, almost certainly came from a
    transient upstream failure and should not be cached for 6 hours. Lists
    with zero entries are also suspect."""
    if result is None:
        return True
    if isinstance(result, dict):
        if not result:
            return True
        if result.get("error") or result.get("blocked") or result.get("empty") is True:
            return True
        # All values empty/None/False?
        def _empty(v):
            if v is None or v is False:
                return True
            if isinstance(v, (list, dict, str)) and len(v) == 0:
                return True
            return False
        if all(_empty(v) for v in result.values()):
            return True
    if isinstance(result, list) and len(result) == 0:
        return True
    return False


def cached(key: str, fn):
    now = time.time()
    entry = _cache.get(key)
    if entry and now - entry["ts"] < CACHE_TTL:
        return entry["data"]
    result = fn()
    # Don't cache obviously-broken empty responses — they tend to come from
    # transient upstream errors and poison the dashboard for 6 hours.
    if not _is_likely_failure_response(result):
        _cache[key] = {"data": result, "ts": now}
    return result


@app.route("/api/admin/cache-clear")
def api_admin_cache_clear():
    """Drop entries from the in-process `_cache` dict (which is shared by
    both `cached()` and `cached_ttl()`). Useful after a transient NHL API
    failure poisons the cache with an empty payload. Accepts an optional
    ?keys=k1,k2 to clear only specific keys; default clears everything."""
    requested = request.args.get("keys")
    before = len(_cache)
    if requested:
        cleared = []
        for k in [x.strip() for x in requested.split(",") if x.strip()]:
            if k in _cache:
                del _cache[k]
                cleared.append(k)
        return jsonify({"cleared_keys": cleared, "remaining": len(_cache)})
    _cache.clear()
    return jsonify({"cleared_count": before, "remaining": len(_cache)})


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def get_json(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()


def get_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return pd.read_csv(StringIO(r.text))


# ---------------------------------------------------------------------------
# Season auto-detection
# ---------------------------------------------------------------------------
def _date_based_season() -> str:
    """Fallback: derive season string from today's date (Oct+ = YYYY YYYY+1)."""
    t = date.today()
    start = t.year if t.month >= 10 else t.year - 1
    return f"{start}{start + 1}"


def _resolve_season_info() -> dict:
    """Hit /v1/schedule/now (primary) then /v1/season (fallback) then date math.

    Returns {'season': '20252026', 'label': '2025-26', 'regular_start': ...,
             'regular_end': ..., 'playoff_end': ..., 'preseason_start': ...}.
    """
    info = {
        "season": None,
        "label": None,
        "regular_start": None,
        "regular_end": None,
        "playoff_end": None,
        "preseason_start": None,
    }
    try:
        sched = get_json(f"{NHL_API}/schedule/now")
        for day in (sched.get("gameWeek") or []):
            for g in (day.get("games") or []):
                if g.get("season"):
                    info["season"] = str(g["season"])
                    break
            if info["season"]:
                break
        info["regular_start"] = sched.get("regularSeasonStartDate")
        info["regular_end"] = sched.get("regularSeasonEndDate")
        info["playoff_end"] = sched.get("playoffEndDate")
        info["preseason_start"] = sched.get("preSeasonStartDate")
    except Exception as e:
        log.info("schedule/now season probe failed: %s", e)

    if not info["season"]:
        try:
            seasons = get_json(f"{NHL_API}/season")
            if isinstance(seasons, list) and seasons:
                info["season"] = str(seasons[-1])
        except Exception as e:
            log.info("season list fallback failed: %s", e)

    if not info["season"]:
        info["season"] = _date_based_season()

    s = info["season"]
    if len(s) == 8 and s.isdigit():
        info["label"] = f"{s[:4]}-{s[6:8]}"
    else:
        info["label"] = s

    return info


def season_info() -> dict:
    return cached("nhl_season_info", _resolve_season_info)


def current_season() -> str:
    """Current NHL season string, e.g. '20252026'. Cached for CACHE_TTL."""
    return season_info()["season"]


def season_label() -> str:
    """UI-friendly season label, e.g. '2025-26'."""
    return season_info()["label"]


def season_state() -> str:
    """Return 'preseason' | 'regular' | 'playoffs' | 'offseason' based on today vs API dates."""
    def _resolve():
        info = season_info()
        today = date.today().isoformat()
        pre = info.get("preseason_start")
        rs = info.get("regular_start")
        re_ = info.get("regular_end")
        po = info.get("playoff_end")
        if rs and re_ and rs <= today <= re_:
            return "regular"
        if re_ and po and re_ < today <= po:
            return "playoffs"
        if pre and rs and pre <= today < rs:
            return "preseason"
        # Date-based fallback when API dates are missing or today is outside known windows
        t = date.today()
        if t.month >= 10 or t.month <= 3:
            return "regular"
        if 4 <= t.month <= 6:
            return "playoffs"
        return "offseason"
    return cached("nhl_season_state", _resolve)


# ---------------------------------------------------------------------------
# 3-tier fallback
# ---------------------------------------------------------------------------
def with_fallback(*provider_fns):
    for fn in provider_fns:
        try:
            result = fn()
            if result is not None:
                return result
        except Exception as e:
            log.warning("%s failed: %s", fn.__name__, e)
    return dict(BLOCKED_SENTINEL)


# ---------------------------------------------------------------------------
# NHL API helpers
# ---------------------------------------------------------------------------
def _format_player_leader(p: dict) -> dict:
    return {
        "id": p["id"],
        "name": f"{p['firstName']['default']} {p['lastName']['default']}",
        "number": p.get("sweaterNumber"),
        "headshot": p.get("headshot", ""),
        "team": p.get("teamAbbrev", ""),
        "team_logo": p.get("teamLogo", ""),
        "position": p.get("position", ""),
        "value": p.get("value", 0),
    }


# ---------------------------------------------------------------------------
# ESPN game URL lookup — maps NHL game date + team abbrevs → ESPN game page
# ---------------------------------------------------------------------------
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"


def _espn_game_links_for_date(yyyy_mm_dd: str) -> dict:
    """
    Returns a dict mapping a frozenset({away_abbrev, home_abbrev}) → ESPN game URL.
    Cached for the duration of the cache TTL.
    """
    date_compact = yyyy_mm_dd.replace("-", "")

    def _fetch():
        try:
            r = requests.get(ESPN_SCOREBOARD, params={"dates": date_compact},
                             headers=HEADERS, timeout=10)
            r.raise_for_status()
            data = r.json()
            mapping = {}
            for event in data.get("events", []) or []:
                competitions = event.get("competitions", []) or []
                if not competitions:
                    continue
                competitors = competitions[0].get("competitors", []) or []
                abbrevs = []
                for c in competitors:
                    abbr = (c.get("team") or {}).get("abbreviation", "")
                    if abbr:
                        abbrevs.append(abbr.upper())
                if len(abbrevs) != 2:
                    continue
                link = ""
                for l in event.get("links", []) or []:
                    if l.get("href"):
                        link = l["href"]
                        break
                if link:
                    mapping[frozenset(abbrevs)] = link
            return mapping
        except Exception as e:
            log.warning("ESPN scoreboard lookup failed for %s: %s", date_compact, e)
            return {}

    return cached(f"espn_links_{date_compact}", _fetch)


# NHL → ESPN team abbreviation map (only differences; identical ones omitted)
_NHL_TO_ESPN_ABBREV = {
    "TBL": "TB",
    "LAK": "LA",
    "SJS": "SJ",
    "NJD": "NJ",
    "WSH": "WSH",
}


def _espn_link_for_game(yyyy_mm_dd: str, away_abbrev: str, home_abbrev: str) -> str:
    if not yyyy_mm_dd or not away_abbrev or not home_abbrev:
        return ""
    mapping = _espn_game_links_for_date(yyyy_mm_dd)
    if not mapping:
        return ""
    away_e = _NHL_TO_ESPN_ABBREV.get(away_abbrev, away_abbrev).upper()
    home_e = _NHL_TO_ESPN_ABBREV.get(home_abbrev, home_abbrev).upper()
    return mapping.get(frozenset([away_e, home_e]), "")


# ---------------------------------------------------------------------------
# Routes — standard NHL data
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    info = season_info()
    return render_template(
        "index.html",
        season=info["season"],
        season_label=info["label"],
        season_state=season_state(),
    )


@app.route("/api/season-info")
def api_season_info():
    """Surface the auto-detected season + state to the frontend."""
    info = season_info()
    s = info["season"]
    end_year = None
    if s and len(s) == 8 and s.isdigit():
        end_year = int(s[4:8])
    return jsonify({
        "season": info["season"],
        "label": info["label"],
        "end_year": end_year,
        "state": season_state(),
        "regular_start": info.get("regular_start"),
        "regular_end": info.get("regular_end"),
        "playoff_end": info.get("playoff_end"),
    })


def _format_score_game(g: dict, day: str) -> dict:
    """Convert a /v1/score game payload into the shape the dashboard expects."""
    state = g.get("gameState", "")
    # Preseason finals report gameState "FINAL" and never advance to "OFF"
    # (only regular-season/playoff games reach "OFF"). Treat both as completed
    # so preseason scores show. Matches the pattern at api_next_playoff_games.
    completed = state in ("OFF", "FINAL")
    away = g.get("awayTeam", {})
    home = g.get("homeTeam", {})
    outcome = g.get("gameOutcome", {}).get("lastPeriodType", "REG") if completed else None
    return {
        "id": g.get("id"),
        "date": g.get("gameDate", day),
        "venue": g.get("venue", {}).get("default", ""),
        "state": state,
        "outcome": outcome,
        "away": {
            "abbrev": away.get("abbrev", ""),
            "name": away.get("name", {}).get("default", ""),
            "score": away.get("score", 0) if completed else None,
            "sog": away.get("sog", 0) if completed else None,
            "logo": away.get("logo", ""),
        },
        "home": {
            "abbrev": home.get("abbrev", ""),
            "name": home.get("name", {}).get("default", ""),
            "score": home.get("score", 0) if completed else None,
            "sog": home.get("sog", 0) if completed else None,
            "logo": home.get("logo", ""),
        },
        "series": g.get("seriesStatus"),
        "gamecenter_link": g.get("gameCenterLink", ""),
        "recap_link": g.get("threeMinRecap", ""),
        "condensed_link": g.get("condensedGame", ""),
        "espn_link": _espn_link_for_game(
            g.get("gameDate", day),
            away.get("abbrev", ""),
            home.get("abbrev", ""),
        ),
        "sportsnet_link": f"https://www.sportsnet.ca/hockey/nhl/scoreboard/?date={g.get('gameDate', day)}",
    }


@app.route("/api/games")
def api_games():
    """Return NHL games for a specific date, or the most recent date with games.

    With ?date=YYYY-MM-DD: returns all games on that date, all game types and states.
    Without a date: walks back from today to find the most recent date that had any
    games, and returns those games (used for the default carousel selection).
    """
    MAX_DAYS_BACK = 21

    def _fetch_day(day: str) -> list:
        def _inner():
            try:
                data = get_json(f"{NHL_API}/score/{day}")
            except Exception as e:
                log.warning("score fetch %s failed: %s", day, e)
                return []
            return data.get("games", []) or []
        return cached(f"score_{day}", _inner)

    def _games_for_day(day: str) -> list:
        out = []
        for g in _fetch_day(day):
            out.append(_format_score_game(g, day))
        out.sort(key=lambda x: x["id"])
        return out

    date_param = (request.args.get("date") or "").strip()

    try:
        if date_param:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_param):
                return jsonify({"date": None, "games": [], "error": "invalid date"}), 400
            games = _games_for_day(date_param)
            return jsonify({"date": date_param, "games": games})

        for delta in range(MAX_DAYS_BACK):
            day = (date.today() - timedelta(days=delta)).strftime("%Y-%m-%d")
            games = _games_for_day(day)
            if games:
                return jsonify({"date": day, "games": games})
        return jsonify({"date": None, "games": []})
    except Exception as e:
        log.error("api_games error: %s", e)
        return jsonify({"date": None, "games": [], "error": str(e)}), 500


@app.route("/api/game-details/<int:game_id>")
def api_game_details(game_id: int):
    """Fetch enriched details for a single game: three stars, scoring, goalies, video links."""
    def _fetch():
        landing = get_json(f"{NHL_API}/gamecenter/{game_id}/landing")
        boxscore = get_json(f"{NHL_API}/gamecenter/{game_id}/boxscore")

        summary = landing.get("summary", {}) or {}
        away = landing.get("awayTeam", {})
        home = landing.get("homeTeam", {})
        away_abbrev = (away.get("abbrev") if isinstance(away.get("abbrev"), str)
                       else away.get("abbrev", {}).get("default", ""))
        home_abbrev = (home.get("abbrev") if isinstance(home.get("abbrev"), str)
                       else home.get("abbrev", {}).get("default", ""))

        # Three stars
        stars = []
        for s in summary.get("threeStars", []) or []:
            stars.append({
                "star": s.get("star"),
                "name": s.get("name", {}).get("default", ""),
                "team": s.get("teamAbbrev", ""),
                "position": s.get("position", ""),
                "goals": s.get("goals"),
                "assists": s.get("assists"),
                "points": s.get("points"),
                "save_pct": s.get("savePctg"),
                "headshot": s.get("headshot", ""),
            })

        # Scoring summary — flatten goals across periods
        scoring = []
        top_clip = None
        for period_block in summary.get("scoring", []) or []:
            pd = period_block.get("periodDescriptor", {}) or {}
            period_num = pd.get("number")
            period_type = pd.get("periodType", "REG")
            for g in period_block.get("goals", []) or []:
                team_abbrev_raw = g.get("teamAbbrev", "")
                t_abbrev = (team_abbrev_raw if isinstance(team_abbrev_raw, str)
                            else team_abbrev_raw.get("default", ""))
                assists = []
                for a in g.get("assists", []) or []:
                    assists.append(a.get("name", {}).get("default", ""))
                clip = g.get("highlightClipSharingUrl") or ""
                if clip and top_clip is None:
                    top_clip = {
                        "scorer": f"{g.get('firstName',{}).get('default','')} {g.get('lastName',{}).get('default','')}",
                        "url": clip,
                    }
                scoring.append({
                    "period": period_num,
                    "period_type": period_type,
                    "time": g.get("timeInPeriod", ""),
                    "team": t_abbrev,
                    "scorer": f"{g.get('firstName',{}).get('default','')} {g.get('lastName',{}).get('default','')}",
                    "assists": assists,
                    "strength": g.get("strength", "ev"),  # ev / pp / sh
                    "shot_type": g.get("shotType", ""),
                    "goal_modifier": g.get("goalModifier", ""),
                    "away_score": g.get("awayScore"),
                    "home_score": g.get("homeScore"),
                })

        # Goalie line — find starter for each team
        goalies = {"away": None, "home": None}
        for side_key, side_data in (("away", "awayTeam"), ("home", "homeTeam")):
            for gg in boxscore.get("playerByGameStats", {}).get(side_data, {}).get("goalies", []):
                if gg.get("starter"):
                    goalies[side_key] = {
                        "name": gg.get("name", {}).get("default", ""),
                        "saves": gg.get("saves", 0),
                        "shots_against": gg.get("shotsAgainst", 0),
                        "goals_against": gg.get("goalsAgainst", 0),
                        "save_pct": gg.get("savePctg"),
                        "toi": gg.get("toi", ""),
                    }
                    break

        return {
            "id": game_id,
            "away_abbrev": away_abbrev,
            "home_abbrev": home_abbrev,
            "stars": stars,
            "scoring": scoring,
            "goalies": goalies,
            "top_clip": top_clip,
        }

    try:
        return jsonify(cached(f"game_details_{game_id}", _fetch))
    except Exception as e:
        log.error("api_game_details error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/leaders")
def api_leaders():
    def _fetch():
        skaters = {}
        for cat in ["points", "goals", "assists"]:
            raw = get_json(
                f"{NHL_API}/skater-stats-leaders/{current_season()}/2",
                params={"categories": cat, "limit": 5},
            )
            skaters[cat] = [_format_player_leader(p) for p in raw.get(cat, [])]

        goalie_raw = get_json(
            f"{NHL_API}/goalie-stats-leaders/{current_season()}/2",
            params={"categories": "wins", "limit": 5},
        )
        goalies = [_format_player_leader(p) for p in goalie_raw.get("wins", [])]
        return {"skaters": skaters, "goalies": goalies}

    try:
        return jsonify(cached("leaders", _fetch))
    except Exception as e:
        log.error("api_leaders error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/standings")
def api_standings():
    def _fetch():
        raw = get_json(f"{NHL_API}/standings/now")
        result: dict = {}
        for team in raw.get("standings", []):
            conf = team.get("conferenceName", "")
            div = team.get("divisionName", "")
            if not conf:
                continue
            # teamAbbrev and teamName are both {"default": "..."} dicts
            abbrev = team.get("teamAbbrev", {})
            abbrev = abbrev.get("default", "") if isinstance(abbrev, dict) else str(abbrev)
            result.setdefault(conf, {}).setdefault(div, []).append({
                "abbrev": abbrev,
                "name": team.get("teamName", {}).get("default", ""),
                "logo": f"https://assets.nhle.com/logos/nhl/svg/{abbrev}_light.svg",
                "gp": team.get("gamesPlayed", 0),
                "w": team.get("wins", 0),
                "l": team.get("losses", 0),
                "otl": team.get("otLosses", 0),
                "pts": team.get("points", 0),
                "div_rank": team.get("divisionSequence", 99),
                "conf_rank": team.get("conferenceSequence", 99),
                "wc_rank": team.get("wildcardSequence"),
                "clinch": team.get("clinchIndicator", ""),
            })
        # Sort each division by division rank
        for conf in result:
            for div in result[conf]:
                result[conf][div].sort(key=lambda t: t["div_rank"])
        return result

    try:
        return jsonify(cached("standings", _fetch))
    except Exception as e:
        log.error("api_standings error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Advanced stats — sourced from my own models (see docs/dashboard_self_generation_audit.md)
# ---------------------------------------------------------------------------
# Evolving Hockey and Natural Stat Trick scrapes were removed in the
# 2026-06-06 self-generation refactor. Every advanced-stat endpoint now
# reads from a CSV in `model/` produced by my own pipeline:
#   - skater GAR  → composite_ratings_sim.csv (composite_war column)
#   - skater xGAR → skater_xgar_self_generated.csv (my xG model applied
#                    to individual shots, in wins units)
#   - goalie GSAX → goalie_gsax_self_generated.csv (my xG model applied
#                    to shots faced via goalie shifts intersection)
#   - team xGF%   → team_xgf_self_generated.csv
#   - skater on-ice xGF% → skater_onice_xgf_self_generated.csv
#   - Game Score  → skater_game_score_self_generated.csv (Galamini formula)


def _nst_scrape(url: str) -> list | None:
    """Attempt Natural Stat Trick scrape — typically returns 403.
    Retained only for the line-analytics endpoint, which has no MP equivalent."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            return None
        rows = []
        header_row = tables[0].find("tr")
        if not header_row:
            return None
        cols = [th.get_text(strip=True) for th in header_row.find_all(["th", "td"])]
        for tr in tables[0].find_all("tr")[1:]:
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cells:
                rows.append(dict(zip(cols, cells)))
        return rows if rows else None
    except Exception as e:
        log.warning("NST scrape failed: %s", e)
        return None


def _mp_skaters() -> pd.DataFrame:
    df = cached("mp_skaters_csv", lambda: get_csv(f"{MP_BASE}/skaters.csv"))
    for col in ["gameScore", "I_F_flurryScoreVenueAdjustedxGoals", "icetime", "games_played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _mp_goalies() -> pd.DataFrame:
    df = cached("mp_goalies_csv", lambda: get_csv(f"{MP_BASE}/goalies.csv"))
    for col in ["xGoals", "goals", "icetime", "games_played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def _mp_teams() -> pd.DataFrame:
    df = cached("mp_teams_csv", lambda: get_csv(f"{MP_BASE}/teams.csv"))
    for col in ["xGoalsPercentage", "corsiPercentage", "highDangerShotsFor", "highDangerShotsAgainst", "games_played"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


# GAR leaderboard
def _position_filter_df(df, position: str):
    """Filter a MoneyPuck skater DataFrame by position bucket: 'F', 'D', or 'all'."""
    if position == "F":
        return df[df["position"].isin(["C", "L", "R"])]
    if position == "D":
        return df[df["position"] == "D"]
    return df


@app.route("/api/gar-leaders")
def api_gar_leaders():
    """Top 10 skater GAR — sourced from my composite_war model
    (model/composite_ratings_sim.csv)."""
    position = (request.args.get("position") or "all").upper()

    def _fetch():
        df_meta = _mp_skaters()
        df_meta = df_meta[df_meta["situation"] == "all"].copy()
        df_meta = _position_filter_df(df_meta, position)
        meta_by_pid = {int(r["playerId"]): r for _, r in df_meta.iterrows()}

        comp = _load_composite_war()
        rows = []
        for pid, c in comp.items():
            if pid not in meta_by_pid:
                continue
            m = meta_by_pid[pid]
            rows.append({
                "name": m["name"],
                "team": m["team"],
                "position": m["position"],
                # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
                "value": round(float(c["composite_war"]) * 12, 2),
                "games": int(m.get("games_played", 0)),
            })
        rows.sort(key=lambda r: r["value"], reverse=True)
        return {
            "source": "self_generated",
            "metric": "Composite WAR (RAPM + individual + playmaking + PP + PK blend)",
            "position": position,
            "players": rows[:10],
        }

    return jsonify(cached(f"gar_leaders_{position}", _fetch))


# xGAR leaderboard
@app.route("/api/xgar-leaders")
def api_xgar_leaders():
    """Top 10 skater xGAR — sourced from my self-generated xG-driven xGAR
    (model/skater_xgar_self_generated.csv, from my xG model on individual shots)."""
    position = (request.args.get("position") or "all").upper()

    def _fetch():
        df_meta = _mp_skaters()
        df_meta = df_meta[df_meta["situation"] == "all"].copy()
        df_meta = _position_filter_df(df_meta, position)
        meta_by_pid = {int(r["playerId"]): r for _, r in df_meta.iterrows()}

        xgar_lookup = _load_skater_xgar()
        comp_lookup = _load_composite_war()
        rows = []
        for pid, x in xgar_lookup.items():
            if pid not in meta_by_pid:
                continue
            m = meta_by_pid[pid]
            gar_val = comp_lookup.get(pid, {}).get("composite_war", 0.0)
            rows.append({
                "name": m["name"],
                "team": m["team"],
                "position": m["position"],
                "xgar_value": round(float(x["xgar"]), 2),
                # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
                "gar_value": round(float(gar_val) * 12, 2),
                "games": int(m.get("games_played", 0)),
            })
        rows.sort(key=lambda r: r["xgar_value"], reverse=True)
        return {
            "source": "self_generated",
            "metric": "xGAR (my xG model applied to individual shots, wins-units)",
            "players": rows[:10],
        }

    return jsonify(cached(f"xgar_leaders_{position}", _fetch))


# Team analytics
@app.route("/api/team-analytics")
def api_team_analytics():
    """Team analytics. xGF% is sourced from my own xG model
    (team_xgf_self_generated.csv); CF% and HDCF% come from MoneyPuck raw counts."""
    def _fetch():
        df = _mp_teams()
        df5 = df[df["situation"] == "5on5"].copy()
        my_team_xgf = _load_team_xgf()
        teams = []
        for _, row in df5.iterrows():
            tm = str(row["team"])
            hd_for = float(row["highDangerShotsFor"])
            hd_ag = float(row["highDangerShotsAgainst"])
            hdcf_pct = (hd_for / (hd_for + hd_ag) * 100) if (hd_for + hd_ag) > 0 else 50.0
            my_x = my_team_xgf.get(tm)
            xgf_pct = (float(my_x["xgf_pct"]) if my_x
                        else round(float(row["xGoalsPercentage"]) * 100, 1))
            teams.append({
                "team": tm,
                "gp": int(row["games_played"]),
                "xgf_pct": round(xgf_pct, 1),
                "cf_pct": round(float(row["corsiPercentage"]) * 100, 1),
                "hdcf_pct": round(hdcf_pct, 1),
            })
        teams.sort(key=lambda t: t["xgf_pct"], reverse=True)
        return {"source": "self_generated_xgf+moneypuck_counts", "teams": teams}

    return jsonify(cached("team_analytics", _fetch))


# Goalie analytics
@app.route("/api/goalie-analytics")
def api_goalie_analytics():
    def _from_self_generated():
        """Self-generated GSAX from my xG model + goalie shifts."""
        df_g = _mp_goalies()
        df_all = df_g[(df_g["situation"] == "all") & (df_g["games_played"] >= 10)].copy()
        my_gsax = _load_goalie_gsax()
        rows = []
        for _, r in df_all.iterrows():
            pid = int(r["playerId"])
            my_g = my_gsax.get(pid)
            if my_g is None:
                continue
            rows.append({
                "name": r["name"],
                "team": r["team"],
                "gsax": round(float(my_g["gsax"]), 2),
                "games": int(r["games_played"]),
                "toi_min": round(float(r["icetime"]), 0),
            })
        rows.sort(key=lambda g: g["gsax"], reverse=True)
        return {
            "source": "self_generated",
            "metric": "GSAX (Goals Saved Above Expected, my xG model)",
            "goalies": rows[:10],
        }

    def _fetch():
        return _from_self_generated()

    return jsonify(cached("goalie_analytics", _fetch))


# Line analytics — no MoneyPuck fallback (no line combination data)
@app.route("/api/line-analytics")
def api_line_analytics():
    """Forward-line combinations. There is no public data source for line
    combinations (NST is the only one and blocks bots), so this endpoint
    returns a blocked-state message rather than displaying third-party data.
    Building line combinations from my shift data is feasible (group by
    (game_id, abs_secs) and find the 3-forward set on ice), but is deferred
    pending demand."""
    return jsonify({
        "blocked": True,
        "message": ("Line combinations are not currently surfaced from a "
                    "self-generated table. Forward-trio identification from "
                    "my shifts dataset is a planned extension."),
    })


# ---------------------------------------------------------------------------
# Morning Brief Zone 2/3 — Playoff Form, News, Bracket
# ---------------------------------------------------------------------------
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import timezone


def _team_conference_map() -> dict:
    """Build a {abbrev: 'Eastern'|'Western'} map from current standings."""
    def _fetch():
        raw = get_json(f"{NHL_API}/standings/now")
        out = {}
        for t in raw.get("standings", []):
            abbrev = t.get("teamAbbrev", {})
            abbrev = abbrev.get("default", "") if isinstance(abbrev, dict) else abbrev
            conf = t.get("conferenceName", "")
            if abbrev and conf:
                out[abbrev] = conf
        return out
    return cached("conf_map", _fetch)


def _team_last_10(abbrev: str, season: str | None = None) -> dict:
    """Last 10 completed games for a team — returns list of W/L/OTL with GF/GA + streak."""
    season = season or current_season()
    def _fetch():
        try:
            data = get_json(f"{NHL_API}/club-schedule-season/{abbrev}/{season}")
        except Exception as e:
            log.warning("schedule fetch %s failed: %s", abbrev, e)
            return {"games": [], "streak": "—", "gf": 0, "ga": 0}

        completed = [g for g in (data.get("games") or []) if g.get("gameState") == "OFF"]
        last10 = completed[-10:]

        results = []
        gf_total = 0
        ga_total = 0
        for g in last10:
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            is_home = home.get("abbrev") == abbrev
            our_score = (home if is_home else away).get("score", 0) or 0
            opp_score = (away if is_home else home).get("score", 0) or 0
            opp_abbrev = (away if is_home else home).get("abbrev", "")
            last_period = g.get("gameOutcome", {}).get("lastPeriodType", "REG")
            if our_score > opp_score:
                result = "W"
            elif last_period in ("OT", "SO"):
                result = "OTL"
            else:
                result = "L"
            gf_total += our_score
            ga_total += opp_score
            results.append({
                "date": g.get("gameDate"),
                "opp": opp_abbrev,
                "is_home": is_home,
                "gf": our_score,
                "ga": opp_score,
                "result": result,
                "last_period": last_period,
            })

        # Streak = run of identical outcomes at the end (W/L combine OTL into L for streak)
        streak = "—"
        if results:
            last = results[-1]["result"]
            streak_kind = "W" if last == "W" else "L"
            count = 1
            for i in range(len(results) - 2, -1, -1):
                r = results[i]["result"]
                r_kind = "W" if r == "W" else "L"
                if r_kind == streak_kind:
                    count += 1
                else:
                    break
            streak = f"{streak_kind}{count}"

        return {"games": results, "streak": streak, "gf": gf_total, "ga": ga_total}

    return cached(f"last10_{abbrev}_{season}", _fetch)


def _build_playoff_carousel() -> dict:
    """Fetch the playoff carousel; returns parsed structure with current round info."""
    def _fetch():
        return get_json(f"{NHL_API}/playoff-series/carousel/{current_season()}")
    return cached("playoff_carousel", _fetch)


@app.route("/api/playoff-form")
def api_playoff_form():
    def _fetch():
        try:
            carousel = _build_playoff_carousel()
        except Exception as e:
            log.warning("playoff carousel failed: %s", e)
            return {"teams": [], "round": None, "error": "Could not load playoff data"}

        current_round = carousel.get("currentRound", 1)
        rounds = carousel.get("rounds", []) or []

        # Find the rounds at-or-before current_round; the team's most recent series
        # is the highest round in which they still appear.
        team_series = {}  # abbrev → series_info
        for rnd in rounds:
            rnd_num = rnd.get("roundNumber")
            if rnd_num is None or rnd_num > current_round:
                continue
            for s in (rnd.get("series") or []):
                top = s.get("topSeed", {}) or {}
                bot = s.get("bottomSeed", {}) or {}
                top_ab = top.get("abbrev")
                bot_ab = bot.get("abbrev")
                needed = s.get("neededToWin", 4)
                top_w = top.get("wins", 0)
                bot_w = bot.get("wins", 0)
                series_complete = (top_w >= needed) or (bot_w >= needed)
                winning_id = s.get("winningTeamId")
                if top_ab:
                    team_series[top_ab] = {
                        "round": rnd_num,
                        "wins": top_w, "opp_wins": bot_w,
                        "opp": bot_ab, "complete": series_complete,
                        "advanced": (winning_id == top.get("id")) if series_complete else None,
                    }
                if bot_ab:
                    team_series[bot_ab] = {
                        "round": rnd_num,
                        "wins": bot_w, "opp_wins": top_w,
                        "opp": top_ab, "complete": series_complete,
                        "advanced": (winning_id == bot.get("id")) if series_complete else None,
                    }

        # Keep only teams whose most-recent series is the current round AND they haven't been eliminated.
        # A team is eliminated if their current-round series is complete and they didn't advance.
        alive = {}
        for ab, info in team_series.items():
            if info["round"] != current_round:
                continue  # team's last appearance was earlier — they're already out
            if info["complete"] and info["advanced"] is False:
                continue  # lost in current round
            alive[ab] = info

        # Lookup conferences
        try:
            conf_map = _team_conference_map()
        except Exception as e:
            log.warning("conference map failed: %s", e)
            conf_map = {}

        # Pull team metadata from carousel (logos, names)
        team_meta = {}
        for rnd in rounds:
            for s in (rnd.get("series") or []):
                for seed_key in ("topSeed", "bottomSeed"):
                    seed = s.get(seed_key, {}) or {}
                    ab = seed.get("abbrev")
                    if ab and ab not in team_meta:
                        team_meta[ab] = {
                            "id": seed.get("id"),
                            "abbrev": ab,
                            "logo": seed.get("logo"),
                        }

        results = []
        for ab, info in alive.items():
            l10 = _team_last_10(ab)
            meta = team_meta.get(ab, {"abbrev": ab})
            conf = conf_map.get(ab, "Unknown")

            # Series situation text
            if info["complete"]:
                situation = f"won series {info['wins']}-{info['opp_wins']} vs {info['opp']}"
            else:
                if info["wins"] > info["opp_wins"]:
                    situation = f"leads {info['wins']}-{info['opp_wins']} vs {info['opp']}"
                elif info["wins"] < info["opp_wins"]:
                    situation = f"trails {info['wins']}-{info['opp_wins']} vs {info['opp']}"
                else:
                    situation = f"tied {info['wins']}-{info['opp_wins']} vs {info['opp']}"

            results.append({
                "abbrev": ab,
                "logo": meta.get("logo"),
                "conference": conf,
                "series_situation": situation,
                "series_round": info["round"],
                "series_complete": info["complete"],
                "last10": l10["games"],
                "streak": l10["streak"],
                "gf": l10["gf"],
                "ga": l10["ga"],
            })

        # Sort: Eastern first, then Western, then alphabetical
        conf_order = {"Eastern": 0, "Western": 1}
        results.sort(key=lambda t: (conf_order.get(t["conference"], 9), t["abbrev"]))

        return {"round": current_round, "teams": results}

    try:
        return jsonify(cached("playoff_form", _fetch))
    except Exception as e:
        log.error("api_playoff_form error: %s", e)
        return jsonify({"error": str(e), "teams": []}), 500


# ---------------------------------------------------------------------------
# News — RSS chain
# ---------------------------------------------------------------------------
NEWS_SOURCES = [
    {"name": "NHL.com",     "url": "https://www.nhl.com/rss/news.xml"},
    {"name": "TSN",         "url": "https://www.tsn.ca/rss/tsn_nhl.xml"},
    {"name": "ESPN",        "url": "https://www.espn.com/espn/rss/nhl/news"},
    {"name": "Sportsnet",   "url": "https://www.sportsnet.ca/hockey/nhl/feed/"},
]

TXN_KEYWORDS = ("trade", "trad", "sign", "recall", "waive", "waiv",
                "injur", " ir ", " ir.", "ir)", "(ir", "suspen", "extension",
                "extends", "extend", "claim", "place on", "activates")


def _parse_rss(xml_text: str, source_name: str) -> list:
    """Parse RSS XML → list of {title, link, pub_date_iso, source}."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        log.warning("RSS parse error from %s: %s", source_name, e)
        return []

    channel = root.find("channel") or root
    items = []
    for it in channel.findall("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        if not title or not link:
            continue
        # Parse pub date to ISO
        pub_iso = None
        if pub:
            try:
                dt = parsedate_to_datetime(pub)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                pub_iso = dt.astimezone(timezone.utc).isoformat()
            except (TypeError, ValueError):
                pub_iso = None
        items.append({
            "title": title,
            "link": link,
            "pub_date": pub_iso,
            "source": source_name,
        })
    return items


NEWS_MAX_AGE_DAYS = 14         # drop anything older than this
# Sportsnet's NHL feed ships a few perennial "section landing" items
# (titled "NHL Featured", "NHL Headlines", "NHL Page Featured 2 Items",
# etc.) with stale pub_dates spanning weeks to years. They're not articles;
# filter them out by title.
NEWS_TITLE_DENYLIST = re.compile(
    r"^(?:nhl\s+(?:featured|headlines?|page\s+featured.*))$",
    re.IGNORECASE,
)


def _filter_fresh_news(items: list) -> list:
    """Drop items older than NEWS_MAX_AGE_DAYS or whose title matches the
    section-landing denylist, then sort by pub_date desc."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS)
    fresh = []
    for it in items:
        if NEWS_TITLE_DENYLIST.match((it.get("title") or "").strip()):
            continue
        pub_iso = it.get("pub_date")
        if not pub_iso:
            # no date — keep but treat as old in sort
            fresh.append(it)
            continue
        try:
            dt = datetime.fromisoformat(pub_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            fresh.append(it)
            continue
        if dt < cutoff:
            continue
        fresh.append(it)
    # Sort newest first; items without a parseable date go to the bottom
    def _sort_key(it):
        pub_iso = it.get("pub_date") or ""
        try:
            dt = datetime.fromisoformat(pub_iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (TypeError, ValueError):
            return datetime.min.replace(tzinfo=timezone.utc)
    fresh.sort(key=_sort_key, reverse=True)
    return fresh


def _news_pool(limit=25):
    """Return (source_name, [items]) from the first news source that yields
    fresh items. Shared by /api/news and the Morning Brief combined feed."""
    for src in NEWS_SOURCES:
        try:
            r = requests.get(src["url"], headers=HEADERS, timeout=8)
            if r.status_code != 200:
                log.info("news source %s returned %s", src["name"], r.status_code)
                continue
            items = _filter_fresh_news(_parse_rss(r.text, src["name"]))
            if items:
                return src["name"], items[:limit]
        except Exception as e:
            log.info("news source %s error: %s", src["name"], e)
            continue
    return None, []


@app.route("/api/news")
def api_news():
    def _fetch():
        source, items = _news_pool(limit=10)
        if items:
            return {"source": source, "items": items}
        return {"blocked": True, "items": [], "message": "News temporarily unavailable"}

    # Shorter cache for news (10 min, matches frontend refresh)
    now = time.time()
    entry = _cache.get("news")
    if entry and now - entry["ts"] < 600:
        return jsonify(entry["data"])
    data = _fetch()
    _cache["news"] = {"data": data, "ts": now}
    return jsonify(data)


# ---------------------------------------------------------------------------
# Transactions — structured feed for the Transactions tab
# ---------------------------------------------------------------------------
TRANSACTION_PRIMARY = {
    "name": "NHL.com",
    "url": "https://www.nhl.com/rss/transactions.xml",
}
TRANSACTION_FALLBACK = {
    "name": "Sportsnet",
    "url": "https://www.sportsnet.ca/hockey/nhl/feed/",
}

TRANSACTION_FILTER_RE = re.compile(
    r"\b(trade[ds]?|trading|acquir(?:e|es|ed)|"
    r"sign(?:s|ed|ing)?|re-?sign(?:s|ed|ing)?|extension|extends?|"
    r"recall(?:s|ed)?|waive(?:s|d)?|waivers?|claim(?:s|ed)?|"
    r"suspen(?:d|ds|ded|sion)|"
    r"injured reserve|\bIR\b|LTIR|"
    r"release(?:s|d)?|loaned?|assigns?|assigned)\b",
    re.IGNORECASE,
)


def _infer_transaction_type(title: str) -> str:
    """Return one of TRADE / SIGNING / RECALL / WAIVER / IR / SUSPENSION / OTHER."""
    t = title.lower()
    # Order matters: more specific keywords first.
    if re.search(r"\bsuspen", t):
        return "SUSPENSION"
    if re.search(r"\b(injured reserve|ltir|\bir\b|placed on (?:long-term )?injured)", t):
        return "IR"
    if re.search(r"\b(waiv|claim)", t):
        return "WAIVER"
    if re.search(r"\brecall", t):
        return "RECALL"
    # sign-and-trade is a trade despite containing "sign"
    if re.search(r"sign-?and-?trade", t):
        return "TRADE"
    # Signing verbs before trade: "deal" is an ambiguous noun (a contract is
    # also a "deal"), so check explicit sign/extension language first.
    if re.search(r"\b(re-?sign|signs?|signed|signing|inks?|extension|extends?)\b", t):
        return "SIGNING"
    if re.search(r"\b(trade[ds]?|trading|acquir)\b", t):
        return "TRADE"
    return "OTHER"


def _fetch_transactions_from(src: dict, *, require_filter: bool) -> list:
    """Fetch one RSS source and return parsed/structured transaction items.

    require_filter=False: feed is already transactions-only (NHL.com); return all items.
    require_filter=True:  feed is general news (Sportsnet); keep only items whose title
                          matches TRANSACTION_FILTER_RE.
    """
    try:
        r = requests.get(src["url"], headers=HEADERS, timeout=8, allow_redirects=True)
    except Exception as e:
        log.info("transactions source %s error: %s", src["name"], e)
        return []
    if r.status_code != 200:
        log.info("transactions source %s returned %s", src["name"], r.status_code)
        return []
    raw = _parse_rss(r.text, src["name"])
    out = []
    for it in raw:
        title = it.get("title") or ""
        if require_filter and not TRANSACTION_FILTER_RE.search(title):
            continue
        out.append({
            "title": title,
            "link": it.get("link"),
            "pub_date": it.get("pub_date"),
            "source": it.get("source"),
            "type": _infer_transaction_type(title),
        })
    return out


def _transactions_pool(limit=40):
    """Return (source_name, [items]) of structured transaction items (each
    carrying an inferred `type`). Shared by /api/transactions and the
    Morning Brief combined feed."""
    items = _fetch_transactions_from(TRANSACTION_PRIMARY, require_filter=False)
    if items:
        return TRANSACTION_PRIMARY["name"], items[:limit]
    items = _fetch_transactions_from(TRANSACTION_FALLBACK, require_filter=True)
    if items:
        return TRANSACTION_FALLBACK["name"], items[:limit]
    return None, []


@app.route("/api/transactions")
def api_transactions():
    """Latest NHL transactions as a structured list.

    Primary source: nhl.com/rss/transactions.xml (transactions-only feed).
    Fallback: Sportsnet NHL RSS, filtered by transaction keywords.
    Returns up to 15 entries. Cached for 10 minutes.
    """
    def _fetch():
        source, items = _transactions_pool(limit=15)
        if items:
            return {"source": source, "items": items}
        return {"blocked": True, "items": [], "message": "Transactions temporarily unavailable"}

    now = time.time()
    entry = _cache.get("transactions")
    if entry and now - entry["ts"] < 600:
        return jsonify(entry["data"])
    data = _fetch()
    _cache["transactions"] = {"data": data, "ts": now}
    return jsonify(data)


# ---------------------------------------------------------------------------
# Morning Brief combined feed — League Pulse / Ink Report / Trade Block
# ---------------------------------------------------------------------------
# Section caps. Pulse tuned to the typical big-news volume of the feed; Ink and
# Trade kept comprehensive within an editorial cap.
PULSE_CAP = 10
INK_CAP = 10
TRADE_CAP = 10


@app.route("/api/morning-brief-feed")
def api_morning_brief_feed():
    """Three editorial sections in one call so all classification lives in the
    data layer (see news_classifier.py):

      league_pulse — biggest league-wide stories, curated by big_news_score.
                     Draws from the news pool AND the transaction pool, so a
                     blockbuster trade / star signing can appear here *and* in
                     its transaction section (duplication is intentional). Also
                     the home for non-TRADE/SIGNING moves (suspensions, etc.).
      ink_report   — executed SIGNING moves only (opinion/video filtered out).
      trade_block  — executed TRADE moves only (opinion/video filtered out).
    """
    def _fetch():
        news_source, news_items = _news_pool(limit=25)
        txn_source, txn_items = _transactions_pool(limit=40)

        if not news_items and not txn_items:
            return {"blocked": True, "league_pulse": [], "ink_report": [],
                    "trade_block": [], "message": "League feed temporarily unavailable"}

        ink = [t for t in txn_items
               if t.get("type") == "SIGNING" and is_real_move(t)][:INK_CAP]
        trade = [t for t in txn_items
                 if t.get("type") == "TRADE" and is_real_move(t)][:TRADE_CAP]

        # Pulse candidates = news pool + every transaction item. rank_big_news
        # de-dupes within Pulse by link; cross-section duplication is allowed.
        pulse = rank_big_news(news_items + txn_items, limit=PULSE_CAP)

        return {
            "league_pulse": pulse,
            "ink_report": ink,
            "trade_block": trade,
            "news_source": news_source,
            "txn_source": txn_source,
        }

    now = time.time()
    entry = _cache.get("mb_feed")
    if entry and now - entry["ts"] < 600:
        return jsonify(entry["data"])
    data = _fetch()
    _cache["mb_feed"] = {"data": data, "ts": now}
    return jsonify(data)


# ---------------------------------------------------------------------------
# Playoff Bracket
# ---------------------------------------------------------------------------
@app.route("/api/playoff-bracket")
def api_playoff_bracket():
    def _fetch():
        try:
            carousel = _build_playoff_carousel()
        except Exception as e:
            log.warning("bracket carousel failed: %s", e)
            return {"empty": True, "message": "Playoffs not yet started", "rounds": []}

        rounds_raw = carousel.get("rounds") or []
        if not rounds_raw:
            return {"empty": True, "rounds": []}

        # Conference inference: in NHL bracketing, series A–D top half is one conference,
        # E–H is the other. Read team conferences from standings to be sure.
        try:
            conf_map = _team_conference_map()
        except Exception:
            conf_map = {}

        rounds_out = []
        for rnd in rounds_raw:
            r_num = rnd.get("roundNumber")
            r_label = rnd.get("roundLabel", f"Round {r_num}")
            series_out = []
            for s in (rnd.get("series") or []):
                top = s.get("topSeed", {}) or {}
                bot = s.get("bottomSeed", {}) or {}
                top_w = top.get("wins", 0)
                bot_w = bot.get("wins", 0)
                needed = s.get("neededToWin", 4)
                complete = (top_w >= needed) or (bot_w >= needed)
                winning_id = s.get("winningTeamId")

                # Conference: use the top seed's conference (both teams in same conf for R1–3)
                top_ab = top.get("abbrev", "")
                bot_ab = bot.get("abbrev", "")
                series_conf = conf_map.get(top_ab) or conf_map.get(bot_ab) or ""
                if r_num == 4:
                    series_conf = "Final"

                series_out.append({
                    "series_letter": s.get("seriesLetter"),
                    "round": r_num,
                    "conference": series_conf,
                    "top_seed": {
                        "id": top.get("id"),
                        "abbrev": top_ab,
                        "logo": top.get("logo"),
                        "wins": top_w,
                        "won": complete and winning_id == top.get("id"),
                    } if top_ab else None,
                    "bottom_seed": {
                        "id": bot.get("id"),
                        "abbrev": bot_ab,
                        "logo": bot.get("logo"),
                        "wins": bot_w,
                        "won": complete and winning_id == bot.get("id"),
                    } if bot_ab else None,
                    "needed_to_win": needed,
                    "complete": complete,
                    "started": (top_ab and bot_ab and (top_w > 0 or bot_w > 0 or complete)),
                })
            rounds_out.append({
                "round_number": r_num,
                "label": r_label,
                "series": series_out,
            })

        # Next games — today's and tomorrow's playoff (gameType=3) games
        next_games = []
        try:
            today = date.today().strftime("%Y-%m-%d")
            tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
            for day in (today, tomorrow):
                try:
                    sched = get_json(f"{NHL_API}/schedule/{day}")
                    for d in (sched.get("gameWeek") or []):
                        if d.get("date") != day:
                            continue
                        for g in (d.get("games") or []):
                            if g.get("gameType") != 3:  # playoff
                                continue
                            state = g.get("gameState", "")
                            if state in ("OFF", "FINAL"):
                                continue
                            away = g.get("awayTeam", {})
                            home = g.get("homeTeam", {})
                            next_games.append({
                                "date": day,
                                "start_time_utc": g.get("startTimeUTC"),
                                "away": {
                                    "abbrev": away.get("abbrev"),
                                    "logo": away.get("logo"),
                                },
                                "home": {
                                    "abbrev": home.get("abbrev"),
                                    "logo": home.get("logo"),
                                },
                            })
                except Exception as e:
                    log.info("next playoff games for %s failed: %s", day, e)
        except Exception as e:
            log.warning("next playoff games block failed: %s", e)

        return {
            "current_round": carousel.get("currentRound"),
            "rounds": rounds_out,
            "next_games": next_games,
        }

    try:
        return jsonify(cached("playoff_bracket", _fetch))
    except Exception as e:
        log.error("api_playoff_bracket error: %s", e)
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Skater on-ice leaderboards (CF / xGF / HDCF / RAPM-proxy)
# ---------------------------------------------------------------------------
MIN_TOI_SECONDS_500 = 500 * 60  # 500 minutes in seconds


def _mp_skaters_5on5():
    df = _mp_skaters()
    return df[df["situation"] == "5on5"].copy()


@app.route("/api/skater-onice-leaders")
def api_skater_onice_leaders():
    """Top 25 skaters by chosen on-ice metric at 5v5 with min 500 min TOI."""
    metric = (request.args.get("metric") or "cf").lower()
    position = (request.args.get("position") or "all").upper()

    def _fetch():
        df = _mp_skaters_5on5()
        df = _position_filter_df(df, position)
        # Coerce all needed columns to numeric
        cols = [
            "onIce_corsiPercentage", "onIce_xGoalsPercentage",
            "OnIce_F_highDangerShots", "OnIce_A_highDangerShots",
            "OnIce_F_xGoals", "OnIce_A_xGoals",
            "icetime", "games_played",
        ]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df = df[df["icetime"] >= MIN_TOI_SECONDS_500].copy()

        # HDCF%
        hd_total = df["OnIce_F_highDangerShots"] + df["OnIce_A_highDangerShots"]
        df["hdcf_pct"] = (df["OnIce_F_highDangerShots"] / hd_total.replace(0, 1)) * 100

        # RAPM proxy: on-ice xG differential per 60 minutes
        df["onice_xg_diff_60"] = (
            (df["OnIce_F_xGoals"] - df["OnIce_A_xGoals"]) / (df["icetime"] / 3600)
        )

        df["cf_pct"] = df["onIce_corsiPercentage"] * 100
        # On-ice xGF% sourced from my self-generated table (intersects my
        # xG-scored shots with skater shifts); MoneyPuck on-ice xGF% used as
        # fallback for players not in the self-gen table.
        my_onice = _load_onice_xgf()
        def _xgf(r):
            pid = int(r.get("playerId", 0))
            if pid in my_onice:
                return float(my_onice[pid]["xgf_pct"])
            return float(r["onIce_xGoalsPercentage"]) * 100
        df["xgf_pct"] = df.apply(_xgf, axis=1)
        df["toi_min"] = (df["icetime"] / 60).round(0)

        sort_col_map = {
            "cf": "cf_pct",
            "xgf": "xgf_pct",
            "hdcf": "hdcf_pct",
            "rapm": "onice_xg_diff_60",
        }
        sort_col = sort_col_map.get(metric, "cf_pct")
        df = df.sort_values(sort_col, ascending=False).head(25)

        return {
            "source": "self_generated_onice_xgf+moneypuck_counts",
            "metric": metric,
            "min_toi_min": 500,
            "players": [
                {
                    "name": r["name"],
                    "team": r["team"],
                    "position": r["position"],
                    "games": int(r["games_played"]),
                    "toi_min": int(r["toi_min"]),
                    "cf_pct": round(float(r["cf_pct"]), 2),
                    "xgf_pct": round(float(r["xgf_pct"]), 2),
                    "hdcf_pct": round(float(r["hdcf_pct"]), 2),
                    "onice_xg_diff_60": round(float(r["onice_xg_diff_60"]), 3),
                }
                for _, r in df.iterrows()
            ],
        }

    return jsonify(cached(f"skater_onice_{metric}_{position}", _fetch))


# ---------------------------------------------------------------------------
# Team special teams (PP / PK) from MoneyPuck
# ---------------------------------------------------------------------------
@app.route("/api/team-special-teams")
def api_team_special_teams():
    """All 32 teams ranked by PP / PK metrics derived from MoneyPuck team CSV."""
    def _fetch():
        df = _mp_teams()
        for c in ["xGoalsFor", "goalsFor", "xGoalsAgainst", "goalsAgainst",
                  "iceTime", "games_played"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        pp = df[df["situation"] == "5on4"].copy()
        pk = df[df["situation"] == "4on5"].copy()

        teams = {}
        for _, r in pp.iterrows():
            ice60 = float(r["iceTime"]) / 3600 if float(r["iceTime"]) > 0 else 0.01
            # PP% estimate: avg PP duration ~90s; opportunities = iceTime / 90
            opps_est = float(r["iceTime"]) / 90 if float(r["iceTime"]) > 0 else 1
            teams[r["team"]] = {
                "team": r["team"],
                "pp_xgf_60": round(float(r["xGoalsFor"]) / ice60, 2),
                "pp_g_60":   round(float(r["goalsFor"]) / ice60, 2),
                "pp_pct":    round(float(r["goalsFor"]) / opps_est * 100, 1),
                "pp_toi_min": int(float(r["iceTime"]) / 60),
            }
        for _, r in pk.iterrows():
            ice60 = float(r["iceTime"]) / 3600 if float(r["iceTime"]) > 0 else 0.01
            opps_est = float(r["iceTime"]) / 90 if float(r["iceTime"]) > 0 else 1
            teams.setdefault(r["team"], {"team": r["team"]})
            teams[r["team"]].update({
                "pk_xga_60": round(float(r["xGoalsAgainst"]) / ice60, 2),
                "pk_ga_60":  round(float(r["goalsAgainst"]) / ice60, 2),
                "pk_pct":    round(100 - (float(r["goalsAgainst"]) / opps_est * 100), 1),
                "pk_toi_min": int(float(r["iceTime"]) / 60),
            })

        return {"source": "moneypuck", "teams": list(teams.values())}

    return jsonify(cached("team_special_teams", _fetch))


# ---------------------------------------------------------------------------
# Team style — pace / SH% / SV% / PDO
# ---------------------------------------------------------------------------
@app.route("/api/team-style")
def api_team_style():
    """Pace (SA/60), SH%, SV%, PDO for all 32 teams (5v5)."""
    def _fetch():
        df = _mp_teams()
        for c in ["shotsOnGoalFor", "shotsOnGoalAgainst", "shotAttemptsFor",
                  "goalsFor", "goalsAgainst", "iceTime", "games_played"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df5 = df[df["situation"] == "5on5"].copy()
        out = []
        for _, r in df5.iterrows():
            ice60 = float(r["iceTime"]) / 3600 if float(r["iceTime"]) > 0 else 0.01
            sa_for = float(r["shotAttemptsFor"])
            sog_for = float(r["shotsOnGoalFor"])
            sog_against = float(r["shotsOnGoalAgainst"])
            gf = float(r["goalsFor"])
            ga = float(r["goalsAgainst"])
            sh_pct = (gf / sog_for * 100) if sog_for > 0 else 0
            sv_pct = ((sog_against - ga) / sog_against * 100) if sog_against > 0 else 0
            pdo = sh_pct + sv_pct
            out.append({
                "team": r["team"],
                "gp": int(r["games_played"]),
                "pace_sa_60": round(sa_for / ice60, 2),
                "sh_pct": round(sh_pct, 2),
                "sv_pct": round(sv_pct, 2),
                "pdo": round(pdo, 2),
            })
        out.sort(key=lambda t: t["pdo"], reverse=True)
        return {"source": "moneypuck", "teams": out}

    return jsonify(cached("team_style_pdo", _fetch))


# ---------------------------------------------------------------------------
# Team head-to-head (current season schedule proxy)
# ---------------------------------------------------------------------------
@app.route("/api/team-head-to-head")
def api_team_head_to_head():
    """Current-season head-to-head record between two teams.

    Used as a proxy for all-time playoff series history (no free API for that).
    """
    t1 = (request.args.get("team1") or "").upper()
    t2 = (request.args.get("team2") or "").upper()
    if not t1 or not t2 or t1 == t2:
        return jsonify({"error": "Provide two distinct team abbrevs as ?team1=&team2="}), 400

    def _fetch():
        try:
            sched1 = get_json(f"{NHL_API}/club-schedule-season/{t1}/{current_season()}")
        except Exception as e:
            log.warning("h2h schedule %s failed: %s", t1, e)
            return {"error": f"could not load {t1} schedule"}
        games = sched1.get("games", []) or []
        meetings = []
        t1_wins = t2_wins = 0
        gf_t1 = gf_t2 = 0
        for g in games:
            if g.get("gameState") != "OFF":
                continue
            away = g.get("awayTeam", {})
            home = g.get("homeTeam", {})
            if t2 not in (away.get("abbrev"), home.get("abbrev")):
                continue
            t1_is_home = home.get("abbrev") == t1
            t1_score = (home if t1_is_home else away).get("score", 0)
            t2_score = (away if t1_is_home else home).get("score", 0)
            gf_t1 += t1_score
            gf_t2 += t2_score
            last_period = (g.get("gameOutcome") or {}).get("lastPeriodType", "REG")
            if t1_score > t2_score:
                t1_wins += 1
                result = f"{t1} W"
            else:
                t2_wins += 1
                result = f"{t2} W"
            meetings.append({
                "date": g.get("gameDate"),
                "game_type": g.get("gameType"),
                "t1_score": t1_score,
                "t2_score": t2_score,
                "result": result,
                "ot_so": last_period if last_period != "REG" else None,
                "venue": "home" if t1_is_home else "away",
            })

        return {
            "team1": t1,
            "team2": t2,
            "season": current_season(),
            "note": "Showing current-season head-to-head; all-time playoff history would require an external source.",
            "t1_wins": t1_wins,
            "t2_wins": t2_wins,
            "gf_t1": gf_t1,
            "gf_t2": gf_t2,
            "meetings": meetings,
        }

    return jsonify(cached(f"h2h_{t1}_{t2}", _fetch))


# ---------------------------------------------------------------------------
# Goalies extended (SV% / HDSV% / MDSV% / workload / QS% / GSAX/60)
# ---------------------------------------------------------------------------
def _quality_starts_for_goalie(player_id: int, season: str | None = None) -> dict:
    """Fetch game log + compute Quality Start % per Brooks/MeisterStats convention:
    QS = SV% >= .917 OR (GA <= 2 AND SA >= 1).

    Returns dict with starts, qs, qs_pct.
    """
    season = season or current_season()
    def _fetch():
        try:
            data = get_json(f"{NHL_API}/player/{player_id}/game-log/{season}/2")
        except Exception as e:
            log.info("game-log fetch failed pid=%s: %s", player_id, e)
            return {"starts": 0, "qs": 0, "qs_pct": None}
        games = data.get("gameLog", []) or []
        starts = 0
        qs = 0
        for g in games:
            if not g.get("gamesStarted"):
                continue
            sa = g.get("shotsAgainst", 0) or 0
            ga = g.get("goalsAgainst", 0) or 0
            svp = g.get("savePctg")
            try:
                svp = float(svp) if svp is not None else None
            except (TypeError, ValueError):
                svp = None
            if sa < 1:
                continue
            starts += 1
            is_qs = (svp is not None and svp >= 0.917) or (ga <= 2 and sa >= 1)
            if is_qs:
                qs += 1
        return {
            "starts": starts,
            "qs": qs,
            "qs_pct": round(qs / starts * 100, 1) if starts else None,
        }
    return cached(f"qs_{player_id}_{season}", _fetch)


@app.route("/api/goalies-extended")
def api_goalies_extended():
    """Full goalie table: SV% / HDSV% / MDSV% / starts / TOI / GAA / QS% / GSAX-per-60.

    First-load slowness was a sequential loop of NHL game-log fetches (one
    per qualified goalie, ~75-80 calls × ~200ms each = 15-20s). Parallelised
    via ThreadPoolExecutor with 12 workers — first load now ~2-3s, cached
    runs are instant. Each per-goalie QS result is independently cached, so
    even on a cold global cache the per-player caches survive across calls."""
    def _fetch():
        df = _mp_goalies()
        cols = ["xGoals", "goals", "icetime", "games_played", "ongoal", "playerId",
                "lowDangerShots", "mediumDangerShots", "highDangerShots",
                "lowDangerGoals", "mediumDangerGoals", "highDangerGoals"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # Qualified = situation "all" + games played >= 10
        df_all = df[(df["situation"] == "all") & (df["games_played"] >= 10)].copy()

        # Pre-fetch all per-goalie QS results in parallel (the slow step).
        from concurrent.futures import ThreadPoolExecutor
        pids = [int(r["playerId"]) for _, r in df_all.iterrows()]
        qs_by_pid: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=12) as ex:
            for pid, res in zip(pids, ex.map(_quality_starts_for_goalie, pids)):
                qs_by_pid[pid] = res

        # Self-generated GSAX lookup (xG against from my model − actual GA)
        my_gsax = _load_goalie_gsax()

        out = []
        for _, r in df_all.iterrows():
            shots = float(r["ongoal"]) or 0
            goals = float(r["goals"]) or 0
            sv_pct = ((shots - goals) / shots * 100) if shots > 0 else 0

            hd_shots = float(r["highDangerShots"])
            hd_goals = float(r["highDangerGoals"])
            md_shots = float(r["mediumDangerShots"])
            md_goals = float(r["mediumDangerGoals"])
            hdsv_pct = ((hd_shots - hd_goals) / hd_shots * 100) if hd_shots > 0 else 0
            mdsv_pct = ((md_shots - md_goals) / md_shots * 100) if md_shots > 0 else 0

            toi_min = float(r["icetime"]) / 60
            # Self-generated GSAX (my xG model applied to shots the goalie
            # was on the ice for, minus actual goals against)
            pid = int(r["playerId"])
            my_g = my_gsax.get(pid)
            gsax = float(my_g["gsax"]) if my_g else (float(r["xGoals"]) - goals)
            gsax_60 = (gsax / (float(r["icetime"]) / 3600)) if float(r["icetime"]) > 0 else 0
            gp = int(r["games_played"])
            gaa = (goals / (toi_min / 60)) if toi_min > 0 else 0

            qs = qs_by_pid.get(pid, {"starts": 0, "qs": 0, "qs_pct": None})

            out.append({
                "name": r["name"],
                "team": r["team"],
                "playerId": pid,
                "games": gp,
                "starts": qs["starts"],
                "toi_min": int(round(toi_min)),
                "sv_pct": round(sv_pct, 2),
                "hdsv_pct": round(hdsv_pct, 2),
                "mdsv_pct": round(mdsv_pct, 2),
                "gsax": round(gsax, 2),
                "gsax_60": round(gsax_60, 3),
                "gaa": round(gaa, 2),
                "qs": qs["qs"],
                "qs_pct": qs["qs_pct"],
            })
        out.sort(key=lambda g: g["gsax"], reverse=True)
        return {"source": "self_generated_gsax+moneypuck_counts", "goalies": out}

    return jsonify(cached("goalies_extended", _fetch))


# ---------------------------------------------------------------------------
# Playoff stat leaders (gameType=3)
# ---------------------------------------------------------------------------
@app.route("/api/playoff-stats-leaders")
def api_playoff_stats_leaders():
    def _fetch():
        result = {}
        for cat in ["points", "goals", "assists", "plusMinus"]:
            try:
                raw = get_json(
                    f"{NHL_API}/skater-stats-leaders/{current_season()}/3",
                    params={"categories": cat, "limit": 10},
                )
                result[cat] = [_format_player_leader(p) for p in raw.get(cat, [])]
            except Exception as e:
                log.info("playoff leaders %s failed: %s", cat, e)
                result[cat] = []
        try:
            goalie_raw = get_json(
                f"{NHL_API}/goalie-stats-leaders/{current_season()}/3",
                params={"categories": "wins", "limit": 10},
            )
            result["goalie_wins"] = [_format_player_leader(p) for p in goalie_raw.get("wins", [])]
        except Exception as e:
            log.info("playoff goalie leaders failed: %s", e)
            result["goalie_wins"] = []
        return result

    return jsonify(cached("playoff_stats_leaders", _fetch))


# ---------------------------------------------------------------------------
# Playoff team analytics (MoneyPuck playoffs CSV)
# ---------------------------------------------------------------------------
MP_PLAYOFFS_TEAMS_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/playoffs/teams.csv"


@app.route("/api/playoff-team-analytics")
def api_playoff_team_analytics():
    def _fetch():
        try:
            df = cached("mp_playoff_teams_csv", lambda: get_csv(MP_PLAYOFFS_TEAMS_URL))
        except Exception as e:
            log.warning("MoneyPuck playoffs CSV failed, falling back to regular season: %s", e)
            df = _mp_teams()
            fallback_note = "MoneyPuck playoff CSV unavailable — showing regular-season 5v5 stats as proxy."
        else:
            fallback_note = None

        for c in ["xGoalsPercentage", "corsiPercentage", "highDangerShotsFor",
                  "highDangerShotsAgainst", "games_played"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df5 = df[df["situation"] == "5on5"].copy() if "situation" in df.columns else df

        # xGF% uses my self-generated regular-season xG aggregation as a
        # baseline (playoff-specific aggregation would require filtering
        # shots to game-type 3; left for a future build-script extension).
        my_team_xgf = _load_team_xgf()
        out = []
        for _, r in df5.iterrows():
            tm = str(r["team"])
            hd_for = float(r.get("highDangerShotsFor", 0))
            hd_ag = float(r.get("highDangerShotsAgainst", 0))
            hdcf_pct = (hd_for / (hd_for + hd_ag) * 100) if (hd_for + hd_ag) > 0 else 50.0
            my_x = my_team_xgf.get(tm)
            xgf_pct = (float(my_x["xgf_pct"]) if my_x
                        else round(float(r.get("xGoalsPercentage", 0)) * 100, 1))
            out.append({
                "team": tm,
                "gp": int(r.get("games_played", 0)),
                "xgf_pct": round(xgf_pct, 1),
                "cf_pct": round(float(r.get("corsiPercentage", 0)) * 100, 1),
                "hdcf_pct": round(hdcf_pct, 1),
            })
        out.sort(key=lambda t: t["xgf_pct"], reverse=True)
        return {
            "source": "self_generated_xgf_regular_season+moneypuck_playoff_counts",
            "fallback_note": fallback_note or "xGF% is the regular-season value from my own xG model; playoff CF% and HDCF% are from MoneyPuck playoff counts.",
            "teams": out,
        }

    return jsonify(cached("playoff_team_analytics", _fetch))


# ---------------------------------------------------------------------------
# Contract Value — capwages snapshot → PuckPedia → manual JSON → MoneyPuck merge
# ---------------------------------------------------------------------------
PUCKPEDIA_URL = "https://puckpedia.com/contracts"
MANUAL_CONTRACTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "contracts_manual.json"
)
# League-wide snapshot written by data/fetch_contracts.py on a schedule.
CURRENT_CONTRACTS_PATH = os.path.join(
    os.path.dirname(__file__), "data", "contracts_current.json"
)


def _norm_name(s: str) -> str:
    """Normalize player name for matching across sources."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# MoneyPuck has an encoding bug where names with diacritics drop the accented
# character entirely (Stützle → Sttzle). Map canonical normalized name → the
# normalized form MoneyPuck actually serves.
_MP_NAME_ALIASES = {
    "timstutzle": "timsttzle",
}


# --- Self-generated dashboard analytics (2026-06-07 single-season refactor) -
# Players-section analytical metrics reflect the 2025-26 regular season only.
# The simulator still uses multi-season talent estimates (composite_ratings_sim.csv).
#
#   Players-section paths (single-season 25-26):
#     GAR        → composite_ratings_single_season.csv (composite_war column)
#     RAPM       → rapm_single_season.csv
#     xGAR       → skater_xgar_self_generated.csv
#     Game Score → skater_game_score_self_generated.csv
#
#   Common paths (single-season facts about 25-26):
#     Goalie GSAX     → goalie_gsax_self_generated.csv
#     Team xGF%       → team_xgf_self_generated.csv
#     Skater on-ice   → skater_onice_xgf_self_generated.csv
#
# Build these with:
#     python3 model/build_self_generated_stats.py
#     python3 model/train_rapm_single_season.py
#     python3 model/build_composite_single_season.py --save
SKATER_XGAR_PATH = os.path.join(os.path.dirname(__file__), "model",
                                  "skater_xgar_self_generated.csv")
GOALIE_GSAX_PATH = os.path.join(os.path.dirname(__file__), "model",
                                  "goalie_gsax_self_generated.csv")
TEAM_XGF_PATH = os.path.join(os.path.dirname(__file__), "model",
                              "team_xgf_self_generated.csv")
ONICE_XGF_PATH = os.path.join(os.path.dirname(__file__), "model",
                               "skater_onice_xgf_self_generated.csv")
GAME_SCORE_PATH = os.path.join(os.path.dirname(__file__), "model",
                                "skater_game_score_self_generated.csv")
# composite_war used by the Players section now points at the single-season
# build. The simulator still reads composite_ratings_sim.csv directly via
# model/build_team_strength.py and model/simulate_season.py — both unchanged.
COMPOSITE_PLAYERS_PATH = os.path.join(os.path.dirname(__file__), "model",
                                       "composite_ratings_single_season.csv")


def _load_skater_xgar() -> dict[int, dict]:
    """player_id → {xgar, ixg, goals, goals_above_expected, n_shots} from
    model/skater_xgar_self_generated.csv. Cached."""
    def _fn():
        if not os.path.exists(SKATER_XGAR_PATH):
            return {}
        df = pd.read_csv(SKATER_XGAR_PATH)
        return {int(r.player_id): {
            "xgar": float(r.xgar),
            "ixg": float(r.ixg),
            "goals": int(r.goals),
            "goals_above_expected": float(r.goals_above_expected),
            "n_shots": int(r.n_shots),
        } for r in df.itertuples(index=False)}
    return cached("skater_xgar_lookup", _fn)


def _load_composite_war() -> dict[int, dict]:
    """player_id → {composite_war, composite_rating, ...components} from
    model/composite_ratings_single_season.csv (the 25-26-only build used by
    the Players section). The simulator does NOT call this helper — it reads
    composite_ratings_sim.csv (multi-season) directly via
    build_team_strength.load_composite_lookup()."""
    def _fn():
        if not os.path.exists(COMPOSITE_PLAYERS_PATH):
            return {}
        df = pd.read_csv(COMPOSITE_PLAYERS_PATH)
        return {int(r.player_id): {
            "composite_war": float(r.composite_war),
            "composite_rating": float(r.composite_rating),
            "rapm_component": float(r.rapm_component),
            "individual_component": float(r.individual_component),
            "playmaking_component": float(r.playmaking_component),
            "power_play_component": float(r.power_play_component),
            "penalty_kill_component": float(r.penalty_kill_component),
            "sample_size_flag": str(r.sample_size_flag),
        } for r in df.itertuples(index=False)}
    return cached("composite_war_lookup", _fn)


def _load_goalie_gsax() -> dict[int, dict]:
    """player_id → {gsax, gsax_war, xg_against, goals_against, shots_faced}."""
    def _fn():
        if not os.path.exists(GOALIE_GSAX_PATH):
            return {}
        df = pd.read_csv(GOALIE_GSAX_PATH)
        return {int(r.player_id): {
            "gsax": float(r.gsax),
            "gsax_war": float(r.gsax_war),
            "xg_against": float(r.xg_against),
            "goals_against": int(r.goals_against),
            "shots_faced": int(r.shots_faced),
        } for r in df.itertuples(index=False)}
    return cached("goalie_gsax_lookup", _fn)


def _load_team_xgf() -> dict[str, dict]:
    """team_abbrev → {xgf, xga, xgf_pct} from my self-generated team table."""
    def _fn():
        if not os.path.exists(TEAM_XGF_PATH):
            return {}
        df = pd.read_csv(TEAM_XGF_PATH)
        df = df[df["team"] != "?"]   # drop the unmapped-game-id orphan
        return {str(r.team): {
            "xgf": float(r.xgf),
            "xga": float(r.xga),
            "xgf_pct": float(r.xgf_pct),
        } for r in df.itertuples(index=False)}
    return cached("team_xgf_lookup", _fn)


def _load_onice_xgf() -> dict[int, dict]:
    """player_id → {xgf, xga, xgf_pct, rel_x, shifts, team}."""
    def _fn():
        if not os.path.exists(ONICE_XGF_PATH):
            return {}
        df = pd.read_csv(ONICE_XGF_PATH)
        # If a player has multiple stints (traded), use highest-shifts row.
        df = df.sort_values("shifts", ascending=False)
        df = df.drop_duplicates("player_id", keep="first")
        return {int(r.player_id): {
            "xgf": float(r.xgf),
            "xga": float(r.xga),
            "xgf_pct": float(r.xgf_pct),
            "rel_x": float(r.rel_x),
            "shifts": int(r.shifts),
            "team": str(r.team),
        } for r in df.itertuples(index=False)}
    return cached("onice_xgf_lookup", _fn)


def _load_game_score() -> dict[int, dict]:
    """player_id → {game_score, game_score_per_game, ...}."""
    def _fn():
        if not os.path.exists(GAME_SCORE_PATH):
            return {}
        df = pd.read_csv(GAME_SCORE_PATH)
        return {int(r.player_id): {
            "game_score": float(r.game_score),
            "game_score_per_game": float(r.game_score_per_game),
        } for r in df.itertuples(index=False)}
    return cached("game_score_lookup", _fn)


QOC_QOT_PATH = os.path.join(os.path.dirname(__file__), "model",
                              "skater_qoc_qot_single_season.csv")


def _load_qoc_qot() -> dict[int, dict]:
    """player_id → {qoc, qot, qoc_toi_secs, qot_toi_secs} from my own
    shift-overlap-weighted GAR aggregation. See model/build_qoc_qot.py."""
    def _fn():
        if not os.path.exists(QOC_QOT_PATH):
            return {}
        df = pd.read_csv(QOC_QOT_PATH)
        return {int(r.player_id): {
            "qoc": float(r.qoc) if pd.notna(r.qoc) else None,
            "qot": float(r.qot) if pd.notna(r.qot) else None,
            "qoc_toi_secs": int(r.qoc_toi_secs),
            "qot_toi_secs": int(r.qot_toi_secs),
        } for r in df.itertuples(index=False)}
    return cached("qoc_qot_lookup", _fn)


def _puckpedia_contracts() -> list | None:
    """Attempt to scrape PuckPedia top contracts. Cloudflare-protected — typically fails."""
    try:
        r = requests.get(PUCKPEDIA_URL, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            log.info("PuckPedia returned %s — falling through", r.status_code)
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        tables = soup.find_all("table")
        if not tables:
            log.info("PuckPedia: no tables in page (Cloudflare challenge or SPA shell)")
            return None
        # If a table is found we'd parse it here. PuckPedia is JS-rendered, so this
        # rarely returns useful HTML server-side. Returning None keeps fallback honest.
        return None
    except Exception as e:
        log.warning("PuckPedia scrape failed: %s", e)
        return None


def _manual_contracts() -> dict | None:
    """Load the manually-curated top contracts file."""
    try:
        with open(MANUAL_CONTRACTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        contracts = data.get("contracts", [])
        if not contracts:
            return None
        return {"source": "manual", "meta": data.get("_meta", {}), "contracts": contracts}
    except Exception as e:
        log.warning("Manual contracts load failed: %s", e)
        return None


def _current_contracts() -> dict | None:
    """Load the league-wide contract snapshot from data/contracts_current.json.

    Written on a schedule by data/fetch_contracts.py (capwages source). Each row
    already carries `age` and `playerId` resolved at scrape time, so the request
    path does not hit the NHL API to enrich ~1,400 players."""
    try:
        with open(CURRENT_CONTRACTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        contracts = data.get("contracts", [])
        if not contracts:
            return None
        return {"source": "capwages", "meta": data.get("_meta", {}), "contracts": contracts}
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning("Current contracts load failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Scheduled contract refresh — keeps contracts_current.json current league-wide
# ---------------------------------------------------------------------------
# A daemon thread runs data/fetch_contracts.py (the standalone scraper) once a
# day. It works the same whether the app is started with `python app.py` or
# under gunicorn with multiple workers: a single-flight lock file ensures only
# one process actually fetches, and the snapshot write is atomic, so requests
# always read a complete file. Set DISABLE_CONTRACT_REFRESH=1 to turn it off
# (e.g. if you drive the refresh from external cron instead).
_FETCH_SCRIPT = os.path.join(os.path.dirname(__file__), "data", "fetch_contracts.py")
_CONTRACT_LOCK = os.path.join(os.path.dirname(__file__), "data", ".contracts_refresh.lock")
CONTRACT_MAX_AGE = 24 * 3600      # refresh when snapshot is older than a day
_CONTRACT_LOCK_STALE = 30 * 60    # a lock older than this is treated as abandoned


def _contract_snapshot_age() -> float | None:
    try:
        return time.time() - os.path.getmtime(CURRENT_CONTRACTS_PATH)
    except OSError:
        return None  # missing snapshot


def _run_contract_refresh() -> None:
    """Run the scraper once, guarded by a single-flight lock. Never raises."""
    try:
        if os.path.exists(_CONTRACT_LOCK):
            if time.time() - os.path.getmtime(_CONTRACT_LOCK) < _CONTRACT_LOCK_STALE:
                return  # another process is already refreshing
            os.remove(_CONTRACT_LOCK)  # stale lock from a crashed run
        fd = os.open(_CONTRACT_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return  # lost the race to another worker
    except OSError as e:
        log.warning("contract refresh lock error: %s", e)
        return

    try:
        log.info("Refreshing contract snapshot via %s", _FETCH_SCRIPT)
        subprocess.run([sys.executable, _FETCH_SCRIPT], timeout=600, check=False)
        _cache.pop("contract_values", None)  # force reload on next request
    except Exception as e:
        log.warning("contract refresh failed: %s", e)
    finally:
        try:
            os.remove(_CONTRACT_LOCK)
        except OSError:
            pass


def _contract_refresher_loop() -> None:
    while True:
        try:
            age = _contract_snapshot_age()
            if age is None or age >= CONTRACT_MAX_AGE:
                _run_contract_refresh()
        except Exception as e:
            log.warning("contract refresher loop error: %s", e)
        time.sleep(3600)  # re-check hourly


def _start_contract_refresher() -> None:
    if os.environ.get("DISABLE_CONTRACT_REFRESH"):
        log.info("Contract refresher disabled via DISABLE_CONTRACT_REFRESH")
        return
    threading.Thread(
        target=_contract_refresher_loop, daemon=True, name="contract-refresher"
    ).start()


# ---------------------------------------------------------------------------
# Contract value model — Off/Def split, PP/PK GAR, contract type, surplus value
# ---------------------------------------------------------------------------
MARKET_VALUE_PER_GAR = 1.85   # $1.85M per GAR unit (open-market reference price)
ELC_CAP_HIT_THRESHOLD = 1.1   # ELC contracts have cap hit at or below this ($M)
BRIDGE_CAP_HIT_THRESHOLD = 5.0  # bridge contracts: post-ELC, under $5M


def _classify_contract(c: dict, age: int | None) -> str:
    """Return 'ELC' | 'Bridge' | 'Market Rate' | 'Veteran'."""
    cap = float(c["cap_hit"])
    if cap <= ELC_CAP_HIT_THRESHOLD and (age is None or age <= 23):
        return "ELC"
    if age is not None and age >= 32:
        return "Veteran"
    if cap < BRIDGE_CAP_HIT_THRESHOLD and (age is None or age <= 27):
        return "Bridge"
    return "Market Rate"


def _age_curve_projection(gar: float, position: str, current_age: int, year_offset: int) -> float:
    """Project GAR `year_offset` years forward using a simple aging curve.
    Forwards decline 8%/yr after age 30; defensemen decline 8%/yr after age 33."""
    if gar is None or current_age is None:
        return gar
    is_d = position == "D"
    decline_start = 33 if is_d else 30
    projected = float(gar)
    for y in range(1, year_offset + 1):
        target_age = current_age + y
        if target_age >= decline_start:
            projected *= 0.92
    return round(projected, 2)


def _player_age_from_landing(player_id: int) -> int | None:
    """Fetch landing endpoint, return current age or None."""
    landing = _player_landing(player_id)
    if not landing or not landing.get("birthDate"):
        return None
    try:
        from datetime import datetime as _dt
        bd = _dt.fromisoformat(landing["birthDate"]).date()
        today = date.today()
        return today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
    except Exception:
        return None


def _resolve_player_id(name: str, team: str) -> int | None:
    """Look up a player ID via the search API. Cached 24h."""
    def _fetch():
        try:
            r = requests.get(NHL_SEARCH, params={"culture": "en-us", "limit": 8, "q": name, "active": "true"},
                             headers=HEADERS, timeout=6)
            r.raise_for_status()
            data = r.json() or []
        except Exception:
            return None
        for hit in data:
            hit_team = hit.get("teamAbbrev") or hit.get("lastTeamAbbrev") or ""
            if str(hit_team).upper() == team.upper():
                return int(hit.get("playerId") or hit.get("id") or 0) or None
        # No team-match — return first result as best-effort fallback
        if data:
            try:
                return int(data[0].get("playerId") or data[0].get("id") or 0) or None
            except Exception:
                return None
        return None
    return cached_ttl(f"resolve:{_norm_name(name)}:{team}", 86400, _fetch)


def _bulk_fetch_ages(contracts: list[dict]) -> dict:
    """Resolve playerId for each contract, fetch landing in parallel, return {name -> age}."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    name_to_age = {}
    name_to_pid = {}

    def _resolve_and_age(c):
        # Snapshot rows (capwages) already carry playerId + age — use them and
        # skip the NHL API entirely. Only fall back to live resolution for
        # rows that lack them (e.g. the small manual/PuckPedia fallback set).
        pid = c.get("playerId") or _resolve_player_id(c["name"], c["team"])
        if not pid:
            return c["name"], None, c.get("age")
        age = c.get("age")
        if age is None:
            age = _player_age_from_landing(pid)
        return c["name"], pid, age

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_resolve_and_age, c) for c in contracts]
        for fut in as_completed(futures):
            try:
                name, pid, age = fut.result()
                name_to_age[name] = age
                name_to_pid[name] = pid
            except Exception as e:
                log.info("age resolve failed: %s", e)

    return {"ages": name_to_age, "playerIds": name_to_pid}


def _build_pp_pk_gar_index() -> dict:
    """Build {player_id -> {pp_gar, pk_gar}} from my composite-rating model.

    The composite already splits special-teams impact into `power_play_component`
    and `penalty_kill_component` (z-scored across the qualified pool, with a
    realistic ~0.5σ scaling to keep them in the 0-5 range like the rest of the
    GAR components). Returned dict is keyed by `player_id` (the swap from
    `norm_name` is intentional — name normalisation across sources was a
    constant source of mismatch)."""
    comp = _load_composite_war()
    out = {}
    for pid, c in comp.items():
        out[pid] = {
            "pp_gar": round(float(c.get("power_play_component", 0.0)), 2),
            "pk_gar": round(float(c.get("penalty_kill_component", 0.0)), 2),
        }
    return out


def _split_off_def_gar(total_gar: float, position: str, ixg_60: float, all_ixg_60: list[float]) -> tuple[float, float]:
    """Approximate Off/Def split using ixG/60 as the offensive contribution signal.
    Caps at position-realistic baselines."""
    if total_gar is None:
        return None, None
    # Normalize this player's ixG/60 against league-wide max for their role
    max_ixg = max(all_ixg_60 or [0]) or 1.0
    raw_share = (ixg_60 or 0) / max_ixg
    is_d = position == "D"
    # Position-realistic baselines: forwards skew offensive, D skew defensive
    if is_d:
        off_share = max(0.20, min(0.55, 0.25 + raw_share * 0.45))
    else:
        off_share = max(0.45, min(0.85, 0.50 + raw_share * 0.50))
    # If total GAR is negative, the split semantics flip — keep proportionality
    off_gar = total_gar * off_share
    def_gar = total_gar - off_gar
    return round(off_gar, 2), round(def_gar, 2)


def _merge_contracts_with_moneypuck(contracts: list[dict]) -> list[dict]:
    """
    Enrich contract rows with full analytical model: Total/Off/Def/PP/PK GAR, xGAR,
    Sustainability Score, Surplus Value, contract type, age, age-curve projection.
    """
    skaters = _mp_skaters()
    skaters_all = skaters[skaters["situation"] == "all"].copy()
    skaters_5on5 = skaters[skaters["situation"] == "5on5"].copy()

    skaters_all["norm"] = skaters_all["name"].apply(_norm_name)
    skaters_5on5["norm"] = skaters_5on5["name"].apply(_norm_name)

    skaters_all_idx = skaters_all.set_index("norm")
    skaters_5on5_idx = skaters_5on5.set_index("norm")

    goalies = _mp_goalies()
    goalies_all = goalies[(goalies["situation"] == "all")].copy()
    goalies_all["norm"] = goalies_all["name"].apply(_norm_name)
    goalies_all_idx = goalies_all.set_index("norm")

    # Self-generated lookups for GAR / xGAR / GSAX / PP-PK
    composite_by_pid = _load_composite_war()
    xgar_by_pid      = _load_skater_xgar()
    gsax_by_pid      = _load_goalie_gsax()
    pp_pk_index_by_pid = _build_pp_pk_gar_index()

    age_data = _bulk_fetch_ages(contracts)
    ages = age_data["ages"]
    pids = age_data["playerIds"]

    # League-wide ixG/60 used by _split_off_def_gar — derived from my own xGAR
    # totals (ixG per player from the self-generated CSV) divided by their
    # 5on5 ice time. Cap qualifying pool at 200 min 5v5 TOI as before.
    skaters_5on5["icetime_num"] = pd.to_numeric(skaters_5on5["icetime"], errors="coerce").fillna(0)
    qualified_5on5 = skaters_5on5[skaters_5on5["icetime_num"] >= 200 * 60].copy()
    qualified_5on5["pid_int"] = pd.to_numeric(qualified_5on5["playerId"],
                                                errors="coerce").fillna(0).astype(int)
    qualified_5on5["my_ixg"] = qualified_5on5["pid_int"].map(
        lambda p: xgar_by_pid.get(p, {}).get("ixg", 0.0))
    qualified_5on5["ixg_60"] = qualified_5on5.apply(
        lambda r: (float(r["my_ixg"]) / (r["icetime_num"] / 3600))
                   if r["icetime_num"] > 0 else 0, axis=1)
    league_ixg_60 = qualified_5on5["ixg_60"].tolist()

    out = []
    for c in contracts:
        canonical_norm = _norm_name(c["name"])
        norm = _MP_NAME_ALIASES.get(canonical_norm, canonical_norm)
        cap_hit = float(c["cap_hit"])
        age = ages.get(c["name"])
        pid = pids.get(c["name"])

        contract_type = _classify_contract(c, age)
        years_remaining = max(0, int(c["expiry_year"]) - date.today().year)

        row = {
            "name": c["name"],
            "team": c["team"],
            "position": c["position"],
            "is_goalie": c["position"] == "G",
            "cap_hit": cap_hit,
            "expiry_year": c["expiry_year"],
            "years_remaining": years_remaining,
            "age": age,
            "playerId": pid,
            "contract_type": contract_type,
            "headshot": f"https://assets.nhle.com/mugs/nhl/{current_season()}/{c['team']}/{pid}.png" if pid else "",
            "team_logo": f"https://assets.nhle.com/logos/nhl/svg/{c['team']}_light.svg",
        }

        if c["position"] == "G":
            # Goalie "GAR" is now self-generated GSAX (xG against from my model
            # minus actual goals), looked up by player_id. We still need the
            # MoneyPuck goalie row for GP + ice time facts.
            mp_gp = 0
            mp_toi_min = 0
            if norm in goalies_all_idx.index:
                g = goalies_all_idx.loc[norm]
                if isinstance(g, pd.DataFrame):
                    g = g.iloc[0]
                mp_gp = int(g["games_played"])
                mp_toi_min = round(float(g["icetime"]) / 60, 0)

            my_gsax = gsax_by_pid.get(pid) if pid else None
            if my_gsax is not None:
                row["gar"] = round(float(my_gsax["gsax"]), 2)
                row["games"] = mp_gp
                row["toi_min"] = mp_toi_min
                row["matched"] = True
            else:
                row["gar"] = None
                row["games"] = mp_gp
                row["toi_min"] = mp_toi_min
                row["matched"] = False
            row["xgar"]    = None
            row["off_gar"] = None
            row["def_gar"] = None
            row["pp_gar"]  = None
            row["pk_gar"]  = None
        else:
            # Skater GAR = composite_war (z-blend of RAPM + individual +
            # relative + playmaking + PP + PK); xGAR = self-generated from xG
            # model predictions on individual shots. Both keyed by player_id.
            my_comp = composite_by_pid.get(pid) if pid else None
            my_xgar = xgar_by_pid.get(pid) if pid else None

            # Need MP "all" row for GP + total ice time facts
            mp_gp = 0
            mp_toi_min = 0
            if norm in skaters_all_idx.index:
                s_all = skaters_all_idx.loc[norm]
                if isinstance(s_all, pd.DataFrame):
                    s_all = s_all.iloc[0]
                mp_gp = int(s_all["games_played"])
                mp_toi_min = round(float(s_all["icetime"]) / 60, 0)
            # 5v5 ixG/60 — from my xGAR table (sum of my model's xG) ÷ 5v5 TOI
            ixg_60 = 0
            if norm in skaters_5on5_idx.index:
                s5 = skaters_5on5_idx.loc[norm]
                if isinstance(s5, pd.DataFrame):
                    s5 = s5.iloc[0]
                icetime_5 = float(s5.get("icetime", 0))
                my_ixg = float(my_xgar["ixg"]) if my_xgar else 0.0
                ixg_60 = (my_ixg / (icetime_5 / 3600)) if icetime_5 > 0 else 0

            if my_comp is not None:
                # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
                row["gar"] = round(float(my_comp["composite_war"]) * 12, 2)
                row["games"] = mp_gp
                row["toi_min"] = mp_toi_min
                row["matched"] = True
            else:
                row["gar"] = None
                row["games"] = mp_gp
                row["toi_min"] = mp_toi_min
                row["matched"] = False
            row["xgar"] = round(float(my_xgar["xgar"]), 2) if my_xgar else None

            if row["gar"] is not None:
                off_g, def_g = _split_off_def_gar(row["gar"], c["position"],
                                                    ixg_60, league_ixg_60)
                row["off_gar"] = off_g
                row["def_gar"] = def_g
            else:
                row["off_gar"] = None
                row["def_gar"] = None

            ppk = pp_pk_index_by_pid.get(pid) or {} if pid else {}
            row["pp_gar"] = ppk.get("pp_gar", 0.0)
            row["pk_gar"] = ppk.get("pk_gar", 0.0)

        # Per-$1M metrics
        if row["gar"] is not None and cap_hit > 0:
            row["gar_per_million"] = round(row["gar"] / cap_hit, 2)
        else:
            row["gar_per_million"] = None

        if row.get("off_gar") is not None and cap_hit > 0:
            row["off_gar_per_million"] = round(row["off_gar"] / cap_hit, 2)
        else:
            row["off_gar_per_million"] = None

        if row.get("def_gar") is not None and cap_hit > 0:
            row["def_gar_per_million"] = round(row["def_gar"] / cap_hit, 2)
        else:
            row["def_gar_per_million"] = None

        # Sustainability Score = GAR - xGAR
        if row.get("gar") is not None and row.get("xgar") is not None:
            row["sustainability_score"] = round(row["gar"] - row["xgar"], 2)
        else:
            row["sustainability_score"] = None

        # Surplus Value = (GAR * MARKET_VALUE_PER_GAR) - cap_hit
        # For goalies, GSAX scaled differently — use a goalie-tuned multiplier
        if row.get("gar") is not None:
            multiplier = 0.45 if c["position"] == "G" else MARKET_VALUE_PER_GAR  # GSAX → $0.45M per save
            expected_market = row["gar"] * multiplier
            row["expected_market_value"] = round(expected_market, 2)
            row["surplus_value"] = round(expected_market - cap_hit, 2)
        else:
            row["expected_market_value"] = None
            row["surplus_value"] = None

        # Age-curve flag (three tiers driving the Age column badge):
        #   pre_peak  → under 25 (rising on the public aging curve)
        #   peak      → 25-29   (career-peak window)
        #   post_peak → 30+     (declining)
        if age is not None:
            if age < 25:
                row["age_flag"] = "pre_peak"
            elif age <= 29:
                row["age_flag"] = "peak"
            else:
                row["age_flag"] = "post_peak"
        else:
            row["age_flag"] = None

        # Year-3 projection (for Aging Contracts view)
        if row.get("gar") is not None and age is not None and not row["is_goalie"]:
            row["projected_gar_y3"] = _age_curve_projection(row["gar"], c["position"], age, 3)
            if row["projected_gar_y3"] is not None and cap_hit > 0:
                expected_y3 = row["projected_gar_y3"] * MARKET_VALUE_PER_GAR
                row["projected_surplus_y3"] = round(expected_y3 - cap_hit, 2)
            else:
                row["projected_surplus_y3"] = None
        else:
            row["projected_gar_y3"] = None
            row["projected_surplus_y3"] = None

        out.append(row)
    return out


@app.route("/api/contract-values")
def api_contract_values():
    def _fetch():
        # Tier 1: league-wide capwages snapshot (refreshed on a schedule)
        current = _current_contracts()
        if current:
            log.info("Using capwages snapshot (%d contracts)", len(current["contracts"]))
            contracts = current["contracts"]
            source = "capwages"
            meta = current["meta"]
        else:
            # Tier 2: PuckPedia (typically Cloudflare-blocked)
            pp = _puckpedia_contracts()
            if pp:
                log.info("Using PuckPedia contracts")
                contracts = pp
                source = "puckpedia"
                meta = {}
            else:
                # Tier 3: manually-curated fallback JSON
                manual = _manual_contracts()
                if not manual:
                    return {"blocked": True, "message": "No contract data sources available."}
                contracts = manual["contracts"]
                source = "manual"
                meta = manual["meta"]

        try:
            merged = _merge_contracts_with_moneypuck(contracts)
        except Exception as e:
            log.error("MoneyPuck merge failed: %s", e)
            merged = []
            for c in contracts:
                merged.append({
                    "name": c["name"], "team": c["team"], "position": c["position"],
                    "is_goalie": c["position"] == "G", "cap_hit": float(c["cap_hit"]),
                    "expiry_year": c["expiry_year"], "gar": None, "xgar": None,
                    "games": 0, "toi_min": 0, "matched": False, "gar_per_million": None,
                })

        return {"source": source, "meta": meta, "players": merged}

    try:
        return jsonify(cached("contract_values", _fetch))
    except Exception as e:
        log.error("api_contract_values error: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-cap-efficiency")
def api_team_cap_efficiency():
    """Aggregate Surplus Value per team from /api/contract-values data."""
    def _fetch():
        # Reuse the contract-values cache by calling the underlying fetcher
        cv = cached("contract_values", lambda: None)
        if cv is None:
            # Trigger a fetch via the route's logic — easiest path: call its inner function
            # by replicating the loader chain
            current = _current_contracts()
            if current:
                contracts = current["contracts"]; source = "capwages"
            elif (pp := _puckpedia_contracts()):
                contracts = pp; source = "puckpedia"
            else:
                manual = _manual_contracts()
                if not manual:
                    return {"blocked": True, "message": "No contract data available."}
                contracts = manual["contracts"]; source = "manual"
            try:
                merged = _merge_contracts_with_moneypuck(contracts)
                cv = {"source": source, "players": merged}
                _cache["contract_values"] = {"data": cv, "ts": time.time()}
            except Exception as e:
                return {"error": str(e)}

        teams = {}
        for p in cv.get("players", []):
            t = p.get("team")
            if not t:
                continue
            entry = teams.setdefault(t, {
                "team": t, "team_logo": f"https://assets.nhle.com/logos/nhl/svg/{t}_light.svg",
                "players_counted": 0, "total_cap": 0.0, "total_gar": 0.0, "total_surplus": 0.0,
            })
            entry["players_counted"] += 1
            entry["total_cap"] += float(p.get("cap_hit") or 0)
            if p.get("gar") is not None:
                entry["total_gar"] += float(p["gar"])
            if p.get("surplus_value") is not None:
                entry["total_surplus"] += float(p["surplus_value"])

        rows = []
        for t, e in teams.items():
            avg_per_dollar = round(e["total_gar"] / e["total_cap"], 2) if e["total_cap"] > 0 else 0
            rows.append({
                **e,
                "total_cap": round(e["total_cap"], 2),
                "total_gar": round(e["total_gar"], 2),
                "total_surplus": round(e["total_surplus"], 2),
                "avg_gar_per_dollar": avg_per_dollar,
            })
        rows.sort(key=lambda r: r["total_surplus"], reverse=True)
        return {"teams": rows, "note": "Aggregated only over contracts in the curated dataset (~70 highest cap hits + bargains)."}

    return jsonify(cached_ttl("team_cap_efficiency", 21600, _fetch))


# ---------------------------------------------------------------------------
# Player Lookup + Compare endpoints
# ---------------------------------------------------------------------------
NHL_SEARCH = "https://search.d3.nhle.com/api/v1/search/player"

# Per-key TTL overrides for the cached() helper
_cache_ttls: dict[str, int] = {}


def cached_ttl(key: str, ttl_seconds: int, fn):
    """cached() variant with a per-key TTL override."""
    now = time.time()
    entry = _cache.get(key)
    if entry and now - entry["ts"] < ttl_seconds:
        return entry["data"]
    result = fn()
    _cache[key] = {"data": result, "ts": now}
    return result


def _slugify_name(name: str) -> str:
    base = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")


@app.route("/api/player-search")
def api_player_search():
    """Live search: hit NHL d3 search endpoint and pass through normalized hits."""
    q = (request.args.get("q") or "").strip()
    if len(q) < 2:
        return jsonify({"query": q, "results": []})
    limit = int(request.args.get("limit", 10))

    def _fetch():
        try:
            params = {"culture": "en-us", "limit": limit, "q": q, "active": "true"}
            r = requests.get(NHL_SEARCH, params=params, headers=HEADERS, timeout=8)
            r.raise_for_status()
            data = r.json() or []
        except Exception as e:
            log.info("player-search failed q=%s: %s", q, e)
            return {"query": q, "results": [], "error": str(e)}

        results = []
        for hit in data:
            pid = hit.get("playerId") or hit.get("id")
            if not pid:
                continue
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            name = hit.get("name") or " ".join(filter(None, [hit.get("firstName"), hit.get("lastName")]))
            team_abbrev = hit.get("teamAbbrev") or hit.get("lastTeamAbbrev") or ""
            position = hit.get("positionCode") or hit.get("position") or ""
            sweater = hit.get("sweaterNumber") or hit.get("lastTeamId") or ""
            results.append({
                "id": pid_int,
                "name": name,
                "team": team_abbrev,
                "position": position,
                "sweater": sweater,
                "headshot": f"https://assets.nhle.com/mugs/nhl/{current_season()}/{team_abbrev}/{pid_int}.png" if team_abbrev else "",
                "team_logo": f"https://assets.nhle.com/logos/nhl/svg/{team_abbrev}_light.svg" if team_abbrev else "",
                "slug": _slugify_name(name),
            })
        return {"query": q, "results": results}

    # Cache search results 24h (rosters change rarely)
    return jsonify(cached_ttl(f"player_search:{q.lower()}:{limit}", 86400, _fetch))


def _years_in_league(landing: dict) -> int:
    """Count distinct NHL seasons from career totals seasonTotals (regular only)."""
    seasons = set()
    for entry in (landing.get("seasonTotals") or []):
        if entry.get("leagueAbbrev") == "NHL" and entry.get("gameTypeId") == 2:
            s = entry.get("season")
            if s:
                seasons.add(s)
    return len(seasons)


def _height_str(in_inches: int | None) -> str:
    if not in_inches:
        return "—"
    feet, inches = divmod(int(in_inches), 12)
    return f"{feet}'{inches}\""


def _moneypuck_skater_row(name: str) -> dict | None:
    """Return advanced row for a skater (MoneyPuck raw counts + my-model GAR/xGAR).

    `gar` is my composite_war (from composite_ratings_sim.csv); `xgar` is
    self-generated from my xG model (skater_xgar_self_generated.csv). All
    other fields (counts, on-ice rates) come from MoneyPuck raw counts."""
    norm = _MP_NAME_ALIASES.get(_norm_name(name), _norm_name(name))
    df = _mp_skaters()
    df_all = df[df["situation"] == "all"].copy()
    df_5 = df[df["situation"] == "5on5"].copy()
    df_all["norm"] = df_all["name"].apply(_norm_name)
    df_5["norm"] = df_5["name"].apply(_norm_name)

    row_all = df_all[df_all["norm"] == norm]
    row_5 = df_5[df_5["norm"] == norm]
    if row_all.empty and row_5.empty:
        return None

    # Pull player_id for self-gen lookups
    pid = None
    if not row_all.empty:
        pid = int(row_all.iloc[0].get("playerId", 0)) or None
    elif not row_5.empty:
        pid = int(row_5.iloc[0].get("playerId", 0)) or None

    my_comp = _load_composite_war().get(pid) if pid else None
    my_xgar = _load_skater_xgar().get(pid) if pid else None

    out: dict = {}
    if not row_all.empty:
        r = row_all.iloc[0]
        # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
        out["gar"] = round(float(my_comp["composite_war"]) * 12, 2) if my_comp else None
        out["games"] = int(r.get("games_played", 0))
        out["icf"] = int(r.get("I_F_shotAttempts", 0)) if "I_F_shotAttempts" in r else None
        out["iff"] = int(r.get("I_F_unblockedShotAttempts", 0)) if "I_F_unblockedShotAttempts" in r else None
        out["ixg"] = round(float(my_xgar["ixg"]), 2) if my_xgar else None
        out["ihdcf"] = int(r.get("I_F_highDangerShots", 0)) if "I_F_highDangerShots" in r else None
        out["icf_60"] = round(out["icf"] / (float(r.get("icetime", 1)) / 3600), 2) if out.get("icf") and r.get("icetime", 0) > 0 else None
        out["ixg_60"] = round(out["ixg"] / (float(r.get("icetime", 1)) / 3600), 2) if out.get("ixg") and r.get("icetime", 0) > 0 else None
        out["icetime_min"] = round(float(r.get("icetime", 0)) / 60, 0)
    if not row_5.empty:
        r5 = row_5.iloc[0]
        out["xgar"] = round(float(my_xgar["xgar"]), 2) if my_xgar else None
        # On-ice xGF% from my self-generated table (intersects my-xG-scored
        # shots with the player's shifts). Falls back to MoneyPuck if missing.
        my_onice = _load_onice_xgf().get(pid) if pid else None
        if my_onice is not None:
            out["onice_xgf_pct"] = round(float(my_onice["xgf_pct"]), 1)
        else:
            on_xg_for = float(r5.get("OnIce_F_xGoals", 0)) if "OnIce_F_xGoals" in r5 else 0
            on_xg_ag = float(r5.get("OnIce_A_xGoals", 0)) if "OnIce_A_xGoals" in r5 else 0
            if on_xg_for + on_xg_ag > 0:
                out["onice_xgf_pct"] = round(on_xg_for / (on_xg_for + on_xg_ag) * 100, 1)
        # Corsi % from raw counts (factual)
        on_corsi_for = float(r5.get("OnIce_F_shotAttempts", 0)) if "OnIce_F_shotAttempts" in r5 else 0
        on_corsi_ag = float(r5.get("OnIce_A_shotAttempts", 0)) if "OnIce_A_shotAttempts" in r5 else 0
        if on_corsi_for + on_corsi_ag > 0:
            out["onice_cf_pct"] = round(on_corsi_for / (on_corsi_for + on_corsi_ag) * 100, 1)
        # Zone start %
        oz = float(r5.get("I_F_offZoneStart", 0)) if "I_F_offZoneStart" in r5 else 0
        dz = float(r5.get("I_F_defZoneStart", 0)) if "I_F_defZoneStart" in r5 else 0
        nz = float(r5.get("I_F_neuZoneStart", 0)) if "I_F_neuZoneStart" in r5 else 0
        if oz + dz > 0:
            out["zone_start_pct"] = round(oz / (oz + dz) * 100, 1)
        out["position"] = r5.get("position", "")
        out["team"] = r5.get("team", "")
    return out or None


def _moneypuck_goalie_row(name: str) -> dict | None:
    norm = _MP_NAME_ALIASES.get(_norm_name(name), _norm_name(name))
    df = _mp_goalies()
    df_all = df[df["situation"] == "all"].copy()
    df_all["norm"] = df_all["name"].apply(_norm_name)
    row = df_all[df_all["norm"] == norm]
    if row.empty:
        return None
    r = row.iloc[0]
    shots = float(r.get("ongoal", 0))
    goals = float(r.get("goals", 0))
    icetime = float(r.get("icetime", 0))
    gsax = float(r.get("xGoals", 0)) - goals
    out = {
        "team": r.get("team", ""),
        "games": int(r.get("games_played", 0)),
        "gsax": round(gsax, 2),
        "gsax_60": round(gsax / (icetime / 3600), 3) if icetime > 0 else None,
        "sv_pct": round((shots - goals) / shots * 100, 2) if shots > 0 else None,
    }
    hd_s = float(r.get("highDangerShots", 0))
    hd_g = float(r.get("highDangerGoals", 0))
    md_s = float(r.get("mediumDangerShots", 0))
    md_g = float(r.get("mediumDangerGoals", 0))
    ld_s = float(r.get("lowDangerShots", 0))
    ld_g = float(r.get("lowDangerGoals", 0))
    if hd_s > 0:
        out["hdsv_pct"] = round((hd_s - hd_g) / hd_s * 100, 2)
    if md_s > 0:
        out["mdsv_pct"] = round((md_s - md_g) / md_s * 100, 2)
    if ld_s > 0:
        out["ldsv_pct"] = round((ld_s - ld_g) / ld_s * 100, 2)
    pid = int(r.get("playerId", 0))
    if pid:
        try:
            qs = _quality_starts_for_goalie(pid)
            out.update({"qs": qs.get("qs"), "qs_pct": qs.get("qs_pct"), "starts": qs.get("starts")})
        except Exception:
            pass
    return out


def _player_landing(player_id: int) -> dict | None:
    try:
        return get_json(f"{NHL_API}/player/{player_id}/landing")
    except Exception as e:
        log.info("landing fetch failed pid=%s: %s", player_id, e)
        return None


def _player_game_log(player_id: int, season: str, game_type: int) -> list:
    try:
        data = get_json(f"{NHL_API}/player/{player_id}/game-log/{season}/{game_type}")
        return data.get("gameLog") or []
    except Exception as e:
        log.info("game-log fetch failed pid=%s season=%s type=%s: %s", player_id, season, game_type, e)
        return []


@app.route("/api/player/<int:player_id>")
def api_player_profile(player_id: int):
    """Return full player profile: bio, current-season stats, advanced stats, career stats."""
    def _fetch():
        landing = _player_landing(player_id)
        if not landing:
            return {"error": "Player not found", "id": player_id}

        first = (landing.get("firstName") or {}).get("default", "")
        last = (landing.get("lastName") or {}).get("default", "")
        full_name = f"{first} {last}".strip()
        position = landing.get("position") or ""
        is_goalie = position == "G"

        bio = {
            "id": player_id,
            "name": full_name,
            "first_name": first,
            "last_name": last,
            "position": position,
            "is_goalie": is_goalie,
            "team": (landing.get("currentTeamAbbrev") or ""),
            "team_name": (landing.get("fullTeamName") or {}).get("default", ""),
            "sweater": landing.get("sweaterNumber"),
            "headshot": landing.get("headshot") or "",
            "hero_image": landing.get("heroImage") or "",
            "team_logo": landing.get("teamLogo") or "",
            "height": _height_str(landing.get("heightInInches")),
            "weight": f"{landing.get('weightInPounds', 0)} lb" if landing.get("weightInPounds") else "—",
            "shoots": landing.get("shootsCatches") or "—",
            "birth_date": landing.get("birthDate") or "",
            "birth_city": (landing.get("birthCity") or {}).get("default", ""),
            "birth_country": landing.get("birthCountry") or "",
            "birth_state": (landing.get("birthStateProvince") or {}).get("default", "") if landing.get("birthStateProvince") else "",
            "draft": landing.get("draftDetails") or {},
            "years_in_league": _years_in_league(landing),
        }

        # Age from birth date
        if bio["birth_date"]:
            try:
                from datetime import datetime as _dt
                bd = _dt.fromisoformat(bio["birth_date"]).date()
                today = date.today()
                bio["age"] = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
            except Exception:
                bio["age"] = None
        else:
            bio["age"] = None

        # Hometown string
        hometown_parts = [bio["birth_city"], bio["birth_state"], bio["birth_country"]]
        bio["hometown"] = ", ".join([p for p in hometown_parts if p])

        # Current-season featured stats
        feat = landing.get("featuredStats") or {}
        cur = (feat.get("regularSeason") or {}).get("subSeason") or {}
        career_totals = (feat.get("regularSeason") or {}).get("career") or {}

        if is_goalie:
            current_season_stats = {
                "season": feat.get("season"),
                "team": cur.get("teamAbbrev") or bio["team"],
                "gp": cur.get("gamesPlayed"),
                "gs": cur.get("gamesStarted"),
                "w": cur.get("wins"),
                "l": cur.get("losses"),
                "otl": cur.get("otLosses"),
                "gaa": cur.get("goalsAgainstAvg"),
                "sv_pct": cur.get("savePctg"),
                "so": cur.get("shutouts"),
                "toi": cur.get("timeOnIce"),
            }
        else:
            current_season_stats = {
                "season": feat.get("season"),
                "team": cur.get("teamAbbrev") or bio["team"],
                "gp": cur.get("gamesPlayed"),
                "g": cur.get("goals"),
                "a": cur.get("assists"),
                "pts": cur.get("points"),
                "plus_minus": cur.get("plusMinus"),
                "pim": cur.get("pim"),
                "shots": cur.get("shots"),
                "shooting_pct": cur.get("shootingPctg"),
                "toi_per_game": cur.get("avgToi"),
                "pp_g": cur.get("powerPlayGoals"),
                "pp_pts": cur.get("powerPlayPoints"),
                "sh_g": cur.get("shorthandedGoals"),
                "sh_pts": cur.get("shorthandedPoints"),
                "gw": cur.get("gameWinningGoals"),
                "ot": cur.get("otGoals"),
                "fow_pct": cur.get("faceoffWinningPctg"),
            }

        # Advanced stats from MoneyPuck
        if is_goalie:
            advanced = _moneypuck_goalie_row(full_name) or {}
        else:
            advanced = _moneypuck_skater_row(full_name) or {}

        # Career year-by-year — separated regular vs playoff
        career_regular = []
        career_playoff = []
        for entry in (landing.get("seasonTotals") or []):
            row = {
                "season": entry.get("season"),
                "league": entry.get("leagueAbbrev"),
                "team": entry.get("teamName", {}).get("default") if isinstance(entry.get("teamName"), dict) else entry.get("teamName"),
                "gp": entry.get("gamesPlayed"),
            }
            if is_goalie:
                row.update({
                    "gs": entry.get("gamesStarted"),
                    "w": entry.get("wins"),
                    "l": entry.get("losses"),
                    "otl": entry.get("otLosses"),
                    "gaa": entry.get("goalsAgainstAvg"),
                    "sv_pct": entry.get("savePctg"),
                    "so": entry.get("shutouts"),
                })
            else:
                row.update({
                    "g": entry.get("goals"),
                    "a": entry.get("assists"),
                    "pts": entry.get("points"),
                    "plus_minus": entry.get("plusMinus"),
                    "pim": entry.get("pim"),
                    "shots": entry.get("shots"),
                    "pp_g": entry.get("powerPlayGoals"),
                    "sh_g": entry.get("shorthandedGoals"),
                })
            if entry.get("gameTypeId") == 2:
                career_regular.append(row)
            elif entry.get("gameTypeId") == 3:
                career_playoff.append(row)

        # Last 5 games
        last5 = []
        for g in (landing.get("last5Games") or [])[:5]:
            row = {
                "date": g.get("gameDate"),
                "opponent": g.get("opponentAbbrev"),
                "home_road": "vs" if g.get("homeRoadFlag") == "H" else "@",
                "result": ("W" if g.get("decision") == "W" else ("L" if g.get("decision") == "L" else (g.get("teamPoints") and "OTL")) or ""),
            }
            if is_goalie:
                row.update({
                    "shots_against": g.get("shotsAgainst"),
                    "goals_against": g.get("goalsAgainst"),
                    "saves": (g.get("shotsAgainst") or 0) - (g.get("goalsAgainst") or 0),
                    "sv_pct": g.get("savePctg"),
                    "decision": g.get("decision"),
                    "toi": g.get("toi"),
                })
            else:
                row.update({
                    "goals": g.get("goals"),
                    "assists": g.get("assists"),
                    "points": g.get("points"),
                    "shots": g.get("shots"),
                    "plus_minus": g.get("plusMinus"),
                    "toi": g.get("toi"),
                    "pim": g.get("pim"),
                })
            last5.append(row)

        return {
            "bio": bio,
            "current_season": current_season_stats,
            "career_totals": career_totals,
            "advanced": advanced,
            "career_regular": career_regular,
            "career_playoff": career_playoff,
            "last5": last5,
        }

    # Cache 1 hour per player
    return jsonify(cached_ttl(f"player_profile:{player_id}", 3600, _fetch))


@app.route("/api/player/<int:player_id>/shot-chart")
def api_player_shot_chart(player_id: int):
    """Return shot coordinates from this player's last N games (default 10).
    Each shot: {x, y, type: 'goal'|'shot-on-goal'|'missed'|'blocked', game, opp, period, time}.
    Coordinates are NHL standard rink coords (x: -100 to +100, y: -42.5 to +42.5).
    """
    n_games = int(request.args.get("games", 10))

    def _fetch():
        season = current_season()
        log_entries = _player_game_log(player_id, season, 2)
        log_entries = log_entries[:n_games]
        if not log_entries:
            return {"shots": [], "games_used": 0, "season": season}

        shots = []
        used = 0
        for g in log_entries:
            game_id = g.get("gameId")
            if not game_id:
                continue
            try:
                pbp = cached_ttl(f"pbp:{game_id}", 86400, lambda gid=game_id: get_json(f"{NHL_API}/gamecenter/{gid}/play-by-play"))
            except Exception as e:
                log.info("pbp fetch failed gid=%s: %s", game_id, e)
                continue

            opp = g.get("opponentAbbrev", "")
            game_date = g.get("gameDate", "")

            for play in (pbp.get("plays") or []):
                t = play.get("typeDescKey")
                if t not in {"goal", "shot-on-goal", "missed-shot", "blocked-shot"}:
                    continue
                d = play.get("details") or {}
                shooter = d.get("shootingPlayerId") or d.get("scoringPlayerId")
                if shooter != player_id:
                    continue
                x = d.get("xCoord")
                y = d.get("yCoord")
                if x is None or y is None:
                    continue
                shots.append({
                    "x": x,
                    "y": y,
                    "type": t,
                    "game_id": game_id,
                    "opp": opp,
                    "date": game_date,
                    "period": play.get("periodDescriptor", {}).get("number"),
                    "time": play.get("timeInPeriod"),
                    "shot_type": d.get("shotType") or "",
                })
            used += 1

        return {"shots": shots, "games_used": used, "season": season, "n_games_requested": n_games}

    return jsonify(cached_ttl(f"shot_chart:{player_id}:{n_games}", 3600, _fetch))


@app.route("/api/player/<int:player_id>/similar")
def api_player_similar(player_id: int):
    """Find 3-5 players similar by GAR proxy, position, age."""
    def _fetch():
        landing = _player_landing(player_id)
        if not landing:
            return {"similar": [], "error": "Player not found"}
        position = landing.get("position") or ""
        if position == "G":
            return {"similar": [], "note": "Similar-player matching is currently skater-only."}
        first = (landing.get("firstName") or {}).get("default", "")
        last = (landing.get("lastName") or {}).get("default", "")
        full = f"{first} {last}".strip()
        norm_self = _norm_name(full)

        # Compute self age
        age_self = None
        if landing.get("birthDate"):
            try:
                from datetime import datetime as _dt
                bd = _dt.fromisoformat(landing["birthDate"]).date()
                today = date.today()
                age_self = today.year - bd.year - ((today.month, today.day) < (bd.month, bd.day))
            except Exception:
                pass

        # Self GAR
        self_row = _moneypuck_skater_row(full) or {}
        gar_self = self_row.get("gar")
        ozs_self = self_row.get("zone_start_pct")

        # Score every other skater
        df = _mp_skaters()
        df_all = df[df["situation"] == "all"].copy()
        df_5 = df[df["situation"] == "5on5"].copy()
        df_5_idx = df_5.set_index(df_5["name"].apply(_norm_name))

        # Position bucket: forwards group together, D separately
        pos_bucket = {"C", "L", "R"} if position in {"C", "L", "R"} else {position}
        df_all = df_all[df_all["position"].isin(pos_bucket)]
        df_all["norm"] = df_all["name"].apply(_norm_name)
        df_all = df_all[df_all["norm"] != norm_self]

        # Roster age data: use draft year as cheap proxy via NHL roster lookup is too expensive;
        # fall back to GAR + zone-start similarity only.
        comp_lookup = _load_composite_war()
        rows = []
        for _, r in df_all.iterrows():
            pid_other = int(r.get("playerId", 0))
            # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
            # (uniform scaling of both sides of the similarity score leaves the ranking unchanged)
            gar_other = float(comp_lookup.get(pid_other, {}).get("composite_war", 0.0)) * 12
            score = 0.0
            if gar_self is not None:
                score += abs(gar_other - gar_self)
            n = r["norm"]
            ozs_other = None
            if n in df_5_idx.index:
                r5 = df_5_idx.loc[n]
                if isinstance(r5, pd.DataFrame):
                    r5 = r5.iloc[0]
                oz = float(r5.get("I_F_offZoneStart", 0))
                dz = float(r5.get("I_F_defZoneStart", 0))
                if oz + dz > 0:
                    ozs_other = oz / (oz + dz) * 100
            if ozs_self is not None and ozs_other is not None:
                score += abs(ozs_other - ozs_self) / 5.0  # weight zone-start lower

            rows.append({
                "name": r["name"],
                "team": r["team"],
                "position": r["position"],
                "gar": round(gar_other, 2),
                "games": int(r.get("games_played", 0)),
                "score": score,
            })

        rows.sort(key=lambda x: x["score"])
        # Filter out tiny-sample players (< 20 GP) so we get meaningful comparisons
        rows = [r for r in rows if r["games"] >= 20][:5]

        return {
            "self": {"name": full, "position": position, "gar": gar_self, "age": age_self},
            "similar": rows,
        }

    return jsonify(cached_ttl(f"similar:{player_id}", 3600, _fetch))


# ---------------------------------------------------------------------------
# Bulk leaderboards — all skaters / goalies for the new Players page
# ---------------------------------------------------------------------------
def _safe_div(a, b):
    return a / b if b else 0.0


def _nhl_plus_minus_by_pid(team_abbrevs) -> dict[int, int]:
    """Bulk-fetch NHL plus/minus for every skater across the supplied team set.

    Hits /v1/club-stats/{TEAM}/{season}/2 per team in parallel (32 small JSON
    payloads) and merges into {playerId → plusMinus}. The traditional MoneyPuck
    skaters CSV has no plus/minus column; this is how we get the canonical NHL
    figure that matches each player's profile page.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    season = current_season()
    out: dict[int, int] = {}

    def _pull(team):
        try:
            j = get_json(f"{NHL_API}/club-stats/{team}/{season}/2")
        except Exception:
            return {}
        d = {}
        for s in (j.get("skaters") or []):
            pid = s.get("playerId")
            pm = s.get("plusMinus")
            if pid is not None and pm is not None:
                d[int(pid)] = int(pm)
        return d

    teams = sorted({t for t in team_abbrevs if t})
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_pull, t) for t in teams]
        for f in as_completed(futs):
            try:
                out.update(f.result())
            except Exception:
                continue
    return out


def _build_skater_full_row(r_all, r_5, r_pp=None, r_pk=None, plus_minus_by_pid=None):
    """Build a unified skater row merging 'all' situation + '5on5' situation.
    Optional `r_pp` (5on4) and `r_pk` (4on5) rows are used for the
    power-play and penalty-kill columns (2026-06-07 Change 2).

    GAR / xGAR / Game Score / on-ice xGF% are all sourced from my own model
    files via the loader helpers above; MoneyPuck provides only the raw
    counts (goals, assists, shots, GP, ice time, hits, blocks, zone starts)."""
    icetime_5 = float(r_5.get("icetime", 0)) if r_5 is not None else 0
    icetime_all = float(r_all.get("icetime", 0))
    games = int(r_all.get("games_played", 0))
    pid = int(r_all.get("playerId", 0))

    # Self-generated analytical lookups (cached per request batch)
    composite = _load_composite_war().get(pid)
    my_xgar   = _load_skater_xgar().get(pid)
    my_gs     = _load_game_score().get(pid)
    my_onice  = _load_onice_xgf().get(pid)
    my_qq     = _load_qoc_qot().get(pid)

    # 5on5 on-ice metrics
    if r_5 is not None:
        on_xg_for = float(r_5.get("OnIce_F_xGoals", 0)) if "OnIce_F_xGoals" in r_5 else 0
        on_xg_ag = float(r_5.get("OnIce_A_xGoals", 0)) if "OnIce_A_xGoals" in r_5 else 0
        on_corsi_for = float(r_5.get("OnIce_F_shotAttempts", 0)) if "OnIce_F_shotAttempts" in r_5 else 0
        on_corsi_ag = float(r_5.get("OnIce_A_shotAttempts", 0)) if "OnIce_A_shotAttempts" in r_5 else 0
        on_fenwick_for = float(r_5.get("OnIce_F_unblockedShotAttempts", 0)) if "OnIce_F_unblockedShotAttempts" in r_5 else 0
        on_fenwick_ag = float(r_5.get("OnIce_A_unblockedShotAttempts", 0)) if "OnIce_A_unblockedShotAttempts" in r_5 else 0
        on_hd_for = float(r_5.get("OnIce_F_highDangerShots", 0)) if "OnIce_F_highDangerShots" in r_5 else 0
        on_hd_ag = float(r_5.get("OnIce_A_highDangerShots", 0)) if "OnIce_A_highDangerShots" in r_5 else 0
        # Zone starts
        oz = float(r_5.get("I_F_offZoneStart", 0)) if "I_F_offZoneStart" in r_5 else 0
        dz = float(r_5.get("I_F_defZoneStart", 0)) if "I_F_defZoneStart" in r_5 else 0
        # Individual rates. ixG comes from my self-generated xGAR table
        # (sum of per-shot predictions from my xG model on this player's
        # individual shots). Corsi/Fenwick/HD are raw shot counts.
        i_xg_5 = float(my_xgar["ixg"]) if my_xgar else 0.0
        i_corsi_5 = float(r_5.get("I_F_shotAttempts", 0)) if "I_F_shotAttempts" in r_5 else 0
        i_fenwick_5 = float(r_5.get("I_F_unblockedShotAttempts", 0)) if "I_F_unblockedShotAttempts" in r_5 else 0
        i_hd_5 = float(r_5.get("I_F_highDangerShots", 0)) if "I_F_highDangerShots" in r_5 else 0
        # Rush / rebound
        i_rush = float(r_5.get("I_F_rush_attempts", 0)) if "I_F_rush_attempts" in r_5 else 0
        i_reb = float(r_5.get("I_F_rebound_attempts", 0)) if "I_F_rebound_attempts" in r_5 else 0
    else:
        on_xg_for = on_xg_ag = on_corsi_for = on_corsi_ag = 0
        on_fenwick_for = on_fenwick_ag = on_hd_for = on_hd_ag = 0
        oz = dz = i_xg_5 = i_corsi_5 = i_fenwick_5 = i_hd_5 = i_rush = i_reb = 0

    cf_pct = _safe_div(on_corsi_for, on_corsi_for + on_corsi_ag) * 100
    ff_pct = _safe_div(on_fenwick_for, on_fenwick_for + on_fenwick_ag) * 100
    # On-ice xGF% — self-generated from my xG model + shifts. Falls back to
    # MoneyPuck-derived on-ice xG count when the self-gen lookup is missing
    # (only happens for players who haven't taken/been on the ice for any
    # shot in the 25-26 shot dataset — rookies just called up).
    if my_onice is not None:
        xgf_pct = float(my_onice["xgf_pct"])
        xga_pct = round(100.0 - xgf_pct, 2) if xgf_pct > 0 else 0.0
    else:
        xgf_pct = _safe_div(on_xg_for, on_xg_for + on_xg_ag) * 100
        xga_pct = _safe_div(on_xg_ag, on_xg_for + on_xg_ag) * 100
    hdcf_pct = _safe_div(on_hd_for, on_hd_for + on_hd_ag) * 100
    zone_start_pct = _safe_div(oz, oz + dz) * 100

    # Per-60 rates (5on5)
    hours_5 = icetime_5 / 3600 if icetime_5 > 0 else 0
    icf_60 = i_corsi_5 / hours_5 if hours_5 > 0 else 0
    iff_60 = i_fenwick_5 / hours_5 if hours_5 > 0 else 0
    ixg_60 = i_xg_5 / hours_5 if hours_5 > 0 else 0
    ihdcf_60 = i_hd_5 / hours_5 if hours_5 > 0 else 0
    rush_pct = _safe_div(i_rush, i_corsi_5) * 100
    rebound_pct = _safe_div(i_reb, i_corsi_5) * 100

    # All-situation totals
    goals = int(r_all.get("I_F_goals", 0)) if "I_F_goals" in r_all else 0
    pri_a = int(r_all.get("I_F_primaryAssists", 0)) if "I_F_primaryAssists" in r_all else 0
    sec_a = int(r_all.get("I_F_secondaryAssists", 0)) if "I_F_secondaryAssists" in r_all else 0
    assists = pri_a + sec_a
    points = goals + assists
    shots = int(r_all.get("I_F_shotsOnGoal", 0)) if "I_F_shotsOnGoal" in r_all else 0
    sh_pct = _safe_div(goals, shots) * 100
    # Plus/Minus: MoneyPuck doesn't publish I_F_plusMinus. Pull canonical NHL
    # plus/minus from /v1/club-stats (bulk-loaded above into plus_minus_by_pid).
    # Falls back to 0 only if the caller didn't supply the dict (e.g. an
    # ad-hoc test path) — production routes always pass it in.
    _pid = int(r_all.get("playerId", 0)) if r_all.get("playerId") is not None else 0
    plus_minus = int((plus_minus_by_pid or {}).get(_pid, 0))
    pim = int(r_all.get("penalityMinutes", 0)) if "penalityMinutes" in r_all else 0

    # Self-generated GAR + xGAR + Game Score (replaces the MoneyPuck proxies)
    # TEMP STOPGAP: GAR x12 (undo WAR_UNIT 0.5, times 6 goals per win), to be replaced by the composite rebuild
    gar = round(float(composite["composite_war"]) * 12, 2) if composite else None
    xgar = round(float(my_xgar["xgar"]), 2) if my_xgar else None
    game_score = round(float(my_gs["game_score"]), 2) if my_gs else 0.0
    toi_per_game_sec = _safe_div(icetime_all, games)

    # PP (5on4) counts. Per-game ice time uses the player's all-situation GP
    # so a forward with 0 PP shifts in a given game contributes that game to
    # the denominator (consistent with the way "TOI on PP per game" is
    # reported industry-wide).
    if r_pp is not None:
        pp_goals = int(r_pp.get("I_F_goals", 0))
        pp_a1    = int(r_pp.get("I_F_primaryAssists", 0))
        pp_a2    = int(r_pp.get("I_F_secondaryAssists", 0))
        pp_icetime_sec = float(r_pp.get("icetime", 0))
    else:
        pp_goals = pp_a1 = pp_a2 = 0
        pp_icetime_sec = 0.0
    pp_assists = pp_a1 + pp_a2
    pp_points  = pp_goals + pp_assists
    pp_toi_min       = round(pp_icetime_sec / 60, 1)
    pp_toi_per_game_min = round(_safe_div(pp_icetime_sec, games) / 60, 2) if games > 0 else 0.0

    # PK (4on5) counts. PK +/- = shorthanded goals scored − power-play goals
    # allowed, both at on-ice level in 4on5. MoneyPuck doesn't ship a direct
    # I_F_plusMinus while shorthanded so we derive it from OnIce counts.
    if r_pk is not None:
        pk_icetime_sec  = float(r_pk.get("icetime", 0))
        pk_onice_goals_f = float(r_pk.get("OnIce_F_goals", 0))
        pk_onice_goals_a = float(r_pk.get("OnIce_A_goals", 0))
    else:
        pk_icetime_sec = pk_onice_goals_f = pk_onice_goals_a = 0.0
    pk_toi_min       = round(pk_icetime_sec / 60, 1)
    pk_toi_per_game_min = round(_safe_div(pk_icetime_sec, games) / 60, 2) if games > 0 else 0.0
    pk_pm = int(pk_onice_goals_f - pk_onice_goals_a)

    return {
        "playerId": int(r_all.get("playerId", 0)),
        "name": r_all["name"],
        "team": r_all["team"],
        "position": r_all["position"],
        # Scoring
        "gp": games,
        "goals": goals,
        "assists": assists,
        "points": points,
        "shots": shots,
        "shooting_pct": round(sh_pct, 2),
        "plus_minus": plus_minus,
        "pim": pim,
        "toi_min": round(icetime_all / 60),
        "toi_per_game_sec": round(toi_per_game_sec),
        "toi_per_game_min": round(toi_per_game_sec / 60, 2),
        # Possession (5v5 on-ice)
        "cf_pct": round(cf_pct, 2),
        "ff_pct": round(ff_pct, 2),
        "xgf_pct": round(xgf_pct, 2),
        "xga_pct": round(xga_pct, 2),
        "hdcf_pct": round(hdcf_pct, 2),
        # Individual impact. `gar` and `xgar` may be None for players who are
        # in MoneyPuck but not in my composite/xGAR self-gen tables (e.g., new
        # call-ups whose shots aren't in the augmented shot dataset yet).
        # Frontend formatters guard for null; pass through without rounding.
        "gar": gar,
        "xgar": xgar,
        "game_score": game_score,
        "ixg_60": round(ixg_60, 2),
        "icf_60": round(icf_60, 2),
        "iff_60": round(iff_60, 2),
        "ihdcf_60": round(ihdcf_60, 2),
        "rush_pct": round(rush_pct, 2),
        "rebound_pct": round(rebound_pct, 2),
        # Usage
        "zone_start_pct": round(zone_start_pct, 2),
        # QoC / QoT — weighted avg GAR of opponents / teammates from
        # 25-26 shift overlaps. See model/build_qoc_qot.py.
        "qoc": round(float(my_qq["qoc"]), 3) if my_qq and my_qq.get("qoc") is not None else None,
        "qot": round(float(my_qq["qot"]), 3) if my_qq and my_qq.get("qot") is not None else None,
        # Power play (5on4 situation counts from MoneyPuck)
        "pp_goals":   pp_goals,
        "pp_assists": pp_assists,
        "pp_points":  pp_points,
        "pp_toi_min": pp_toi_min,
        "pp_toi_per_game_min": pp_toi_per_game_min,
        # Penalty kill (4on5 situation counts from MoneyPuck)
        "pk_toi_min": pk_toi_min,
        "pk_toi_per_game_min": pk_toi_per_game_min,
        "pk_pm":      pk_pm,
        # Headshot
        "headshot": f"https://assets.nhle.com/mugs/nhl/{current_season()}/{r_all['team']}/{int(r_all.get('playerId', 0))}.png",
        "team_logo": f"https://assets.nhle.com/logos/nhl/svg/{r_all['team']}_light.svg",
    }


def _compute_relative_metrics(rows: list[dict]) -> list[dict]:
    """Add Relative CF% and Relative xGF% — diff vs team average among non-this players."""
    by_team: dict[str, list[dict]] = {}
    for r in rows:
        by_team.setdefault(r["team"], []).append(r)

    for r in rows:
        team_rows = by_team.get(r["team"], [])
        others = [o for o in team_rows if o["playerId"] != r["playerId"]]
        if others:
            team_cf = sum(o["cf_pct"] for o in others) / len(others)
            team_xgf = sum(o["xgf_pct"] for o in others) / len(others)
            r["rel_cf_pct"] = round(r["cf_pct"] - team_cf, 2)
            r["rel_xgf_pct"] = round(r["xgf_pct"] - team_xgf, 2)
        else:
            r["rel_cf_pct"] = None
            r["rel_xgf_pct"] = None
    return rows


@app.route("/api/players-full")
def api_players_full():
    """Full skater dataset from MoneyPuck — every player with all displayed stats.
    Optional ?min_toi_min=200 filter."""
    min_toi_min = int(request.args.get("min_toi_min", 0))

    def _fetch():
        df = _mp_skaters()
        df_all = df[df["situation"] == "all"].copy()
        df_5  = df[df["situation"] == "5on5"].copy()
        df_pp = df[df["situation"] == "5on4"].copy()  # PP = team has man advantage
        df_pk = df[df["situation"] == "4on5"].copy()  # PK = team is shorthanded
        df_5_idx  = df_5.set_index(df_5["playerId"].astype(int))
        df_pp_idx = df_pp.set_index(df_pp["playerId"].astype(int))
        df_pk_idx = df_pk.set_index(df_pk["playerId"].astype(int))

        # One bulk NHL-API fetch for plus/minus across all teams in this dataset.
        pm_by_pid = cached(
            "nhl_plus_minus_by_pid",
            lambda: _nhl_plus_minus_by_pid(df_all["team"].dropna().unique().tolist()),
        )

        rows = []
        for _, r_all in df_all.iterrows():
            pid = int(r_all.get("playerId", 0))
            def _row_for(idx):
                if pid not in idx.index:
                    return None
                rr = idx.loc[pid]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
                return rr
            r_5  = _row_for(df_5_idx)
            r_pp = _row_for(df_pp_idx)
            r_pk = _row_for(df_pk_idx)
            rows.append(_build_skater_full_row(r_all, r_5, r_pp, r_pk,
                                                plus_minus_by_pid=pm_by_pid))

        rows = _compute_relative_metrics(rows)
        return {"source": "self_generated_analytics+moneypuck_counts",
                "season": current_season(), "players": rows}

    data = cached_ttl("players_full", 3600, _fetch)
    if min_toi_min > 0:
        filtered = [p for p in data["players"] if p["toi_min"] >= min_toi_min]
        return jsonify({**data, "players": filtered, "min_toi_min": min_toi_min})
    return jsonify(data)


@app.route("/api/goalies-full")
def api_goalies_full():
    """Full goalie dataset. GSAX is self-generated (model/goalie_gsax_self_generated.csv,
    built from my xG model + goalie shifts); raw counts (GP, shots, goals
    by danger) still come from MoneyPuck. Optional ?min_games=10."""
    min_games = int(request.args.get("min_games", 0))

    def _fetch():
        df = _mp_goalies()
        df_all = df[df["situation"] == "all"].copy()

        cols = ["xGoals", "goals", "icetime", "games_played", "ongoal", "playerId",
                "lowDangerShots", "mediumDangerShots", "highDangerShots",
                "lowDangerGoals", "mediumDangerGoals", "highDangerGoals"]
        for c in cols:
            if c in df_all.columns:
                df_all[c] = pd.to_numeric(df_all[c], errors="coerce").fillna(0)

        my_gsax = _load_goalie_gsax()
        rows = []
        for _, r in df_all.iterrows():
            pid = int(r.get("playerId", 0))
            shots = float(r.get("ongoal", 0))
            goals = float(r.get("goals", 0))
            icetime = float(r.get("icetime", 0))
            # Self-generated GSAX (xG against from my model − actual goals)
            my_g = my_gsax.get(pid)
            gsax = float(my_g["gsax"]) if my_g else (float(r.get("xGoals", 0)) - goals)
            hd_s = float(r.get("highDangerShots", 0))
            hd_g = float(r.get("highDangerGoals", 0))
            md_s = float(r.get("mediumDangerShots", 0))
            md_g = float(r.get("mediumDangerGoals", 0))
            rows.append({
                "playerId": int(r.get("playerId", 0)),
                "name": r.get("name", ""),
                "team": r.get("team", ""),
                "position": "G",
                "gp": int(r.get("games_played", 0)),
                "toi_min": round(icetime / 60),
                "gsax": round(gsax, 2),
                "gsax_60": round(gsax / (icetime / 3600), 3) if icetime > 0 else 0,
                "sv_pct": round((shots - goals) / shots * 100, 2) if shots > 0 else 0,
                "hdsv_pct": round((hd_s - hd_g) / hd_s * 100, 2) if hd_s > 0 else 0,
                "mdsv_pct": round((md_s - md_g) / md_s * 100, 2) if md_s > 0 else 0,
                "headshot": f"https://assets.nhle.com/mugs/nhl/{current_season()}/{r.get('team', '')}/{int(r.get('playerId', 0))}.png",
                "team_logo": f"https://assets.nhle.com/logos/nhl/svg/{r.get('team', '')}_light.svg",
            })
        return {"source": "self_generated_gsax+moneypuck_counts",
                "season": current_season(), "goalies": rows}

    data = cached_ttl("goalies_full", 3600, _fetch)
    if min_games > 0:
        filtered = [g for g in data["goalies"] if g["gp"] >= min_games]
        return jsonify({**data, "goalies": filtered, "min_games": min_games})
    return jsonify(data)


@app.route("/api/player/<int:player_id>/rankings")
def api_player_rankings(player_id: int):
    """Return this player's league rank in key categories.
    Used by Player Search to show 'GAR: 3rd in NHL' pills under headshot."""
    def _fetch():
        landing = _player_landing(player_id)
        if not landing:
            return {"error": "Player not found"}
        position = landing.get("position") or ""
        is_goalie = position == "G"

        if is_goalie:
            # Avoid recursive route call — fetch directly
            df = _mp_goalies()
            df_all = df[(df["situation"] == "all") & (df["games_played"] >= 10)].copy()
            df_all["icetime"] = pd.to_numeric(df_all["icetime"], errors="coerce").fillna(0)
            df_all["xGoals"] = pd.to_numeric(df_all["xGoals"], errors="coerce").fillna(0)
            df_all["goals"] = pd.to_numeric(df_all["goals"], errors="coerce").fillna(0)
            df_all["ongoal"] = pd.to_numeric(df_all["ongoal"], errors="coerce").fillna(0)
            # Self-generated GSAX overrides MoneyPuck xG when available
            my_gsax = _load_goalie_gsax()
            df_all["gsax"] = df_all.apply(
                lambda r: float(my_gsax[int(r["playerId"])]["gsax"])
                          if int(r["playerId"]) in my_gsax
                          else float(r["xGoals"] - r["goals"]), axis=1)
            df_all["sv_pct"] = ((df_all["ongoal"] - df_all["goals"]) / df_all["ongoal"].replace(0, 1)) * 100

            rankings = []
            for stat_key, label, asc in [
                ("gsax", "GSAX", False),
                ("sv_pct", "SV%", False),
            ]:
                ranked = df_all.sort_values(stat_key, ascending=asc).reset_index(drop=True)
                row = ranked[ranked["playerId"] == player_id]
                if not row.empty:
                    rank = int(row.index[0]) + 1
                    total = len(ranked)
                    rankings.append({
                        "stat": label,
                        "rank": rank,
                        "total": total,
                        "value": round(float(row.iloc[0][stat_key]), 2),
                        "scope": "qualified goalies",
                    })
            return {"player_id": player_id, "rankings": rankings}

        # Skater rankings — pull bulk skater data and compute ranks for key stats
        full = cached_ttl("players_full", 3600, lambda: api_players_full().get_json())
        # api_players_full returns wrapped Response; safer to compute fresh:
        df = _mp_skaters()
        df_all = df[df["situation"] == "all"].copy()
        df_5 = df[df["situation"] == "5on5"].copy()
        df_5_idx = df_5.set_index(df_5["playerId"].astype(int))

        pm_by_pid = cached(
            "nhl_plus_minus_by_pid",
            lambda: _nhl_plus_minus_by_pid(df_all["team"].dropna().unique().tolist()),
        )

        rows = []
        for _, r_all in df_all.iterrows():
            pid = int(r_all.get("playerId", 0))
            r_5 = df_5_idx.loc[pid] if pid in df_5_idx.index else None
            if isinstance(r_5, pd.DataFrame):
                r_5 = r_5.iloc[0]
            rows.append(_build_skater_full_row(r_all, r_5,
                                                plus_minus_by_pid=pm_by_pid))

        # Filter out tiny-sample players for ranking purposes
        qualified = [r for r in rows if r["toi_min"] >= 200]
        # Position-bucket rankings for forward stats
        pos_bucket = {"C", "L", "R"} if position in {"C", "L", "R"} else {"D"}
        pos_qualified = [r for r in qualified if r["position"] in pos_bucket]
        pos_label = "forwards" if position in {"C", "L", "R"} else "defensemen"

        def _rank(rows_list, key, scope_label, descending=True):
            ranked = sorted(rows_list, key=lambda r: r.get(key, 0) or 0, reverse=descending)
            for i, r in enumerate(ranked, start=1):
                if r["playerId"] == player_id:
                    return {"rank": i, "total": len(ranked), "value": r.get(key), "scope": scope_label}
            return None

        rankings = []
        rank_specs = [
            ("points", "Points", "skaters", True),
            ("goals", "Goals", "skaters", True),
            ("assists", "Assists", "skaters", True),
            ("gar", "GAR", "skaters", True),
            ("xgar", "xGAR", "skaters", True),
            ("xgf_pct", "xGF%", pos_label, True),
            ("hdcf_pct", "HDCF%", pos_label, True),
            ("ixg_60", "ixG/60", pos_label, True),
            ("rel_cf_pct", "Rel CF%", pos_label, True),
        ]

        for key, label, scope, desc in rank_specs:
            scope_rows = pos_qualified if scope == pos_label else qualified
            r = _rank(scope_rows, key, scope, desc)
            if r:
                rankings.append({"stat": label, **r})

        return {"player_id": player_id, "rankings": rankings, "position": position}

    return jsonify(cached_ttl(f"rankings:{player_id}", 3600, _fetch))


# ---------------------------------------------------------------------------
# xG model — Phase 1
# ---------------------------------------------------------------------------
XG_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "xg_model.pkl")
XG_META_PATH = os.path.join(os.path.dirname(__file__), "model", "xg_model_meta.json")


# RAPM endpoint serves the SINGLE-SEASON 25-26 build for the Players section.
# The multi-season `rapm_results.csv` is unchanged and is still consumed by
# the simulation composite (build_composite_sim.py).
RAPM_CSV_PATH = os.path.join(os.path.dirname(__file__), "model", "rapm_single_season.csv")
RAPM_META_PATH = os.path.join(os.path.dirname(__file__), "model", "rapm_single_season_meta.json")


@app.route("/api/rapm-leaders")
def api_rapm_leaders():
    """Return top 25 by total / offensive / defensive RAPM. 24h cache."""
    def _fetch():
        if not os.path.exists(RAPM_CSV_PATH):
            return {
                "trained": False,
                "message": "RAPM model has not been trained yet. Run: python3 model/train_rapm.py",
            }
        try:
            df = pd.read_csv(RAPM_CSV_PATH)
        except Exception as e:
            return {"trained": False, "error": str(e)}

        meta = {}
        if os.path.exists(RAPM_META_PATH):
            try:
                with open(RAPM_META_PATH) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        # The single-season CSV was extended by model/apply_rapm_shrinkage.py
        # with Bayesian-shrunk values (K=1000, shrunk_*_rapm columns) plus the
        # raw multi-season prior (ms_*_rapm). Headline sort is the shrunk
        # total; the SS and MS columns ride along so the frontend can render
        # a dual-view comparison.
        has_shrunk = "shrunk_total_rapm" in df.columns

        def _f(v):
            return float(v) if pd.notna(v) else None

        def _to_records(sub):
            out = []
            for r in sub.itertuples(index=False):
                rec = {
                    "player_id": int(r.player_id) if not pd.isna(r.player_id) else None,
                    "player_name": str(r.player_name),
                    "team": str(r.team) if not pd.isna(r.team) else "",
                    "season": str(r.season) if not pd.isna(r.season) else "",
                    "total_rapm": float(r.total_rapm),
                    "offensive_rapm": float(r.offensive_rapm),
                    "defensive_rapm": float(r.defensive_rapm),
                    "toi_minutes": float(r.toi_minutes),
                    "shift_count": int(r.shift_count),
                }
                if has_shrunk:
                    rec.update({
                        "shrunk_total_rapm": _f(getattr(r, "shrunk_total_rapm", None)),
                        "shrunk_off_rapm":   _f(getattr(r, "shrunk_off_rapm", None)),
                        "shrunk_def_rapm":   _f(getattr(r, "shrunk_def_rapm", None)),
                        "ms_total_rapm":     _f(getattr(r, "ms_total_rapm", None)),
                        "ms_off_rapm":       _f(getattr(r, "ms_off_rapm", None)),
                        "ms_def_rapm":       _f(getattr(r, "ms_def_rapm", None)),
                        "ms_toi_minutes":    _f(getattr(r, "ms_toi_minutes", None)),
                        "shrink_weight_ss":  _f(getattr(r, "shrink_weight_ss", None)),
                    })
                out.append(rec)
            return out

        # Default sort is the shrunk value when available; falls back to raw
        # single-season if the shrinkage step hasn't been run yet.
        total_sort = "shrunk_total_rapm" if has_shrunk else "total_rapm"
        off_sort   = "shrunk_off_rapm"   if has_shrunk else "offensive_rapm"
        def_sort   = "shrunk_def_rapm"   if has_shrunk else "defensive_rapm"

        return {
            "trained": True,
            "n_players": int(len(df)),
            "shrinkage_applied": bool(has_shrunk),
            "shrinkage_K": int(df["shrink_K"].iloc[0]) if has_shrunk and "shrink_K" in df.columns else None,
            "meta": meta,
            "top_total": _to_records(df.sort_values(total_sort, ascending=False).head(25)),
            "top_offensive": _to_records(df.sort_values(off_sort, ascending=False).head(25)),
            # Defensive RAPM in our doubled-column encoding: POSITIVE = better defender
            # (we negate the raw coef inside training so positive = suppresses xGA).
            "top_defensive": _to_records(df.sort_values(def_sort, ascending=False).head(25)),
            "full_dataset": _to_records(df.sort_values(total_sort, ascending=False)),
        }

    return jsonify(cached_ttl("rapm_leaders", 86400, _fetch))


RAPM_BAYES_CSV_PATH = os.path.join(os.path.dirname(__file__), "model", "rapm_bayesian_results.csv")
RAPM_BAYES_META_PATH = os.path.join(os.path.dirname(__file__), "model", "rapm_bayesian_meta.json")


@app.route("/api/rapm-bayesian-leaders")
def api_rapm_bayesian_leaders():
    """Return top 25 by total Bayesian (prior-informed) RAPM. Experimental
    parallel model — not wired into the composite rating. 24h cache."""
    def _fetch():
        if not os.path.exists(RAPM_BAYES_CSV_PATH):
            return {
                "trained": False,
                "message": ("Bayesian RAPM has not been trained yet. "
                            "Run: python3 model/train_rapm_bayesian.py"),
            }
        try:
            df = pd.read_csv(RAPM_BAYES_CSV_PATH)
        except Exception as e:
            return {"trained": False, "error": str(e)}

        meta = {}
        if os.path.exists(RAPM_BAYES_META_PATH):
            try:
                with open(RAPM_BAYES_META_PATH) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        def _to_records(sub):
            return [
                {
                    "player_id": int(r.player_id) if not pd.isna(r.player_id) else None,
                    "player_name": str(r.player_name),
                    "team": str(r.team) if not pd.isna(r.team) else "",
                    "season": str(r.season) if not pd.isna(r.season) else "",
                    "total_rapm_bayes": float(r.total_rapm_bayes),
                    "offensive_rapm_bayes": float(r.offensive_rapm_bayes),
                    "defensive_rapm_bayes": float(r.defensive_rapm_bayes),
                    "off_prior": float(r.off_prior),
                    "def_prior": float(r.def_prior),
                    "ixg_60": float(r.ixg_60),
                    "onice_xga_60": float(r.onice_xga_60),
                    "toi_minutes": float(r.toi_minutes),
                    "shift_count": int(r.shift_count),
                }
                for r in sub.itertuples(index=False)
            ]

        return {
            "trained": True,
            "experimental": True,
            "n_players": int(len(df)),
            "meta": meta,
            "top_total": _to_records(
                df.sort_values("total_rapm_bayes", ascending=False).head(25)),
            "top_offensive": _to_records(
                df.sort_values("offensive_rapm_bayes", ascending=False).head(25)),
            "top_defensive": _to_records(
                df.sort_values("defensive_rapm_bayes", ascending=False).head(25)),
            "full_dataset": _to_records(
                df.sort_values("total_rapm_bayes", ascending=False)),
        }

    return jsonify(cached_ttl("rapm_bayesian_leaders", 86400, _fetch))


# Composite endpoint serves the SINGLE-SEASON 25-26 build for the Players section.
# The 16-season `composite_ratings.csv` and the simulator-targeted
# `composite_ratings_sim.csv` are unchanged.
COMPOSITE_CSV_PATH = os.path.join(os.path.dirname(__file__), "model",
                                    "composite_ratings_single_season.csv")
COMPOSITE_META_PATH = os.path.join(os.path.dirname(__file__), "model",
                                     "composite_meta_single_season.json")


# ===========================================================================
# Phase 4 — Simulation endpoints
# ===========================================================================
TEAM_STRENGTH_PATH = os.path.join(os.path.dirname(__file__), "model", "team_strength.csv")
SEASON_SIM_PATH = os.path.join(os.path.dirname(__file__), "model", "season_simulation_results.csv")
PLAYOFF_PROJ_PATH = os.path.join(os.path.dirname(__file__), "model", "playoff_projection_now.csv")


def _load_sim_engine():
    """Lazy-import the simulate module + helpers (heavy imports cached at module level)."""
    import sys as _sys
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if model_dir not in _sys.path:
        _sys.path.insert(0, model_dir)
    import simulate as _simulate
    import build_team_strength as _bts
    return _simulate, _bts


@app.route("/api/team-strength")
def api_team_strength():
    """Return team_strength.csv (32 teams) sorted by team_strength desc."""
    def _fetch():
        if not os.path.exists(TEAM_STRENGTH_PATH):
            return {"built": False,
                    "message": "team_strength.csv not built. Run: "
                               "python3 model/build_team_strength.py"}
        df = pd.read_csv(TEAM_STRENGTH_PATH).sort_values("team_strength", ascending=False)
        return {"built": True, "n_teams": int(len(df)),
                "teams": df.to_dict(orient="records")}
    return jsonify(cached_ttl("team_strength", 21600, _fetch))


@app.route("/api/simulate-game")
def api_simulate_game():
    """Single-game simulation. Query params: home, away, n_sims (default 5000).
    Returns expected score, win probabilities, OT/SO probabilities, and a
    sample simulated result for illustration."""
    home = (request.args.get("home") or "").upper().strip()
    away = (request.args.get("away") or "").upper().strip()
    n_sims = max(100, min(50000, int(request.args.get("n_sims") or 5000)))
    if not home or not away:
        return jsonify({"error": "home and away query params required"}), 400
    if home == away:
        return jsonify({"error": "home and away must differ"}), 400

    simulate, _ = _load_sim_engine()
    ts = simulate.load_team_strengths()
    if home not in ts or away not in ts:
        return jsonify({"error": f"unknown team(s): home={home} away={away}",
                        "valid_teams": sorted(ts.keys())}), 400

    import random
    # Run a full simulation loop ourselves so we can collect the per-game goal
    # distributions for histogram rendering (the win_probability helper only
    # returns aggregates).
    rng = random.Random(42)
    HIST_MAX = 10   # bin labels 0..9, plus a "10+" overflow bucket
    home_dist = [0] * (HIST_MAX + 1)
    away_dist = [0] * (HIST_MAX + 1)
    h_wins = 0; ot_count = 0; so_count = 0
    h_sum = 0.0; a_sum = 0.0
    for _ in range(n_sims):
        g = simulate.simulate_game(home, away, ts, rng)
        hs, as_ = g["home_score"], g["away_score"]
        home_dist[min(hs, HIST_MAX)] += 1
        away_dist[min(as_, HIST_MAX)] += 1
        h_wins += int(g["winner"] == home)
        ot_count += int(g["ot"])
        so_count += int(g["so"])
        h_sum += hs; a_sum += as_
    sample = simulate.simulate_game(home, away, ts, rng=random.Random(7))
    xgh, xga = simulate.expected_goals(home, away, ts)
    hist_labels = [str(i) for i in range(HIST_MAX)] + [f"{HIST_MAX}+"]
    return jsonify({
        "home": home, "away": away, "n_sims": n_sims,
        "expected_goals_home": round(xgh, 3),
        "expected_goals_away": round(xga, 3),
        "win_pct": {
            "home": round(h_wins / n_sims, 4),
            "away": round(1 - h_wins / n_sims, 4),
        },
        "expected_score": {
            "home": round(h_sum / n_sims, 2),
            "away": round(a_sum / n_sims, 2),
        },
        "ot_pct": round(ot_count / n_sims, 4),
        "so_pct": round(so_count / n_sims, 4),
        # Regulation result percentage = 1 - (ot OR so happened); OT-no-SO and SO
        # are mutually exclusive in our model.
        "regulation_pct": round(1 - ot_count / n_sims, 4),
        "ot_only_pct": round((ot_count - so_count) / n_sims, 4),
        "sample_result": {
            "home_score": sample["home_score"],
            "away_score": sample["away_score"],
            "winner": sample["winner"],
            "ot": sample["ot"], "so": sample["so"],
        },
        "goal_distribution": {
            "labels": hist_labels,
            "home_counts": home_dist,
            "away_counts": away_dist,
        },
        "calibration": {
            "league_gpg": simulate.LEAGUE_GPG,
            "alpha": simulate.ALPHA,
            "home_ice": simulate.HOME_ICE,
        },
    })


@app.route("/api/team-strength-custom", methods=["POST"])
def api_team_strength_custom():
    """Recalculate team_strength for one team given an optional roster override.

    Request JSON: {
      "team": "EDM",
      "roster_override": [
         {"player_id": 8478402, "position": "C", "roster_role": "F"},
         ...
      ]
    }
    If roster_override is omitted/null, falls back to the default NHL API roster.

    Response includes the recomputed team_strength PLUS the default for
    comparison and a delta breakdown."""
    body = request.get_json(silent=True) or {}
    team = (body.get("team") or "").upper().strip()
    override = body.get("roster_override")
    if not team:
        return jsonify({"error": "team field required"}), 400

    simulate, bts = _load_sim_engine()
    comp_lookup = bts.load_composite_lookup()
    goalies_csv = bts.load_goalies_csv()

    # Default strength (cached on disk from the most-recent build)
    default_df = pd.read_csv(TEAM_STRENGTH_PATH)
    default_row = default_df[default_df["team"] == team]
    if default_row.empty:
        return jsonify({"error": f"unknown team: {team}"}), 400
    default_strength = default_row.iloc[0].to_dict()

    # Build roster_override into the shape the engine expects, then recompute
    # the RAW (pre-z-normalized) component values.
    if override:
        for p in override:
            p["player_id"] = int(p["player_id"])
            p["position"] = p.get("position", "F")
            p["roster_role"] = p.get("roster_role", p["position"] if p["position"] in ("D","G") else "F")
            p.setdefault("first_name", "")
            p.setdefault("last_name", "")
        roster = override
    else:
        roster = bts.fetch_roster(team)

    raw = bts.compute_team_strength(team, comp_lookup, goalies_csv,
                                     roster_override=roster)
    # To match the saved z-scored team_strength we'd need full-league context.
    # Simplest: compute every team's raw values, re-z-normalize, return THIS
    # team's z-score under the override.
    rows = []
    for t in bts.TEAMS:
        if t == team:
            rows.append(raw)
        else:
            rows.append(bts.compute_team_strength(t, comp_lookup, goalies_csv))
    z_df = bts._z_normalize_components(pd.DataFrame(rows))
    new_row = z_df[z_df["team"] == team].iloc[0].to_dict()

    return jsonify({
        "team": team,
        "default": {k: default_strength.get(k) for k in default_strength
                    if k not in ("error",)},
        "custom":  {k: new_row.get(k) for k in new_row if k not in ("error",)},
        "delta_team_strength": round(new_row["team_strength"] - default_strength["team_strength"], 4),
        "delta_top6_F": round(new_row["top6_forward_strength"]
                              - default_strength["top6_forward_strength"], 4),
        "delta_top_pair_D": round(new_row["top_pair_d_strength"]
                                  - default_strength["top_pair_d_strength"], 4),
        "delta_goalie": round(new_row["goalie_strength"]
                              - default_strength["goalie_strength"], 4),
        "n_override_players": len(roster),
    })


@app.route("/api/sim-team-roster/<team>")
def api_sim_team_roster(team):
    """Helper for the Lineup Editor — returns the team's 25-26 active roster
    with each player's composite_war, expected_gp_share, and the tier role
    they'd be assigned given the current roster (top line F, 2nd line F, etc.).
    Plus the two top goalies by GP from MoneyPuck."""
    team = team.upper().strip()
    simulate, bts = _load_sim_engine()
    if team not in bts.TEAMS:
        return jsonify({"error": f"unknown team: {team}"}), 400
    comp_lookup = bts.load_composite_lookup()
    toi_lookup = bts.load_toi_lookup()
    goalies_csv = bts.load_goalies_csv()

    def _fetch():
        roster = bts.fetch_roster(team)
        if not roster:
            return {"team": team, "error": "no roster"}
        fwds_raw = [p for p in roster if p["roster_role"] == "F"]
        dfns_raw = [p for p in roster if p["roster_role"] == "D"]
        glxs_raw = [p for p in roster if p["roster_role"] == "G"]
        fwds = bts._attach_war(fwds_raw, comp_lookup, toi_lookup)
        dfns = bts._attach_war(dfns_raw, comp_lookup, toi_lookup)
        # Sort by historical avg ice time — coaches deploy by TOI, not WAR.
        fwds.sort(key=lambda p: p["avg_toi_per_game"], reverse=True)
        dfns.sort(key=lambda p: p["avg_toi_per_game"], reverse=True)

        # Tier labels by position
        F_TIERS_LABEL = ["Top Line", "Top Line", "Top Line",
                         "Second Line", "Second Line", "Second Line",
                         "Third Line", "Third Line", "Third Line",
                         "Fourth Line", "Fourth Line", "Fourth Line"]
        D_TIERS_LABEL = ["Top Pair", "Top Pair",
                         "Second Pair", "Second Pair",
                         "Third Pair", "Third Pair"]

        out_fwds = []
        for i, p in enumerate(fwds[:12]):
            out_fwds.append({
                **{k: p[k] for k in ("player_id","first_name","last_name","position",
                                       "composite_war","expected_gp_share","rated")},
                "tier_label": F_TIERS_LABEL[i] if i < 12 else "scratched",
                "tier_minutes": simulate.__dict__ and bts.FWD_TIERS[i],
            })
        scratched_fwd = []
        for p in fwds[12:]:
            scratched_fwd.append({
                **{k: p[k] for k in ("player_id","first_name","last_name","position",
                                       "composite_war","expected_gp_share","rated")},
                "tier_label": "Scratched",
                "tier_minutes": 0,
            })
        out_dfns = []
        for i, p in enumerate(dfns[:6]):
            out_dfns.append({
                **{k: p[k] for k in ("player_id","first_name","last_name","position",
                                       "composite_war","expected_gp_share","rated")},
                "tier_label": D_TIERS_LABEL[i],
                "tier_minutes": bts.DEF_TIERS[i],
            })
        scratched_def = []
        for p in dfns[6:]:
            scratched_def.append({
                **{k: p[k] for k in ("player_id","first_name","last_name","position",
                                       "composite_war","expected_gp_share","rated")},
                "tier_label": "Scratched",
                "tier_minutes": 0,
            })

        # Goalies — top 2 by GP from MoneyPuck restricted to rostered
        rostered_g_ids = {g["player_id"] for g in glxs_raw}
        g_rows = goalies_csv[goalies_csv["playerId"].isin(rostered_g_ids)] \
                    .sort_values("games_played", ascending=False).head(2)
        total_gp = float(g_rows["games_played"].sum()) if not g_rows.empty else 0.0
        goalies_out = []
        for i, gr in enumerate(g_rows.itertuples(index=False)):
            goalies_out.append({
                "player_id": int(gr.playerId),
                "name": gr.name,
                "games_played": int(gr.games_played),
                "gp_share": round(gr.games_played / total_gp, 3) if total_gp > 0 else 0.0,
                "gsaa": round(float(gr.gsaa), 2),
                "goalie_war": round(float(gr.goalie_war), 3),
                "role": "Starter" if i == 0 else "Backup",
            })

        return {
            "team": team,
            "forwards_active": out_fwds,
            "forwards_scratched": scratched_fwd,
            "defense_active": out_dfns,
            "defense_scratched": scratched_def,
            "goalies": goalies_out,
        }

    return jsonify(cached_ttl(f"sim_roster:{team}", 86400, _fetch))


@app.route("/api/simulate-season-custom", methods=["POST"])
def api_simulate_season_custom():
    """Phase 5.1 season simulator.

    POST body:
        {
          "team": "EDM",
          "lineup": {
             "forwards": [{player_id, position, first_name, last_name}, … 12 slots …],
             "defense":  [… 6 slots …],
             "goalies":  [… up to 2 …]
          },
          "n_sims": 250,
          "compare_to_default": true|false
        }

    Returns the simulator result for the custom lineup (regular_season /
    player_stats / playoff_outcome). When `compare_to_default` is true the
    response also includes a `default` block with the same shape, computed
    from the team's NHL-API default roster, and a per-key `delta_*` summary
    for the regular-season block.
    """
    body = request.get_json(silent=True) or {}
    team = (body.get("team") or "").upper().strip()
    lineup = body.get("lineup") or {}
    n_sims = max(10, min(1000, int(body.get("n_sims") or 250)))
    compare = bool(body.get("compare_to_default"))
    if not team:
        return jsonify({"error": "team field required"}), 400
    if not lineup.get("forwards") and not lineup.get("defense"):
        return jsonify({"error": "lineup field required"}), 400

    # Lazy import — these are heavy modules
    simulate, bts = _load_sim_engine()
    import sys as _sys
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if model_dir not in _sys.path:
        _sys.path.insert(0, model_dir)
    import simulate_season as ss

    # First compute the custom team's strength using the existing endpoint logic.
    comp_lookup = bts.load_composite_lookup()
    goalies_csv = bts.load_goalies_csv()
    roster_override = []
    for slot_kind, key in (("F", "forwards"), ("D", "defense"), ("G", "goalies")):
        for p in lineup.get(key, []) or []:
            if not p or not p.get("player_id"):
                continue
            roster_override.append({
                "player_id": int(p["player_id"]),
                "position": p.get("position") or slot_kind,
                "roster_role": slot_kind,
                "first_name": p.get("first_name", ""),
                "last_name":  p.get("last_name", ""),
            })
    raw = bts.compute_team_strength(team, comp_lookup, goalies_csv,
                                     roster_override=roster_override)
    # z-normalise against league for the team strength used by the sim engine
    rows = []
    for t in bts.TEAMS:
        rows.append(raw if t == team else
                    bts.compute_team_strength(t, comp_lookup, goalies_csv))
    z_df = bts._z_normalize_components(pd.DataFrame(rows))
    custom_ts = float(z_df[z_df["team"] == team].iloc[0]["team_strength"])

    # Run the custom simulation
    custom_result = ss.monte_carlo_custom_season(team, lineup, custom_ts,
                                                  n_sims=n_sims)

    payload = {
        "team": team,
        "n_sims": n_sims,
        "custom_team_strength": round(custom_ts, 4),
        "custom": custom_result,
    }

    if compare:
        # Build the default roster from the NHL API and z-normalised default strength
        default_roster = bts.fetch_roster(team)
        fwd = sorted(bts._attach_war([p for p in default_roster
                                       if p["roster_role"] == "F"], comp_lookup),
                     key=lambda p: p["effective_war"], reverse=True)
        dfn = sorted(bts._attach_war([p for p in default_roster
                                       if p["roster_role"] == "D"], comp_lookup),
                     key=lambda p: p["effective_war"], reverse=True)
        glx = [p for p in default_roster if p["roster_role"] == "G"][:2]
        default_lineup = {
            "forwards": fwd[:12] + [None] * max(0, 12 - len(fwd)),
            "defense":  dfn[:6]  + [None] * max(0, 6 - len(dfn)),
            "goalies":  glx,
        }
        default_ts = pd.read_csv(TEAM_STRENGTH_PATH).query("team == @team")
        default_ts_val = float(default_ts.iloc[0]["team_strength"]) if not default_ts.empty else 0.0
        default_result = ss.monte_carlo_custom_season(team, default_lineup,
                                                      default_ts_val,
                                                      n_sims=n_sims)
        payload["default_team_strength"] = round(default_ts_val, 4)
        payload["default"] = default_result
        rs_c = custom_result["regular_season"]
        rs_d = default_result["regular_season"]
        payload["delta_regular_season"] = {
            "wins":   round(rs_c["avg_wins"]   - rs_d["avg_wins"], 1),
            "losses": round(rs_c["avg_losses"] - rs_d["avg_losses"], 1),
            "otl":    round(rs_c["avg_otl"]    - rs_d["avg_otl"], 1),
            "points": round(rs_c["avg_points"] - rs_d["avg_points"], 1),
            "gf":     round(rs_c["avg_gf"]     - rs_d["avg_gf"], 1),
            "ga":     round(rs_c["avg_ga"]     - rs_d["avg_ga"], 1),
            "playoff_pct":
                round(rs_c["playoff_pct"] - rs_d["playoff_pct"], 4),
        }

    return jsonify(payload)


@app.route("/api/sim-player-pool")
def api_sim_player_pool():
    """Search-friendly list of all rated players for the Lineup Editor swap
    dropdown. Returns {player_id, name, position, composite_war,
    expected_gp_share, team}. Goalies are included from the MoneyPuck
    25-26 goalies CSV with their `goalie_war`/`games_played` surfaced for
    the goalie cell renderer (and aliased into composite_war /
    expected_gp_share for the generic swap result row)."""
    def _fetch():
        path = os.path.join(os.path.dirname(__file__), "model",
                            "composite_ratings_sim.csv")
        if not os.path.exists(path):
            return {"built": False, "message": "composite_ratings_sim.csv not found"}
        df = pd.read_csv(path)
        df = df[df["sample_size_flag"] == "ok"].sort_values("composite_war", ascending=False)
        recs = [{
            "player_id": int(r.player_id),
            "name": str(r.player_name),
            "position": str(r.position),
            "team": str(r.team),
            "composite_war": float(r.composite_war),
            "expected_gp_share": float(r.expected_gp_share),
        } for r in df.itertuples(index=False)]

        # Append goalies from MoneyPuck 25-26 so the Lineup Editor's goalie
        # swap dropdown is populated. (composite_ratings_sim.csv is skater-
        # only.) Use the same loader the team_strength builder uses.
        try:
            _simulate, bts = _load_sim_engine()
            g_df = bts.load_goalies_csv()
            g_df = g_df.sort_values("goalie_war", ascending=False)
            for r in g_df.itertuples(index=False):
                gp = int(r.games_played) if not pd.isna(r.games_played) else 0
                gw = float(r.goalie_war) if not pd.isna(r.goalie_war) else 0.0
                recs.append({
                    "player_id": int(r.playerId),
                    "name": str(r.name),
                    "position": "G",
                    "team": str(r.team),
                    # Generic fields used by the swap result row UI
                    "composite_war": gw,
                    "expected_gp_share": (gp / 82.0) if gp > 0 else 0.0,
                    # Goalie-specific fields used by the goalie cell renderer
                    "goalie_war": gw,
                    "games_played": gp,
                    "gsaa": round(float(r.gsaa), 2) if not pd.isna(r.gsaa) else 0.0,
                })
        except Exception as e:
            # Don't fail the whole pool if goalies CSV is unavailable;
            # skater swaps still work.
            print(f"[api_sim_player_pool] WARN: goalies not loaded — {e}",
                  flush=True)

        return {"built": True, "n_players": len(recs), "players": recs}
    return jsonify(cached_ttl("sim_player_pool", 86400, _fetch))


def _rerun_playoffs_now_mc(n_sims: int = 10000) -> dict:
    """Re-fetch live playoff state and run a fresh playoffs-only Monte Carlo
    with a random seed. Persists the new results to PLAYOFF_PROJ_PATH and
    returns the in-memory dict."""
    import random as _r
    import time as _t
    simulate, _ = _load_sim_engine()
    # Import the driver helpers (live fetchers) — defined in run_season_simulation.
    import sys as _sys
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if model_dir not in _sys.path:
        _sys.path.insert(0, model_dir)
    import run_season_simulation as _drv

    ts = simulate.load_team_strengths()
    cf_pairings, cf_state = _drv.fetch_playoff_state()
    seed = _r.randint(0, 10**9)
    rng = _r.Random(seed)
    counts = {t: {"conf_final": 0, "final": 0, "cup": 0}
              for t in simulate.TEAM_TO_DIVISION}
    # Teams alive in the CF round get conf_final=n_sims by definition.
    alive_now = []
    for conf in ("east", "west"):
        for (a, b) in cf_pairings.get(conf, []):
            counts[a]["conf_final"] = n_sims
            counts[b]["conf_final"] = n_sims
            alive_now.extend([a, b])

    for _ in range(n_sims):
        conf_winners = {}
        for conf in ("east", "west"):
            if not cf_pairings.get(conf):
                continue
            (a, b) = cf_pairings[conf][0]
            init = cf_state.get((a, b), {"a_wins": 0, "b_wins": 0})
            higher = a if ts.get(a, 0) >= ts.get(b, 0) else b
            res = simulate.simulate_series(
                a, b, ts, higher_seed=higher,
                a_wins=init["a_wins"], b_wins=init["b_wins"], rng=rng,
            )
            conf_winners[conf] = res["winner"]
            counts[res["winner"]]["final"] += 1
        if "east" in conf_winners and "west" in conf_winners:
            e, w = conf_winners["east"], conf_winners["west"]
            higher = e if ts.get(e, 0) >= ts.get(w, 0) else w
            cup = simulate.simulate_series(e, w, ts, higher_seed=higher, rng=rng)
            counts[cup["winner"]]["cup"] += 1

    rows = []
    for t, c in counts.items():
        if c["conf_final"] == 0 and c["final"] == 0 and c["cup"] == 0:
            continue
        rows.append({
            "team": t,
            "in_round": "R3" if c["conf_final"] > 0 else "eliminated",
            "conf_final_pct": round(c["conf_final"] / n_sims, 4),
            "final_pct": round(c["final"] / n_sims, 4),
            "cup_pct": round(c["cup"] / n_sims, 4),
        })
    df = pd.DataFrame(rows).sort_values("cup_pct", ascending=False).reset_index(drop=True)
    df.to_csv(PLAYOFF_PROJ_PATH, index=False)
    return {
        "built": True, "n_teams": int(len(df)),
        "n_sims": n_sims,
        "seed": seed,
        "rebuilt_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        "teams": df.to_dict(orient="records"),
        "note": ("In-progress 2025-26 playoffs (conf finals). "
                 "R1 and R2 results locked; only R3 + R4 simulated."),
    }


@app.route("/api/season-projections")
def api_season_projections():
    """Return both Monte Carlo outputs: the full-season 25-26 backtest and the
    live playoffs-only forecast. 6h cache by default. Pass ?force=true to
    bypass the cache AND re-run the playoffs-only Monte Carlo with a fresh
    random seed; results will vary slightly between successive force-runs
    within MC statistical noise."""
    force = (request.args.get("force") or "").lower() in ("1", "true", "yes")

    def _fetch():
        out = {"backtest": {"built": False}, "playoffs_now": {"built": False}}
        if os.path.exists(SEASON_SIM_PATH):
            df = pd.read_csv(SEASON_SIM_PATH).sort_values("cup_pct", ascending=False)
            out["backtest"] = {
                "built": True, "n_teams": int(len(df)),
                "n_sims": 10000,
                "teams": df.to_dict(orient="records"),
            }
        if force:
            # Fresh Monte Carlo with a random seed.
            out["playoffs_now"] = _rerun_playoffs_now_mc(n_sims=10000)
        elif os.path.exists(PLAYOFF_PROJ_PATH):
            df = pd.read_csv(PLAYOFF_PROJ_PATH).sort_values("cup_pct", ascending=False)
            out["playoffs_now"] = {
                "built": True, "n_teams": int(len(df)),
                "n_sims": 10000,
                "teams": df.to_dict(orient="records"),
                "note": ("In-progress 2025-26 playoffs (conf finals). "
                         "R1 and R2 results locked; only R3 + R4 simulated."),
            }
        return out

    if force:
        # Bypass the cache entirely — the user wants a fresh roll.
        return jsonify(_fetch())
    return jsonify(cached_ttl("season_projections", 21600, _fetch))
# ===========================================================================
# END Phase 4 endpoints
# ===========================================================================


@app.route("/api/composite-ratings")
def api_composite_ratings():
    """Composite player ratings (Phase 2.5) — top 50 + full dataset. 24h cache.
    The composite rating is the primary player-strength metric feeding the simulation."""
    def _fetch():
        if not os.path.exists(COMPOSITE_CSV_PATH):
            return {
                "built": False,
                "message": "Composite ratings not built yet. Run: python3 model/build_composite.py",
            }
        try:
            df = pd.read_csv(COMPOSITE_CSV_PATH)
        except Exception as e:
            return {"built": False, "error": str(e)}

        meta = {}
        if os.path.exists(COMPOSITE_META_PATH):
            try:
                with open(COMPOSITE_META_PATH) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        def _records(sub):
            return [
                {
                    "player_id": int(r.player_id) if not pd.isna(r.player_id) else None,
                    "player_name": str(r.player_name),
                    "team": str(r.team) if not pd.isna(r.team) else "",
                    "position": str(r.position) if not pd.isna(r.position) else "",
                    "rapm_component": float(r.rapm_component),
                    "individual_component": float(r.individual_component),
                    "relative_xgf_component": float(r.relative_xgf_component),
                    "playmaking_component": float(r.playmaking_component),
                    "power_play_component": float(r.power_play_component),
                    "penalty_kill_component": float(r.penalty_kill_component),
                    "composite_rating": float(r.composite_rating),
                    "toi_minutes": float(r.toi_minutes),
                    "sample_size_flag": str(r.sample_size_flag),
                }
                for r in sub.itertuples(index=False)
            ]

        # Top 50 is taken from the qualified pool only (small-sample players are
        # excluded from the leaderboard, included in the full dataset).
        qualified = df[df["sample_size_flag"] == "ok"]
        return {
            "built": True,
            "n_players": int(len(df)),
            "n_qualified": int(len(qualified)),
            "meta": meta,
            "top_50": _records(qualified.sort_values("composite_rating", ascending=False).head(50)),
            "full_dataset": _records(df),
        }

    return jsonify(cached_ttl("composite_ratings", 86400, _fetch))


@app.route("/api/xg-model-stats")
def api_xg_model_stats():
    """Return AUC, log loss, training info, and feature importances for the xG model."""
    if not os.path.exists(XG_META_PATH):
        return jsonify({
            "trained": False,
            "message": "xG model has not been trained yet. Run: python3 model/train_xg.py",
        }), 200
    try:
        with open(XG_META_PATH) as f:
            meta = json.load(f)
        meta["model_loaded"] = os.path.exists(XG_MODEL_PATH)
        meta["trained"] = True
        return jsonify(meta)
    except Exception as e:
        log.error("xg-model-stats failed: %s", e)
        return jsonify({"trained": False, "error": str(e)}), 500


# Kick off the daily league-wide contract refresh. Runs under both `python
# app.py` and gunicorn (module import), single-flight-guarded across workers.
_start_contract_refresher()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run on (default 5000). macOS tip: disable AirPlay Receiver if 5000 is taken.")
    parser.add_argument("--no-reload", action="store_true",
                        help="Disable auto-reload (for stable demos / production-like runs).")
    args = parser.parse_args()
    print(f"Starting NHL Morning Brief on http://localhost:{args.port}")
    # debug=True turns on the Werkzeug auto-reloader (restarts on .py changes)
    # plus templates_auto_reload + browser cache disabled (set above) gives a
    # full live-edit workflow: save file → refresh browser → see change.
    app.run(host="0.0.0.0", port=args.port, debug=not args.no_reload, use_reloader=not args.no_reload)
