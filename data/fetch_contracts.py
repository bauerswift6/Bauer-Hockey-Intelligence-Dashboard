#!/usr/bin/env python3
"""
Fetch league-wide NHL contract data and write data/contracts_current.json.

Source: capwages.com (a CapFriendly successor). The NHL's own public API does
NOT expose any salary / cap-hit data, so contracts have to come from a cap
tracker. capwages is a Next.js site whose page data is available as clean JSON
at /_next/data/<buildId>/... — no Cloudflare challenge (unlike PuckPedia /
Spotrac), and it is kept current (new signings appear within a day).

This runs on a schedule (see the scheduler config). It is deliberately
defensive: if anything fails or the decode looks wrong, it leaves the existing
contracts_current.json untouched and exits non-zero, so the dashboard keeps
serving the last-good snapshot rather than crashing or going blank.

Bulk payload decode (the /players/active page ships each player as a positional
array; column indices verified empirically against authoritative per-player
pages — Matthews $13.25M, McDavid $12.5M, Draisaitl $14.0M, Makar $9.0M):

    [0]  name "Last, First"        [8]  age
    [2]  team tricode              [18] current cap hit  (value / 10 == $M)
    [3]  position                  [29] expiry year mod 100  (2000 + v)
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "contracts_current.json")

CAPWAGES = "https://capwages.com"
NHL_SEARCH = "https://search.d3.nhle.com/api/v1/search/player"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
    ),
    "Accept": "application/json,text/html",
}

# Column indices in the capwages /players/active positional array.
COL_NAME, COL_TEAM, COL_POS, COL_AGE = 0, 2, 3, 8
COL_CAP_HIT, COL_EXPIRY_YY = 18, 29

# Known players used to validate the cap-hit decode each run. If capwages ever
# reorders its columns, these assertions fail loudly and we keep the old file.
VALIDATION = {
    # slug: expected current cap hit in $M
    "connor-mcdavid": 12.5,
    "leon-draisaitl": 14.0,
    "cale-makar": 9.0,
}
VALIDATION_TOL = 0.001  # $M


def log(msg: str) -> None:
    print(f"[fetch_contracts] {msg}", file=sys.stderr, flush=True)


def fetch_build_id() -> str:
    """Scrape the current Next.js buildId from the capwages homepage."""
    r = requests.get(CAPWAGES + "/", headers=HEADERS, timeout=20)
    r.raise_for_status()
    m = re.search(r'"buildId":"([^"]+)"', r.text)
    if not m:
        raise RuntimeError("could not locate buildId on capwages homepage")
    return m.group(1)


def _data_url(build_id: str, route: str) -> str:
    return f"{CAPWAGES}/_next/data/{build_id}/{route}.json"


def fetch_active_players(build_id: str) -> list[list]:
    """Return the raw positional-array roster for all active players."""
    r = requests.get(
        _data_url(build_id, "players/active"),
        headers={**HEADERS, "x-nextjs-data": "1"},
        timeout=30,
    )
    r.raise_for_status()
    arr = r.json().get("pageProps", {}).get("playersArray")
    if not isinstance(arr, list) or not arr:
        raise RuntimeError("active players payload missing/empty")
    return arr


def _flip_name(name: str) -> str:
    """"Draisaitl, Leon" -> "Leon Draisaitl"; leave already-flipped names alone."""
    if "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name.strip()


def _norm_position(pos: str) -> str:
    """Collapse capwages position strings to a single canonical token.

    The app treats "G" as goalie and "D" as defenseman; forwards keep their
    listed primary token (C/LW/RW). "RD, LD" -> "D", "RW, LW" -> "RW"."""
    if not pos:
        return ""
    primary = pos.split(",")[0].strip().upper()
    if "G" in primary:
        return "G"
    if "D" in primary:
        return "D"
    return primary


def decode_row(row: list) -> dict | None:
    """Decode one positional array into a contract dict, or None to skip."""
    try:
        name = _flip_name(str(row[COL_NAME]))
        team = str(row[COL_TEAM] or "").strip().upper()
        pos = _norm_position(str(row[COL_POS] or ""))
        cap_raw = row[COL_CAP_HIT]
        yy = row[COL_EXPIRY_YY]
        age = row[COL_AGE]
    except (IndexError, TypeError):
        return None

    if not name or not team or not isinstance(cap_raw, (int, float)):
        return None

    cap_hit = round(float(cap_raw) / 10.0, 4)  # value is $M * 10
    if cap_hit <= 0:
        return None  # unsigned RFA/UFA — no active contract

    if not isinstance(yy, (int, float)) or yy <= 0:
        return None
    expiry_year = 2000 + int(yy)

    return {
        "name": name,
        "team": team,
        "position": pos,
        "cap_hit": cap_hit,
        "expiry_year": expiry_year,
        "age": int(age) if isinstance(age, (int, float)) and age else None,
    }


def validate_decode(build_id: str) -> None:
    """Cross-check a few decoded cap hits against authoritative per-player pages."""
    for slug, expected in VALIDATION.items():
        r = requests.get(
            _data_url(build_id, f"players/{slug}"),
            headers={**HEADERS, "x-nextjs-data": "1"},
            timeout=20,
        )
        r.raise_for_status()
        player = r.json().get("pageProps", {}).get("player", {})
        got = None
        for c in player.get("contracts", []):
            for d in c.get("details", []):
                cap = re.sub(r"[^0-9]", "", str(d.get("capHit", "")))
                if cap:
                    got = int(cap) / 1_000_000
                    break
            if got is not None:
                break
        if got is None or abs(got - expected) > VALIDATION_TOL:
            raise RuntimeError(
                f"validation failed for {slug}: expected {expected}M, "
                f"detail page shows {got}M"
            )
    log(f"validation ok ({len(VALIDATION)} players)")


def _search_player_id(name: str, team: str) -> int | None:
    try:
        r = requests.get(
            NHL_SEARCH,
            params={"culture": "en-us", "limit": 8, "q": name, "active": "true"},
            headers=HEADERS,
            timeout=8,
        )
        r.raise_for_status()
        data = r.json() or []
    except Exception:
        return None
    for hit in data:
        hit_team = (hit.get("teamAbbrev") or hit.get("lastTeamAbbrev") or "")
        if str(hit_team).upper() == team.upper():
            try:
                return int(hit.get("playerId") or hit.get("id") or 0) or None
            except (TypeError, ValueError):
                pass
    if data:
        try:
            return int(data[0].get("playerId") or data[0].get("id") or 0) or None
        except (TypeError, ValueError):
            return None
    return None


def resolve_player_ids(players: list[dict], prev_by_name: dict[str, int]) -> None:
    """Fill each player's `playerId` in place.

    Re-uses previously resolved IDs (keyed by name) so a normal refresh only
    hits the NHL search API for players it has never seen before."""
    todo = []
    for p in players:
        pid = prev_by_name.get(p["name"])
        if pid:
            p["playerId"] = pid
        else:
            todo.append(p)

    if not todo:
        log("all player IDs served from cache")
        return

    log(f"resolving {len(todo)} new player IDs via NHL search")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(_search_player_id, p["name"], p["team"]): p for p in todo}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                p["playerId"] = fut.result()
            except Exception:
                p["playerId"] = None


def load_previous() -> dict:
    try:
        with open(OUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main() -> int:
    started = time.time()
    prev = load_previous()
    prev_by_name = {
        c["name"]: c["playerId"]
        for c in prev.get("contracts", [])
        if c.get("playerId")
    }

    try:
        build_id = fetch_build_id()
        log(f"buildId={build_id}")
        raw = fetch_active_players(build_id)
        log(f"fetched {len(raw)} active player rows")

        validate_decode(build_id)

        contracts = [d for d in (decode_row(r) for r in raw) if d]
        if len(contracts) < 500:
            raise RuntimeError(
                f"decoded only {len(contracts)} contracts — refusing to overwrite"
            )
        log(f"decoded {len(contracts)} contracts")

        resolve_player_ids(contracts, prev_by_name)
    except Exception as e:
        log(f"ERROR: {e} — keeping existing snapshot")
        return 1

    payload = {
        "_meta": {
            "source": "capwages.com",
            "source_url": CAPWAGES + "/players/active",
            "build_id": build_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "count": len(contracts),
            "note": (
                "League-wide active NHL contracts. Cap hits in $M. Decoded from "
                "the capwages bulk roster payload; cross-validated against "
                "per-player pages each run."
            ),
        },
        "contracts": contracts,
    }

    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, OUT_PATH)

    resolved = sum(1 for c in contracts if c.get("playerId"))
    log(
        f"wrote {len(contracts)} contracts ({resolved} with playerId) "
        f"in {time.time() - started:.1f}s -> {OUT_PATH}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
