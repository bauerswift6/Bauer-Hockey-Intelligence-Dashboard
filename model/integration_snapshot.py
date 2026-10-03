"""
integration_snapshot.py
-----------------------
Capture a BEFORE snapshot of the dashboard API (old metrics) for the integration
diagnostic. Starts app.py locally, requests the player + goalie + leaderboard
endpoints, and saves each JSON response to integration_baseline/.

Guardrail compliance: app.py's contract refresher (a daemon started at import,
line ~4529) would run data/fetch_contracts.py and rewrite data/contracts_current.json
+ a lock file when the snapshot is stale (it is, ~840h). We therefore launch with
DISABLE_CONTRACT_REFRESH=1 (app.py's own supported guard) so the app creates/
modifies NO project file. The only endpoint that writes (/api/playoff-projection)
is not requested. We snapshot the project file listing before and after and abort
with a report if any project file (other than integration_baseline/ outputs and
__pycache__ bytecode) changes.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTDIR = os.path.join(ROOT, "integration_baseline")
PORT = 5077
BASE = f"http://127.0.0.1:{PORT}"

PLAYERS = {
    "mcdavid": 8478402, "kucherov": 8476453, "mackinnon": 8477492,
    "brady_tkachuk": 8480801, "mark_stone": 8475913,
    "logan_thompson": 8480313, "jeremy_swayman": 8480280, "jordan_binnington": 8476412,
}
ENDPOINTS = {
    "players_full": "/api/players-full",
    "goalies_full": "/api/goalies-full",
    "goalie_analytics": "/api/goalie-analytics",
    "leaders": "/api/leaders",
    "rapm_leaders": "/api/rapm-leaders",
}


def file_listing():
    out = {}
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__", "node_modules",
                                              "integration_baseline")]
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                st = os.stat(p)
                out[os.path.relpath(p, ROOT)] = (round(st.st_mtime, 3), st.st_size)
            except OSError:
                pass
    return out


def get(path):
    req = urllib.request.Request(BASE + path, headers={"User-Agent": "integration-snapshot"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    before = file_listing()

    env = dict(os.environ, DISABLE_CONTRACT_REFRESH="1")
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "app.py"),
                             "--port", str(PORT), "--no-reload"],
                            cwd=ROOT, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # wait for readiness
        ready = False
        for _ in range(60):
            try:
                urllib.request.urlopen(BASE + "/", timeout=5)
                ready = True; break
            except Exception:
                time.sleep(1)
        if not ready:
            print("APP DID NOT START"); return

        saved = []
        for name, pid in PLAYERS.items():
            try:
                d = get(f"/api/player/{pid}")
                json.dump(d, open(os.path.join(OUTDIR, f"player_{name}_{pid}.json"), "w"), indent=2)
                saved.append(f"player_{name}")
            except Exception as e:
                print("ERR player", name, e)
        for name, path in ENDPOINTS.items():
            try:
                d = get(path)
                json.dump(d, open(os.path.join(OUTDIR, f"{name}.json"), "w"), indent=2)
                saved.append(name)
            except Exception as e:
                print("ERR", name, e)
        print("SAVED", len(saved), "responses:", ", ".join(saved))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()

    after = file_listing()
    changed = [f for f in after if f not in before or after[f] != before.get(f)]
    removed = [f for f in before if f not in after]
    changed = [f for f in changed if not f.startswith("integration_baseline")]
    if changed or removed:
        print("!!! PROJECT FILES CHANGED (excluding integration_baseline, __pycache__):")
        for f in changed: print("   CHANGED/NEW", f)
        for f in removed: print("   REMOVED", f)
    else:
        print("OK: no project file created or modified (besides integration_baseline/ and __pycache__).")


if __name__ == "__main__":
    main()
