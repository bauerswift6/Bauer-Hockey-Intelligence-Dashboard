# NHL Morning Brief

A local NHL analytics dashboard with live scores, advanced stats, and standings.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

> **macOS note:** If port 5000 is already in use (common on macOS due to AirPlay Receiver), go to
> **System Settings → General → AirDrop & Handoff → AirPlay Receiver** and turn it **Off**.
> Then re-run `python app.py`. Alternatively, run on a different port:
> ```bash
> python app.py --port 5001
> ```

## Data Sources

| Section | Source |
|---|---|
| Last Night's Games | NHL Public API |
| Season Leaders | NHL Public API |
| Standings | NHL Public API |
| GAR / xGAR Leaderboards | Evolving Hockey → MoneyPuck (proxy) |
| Team Analytics (xGF%, CF%, HDCF%) | Evolving Hockey → MoneyPuck |
| Goalie Analytics (GSAX) | Evolving Hockey → MoneyPuck |
| Top Lines | Evolving Hockey (subscription required) |

Advanced stats via Evolving Hockey fall back to MoneyPuck CSVs automatically.
If Evolving Hockey data is available, it takes priority.

## Notes

- Advanced stats cache for 6 hours — restart the server to force a refresh.
- Line analytics require an Evolving Hockey subscription; a link is shown instead.
- The dashboard auto-detects playoff vs. regular season game context.
