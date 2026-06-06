/* =========================================================
   NHL Morning Brief — Dashboard JS
   ========================================================= */

let garChart = null;
let xgarChart = null;
let teamSortCol = "xgf_pct";
let teamSortAsc = false;
let teamData = [];

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Only kick off fetches for sections whose DOM containers still exist.
  // (The Players-section restructure removed the GAR/xGAR/Leaders/Lines panels
  // but their fetch functions remain in case other callers want them.)
  if (document.getElementById("games-section")) fetchGames();
  if (document.getElementById("leaders-section")) fetchLeaders();
  if (document.getElementById("standings-section")) fetchStandings();
  if (document.getElementById("gar-chart")) fetchGarLeaders();
  if (document.getElementById("xgar-chart")) fetchXgarLeaders();
  if (document.getElementById("team-analytics-section")) fetchTeamAnalytics();
  if (document.getElementById("goalie-analytics-section")) fetchGoalieAnalytics();
  if (document.getElementById("lines-section")) fetchLineAnalytics();
});

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function show(id) {
  const el = document.getElementById(id);
  if (el) el.hidden = false;
}

function hide(id) {
  const el = document.getElementById(id);
  if (el) el.hidden = true;
}

function showError(prefix) {
  hide(`${prefix}-spinner`);
  show(`${prefix}-error`);
}

function showBlocked(prefix, data) {
  hide(`${prefix}-spinner`);
  const el = document.getElementById(`${prefix}-blocked`);
  if (!el) return;
  el.innerHTML = `
    <strong>Stats unavailable</strong><br>
    ${escHtml(data.message || "This metric is not currently sourced from a self-generated table.")}`;
  show(`${prefix}-blocked`);
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fmtPct(v) {
  return Number(v).toFixed(1) + "%";
}

function metricColor(value) {
  // value: 0–100; white at 50, green above, red below
  const t = Math.min(1, Math.abs(value - 50) / 15);
  if (value >= 50) return lerpColor("#8899BB", "#1DB954", t);
  return lerpColor("#8899BB", "#E8302A", t);
}

function lerpColor(a, b, t) {
  const ah = parseInt(a.slice(1), 16);
  const bh = parseInt(b.slice(1), 16);
  const ar = (ah >> 16) & 0xff, ag = (ah >> 8) & 0xff, ab = ah & 0xff;
  const br = (bh >> 16) & 0xff, bg = (bh >> 8) & 0xff, bb = bh & 0xff;
  const rr = Math.round(ar + t * (br - ar));
  const rg = Math.round(ag + t * (bg - ag));
  const rb = Math.round(ab + t * (bb - ab));
  return `rgb(${rr},${rg},${rb})`;
}

function sourceLabel(source, metric) {
  const labels = {
    self_generated: "Hockey Intelligence Hub models",
    self_generated_xgf_regular_season: "Hockey Intelligence Hub models (regular-season xGF%)",
    moneypuck: "MoneyPuck (raw counts)",
    natural_stat_trick: "Natural Stat Trick",
  };
  // Multi-source labels like "self_generated_xgf+moneypuck_counts" → readable
  let src = labels[source];
  if (!src && source && source.includes("+")) {
    src = source
      .split("+")
      .map((s) => labels[s.trim()] || s.replaceAll("_", " "))
      .join(" + ");
  }
  src = src || source;
  return metric ? `Source: ${src} — ${metric}` : `Source: ${src}`;
}

// ---------------------------------------------------------------------------
// Fetch: Games — date carousel + per-date load
// ---------------------------------------------------------------------------
const GAMES_DATE_RANGE_DAYS = 14;
const GAMES_CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const _gamesCache = {}; // { 'YYYY-MM-DD': { ts, data } }
let _selectedGameDate = null;

function _todayYmd() {
  // Local-date YYYY-MM-DD so the carousel reflects the user's calendar day.
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function _addDaysYmd(ymd, deltaDays) {
  const d = new Date(ymd + "T12:00:00");
  d.setDate(d.getDate() + deltaDays);
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function formatGameDateHeader(yyyyMmDd) {
  const d = new Date(yyyyMmDd + "T12:00:00");
  return d.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
  });
}

function _formatPillLabel(yyyyMmDd) {
  const d = new Date(yyyyMmDd + "T12:00:00");
  const dow = d.toLocaleDateString("en-US", { weekday: "short" }).toUpperCase();
  const md = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  return { dow, md };
}

function _renderDatePills() {
  const rail = document.getElementById("games-carousel-rail");
  if (!rail) return;
  const today = _todayYmd();
  // Pills oldest → newest (today on the right), 14 days back + today = 15 pills total.
  const days = [];
  for (let i = GAMES_DATE_RANGE_DAYS; i >= 0; i--) {
    days.push(_addDaysYmd(today, -i));
  }
  rail.innerHTML = days.map((ymd) => {
    const { dow, md } = _formatPillLabel(ymd);
    const active = ymd === _selectedGameDate ? " active" : "";
    return `
      <button type="button" class="date-pill${active}" role="tab"
              aria-selected="${ymd === _selectedGameDate ? "true" : "false"}"
              data-date="${ymd}">
        <span class="pill-dow">${escHtml(dow)}</span>
        <span class="pill-date">${escHtml(md)}</span>
      </button>
    `;
  }).join("");

  rail.querySelectorAll(".date-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      const ymd = btn.dataset.date;
      if (ymd && ymd !== _selectedGameDate) loadGamesForDate(ymd);
    });
  });

  // Scroll the selected pill into view (anchor on the right edge — today).
  const activeEl = rail.querySelector(".date-pill.active");
  if (activeEl) {
    requestAnimationFrame(() => {
      activeEl.scrollIntoView({ inline: "center", block: "nearest" });
    });
  }
}

function _setActivePill(ymd) {
  const rail = document.getElementById("games-carousel-rail");
  if (!rail) return;
  rail.querySelectorAll(".date-pill").forEach((btn) => {
    const on = btn.dataset.date === ymd;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
    if (on) btn.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  });
}

function _setGamesTitle(ymd) {
  const el = document.getElementById("games-title");
  if (!el) return;
  el.textContent = ymd ? formatGameDateHeader(ymd) : "Recent Games";
}

function _renderGamesForDate(ymd, games) {
  const container = document.getElementById("games-content");
  const empty = document.getElementById("games-empty");
  if (!container) return;

  if (!games || games.length === 0) {
    container.innerHTML = "";
    container.hidden = true;
    if (empty) empty.hidden = false;
  } else {
    if (empty) empty.hidden = true;
    container.innerHTML = `<div class="games-grid">${games.map(renderGameCard).join("")}</div>`;
    container.hidden = false;
    bindDetailsToggles();
  }

  // Key Stories — reflect the currently selected date
  const finished = (games || []).filter((g) => g.state === "OFF");
  const stories = generateStories(finished);
  const list = document.getElementById("stories-list");
  const storiesSection = document.getElementById("stories-section");
  if (list) {
    if (stories.length) {
      list.innerHTML = stories.map((s) => `<li>${escHtml(s)}</li>`).join("");
      if (storiesSection) storiesSection.hidden = false;
    } else {
      list.innerHTML = "";
      if (storiesSection) storiesSection.hidden = true;
    }
  }

  // Brief header date label (already wired in the page header)
  const briefDateEl = document.getElementById("brief-date");
  if (ymd && briefDateEl) {
    briefDateEl.textContent = formatGameDateHeader(ymd);
  }
}

async function _fetchGamesData(ymd) {
  const cached = _gamesCache[ymd];
  if (cached && Date.now() - cached.ts < GAMES_CACHE_TTL_MS) {
    return cached.data;
  }
  const url = ymd ? `/api/games?date=${encodeURIComponent(ymd)}` : "/api/games";
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const data = await r.json();
  if (data && data.date) {
    _gamesCache[data.date] = { ts: Date.now(), data };
  }
  return data;
}

async function loadGamesForDate(ymd) {
  const spinner = document.getElementById("games-spinner");
  const content = document.getElementById("games-content");
  const empty = document.getElementById("games-empty");
  const err = document.getElementById("games-error");
  if (err) err.hidden = true;
  if (empty) empty.hidden = true;
  if (content) content.hidden = true;
  if (spinner) spinner.hidden = false;

  _selectedGameDate = ymd;
  _setActivePill(ymd);
  _setGamesTitle(ymd);

  try {
    const data = await _fetchGamesData(ymd);
    if (spinner) spinner.hidden = true;
    _renderGamesForDate(ymd, data.games || []);
  } catch (e) {
    console.error("loadGamesForDate:", e);
    if (spinner) spinner.hidden = true;
    if (err) err.hidden = false;
  }
}

function _initGamesCarouselArrows() {
  const rail = document.getElementById("games-carousel-rail");
  const prev = document.getElementById("games-carousel-prev");
  const next = document.getElementById("games-carousel-next");
  if (!rail) return;
  const step = () => Math.max(120, Math.round(rail.clientWidth * 0.7));
  if (prev) prev.addEventListener("click", () => rail.scrollBy({ left: -step(), behavior: "smooth" }));
  if (next) next.addEventListener("click", () => rail.scrollBy({ left: step(), behavior: "smooth" }));
}

async function fetchGames() {
  _renderDatePills();
  _initGamesCarouselArrows();
  const spinner = document.getElementById("games-spinner");
  if (spinner) spinner.hidden = false;

  try {
    // Default-load: backend walks back to the most recent date with games.
    const data = await _fetchGamesData(null);
    const ymd = data.date || _todayYmd();
    _selectedGameDate = ymd;
    _setActivePill(ymd);
    _setGamesTitle(ymd);
    if (spinner) spinner.hidden = true;
    _renderGamesForDate(ymd, data.games || []);
  } catch (e) {
    console.error("fetchGames:", e);
    if (spinner) spinner.hidden = true;
    showError("games");
  }
}

function renderGameCard(g) {
  const outcome = g.outcome || "REG";
  const badgeHtml = outcome !== "REG"
    ? `<span class="outcome-badge ${outcome.toLowerCase()}">${outcome}</span>`
    : "";

  const awaySog = g.away.sog != null ? `${g.away.sog} SOG` : "";
  const homeSog = g.home.sog != null ? `${g.home.sog} SOG` : "";

  const awayLogo = g.away.logo
    ? `<img src="${escHtml(g.away.logo)}" alt="${escHtml(g.away.abbrev)}" class="team-logo" loading="lazy" onerror="this.style.display='none'" />`
    : "";
  const homeLogo = g.home.logo
    ? `<img src="${escHtml(g.home.logo)}" alt="${escHtml(g.home.abbrev)}" class="team-logo" loading="lazy" onerror="this.style.display='none'" />`
    : "";

  const awayScore = g.away.score ?? "–";
  const homeScore = g.home.score ?? "–";

  // Article / video links
  const matchupQuery = encodeURIComponent(
    `${g.away.name} vs ${g.home.name} NHL ${g.date || ""}`
  );
  const links = [];
  if (g.gamecenter_link) {
    links.push(`<a href="https://www.nhl.com${escHtml(g.gamecenter_link)}" target="_blank" rel="noopener" class="link-btn link-nhl">NHL.com</a>`);
  }
  // Highlights — always search the official NHL YouTube channel for this matchup
  const highlightsQuery = encodeURIComponent(
    `${g.away.name} ${g.home.name} highlights`
  );
  links.push(`<a href="https://www.youtube.com/@NHL/search?query=${highlightsQuery}" target="_blank" rel="noopener" class="link-btn link-video">Highlights</a>`);
  // ESPN direct game page (preferred) or fallback to search
  const espnHref = g.espn_link
    ? g.espn_link
    : `https://www.espn.com/search/_/q/${matchupQuery}`;
  links.push(`<a href="${escHtml(espnHref)}" target="_blank" rel="noopener" class="link-btn link-news">ESPN</a>`);

  return `
    <div class="game-card" data-game-id="${g.id}">
      <div class="game-venue">${escHtml(g.venue || "")}</div>
      <div class="game-teams">
        <div class="team-side">
          ${awayLogo}
          <span class="team-abbrev">${escHtml(g.away.abbrev)}</span>
          <span class="team-sog">${escHtml(awaySog)}</span>
        </div>
        <div class="game-score-block">
          <div class="score-display">
            <span>${awayScore}</span>
            <span class="score-sep">–</span>
            <span>${homeScore}</span>
          </div>
          ${badgeHtml}
        </div>
        <div class="team-side">
          ${homeLogo}
          <span class="team-abbrev">${escHtml(g.home.abbrev)}</span>
          <span class="team-sog">${escHtml(homeSog)}</span>
        </div>
      </div>
      <div class="game-links">${links.join("")}</div>
      <button class="details-toggle" data-game-id="${g.id}" aria-expanded="false">
        ▾ Show full briefing
      </button>
      <div class="game-details" id="details-${g.id}" hidden>
        <div class="details-spinner spinner-sm"></div>
      </div>
    </div>`;
}

// ---------------------------------------------------------------------------
// Game Details — fetched on demand when toggle is clicked
// ---------------------------------------------------------------------------
const _detailsCache = {};

async function loadGameDetails(gameId) {
  if (_detailsCache[gameId]) return _detailsCache[gameId];
  const data = await fetch(`/api/game-details/${gameId}`).then(r => r.json());
  _detailsCache[gameId] = data;
  return data;
}

function bindDetailsToggles() {
  document.querySelectorAll(".details-toggle").forEach(btn => {
    btn.addEventListener("click", async () => {
      const id = btn.dataset.gameId;
      const panel = document.getElementById(`details-${id}`);
      const isOpen = !panel.hidden;
      if (isOpen) {
        panel.hidden = true;
        btn.textContent = "▾ Show full briefing";
        btn.setAttribute("aria-expanded", "false");
        return;
      }
      btn.textContent = "▴ Hide full briefing";
      btn.setAttribute("aria-expanded", "true");
      panel.hidden = false;
      if (!panel.dataset.loaded) {
        try {
          const data = await loadGameDetails(id);
          panel.innerHTML = renderDetailsPanel(data);
          panel.dataset.loaded = "1";
        } catch (e) {
          panel.innerHTML = `<p class="error-msg">Failed to load details.</p>`;
        }
      }
    });
  });
}

function renderDetailsPanel(d) {
  if (d.error) return `<p class="error-msg">${escHtml(d.error)}</p>`;

  // Three Stars
  let starsHtml = "";
  if (d.stars && d.stars.length) {
    starsHtml = `
      <div class="details-block">
        <h4 class="details-heading">Three Stars</h4>
        <ol class="stars-list">
          ${d.stars.map(s => {
            const line = s.save_pct != null
              ? `SV% ${(s.save_pct * 100).toFixed(1)}%`
              : `${s.goals ?? 0}G ${s.assists ?? 0}A · ${s.points ?? 0} PTS`;
            return `
              <li class="star-item">
                <span class="star-rank">${s.star ?? ""}</span>
                ${s.headshot
                  ? `<img src="${escHtml(s.headshot)}" alt="" class="star-headshot" loading="lazy" onerror="this.style.display='none'" />`
                  : ""}
                <div class="star-info">
                  <div class="star-name">${escHtml(s.name)}</div>
                  <div class="star-team">${escHtml(s.team)}${s.position ? ` · ${escHtml(s.position)}` : ""}</div>
                </div>
                <span class="star-line">${escHtml(line)}</span>
              </li>`;
          }).join("")}
        </ol>
      </div>`;
  }

  // Goalie Line
  let goalieHtml = "";
  if (d.goalies && (d.goalies.away || d.goalies.home)) {
    const rows = [];
    for (const side of ["away", "home"]) {
      const g = d.goalies[side];
      if (!g) continue;
      const team = side === "away" ? d.away_abbrev : d.home_abbrev;
      const svpct = g.save_pct != null ? (g.save_pct * 100).toFixed(1) + "%" : "—";
      rows.push(`
        <tr>
          <td>${escHtml(team)}</td>
          <td>${escHtml(g.name)}</td>
          <td>${g.saves}/${g.shots_against}</td>
          <td>${svpct}</td>
          <td>${g.goals_against}</td>
          <td>${escHtml(g.toi)}</td>
        </tr>`);
    }
    if (rows.length) {
      goalieHtml = `
        <div class="details-block">
          <h4 class="details-heading">Goalie Line</h4>
          <table class="goalie-line-table">
            <thead><tr><th>Team</th><th>Goalie</th><th>SV/SA</th><th>SV%</th><th>GA</th><th>TOI</th></tr></thead>
            <tbody>${rows.join("")}</tbody>
          </table>
        </div>`;
    }
  }

  // Scoring Summary
  let scoringHtml = "";
  if (d.scoring && d.scoring.length) {
    const rows = d.scoring.map(g => {
      const periodLabel = g.period_type === "OT" ? "OT" :
                          g.period_type === "SO" ? "SO" :
                          `P${g.period}`;
      const strengthBadge =
        g.strength === "pp" ? `<span class="str-badge str-pp">PP</span>` :
        g.strength === "sh" ? `<span class="str-badge str-sh">SH</span>` : "";
      const assists = g.assists && g.assists.length
        ? `<span class="assists">(${g.assists.map(escHtml).join(", ")})</span>`
        : "";
      return `
        <tr>
          <td class="scoring-period">${periodLabel}</td>
          <td class="scoring-time">${escHtml(g.time)}</td>
          <td class="scoring-team">${escHtml(g.team)}</td>
          <td class="scoring-goal">
            ${escHtml(g.scorer)} ${strengthBadge} ${assists}
          </td>
          <td class="scoring-score">${g.away_score}–${g.home_score}</td>
        </tr>`;
    }).join("");
    scoringHtml = `
      <div class="details-block">
        <h4 class="details-heading">Scoring Summary</h4>
        <table class="scoring-table">
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }

  // Top highlight clip
  let clipHtml = "";
  if (d.top_clip && d.top_clip.url) {
    clipHtml = `
      <div class="details-block">
        <h4 class="details-heading">Highlight</h4>
        <a href="${escHtml(d.top_clip.url)}" target="_blank" rel="noopener" class="clip-link">
          ▶ Watch ${escHtml(d.top_clip.scorer)}'s goal
        </a>
      </div>`;
  }

  return starsHtml + goalieHtml + scoringHtml + clipHtml ||
         `<p class="no-details">No additional details available.</p>`;
}

// ---------------------------------------------------------------------------
// Key Stories generation
// ---------------------------------------------------------------------------
function generateStories(games) {
  const stories = [];
  for (const g of games) {
    if (g.state !== "OFF") continue;

    const awayScore = g.away.score ?? 0;
    const homeScore = g.home.score ?? 0;
    const awaySog = g.away.sog ?? 0;
    const homeSog = g.home.sog ?? 0;

    const winner = awayScore > homeScore ? g.away : g.home;
    const loser  = awayScore > homeScore ? g.home : g.away;
    const winScore = Math.max(awayScore, homeScore);
    const loseScore = Math.min(awayScore, homeScore);
    const margin = winScore - loseScore;
    const outcome = g.outcome || "REG";

    const suffix = outcome === "OT" ? " in overtime" : outcome === "SO" ? " in a shootout" : "";
    stories.push(`${winner.name} defeated ${loser.name} ${winScore}–${loseScore}${suffix} at ${g.venue || "home"}.`);

    const sogDiff = Math.abs(awaySog - homeSog);
    if (sogDiff >= 10) {
      const sogWinner = awaySog > homeSog ? g.away : g.home;
      const sogLoser  = awaySog > homeSog ? g.home : g.away;
      stories.push(
        `${sogWinner.name} dominated possession, outshooting ${sogLoser.name} ${Math.max(awaySog, homeSog)}–${Math.min(awaySog, homeSog)}.`
      );
    }

    if (margin >= 4) {
      stories.push(`${winner.name} won convincingly by ${margin} goals.`);
    }

    if (g.series) {
      const s = g.series;
      const topW = s.topSeedWins ?? 0;
      const botW = s.bottomSeedWins ?? 0;
      const need = s.neededToWin ?? 4;

      if (topW >= need) {
        stories.push(
          `${s.topSeedTeamAbbrev} wins the ${s.seriesTitle}, defeating ${s.bottomSeedTeamAbbrev} ${topW}–${botW}.`
        );
      } else if (botW >= need) {
        stories.push(
          `${s.bottomSeedTeamAbbrev} wins the ${s.seriesTitle}, defeating ${s.topSeedTeamAbbrev} ${botW}–${topW}.`
        );
      } else if (topW === botW) {
        stories.push(
          `${s.seriesTitle} series is tied ${topW}–${botW}. Game ${topW + botW + 1} is next.`
        );
      } else {
        const leader  = topW > botW ? s.topSeedTeamAbbrev : s.bottomSeedTeamAbbrev;
        const trailer = topW > botW ? s.bottomSeedTeamAbbrev : s.topSeedTeamAbbrev;
        const leadW   = Math.max(topW, botW);
        const trailW  = Math.min(topW, botW);
        stories.push(
          `${leader} leads the ${s.seriesTitle} ${leadW}–${trailW} over ${trailer}.`
        );
      }
    }
  }
  return stories;
}

// ---------------------------------------------------------------------------
// Fetch: Season Leaders
// ---------------------------------------------------------------------------
async function fetchLeaders() {
  try {
    const data = await fetch("/api/leaders").then(r => r.json());
    hide("leaders-spinner");

    const container = document.getElementById("leaders-content");
    const sections = [
      { key: "points",  title: "Points",      from: "skaters" },
      { key: "goals",   title: "Goals",       from: "skaters" },
      { key: "assists", title: "Assists",      from: "skaters" },
      { key: null,      title: "Goalie Wins",  from: "goalies" },
    ];

    container.innerHTML = sections.map(sec => {
      const players = sec.from === "goalies"
        ? (data.goalies || [])
        : (data.skaters?.[sec.key] || []);
      return renderLeadersCard(sec.title, players);
    }).join("");

    show("leaders-content");
  } catch (e) {
    console.error("fetchLeaders:", e);
    showError("leaders");
  }
}

function renderLeadersCard(title, players) {
  const rows = players.map((p, i) => `
    <li>
      <span class="leader-rank">${i + 1}</span>
      ${p.headshot ? `<img src="${escHtml(p.headshot)}" alt="" class="leader-headshot" loading="lazy" onerror="this.style.display='none'" />` : ""}
      <div class="leader-info">
        <div class="leader-name">${escHtml(p.name)}</div>
        <div class="leader-team">${escHtml(p.team)} · ${escHtml(p.position || "")}</div>
      </div>
      <span class="leader-value">${p.value}</span>
    </li>`).join("");

  return `
    <div class="leaders-card">
      <div class="leaders-card-title">${escHtml(title)}</div>
      <ul class="leaders-list">${rows}</ul>
    </div>`;
}

// ---------------------------------------------------------------------------
// Fetch: Standings
// ---------------------------------------------------------------------------
async function fetchStandings() {
  try {
    const data = await fetch("/api/standings").then(r => r.json());
    hide("standings-spinner");

    const container = document.getElementById("standings-content");
    container.innerHTML = Object.entries(data)
      .map(([conf, divs]) => renderConferenceStandings(conf, divs))
      .join("");

    show("standings-content");
  } catch (e) {
    console.error("fetchStandings:", e);
    showError("standings");
  }
}

function renderConferenceStandings(confName, divisions) {
  // Collect all teams and compute playoff spots
  const allTeams = [];
  for (const [div, teams] of Object.entries(divisions)) {
    teams.forEach((t, i) => allTeams.push({ ...t, div, divPos: i }));
  }

  const inPlayoffs = computePlayoffSpots(divisions);

  // Sort by conf_rank for rendering
  const sorted = [...allTeams].sort((a, b) => a.conf_rank - b.conf_rank);

  const rows = sorted.map((t, idx) => {
    const isIn = inPlayoffs.has(t.abbrev);
    const isCutline = idx === 7; // after 8th team

    const logo = `<img src="${escHtml(t.logo)}" alt="" class="std-logo" loading="lazy" onerror="this.style.display='none'" />`;
    const clinch = t.clinch ? `<span class="clinch-badge">${escHtml(t.clinch)}</span>` : "";
    const ptsCell = `<td class="pts-cell">${t.pts}</td>`;

    return `
      <tr class="${isIn ? "in-playoffs" : ""}${isCutline ? " cutline-row" : ""}">
        <td class="rank-col">${t.conf_rank}</td>
        <td>${logo}<span class="std-abbrev">${escHtml(t.abbrev)}</span>${clinch}</td>
        <td>${t.gp}</td>
        <td>${t.w}</td>
        <td>${t.l}</td>
        <td>${t.otl}</td>
        ${ptsCell}
      </tr>`;
  }).join("");

  return `
    <div class="standings-conf">
      <div class="standings-conf-title">${escHtml(confName)} Conference</div>
      <table class="standings-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Team</th>
            <th>GP</th>
            <th>W</th>
            <th>L</th>
            <th>OTL</th>
            <th>PTS</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function computePlayoffSpots(divisions) {
  const inPlayoffs = new Set();
  const wildcardPool = [];

  for (const [, teams] of Object.entries(divisions)) {
    const sorted = [...teams].sort((a, b) => a.div_rank - b.div_rank);
    sorted.forEach((t, i) => {
      if (i < 3) inPlayoffs.add(t.abbrev);
      else wildcardPool.push(t);
    });
  }

  // Top 2 from wildcard pool by points (tiebreak: wins)
  wildcardPool
    .sort((a, b) => b.pts - a.pts || b.w - a.w)
    .slice(0, 2)
    .forEach(t => inPlayoffs.add(t.abbrev));

  return inPlayoffs;
}

// ---------------------------------------------------------------------------
// Fetch: GAR Leaderboard
// ---------------------------------------------------------------------------
async function fetchGarLeaders(position = "all") {
  try {
    show("gar-spinner");
    hide("gar-content");
    hide("gar-error");
    hide("gar-blocked");
    const data = await fetch(`/api/gar-leaders?position=${position}`).then(r => r.json());
    hide("gar-spinner");

    if (data.blocked) { showBlocked("gar", data); return; }

    document.getElementById("gar-source-label").textContent =
      sourceLabel(data.source, data.metric);

    const players = data.players || [];
    const labels  = players.map(p => `${p.name} (${p.team})`);
    const values  = players.map(p => p.value);

    if (garChart) garChart.destroy();
    const ctx = document.getElementById("gar-chart").getContext("2d");
    garChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: data.metric || "GAR",
          data: values,
          backgroundColor: "rgba(232, 160, 32, 0.8)",
          borderColor: "#E8A020",
          borderWidth: 1,
          borderRadius: 3,
        }],
      },
      options: garChartOptions(data.metric || "GAR"),
    });

    show("gar-content");
  } catch (e) {
    console.error("fetchGarLeaders:", e);
    showError("gar");
  }
}

// ---------------------------------------------------------------------------
// Fetch: xGAR Leaderboard
// ---------------------------------------------------------------------------
async function fetchXgarLeaders(position = "all") {
  try {
    show("xgar-spinner");
    hide("xgar-content");
    hide("xgar-error");
    hide("xgar-blocked");
    const data = await fetch(`/api/xgar-leaders?position=${position}`).then(r => r.json());
    hide("xgar-spinner");

    if (data.blocked) { showBlocked("xgar", data); return; }

    document.getElementById("xgar-source-label").textContent =
      sourceLabel(data.source, data.metric);

    const players = data.players || [];
    const labels  = players.map(p => `${p.name} (${p.team})`);
    const xvals   = players.map(p => p.xgar_value ?? p.value ?? 0);
    const gvals   = players.map(p => p.gar_value ?? 0);

    if (xgarChart) xgarChart.destroy();
    const ctx = document.getElementById("xgar-chart").getContext("2d");

    const datasets = [{
      label: data.metric || "xGAR",
      data: xvals,
      backgroundColor: "rgba(43, 107, 240, 0.8)",
      borderColor: "#2B6BF0",
      borderWidth: 1,
      borderRadius: 3,
    }];

    if (gvals.some(v => v !== 0)) {
      datasets.push({
        label: "GAR (Total)",
        data: gvals,
        backgroundColor: "rgba(232, 160, 32, 0.5)",
        borderColor: "#E8A020",
        borderWidth: 1,
        borderRadius: 3,
      });
    }

    xgarChart = new Chart(ctx, {
      type: "bar",
      data: { labels, datasets },
      options: garChartOptions(data.metric || "xGAR", true),
    });

    show("xgar-content");
  } catch (e) {
    console.error("fetchXgarLeaders:", e);
    showError("xgar");
  }
}

function garChartOptions(metricLabel, grouped = false) {
  return {
    indexAxis: "y",
    responsive: true,
    plugins: {
      legend: {
        display: grouped,
        labels: { color: "#8899BB", font: { family: "'DM Mono', monospace", size: 11 } },
      },
      tooltip: {
        backgroundColor: "#080F1F",
        borderColor: "#0E1E35",
        borderWidth: 1,
        titleColor: "#F0F4FF",
        bodyColor: "#8899BB",
        callbacks: {
          label: ctx => ` ${metricLabel}: ${ctx.parsed.x}`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: "rgba(14, 30, 53, 0.8)" },
        ticks: { color: "#8899BB", font: { family: "'DM Mono', monospace", size: 11 } },
      },
      y: {
        grid: { color: "rgba(14, 30, 53, 0.4)" },
        ticks: { color: "#C0CCDD", font: { family: "'DM Mono', monospace", size: 11 } },
      },
    },
  };
}

// ---------------------------------------------------------------------------
// Fetch: Team Analytics
// ---------------------------------------------------------------------------
async function fetchTeamAnalytics() {
  try {
    const data = await fetch("/api/team-analytics").then(r => r.json());
    hide("team-spinner");

    if (data.blocked) { showBlocked("team", data); return; }

    document.getElementById("team-source-label").textContent =
      sourceLabel(data.source, "5v5 xGF% · CF% · HDCF%");

    teamData = data.teams || [];
    renderTeamTable(teamData, teamSortCol, teamSortAsc);
    bindTeamSort();
    show("team-content");
  } catch (e) {
    console.error("fetchTeamAnalytics:", e);
    showError("team");
  }
}

function renderTeamTable(teams, sortCol, asc) {
  const sorted = [...teams].sort((a, b) => {
    const diff = a[sortCol] - b[sortCol];
    return asc ? diff : -diff;
  });

  const tbody = document.getElementById("team-tbody");
  tbody.innerHTML = sorted.map((t, i) => {
    const logo = `https://assets.nhle.com/logos/nhl/svg/${t.team}_light.svg`;
    return `
      <tr>
        <td class="rank-col">${i + 1}</td>
        <td>
          <div class="team-abbrev-cell">
            <img src="${escHtml(logo)}" alt="" class="team-logo-sm" loading="lazy" onerror="this.style.display='none'" />
            ${escHtml(t.team)}
          </div>
        </td>
        <td class="metric-cell" style="color:${metricColor(t.xgf_pct)}">${fmtPct(t.xgf_pct)}</td>
        <td class="metric-cell" style="color:${metricColor(t.cf_pct)}">${fmtPct(t.cf_pct)}</td>
        <td class="metric-cell" style="color:${metricColor(t.hdcf_pct)}">${fmtPct(t.hdcf_pct)}</td>
        <td>${t.gp}</td>
      </tr>`;
  }).join("");
}

function bindTeamSort() {
  document.querySelectorAll("#team-table th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (teamSortCol === col) {
        teamSortAsc = !teamSortAsc;
      } else {
        teamSortCol = col;
        teamSortAsc = false;
      }
      document.querySelectorAll("#team-table th.sortable").forEach(h => {
        h.classList.remove("active-sort", "asc");
      });
      th.classList.add("active-sort");
      if (teamSortAsc) th.classList.add("asc");
      renderTeamTable(teamData, teamSortCol, teamSortAsc);
    });
  });
}

// ---------------------------------------------------------------------------
// Fetch: Goalie Analytics
// ---------------------------------------------------------------------------
async function fetchGoalieAnalytics() {
  try {
    const data = await fetch("/api/goalie-analytics").then(r => r.json());
    hide("goalie-spinner");

    if (data.blocked) { showBlocked("goalie", data); return; }

    document.getElementById("goalie-source-label").textContent =
      sourceLabel(data.source, data.metric);

    const goalies = data.goalies || [];
    const tbody = document.getElementById("goalie-tbody");
    tbody.innerHTML = goalies.map((g, i) => {
      const gsaxClass = g.gsax >= 0 ? "gsax-pos" : "gsax-neg";
      const gsaxSign  = g.gsax >= 0 ? "+" : "";
      const toi = g.toi_min ? `${Math.round(g.toi_min / 60)}h ${Math.round(g.toi_min % 60)}m` : "—";
      return `
        <tr>
          <td class="rank-col">${i + 1}</td>
          <td>${escHtml(g.name)}</td>
          <td>
            <div class="team-abbrev-cell">
              <img src="https://assets.nhle.com/logos/nhl/svg/${escHtml(g.team)}_light.svg"
                   alt="" class="team-logo-sm" loading="lazy" onerror="this.style.display='none'" />
              ${escHtml(g.team)}
            </div>
          </td>
          <td class="${gsaxClass}">${gsaxSign}${Number(g.gsax).toFixed(2)}</td>
          <td>${g.games}</td>
          <td>${toi}</td>
        </tr>`;
    }).join("");

    show("goalie-content");
  } catch (e) {
    console.error("fetchGoalieAnalytics:", e);
    showError("goalie");
  }
}

// ---------------------------------------------------------------------------
// Fetch: Line Analytics
// ---------------------------------------------------------------------------
async function fetchLineAnalytics() {
  try {
    const data = await fetch("/api/line-analytics").then(r => r.json());
    hide("lines-spinner");

    if (data.blocked) { showBlocked("lines", data); return; }

    const lines = data.lines || [];
    if (!lines.length) { showBlocked("lines", { message: "Line combinations are not currently surfaced — forward-trio identification from my shifts dataset is a planned extension." }); return; }

    const container = document.getElementById("lines-content");
    const cols = lines[0] ? Object.keys(lines[0]) : [];
    const rows = lines.map(l =>
      `<tr>${cols.map(c => `<td>${escHtml(String(l[c] ?? ""))}</td>`).join("")}</tr>`
    ).join("");

    container.innerHTML = `
      <div class="table-wrap">
        <table class="data-table">
          <thead><tr>${cols.map(c => `<th>${escHtml(c)}</th>`).join("")}</tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;

    show("lines-content");
  } catch (e) {
    console.error("fetchLineAnalytics:", e);
    showError("lines");
  }
}
