/* =========================================================
   Player Lookup — live search, profile card, advanced stats,
   career tables, shot chart, similar players.
   Exposes window.playerLookup.{search, loadProfile} for compare.
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = window.dashUtil ? window.dashUtil.escHtml : (s) => String(s ?? "");

  let searchTimer = null;
  let activeIndex = -1;
  let dropdownItems = [];
  let lastQuery = "";
  let shotChartCache = null;

  // ---------------------------------------------------------------------------
  // Generic live-search dropdown helper. Returns a controller for any input.
  // onSelect(playerObj) is called when a result is chosen.
  // ---------------------------------------------------------------------------
  function attachLiveSearch(input, dropdown, onSelect) {
    let timer = null;
    let items = [];
    let cursor = -1;

    function renderDropdown(results) {
      items = results;
      cursor = -1;
      if (!results || !results.length) {
        dropdown.innerHTML = `<div class="player-search-empty">No players found.</div>`;
        dropdown.hidden = false;
        return;
      }
      dropdown.innerHTML = results.map((p, i) => `
        <div class="player-search-item" data-index="${i}" role="option">
          <img class="player-search-headshot" src="${esc(p.headshot)}" alt="" onerror="this.style.display='none'" />
          <div class="player-search-meta">
            <div class="player-search-name">${esc(p.name)}</div>
            <div class="player-search-sub">
              ${p.team ? `<img class="player-search-logo" src="${esc(p.team_logo)}" alt="" onerror="this.style.display='none'" />` : ""}
              <span>${esc(p.team || "—")}</span>
              <span>&middot; ${esc(p.position || "—")}</span>
              ${p.sweater ? `<span>&middot; #${esc(p.sweater)}</span>` : ""}
            </div>
          </div>
        </div>
      `).join("");
      dropdown.hidden = false;
    }

    function clearDropdown() {
      dropdown.hidden = true;
      dropdown.innerHTML = "";
      items = [];
      cursor = -1;
    }

    async function runSearch(q) {
      try {
        const r = await fetch(`/api/player-search?q=${encodeURIComponent(q)}&limit=10`);
        const data = await r.json();
        renderDropdown(data.results || []);
      } catch (e) {
        dropdown.innerHTML = `<div class="player-search-empty">Search failed.</div>`;
        dropdown.hidden = false;
      }
    }

    input.addEventListener("input", () => {
      const q = input.value.trim();
      clearTimeout(timer);
      if (q.length < 2) {
        clearDropdown();
        return;
      }
      timer = setTimeout(() => runSearch(q), 180);
    });

    input.addEventListener("keydown", (e) => {
      if (dropdown.hidden || !items.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        cursor = Math.min(cursor + 1, items.length - 1);
        updateActive();
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        cursor = Math.max(cursor - 1, 0);
        updateActive();
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (cursor >= 0) selectItem(cursor);
      } else if (e.key === "Escape") {
        clearDropdown();
      }
    });

    function updateActive() {
      $$(".player-search-item", dropdown).forEach((el, i) => {
        el.classList.toggle("active", i === cursor);
      });
    }

    function selectItem(i) {
      const p = items[i];
      if (!p) return;
      input.value = p.name;
      clearDropdown();
      onSelect(p);
    }

    dropdown.addEventListener("click", (e) => {
      const item = e.target.closest(".player-search-item");
      if (!item) return;
      selectItem(Number(item.dataset.index));
    });

    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target) && e.target !== input) {
        clearDropdown();
      }
    });

    return { clear: clearDropdown };
  }

  // ---------------------------------------------------------------------------
  // Profile rendering
  // ---------------------------------------------------------------------------
  function fmt(v, suffix = "", dp = null) {
    if (v == null || v === "" || (typeof v === "number" && isNaN(v))) return "—";
    if (dp != null && typeof v === "number") return v.toFixed(dp) + suffix;
    return String(v) + suffix;
  }

  function fmtToi(t) {
    if (!t) return "—";
    // Already MM:SS string from NHL API
    return String(t);
  }

  function fmtPct(v, dp = 1) {
    if (v == null || isNaN(v)) return "—";
    // NHL API returns shooting/save pct as a fraction (0.123) — convert
    if (v < 1) return (v * 100).toFixed(dp) + "%";
    return Number(v).toFixed(dp) + "%";
  }

  function profileTopHTML(p) {
    const bio = p.bio;
    const cs = p.current_season || {};
    const team = bio.team || "—";
    const teamName = bio.team_name || team;
    return `
      <header class="player-profile-head">
        <div class="player-headshot-wrap">
          <img class="player-headshot" src="${esc(bio.headshot)}" alt="${esc(bio.name)}" onerror="this.style.display='none'" />
          ${bio.sweater ? `<div class="player-sweater">#${esc(bio.sweater)}</div>` : ""}
        </div>
        <div class="player-bio">
          <div class="player-breadcrumb">
            <a href="#" class="player-team-link" data-team="${esc(team)}">${esc(teamName)}</a>
            <span class="player-breadcrumb-sep">›</span>
            <span>${esc(bio.position)}</span>
          </div>
          <h2 class="player-name">${esc(bio.name)}</h2>
          <div class="player-bio-grid">
            <div><span class="bio-label">Age</span> <span class="bio-value">${fmt(bio.age)}</span></div>
            <div><span class="bio-label">Height</span> <span class="bio-value">${esc(bio.height)}</span></div>
            <div><span class="bio-label">Weight</span> <span class="bio-value">${esc(bio.weight)}</span></div>
            <div><span class="bio-label">Shoots</span> <span class="bio-value">${esc(bio.shoots)}</span></div>
            <div><span class="bio-label">Born</span> <span class="bio-value">${esc(bio.birth_date)}</span></div>
            <div><span class="bio-label">Hometown</span> <span class="bio-value">${esc(bio.hometown || "—")}</span></div>
            <div><span class="bio-label">NHL seasons</span> <span class="bio-value">${fmt(bio.years_in_league)}</span></div>
            ${bio.draft && bio.draft.year ? `<div><span class="bio-label">Draft</span> <span class="bio-value">${esc(bio.draft.year)} #${esc(bio.draft.overallPick || "?")} (${esc(bio.draft.teamAbbrev || "?")})</span></div>` : ""}
          </div>
        </div>
        <div class="player-team-block">
          <img class="player-team-logo" src="${esc(bio.team_logo)}" alt="${esc(team)}" onerror="this.style.display='none'" />
          <div class="player-team-abbrev">${esc(team)}</div>
        </div>
      </header>

      <!-- League rankings populated async by loadRankings() -->
      <div class="player-rank-pills" id="player-rank-pills" data-loading="1">
        <span class="rank-pills-label">League rank</span>
        <span class="muted">Loading…</span>
      </div>
    `;
  }

  function ordinal(n) {
    if (n == null) return "—";
    const s = ["th", "st", "nd", "rd"];
    const v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  async function loadRankings(playerId) {
    const wrap = document.getElementById("player-rank-pills");
    if (!wrap) return;
    try {
      const r = await fetch(`/api/player/${playerId}/rankings`);
      const data = await r.json();
      const ranks = data.rankings || [];
      if (!ranks.length) {
        wrap.innerHTML = `<span class="rank-pills-label">League rank</span><span class="muted">No qualifying ranks (player below TOI threshold)</span>`;
        return;
      }
      const pillsHtml = ranks.map((rk) => {
        const tierCls = rk.rank <= 5 ? "rank-pill-elite" : rk.rank <= 25 ? "rank-pill-good" : rk.rank <= 100 ? "rank-pill-mid" : "rank-pill-low";
        return `<span class="rank-pill ${tierCls}" title="Rank ${rk.rank} of ${rk.total} ${rk.scope}">
          <span class="rank-pill-stat">${esc(rk.stat)}</span>
          <strong>${ordinal(rk.rank)}</strong>
          <span class="rank-pill-scope">in ${esc(rk.scope)}</span>
        </span>`;
      }).join("");
      wrap.innerHTML = `<span class="rank-pills-label">League rank</span>${pillsHtml}`;
    } catch (e) {
      wrap.innerHTML = `<span class="rank-pills-label">League rank</span><span class="muted">Unavailable.</span>`;
    }
  }

  function statTableHTML(title, rows) {
    return `
      <div class="player-stat-card">
        <h3 class="player-stat-card-title">${esc(title)}</h3>
        <table class="player-stat-table">
          <tbody>
            ${rows.map((r) => `
              <tr><td class="stat-label">${esc(r.label)}</td><td class="stat-value">${r.value}</td></tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;
  }

  function skaterCurrentSeasonRows(cs) {
    return [
      { label: "GP", value: fmt(cs.gp) },
      { label: "G", value: fmt(cs.g) },
      { label: "A", value: fmt(cs.a) },
      { label: "PTS", value: fmt(cs.pts) },
      { label: "+/-", value: fmt(cs.plus_minus) },
      { label: "PIM", value: fmt(cs.pim) },
      { label: "Shots", value: fmt(cs.shots) },
      { label: "Shooting %", value: fmtPct(cs.shooting_pct) },
      { label: "TOI per game", value: fmtToi(cs.toi_per_game) },
      { label: "PP Points", value: fmt(cs.pp_pts) },
      { label: "SH Points", value: fmt(cs.sh_pts) },
    ];
  }

  function skaterAdvancedRows(adv) {
    if (!adv || !Object.keys(adv).length) {
      return [{ label: "Advanced stats", value: '<span class="muted">Not available from MoneyPuck for this player</span>' }];
    }
    return [
      { label: "GAR", value: fmt(adv.gar) },
      { label: "xGAR", value: fmt(adv.xgar) },
      { label: "iCF", value: fmt(adv.icf) },
      { label: "iFF", value: fmt(adv.iff) },
      { label: "ixG", value: fmt(adv.ixg) },
      { label: "iHDCF", value: fmt(adv.ihdcf) },
      { label: "On-Ice xGF%", value: adv.onice_xgf_pct != null ? fmt(adv.onice_xgf_pct, "%", 1) : "—" },
      { label: "On-Ice CF%", value: adv.onice_cf_pct != null ? fmt(adv.onice_cf_pct, "%", 1) : "—" },
      { label: "Zone Start %", value: adv.zone_start_pct != null ? fmt(adv.zone_start_pct, "%", 1) : "—" },
    ];
  }

  function goalieCurrentSeasonRows(cs) {
    return [
      { label: "GP", value: fmt(cs.gp) },
      { label: "GS", value: fmt(cs.gs) },
      { label: "W", value: fmt(cs.w) },
      { label: "L", value: fmt(cs.l) },
      { label: "OTL", value: fmt(cs.otl) },
      { label: "GAA", value: fmt(cs.gaa, "", 2) },
      { label: "SV%", value: fmtPct(cs.sv_pct, 3) },
      { label: "SO", value: fmt(cs.so) },
    ];
  }

  function goalieAdvancedRows(adv) {
    if (!adv || !Object.keys(adv).length) {
      return [{ label: "Advanced stats", value: '<span class="muted">Not available from MoneyPuck</span>' }];
    }
    return [
      { label: "GSAX", value: fmt(adv.gsax, "", 2) },
      { label: "GSAx per 60", value: fmt(adv.gsax_60, "", 3) },
      { label: "High Danger SV%", value: adv.hdsv_pct != null ? fmt(adv.hdsv_pct, "%", 2) : "—" },
      { label: "Medium Danger SV%", value: adv.mdsv_pct != null ? fmt(adv.mdsv_pct, "%", 2) : "—" },
      { label: "Low Danger SV%", value: adv.ldsv_pct != null ? fmt(adv.ldsv_pct, "%", 2) : "—" },
      { label: "Quality Start %", value: adv.qs_pct != null ? fmt(adv.qs_pct, "%", 1) : "—" },
      { label: "Starts", value: fmt(adv.starts) },
    ];
  }

  function last5HTML(p) {
    const isG = p.bio.is_goalie;
    const games = p.last5 || [];
    if (!games.length) {
      return '<div class="player-stat-card"><h3 class="player-stat-card-title">Last 5 Games</h3><p class="muted">No recent games available.</p></div>';
    }

    const headers = isG
      ? ["Date", "Opp", "Decision", "SA", "GA", "SV%", "TOI"]
      : ["Date", "Opp", "G", "A", "PTS", "Shots", "+/-", "TOI", "PIM"];

    const rows = games.map((g) => {
      const opp = `${g.home_road || ""} ${g.opponent || ""}`.trim();
      if (isG) {
        return `<tr>
          <td>${esc(g.date)}</td>
          <td>${esc(opp)}</td>
          <td>${esc(g.decision || "—")}</td>
          <td>${fmt(g.shots_against)}</td>
          <td>${fmt(g.goals_against)}</td>
          <td>${fmtPct(g.sv_pct, 3)}</td>
          <td>${fmtToi(g.toi)}</td>
        </tr>`;
      }
      return `<tr>
        <td>${esc(g.date)}</td>
        <td>${esc(opp)}</td>
        <td>${fmt(g.goals)}</td>
        <td>${fmt(g.assists)}</td>
        <td>${fmt(g.points)}</td>
        <td>${fmt(g.shots)}</td>
        <td>${fmt(g.plus_minus)}</td>
        <td>${fmtToi(g.toi)}</td>
        <td>${fmt(g.pim)}</td>
      </tr>`;
    }).join("");

    return `
      <div class="player-stat-card">
        <h3 class="player-stat-card-title">Last 5 Games</h3>
        <div class="table-wrap">
          <table class="data-table player-game-log-table">
            <thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  function careerTableHTML(p, type) {
    const rows = type === "playoff" ? p.career_playoff : p.career_regular;
    const isG = p.bio.is_goalie;
    const title = type === "playoff" ? "Career — Playoffs" : "Career — Regular Season";

    if (!rows || !rows.length) {
      return `<div class="player-stat-card"><h3 class="player-stat-card-title">${title}</h3><p class="muted">No ${type} stats on file.</p></div>`;
    }

    const headers = isG
      ? ["Season", "Team", "League", "GP", "W", "L", "OTL", "GAA", "SV%", "SO"]
      : ["Season", "Team", "League", "GP", "G", "A", "PTS", "+/-", "PIM", "Shots"];

    const seasonFmt = (s) => {
      if (!s) return "—";
      const ss = String(s);
      if (ss.length === 8) return `${ss.slice(0, 4)}-${ss.slice(6, 8)}`;
      return ss;
    };

    const body = rows.map((r) => {
      if (isG) {
        return `<tr>
          <td>${esc(seasonFmt(r.season))}</td>
          <td>${esc(r.team || "—")}</td>
          <td>${esc(r.league || "—")}</td>
          <td>${fmt(r.gp)}</td>
          <td>${fmt(r.w)}</td>
          <td>${fmt(r.l)}</td>
          <td>${fmt(r.otl)}</td>
          <td>${fmt(r.gaa, "", 2)}</td>
          <td>${fmtPct(r.sv_pct, 3)}</td>
          <td>${fmt(r.so)}</td>
        </tr>`;
      }
      return `<tr>
        <td>${esc(seasonFmt(r.season))}</td>
        <td>${esc(r.team || "—")}</td>
        <td>${esc(r.league || "—")}</td>
        <td>${fmt(r.gp)}</td>
        <td>${fmt(r.g)}</td>
        <td>${fmt(r.a)}</td>
        <td>${fmt(r.pts)}</td>
        <td>${fmt(r.plus_minus)}</td>
        <td>${fmt(r.pim)}</td>
        <td>${fmt(r.shots)}</td>
      </tr>`;
    }).join("");

    return `
      <div class="player-stat-card">
        <h3 class="player-stat-card-title">${title}</h3>
        <div class="table-wrap">
          <table class="data-table player-career-table sortable-table">
            <thead><tr>${headers.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead>
            <tbody>${body}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Shot chart — half rink, last N games
  // ---------------------------------------------------------------------------
  function shotChartCardHTML() {
    return `
      <div class="player-stat-card shot-chart-card">
        <h3 class="player-stat-card-title">Shot Map <span class="subtitle">— last 10 games, this season</span></h3>
        <div id="player-shot-chart-status" class="muted shot-chart-status">Loading shot data…</div>
        <div class="shot-chart-wrap" id="player-shot-chart-wrap" hidden>
          <svg id="player-shot-chart-svg" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet"></svg>
          <div class="shot-chart-legend">
            <span class="legend-item"><span class="shot-dot shot-goal"></span>Goal</span>
            <span class="legend-item"><span class="shot-dot shot-onnet"></span>On Target</span>
            <span class="legend-item"><span class="shot-dot shot-miss"></span>Missed / Blocked</span>
          </div>
        </div>
      </div>
    `;
  }

  function renderShotChart(shots) {
    const svg = document.getElementById("player-shot-chart-svg");
    const wrap = document.getElementById("player-shot-chart-wrap");
    const status = document.getElementById("player-shot-chart-status");
    if (!svg) return;

    if (!shots || !shots.length) {
      status.textContent = "No shot data available for this player's last 10 games.";
      return;
    }

    // Show wrap, hide loading status
    status.style.display = "none";
    wrap.hidden = false;

    // SVG viewBox is 400 wide x 300 tall.
    // We map an offensive zone half-rink: x from 25 to 100 (NHL feet from center → goal),
    // y from -42.5 to 42.5. We mirror so that goals are always on the right.
    // svg coords: convert (rinkX, rinkY) → (svgX, svgY)
    // svgX = ((rinkX - (-42)) / (100 - (-42))) * 400 ... too wide.
    // Use a half-rink at the offensive end: x ∈ [-10, 100], y ∈ [-42.5, 42.5]
    // svgX = (rinkX + 10) * (400 / 110)
    // svgY = (-rinkY + 42.5) * (300 / 85)

    function project(x, y) {
      // Mirror shots taken in the defensive zone so all show on offensive half
      let rx = x;
      let ry = y;
      if (rx < 0) {
        rx = -rx;
        ry = -ry;
      }
      const svgX = (rx + 10) * (400 / 110);
      const svgY = (-ry + 42.5) * (300 / 85);
      return [svgX, svgY];
    }

    const rink = `
      <!-- Rink background -->
      <rect x="0" y="0" width="400" height="300" rx="14" ry="14" fill="#0A1525" stroke="#0E1E35" stroke-width="2" />
      <!-- Center red line on left edge -->
      <line x1="36.4" y1="0" x2="36.4" y2="300" stroke="#E8302A" stroke-width="2" opacity="0.6" />
      <!-- Blue line at NHL x=25 → svgX = (25+10)*(400/110) ≈ 127.3 -->
      <line x1="127.3" y1="0" x2="127.3" y2="300" stroke="#2B6BF0" stroke-width="3" opacity="0.7" />
      <!-- Goal line at NHL x=89 → svgX = (89+10)*(400/110) ≈ 360 -->
      <line x1="360" y1="40" x2="360" y2="260" stroke="#E8302A" stroke-width="1.5" opacity="0.7" />
      <!-- Goal crease (semicircle in front of net) -->
      <path d="M 360 130 A 22 22 0 0 0 360 170 Z" fill="rgba(43,107,240,0.12)" stroke="#2B6BF0" stroke-width="1" opacity="0.5" />
      <!-- Faceoff circles at offensive end -->
      <circle cx="295" cy="80" r="20" fill="none" stroke="#E8302A" stroke-width="1" opacity="0.5" />
      <circle cx="295" cy="220" r="20" fill="none" stroke="#E8302A" stroke-width="1" opacity="0.5" />
      <!-- Slot dashed -->
      <path d="M 295 80 L 360 130 M 295 220 L 360 170" stroke="#E8A020" stroke-width="0.8" stroke-dasharray="3 3" opacity="0.35" />
    `;

    function dotCls(t) {
      if (t === "goal") return "shot-dot-svg shot-goal";
      if (t === "shot-on-goal") return "shot-dot-svg shot-onnet";
      return "shot-dot-svg shot-miss";
    }
    function dotR(t) {
      return t === "goal" ? 6 : 4;
    }

    const shotsHtml = shots.map((s) => {
      const [cx, cy] = project(s.x, s.y);
      return `<circle class="${dotCls(s.type)}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${dotR(s.type)}">
        <title>${esc(`${s.type} vs ${s.opp} — ${s.date} (${s.shot_type || ""})`)}</title>
      </circle>`;
    }).join("");

    svg.innerHTML = rink + shotsHtml;
  }

  async function loadShotChart(playerId) {
    try {
      const r = await fetch(`/api/player/${playerId}/shot-chart?games=10`);
      const data = await r.json();
      renderShotChart(data.shots || []);
    } catch (e) {
      const status = document.getElementById("player-shot-chart-status");
      if (status) status.textContent = "Failed to load shot chart.";
    }
  }

  // ---------------------------------------------------------------------------
  // Similar players
  // ---------------------------------------------------------------------------
  async function loadSimilar(playerId) {
    try {
      const r = await fetch(`/api/player/${playerId}/similar`);
      const data = await r.json();
      const wrap = document.getElementById("player-similar-wrap");
      if (!wrap) return;
      if (!data.similar || !data.similar.length) {
        wrap.innerHTML = `
          <h3 class="player-stat-card-title">Similar Players</h3>
          <p class="muted">${esc(data.note || "No similar-player matches available.")}</p>
        `;
        return;
      }
      wrap.innerHTML = `
        <h3 class="player-stat-card-title">Similar Players</h3>
        <p class="player-similar-sub">Closest matches by GAR proxy and zone-start usage.</p>
        <div class="similar-grid">
          ${data.similar.map((s) => `
            <div class="similar-card" data-name="${esc(s.name)}">
              <div class="similar-name">${esc(s.name)}</div>
              <div class="similar-meta">${esc(s.team)} &middot; ${esc(s.position)}</div>
              <div class="similar-stat"><span>GAR</span> <strong>${fmt(s.gar)}</strong></div>
              <div class="similar-stat"><span>GP</span> <strong>${fmt(s.games)}</strong></div>
            </div>
          `).join("")}
        </div>
      `;
    } catch (e) {
      const wrap = document.getElementById("player-similar-wrap");
      if (wrap) wrap.innerHTML = '<p class="muted">Failed to load similar players.</p>';
    }
  }

  // ---------------------------------------------------------------------------
  // Analytics section — composite rating + RAPM (Phase 3)
  // ---------------------------------------------------------------------------
  let _compositeCache = null;
  let _rapmCache = null;

  async function getComposite() {
    if (_compositeCache) return _compositeCache;
    try {
      const r = await fetch("/api/composite-ratings");
      _compositeCache = await r.json();
    } catch (e) { _compositeCache = { built: false }; }
    return _compositeCache;
  }
  async function getRapm() {
    if (_rapmCache) return _rapmCache;
    try {
      const r = await fetch("/api/rapm-leaders");
      _rapmCache = await r.json();
    } catch (e) { _rapmCache = { trained: false }; }
    return _rapmCache;
  }

  const COMPONENT_LABELS = [
    ["rapm_component", "RAPM"],
    ["individual_component", "Individual"],
    ["relative_xgf_component", "Relative xGF%"],
    ["playmaking_component", "Playmaking"],
    ["power_play_component", "Power Play"],
    ["penalty_kill_component", "Penalty Kill"],
  ];

  function componentBar(label, z) {
    const v = z == null ? 0 : z;
    const mag = Math.min(Math.abs(v) / 3, 1) * 50;   // % of half-track
    const cls = v >= 0 ? "pa-bar-pos" : "pa-bar-neg";
    const style = v >= 0
      ? `left:50%; width:${mag}%`
      : `left:${50 - mag}%; width:${mag}%`;
    return `
      <div class="pa-bar-row">
        <span class="pa-bar-label">${esc(label)}</span>
        <div class="pa-bar-track">
          <div class="pa-bar-center"></div>
          <div class="pa-bar-fill ${cls}" style="${style}"></div>
        </div>
        <span class="pa-bar-val">${v >= 0 ? "+" : ""}${v.toFixed(2)}</span>
      </div>`;
  }

  function buildInterpretation(row, position) {
    const c = {
      RAPM: row.rapm_component, individual: row.individual_component,
      "two-way play-driving": row.relative_xgf_component, playmaking: row.playmaking_component,
      "power play": row.power_play_component, "penalty kill": row.penalty_kill_component,
    };
    const comp = row.composite_rating;
    const posWord = position === "D" ? "defenseman" : "forward";
    const tier = comp >= 1.5 ? "Elite" : comp >= 0.7 ? "Strong" : comp >= 0.2 ? "Solid"
               : comp >= -0.3 ? "Average" : "Below-average";
    const strong = Object.entries(c).filter(([, v]) => v >= 1.0).map(([k]) => k);
    const weak = Object.entries(c).filter(([, v]) => v <= -0.6).map(([k]) => k);
    const aboveAvg = Object.values(c).filter((v) => v >= 0.3).length;
    const offense = (c.individual + c.playmaking + c["power play"]) / 3;
    const defense = (c["penalty kill"] + c["two-way play-driving"]) / 2;

    let archetype;
    if (aboveAvg >= 5) {
      archetype = `two-way ${posWord} with above-average contributions across nearly all six components`;
    } else if (offense > 0.8 && defense < 0.25) {
      archetype = `offensive ${posWord} whose value is concentrated in ${strong.slice(0, 2).join(" and ") || "individual offense"}`;
    } else if (defense > 0.6 && offense < 0.3) {
      archetype = `defensive ${posWord} — value concentrated in chance suppression and special-teams defense`;
    } else if (strong.length) {
      archetype = `${posWord} whose standout strengths are ${strong.join(", ")}`;
    } else {
      archetype = `${posWord} with a balanced but unremarkable analytical profile`;
    }
    let s = `${tier} ${archetype}.`;
    if (weak.length) s += ` Relative weakness: ${weak.join(", ")}.`;
    return s;
  }

  async function loadAnalytics(playerId, isGoalie, name) {
    const wrap = document.getElementById("player-analytics");
    if (!wrap) return;

    if (isGoalie) {
      wrap.innerHTML = `
        <h3 class="player-stat-card-title">Analytics</h3>
        <p class="muted">Composite Rating and RAPM are skater metrics — not computed for goalies.
        See the Goalies page for goalie-specific advanced stats.</p>`;
      return;
    }

    const [comp, rapm] = await Promise.all([getComposite(), getRapm()]);

    const compRows = (comp && comp.built && comp.full_dataset) ? comp.full_dataset : [];
    const qualified = compRows.filter((r) => r.sample_size_flag === "ok");
    const row = qualified.find((r) => r.player_id === playerId);
    const rank = row ? qualified.findIndex((r) => r.player_id === playerId) + 1 : null;

    const rapmRows = (rapm && rapm.trained && (rapm.full_dataset || [])) || [];
    const rapmRow = rapmRows.find((r) => r.player_id === playerId);

    if (!row) {
      wrap.innerHTML = `
        <h3 class="player-stat-card-title">Analytics <span class="subtitle">— composite rating model</span></h3>
        <p class="muted">${esc(name)} did not clear the 300 even-strength-minute threshold this
        season, so no composite rating was computed.</p>`;
      return;
    }

    const compCls = row.composite_rating > 1.5 ? "lb-comp-green"
                  : row.composite_rating >= 0.5 ? "lb-comp-gold"
                  : row.composite_rating >= 0 ? "lb-comp-neutral" : "lb-comp-red";
    const bars = COMPONENT_LABELS.map(([k, lbl]) => componentBar(lbl, row[k])).join("");
    const interp = buildInterpretation(row, row.position);

    let rapmHtml = '<p class="muted">RAPM not available for this player.</p>';
    if (rapmRow) {
      const f = (v) => (v >= 0 ? "+" : "") + Number(v).toFixed(3);
      rapmHtml = `
        <div class="pa-rapm-row">
          <div class="pa-rapm-stat"><span>Total RAPM</span><strong>${f(rapmRow.total_rapm)}</strong></div>
          <div class="pa-rapm-stat"><span>Offensive</span><strong>${f(rapmRow.offensive_rapm)}</strong></div>
          <div class="pa-rapm-stat"><span>Defensive</span><strong>${f(rapmRow.defensive_rapm)}</strong></div>
        </div>`;
    }

    // 2026-06-07: section was titled "Composite Rating" until GAR was
    // unified with composite_war on the same display scale. We now show the
    // six component z-scores under a "GAR Component Breakdown" header so the
    // user sees exactly how their GAR was constructed.
    // REBUILD HIDDEN: the GAR Component Breakdown (composite_war headline + the six
    // component z-score bars) is a DERIVED, not-rebuilt surface — hidden until the
    // composite rebuild. The RAPM block below is the rebuilt 5v5 RAPM and stays.
    // (compCls / row.composite_rating / bars / interp retained above for restore.)
    wrap.innerHTML = `
      <h3 class="player-stat-card-title">RAPM <span class="subtitle">— 2025-26 5v5 (xG/60)</span></h3>
      <div class="pa-section-label">RAPM — even-strength impact</div>
      ${rapmHtml}
    `;
  }

  // ---------------------------------------------------------------------------
  // Render the full profile
  // ---------------------------------------------------------------------------
  function renderProfile(p) {
    const wrap = $("#player-lookup-profile");
    const empty = $("#player-lookup-empty");
    const isG = p.bio.is_goalie;

    wrap.innerHTML = `
      ${profileTopHTML(p)}

      <div class="player-stats-grid">
        ${statTableHTML(`Current Season ${p.current_season.season ? `(${formatSeason(p.current_season.season)})` : ""}`,
          isG ? goalieCurrentSeasonRows(p.current_season) : skaterCurrentSeasonRows(p.current_season))}
        ${statTableHTML("Advanced Stats", isG ? goalieAdvancedRows(p.advanced) : skaterAdvancedRows(p.advanced))}
      </div>

      <div class="player-stat-card player-analytics-card" id="player-analytics">
        <h3 class="player-stat-card-title">RAPM <span class="subtitle">— 2025-26 5v5 (xG/60)</span></h3>
        <p class="muted">Loading…</p>
      </div>

      ${last5HTML(p)}
      ${shotChartCardHTML()}
      ${careerTableHTML(p, "regular")}
      ${careerTableHTML(p, "playoff")}

      <div class="player-stat-card" id="player-similar-wrap">
        <h3 class="player-stat-card-title">Similar Players</h3>
        <p class="muted">Loading…</p>
      </div>

      <div class="player-bottom-actions">
        <a href="?ptab=compare&p1=${encodeURIComponent(slug(p.bio.name))}" class="btn-eh" id="player-compare-btn">Compare with another player →</a>
      </div>
    `;
    empty.hidden = true;
    wrap.hidden = false;

    // Always load league rankings under the headshot
    loadRankings(p.bio.id);

    // Analytics section — composite rating + RAPM (skaters only)
    loadAnalytics(p.bio.id, isG, p.bio.name);

    // Skaters get shot chart; for goalies, show a skip note
    if (!isG) {
      loadShotChart(p.bio.id);
      loadSimilar(p.bio.id);
    } else {
      const status = document.getElementById("player-shot-chart-status");
      if (status) status.textContent = "Shot map is for skaters only.";
      const sim = document.getElementById("player-similar-wrap");
      if (sim) sim.innerHTML = '<h3 class="player-stat-card-title">Similar Players</h3><p class="muted">Similar-goalie matching coming soon.</p>';
    }

    // Compare-button: jump to compare tab with player pre-loaded
    const cmpBtn = document.getElementById("player-compare-btn");
    if (cmpBtn) {
      cmpBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const url = new URL(window.location.href);
        url.searchParams.set("ptab", "compare");
        url.searchParams.set("p1", slug(p.bio.name));
        history.replaceState({}, "", url);
        const tab = document.querySelector('.players-tab[data-ptab="compare"]');
        if (tab) tab.click();
        if (window.playerCompare && window.playerCompare.loadPlayerByIdIntoSlot) {
          window.playerCompare.loadPlayerByIdIntoSlot(p.bio.id, 1);
        }
      });
    }
  }

  function formatSeason(s) {
    if (!s) return "";
    const ss = String(s);
    if (ss.length === 8) return `${ss.slice(0, 4)}-${ss.slice(6, 8)}`;
    return ss;
  }

  function slug(name) {
    return String(name || "")
      .normalize("NFKD")
      .replace(/[̀-ͯ]/g, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  // ---------------------------------------------------------------------------
  // Load profile by ID
  // ---------------------------------------------------------------------------
  async function loadProfile(playerId, playerName) {
    const empty = $("#player-lookup-empty");
    const error = $("#player-lookup-error");
    const wrap = $("#player-lookup-profile");
    const loading = $("#player-lookup-loading");

    empty.hidden = true;
    error.hidden = true;
    wrap.hidden = true;
    loading.hidden = false;

    try {
      const r = await fetch(`/api/player/${playerId}`);
      const data = await r.json();
      loading.hidden = true;
      if (data.error) {
        error.textContent = data.error;
        error.hidden = false;
        empty.hidden = false;
        return;
      }
      renderProfile(data);

      // Update URL with ?player=slug
      const url = new URL(window.location.href);
      url.searchParams.set("player", slug(data.bio.name));
      history.replaceState({}, "", url);
    } catch (e) {
      loading.hidden = true;
      error.textContent = "Failed to load player profile.";
      error.hidden = false;
      empty.hidden = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    const input = $("#player-lookup-search");
    const dropdown = $("#player-lookup-dropdown");
    if (!input || !dropdown) return;

    attachLiveSearch(input, dropdown, (player) => {
      loadProfile(player.id, player.name);
    });

    // URL param: load player on init if ?player=slug specified
    const params = new URLSearchParams(window.location.search);
    const initSlug = params.get("player");
    if (initSlug) {
      // Use search to resolve slug→id (search by reformatted slug)
      const query = initSlug.replace(/-/g, " ");
      fetch(`/api/player-search?q=${encodeURIComponent(query)}&limit=10`)
        .then((r) => r.json())
        .then((data) => {
          const match = (data.results || []).find((p) => slug(p.name) === initSlug) || (data.results || [])[0];
          if (match) {
            input.value = match.name;
            loadProfile(match.id, match.name);
            // Switch to Player Search tab
            const tab = document.querySelector('.players-tab[data-ptab="player-search"]');
            if (tab) tab.click();
          }
        });
    }

    // Click on similar-player card jumps to that player's profile
    document.addEventListener("click", (e) => {
      const card = e.target.closest(".similar-card");
      if (!card) return;
      const name = card.dataset.name;
      if (!name) return;
      // Search and load
      fetch(`/api/player-search?q=${encodeURIComponent(name)}&limit=5`)
        .then((r) => r.json())
        .then((data) => {
          const m = (data.results || []).find((p) => p.name === name) || (data.results || [])[0];
          if (m) {
            input.value = m.name;
            loadProfile(m.id, m.name);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }
        });
    });
  }

  // Expose for player_compare.js
  window.playerLookup = {
    attachLiveSearch,
    loadProfile,
    slug,
    fmt,
    fmtPct,
    fmtToi,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
