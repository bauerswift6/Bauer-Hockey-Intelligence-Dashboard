/* =========================================================
   Player Compare — dual/triple comparison with radar chart,
   season trend, shot charts, career trajectory, contract value.
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = window.dashUtil ? window.dashUtil.escHtml : (s) => String(s ?? "");
  const lookup = window.playerLookup;

  const COLORS = ["#2B6BF0", "#E8A020", "#1DB954"]; // Player 1, 2, 3

  const state = {
    players: [null, null, null],   // slots 1, 2, 3 (index 0/1/2)
    template: "all",
    thirdEnabled: false,
    radarChart: null,
    trendChart: null,
    careerChart: null,
  };

  // ---------------------------------------------------------------------------
  // Helpers — re-use formatters from playerLookup
  // ---------------------------------------------------------------------------
  const fmt = lookup ? lookup.fmt : (v) => v ?? "—";
  const fmtPct = lookup ? lookup.fmtPct : (v) => v ?? "—";
  const fmtToi = lookup ? lookup.fmtToi : (v) => v ?? "—";
  const slug = lookup ? lookup.slug : (s) => String(s).toLowerCase().replace(/\s+/g, "-");

  function tip(text) {
    return `<span class="stat-tip" title="${esc(text)}">${esc(text)}</span>`;
  }

  // ---------------------------------------------------------------------------
  // Bio strip
  // ---------------------------------------------------------------------------
  function bioStripHTML(players) {
    const cells = players.filter(Boolean).map((p, i) => {
      const c = COLORS[i];
      return `
        <div class="compare-bio-cell" style="border-top: 4px solid ${c}">
          <img class="compare-bio-headshot" src="${esc(p.bio.headshot)}" alt="${esc(p.bio.name)}" onerror="this.style.display='none'" />
          <div class="compare-bio-name">${esc(p.bio.name)}</div>
          <div class="compare-bio-team">
            <img class="compare-bio-logo" src="${esc(p.bio.team_logo)}" alt="${esc(p.bio.team)}" onerror="this.style.display='none'" />
            <span>${esc(p.bio.team)}</span>
            <span>&middot; ${esc(p.bio.position)}</span>
          </div>
          <div class="compare-bio-meta">
            Age ${fmt(p.bio.age)} &middot; ${esc(p.bio.height)} &middot; ${esc(p.bio.weight)}
          </div>
        </div>
      `;
    }).join("");
    return `<div class="compare-bio-strip">${cells}</div>`;
  }

  // ---------------------------------------------------------------------------
  // Stats comparison table — shared row builder
  // Each rowDef: { label, key, isGoalie?, getValue(p), better: 'high'|'low', tip?, fmt?, dp? }
  // ---------------------------------------------------------------------------
  function compareTable(title, rowDefs, players) {
    const slots = players.filter(Boolean);
    if (!slots.length) return "";

    // Determine leader for each row
    const rowsHtml = rowDefs.map((rd) => {
      const vals = slots.map((p) => rd.getValue(p));
      // Identify the best — ignore null/undefined
      let bestIdx = -1;
      let bestVal = null;
      vals.forEach((v, i) => {
        if (v == null || isNaN(v)) return;
        if (bestVal == null) { bestVal = v; bestIdx = i; return; }
        if (rd.better === "low" ? v < bestVal : v > bestVal) {
          bestVal = v; bestIdx = i;
        }
      });
      // Detect ties
      const tieIdxs = vals
        .map((v, i) => (v != null && !isNaN(v) && v === bestVal) ? i : -1)
        .filter((i) => i >= 0);
      const isTie = tieIdxs.length > 1;

      const cells = vals.map((v, i) => {
        let display;
        if (rd.fmtFn) display = rd.fmtFn(v);
        else if (rd.dp != null && typeof v === "number") display = v.toFixed(rd.dp);
        else display = (v == null) ? "—" : String(v);

        let cls = "cmp-val";
        if (isTie && tieIdxs.includes(i)) cls += " cmp-tie";
        else if (i === bestIdx) cls += " cmp-leader";
        else cls += " cmp-loser";

        return `<td class="${cls}">${esc(display)}</td>`;
      }).join("");

      const labelHtml = rd.tip
        ? `<td class="cmp-label" title="${esc(rd.tip)}">${esc(rd.label)} <span class="cmp-info">i</span></td>`
        : `<td class="cmp-label">${esc(rd.label)}</td>`;

      return `<tr>${labelHtml}${cells}</tr>`;
    }).join("");

    const headerCells = slots.map((p, i) => {
      const c = COLORS[i];
      return `<th style="color: ${c}">${esc(p.bio.name)}</th>`;
    }).join("");

    return `
      <div class="compare-stat-block">
        <h3 class="compare-stat-block-title">${esc(title)}</h3>
        <div class="table-wrap">
          <table class="data-table compare-stat-table">
            <thead><tr><th>Stat</th>${headerCells}</tr></thead>
            <tbody>${rowsHtml}</tbody>
          </table>
        </div>
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Row defs (skater traditional + advanced)
  // ---------------------------------------------------------------------------
  const TRAD_ROWS = [
    { label: "GP", getValue: (p) => p.current_season?.gp, better: "high" },
    { label: "G", getValue: (p) => p.current_season?.g, better: "high" },
    { label: "A", getValue: (p) => p.current_season?.a, better: "high" },
    { label: "PTS", getValue: (p) => p.current_season?.pts, better: "high" },
    { label: "+/-", getValue: (p) => p.current_season?.plus_minus, better: "high" },
    { label: "PIM", getValue: (p) => p.current_season?.pim, better: "low" },
    { label: "Shots", getValue: (p) => p.current_season?.shots, better: "high" },
    { label: "Shooting %", getValue: (p) => p.current_season?.shooting_pct, better: "high",
      fmtFn: (v) => v == null ? "—" : (v < 1 ? (v * 100).toFixed(1) + "%" : v.toFixed(1) + "%") },
    { label: "TOI/GP", getValue: (p) => p.current_season?.toi_per_game, better: "high",
      fmtFn: (v) => v ? String(v) : "—" },
    { label: "PP Points", getValue: (p) => p.current_season?.pp_pts, better: "high" },
  ];

  const ADV_ROWS = [
    { label: "GAR", getValue: (p) => p.advanced?.gar, better: "high",
      tip: "Goals Above Replacement — total goal value vs a replacement-level player.", dp: 2 },
    { label: "xGAR", getValue: (p) => p.advanced?.xgar, better: "high",
      tip: "Expected GAR — luck-stripped GAR built from expected goals.", dp: 2 },
    { label: "iCF / 60", getValue: (p) => p.advanced?.icf_60, better: "high",
      tip: "Individual Corsi For per 60 — shot-attempt rate.", dp: 2 },
    { label: "iFF (total)", getValue: (p) => p.advanced?.iff, better: "high",
      tip: "Individual Fenwick For — total unblocked attempts." },
    { label: "ixG / 60", getValue: (p) => p.advanced?.ixg_60, better: "high",
      tip: "Individual Expected Goals per 60.", dp: 2 },
    { label: "iHDCF (total)", getValue: (p) => p.advanced?.ihdcf, better: "high",
      tip: "Individual high-danger chances for." },
    { label: "On-Ice xGF%", getValue: (p) => p.advanced?.onice_xgf_pct, better: "high",
      tip: "Expected goals share with player on-ice.", fmtFn: (v) => v == null ? "—" : v.toFixed(1) + "%" },
    { label: "On-Ice CF%", getValue: (p) => p.advanced?.onice_cf_pct, better: "high",
      tip: "Corsi share with player on-ice.", fmtFn: (v) => v == null ? "—" : v.toFixed(1) + "%" },
    { label: "Zone Start %", getValue: (p) => p.advanced?.zone_start_pct, better: "high",
      tip: "Share of non-neutral starts in offensive zone.", fmtFn: (v) => v == null ? "—" : v.toFixed(1) + "%" },
  ];

  // Goalie row defs — used when both selected players are goalies
  const GOALIE_TRAD_ROWS = [
    { label: "GP", getValue: (p) => p.current_season?.gp, better: "high" },
    { label: "GS", getValue: (p) => p.current_season?.gs, better: "high" },
    { label: "W", getValue: (p) => p.current_season?.w, better: "high" },
    { label: "L", getValue: (p) => p.current_season?.l, better: "low" },
    { label: "OTL", getValue: (p) => p.current_season?.otl, better: "low" },
    { label: "GAA", getValue: (p) => p.current_season?.gaa, better: "low", dp: 2 },
    { label: "SV%", getValue: (p) => p.current_season?.sv_pct, better: "high",
      fmtFn: (v) => v == null ? "—" : (v * 100).toFixed(2) + "%" },
    { label: "SO", getValue: (p) => p.current_season?.so, better: "high" },
  ];

  const GOALIE_ADV_ROWS = [
    { label: "GSAX", getValue: (p) => p.advanced?.gsax, better: "high", dp: 2,
      tip: "Goals Saved Above Expected." },
    { label: "GSAx / 60", getValue: (p) => p.advanced?.gsax_60, better: "high", dp: 3,
      tip: "GSAX per 60 minutes." },
    { label: "HDSV%", getValue: (p) => p.advanced?.hdsv_pct, better: "high",
      tip: "Save % on high-danger shots.", fmtFn: (v) => v == null ? "—" : v.toFixed(2) + "%" },
    { label: "MDSV%", getValue: (p) => p.advanced?.mdsv_pct, better: "high",
      tip: "Save % on medium-danger shots.", fmtFn: (v) => v == null ? "—" : v.toFixed(2) + "%" },
    { label: "Quality Start %", getValue: (p) => p.advanced?.qs_pct, better: "high",
      tip: "Share of starts above quality threshold.", fmtFn: (v) => v == null ? "—" : v.toFixed(1) + "%" },
  ];

  // ---------------------------------------------------------------------------
  // Radar chart (Chart.js)
  // ---------------------------------------------------------------------------
  function destroyChart(c) {
    if (c) try { c.destroy(); } catch (_) {}
  }

  function renderRadar(players) {
    const canvas = document.getElementById("compare-radar");
    if (!canvas) return;
    destroyChart(state.radarChart);

    // Six axes:
    // Offense (ixG/60), Defense (inverted xGA/60), Possession (CF%), Shot Quality (HDCF%),
    // Durability (GP), Value (GAR)
    // Each player normalized to 0-100 against a fixed scale max for that axis.
    const SCALES = {
      offense: 1.5,         // ixG/60 max
      defense: 3.0,         // xGA/60 max (inverted)
      possession: 60,       // on-ice CF% max
      shot_quality: 60,     // on-ice xGF% as proxy for shot quality
      durability: 82,       // GP max
      value: 18,            // GAR max
    };

    const slots = players.filter(Boolean).slice(0, 3);
    const datasets = slots.map((p, i) => {
      const adv = p.advanced || {};
      const cs = p.current_season || {};
      const offense = clamp((adv.ixg_60 || 0) / SCALES.offense * 100, 0, 100);
      // No xGA/60 in our advanced row — proxy: 100 - on-ice xGF% (lower xGA → higher)
      const defense = clamp((adv.onice_xgf_pct ? (adv.onice_xgf_pct - 40) * 5 : 0), 0, 100);
      const possession = clamp(((adv.onice_cf_pct || 0) - 40) * 5, 0, 100);
      const shot_quality = clamp(((adv.onice_xgf_pct || 0) - 40) * 5, 0, 100);
      const durability = clamp((cs.gp || 0) / SCALES.durability * 100, 0, 100);
      const value = clamp((adv.gar || 0) / SCALES.value * 100, 0, 100);

      return {
        label: p.bio.name,
        data: [offense, defense, possession, shot_quality, durability, value],
        backgroundColor: hexToRgba(COLORS[i], 0.18),
        borderColor: COLORS[i],
        borderWidth: 2,
        pointBackgroundColor: COLORS[i],
        pointRadius: 4,
      };
    });

    state.radarChart = new Chart(canvas, {
      type: "radar",
      data: {
        labels: ["Offense (ixG/60)", "Defense (xGA inv)", "Possession (CF%)", "Shot Quality (xGF%)", "Durability (GP)", "Value (GAR)"],
        datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: "rgba(120, 140, 180, 0.18)" },
            grid: { color: "rgba(120, 140, 180, 0.18)" },
            pointLabels: { color: "#C0CCDD", font: { size: 11 } },
            ticks: { display: false, beginAtZero: true, max: 100 },
            min: 0, max: 100,
          },
        },
        plugins: {
          legend: { labels: { color: "#F0F4FF" } },
        },
      },
    });
  }

  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }
  function hexToRgba(hex, a) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${a})`;
  }

  // ---------------------------------------------------------------------------
  // Season trend chart — cumulative points across last-N games
  // ---------------------------------------------------------------------------
  function renderTrend(players) {
    const canvas = document.getElementById("compare-trend");
    if (!canvas) return;
    destroyChart(state.trendChart);

    const slots = players.filter(Boolean).slice(0, 3);
    // We use the last5 game log as a small but real trend.
    // Each player's last5 list comes most-recent first; reverse to chronological.
    const datasets = slots.map((p, i) => {
      const games = (p.last5 || []).slice().reverse();
      let cum = 0;
      const data = games.map((g) => {
        cum += (g.points || ((g.goals || 0) + (g.assists || 0)) || 0);
        return cum;
      });
      return {
        label: p.bio.name,
        data,
        borderColor: COLORS[i],
        backgroundColor: hexToRgba(COLORS[i], 0.12),
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 4,
      };
    });

    // Use the longest game count for x labels
    const maxLen = datasets.reduce((m, d) => Math.max(m, d.data.length), 0);
    const labels = Array.from({ length: maxLen }, (_, i) => `Game ${i + 1}`);

    state.trendChart = new Chart(canvas, {
      type: "line",
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.1)" } },
          y: { beginAtZero: true, ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.1)" }, title: { text: "Cumulative Points", display: true, color: "#C0CCDD" } },
        },
        plugins: { legend: { labels: { color: "#F0F4FF" } } },
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Career trajectory — last 5 NHL regular seasons, points (skaters) or wins (goalies)
  // ---------------------------------------------------------------------------
  function renderCareerTrajectory(players) {
    const canvas = document.getElementById("compare-career");
    if (!canvas) return;
    destroyChart(state.careerChart);

    const slots = players.filter(Boolean).slice(0, 3);

    // Build a unified set of last-5 NHL seasons across all players
    const seasonSet = new Set();
    slots.forEach((p) => {
      (p.career_regular || [])
        .filter((r) => r.league === "NHL")
        .slice(-6)
        .forEach((r) => seasonSet.add(r.season));
    });
    const seasons = Array.from(seasonSet).sort().slice(-6);

    const labelize = (s) => {
      const ss = String(s);
      if (ss.length === 8) return `${ss.slice(0, 4)}-${ss.slice(6, 8)}`;
      return ss;
    };

    const datasets = slots.map((p, i) => {
      const isG = p.bio.is_goalie;
      const lookup = {};
      (p.career_regular || []).forEach((r) => { lookup[r.season] = r; });
      const data = seasons.map((s) => {
        const r = lookup[s];
        if (!r) return null;
        return isG ? (r.w || 0) : (r.pts || 0);
      });
      return {
        label: p.bio.name,
        data,
        backgroundColor: COLORS[i],
        borderColor: COLORS[i],
        borderWidth: 1,
      };
    });

    state.careerChart = new Chart(canvas, {
      type: "bar",
      data: { labels: seasons.map(labelize), datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.08)" } },
          y: { beginAtZero: true, ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.1)" }, title: { text: slots[0]?.bio.is_goalie ? "Wins" : "Points", display: true, color: "#C0CCDD" } },
        },
        plugins: { legend: { labels: { color: "#F0F4FF" } } },
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Shot location comparison — side-by-side mini rinks
  // ---------------------------------------------------------------------------
  async function renderShotComparison(players) {
    const wrap = document.getElementById("compare-shots");
    if (!wrap) return;
    const slots = players.filter(Boolean).slice(0, 3);
    wrap.innerHTML = slots.map((p, i) => `
      <div class="compare-shot-cell" data-player-id="${esc(p.bio.id)}">
        <div class="compare-shot-name" style="color: ${COLORS[i]}">${esc(p.bio.name)}</div>
        <svg class="compare-shot-svg" data-player-id="${esc(p.bio.id)}" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet"></svg>
        <div class="compare-shot-status muted" data-player-id="${esc(p.bio.id)}">Loading…</div>
      </div>
    `).join("");

    for (const p of slots) {
      if (p.bio.is_goalie) {
        const status = wrap.querySelector(`.compare-shot-status[data-player-id="${p.bio.id}"]`);
        if (status) status.textContent = "Goalie — no shot map.";
        continue;
      }
      try {
        const r = await fetch(`/api/player/${p.bio.id}/shot-chart?games=10`);
        const data = await r.json();
        renderShotsSVG(p.bio.id, data.shots || []);
      } catch (e) {
        const status = wrap.querySelector(`.compare-shot-status[data-player-id="${p.bio.id}"]`);
        if (status) status.textContent = "Failed to load.";
      }
    }
  }

  function renderShotsSVG(playerId, shots) {
    const svg = document.querySelector(`.compare-shot-svg[data-player-id="${playerId}"]`);
    const status = document.querySelector(`.compare-shot-status[data-player-id="${playerId}"]`);
    if (!svg) return;
    if (!shots.length) {
      if (status) status.textContent = "No shots tracked in last 10 games.";
      return;
    }
    if (status) status.style.display = "none";

    function project(x, y) {
      let rx = x, ry = y;
      if (rx < 0) { rx = -rx; ry = -ry; }
      const svgX = (rx + 10) * (400 / 110);
      const svgY = (-ry + 42.5) * (300 / 85);
      return [svgX, svgY];
    }

    const rink = `
      <rect x="0" y="0" width="400" height="300" rx="14" ry="14" fill="#0A1525" stroke="#0E1E35" stroke-width="2" />
      <line x1="36.4" y1="0" x2="36.4" y2="300" stroke="#E8302A" stroke-width="2" opacity="0.6" />
      <line x1="127.3" y1="0" x2="127.3" y2="300" stroke="#2B6BF0" stroke-width="3" opacity="0.7" />
      <line x1="360" y1="40" x2="360" y2="260" stroke="#E8302A" stroke-width="1.5" opacity="0.7" />
      <path d="M 360 130 A 22 22 0 0 0 360 170 Z" fill="rgba(43,107,240,0.12)" stroke="#2B6BF0" stroke-width="1" opacity="0.5" />
      <circle cx="295" cy="80" r="20" fill="none" stroke="#E8302A" stroke-width="1" opacity="0.5" />
      <circle cx="295" cy="220" r="20" fill="none" stroke="#E8302A" stroke-width="1" opacity="0.5" />
    `;

    function dotCls(t) {
      if (t === "goal") return "shot-dot-svg shot-goal";
      if (t === "shot-on-goal") return "shot-dot-svg shot-onnet";
      return "shot-dot-svg shot-miss";
    }

    const dots = shots.map((s) => {
      const [cx, cy] = project(s.x, s.y);
      const r = s.type === "goal" ? 5 : 3.5;
      return `<circle class="${dotCls(s.type)}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${r}"><title>${esc(s.type + " vs " + s.opp)}</title></circle>`;
    }).join("");

    svg.innerHTML = rink + dots;
  }

  // ---------------------------------------------------------------------------
  // Contract value — pulled from /api/contract-values, matched by name
  // ---------------------------------------------------------------------------
  let contractCache = null;
  async function getContracts() {
    if (contractCache) return contractCache;
    try {
      const r = await fetch("/api/contract-values");
      contractCache = await r.json();
    } catch (e) { contractCache = { players: [] }; }
    return contractCache;
  }

  async function renderContractValue(players) {
    const wrap = document.getElementById("compare-contract");
    if (!wrap) return;
    const slots = players.filter(Boolean).slice(0, 3);
    const data = await getContracts();
    const matchByName = (name) => {
      const norm = String(name || "").toLowerCase().replace(/[^a-z]/g, "");
      return (data.players || []).find((c) => String(c.name).toLowerCase().replace(/[^a-z]/g, "") === norm);
    };

    const cells = slots.map((p, i) => {
      const c = matchByName(p.bio.name);
      const cap = c?.cap_hit;
      const gar = c?.gar ?? p.advanced?.gar ?? null;
      const xgar = c?.xgar ?? p.advanced?.xgar ?? null;
      const garPerMil = c?.gar_per_million ?? (cap > 0 && gar != null ? +(gar / cap).toFixed(2) : null);
      const xgarPerMil = (cap > 0 && xgar != null) ? +(xgar / cap).toFixed(2) : null;
      return {
        idx: i,
        name: p.bio.name,
        cap: cap,
        gar,
        xgar,
        garPerMil,
        xgarPerMil,
      };
    });

    // Verdict — only when 2 players selected
    let verdict = "";
    if (cells.length >= 2 && cells[0].garPerMil != null && cells[1].garPerMil != null) {
      const v1 = cells[0].garPerMil;
      const v2 = cells[1].garPerMil;
      if (v1 > 0 && v2 > 0) {
        if (v1 > v2) {
          const pct = Math.round((v1 / v2 - 1) * 100);
          verdict = `<strong>${esc(cells[0].name)}</strong> provides ${pct}% more value per dollar of cap hit than <strong>${esc(cells[1].name)}</strong>.`;
        } else if (v2 > v1) {
          const pct = Math.round((v2 / v1 - 1) * 100);
          verdict = `<strong>${esc(cells[1].name)}</strong> provides ${pct}% more value per dollar of cap hit than <strong>${esc(cells[0].name)}</strong>.`;
        } else {
          verdict = "Both players deliver identical value per dollar of cap hit.";
        }
      }
    }

    wrap.innerHTML = `
      <h3 class="compare-stat-block-title">Contract Value</h3>
      <div class="contract-grid">
        ${cells.map((c) => `
          <div class="contract-card" style="border-top: 4px solid ${COLORS[c.idx]}">
            <div class="contract-name">${esc(c.name)}</div>
            <div class="contract-row"><span>Cap Hit</span><strong>${c.cap != null ? "$" + (c.cap).toFixed(2) + "M" : "—"}</strong></div>
            <div class="contract-row"><span>GAR / $1M</span><strong>${c.garPerMil != null ? c.garPerMil : "—"}</strong></div>
            <div class="contract-row"><span>xGAR / $1M</span><strong>${c.xgarPerMil != null ? c.xgarPerMil : "—"}</strong></div>
          </div>
        `).join("")}
      </div>
      ${verdict ? `<p class="contract-verdict">${verdict}</p>` : `<p class="muted">Cap-hit data unavailable for one or more selected players.</p>`}
    `;
  }

  // ---------------------------------------------------------------------------
  // "You Might Also Compare" — suggested matchups by position+age
  // ---------------------------------------------------------------------------
  async function renderSuggestedMatchups(players) {
    const wrap = document.getElementById("compare-suggested");
    if (!wrap) return;
    const slots = players.filter(Boolean).slice(0, 2);
    if (slots.length < 2) {
      wrap.innerHTML = "";
      return;
    }
    const pos = slots[0].bio.position;
    const ageBand = slots[0].bio.age != null ? `${Math.max(20, slots[0].bio.age - 2)}-${slots[0].bio.age + 2}` : "any";

    // Use similar-player API on player 1 to find matching peers
    let suggestions = [];
    try {
      const r = await fetch(`/api/player/${slots[0].bio.id}/similar`);
      const data = await r.json();
      suggestions = (data.similar || []).slice(0, 3);
    } catch (e) {}

    if (!suggestions.length) {
      wrap.innerHTML = "";
      return;
    }

    wrap.innerHTML = `
      <h3 class="compare-stat-block-title">You Might Also Compare</h3>
      <p class="muted">${esc(`Other ${pos === 'G' ? 'goalies' : 'skaters'} similar to ${slots[0].bio.name}`)}</p>
      <div class="similar-grid">
        ${suggestions.map((s) => `
          <div class="similar-card suggest-card" data-name="${esc(s.name)}">
            <div class="similar-name">${esc(s.name)}</div>
            <div class="similar-meta">${esc(s.team)} &middot; ${esc(s.position)}</div>
            <div class="similar-stat"><span>GAR</span> <strong>${s.gar != null ? s.gar : "—"}</strong></div>
            <button class="add-suggested-btn">Compare with ${esc(slots[0].bio.name.split(' ').pop())}</button>
          </div>
        `).join("")}
      </div>
    `;
  }

  // ---------------------------------------------------------------------------
  // Rendering pipeline
  // ---------------------------------------------------------------------------
  function showResults(players) {
    const results = $("#compare-results");
    const empty = $("#compare-empty");
    const filled = players.filter(Boolean);
    if (filled.length < 2) {
      results.hidden = true;
      results.innerHTML = "";
      empty.hidden = false;
      return;
    }
    empty.hidden = true;

    const isGoalieCompare = filled.every((p) => p.bio.is_goalie);
    const tradRows = isGoalieCompare ? GOALIE_TRAD_ROWS : TRAD_ROWS;
    const advRows = isGoalieCompare ? GOALIE_ADV_ROWS : ADV_ROWS;

    // Apply template filter — hide irrelevant rows
    const filteredTrad = filterByTemplate(tradRows, state.template);
    const filteredAdv = filterByTemplate(advRows, state.template);

    results.innerHTML = `
      ${bioStripHTML(players)}

      ${compareTable("Traditional Stats", filteredTrad, players)}
      ${compareTable("Advanced Stats", filteredAdv, players)}

      <div class="compare-charts-grid">
        <div class="compare-chart-card">
          <h3 class="compare-stat-block-title">Radar Comparison</h3>
          <div class="chart-wrap radar-wrap"><canvas id="compare-radar"></canvas></div>
        </div>
        <div class="compare-chart-card">
          <h3 class="compare-stat-block-title">Season Trend <span class="subtitle">— cumulative points (last games)</span></h3>
          <div class="chart-wrap trend-wrap"><canvas id="compare-trend"></canvas></div>
        </div>
      </div>

      <div class="compare-stat-block">
        <h3 class="compare-stat-block-title">Shot Location Comparison <span class="subtitle">— last 10 games</span></h3>
        <div class="compare-shots-grid" id="compare-shots"></div>
      </div>

      <div class="compare-stat-block">
        <h3 class="compare-stat-block-title">Career Trajectory <span class="subtitle">— last 5 NHL seasons</span></h3>
        <div class="chart-wrap career-wrap"><canvas id="compare-career"></canvas></div>
      </div>

      <div class="compare-stat-block" id="compare-contract"></div>
      <div class="compare-stat-block" id="compare-suggested"></div>
    `;

    results.hidden = false;

    // Render charts after DOM injection
    renderRadar(players);
    renderTrend(players);
    renderCareerTrajectory(players);
    renderShotComparison(players);
    renderContractValue(players);
    renderSuggestedMatchups(players);
  }

  function filterByTemplate(rows, template) {
    if (template === "all") return rows;
    if (template === "offense") {
      return rows.filter((r) => /G|A|PTS|Shot|ixG|iCF|iFF|iHDCF|xGF|Shooting/i.test(r.label));
    }
    if (template === "defense") {
      return rows.filter((r) => /\+\/-|HDSV|MDSV|GAA|SV%|GSAX|xGA|CF%|xGF%/i.test(r.label));
    }
    if (template === "contract") {
      // Show only the flag GAR/xGAR rows; verdict block does the work
      return rows.filter((r) => /GAR|GP/i.test(r.label));
    }
    if (template === "playoff") {
      // Trim to durability + scoring; playoff data isn't in current_season
      return rows.filter((r) => /GP|G|A|PTS|\+\/-|TOI/i.test(r.label));
    }
    return rows;
  }

  // ---------------------------------------------------------------------------
  // Slot management
  // ---------------------------------------------------------------------------
  async function loadPlayerByIdIntoSlot(playerId, slot) {
    try {
      const r = await fetch(`/api/player/${playerId}`);
      const data = await r.json();
      if (data.error) return;
      state.players[slot - 1] = data;

      // Sync input
      const input = document.querySelector(`.compare-search-input[data-slot="${slot}"]`);
      if (input) input.value = data.bio.name;

      // Sync URL
      const url = new URL(window.location.href);
      url.searchParams.set(`p${slot}`, slug(data.bio.name));
      history.replaceState({}, "", url);

      showResults(state.players);
    } catch (e) {
      console.error("Failed to load player into slot", slot, e);
    }
  }

  function attachSlotSearches() {
    [1, 2, 3].forEach((slot) => {
      const input = document.querySelector(`.compare-search-input[data-slot="${slot}"]`);
      const dropdown = document.querySelector(`.player-search-dropdown[data-slot="${slot}"]`);
      if (!input || !dropdown || !lookup) return;
      lookup.attachLiveSearch(input, dropdown, (player) => {
        loadPlayerByIdIntoSlot(player.id, slot);
      });
    });
  }

  function setupTemplateSelector() {
    const sel = $("#compare-template");
    if (!sel) return;
    sel.addEventListener("change", (e) => {
      state.template = e.target.value;
      showResults(state.players);
    });
  }

  function setupSwap() {
    const btn = $("#compare-swap");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const tmp = state.players[0];
      state.players[0] = state.players[1];
      state.players[1] = tmp;

      [1, 2].forEach((slot) => {
        const input = document.querySelector(`.compare-search-input[data-slot="${slot}"]`);
        if (input) input.value = state.players[slot - 1]?.bio.name || "";
      });

      // Update URL
      const url = new URL(window.location.href);
      const p1 = state.players[0]?.bio.name;
      const p2 = state.players[1]?.bio.name;
      if (p1) url.searchParams.set("p1", slug(p1)); else url.searchParams.delete("p1");
      if (p2) url.searchParams.set("p2", slug(p2)); else url.searchParams.delete("p2");
      history.replaceState({}, "", url);

      showResults(state.players);
    });
  }

  function setupShare() {
    const btn = $("#compare-share");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const url = new URL(window.location.href);
      // Make sure ptab=compare is set
      url.searchParams.set("ptab", "compare");
      const txt = url.toString();
      try {
        await navigator.clipboard.writeText(txt);
        btn.textContent = "Link copied!";
        setTimeout(() => (btn.textContent = "Share Link"), 1400);
      } catch (e) {
        prompt("Copy this URL:", txt);
      }
    });
  }

  function setupPrint() {
    const btn = $("#compare-print");
    if (!btn) return;
    btn.addEventListener("click", () => {
      // Add print-mode class so CSS print stylesheet kicks in
      document.body.classList.add("print-comparison");
      window.print();
      setTimeout(() => document.body.classList.remove("print-comparison"), 500);
    });
  }

  function setupThirdToggle() {
    const btn = $("#compare-toggle-third");
    if (!btn) return;
    btn.addEventListener("click", () => {
      state.thirdEnabled = !state.thirdEnabled;
      const slot = document.querySelector('.compare-search-slot[data-slot="3"]');
      if (slot) slot.hidden = !state.thirdEnabled;
      const row = document.querySelector('.compare-search-row');
      if (row) row.classList.toggle("with-third", state.thirdEnabled);
      btn.textContent = state.thirdEnabled ? "− Remove Player 3" : "+ Add Player 3";
      if (!state.thirdEnabled) {
        state.players[2] = null;
        const input = document.querySelector('.compare-search-input[data-slot="3"]');
        if (input) input.value = "";
        const url = new URL(window.location.href);
        url.searchParams.delete("p3");
        history.replaceState({}, "", url);
      }
      showResults(state.players);
    });
  }

  function setupSuggestedClicks() {
    document.addEventListener("click", (e) => {
      const card = e.target.closest(".suggest-card, .similar-card");
      if (!card) return;
      // In compare context, set into slot 2 (preserving slot 1)
      const isCompareTab = !document.querySelector('.players-tab-panel[data-ptab="compare"]')?.hidden;
      if (!isCompareTab) return;
      const name = card.dataset.name;
      if (!name) return;
      fetch(`/api/player-search?q=${encodeURIComponent(name)}&limit=5`)
        .then((r) => r.json())
        .then((data) => {
          const m = (data.results || []).find((p) => p.name === name) || (data.results || [])[0];
          if (m) loadPlayerByIdIntoSlot(m.id, 2);
        });
    });
  }

  // URL state load
  function loadFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const slugs = [params.get("p1"), params.get("p2"), params.get("p3")];
    slugs.forEach((s, i) => {
      if (!s) return;
      const slot = i + 1;
      if (slot === 3) {
        // Auto-enable third slot
        state.thirdEnabled = true;
        const slotEl = document.querySelector('.compare-search-slot[data-slot="3"]');
        if (slotEl) slotEl.hidden = false;
        const row = document.querySelector('.compare-search-row');
        if (row) row.classList.add("with-third");
        const btn = $("#compare-toggle-third");
        if (btn) btn.textContent = "− Remove Player 3";
      }
      const query = s.replace(/-/g, " ");
      fetch(`/api/player-search?q=${encodeURIComponent(query)}&limit=10`)
        .then((r) => r.json())
        .then((data) => {
          const match = (data.results || []).find((p) => slug(p.name) === s) || (data.results || [])[0];
          if (match) loadPlayerByIdIntoSlot(match.id, slot);
        });
    });

    // Switch to compare tab if ptab=compare in URL
    if (params.get("ptab") === "compare") {
      const tab = document.querySelector('.players-tab[data-ptab="compare"]');
      if (tab) tab.click();
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    if (!document.querySelector('.players-tab-panel[data-ptab="compare"]')) return;
    attachSlotSearches();
    setupTemplateSelector();
    setupSwap();
    setupShare();
    setupPrint();
    setupThirdToggle();
    setupSuggestedClicks();
    loadFromUrl();
  }

  // Expose for player_lookup.js cross-talk
  window.playerCompare = { loadPlayerByIdIntoSlot };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
