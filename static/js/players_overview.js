/* =========================================================
   Players → Overview tab — preview cards grid, top 5 per stat
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = window.dashUtil ? window.dashUtil.escHtml : (s) => String(s ?? "");

  // Stat preview definitions
  const STAT_CARDS = [
    { id: "composite", label: "Composite Rating", source: "composite", key: "composite_rating",
      fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2), suffix: "" },
    { id: "points", label: "Points",          source: "skaters", key: "points",     fmt: (v) => v, suffix: "" },
    { id: "goals",  label: "Goals",           source: "skaters", key: "goals",      fmt: (v) => v, suffix: "" },
    { id: "assists",label: "Assists",         source: "skaters", key: "assists",    fmt: (v) => v, suffix: "" },
    { id: "gar",    label: "GAR",             source: "skaters", key: "gar",        fmt: (v) => v.toFixed(2), suffix: "" },
    { id: "xgar",   label: "xGAR",            source: "skaters", key: "xgar",       fmt: (v) => v.toFixed(2), suffix: "" },
    { id: "xgf_pct",label: "xGF%",            source: "skaters", key: "xgf_pct",    fmt: (v) => v.toFixed(2), suffix: "%", minToi: 300 },
    { id: "cf_pct", label: "CF% (Corsi)",     source: "skaters", key: "cf_pct",     fmt: (v) => v.toFixed(2), suffix: "%", minToi: 300 },
    { id: "hdcf_pct",label: "HDCF%",          source: "skaters", key: "hdcf_pct",   fmt: (v) => v.toFixed(2), suffix: "%", minToi: 300 },
    { id: "ixg_60", label: "ixG per 60",      source: "skaters", key: "ixg_60",     fmt: (v) => v.toFixed(2), suffix: "", minToi: 200 },
    { id: "icf_60", label: "iCF per 60",      source: "skaters", key: "icf_60",     fmt: (v) => v.toFixed(2), suffix: "", minToi: 200 },
    { id: "rel_cf_pct",label: "Relative CF%", source: "skaters", key: "rel_cf_pct", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2), suffix: "%", minToi: 300 },
    { id: "zone_start_pct",label: "Zone Start %", source: "skaters", key: "zone_start_pct", fmt: (v) => v.toFixed(1), suffix: "%", minToi: 300 },
    { id: "gsax",   label: "GSAX (Goalies)",  source: "goalies", key: "gsax",       fmt: (v) => v.toFixed(2), suffix: "" },
  ];

  let dataCache = { skaters: null, goalies: null, composite: null };

  async function fetchData() {
    if (!dataCache.skaters) {
      try {
        const r = await fetch("/api/players-full?min_toi_min=100");
        const d = await r.json();
        dataCache.skaters = d.players || [];
      } catch (e) {
        dataCache.skaters = [];
      }
    }
    if (!dataCache.goalies) {
      try {
        const r = await fetch("/api/goalies-full?min_games=10");
        const d = await r.json();
        dataCache.goalies = d.goalies || [];
      } catch (e) {
        dataCache.goalies = [];
      }
    }
    if (!dataCache.composite) {
      try {
        const r = await fetch("/api/composite-ratings");
        const d = await r.json();
        // qualified players only; normalize schema for renderCard()
        dataCache.composite = (d.full_dataset || [])
          .filter((p) => p.sample_size_flag === "ok")
          .map((p) => ({
            ...p,
            name: p.player_name,
            playerId: p.player_id,
            toi_min: p.toi_minutes,
            team_logo: `https://assets.nhle.com/logos/nhl/svg/${p.team}_light.svg`,
          }));
      } catch (e) {
        dataCache.composite = [];
      }
    }
  }

  function applyPositionFilter(rows, position) {
    if (position === "all") return rows;
    if (position === "F") return rows.filter((r) => ["C", "L", "R"].includes(r.position));
    return rows.filter((r) => r.position === position);
  }

  function topN(rows, key, n = 5, minToi = 0) {
    return rows
      .filter((r) => r.toi_min >= minToi && r[key] != null)
      .slice() // copy
      .sort((a, b) => (b[key] || 0) - (a[key] || 0))
      .slice(0, n);
  }

  function renderCard(card, rows) {
    const top = topN(rows, card.key, 5, card.minToi || 0);
    if (!top.length) {
      return `
        <div class="overview-card">
          <header class="overview-card-head">
            <h3 class="overview-card-title">${esc(card.label)}</h3>
          </header>
          <p class="muted" style="padding: 0.7rem 0;">No qualified players for current filters.</p>
        </div>
      `;
    }

    const rowsHtml = top.map((p, i) => `
      <li class="overview-row" data-player-id="${esc(p.playerId)}" data-player-name="${esc(p.name)}">
        <span class="overview-rank">${i + 1}</span>
        <img class="overview-team-logo" src="${esc(p.team_logo)}" alt="" onerror="this.style.display='none'" />
        <span class="overview-name">${esc(p.name)}</span>
        <span class="overview-team">${esc(p.team)}</span>
        <span class="overview-value">${esc(card.fmt(p[card.key]))}${esc(card.suffix)}</span>
      </li>
    `).join("");

    return `
      <div class="overview-card" data-stat-id="${esc(card.id)}">
        <header class="overview-card-head">
          <h3 class="overview-card-title">${esc(card.label)}</h3>
          <span class="overview-card-source">${card.source === "goalies" ? "Goalies" : "Skaters"}</span>
        </header>
        <ol class="overview-rows">${rowsHtml}</ol>
        <button class="overview-jump-btn" data-stat-id="${esc(card.id)}">View Full Leaderboard →</button>
      </div>
    `;
  }

  function renderAll() {
    const grid = $("#overview-grid");
    const spinner = $("#overview-spinner");
    if (!grid) return;

    // Honor page-level position filter
    const posBtn = document.querySelector("#player-pos-filter .pos-btn.active");
    const position = posBtn?.dataset.pos || "all";
    const filtered = applyPositionFilter(dataCache.skaters || [], position);
    const filteredComposite = applyPositionFilter(dataCache.composite || [], position);

    const html = STAT_CARDS.map((card) => {
      let rows;
      if (card.source === "goalies") rows = dataCache.goalies || [];
      else if (card.source === "composite") rows = filteredComposite;
      else rows = filtered;
      return renderCard(card, rows);
    }).join("");

    grid.innerHTML = html;
    grid.hidden = false;
    spinner.hidden = true;
  }

  function setupJumpButtons() {
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".overview-jump-btn");
      if (!btn) return;
      const statId = btn.dataset.statId;
      // Switch tabs to leaderboards and pre-select the stat
      const tab = document.querySelector('.players-tab[data-ptab="leaderboards"]');
      if (tab) tab.click();
      if (window.playersLeaderboards) {
        window.playersLeaderboards.selectStat(statId);
      }
    });
  }

  function setupRowClicks() {
    document.addEventListener("click", (e) => {
      const row = e.target.closest(".overview-row");
      if (!row) return;
      const pid = row.dataset.playerId;
      if (!pid) return;
      // Switch to Player Search and load
      const tab = document.querySelector('.players-tab[data-ptab="player-search"]');
      if (tab) tab.click();
      if (window.playerLookup && window.playerLookup.loadProfile) {
        window.playerLookup.loadProfile(Number(pid));
      }
    });
  }

  let firstLoaded = false;

  async function load() {
    if (firstLoaded) return;
    firstLoaded = true;
    const spinner = $("#overview-spinner");
    if (spinner) spinner.hidden = false;
    await fetchData();
    renderAll();
  }

  function watchPositionFilter() {
    const bar = document.querySelector("#player-pos-filter");
    if (!bar) return;
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".pos-btn");
      if (!btn) return;
      // Re-render after a tick so the active class settles
      setTimeout(() => {
        if (firstLoaded) renderAll();
      }, 0);
    });
  }

  function watchTabActivation() {
    const tab = document.querySelector('.players-tab[data-ptab="overview"]');
    if (!tab) return;
    tab.addEventListener("click", load);
    // Eager-load if Overview is already active on first paint
    if (tab.classList.contains("active")) load();
  }

  function init() {
    setupJumpButtons();
    setupRowClicks();
    watchPositionFilter();
    watchTabActivation();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
