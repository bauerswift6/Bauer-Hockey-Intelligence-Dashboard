/* =========================================================
   NHL → Teams page: Power Play, Penalty Kill, Team Style/PDO,
   Head-to-Head comparison.
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const util = window.dashUtil;

  const state = { loaded: false };

  // ---------------------------------------------------------------------------
  // Glossary blurbs
  // ---------------------------------------------------------------------------
  const BLURBS = {
    pp: "How dangerous each team is with the man advantage. xGF/60 captures shot quality on the power play; PP% is the rough percentage of opportunities converted. xG-based metrics are more predictive of future PP success.",
    pk: "How well each team suppresses chances while shorthanded. xGA/60 captures the quality of shots allowed; PK% is the rough percentage of opposing power plays killed off. Lower xGA/60 is better.",
    style: "How each team plays at 5v5: pace (shot attempts per 60), shooting and save percentages, and PDO (SH% + SV%). PDO regresses to ~100 over time — teams above 100 are typically running hot on percentages, those below are due for a bounce-back.",
    h2h: "Compare any two teams' season-long stats side by side. Useful for matchup previews, trade-deadline analysis, or settling debates.",
  };

  // ---------------------------------------------------------------------------
  // PP / PK / Style leaderboards
  // ---------------------------------------------------------------------------
  function colorPdoCell(v) {
    if (v == null) return "—";
    const n = Number(v);
    const cls = n > 100.5 ? "metric-elite" : n < 99.5 ? "metric-poor" : "metric-mid";
    return `<span class="metric-pill ${cls}">${n.toFixed(2)}</span>`;
  }

  async function fetchJson(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.json();
  }

  async function buildPP() {
    const container = $("#pp-leaders-container");
    if (!container) return;
    container.innerHTML = `<section class="section leaderboard-section"><h2 class="section-title">Power Play Analytics</h2><div class="section-body"><div class="spinner"></div></div></section>`;
    let rows = [];
    try {
      const data = await fetchJson("/api/team-special-teams");
      rows = (data.teams || []).filter((t) => t.pp_xgf_60 != null);
    } catch (e) {
      container.innerHTML = `<section class="section"><div class="section-body"><p class="error-msg">Failed to load PP data.</p></div></section>`;
      return;
    }
    util.buildLeaderboard({
      container,
      id: "pp-board",
      title: "Power Play Analytics",
      subtitle: BLURBS.pp,
      ehBanner: true,
      sourceLabel: "Source: MoneyPuck (5on4 situation)",
      columns: [
        { key: "team", label: "Team" },
        { key: "pp_xgf_60", label: "xGF/60", title: "Expected goals per 60 min of PP time", fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "pp_g_60",   label: "G/60",   title: "Actual goals per 60 min of PP time", fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "pp_pct",    label: "PP% (est.)", title: "Rough PP% estimate (goals / (iceTime/90s)); see source notes", fmt: (v) => util.formatPct(v, 1), align: "right" },
        { key: "pp_toi_min", label: "PP TOI", title: "Total minutes spent on the power play this season", align: "right" },
      ],
      activeSort: { key: "pp_xgf_60", dir: "desc" },
      getRows: () => rows,
      searchKeys: ["team"],
    });
  }

  async function buildPK() {
    const container = $("#pk-leaders-container");
    if (!container) return;
    container.innerHTML = `<section class="section leaderboard-section"><h2 class="section-title">Penalty Kill Analytics</h2><div class="section-body"><div class="spinner"></div></div></section>`;
    let rows = [];
    try {
      const data = await fetchJson("/api/team-special-teams");
      rows = (data.teams || []).filter((t) => t.pk_xga_60 != null);
    } catch (e) {
      container.innerHTML = `<section class="section"><div class="section-body"><p class="error-msg">Failed to load PK data.</p></div></section>`;
      return;
    }
    util.buildLeaderboard({
      container,
      id: "pk-board",
      title: "Penalty Kill Analytics",
      subtitle: BLURBS.pk,
      ehBanner: true,
      sourceLabel: "Source: MoneyPuck (4on5 situation)",
      columns: [
        { key: "team", label: "Team" },
        { key: "pk_xga_60", label: "xGA/60", title: "Expected goals against per 60 min of PK time — lower is better", fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "pk_ga_60",  label: "GA/60",  title: "Actual goals against per 60 min of PK time", fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "pk_pct",    label: "PK% (est.)", title: "Rough PK% estimate; see source notes", fmt: (v) => util.formatPct(v, 1), align: "right" },
        { key: "pk_toi_min", label: "PK TOI", title: "Total minutes spent on the penalty kill this season", align: "right" },
      ],
      activeSort: { key: "pk_xga_60", dir: "asc" },  // lower xGA/60 = better
      getRows: () => rows,
      searchKeys: ["team"],
    });
  }

  async function buildStyle() {
    const container = $("#team-style-container");
    if (!container) return;
    container.innerHTML = `<section class="section leaderboard-section"><h2 class="section-title">Team Style — Pace, SH%, SV%, PDO</h2><div class="section-body"><div class="spinner"></div></div></section>`;
    let rows = [];
    try {
      const data = await fetchJson("/api/team-style");
      rows = data.teams || [];
    } catch (e) {
      container.innerHTML = `<section class="section"><div class="section-body"><p class="error-msg">Failed to load team style.</p></div></section>`;
      return;
    }
    util.buildLeaderboard({
      container,
      id: "style-board",
      title: "Team Style — Pace, SH%, SV%, PDO",
      subtitle: BLURBS.style,
      ehBanner: true,
      sourceLabel: "Source: MoneyPuck (5v5 only). PDO = SH% + SV%; regresses to 100 over time.",
      columns: [
        { key: "team", label: "Team" },
        { key: "gp",          label: "GP",   align: "right" },
        { key: "pace_sa_60",  label: "Pace (SA/60)", title: "Shot attempts for per 60 minutes at 5v5", fmt: (v) => util.formatNum(v, 1), align: "right" },
        { key: "sh_pct",      label: "SH%",  title: "Team shooting percentage at 5v5", fmt: (v) => util.formatPct(v, 2), align: "right" },
        { key: "sv_pct",      label: "SV%",  title: "Team save percentage at 5v5", fmt: (v) => util.formatPct(v, 2), align: "right" },
        { key: "pdo",         label: "PDO",  title: "SH% + SV%. Above 100 = lucky/hot; below 100 = unlucky/cold. Regresses to 100.", fmt: colorPdoCell, align: "right" },
      ],
      activeSort: { key: "pdo", dir: "desc" },
      getRows: () => rows,
      searchKeys: ["team"],
    });
  }

  // ---------------------------------------------------------------------------
  // Head-to-Head
  // ---------------------------------------------------------------------------
  async function populateTeamDropdowns() {
    try {
      const data = await fetchJson("/api/team-style");
      const teams = (data.teams || []).map((t) => t.team).sort();
      const opt = (t) => `<option value="${util.escHtml(t)}">${util.escHtml(t)}</option>`;
      const html = teams.map(opt).join("");
      const sel1 = $("#h2h-team1");
      const sel2 = $("#h2h-team2");
      if (sel1) sel1.innerHTML = `<option value="">— Select —</option>${html}`;
      if (sel2) sel2.innerHTML = `<option value="">— Select —</option>${html}`;
      // Sensible defaults — top two by PDO
      if (sel1 && teams.length) sel1.value = teams[0];
      if (sel2 && teams.length > 1) sel2.value = teams[1];
    } catch (e) {
      // dropdowns stay empty
    }
  }

  function renderH2H(data, t1Style, t2Style) {
    const container = $("#h2h-result");
    if (!container) return;
    if (data.error) {
      container.innerHTML = `<p class="error-msg">${util.escHtml(data.error)}</p>`;
      return;
    }
    const t1 = data.team1;
    const t2 = data.team2;

    const styleRow = (label, key, dp = 2, isPct = false) => {
      const a = t1Style?.[key];
      const b = t2Style?.[key];
      const fmt = (v) => v == null ? "—" : isPct ? util.formatPct(v, dp) : util.formatNum(v, dp);
      let aCls = "", bCls = "";
      if (typeof a === "number" && typeof b === "number") {
        if (a > b) aCls = "h2h-better"; else if (b > a) bCls = "h2h-better";
      }
      return `<tr>
        <td class="h2h-val ${aCls}">${fmt(a)}</td>
        <td class="h2h-label">${util.escHtml(label)}</td>
        <td class="h2h-val ${bCls}">${fmt(b)}</td>
      </tr>`;
    };

    const meetingsHtml = data.meetings && data.meetings.length
      ? `<table class="data-table h2h-meetings"><thead><tr>
           <th>Date</th><th>${util.escHtml(t1)}</th><th>${util.escHtml(t2)}</th><th>Result</th><th>Venue (${util.escHtml(t1)})</th>
         </tr></thead><tbody>
           ${data.meetings.map((m) => `
             <tr>
               <td>${util.escHtml(m.date)}</td>
               <td>${m.t1_score}</td>
               <td>${m.t2_score}</td>
               <td>${util.escHtml(m.result)}${m.ot_so ? ` (${m.ot_so})` : ""}</td>
               <td>${util.escHtml(m.venue)}</td>
             </tr>
           `).join("")}
         </tbody></table>`
      : `<p class="no-games-msg">No meetings this season yet.</p>`;

    container.innerHTML = `
      <div class="h2h-table-wrap">
        <table class="h2h-table">
          <thead><tr>
            <th class="h2h-team-head">${util.escHtml(t1)}</th>
            <th class="h2h-vs-head">vs</th>
            <th class="h2h-team-head">${util.escHtml(t2)}</th>
          </tr></thead>
          <tbody>
            <tr>
              <td class="h2h-record">${data.t1_wins}</td>
              <td class="h2h-label">Wins this season</td>
              <td class="h2h-record">${data.t2_wins}</td>
            </tr>
            <tr>
              <td class="h2h-val">${data.gf_t1}</td>
              <td class="h2h-label">Goals for in H2H</td>
              <td class="h2h-val">${data.gf_t2}</td>
            </tr>
            ${styleRow("xGF% (5v5 season)", "xgf_pct", 1, true)}
            ${styleRow("Pace (SA/60)", "pace_sa_60", 1)}
            ${styleRow("SH% (5v5)", "sh_pct", 2, true)}
            ${styleRow("SV% (5v5)", "sv_pct", 2, true)}
            ${styleRow("PDO", "pdo", 2)}
          </tbody>
        </table>
      </div>
      <h3 class="h2h-meetings-title">Head-to-head meetings this season</h3>
      ${meetingsHtml}
      <p class="news-source" style="margin-top: 1rem">${util.escHtml(data.note || "")}</p>
    `;
  }

  async function compareTeams() {
    const t1 = $("#h2h-team1")?.value;
    const t2 = $("#h2h-team2")?.value;
    const container = $("#h2h-result");
    if (!t1 || !t2) {
      container.innerHTML = `<p class="error-msg">Pick two teams to compare.</p>`;
      return;
    }
    if (t1 === t2) {
      container.innerHTML = `<p class="error-msg">Pick two different teams.</p>`;
      return;
    }
    container.innerHTML = `<div class="spinner"></div>`;
    try {
      const [h2hData, styleData] = await Promise.all([
        fetchJson(`/api/team-head-to-head?team1=${t1}&team2=${t2}`),
        fetchJson("/api/team-style"),
      ]);
      // Also get xGF% from team-analytics
      let analyticsByTeam = {};
      try {
        const a = await fetchJson("/api/team-analytics");
        (a.teams || []).forEach((t) => { analyticsByTeam[t.team] = t; });
      } catch (e) { /* ignore */ }

      const styleByTeam = {};
      (styleData.teams || []).forEach((t) => { styleByTeam[t.team] = t; });

      const merge = (ab) => ({
        ...(styleByTeam[ab] || {}),
        xgf_pct: analyticsByTeam[ab]?.xgf_pct,
      });

      renderH2H(h2hData, merge(t1), merge(t2));
    } catch (e) {
      container.innerHTML = `<p class="error-msg">Compare failed: ${util.escHtml(e.message || e)}</p>`;
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    // Lazy-load when Teams page is first viewed.
    // Listen for nav clicks to detect activation.
    const teamsTab = document.querySelector('.subpage-tab[data-subpage="teams"]');
    if (teamsTab) {
      teamsTab.addEventListener("click", () => {
        if (state.loaded) return;
        state.loaded = true;
        buildPP();
        buildPK();
        buildStyle();
        populateTeamDropdowns();
      });
    }

    // H2H interactions
    $("#h2h-compare")?.addEventListener("click", compareTeams);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
