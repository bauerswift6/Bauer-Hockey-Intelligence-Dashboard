/* =========================================================
   NHL → Playoffs page: full-page bracket, stat leaders,
   team analytics, series history.
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const util = window.dashUtil;
  const state = { loaded: false };

  // ---------------------------------------------------------------------------
  // Bracket — duplicate of the morning-brief rendering, scoped to playoffs page IDs
  // ---------------------------------------------------------------------------
  function renderSeriesCard(s) {
    if (!s) return `<div class="series-card series-card-tbd"><span>TBD</span></div>`;
    const top = s.top_seed || {};
    const bot = s.bottom_seed || {};
    let statusText = "Not started";
    let statusClass = "status-pending";
    if (s.complete) { statusText = "Complete"; statusClass = "status-complete"; }
    else if (s.started) { statusText = "In progress"; statusClass = "status-active"; }
    const tbd = (seed) => !seed || !seed.abbrev;
    const seedRow = (seed) => {
      if (tbd(seed)) return `<div class="series-team series-team-tbd"><span class="series-team-abbrev">TBD</span></div>`;
      const wonCls = seed.won ? "series-team-won" : "";
      return `<div class="series-team ${wonCls}">
        <img class="series-team-logo" src="${util.escHtml(seed.logo || '')}" alt="" loading="lazy" />
        <span class="series-team-abbrev">${util.escHtml(seed.abbrev)}</span>
        <span class="series-team-wins">${seed.wins ?? 0}</span>
      </div>`;
    };
    return `<div class="series-card ${statusClass}">
      ${seedRow(top)}${seedRow(bot)}
      <div class="series-status">${util.escHtml(statusText)}</div>
    </div>`;
  }

  function groupBracketSeries(rounds) {
    const by = {};
    rounds.forEach((r) => { by[r.round_number] = r.series; });
    const split = (n) => {
      const all = by[n] || [];
      return {
        east: all.filter((s) => s.conference === "Eastern"),
        west: all.filter((s) => s.conference === "Western"),
        final: all.filter((s) => s.conference === "Final" || s.round === 4),
      };
    };
    return { r1: split(1), r2: split(2), r3: split(3), r4: split(4) };
  }

  function renderBracket(data) {
    const content = $("#po-bracket-content");
    if (!content) return;
    const g = groupBracketSeries(data.rounds || []);
    const pad = (a, n) => { const o = a.slice(0, n); while (o.length < n) o.push(null); return o; };
    const col = (label, series) =>
      `<div class="bracket-col"><div class="bracket-col-label">${util.escHtml(label)}</div><div class="bracket-col-series">${series.map(renderSeriesCard).join("")}</div></div>`;
    content.innerHTML = `<div class="bracket-grid">
      ${col("East R1", pad(g.r1.east, 4))}
      ${col("East R2", pad(g.r2.east, 2))}
      ${col("East Final", pad(g.r3.east, 1))}
      ${col("Stanley Cup", pad(g.r4.final, 1))}
      ${col("West Final", pad(g.r3.west, 1))}
      ${col("West R2", pad(g.r2.west, 2))}
      ${col("West R1", pad(g.r1.west, 4))}
    </div>`;
    content.hidden = false;
  }

  async function loadBracket() {
    const spinner = $("#po-bracket-spinner");
    const empty = $("#po-bracket-empty");
    const err = $("#po-bracket-error");
    [empty, err].forEach((el) => el && (el.hidden = true));
    if (spinner) spinner.hidden = false;
    try {
      const r = await fetch("/api/playoff-bracket");
      const data = await r.json();
      if (spinner) spinner.hidden = true;
      if (data.empty || !data.rounds?.length) {
        if (empty) empty.hidden = false;
        return data;
      }
      renderBracket(data);
      return data;
    } catch (e) {
      if (spinner) spinner.hidden = true;
      if (err) err.hidden = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Playoff stat leaders
  // ---------------------------------------------------------------------------
  function renderLeadersCard(title, players, valueKey = "value", valueLabel = "") {
    const items = (players || []).map((p, i) => `
      <li>
        <span class="leader-rank">${i + 1}</span>
        ${p.headshot ? `<img src="${util.escHtml(p.headshot)}" alt="" class="leader-headshot" loading="lazy" onerror="this.style.display='none'" />` : ""}
        <div class="leader-info">
          <div class="leader-name">${util.escHtml(p.name)}</div>
          <div class="leader-team">${util.escHtml(p.team)}${p.position ? ` · ${util.escHtml(p.position)}` : ""}</div>
        </div>
        <span class="leader-value">${util.escHtml(p[valueKey])}</span>
      </li>
    `).join("");
    return `<div class="leaders-card">
      <h3 class="leaders-card-title">${util.escHtml(title)}${valueLabel ? ` <span class="subtitle">${util.escHtml(valueLabel)}</span>` : ""}</h3>
      <ul class="leaders-list">${items || "<li>No data yet</li>"}</ul>
    </div>`;
  }

  async function loadPlayoffLeaders() {
    const spinner = $("#po-leaders-spinner");
    const content = $("#po-leaders-content");
    const err = $("#po-leaders-error");
    [content, err].forEach((el) => el && (el.hidden = true));
    if (spinner) spinner.hidden = false;
    try {
      const r = await fetch("/api/playoff-stats-leaders");
      const data = await r.json();
      if (spinner) spinner.hidden = true;
      content.innerHTML = [
        renderLeadersCard("Points",        data.points,      "value"),
        renderLeadersCard("Goals",         data.goals,       "value"),
        renderLeadersCard("Assists",       data.assists,     "value"),
        renderLeadersCard("Plus/Minus",    data.plusMinus,   "value"),
        renderLeadersCard("Goalie Wins",   data.goalie_wins, "value"),
      ].join("");
      content.hidden = false;
    } catch (e) {
      if (spinner) spinner.hidden = true;
      if (err) err.hidden = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Playoff team analytics
  // ---------------------------------------------------------------------------
  async function loadPlayoffTeamAnalytics() {
    const container = $("#playoff-team-analytics-container");
    if (!container) return;
    container.innerHTML = `<section class="section leaderboard-section"><h2 class="section-title">Playoff Team Analytics — 5v5</h2><div class="section-body"><div class="spinner"></div></div></section>`;
    let rows = [];
    let fallback = null;
    try {
      const r = await fetch("/api/playoff-team-analytics");
      const data = await r.json();
      rows = data.teams || [];
      fallback = data.fallback_note;
    } catch (e) {
      container.innerHTML = `<section class="section"><div class="section-body"><p class="error-msg">Failed to load playoff team analytics.</p></div></section>`;
      return;
    }
    util.buildLeaderboard({
      container,
      id: "po-team-analytics",
      title: "Playoff Team Analytics — 5v5",
      subtitle: "How each remaining team is performing in the playoffs. xGF% is the consensus best possession-quality stat — CF% rewards volume, HDCF% rewards inner-slot chances.",
      ehBanner: true,
      sourceLabel: fallback ? `${fallback}` : "Source: MoneyPuck (playoffs split, 5v5)",
      columns: [
        { key: "team", label: "Team" },
        { key: "gp",   label: "GP", align: "right" },
        { key: "xgf_pct",  label: "xGF%",  title: "Expected Goals For % at 5v5", fmt: (v) => util.formatPct(v, 1), align: "right" },
        { key: "cf_pct",   label: "CF%",   title: "Corsi For % at 5v5",          fmt: (v) => util.formatPct(v, 1), align: "right" },
        { key: "hdcf_pct", label: "HDCF%", title: "High-Danger Corsi For % at 5v5", fmt: (v) => util.formatPct(v, 1), align: "right" },
      ],
      activeSort: { key: "xgf_pct", dir: "desc" },
      getRows: () => rows,
      searchKeys: ["team"],
    });
  }

  // ---------------------------------------------------------------------------
  // Series History (current-season H2H proxy for active series)
  // ---------------------------------------------------------------------------
  async function loadSeriesHistory(bracketData) {
    const container = $("#series-history-content");
    const spinner = $("#series-history-spinner");
    const err = $("#series-history-error");
    if (!container) return;
    [err].forEach((el) => el && (el.hidden = true));
    if (spinner) spinner.hidden = false;

    // Find active series from bracket
    const currentRound = bracketData?.current_round;
    if (!currentRound || !bracketData?.rounds) {
      if (spinner) spinner.hidden = true;
      container.innerHTML = `<p class="no-games-msg">No active playoff series.</p>`;
      container.hidden = false;
      return;
    }
    const activeSeries = [];
    bracketData.rounds.forEach((r) => {
      if (r.round_number > currentRound) return;
      (r.series || []).forEach((s) => {
        if (!s.top_seed || !s.bottom_seed) return;
        if (s.complete && r.round_number < currentRound) return; // already moved on
        activeSeries.push({
          round: r.round_number,
          letter: s.series_letter,
          conference: s.conference,
          t1: s.top_seed.abbrev,
          t2: s.bottom_seed.abbrev,
          t1_logo: s.top_seed.logo,
          t2_logo: s.bottom_seed.logo,
          t1_wins: s.top_seed.wins,
          t2_wins: s.bottom_seed.wins,
          complete: s.complete,
        });
      });
    });

    if (!activeSeries.length) {
      if (spinner) spinner.hidden = true;
      container.innerHTML = `<p class="no-games-msg">No active series right now.</p>`;
      container.hidden = false;
      return;
    }

    try {
      const histories = await Promise.all(
        activeSeries.map((s) =>
          fetch(`/api/team-head-to-head?team1=${s.t1}&team2=${s.t2}`)
            .then((r) => r.json())
            .then((d) => ({ series: s, h2h: d }))
            .catch(() => ({ series: s, h2h: null }))
        )
      );

      container.innerHTML = histories.map(({ series: s, h2h }) => {
        const meetings = h2h?.meetings || [];
        const rows = meetings.length
          ? meetings.map((m) => `
              <tr>
                <td class="mono-tiny">${util.escHtml(m.date)}</td>
                <td>${util.escHtml(m.result)}${m.ot_so ? ` <span class="conf-tag">${util.escHtml(m.ot_so)}</span>` : ""}</td>
                <td>${m.t1_score}–${m.t2_score}</td>
              </tr>
            `).join("")
          : `<tr><td colspan="3" class="no-games-msg">No meetings between these teams this season.</td></tr>`;
        const t1w = h2h?.t1_wins ?? "—";
        const t2w = h2h?.t2_wins ?? "—";
        return `<div class="series-history-card">
          <div class="series-history-head">
            <div class="series-history-team">
              <img src="${util.escHtml(s.t1_logo || '')}" class="form-team-logo" alt="" loading="lazy" />
              <span class="team-abbrev-text">${util.escHtml(s.t1)}</span>
              <span class="series-history-record">${t1w}</span>
            </div>
            <span class="series-history-vs">vs</span>
            <div class="series-history-team">
              <span class="series-history-record">${t2w}</span>
              <span class="team-abbrev-text">${util.escHtml(s.t2)}</span>
              <img src="${util.escHtml(s.t2_logo || '')}" class="form-team-logo" alt="" loading="lazy" />
            </div>
          </div>
          <div class="series-history-meta">Round ${s.round} · series ${s.t1_wins}-${s.t2_wins}${s.complete ? " (complete)" : ""}</div>
          <table class="data-table series-history-table">
            <thead><tr><th>Date</th><th>Result</th><th>Score</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>`;
      }).join("");

      if (spinner) spinner.hidden = true;
      container.hidden = false;
    } catch (e) {
      if (spinner) spinner.hidden = true;
      if (err) err.hidden = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  async function loadAll() {
    if (state.loaded) return;
    state.loaded = true;
    const bracketData = await loadBracket();
    loadPlayoffLeaders();
    loadPlayoffTeamAnalytics();
    loadSeriesHistory(bracketData);
  }

  function init() {
    const tab = document.querySelector('.subpage-tab[data-subpage="playoffs"]');
    if (tab) tab.addEventListener("click", loadAll);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
