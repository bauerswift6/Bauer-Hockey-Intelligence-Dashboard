/* =========================================================
   Contract Value Analysis — multi-dimensional model
   - Surplus Value as headline metric
   - 5 list views + Scatter (3 variants)
   - Side panel with full per-player breakdown
   - ELC Watch + Team Cap Efficiency sections
   - Default: Exclude ELCs, sort by Surplus Value
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const GLOSSARY = (window.GLOSSARY_DATA && window.GLOSSARY_DATA.stats) || [];
  const glossaryById = {};
  GLOSSARY.forEach((s) => { glossaryById[s.id] = s; });
  function tipFor(id) { const s = glossaryById[id]; return s ? s.def : ""; }

  const state = {
    raw: [],            // all rows from /api/contract-values
    teamData: [],       // rows from /api/team-cap-efficiency
    view: "best",       // best | def | off | sustain | aging | scatter
    contractType: "exclude-elc",  // all | exclude-elc | market | elc | bridge | veteran
    position: "all",
    team: "all",
    expiry: "all",
    search: "",
    sortKey: null,      // null = view's default sort
    sortAsc: false,
    scatter: null,      // Chart.js instance
    scatterVariant: "cap-vs-gar",
    loaded: false,
  };

  let CURRENT_SEASON_END_YEAR = new Date().getFullYear();

  async function _resolveCurrentSeasonEndYear() {
    try {
      const r = await fetch("/api/season-info");
      if (!r.ok) return;
      const info = await r.json();
      if (info && Number.isInteger(info.end_year)) CURRENT_SEASON_END_YEAR = info.end_year;
    } catch (e) {}
  }

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  function classifyPosition(pos) {
    if (pos === "G") return "G";
    if (pos === "D") return "D";
    return "F";
  }

  function fmtMoney(v) {
    if (v == null) return "—";
    // U+2212 minus is wider and typographically aligned with U+002B '+'
    const sign = v < 0 ? "−" : "";
    return `${sign}$${Math.abs(v).toFixed(2)}M`;
  }

  function fmtSurplus(v, isElc) {
    if (v == null) return '<span class="cv-na">—</span>';
    const cls = isElc ? "cv-surplus-elc" : (v >= 0 ? "cv-surplus-pos" : "cv-surplus-neg");
    // Always-explicit sign: "+$1.88M" or "−$2.45M". Color reinforces, never
    // sole signal — accessible to red/green-blind readers.
    const sign = v >= 0 ? "+" : "−";
    const body = `$${Math.abs(v).toFixed(2)}M`;
    return `<span class="${cls}">${sign}${body}</span>`;
  }

  function fmtNum(v, dp = 2) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(dp);
  }

  function contractBadge(type) {
    if (!type) return "";
    const cls = "cv-badge cv-badge-" + type.toLowerCase().replace(/\s+/g, "-");
    return `<span class="${cls}" title="${escHtml(type)}">${escHtml(type)}</span>`;
  }

  // Age-curve badge — drives the visual cue in the Age column. Three tiers
  // matching the player's position on the public-model aging curve. Distinct
  // glyph set (↑ / ● / ↓) from anything used near player names so the two
  // can't be confused.
  function ageCurveBadge(flag) {
    if (flag === "pre_peak")  return '<span class="cv-age-badge cv-age-pre"  title="Under 25 — pre-peak (rising)">↑</span>';
    if (flag === "peak")      return '<span class="cv-age-badge cv-age-peak" title="25–29 — peak years">●</span>';
    if (flag === "post_peak") return '<span class="cv-age-badge cv-age-post" title="30+ — post-peak (declining)">↓</span>';
    return "";
  }

  function ageCell(v, r) {
    if (v == null) return "—";
    return `${escHtml(String(v))} ${ageCurveBadge(r.age_flag)}`;
  }

  function expiryCell(v, r) {
    if (v == null) return "—";
    // years_remaining comes from the server (max(0, expiry - today.year)).
    // 0 = contract is in its final year now; 1 = one full season after this;
    // N = N seasons remaining.
    const y = r.years_remaining;
    let suffix = "";
    if (y == null) suffix = "";
    else if (y === 0) suffix = ' <span class="cv-expiry-yrs">(final yr)</span>';
    else if (y === 1) suffix = ' <span class="cv-expiry-yrs">(1 yr)</span>';
    else              suffix = ` <span class="cv-expiry-yrs">(${y} yrs)</span>`;
    return `${escHtml(String(v))}${suffix}`;
  }

  // ---------------------------------------------------------------------------
  // Players in-page tab bar (kept here as the canonical tab controller)
  // ---------------------------------------------------------------------------
  function initPlayersTabs() {
    const tabs = $$(".players-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.dataset.ptab;
        tabs.forEach((t) => {
          const on = t.dataset.ptab === key;
          t.classList.toggle("active", on);
          t.setAttribute("aria-selected", on ? "true" : "false");
        });
        $$(".players-tab-panel").forEach((panel) => {
          panel.hidden = panel.dataset.ptab !== key;
        });
        if (key === "contract-value" && !state.loaded) loadContractValues();
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Fetch
  // ---------------------------------------------------------------------------
  async function loadContractValues() {
    if (state.loaded) return;
    state.loaded = true;

    const spinner = $("#cv-spinner");
    const errEl = $("#cv-error");
    const blockedEl = $("#cv-blocked");
    spinner.hidden = false;
    errEl.hidden = true;
    blockedEl.hidden = true;

    await _resolveCurrentSeasonEndYear();

    try {
      const r = await fetch("/api/contract-values");
      const data = await r.json();
      spinner.hidden = true;

      if (data.error) {
        errEl.textContent = data.error;
        errEl.hidden = false;
        return;
      }
      if (data.blocked) {
        blockedEl.innerHTML = `
          <strong>Contract data unavailable</strong><br>
          ${escHtml(data.message || "No contract data sources available.")}
        `;
        blockedEl.hidden = false;
        return;
      }

      state.raw = data.players || [];
      const SOURCE_LABELS = { capwages: "CapWages", puckpedia: "PuckPedia", manual: "Manual contract dataset" };
      const sourceName = SOURCE_LABELS[data.source] || "Manual contract dataset";
      let fetched = "";
      if (data.meta && data.meta.fetched_at) {
        const d = new Date(data.meta.fetched_at);
        if (!isNaN(d)) fetched = ` · updated ${d.toLocaleDateString()}`;
      }
      $("#cv-source-label").textContent = `Source: ${sourceName} + MoneyPuck${fetched}`;
      $("#cv-source-label").hidden = false;

      populateTeamFilter();
      renderActiveView();
      loadElcWatch();
      loadTeamEfficiency();
    } catch (e) {
      spinner.hidden = true;
      errEl.textContent = "Failed to load contract data.";
      errEl.hidden = false;
    }
  }

  function populateTeamFilter() {
    const sel = $("#cv-team-filter");
    if (!sel) return;
    // Clear all but the first option
    sel.innerHTML = '<option value="all">All teams</option>';
    const teams = Array.from(new Set(state.raw.map((p) => p.team).filter(Boolean))).sort();
    teams.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t; opt.textContent = t;
      sel.appendChild(opt);
    });
  }

  // ---------------------------------------------------------------------------
  // Filter rows
  // ---------------------------------------------------------------------------
  function pageLevelPosition() {
    const btn = document.querySelector("#player-pos-filter .pos-btn.active");
    return btn?.dataset.pos || "all";
  }

  function filterRows() {
    let rows = state.raw.slice();

    // Contract type
    if (state.contractType === "exclude-elc") rows = rows.filter((r) => r.contract_type !== "ELC");
    else if (state.contractType === "elc")     rows = rows.filter((r) => r.contract_type === "ELC");
    else if (state.contractType === "bridge")  rows = rows.filter((r) => r.contract_type === "Bridge");
    else if (state.contractType === "market")  rows = rows.filter((r) => r.contract_type === "Market Rate");
    else if (state.contractType === "veteran") rows = rows.filter((r) => r.contract_type === "Veteran");

    // Position — local
    const posKey = state.position;
    if (posKey === "F") rows = rows.filter((r) => ["C", "L", "R"].includes(r.position));
    else if (posKey === "D") rows = rows.filter((r) => r.position === "D");
    else if (posKey === "G") rows = rows.filter((r) => r.position === "G");

    // Page-level position (overrides if more restrictive than local)
    const pagePos = pageLevelPosition();
    if (pagePos !== "all") {
      if (pagePos === "F") rows = rows.filter((r) => ["C", "L", "R"].includes(r.position));
      else if (pagePos === "D") rows = rows.filter((r) => r.position === "D");
      else if (pagePos === "G") rows = rows.filter((r) => r.position === "G");
    }

    // Team
    if (state.team !== "all") rows = rows.filter((r) => r.team === state.team);

    // Expiry
    if (state.expiry === "this-year") rows = rows.filter((r) => r.expiry_year === CURRENT_SEASON_END_YEAR);
    else if (state.expiry === "next-year") rows = rows.filter((r) => r.expiry_year === CURRENT_SEASON_END_YEAR + 1);
    else if (state.expiry === "2plus") rows = rows.filter((r) => r.expiry_year >= CURRENT_SEASON_END_YEAR + 2);

    // Search
    if (state.search) {
      const q = state.search.toLowerCase();
      rows = rows.filter((r) => (r.name || "").toLowerCase().includes(q));
    }

    return rows;
  }

  // ---------------------------------------------------------------------------
  // Column definitions per view
  // ---------------------------------------------------------------------------
  // Sustain tooltip — explicit about the actual scale. Today's "Sustain" is
  // a raw GAR − xGAR differential in GAR (wins) units, NOT a z-score or
  // percentile. We surface that here so users don't misread −1.01 as "1
  // standard deviation below normal" when it really means "GAR is 1.01 wins
  // below underlying xG-derived expectation".
  const SUSTAIN_TOOLTIP =
    "Legacy, being rebuilt: this mixes legacy GAR with v2 xGAR. " +
    "It uses the legacy composite GAR and will be updated when GAR is rebuilt. " +
    "Sustainability gap = GAR (legacy) minus xGAR (v2), in GAR (wins) units. " +
    "Positive = current GAR is outpacing underlying shot quality (regression risk). " +
    "Negative = underperforming the underlying shot quality (likely to bounce back). " +
    "Flagged at or above +3.0 as Regression Risk; at or below -3.0 typically a buy-low signal.";

  // GAR column note — goalie rows use GSAX (goals saved above expected) not
  // skater-WAR GAR, with a separate $0.45M/save multiplier feeding Surplus.
  const GAR_TOOLTIP =
    "Skaters: composite GAR (wins above replacement, z-blended). " +
    "Goalies: this column shows GSAX (goals saved above expected) on a different scale; " +
    "the Surplus column applies a goalie-specific $0.45M/GSAX multiplier so the dollar " +
    "comparison stays apples-to-apples even though the raw GAR/GSAX numbers are not.";

  const COMMON_COLS = [
    { key: "name",          label: "Player",       align: "left",  isPlayer: true },
    { key: "team",          label: "Team",         align: "left",  isTeam: true },
    { key: "position",      label: "Pos",          align: "left",  isPos: true },
    { key: "age",           label: "Age",          align: "right", fmt: (v, r) => ageCell(v, r) },
    { key: "cap_hit",       label: "Cap",          align: "right", fmt: (v) => fmtMoney(v) },
    { key: "expiry_year",   label: "Expiry",       align: "right", fmt: (v, r) => expiryCell(v, r) },
    { key: "contract_type", label: "Type",         align: "left",  fmt: (v) => contractBadge(v) },
    { key: "gar",           label: "GAR",          align: "right", glossaryId: "gar", titleOverride: GAR_TOOLTIP, fmt: (v) => fmtNum(v) },
  ];

  function bestColsExtra() {
    return [
      { key: "off_gar",            label: "Off GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "def_gar",            label: "Def GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "pp_gar",             label: "PP GAR",      align: "right", fmt: (v) => fmtNum(v) },
      { key: "pk_gar",             label: "PK GAR",      align: "right", fmt: (v) => fmtNum(v) },
      { key: "gar_per_million",    label: "GAR/$1M",     align: "right", glossaryId: "gar-per-mil", fmt: (v) => fmtNum(v) },
      { key: "xgar",               label: "xGAR",        align: "right", glossaryId: "xgar", fmt: (v) => fmtNum(v) },
      { key: "sustainability_score", label: "Sustain (legacy)", align: "right", titleOverride: SUSTAIN_TOOLTIP, infoIcon: true, fmt: (v, r) => sustainCell(v) },
      { key: "surplus_value",      label: "Surplus",     align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC"), highlight: true },
    ];
  }

  function defColsExtra() {
    return [
      { key: "def_gar",             label: "Def GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "def_gar_per_million", label: "Def GAR/$1M", align: "right", fmt: (v) => fmtNum(v), highlight: true },
      { key: "off_gar",             label: "Off GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "surplus_value",       label: "Surplus",     align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC") },
    ];
  }

  function offColsExtra() {
    return [
      { key: "off_gar",             label: "Off GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "off_gar_per_million", label: "Off GAR/$1M", align: "right", fmt: (v) => fmtNum(v), highlight: true },
      { key: "def_gar",             label: "Def GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "surplus_value",       label: "Surplus",     align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC") },
    ];
  }

  function sustainColsExtra() {
    return [
      { key: "xgar",                 label: "xGAR",       align: "right", glossaryId: "xgar", fmt: (v) => fmtNum(v) },
      { key: "sustainability_score", label: "Sustain (legacy)", align: "right", titleOverride: SUSTAIN_TOOLTIP, infoIcon: true, fmt: (v, r) => sustainCell(v), highlight: true },
      { key: "surplus_value",        label: "Surplus",    align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC") },
    ];
  }

  function agingColsExtra() {
    return [
      { key: "off_gar",            label: "Off GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "def_gar",            label: "Def GAR",     align: "right", fmt: (v) => fmtNum(v) },
      { key: "surplus_value",      label: "Surplus",     align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC") },
      { key: "projected_gar_y3",   label: "Y3 GAR proj", align: "right", fmt: (v) => v == null ? "—" : fmtNum(v) },
      { key: "projected_surplus_y3", label: "Y3 Surplus", align: "right", fmt: (v, r) => fmtSurplus(v, r.contract_type === "ELC") },
    ];
  }

  function sustainCell(v) {
    if (v == null) return "—";
    const cls = v >= 3 ? "cv-sustain-warn" : (v <= -3 ? "cv-sustain-buy" : "cv-sustain-neutral");
    const sign = v >= 0 ? "+" : "";
    const tag = v >= 3 ? '<span class="cv-warn-pill">Regression Risk</span>' : "";
    return `<span class="${cls}">${sign}${v.toFixed(2)}</span>${tag}`;
  }

  // ---------------------------------------------------------------------------
  // Render: list views (best / def / off / sustain)
  // ---------------------------------------------------------------------------
  function colsForView() {
    if (state.view === "def") return [...COMMON_COLS, ...defColsExtra()];
    if (state.view === "off") return [...COMMON_COLS, ...offColsExtra()];
    if (state.view === "sustain") return [...COMMON_COLS, ...sustainColsExtra()];
    if (state.view === "aging") return [...COMMON_COLS, ...agingColsExtra()];
    return [...COMMON_COLS, ...bestColsExtra()];
  }

  function defaultSortFor(view) {
    if (view === "best") return { key: "surplus_value", asc: false };
    if (view === "def")  return { key: "def_gar_per_million", asc: false };
    if (view === "off")  return { key: "off_gar_per_million", asc: false };
    if (view === "sustain") return { key: "sustainability_score", asc: true };
    if (view === "aging")   return { key: "surplus_value", asc: false };
    return { key: "surplus_value", asc: false };
  }

  function applyViewFilters(rows) {
    if (state.view === "def") {
      // min defensive GAR floor
      rows = rows.filter((r) => (r.def_gar || 0) >= 1.0);
    }
    if (state.view === "aging") {
      rows = rows.filter((r) => r.age != null && r.age >= 30);
    }
    return rows;
  }

  function viewNote() {
    if (state.view === "def") {
      return "Defensive GAR is a heuristic split (Off share derived from ixG/60, Def is the remainder). Defensive metrics have wider error bars than offensive metrics — supplement with video analysis. Min Def GAR ≥ 1.0 applied.";
    }
    if (state.view === "sustain") {
      return "Players whose GAR most exceeds their xGAR are at highest risk of regression. A gap > 3.0 is flagged 'Regression Risk' — likely benefiting from unsustainable shooting % or finishing.";
    }
    if (state.view === "aging") {
      return "Filtered to players age 30+. Y3 GAR projection applies an 8%/yr decline starting at age 30 (forwards) or 33 (defensemen).";
    }
    return null;
  }

  function renderListView() {
    const note = viewNote();
    const noteEl = $("#cv-view-note");
    if (note) { noteEl.textContent = note; noteEl.hidden = false; }
    else noteEl.hidden = true;

    // Show universal list table panel
    $$(".cv-view").forEach((v) => v.hidden = true);
    if (state.view === "aging") $('[data-cv-view="aging"]').hidden = false;
    else $('[data-cv-view-list="1"]').hidden = false;

    const cols = colsForView();
    const sort = state.sortKey
      ? { key: state.sortKey, asc: state.sortAsc }
      : defaultSortFor(state.view);

    let rows = applyViewFilters(filterRows());

    // Sort
    rows = rows.slice().sort((a, b) => {
      const av = a[sort.key], bv = b[sort.key];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return sort.asc ? av - bv : bv - av;
      return sort.asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });

    const theadId = state.view === "aging" ? "#cv-aging-thead" : "#cv-list-thead";
    const tbodyId = state.view === "aging" ? "#cv-aging-tbody" : "#cv-list-tbody";
    const footerId = state.view === "aging" ? "#cv-aging-footer" : "#cv-list-footer";

    const allCols = [{ key: "_rank", label: "#", align: "right" }, ...cols];
    $(theadId).innerHTML = allCols.map((c) => {
      let cls = c.align === "right" ? "num-th" : "";
      if (c.highlight) cls += " active-stat-th";
      const sortable = !["_rank", "name", "team", "position", "contract_type"].includes(c.key);
      const sortableCls = sortable ? " sortable" : "";
      const arrow = c.key === sort.key ? `<span class="sort-arrow">${sort.asc ? "▲" : "▼"}</span>` : "";
      // titleOverride wins over glossaryId; infoIcon forces an "i" badge even
      // when the column isn't backed by a glossary entry.
      const tipText = c.titleOverride || (c.glossaryId ? tipFor(c.glossaryId) : "");
      const titleAttr = tipText ? ` title="${escHtml(tipText)}"` : "";
      const info = (c.glossaryId || c.infoIcon) ? ' <span class="lb-info-icon">i</span>' : "";
      return `<th class="${cls}${sortableCls}" data-col="${escHtml(c.key)}"${titleAttr}>${escHtml(c.label)}${info}${arrow}</th>`;
    }).join("");

    // Wire sortable
    $$(`${theadId} th.sortable`).forEach((th) => {
      th.onclick = () => {
        const k = th.dataset.col;
        if (state.sortKey === k) state.sortAsc = !state.sortAsc;
        else { state.sortKey = k; state.sortAsc = false; }
        renderListView();
      };
    });

    if (!rows.length) {
      $(tbodyId).innerHTML = `<tr><td colspan="${allCols.length}" class="no-games-msg">No players match the current filters.</td></tr>`;
      $(footerId).textContent = "0 players";
      return;
    }

    $(tbodyId).innerHTML = rows.map((r, i) => {
      const cells = allCols.map((c) => {
        if (c.key === "_rank") return `<td class="num-cell rank-col">${i + 1}</td>`;
        if (c.isPlayer) return `<td class="cv-player-cell"><a href="#" class="cv-player-link" data-player-id="${escHtml(r.playerId || "")}" data-name="${escHtml(r.name)}">${escHtml(r.name)}</a></td>`;
        if (c.isTeam) {
          const logo = r.team_logo ? `<img src="${escHtml(r.team_logo)}" alt="" class="cv-team-logo" onerror="this.style.display='none'" />` : "";
          return `<td class="cv-team-cell">${logo}<span>${escHtml(r.team)}</span></td>`;
        }
        if (c.isPos) {
          const pcls = classifyPosition(r.position);
          return `<td><span class="pos-pill pos-${pcls === 'F' ? 'f' : pcls === 'D' ? 'd' : 'g'}">${escHtml(r.position)}</span></td>`;
        }
        const v = r[c.key];
        const display = c.fmt ? c.fmt(v, r) : (v == null ? "—" : escHtml(v));
        let cls = c.align === "right" ? "num-cell" : "";
        if (c.highlight) cls += " active-stat-cell";
        return `<td class="${cls}">${display}</td>`;
      }).join("");
      return `<tr class="cv-row" data-name="${escHtml(r.name)}">${cells}</tr>`;
    }).join("");

    $(footerId).textContent = `${rows.length} player${rows.length === 1 ? "" : "s"} shown`;
  }

  // ---------------------------------------------------------------------------
  // Render: scatter view (3 variants)
  // ---------------------------------------------------------------------------
  function shapeForType(type) {
    return ({ "ELC": "rectRot", "Bridge": "triangle", "Veteran": "rect", "Market Rate": "circle" })[type] || "circle";
  }

  function colorForRow(r) {
    if (r.contract_type === "ELC") return "#E8A020";
    if (r.surplus_value == null) return "#586478";
    return r.surplus_value >= 0 ? "#1DB954" : "#E8302A";
  }

  function destroyScatter() {
    if (state.scatter) { try { state.scatter.destroy(); } catch (_) {} state.scatter = null; }
  }

  function buildScatterData(variant) {
    const rows = filterRows().filter((r) => r.gar != null);
    return rows.map((r) => {
      let x, y, sz = 6;
      if (variant === "off-vs-def") {
        x = r.off_gar; y = r.def_gar; sz = Math.max(4, Math.min(14, (r.cap_hit || 0) * 1.0));
      } else if (variant === "age-vs-surplus") {
        x = r.age; y = r.surplus_value; sz = 6;
      } else {
        x = r.cap_hit; y = r.gar; sz = 6;
      }
      return {
        x, y,
        r: sz,
        backgroundColor: colorForRow(r),
        borderColor: colorForRow(r),
        pointStyle: shapeForType(r.contract_type),
        radius: sz,
        hoverRadius: sz + 3,
        _row: r,
      };
    }).filter((p) => p.x != null && p.y != null);
  }

  function renderScatter() {
    $$(".cv-view").forEach((v) => v.hidden = true);
    $('[data-cv-view="scatter"]').hidden = false;
    $("#cv-view-note").hidden = true;

    const variant = state.scatterVariant;
    const data = buildScatterData(variant);

    const labels = {
      "cap-vs-gar": { x: "Cap Hit ($M)", y: "GAR" },
      "off-vs-def": { x: "Offensive GAR", y: "Defensive GAR" },
      "age-vs-surplus": { x: "Age", y: "Surplus Value ($M)" },
    }[variant];

    const note = {
      "cap-vs-gar": "Cap hit on the X axis vs. GAR on the Y. Players above the diagonal are out-producing their cap; below, they're underperforming.",
      "off-vs-def": "Two-way balance: top-right corner = elite both ends. Dot size scales with cap hit.",
      "age-vs-surplus": "Which age brackets generate the most contract value across the league?",
    }[variant];
    $("#cv-scatter-note").textContent = note;

    destroyScatter();
    const ctx = $("#cv-scatter").getContext("2d");
    state.scatter = new Chart(ctx, {
      type: "scatter",
      data: { datasets: [{ label: "Players", data, parsing: false }] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { text: labels.x, display: true, color: "#C0CCDD" }, ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.1)" } },
          y: { title: { text: labels.y, display: true, color: "#C0CCDD" }, ticks: { color: "#C0CCDD" }, grid: { color: "rgba(120,140,180,0.1)" } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => {
                const r = item.raw._row;
                return `${r.name} (${r.team}) — Cap $${r.cap_hit}M, GAR ${r.gar}, Surplus ${fmtMoney(r.surplus_value)}`;
              },
            },
          },
        },
        onClick: (evt, items) => {
          if (!items.length) return;
          const r = items[0].element.options._row || items[0].raw._row;
          if (r) openSidePanel(r);
        },
      },
    });
  }

  // ---------------------------------------------------------------------------
  // Side panel
  // ---------------------------------------------------------------------------
  function openSidePanel(row) {
    const panel = $("#cv-side-panel");
    const backdrop = $("#cv-side-backdrop");
    const content = $("#cv-side-content");

    // Compute Off/Def split bar — share of total positive GAR
    const offShare = (row.off_gar != null && row.gar > 0) ? Math.round((row.off_gar / row.gar) * 100) : 50;
    const defShare = 100 - offShare;

    const sustainText = row.sustainability_score == null
      ? "Not enough data to compute a Sustainability Score."
      : (row.sustainability_score >= 3
          ? `Currently outperforming xGAR by ${row.sustainability_score.toFixed(2)} GAR — at meaningful regression risk in future seasons.`
          : (row.sustainability_score <= -3
              ? `Underperforming xGAR by ${Math.abs(row.sustainability_score).toFixed(2)} GAR — likely to bounce back if shot quality holds.`
              : `In line with underlying play (gap of ${row.sustainability_score.toFixed(2)} GAR is normal variance).`));

    const verdict = row.surplus_value == null
      ? ""
      : (row.contract_type === "ELC"
          ? `On a CBA-mandated ELC, this player provides ${fmtMoney(row.surplus_value)} above market — but ELCs are structurally cheap so the comparison is unfair.`
          : (row.surplus_value >= 0
              ? `This player provides ${fmtMoney(row.surplus_value)} in surplus value above their market rate contract.`
              : `This player's contract represents a ${fmtMoney(Math.abs(row.surplus_value))} overpayment relative to their current production.`));

    content.innerHTML = `
      <div class="cv-side-head">
        <img class="cv-side-headshot" src="${escHtml(row.headshot)}" alt="" onerror="this.style.display='none'" />
        <div class="cv-side-bio">
          <div class="cv-side-name">${escHtml(row.name)} ${ageFlagIcon(row.age_flag)}</div>
          <div class="cv-side-meta">
            <img src="${escHtml(row.team_logo)}" alt="" class="cv-side-team-logo" onerror="this.style.display='none'" />
            ${escHtml(row.team)} &middot; ${escHtml(row.position)} ${row.age != null ? `&middot; Age ${row.age}` : ""}
          </div>
          ${contractBadge(row.contract_type)}
        </div>
      </div>

      <div class="cv-side-stats">
        <div class="cv-side-stat-row"><span>Cap Hit</span><strong>${fmtMoney(row.cap_hit)}</strong></div>
        <div class="cv-side-stat-row"><span>Years Remaining</span><strong>${row.years_remaining ?? "—"}</strong></div>
        <div class="cv-side-stat-row"><span>Expiry</span><strong>${row.expiry_year ?? "—"}</strong></div>
      </div>

      <div class="cv-side-section">
        <h4>GAR Breakdown</h4>
        <div class="cv-side-stat-row"><span>Total GAR</span><strong>${fmtNum(row.gar)}</strong></div>
        <div class="cv-side-stat-row"><span>Off GAR</span><strong>${fmtNum(row.off_gar)}</strong></div>
        <div class="cv-side-stat-row"><span>Def GAR</span><strong>${fmtNum(row.def_gar)}</strong></div>
        <div class="cv-side-stat-row"><span>PP GAR</span><strong>${fmtNum(row.pp_gar)}</strong></div>
        <div class="cv-side-stat-row"><span>PK GAR</span><strong>${fmtNum(row.pk_gar)}</strong></div>
        <div class="cv-offdef-bar" title="Offensive vs Defensive contribution share">
          <div class="cv-offdef-off" style="width: ${offShare}%"><span>Off ${offShare}%</span></div>
          <div class="cv-offdef-def" style="width: ${defShare}%"><span>Def ${defShare}%</span></div>
        </div>
      </div>

      <div class="cv-side-section">
        <h4>Sustainability</h4>
        <div class="cv-side-stat-row"><span>GAR − xGAR</span><strong>${row.sustainability_score == null ? "—" : (row.sustainability_score >= 0 ? "+" : "") + row.sustainability_score.toFixed(2)}</strong></div>
        <p class="cv-side-prose">${escHtml(sustainText)}</p>
      </div>

      <div class="cv-side-section">
        <h4>Aging Trajectory</h4>
        <div class="chart-wrap" style="height: 180px;"><canvas id="cv-side-chart"></canvas></div>
        <p class="cv-side-prose-small">Projection applies a simple aging curve (8%/yr decline starting at 30 for forwards, 33 for D).</p>
      </div>

      <div class="cv-side-section cv-side-surplus-section">
        <h4>Surplus Value</h4>
        <div class="cv-side-surplus-big">${row.surplus_value == null ? "—" : (row.surplus_value >= 0 ? "+" : "−") + "$" + Math.abs(row.surplus_value).toFixed(2) + "M"}</div>
        <p class="cv-side-prose">${escHtml(verdict)}</p>
      </div>

      <div class="cv-side-section">
        <h4>Analytical Summary</h4>
        <p class="cv-side-prose">${escHtml(buildAnalyticalSummary(row))}</p>
      </div>
    `;

    panel.hidden = false;
    backdrop.hidden = false;
    setTimeout(() => panel.classList.add("open"), 10);

    // Render aging trajectory chart (synthetic — no historical career data here)
    setTimeout(() => renderAgingChart(row), 30);
  }

  function buildAnalyticalSummary(r) {
    const parts = [];
    if (r.contract_type === "ELC") parts.push("CBA-mandated entry-level deal — value rankings are structurally inflated.");
    else if (r.contract_type === "Veteran") parts.push("Veteran contract: assess against age-curve decline rather than current GAR alone.");
    else if (r.contract_type === "Bridge") parts.push("Bridge deal — short-term, often discounted to the team in exchange for delaying long-term commitment.");

    if (r.surplus_value != null) {
      if (r.surplus_value > 5) parts.push("Major surplus — among the team's most valuable contract assets.");
      else if (r.surplus_value > 0) parts.push("Positive surplus — paying their way at market rate or slightly below.");
      else if (r.surplus_value < -5) parts.push("Significant overpay risk; reduces team flexibility.");
      else parts.push("Slightly under market value at current cap charge.");
    }

    if (r.sustainability_score != null && r.sustainability_score >= 3) {
      parts.push("Running hot vs. underlying chance creation — expect some regression.");
    }
    if (r.sustainability_score != null && r.sustainability_score <= -3) {
      parts.push("Underlying chance creation outpaces actual production — buy-low candidate.");
    }
    if (r.age != null && r.age >= 32) {
      parts.push("Age 32+: contract value will erode through normal aging effects, especially in years 3+.");
    }
    return parts.join(" ");
  }

  let sideChart = null;
  function renderAgingChart(r) {
    const canvas = document.getElementById("cv-side-chart");
    if (!canvas) return;
    if (sideChart) try { sideChart.destroy(); } catch (_) {}

    // Synthetic last-4-seasons + remaining contract years projection
    const labels = [];
    const data = [];
    const colors = [];
    const currentGar = r.gar || 0;
    // Last 4 seasons — placeholder constant + small variation; real history would require additional fetches
    for (let i = 4; i >= 1; i--) {
      labels.push(`-${i} season`);
      data.push(Math.max(0, currentGar - (i * 0.5)));
      colors.push("rgba(43, 107, 240, 0.6)");
    }
    labels.push("This season");
    data.push(currentGar);
    colors.push("rgba(232, 160, 32, 0.85)");
    // Remaining contract years — projected
    const remaining = Math.min(5, r.years_remaining || 0);
    const decay = (r.position === "D") ? 0.92 : 0.92;
    let proj = currentGar;
    for (let i = 1; i <= remaining; i++) {
      labels.push(`+${i}`);
      const startDecline = (r.position === "D") ? 33 : 30;
      if ((r.age || 0) + i >= startDecline) proj *= decay;
      data.push(Math.max(0, +proj.toFixed(2)));
      colors.push("rgba(232, 48, 42, 0.5)");
    }

    sideChart = new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [{ label: "GAR", data, backgroundColor: colors, borderWidth: 0 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#C0CCDD", font: { size: 10 } }, grid: { display: false } },
          y: { ticks: { color: "#C0CCDD", font: { size: 10 } }, grid: { color: "rgba(120,140,180,0.1)" }, beginAtZero: true },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  function closeSidePanel() {
    const panel = $("#cv-side-panel");
    const backdrop = $("#cv-side-backdrop");
    panel.classList.remove("open");
    setTimeout(() => { panel.hidden = true; backdrop.hidden = true; }, 220);
  }

  // ---------------------------------------------------------------------------
  // ELC Watch
  // ---------------------------------------------------------------------------
  function loadElcWatch() {
    const tbody = $("#cv-elc-tbody");
    const thead = $("#cv-elc-thead");
    if (!tbody || !thead) return;

    const elcs = state.raw.filter((r) => r.contract_type === "ELC");
    elcs.sort((a, b) => (b.gar || 0) - (a.gar || 0));

    thead.innerHTML = `
      <tr>
        <th class="rank-col">#</th>
        <th>Player</th>
        <th>Team</th>
        <th>Pos</th>
        <th class="num-th">Age</th>
        <th class="num-th">Current GAR</th>
        <th class="num-th">ELC Expiry</th>
        <th>Status @ Expiry</th>
        <th class="num-th">Proj. GAR @ Ext</th>
        <th class="num-th">Proj. Market Value</th>
        <th>Priority</th>
      </tr>
    `;

    if (!elcs.length) {
      tbody.innerHTML = `<tr><td colspan="11" class="no-games-msg">No ELCs found in current dataset.</td></tr>`;
      return;
    }

    tbody.innerHTML = elcs.map((r, i) => {
      // Project GAR forward to expiry year using the aging curve (assume same year-3 baseline)
      const yearsToExpiry = r.years_remaining || 0;
      const projGar = projectGar(r.gar, r.position, r.age, yearsToExpiry);
      const projMv = projGar != null ? +(projGar * 1.85).toFixed(2) : null;
      // RFA if age < 27, else UFA
      const ageAtExp = (r.age || 0) + yearsToExpiry;
      const status = ageAtExp >= 27 ? "UFA" : "RFA";
      // Priority
      const priority = projMv == null ? "—"
        : (projMv >= 12 ? '<span class="cv-priority cv-priority-high">High</span>'
          : projMv >= 6 ? '<span class="cv-priority cv-priority-med">Medium</span>'
          : '<span class="cv-priority cv-priority-low">Low</span>');

      return `
        <tr class="cv-row" data-name="${escHtml(r.name)}">
          <td class="num-cell rank-col">${i + 1}</td>
          <td><a href="#" class="cv-player-link" data-name="${escHtml(r.name)}">${escHtml(r.name)}</a></td>
          <td>${escHtml(r.team)}</td>
          <td><span class="pos-pill pos-${classifyPosition(r.position) === 'F' ? 'f' : classifyPosition(r.position) === 'D' ? 'd' : 'g'}">${escHtml(r.position)}</span></td>
          <td class="num-cell">${r.age ?? "—"}</td>
          <td class="num-cell">${fmtNum(r.gar)}</td>
          <td class="num-cell">${r.expiry_year}</td>
          <td>${status}</td>
          <td class="num-cell">${fmtNum(projGar)}</td>
          <td class="num-cell">${projMv != null ? "$" + projMv + "M" : "—"}</td>
          <td>${priority}</td>
        </tr>
      `;
    }).join("");
  }

  function projectGar(gar, position, age, yearsForward) {
    if (gar == null || age == null) return null;
    const startDecline = position === "D" ? 33 : 30;
    let projected = gar;
    for (let y = 1; y <= yearsForward; y++) {
      if (age + y >= startDecline) projected *= 0.92;
    }
    return +projected.toFixed(2);
  }

  // ---------------------------------------------------------------------------
  // Team Cap Efficiency
  // ---------------------------------------------------------------------------
  async function loadTeamEfficiency() {
    const tbody = $("#cv-team-eff-tbody");
    if (!tbody) return;
    try {
      const r = await fetch("/api/team-cap-efficiency");
      const data = await r.json();
      const teams = data.teams || [];
      state.teamData = teams;
      tbody.innerHTML = teams.map((t, i) => {
        const surplusCls = t.total_surplus >= 0 ? "cv-surplus-pos" : "cv-surplus-neg";
        const sign = t.total_surplus >= 0 ? "+" : "";
        return `
          <tr>
            <td class="num-cell rank-col">${i + 1}</td>
            <td class="cv-team-cell">
              <img src="${escHtml(t.team_logo)}" alt="" class="cv-team-logo" onerror="this.style.display='none'" />
              <span>${escHtml(t.team)}</span>
            </td>
            <td class="num-cell">${t.players_counted}</td>
            <td class="num-cell">$${t.total_cap.toFixed(2)}M</td>
            <td class="num-cell">${t.total_gar.toFixed(2)}</td>
            <td class="num-cell">${t.avg_gar_per_dollar.toFixed(2)}</td>
            <td class="num-cell ${surplusCls}">${sign}$${Math.abs(t.total_surplus).toFixed(2)}M</td>
          </tr>
        `;
      }).join("");
    } catch (e) {
      tbody.innerHTML = `<tr><td colspan="7" class="no-games-msg">Failed to load team data.</td></tr>`;
    }
  }

  // ---------------------------------------------------------------------------
  // View dispatcher
  // ---------------------------------------------------------------------------
  function renderActiveView() {
    if (state.view === "scatter") renderScatter();
    else renderListView();
  }

  // ---------------------------------------------------------------------------
  // Wire UI
  // ---------------------------------------------------------------------------
  function initViewToggle() {
    $$('.view-toggle-btn[data-cv-view]').forEach((btn) => {
      btn.addEventListener("click", () => {
        $$('.view-toggle-btn[data-cv-view]').forEach((b) => {
          const on = b === btn;
          b.classList.toggle("active", on);
          b.setAttribute("aria-selected", on ? "true" : "false");
        });
        state.view = btn.dataset.cvView;
        state.sortKey = null;
        renderActiveView();
      });
    });

    // Scatter variant
    $$('[data-cv-scatter-variant] .seg-btn').forEach((btn) => {
      btn.addEventListener("click", () => {
        $$('[data-cv-scatter-variant] .seg-btn').forEach((b) => b.classList.toggle("active", b === btn));
        state.scatterVariant = btn.dataset.value;
        renderScatter();
      });
    });
  }

  function initFilterHandlers() {
    // Contract type
    $$('[data-cv-filter="type"] .seg-btn').forEach((btn) => {
      btn.addEventListener("click", () => {
        $$('[data-cv-filter="type"] .seg-btn').forEach((b) => b.classList.toggle("active", b === btn));
        state.contractType = btn.dataset.value;
        state.sortKey = null;
        renderActiveView();
      });
    });
    // Position
    $$('[data-cv-filter="position"] .seg-btn').forEach((btn) => {
      btn.addEventListener("click", () => {
        $$('[data-cv-filter="position"] .seg-btn').forEach((b) => b.classList.toggle("active", b === btn));
        state.position = btn.dataset.value;
        renderActiveView();
      });
    });
    // Team
    const teamSel = $("#cv-team-filter");
    if (teamSel) teamSel.addEventListener("change", (e) => { state.team = e.target.value; renderActiveView(); });
    // Expiry
    const expirySel = $("#cv-expiry-filter");
    if (expirySel) expirySel.addEventListener("change", (e) => { state.expiry = e.target.value; renderActiveView(); });
    // Search
    const search = $("#cv-search");
    if (search) {
      let timer;
      search.addEventListener("input", (e) => {
        clearTimeout(timer);
        state.search = e.target.value;
        timer = setTimeout(() => renderActiveView(), 150);
      });
    }
    // Page-level position
    const pageBar = document.querySelector("#player-pos-filter");
    if (pageBar) pageBar.addEventListener("click", (e) => {
      if (e.target.closest(".pos-btn")) setTimeout(() => renderActiveView(), 0);
    });
  }

  function initRowClicks() {
    document.addEventListener("click", (e) => {
      const link = e.target.closest(".cv-player-link");
      if (!link) return;
      e.preventDefault();
      const name = link.dataset.name;
      const row = state.raw.find((r) => r.name === name);
      if (row) openSidePanel(row);
    });
    $("#cv-side-close")?.addEventListener("click", closeSidePanel);
    $("#cv-side-backdrop")?.addEventListener("click", closeSidePanel);
  }

  function initGlossaryJump() {
    document.addEventListener("click", (e) => {
      const link = e.target.closest('[data-jump-to="glossary"]');
      if (!link) return;
      e.preventDefault();
      const tab = document.querySelector('.subpage-tab[data-subpage="glossary"]');
      if (tab) tab.click();
    });
  }

  function init() {
    initPlayersTabs();
    initViewToggle();
    initFilterHandlers();
    initRowClicks();
    initGlossaryJump();
    // Load eagerly so data is ready when the user clicks the Contract Value tab
    loadContractValues();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
