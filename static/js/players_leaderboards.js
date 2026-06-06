/* =========================================================
   Players → Leaderboards tab — pill selector + sortable filterable table
   Sources: MoneyPuck (skaters/goalies), contract values, the composite
   rating model (/api/composite-ratings), and RAPM (/api/rapm-leaders).
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = window.dashUtil ? window.dashUtil.escHtml : (s) => String(s ?? "");

  // Pull glossary tooltip text for the info icons
  const GLOSSARY = (window.GLOSSARY_DATA && window.GLOSSARY_DATA.stats) || [];
  const glossaryById = {};
  GLOSSARY.forEach((s) => { glossaryById[s.id] = s; });
  function tipFor(glossaryId) {
    const s = glossaryById[glossaryId];
    return s ? s.def : "";
  }

  // z-score / rating formatters (signed)
  const zfmt = (dp) => (v) => (v == null || isNaN(v) ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(dp));
  const toiFmt = (v) => (v == null ? "—" : Math.round(v).toLocaleString() + " min");

  // Composite-rating colour band: green > 1.5, gold 0.5–1.5, neutral 0–0.5, red < 0
  function compositeClass(v) {
    if (v == null) return "";
    if (v > 1.5) return "lb-comp-green";
    if (v >= 0.5) return "lb-comp-gold";
    if (v >= 0) return "lb-comp-neutral";
    return "lb-comp-red";
  }

  // ---------------------------------------------------------------------------
  // Custom column layouts for the model-driven stats
  // ---------------------------------------------------------------------------
  const COMPOSITE_COLS = [
    { key: "name",                   label: "Player",   align: "left", isPlayer: true },
    { key: "team",                   label: "Team",     align: "left", isTeam: true },
    { key: "position",               label: "Pos",      align: "left", isPos: true },
    { key: "composite_rating",       label: "Composite",align: "right", highlight: true,
      glossaryId: "composite-rating", fmt: zfmt(3), colorScale: true },
    { key: "rapm_component",         label: "RAPM",     align: "right", fmt: zfmt(2), glossaryId: "rapm" },
    { key: "individual_component",   label: "Indiv",    align: "right", fmt: zfmt(2) },
    { key: "relative_xgf_component", label: "RelxGF",   align: "right", fmt: zfmt(2), glossaryId: "rel-xgf" },
    { key: "playmaking_component",   label: "Play",     align: "right", fmt: zfmt(2) },
    { key: "power_play_component",   label: "PP",       align: "right", fmt: zfmt(2) },
    { key: "penalty_kill_component", label: "PK",       align: "right", fmt: zfmt(2) },
    { key: "toi_minutes",            label: "TOI",      align: "right", fmt: toiFmt },
  ];

  const RAPM_COLS = [
    { key: "name",            label: "Player",     align: "left", isPlayer: true },
    { key: "team",            label: "Team",       align: "left", isTeam: true },
    { key: "position",        label: "Pos",        align: "left", isPos: true },
    { key: "total_rapm",      label: "Total RAPM", align: "right", highlight: true,
      glossaryId: "rapm", fmt: zfmt(3) },
    { key: "offensive_rapm",  label: "Off RAPM",   align: "right", fmt: zfmt(3) },
    { key: "defensive_rapm",  label: "Def RAPM",   align: "right", fmt: zfmt(3) },
    { key: "toi_minutes",     label: "TOI",        align: "right", fmt: toiFmt },
  ];

  const PP_RATING_COLS = [
    { key: "name",                 label: "Player",    align: "left", isPlayer: true },
    { key: "team",                 label: "Team",      align: "left", isTeam: true },
    { key: "position",             label: "Pos",       align: "left", isPos: true },
    { key: "power_play_component", label: "PP Rating", align: "right", highlight: true,
      glossaryId: "composite-rating", fmt: zfmt(2) },
    { key: "toi_minutes",          label: "TOI",       align: "right", fmt: toiFmt },
  ];

  const PK_RATING_COLS = [
    { key: "name",                   label: "Player",    align: "left", isPlayer: true },
    { key: "team",                   label: "Team",      align: "left", isTeam: true },
    { key: "position",               label: "Pos",       align: "left", isPos: true },
    { key: "penalty_kill_component", label: "PK Rating", align: "right", highlight: true,
      glossaryId: "composite-rating", fmt: zfmt(2) },
    { key: "toi_minutes",            label: "TOI",       align: "right", fmt: toiFmt },
  ];

  const RAPM_NOTE = "Note: RAPM controls for teammates and opponents, but collinearity " +
    "may underrate players with extremely high linemate overlap — co-stars who share " +
    "most of their ice time (e.g. Draisaitl/McDavid, Makar/MacKinnon).";


  // ---------------------------------------------------------------------------
  // Stat catalog
  //   source: 'skaters' | 'goalies' | 'contract' | 'composite' | 'rapm'
  //   customCols: full column layout (overrides universal + extra cols)
  // ---------------------------------------------------------------------------
  const STATS = [
    // ---- Scoring ----
    { id: "points",       label: "Points",        key: "points",       source: "skaters", group: "scoring", fmt: (v) => v, suffix: "" },
    { id: "goals",        label: "Goals",         key: "goals",        source: "skaters", group: "scoring", fmt: (v) => v, suffix: "" },
    { id: "assists",      label: "Assists",       key: "assists",      source: "skaters", group: "scoring", fmt: (v) => v, suffix: "" },
    { id: "plus_minus",   label: "Plus/Minus",    key: "plus_minus",   source: "skaters", group: "scoring", fmt: (v) => (v >= 0 ? "+" : "") + v, suffix: "" },
    { id: "toi_per_game_min", label: "TOI per game", key: "toi_per_game_min", source: "skaters", group: "scoring", fmt: (v) => v.toFixed(2), suffix: "" },
    { id: "shooting_pct", label: "Shooting %",    key: "shooting_pct", source: "skaters", group: "scoring", fmt: (v) => v.toFixed(2), suffix: "%" },

    // ---- Advanced Possession ----
    { id: "cf_pct",       label: "CF% (Corsi)",      key: "cf_pct",      source: "skaters", group: "possession", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "corsi", minToi: 200 },
    { id: "ff_pct",       label: "FF% (Fenwick)",    key: "ff_pct",      source: "skaters", group: "possession", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "fenwick", minToi: 200 },
    { id: "xgf_pct",      label: "xGF%",             key: "xgf_pct",     source: "skaters", group: "possession", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "xgf-pct", minToi: 200 },
    { id: "xga_pct",      label: "xGA%",             key: "xga_pct",     source: "skaters", group: "possession", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "xga-pct", asc: true, minToi: 200 },
    { id: "hdcf_pct",     label: "HDCF%",            key: "hdcf_pct",    source: "skaters", group: "possession", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "hdcf-pct", minToi: 200 },
    { id: "rel_cf_pct",   label: "Relative CF%",     key: "rel_cf_pct",  source: "skaters", group: "possession", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2), suffix: "%", glossaryId: "rel-cf", minToi: 200 },
    { id: "rel_xgf_pct",  label: "Relative xGF%",    key: "rel_xgf_pct", source: "skaters", group: "possession", fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(2), suffix: "%", glossaryId: "rel-xgf", minToi: 200 },

    // ---- Individual Impact ----
    { id: "composite",    label: "Composite Rating", key: "composite_rating", source: "composite", group: "impact",
      glossaryId: "composite-rating", customCols: COMPOSITE_COLS },
    { id: "rapm",         label: "RAPM",            key: "total_rapm", source: "rapm", group: "impact",
      glossaryId: "rapm", customCols: RAPM_COLS, tableNote: RAPM_NOTE },
    { id: "gar",          label: "GAR",             key: "gar",        source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "gar" },
    { id: "xgar",         label: "xGAR",            key: "xgar",       source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "xgar" },
    { id: "game_score",   label: "Game Score",      key: "game_score", source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "game-score" },
    { id: "ixg_60",       label: "ixG per 60",      key: "ixg_60",     source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "ixg", minToi: 200 },
    { id: "icf_60",       label: "iCF per 60",      key: "icf_60",     source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "icf", minToi: 200 },
    { id: "iff_60",       label: "iFF per 60",      key: "iff_60",     source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "iff", minToi: 200 },
    { id: "ihdcf_60",     label: "iHDCF per 60",    key: "ihdcf_60",   source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "ihdcf", minToi: 200 },
    { id: "rush_pct",     label: "Rush Shot %",     key: "rush_pct",   source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "rush-shot-pct", minToi: 200 },
    { id: "rebound_pct",  label: "Rebound Shot %",  key: "rebound_pct",source: "skaters", group: "impact", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "rebound-shot-pct", minToi: 200 },

    // ---- Usage & Context ----
    { id: "zone_start_pct", label: "Zone Start %",  key: "zone_start_pct", source: "skaters", group: "usage", fmt: (v) => v.toFixed(2), suffix: "%", glossaryId: "zone-start-pct", minToi: 200 },
    { id: "es_toi",         label: "TOI at ES",     key: "toi_per_game_min", source: "skaters", group: "usage", fmt: (v) => v.toFixed(2), suffix: " min" },
    { id: "pp_toi",         label: "TOI on PP",     blocked: true, source: "skaters", group: "usage", glossaryId: "pp-toi-game" },
    { id: "pk_toi",         label: "TOI on PK",     blocked: true, source: "skaters", group: "usage", glossaryId: "pk-toi-game" },
    { id: "qoc",            label: "QoC (proxy)",   blocked: true, source: "skaters", group: "usage", glossaryId: "qoc" },
    { id: "qot",            label: "QoT (proxy)",   blocked: true, source: "skaters", group: "usage", glossaryId: "qot" },

    // ---- Special Teams ----
    { id: "pp_rating",  label: "PP Rating",  key: "power_play_component",   source: "composite", group: "special",
      glossaryId: "composite-rating", customCols: PP_RATING_COLS },
    { id: "pk_rating",  label: "PK Rating",  key: "penalty_kill_component", source: "composite", group: "special",
      glossaryId: "composite-rating", customCols: PK_RATING_COLS },
    { id: "pp_points",  label: "PP Points",  blocked: true, source: "skaters", group: "special" },
    { id: "pp_goals",   label: "PP Goals",   blocked: true, source: "skaters", group: "special" },
    { id: "pp_assists", label: "PP Assists", blocked: true, source: "skaters", group: "special" },
    { id: "pk_toi_st",  label: "PK TOI",     blocked: true, source: "skaters", group: "special" },
    { id: "pk_pm",      label: "PK +/-",     blocked: true, source: "skaters", group: "special" },

    // ---- Contract Value ----
    { id: "cap_hit",       label: "Cap Hit",       key: "cap_hit",       source: "contract", group: "contract", fmt: (v) => "$" + v.toFixed(2) + "M", suffix: "" },
    { id: "gar_per_mil",   label: "GAR per $1M",   key: "gar_per_million",source: "contract", group: "contract", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "gar-per-mil" },
    { id: "xgar_per_mil",  label: "xGAR per $1M",  key: "xgar_per_million",source: "contract", group: "contract", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "xgar-per-mil" },
    { id: "pts_per_mil",   label: "Points per $1M",key: "points_per_million",source: "contract", group: "contract", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "pts-per-mil" },

    // ---- Goalies ----
    { id: "gsax",   label: "GSAX (Goalies)",  key: "gsax",     source: "goalies", group: "impact", fmt: (v) => v.toFixed(2), suffix: "", glossaryId: "gsax" },
  ];

  const GROUPS = [
    { id: "scoring",    label: "Scoring" },
    { id: "possession", label: "Advanced Possession" },
    { id: "impact",     label: "Individual Impact" },
    { id: "usage",      label: "Usage & Context" },
    { id: "special",    label: "Special Teams" },
    { id: "contract",   label: "Contract / Value" },
  ];

  // Cached datasets
  const data = { skaters: null, goalies: null, contract: null, composite: null, rapm: null };

  function teamLogo(team) {
    return team ? `https://assets.nhle.com/logos/nhl/svg/${team}_light.svg` : "";
  }

  async function fetchSkaters() {
    if (data.skaters) return data.skaters;
    const r = await fetch("/api/players-full?min_toi_min=0");
    const d = await r.json();
    data.skaters = d.players || [];
    return data.skaters;
  }

  async function fetchGoalies() {
    if (data.goalies) return data.goalies;
    const r = await fetch("/api/goalies-full?min_games=0");
    const d = await r.json();
    data.goalies = d.goalies || [];
    return data.goalies;
  }

  async function fetchContracts() {
    if (data.contract) return data.contract;
    const r = await fetch("/api/contract-values");
    const d = await r.json();
    if (d.blocked) {
      data.contract = { blocked: true, message: d.message };
    } else {
      const skaters = await fetchSkaters();
      const skaterByName = {};
      skaters.forEach((s) => { skaterByName[s.name.toLowerCase()] = s; });
      const players = (d.players || []).map((c) => {
        const sk = skaterByName[c.name.toLowerCase()] || null;
        const points = sk?.points ?? 0;
        const xgar_per_million = c.xgar != null && c.cap_hit > 0 ? +(c.xgar / c.cap_hit).toFixed(2) : null;
        const points_per_million = c.cap_hit > 0 ? +(points / c.cap_hit).toFixed(2) : null;
        return {
          ...c,
          playerId: sk?.playerId,
          headshot: sk?.headshot,
          team_logo: sk?.team_logo || teamLogo(c.team),
          gp: sk?.gp ?? c.games ?? 0,
          toi_min: sk?.toi_min ?? c.toi_min ?? 0,
          xgar_per_million,
          points_per_million,
        };
      });
      data.contract = { players, source: d.source };
    }
    return data.contract;
  }

  // Composite — qualified players only; normalize schema for the generic renderer.
  async function fetchComposite() {
    if (data.composite) return data.composite;
    const r = await fetch("/api/composite-ratings");
    const d = await r.json();
    if (!d.built) {
      data.composite = { blocked: true, message: d.message || "Composite ratings not built." };
      return data.composite;
    }
    const players = (d.full_dataset || [])
      .filter((p) => p.sample_size_flag === "ok")
      .map((p) => ({
        ...p,
        name: p.player_name,
        playerId: p.player_id,
        toi_min: p.toi_minutes,
        team_logo: teamLogo(p.team),
      }));
    data.composite = { players };
    return data.composite;
  }

  // RAPM — full dataset; normalize schema. RAPM rows carry no position field
  // (the model never tracked it), so join position from the MoneyPuck skater set.
  async function fetchRapm() {
    if (data.rapm) return data.rapm;
    const r = await fetch("/api/rapm-leaders");
    const d = await r.json();
    if (!d.trained) {
      data.rapm = { blocked: true, message: d.message || "RAPM model not trained." };
      return data.rapm;
    }
    const skaters = await fetchSkaters();
    const posById = {};
    skaters.forEach((s) => { if (s.playerId) posById[s.playerId] = s.position; });
    const players = (d.full_dataset || d.top_total || []).map((p) => ({
      ...p,
      name: p.player_name,
      playerId: p.player_id,
      position: posById[p.player_id] || "",   // "" for retired/out-of-league players
      toi_min: p.toi_minutes,
      team_logo: teamLogo(p.team),
    }));
    data.rapm = { players };
    return data.rapm;
  }

  // Universal columns shown alongside the active stat (generic stats only)
  function universalCols() {
    return [
      { key: "name",      label: "Player", align: "left", isPlayer: true },
      { key: "team",      label: "Team",   align: "left", isTeam: true },
      { key: "position",  label: "Pos",    align: "left", isPos: true },
      { key: "gp",        label: "GP",     align: "right" },
      { key: "toi_min",   label: "TOI",    align: "right", fmt: (v) => v + " min" },
    ];
  }

  function extraColsFor(stat) {
    if (stat.id === "gar") return [{ key: "xgar", label: "xGAR", align: "right", fmt: (v) => v != null ? v.toFixed(2) : "—" }];
    if (stat.id === "xgar") return [{ key: "gar", label: "GAR", align: "right", fmt: (v) => v != null ? v.toFixed(2) : "—" }];
    if (stat.id === "ixg_60") return [{ key: "icf_60", label: "iCF/60", align: "right", fmt: (v) => v.toFixed(2) }];
    if (stat.id === "rel_cf_pct") return [{ key: "cf_pct", label: "CF%", align: "right", fmt: (v) => v.toFixed(2) + "%" }];
    if (stat.id === "shooting_pct") return [{ key: "shots", label: "Shots", align: "right" }];
    if (stat.id === "gar_per_mil") return [
      { key: "cap_hit", label: "Cap Hit", align: "right", fmt: (v) => v != null ? "$" + v.toFixed(2) + "M" : "—" },
      { key: "gar", label: "GAR", align: "right", fmt: (v) => v != null ? v.toFixed(2) : "—" },
    ];
    if (stat.id === "xgar_per_mil") return [
      { key: "cap_hit", label: "Cap Hit", align: "right", fmt: (v) => v != null ? "$" + v.toFixed(2) + "M" : "—" },
      { key: "xgar", label: "xGAR", align: "right", fmt: (v) => v != null ? v.toFixed(2) : "—" },
    ];
    return [];
  }

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------
  const state = {
    statId: "points", position: "all", team: "all",
    minToi: 200, minGames: 10, search: "",
    sortKey: null, sortAsc: false,
  };

  function activeStat() {
    return STATS.find((s) => s.id === state.statId) || STATS[0];
  }

  // ---------------------------------------------------------------------------
  // Pill selector
  // ---------------------------------------------------------------------------
  function renderPills() {
    const wrap = $("#lb-pill-groups");
    if (!wrap) return;
    wrap.innerHTML = GROUPS.map((g) => {
      const pills = STATS.filter((s) => s.group === g.id).map((s) => {
        const blocked = s.blocked ? " lb-pill-blocked" : "";
        const tooltip = s.glossaryId ? tipFor(s.glossaryId) : "";
        const tipAttr = tooltip ? ` title="${esc(tooltip)}"` : "";
        return `<button class="lb-pill${blocked}" data-stat-id="${esc(s.id)}"${tipAttr}>${esc(s.label)}${s.glossaryId ? ' <span class="lb-pill-info">i</span>' : ""}</button>`;
      }).join("");
      return `<div class="lb-pill-group"><div class="lb-pill-group-label">${esc(g.label)}</div><div class="lb-pill-row">${pills}</div></div>`;
    }).join("");
    updateActivePill();
    wrap.addEventListener("click", (e) => {
      const pill = e.target.closest(".lb-pill");
      if (pill) selectStat(pill.dataset.statId);
    });
  }

  function updateActivePill() {
    $$(".lb-pill").forEach((p) => p.classList.toggle("active", p.dataset.statId === state.statId));
  }

  // ---------------------------------------------------------------------------
  // Filters
  // ---------------------------------------------------------------------------
  function applyPosition(rows, pos) {
    if (pos === "all") return rows;
    if (pos === "F") return rows.filter((r) => ["C", "L", "R"].includes(r.position));
    return rows.filter((r) => r.position === pos);
  }
  function applyTeam(rows, team) {
    return team === "all" ? rows : rows.filter((r) => r.team === team);
  }
  function applySearch(rows, q) {
    if (!q) return rows;
    const ql = q.toLowerCase();
    return rows.filter((r) => (r.name || "").toLowerCase().includes(ql));
  }
  function applyMinToi(rows, stat) {
    // composite / rapm carry their own qualification — don't TOI-filter them
    if (stat.source === "composite" || stat.source === "rapm") return rows;
    if (stat.source === "goalies") return rows.filter((r) => r.gp >= state.minGames);
    const minToi = Math.max(state.minToi, stat.minToi || 0);
    return rows.filter((r) => r.toi_min >= minToi);
  }
  function pageLevelPosition() {
    const btn = document.querySelector("#player-pos-filter .pos-btn.active");
    return btn?.dataset.pos || "all";
  }

  // ---------------------------------------------------------------------------
  // Render table
  // ---------------------------------------------------------------------------
  async function renderTable() {
    const stat = activeStat();
    const spinner = $("#lb-spinner");
    const wrap = $("#lb-table-wrap");
    const error = $("#lb-error");
    const blocked = $("#lb-blocked");
    const sourceLabel = $("#lb-source-label");

    spinner.hidden = false;
    wrap.hidden = true;
    error.hidden = true;
    blocked.hidden = true;

    if (stat.blocked) {
      spinner.hidden = true;
      blocked.innerHTML = `<strong>${esc(stat.label)} unavailable</strong><br>This metric is not currently sourced from a self-generated table.`;
      blocked.hidden = false;
      sourceLabel.textContent = "Source: not currently generated";
      return;
    }

    // Soft handle for invalid metric/position combinations.
    // The Players page exposes both skater and goalie metrics. When the
    // user pairs a skater-only metric with the Goalies position filter
    // (or vice versa), show an informational note in the muted blocked-card
    // style instead of an empty table or a red error.
    const pageLevelPos = pageLevelPosition();
    const isGoalieMetric = stat.source === "goalies";
    const isSkaterMetric = !isGoalieMetric;
    if (pageLevelPos === "G" && isSkaterMetric) {
      spinner.hidden = true;
      blocked.innerHTML = `<strong>${esc(stat.label)} is a skater metric.</strong><br>` +
        `GSAX is the available analytical metric for goalies. ` +
        `Switch to GSAX or change the position filter to view this metric.`;
      blocked.hidden = false;
      sourceLabel.textContent = "";
      return;
    }
    if (pageLevelPos !== "G" && pageLevelPos !== "all" && isGoalieMetric) {
      spinner.hidden = true;
      blocked.innerHTML = `<strong>${esc(stat.label)} is goalie-only.</strong><br>` +
        `Select the Goalies (G) position filter to view this metric.`;
      blocked.hidden = false;
      sourceLabel.textContent = "";
      return;
    }

    let rows, sourceText;
    try {
      if (stat.source === "skaters") {
        rows = await fetchSkaters();
        sourceText = "Source: Hockey Intelligence Hub models · counts from MoneyPuck";
      } else if (stat.source === "goalies") {
        rows = await fetchGoalies();
        sourceText = "Source: Hockey Intelligence Hub models · counts from MoneyPuck";
      } else if (stat.source === "contract") {
        const c = await fetchContracts();
        if (c.blocked) {
          spinner.hidden = true;
          blocked.innerHTML = `<strong>Contract data unavailable</strong><br>${esc(c.message || "")}`;
          blocked.hidden = false;
          sourceLabel.textContent = "Source: PuckPedia (blocked) → manual";
          return;
        }
        rows = c.players;
        sourceText = `Source: ${c.source === "puckpedia" ? "PuckPedia" : "Manual contract dataset"} + MoneyPuck`;
      } else if (stat.source === "composite") {
        const c = await fetchComposite();
        if (c.blocked) {
          spinner.hidden = true;
          blocked.innerHTML = `<strong>Composite ratings unavailable</strong><br>${esc(c.message || "")}`;
          blocked.hidden = false;
          return;
        }
        rows = c.players;
        sourceText = "Source: Composite Rating model (Phase 2.5 v2)";
      } else if (stat.source === "rapm") {
        const c = await fetchRapm();
        if (c.blocked) {
          spinner.hidden = true;
          blocked.innerHTML = `<strong>RAPM unavailable</strong><br>${esc(c.message || "")}`;
          blocked.hidden = false;
          return;
        }
        rows = c.players;
        sourceText = "Source: RAPM ridge-regression model (Phase 2, 6 seasons)";
      }
    } catch (e) {
      spinner.hidden = true;
      error.hidden = false;
      return;
    }

    sourceLabel.textContent = sourceText;

    // Page-level position filter (skater metric × skater pos; mismatches
    // already short-circuited above with a soft message).
    if (pageLevelPos !== "all" && pageLevelPos !== "G") {
      rows = applyPosition(rows, pageLevelPos);
    }

    rows = applyPosition(rows, state.position);
    rows = applyTeam(rows, state.team);
    rows = applyMinToi(rows, stat);
    rows = applySearch(rows, state.search);

    // After all filters: if zero rows survive, show a soft "no players match"
    // message instead of an empty table. (Hard errors during the fetch are
    // still surfaced via the red `error` element in the catch block above.)
    if (!rows.length) {
      spinner.hidden = true;
      const filterBits = [];
      if (pageLevelPos !== "all") filterBits.push(`position ${pageLevelPos}`);
      if (state.team) filterBits.push(`team ${state.team}`);
      if (state.minToi > 0) filterBits.push(`min TOI ${state.minToi}`);
      if (state.search) filterBits.push(`search "${state.search}"`);
      const filterText = filterBits.length
        ? ` (filters: ${filterBits.join(", ")})`
        : "";
      blocked.innerHTML = `<strong>No players match these filters.</strong><br>` +
        `Try clearing the position, team, or min-TOI filter${filterText}.`;
      blocked.hidden = false;
      return;
    }

    const sortKey = state.sortKey || stat.key;
    const ascending = state.sortKey ? state.sortAsc : (stat.asc === true);
    rows = rows.slice().sort((a, b) => {
      const av = a[sortKey], bv = b[sortKey];
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") return ascending ? av - bv : bv - av;
      return ascending ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });

    spinner.hidden = true;
    wrap.hidden = false;
    renderTableHTML(stat, rows);
  }

  function renderTableHTML(stat, rows) {
    const theadRow = $("#lb-thead-row");
    const tbody = $("#lb-tbody");
    const footer = $("#lb-footer");

    // Build column set: custom layout, or universal + active + extras
    let cols = [{ key: "_rank", label: "#", align: "right" }];
    if (stat.customCols) {
      stat.customCols.forEach((c) => cols.push(c));
    } else {
      universalCols().forEach((c) => cols.push(c));
      cols.push({ key: stat.key, label: stat.label, align: "right", isActiveStat: true,
                  highlight: true, glossaryId: stat.glossaryId, fmt: stat.fmt, suffix: stat.suffix });
      extraColsFor(stat).forEach((c) => cols.push(c));
    }

    const curSortKey = state.sortKey || stat.key;
    const curSortAsc = state.sortKey ? state.sortAsc : (stat.asc === true);

    theadRow.innerHTML = cols.map((c) => {
      let cls = c.align === "right" ? "num-th" : "";
      if (c.highlight) cls += " active-stat-th";
      const sortable = !["_rank", "team", "position", "name"].includes(c.key) && !c.isPlayer && !c.isTeam && !c.isPos;
      const sortableCls = sortable ? " sortable" : "";
      const arrow = c.key === curSortKey ? `<span class="sort-arrow">${curSortAsc ? "▲" : "▼"}</span>` : "";
      const titleAttr = c.glossaryId ? ` title="${esc(tipFor(c.glossaryId))}"` : "";
      const infoIcon = c.glossaryId ? ' <span class="lb-info-icon">i</span>' : "";
      return `<th class="${cls}${sortableCls}" data-col="${esc(c.key)}"${titleAttr}>${esc(c.label)}${infoIcon}${arrow}</th>`;
    }).join("");

    $$("#lb-thead-row th.sortable").forEach((th) => {
      th.onclick = () => {
        const k = th.dataset.col;
        if (state.sortKey === k) state.sortAsc = !state.sortAsc;
        else { state.sortKey = k; state.sortAsc = false; }
        renderTable();
      };
    });

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="${cols.length}" class="no-games-msg">No players match the current filters.</td></tr>`;
      footer.textContent = "0 players";
      setTableNote(stat);
      return;
    }

    tbody.innerHTML = rows.map((r, i) => {
      const cells = cols.map((c) => {
        if (c.key === "_rank") return `<td class="num-cell rank-col">${i + 1}</td>`;
        if (c.isPlayer) {
          return `<td class="lb-player-cell"><a href="#" class="lb-player-link" data-player-id="${esc(r.playerId ?? "")}">${esc(r[c.key] ?? r.name)}</a></td>`;
        }
        if (c.isTeam) {
          const logo = r.team_logo ? `<img src="${esc(r.team_logo)}" alt="" class="lb-team-logo" onerror="this.style.display='none'" />` : "";
          return `<td class="lb-team-cell">${logo}<span>${esc(r.team)}</span></td>`;
        }
        if (c.isPos) {
          const p = r.position === "D" ? "d" : r.position === "G" ? "g" : "f";
          return `<td><span class="pos-pill pos-${p}">${esc(r.position)}</span></td>`;
        }
        const v = r[c.key];
        let display;
        if (v == null) display = "—";
        else if (c.fmt) display = c.fmt(v, r);
        else display = String(v);
        if (c.suffix && v != null && c.fmt) display += "";  // suffix already in fmt path below
        if (c.isActiveStat && v != null && c.suffix) display = (c.fmt ? c.fmt(v) : String(v)) + c.suffix;
        let cls = c.align === "right" ? "num-cell" : "";
        if (c.highlight) cls += " active-stat-cell";
        // Composite colour scale
        if (c.colorScale) {
          return `<td class="${cls}"><span class="${compositeClass(v)}">${esc(display)}</span></td>`;
        }
        return `<td class="${cls}">${esc(display)}</td>`;
      }).join("");
      return `<tr>${cells}</tr>`;
    }).join("");

    footer.textContent = `${rows.length} player${rows.length === 1 ? "" : "s"} shown`;
    setTableNote(stat);
  }

  // Optional note rendered below the table (e.g. RAPM collinearity caveat)
  function setTableNote(stat) {
    let note = $("#lb-table-note");
    if (!note) {
      note = document.createElement("p");
      note.id = "lb-table-note";
      note.className = "lb-table-note";
      const wrap = $("#lb-table-wrap");
      if (wrap) wrap.appendChild(note);
    }
    if (stat.tableNote) {
      note.textContent = stat.tableNote;
      note.hidden = false;
    } else {
      note.hidden = true;
    }
  }

  function setupPlayerLinkClicks() {
    document.addEventListener("click", (e) => {
      const link = e.target.closest(".lb-player-link");
      if (!link) return;
      e.preventDefault();
      const pid = link.dataset.playerId;
      if (!pid) return;
      const tab = document.querySelector('.players-tab[data-ptab="player-search"]');
      if (tab) tab.click();
      if (window.playerLookup && window.playerLookup.loadProfile) {
        window.playerLookup.loadProfile(Number(pid));
      }
    });
  }

  // ---------------------------------------------------------------------------
  // Filter wiring
  // ---------------------------------------------------------------------------
  function setupFilters() {
    const posBar = document.querySelector('[data-lb-filter="position"]');
    if (posBar) {
      posBar.addEventListener("click", (e) => {
        const btn = e.target.closest(".seg-btn");
        if (!btn) return;
        $$(".seg-btn", posBar).forEach((b) => b.classList.toggle("active", b === btn));
        state.position = btn.dataset.value;
        state.sortKey = null;
        renderTable();
      });
    }
    const teamSel = $("#lb-team-filter");
    if (teamSel) teamSel.addEventListener("change", (e) => { state.team = e.target.value; renderTable(); });

    const slider = $("#lb-min-toi");
    const readout = $("#lb-min-toi-readout");
    if (slider && readout) {
      slider.addEventListener("input", (e) => {
        const v = Number(e.target.value);
        const stat = activeStat();
        if (stat.source === "goalies") {
          state.minGames = Math.round(v / 50);
          readout.textContent = `${state.minGames} GP`;
        } else {
          state.minToi = v;
          readout.textContent = `${v} min`;
        }
      });
      slider.addEventListener("change", () => renderTable());
    }

    const search = $("#lb-search");
    if (search) {
      let timer = null;
      search.addEventListener("input", (e) => {
        clearTimeout(timer);
        state.search = e.target.value;
        timer = setTimeout(() => renderTable(), 150);
      });
    }

    const pageBar = document.querySelector("#player-pos-filter");
    if (pageBar) {
      pageBar.addEventListener("click", (e) => {
        if (!e.target.closest(".pos-btn")) return;
        setTimeout(() => renderTable(), 0);
      });
    }
  }

  function populateTeamDropdown() {
    const sel = $("#lb-team-filter");
    if (!sel || !data.skaters) return;
    Array.from(new Set(data.skaters.map((p) => p.team).filter(Boolean))).sort().forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t; opt.textContent = t;
      sel.appendChild(opt);
    });
  }

  // ---------------------------------------------------------------------------
  // Public: select a stat (called by Overview "View Full Leaderboard" buttons)
  // ---------------------------------------------------------------------------
  function selectStat(statId) {
    const stat = STATS.find((s) => s.id === statId);
    if (!stat) return;
    state.statId = statId;
    state.sortKey = null;
    state.sortAsc = false;
    if (stat.source === "goalies") {
      state.position = "G";
      $$('[data-lb-filter="position"] .seg-btn').forEach((b) => b.classList.toggle("active", b.dataset.value === "G"));
    }
    updateActivePill();
    renderTable();
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  let firstLoaded = false;
  async function load() {
    if (firstLoaded) return;
    firstLoaded = true;
    renderPills();
    setupFilters();
    setupPlayerLinkClicks();
    await Promise.all([fetchSkaters(), fetchGoalies()]);
    populateTeamDropdown();
    renderTable();
  }

  function watchTabActivation() {
    const tab = document.querySelector('.players-tab[data-ptab="leaderboards"]');
    if (tab) tab.addEventListener("click", load);
  }

  function init() { watchTabActivation(); }

  window.playersLeaderboards = { selectStat };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
