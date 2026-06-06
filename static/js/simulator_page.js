/* =========================================================
   NHL → Simulator page  (Phase 5 — wires Phase 4 engine to UI)

   Four sub-tabs, all using the same `.players-tab` / `.players-tab-panel`
   pattern as the Players section. Each sub-tab lazy-loads on first activation
   so we don't pay the cost up-front for tabs the user never opens.
   ========================================================= */
(function () {
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const esc = (s) => window.dashUtil ? window.dashUtil.escHtml(s) : String(s ?? "");

  // 32 NHL team abbreviations (matches Phase 4 build_team_strength.py)
  const ALL_TEAMS = ["ANA","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET",
    "EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT",
    "SEA","SJS","STL","TBL","TOR","UTA","VAN","VGK","WPG","WSH"];

  const NHL_LOGO = (abbrev) =>
    `https://assets.nhle.com/logos/nhl/svg/${abbrev}_light.svg`;

  // Loaded state per sub-tab (one-shot loads)
  const loaded = {
    teamStrength: false,
    lineupEditor: false,
    gameSimulator: false,
    projections: false,
  };

  // Per-tab module state
  const state = {
    teamStrength: { data: null, handle: null },
    lineupEditor: {
      team: null, defaultStrength: null, roster: null,
      customLineup: null,        // {forwards:[{player_id,...}], defense:[…], goalies:[…]}
      playerPool: null,
      editingSlot: null,         // {kind:"F"|"D"|"G", index:int, panelEl}
      customStrengthData: null,
    },
    gameSimulator: {
      away: "EDM", home: "FLA", nSims: 5000,
      chart: null, lastResults: null,
    },
    projections: { data: null },
  };

  // ---------------------------------------------------------------------------
  // Sub-tab switcher (parallel to contract_value.js initPlayersTabs)
  // ---------------------------------------------------------------------------
  function initSubTabs() {
    const tabs = $$(".sim-tab");
    tabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const key = tab.dataset.stab;
        tabs.forEach((t) => {
          const on = t.dataset.stab === key;
          t.classList.toggle("active", on);
          t.setAttribute("aria-selected", on ? "true" : "false");
        });
        $$(".sim-tab-panel").forEach((panel) => {
          panel.hidden = panel.dataset.stab !== key;
        });
        lazyLoad(key);
      });
    });
    // Eager-load the currently-active sub-tab once the Simulator subpage shows.
    // (We also watch the top-level subpage tab so we load on first navigation.)
    const subpageTab = document.querySelector('.subpage-tab[data-subpage="simulator"]');
    if (subpageTab) {
      subpageTab.addEventListener("click", () => {
        const active = $(".sim-tab.active");
        if (active) lazyLoad(active.dataset.stab);
      });
    }
  }

  function lazyLoad(key) {
    if (key === "team-strength" && !loaded.teamStrength) loadTeamStrength();
    else if (key === "lineup-editor" && !loaded.lineupEditor) loadLineupEditor();
    else if (key === "game-simulator" && !loaded.gameSimulator) loadGameSimulator();
    else if (key === "projections" && !loaded.projections) loadProjections();
  }

  // ===========================================================================
  // SUB-TAB 1 — TEAM STRENGTH
  // ===========================================================================
  function loadTeamStrength() {
    loaded.teamStrength = true;
    const container = $("#sim-team-strength-container");
    if (!container) return;
    container.innerHTML = `<div class="leaderboard-loading">Loading team strengths…</div>`;
    fetch("/api/team-strength")
      .then((r) => r.json())
      .then((d) => {
        if (!d.built || !d.teams) {
          container.innerHTML = `<div class="leaderboard-blocked">team_strength.csv not built. Run <code>python3 model/build_team_strength.py</code>.</div>`;
          return;
        }
        state.teamStrength.data = d.teams;
        renderTeamStrength(d.teams);
      })
      .catch((e) => {
        container.innerHTML = `<div class="leaderboard-blocked">Failed to load team_strength: ${esc(e.message || e)}</div>`;
      });
  }

  function renderTeamStrength(teams) {
    const container = $("#sim-team-strength-container");
    container.innerHTML = "";
    const cellPill = (v, key) => {
      // Color-code: green if above league avg (positive z), red if below
      let cls = "metric-mid";
      if (v >= 0.50) cls = "metric-elite";
      else if (v >= 0.15) cls = "metric-good";
      else if (v <= -0.50) cls = "metric-poor";
      else if (v <= -0.15) cls = "metric-weak";
      return `<span class="metric-pill ${cls}">${(v > 0 ? "+" : "") + v.toFixed(3)}</span>`;
    };
    const logoCell = (abbrev) =>
      `<span class="sim-team-cell">
         <img class="sim-team-logo-sm" src="${NHL_LOGO(abbrev)}" alt="${esc(abbrev)}" />
         <span class="sim-team-abbrev">${esc(abbrev)}</span>
       </span>`;
    const handle = window.dashUtil.buildLeaderboard({
      container,
      id: "sim-ts",
      title: "Team Strength Rankings",
      subtitle: "Sortable. Click a row to load the team in the Season Simulator.",
      sourceLabel: "Source: composite_ratings_sim.csv + NHL API rosters + MoneyPuck 25-26 goalies",
      showSearch: true,
      searchKeys: ["team"],
      activeSort: { key: "team_strength", dir: "desc" },
      getRows: () => state.teamStrength.data,
      columns: [
        { key: "team",  label: "Team",
          fmt: (v) => logoCell(v) },
        { key: "team_strength",  label: "Overall",
          fmt: (v) => cellPill(v, "team_strength"), align: "right" },
        { key: "top6_forward_strength",  label: "Top-6 F",
          fmt: (v) => v.toFixed(3), align: "right" },
        { key: "bottom6_forward_strength", label: "Bottom-6 F",
          fmt: (v) => v.toFixed(3), align: "right" },
        { key: "top_pair_d_strength", label: "Top-pair D",
          fmt: (v) => v.toFixed(3), align: "right" },
        { key: "bottom_pair_d_strength", label: "Bottom-pair D",
          fmt: (v) => v.toFixed(3), align: "right" },
        { key: "goalie_strength", label: "Goalies",
          fmt: (v) => cellPill(v, "goalie_strength"), align: "right" },
      ],
    });
    state.teamStrength.handle = handle;
    // Row click → jump to Lineup Editor with team preloaded
    container.addEventListener("click", (e) => {
      const tr = e.target.closest("tbody tr");
      if (!tr) return;
      const cells = tr.querySelectorAll("td");
      // The team cell is the 2nd <td> (rank is 1st). Pull abbrev from the inner span.
      const abbrev = tr.querySelector(".sim-team-abbrev")?.textContent?.trim();
      if (!abbrev) return;
      jumpToLineupEditor(abbrev);
    });
    const ts = new Date().toLocaleString("en-US", {
      timeZone: "America/New_York", hour12: false,
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
    $("#sim-ts-last-updated").textContent = `· Last updated ${ts}`;
  }

  function jumpToLineupEditor(team) {
    const tab = document.querySelector('.sim-tab[data-stab="lineup-editor"]');
    if (!tab) return;
    tab.click();
    setTimeout(() => {
      const sel = $("#sim-le-team-select");
      if (sel) {
        sel.value = team;
        sel.dispatchEvent(new Event("change"));
      }
    }, 50);
  }

  // ===========================================================================
  // SUB-TAB 2 — LINEUP EDITOR
  // ===========================================================================
  function loadLineupEditor() {
    loaded.lineupEditor = true;
    populateTeamSelect("#sim-le-team-select");
    fetch("/api/sim-player-pool")
      .then((r) => r.json())
      .then((d) => {
        state.lineupEditor.playerPool = d.players || [];
      })
      .catch(() => { state.lineupEditor.playerPool = []; });
    initSeasonSimulator();   // wire the Phase 5.1 controls
    const sel = $("#sim-le-team-select");
    if (sel) {
      sel.addEventListener("change", () => loadTeamRoster(sel.value));
      // Load EDM by default for showpiece (McDavid test)
      sel.value = "EDM";
      loadTeamRoster("EDM");
    }
    $("#sim-le-reset-btn")?.addEventListener("click", () => {
      const team = state.lineupEditor.team;
      if (team) loadTeamRoster(team);
    });
  }

  function populateTeamSelect(sel) {
    const el = typeof sel === "string" ? $(sel) : sel;
    if (!el) return;
    el.innerHTML = ALL_TEAMS
      .map((t) => `<option value="${t}">${t}</option>`)
      .join("");
  }

  function loadTeamRoster(team) {
    const rosterEl = $("#sim-le-roster");
    rosterEl.innerHTML = `<div class="leaderboard-loading">Loading ${esc(team)} roster…</div>`;
    state.lineupEditor.team = team;
    Promise.all([
      fetch(`/api/sim-team-roster/${team}`).then((r) => r.json()),
      fetch("/api/team-strength").then((r) => r.json()),
    ]).then(([roster, ts]) => {
      if (roster.error) {
        rosterEl.innerHTML = `<div class="leaderboard-blocked">Roster load failed: ${esc(roster.error)}</div>`;
        return;
      }
      state.lineupEditor.roster = roster;
      // Reset custom lineup to the default, then reshape forwards into
      // LW/C/RW slot order so the classic-lines layout reads naturally.
      const flatF = (roster.forwards_active || []).map((p) => p);
      state.lineupEditor.customLineup = {
        forwards: assignForwardsToLineSlots(flatF),
        defense:  (roster.defense_active  || []).map((p) => p),
        goalies:  (roster.goalies || []).map((p) => p),
      };
      const def = (ts.teams || []).find((x) => x.team === team) || {};
      state.lineupEditor.defaultStrength = def;
      renderLineupEditor();
      // Default strength → show; custom = default until user edits.
      $("#sim-strength-panel").hidden = false;
      $("#sim-le-default-val").textContent = formatStrengthVal(def.team_strength);
      $("#sim-le-custom-val").textContent = formatStrengthVal(def.team_strength);
      $("#sim-le-delta-val").textContent = "+0.000";
      $("#sim-le-delta-val").className = "sim-strength-value muted";
    }).catch((e) => {
      rosterEl.innerHTML = `<div class="leaderboard-blocked">Roster load failed: ${esc(e.message || e)}</div>`;
    });
  }

  function formatStrengthVal(v) {
    if (v == null || isNaN(v)) return "—";
    return (v >= 0 ? "+" : "") + Number(v).toFixed(3);
  }

  // ---------- Classic-lines layout helpers ----------
  //
  // The lines layout is PURELY VISUAL. The /api/team-strength-custom endpoint
  // re-sorts the roster server-side by effective_war when assigning tier
  // minutes, so where you place a player among the 12 forward slots has no
  // direct effect on the team-strength calculation. The visual structure makes
  // the roster easier to read; the engine still auto-tiers by talent.
  //
  // Each forward slot is one of LW / C / RW; each D slot is LD / RD.
  // Goalies are Starter / Backup.
  const FWD_SLOT_LABELS  = ["LW","C","RW","LW","C","RW","LW","C","RW","LW","C","RW"];
  const D_SLOT_LABELS    = ["LD","RD","LD","RD","LD","RD"];
  const LINE_NAMES       = ["First Line","Second Line","Third Line","Fourth Line"];
  const PAIR_NAMES       = ["First Pair","Second Pair","Third Pair"];

  function assignForwardsToLineSlots(flatForwards) {
    // Take 12 forwards already sorted by composite_war desc (server returns
    // them in that order). Group into 4 lines of 3, then within each line try
    // to slot each player into their listed position (LW/C/RW). Players that
    // don't have a matching natural slot fall into whichever wing is still
    // open. Result: a flat 12-element array indexed [L1_LW, L1_C, L1_RW,
    // L2_LW, ..., L4_RW].
    const out = [];
    for (let i = 0; i < 4; i++) {
      const linePlayers = flatForwards.slice(i*3, i*3+3).filter(Boolean);
      const slots = { LW: null, C: null, RW: null };
      const leftover = [];
      linePlayers.forEach((p) => {
        const code = ((p.position || "C") + "")[0].toUpperCase();
        if (code === "L" && !slots.LW) slots.LW = p;
        else if (code === "C" && !slots.C) slots.C = p;
        else if (code === "R" && !slots.RW) slots.RW = p;
        else leftover.push(p);
      });
      ["LW","C","RW"].forEach((k) => {
        if (!slots[k] && leftover.length) slots[k] = leftover.shift();
      });
      out.push(slots.LW, slots.C, slots.RW);
    }
    return out;
  }

  function renderLineupEditor() {
    const rosterEl = $("#sim-le-roster");
    const r = state.lineupEditor.roster;
    const cl = state.lineupEditor.customLineup;
    if (!r || !cl) return;

    const slotCell = (p, kind, index, slotLabel) => {
      const editBtn = `<button class="sim-slot-edit btn-eh-small" data-kind="${kind}" data-index="${index}">Edit</button>`;
      const labelTag = `<div class="sim-slot-tag">${slotLabel}</div>`;
      if (!p) {
        return `<div class="sim-line-cell empty" data-kind="${kind}" data-index="${index}">
          ${labelTag}
          <div class="sim-cell-body">
            <div class="sim-cell-name muted">— replacement-level —</div>
            <div class="sim-cell-war muted">WAR —</div>
          </div>
          ${editBtn}
        </div>`;
      }
      const war = (p.composite_war != null) ? p.composite_war : 0;
      const name = p.name || `${p.first_name || ""} ${p.last_name || ""}`.trim();
      return `<div class="sim-line-cell" data-kind="${kind}" data-index="${index}">
        ${labelTag}
        <div class="sim-cell-body">
          <div class="sim-cell-name">${esc(name || "—")}</div>
          <div class="sim-cell-war"><span class="mono">${formatStrengthVal(war)}</span></div>
        </div>
        ${editBtn}
      </div>`;
    };

    const goalieCell = (g, index, label) => {
      const editBtn = `<button class="sim-slot-edit btn-eh-small" data-kind="G" data-index="${index}">Edit</button>`;
      if (!g) {
        return `<div class="sim-line-cell sim-goalie-cell empty" data-kind="G" data-index="${index}">
          <div class="sim-slot-tag">${label}</div>
          <div class="sim-cell-body">
            <div class="sim-cell-name muted">— no goalie —</div>
            <div class="sim-cell-war muted">WAR —</div>
          </div>
          ${editBtn}
        </div>`;
      }
      return `<div class="sim-line-cell sim-goalie-cell" data-kind="G" data-index="${index}">
        <div class="sim-slot-tag">${label}</div>
        <div class="sim-cell-body">
          <div class="sim-cell-name">${esc(g.name)}</div>
          <div class="sim-cell-war">
            <span class="mono">WAR ${formatStrengthVal(g.goalie_war)}</span>
            <span class="muted">· ${g.games_played} GP · ${(g.gp_share * 100).toFixed(0)}% share</span>
          </div>
        </div>
        ${editBtn}
      </div>`;
    };

    const F = cl.forwards;
    const D = cl.defense;
    const G = cl.goalies;

    // Forwards — 4 horizontal lines
    const linesHtml = [0,1,2,3].map((li) => {
      const cells = [0,1,2].map((sl) => {
        const idx = li*3 + sl;
        return slotCell(F[idx], "F", idx, FWD_SLOT_LABELS[idx]);
      }).join("");
      return `
        <div class="sim-line-row">
          <div class="sim-line-row-label">${LINE_NAMES[li]}</div>
          <div class="sim-line-row-cells sim-line-row-cells-3">${cells}</div>
        </div>`;
    }).join("");

    // Defense — 3 horizontal pairs
    const pairsHtml = [0,1,2].map((pi) => {
      const cells = [0,1].map((sl) => {
        const idx = pi*2 + sl;
        return slotCell(D[idx], "D", idx, D_SLOT_LABELS[idx]);
      }).join("");
      return `
        <div class="sim-line-row">
          <div class="sim-line-row-label">${PAIR_NAMES[pi]}</div>
          <div class="sim-line-row-cells sim-line-row-cells-2">${cells}</div>
        </div>`;
    }).join("");

    // Goalies — 2 horizontal slots
    const goalieRowHtml = `
      <div class="sim-line-row">
        <div class="sim-line-row-label">Goalies</div>
        <div class="sim-line-row-cells sim-line-row-cells-2">
          ${goalieCell(G[0], 0, "Starter")}
          ${goalieCell(G[1], 1, "Backup")}
        </div>
      </div>`;

    rosterEl.innerHTML = `
      <div class="sim-lineup-team-header">
        <img class="sim-lineup-team-logo" src="${NHL_LOGO(r.team)}" alt="${esc(r.team)}" />
        <h3>${esc(r.team)} — Active Roster</h3>
      </div>

      <div class="sim-lines-section">
        <h4 class="sim-lineup-block-title">Forwards</h4>
        ${linesHtml}
      </div>

      <div class="sim-lines-section">
        <h4 class="sim-lineup-block-title">Defense</h4>
        ${pairsHtml}
      </div>

      <div class="sim-lines-section">
        <h4 class="sim-lineup-block-title">Goalies</h4>
        ${goalieRowHtml}
      </div>

      <div id="sim-swap-panel" class="sim-swap-panel" hidden></div>
    `;
    // Hook up Edit buttons
    rosterEl.querySelectorAll(".sim-slot-edit").forEach((btn) => {
      btn.addEventListener("click", (e) => openSwapPanel(e.currentTarget.dataset));
    });
  }

  function openSwapPanel(slotData) {
    const panel = $("#sim-swap-panel");
    if (!panel) return;
    const { kind, index } = slotData;
    const pool = state.lineupEditor.playerPool || [];
    const filterPos = kind === "D" ? "D" : (kind === "G" ? "G" : "F");
    const filtered = pool.filter((p) => {
      const c = p.position?.[0] || "F";
      return filterPos === "D" ? c === "D" : (filterPos === "F" ? c !== "D" && c !== "G" : c === "G");
    });
    panel.innerHTML = `
      <div class="sim-swap-header">
        <span>Swap slot — ${kind === "F" ? "Forward" : kind === "D" ? "Defenseman" : "Goalie"} #${(+index) + 1}</span>
        <input class="sim-swap-search" type="search" placeholder="Search by player name…" />
        <button class="btn-eh-small" id="sim-swap-cancel">Cancel</button>
        <button class="btn-eh-small" id="sim-swap-remove">Remove player (replacement-level)</button>
      </div>
      <div class="sim-swap-results" id="sim-swap-results"></div>
    `;
    panel.hidden = false;
    state.lineupEditor.editingSlot = { kind, index: +index };
    function renderResults(query) {
      const q = (query || "").toLowerCase().trim();
      let list = filtered;
      if (q) {
        list = filtered.filter((p) => p.name.toLowerCase().includes(q));
      }
      list = list.slice(0, 50);
      $("#sim-swap-results").innerHTML = list.map((p) => `
        <div class="sim-swap-result" data-pid="${p.player_id}">
          <span class="sim-swap-name">${esc(p.name)}</span>
          <span class="sim-swap-pos muted">${esc(p.position)}</span>
          <span class="sim-swap-team muted">${esc(p.team)}</span>
          <span class="sim-swap-war mono">WAR ${formatStrengthVal(p.composite_war)}</span>
          <span class="sim-swap-gp muted">GP ${(p.expected_gp_share * 100).toFixed(0)}%</span>
        </div>
      `).join("") || `<div class="muted">No matches.</div>`;
      $$("#sim-swap-results .sim-swap-result").forEach((row) => {
        row.addEventListener("click", () => {
          const pid = +row.dataset.pid;
          const player = pool.find((p) => p.player_id === pid);
          if (player) commitSwap(player);
        });
      });
    }
    panel.querySelector(".sim-swap-search").addEventListener("input", (e) => renderResults(e.target.value));
    panel.querySelector("#sim-swap-cancel").addEventListener("click", () => { panel.hidden = true; });
    panel.querySelector("#sim-swap-remove").addEventListener("click", () => commitSwap(null));
    renderResults("");
  }

  function commitSwap(player) {
    const slot = state.lineupEditor.editingSlot;
    if (!slot) return;
    const cl = state.lineupEditor.customLineup;
    const arr = slot.kind === "F" ? cl.forwards : slot.kind === "D" ? cl.defense : cl.goalies;
    // Replace by index; player==null = replacement-level (empty slot)
    if (!player) {
      arr[slot.index] = null;
    } else if (slot.kind === "G") {
      // Goalies use a different cell shape than skaters — the goalieCell
      // renderer reads {name, goalie_war, games_played, gp_share}. The
      // backend /api/sim-team-roster delivers goalies with these fields and
      // we mirror that shape here so cosmetic display matches.
      arr[slot.index] = {
        player_id: player.player_id,
        name: player.name,
        goalie_war: player.goalie_war ?? player.composite_war ?? 0,
        games_played: player.games_played ?? 0,
        gsaa: player.gsaa ?? 0,
        gp_share: player.expected_gp_share ?? 0,
        role: slot.index === 0 ? "Starter" : "Backup",
      };
    } else {
      arr[slot.index] = {
        player_id: player.player_id,
        first_name: player.name.split(" ").slice(0, -1).join(" "),
        last_name:  player.name.split(" ").slice(-1).join(""),
        position: player.position,
        composite_war: player.composite_war,
        expected_gp_share: player.expected_gp_share,
        rated: true,
        tier_label: arr[slot.index]?.tier_label || "",
        tier_minutes: arr[slot.index]?.tier_minutes || 0,
      };
    }
    $("#sim-swap-panel").hidden = true;
    renderLineupEditor();
    recomputeCustomStrength();
  }

  function recomputeCustomStrength() {
    const team = state.lineupEditor.team;
    const cl = state.lineupEditor.customLineup;
    if (!team || !cl) return;
    // Flatten the active lineup into a roster_override payload.
    const allPlayers = [
      ...cl.forwards.filter(Boolean).map((p) => ({
        player_id: p.player_id, position: p.position || "C",
        roster_role: "F", first_name: p.first_name, last_name: p.last_name })),
      ...cl.defense.filter(Boolean).map((p) => ({
        player_id: p.player_id, position: "D",
        roster_role: "D", first_name: p.first_name, last_name: p.last_name })),
      ...cl.goalies.filter(Boolean).map((g) => ({
        player_id: g.player_id, position: "G",
        roster_role: "G", first_name: "", last_name: g.name })),
    ];
    fetch("/api/team-strength-custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team, roster_override: allPlayers }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          $("#sim-le-custom-val").textContent = "error";
          return;
        }
        const def = d.default?.team_strength ?? 0;
        const cust = d.custom?.team_strength ?? 0;
        const delta = d.delta_team_strength ?? (cust - def);
        $("#sim-le-default-val").textContent = formatStrengthVal(def);
        $("#sim-le-custom-val").textContent = formatStrengthVal(cust);
        const deltaEl = $("#sim-le-delta-val");
        deltaEl.textContent = (delta > 0 ? "+" : "") + delta.toFixed(3);
        deltaEl.className = "sim-strength-value " + (delta > 0.01 ? "green" : delta < -0.01 ? "red" : "muted");
      })
      .catch(() => {
        $("#sim-le-custom-val").textContent = "error";
      });
  }

  // ===========================================================================
  // SUB-TAB 3 — GAME SIMULATOR
  // ===========================================================================
  function loadGameSimulator() {
    loaded.gameSimulator = true;
    populateTeamSelect("#sim-gs-away-select");
    populateTeamSelect("#sim-gs-home-select");
    const awayEl = $("#sim-gs-away-select");
    const homeEl = $("#sim-gs-home-select");
    awayEl.value = "FLA";  // matches user's validation case
    homeEl.value = "EDM";
    updateGameCard("away");
    updateGameCard("home");
    awayEl.addEventListener("change", () => updateGameCard("away"));
    homeEl.addEventListener("change", () => updateGameCard("home"));

    const nSimsEl = $("#sim-gs-nsims");
    nSimsEl.addEventListener("input", (e) => {
      state.gameSimulator.nSims = +e.target.value;
      $("#sim-gs-nsims-val").textContent = e.target.value;
    });
    $("#sim-gs-simulate-btn").addEventListener("click", runGameSimulation);

    // Preload team_strength so we can show team_strength next to each card.
    if (!state.teamStrength.data) {
      fetch("/api/team-strength").then((r) => r.json()).then((d) => {
        if (d.built) {
          state.teamStrength.data = d.teams;
          updateGameCard("away");
          updateGameCard("home");
        }
      });
    }
  }

  function updateGameCard(side) {
    const team = $(`#sim-gs-${side}-select`).value;
    state.gameSimulator[side] = team;
    $(`#sim-gs-${side}-logo`).src = NHL_LOGO(team);
    $(`#sim-gs-${side}-logo`).alt = team;
    const tsData = state.teamStrength.data;
    if (tsData) {
      const row = tsData.find((t) => t.team === team);
      if (row) {
        $(`#sim-gs-${side}-strength`).textContent =
          `team_strength: ${formatStrengthVal(row.team_strength)}`;
      }
    }
  }

  function runGameSimulation() {
    const away = $("#sim-gs-away-select").value;
    const home = $("#sim-gs-home-select").value;
    const n = state.gameSimulator.nSims;
    if (away === home) {
      alert("Home and away teams must differ.");
      return;
    }
    const btn = $("#sim-gs-simulate-btn");
    btn.disabled = true;
    btn.textContent = "Simulating…";
    fetch(`/api/simulate-game?home=${home}&away=${away}&n_sims=${n}`)
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          alert("Sim failed: " + d.error);
          return;
        }
        state.gameSimulator.lastResults = d;
        renderGameSimulation(d);
      })
      .catch((e) => alert("Sim error: " + e.message))
      .finally(() => {
        btn.disabled = false;
        btn.textContent = "Simulate";
      });
  }

  function renderGameSimulation(d) {
    $("#sim-gs-results").hidden = false;
    // Sample final score (large numerals like Morning Brief)
    $("#sim-gs-final-away").innerHTML =
      `<div class="sim-final-team">${esc(d.away)}</div>
       <div class="sim-final-score-num ${d.sample_result.away_score > d.sample_result.home_score ? "winner" : ""}">${d.sample_result.away_score}</div>`;
    $("#sim-gs-final-home").innerHTML =
      `<div class="sim-final-team">${esc(d.home)}</div>
       <div class="sim-final-score-num ${d.sample_result.home_score > d.sample_result.away_score ? "winner" : ""}">${d.sample_result.home_score}</div>`;
    const cap = d.sample_result.ot
      ? (d.sample_result.so ? "Sample simulated result (decided in shootout)" : "Sample simulated result (decided in OT)")
      : "Sample simulated result (regulation)";
    $("#sim-gs-final-caption").textContent = cap;

    // Win probability bars
    const hp = d.win_pct.home, ap = d.win_pct.away;
    $("#sim-gs-wp-away-label").innerHTML =
      `<span class="sim-team-label">${esc(d.away)}</span> <span class="sim-team-pct">${(ap*100).toFixed(1)}%</span>`;
    $("#sim-gs-wp-home-label").innerHTML =
      `<span class="sim-team-pct">${(hp*100).toFixed(1)}%</span> <span class="sim-team-label">${esc(d.home)}</span>`;
    $("#sim-gs-wp-away-bar").style.flex = `${ap}`;
    $("#sim-gs-wp-home-bar").style.flex = `${hp}`;

    // Expected outcomes
    $("#sim-gs-xg-away").textContent = d.expected_goals_away.toFixed(2);
    $("#sim-gs-xg-home").textContent = d.expected_goals_home.toFixed(2);
    $("#sim-gs-reg").textContent = (d.regulation_pct * 100).toFixed(1) + "%";
    $("#sim-gs-ot").textContent = (d.ot_pct * 100).toFixed(1) + "%";
    $("#sim-gs-so").textContent = (d.so_pct * 100).toFixed(1) + "%";

    // Histogram (Chart.js)
    const canvas = $("#sim-gs-histogram");
    if (state.gameSimulator.chart) {
      state.gameSimulator.chart.destroy();
      state.gameSimulator.chart = null;
    }
    const ctx = canvas.getContext("2d");
    const totalSims = d.n_sims;
    const awayPct = d.goal_distribution.away_counts.map((c) => (c / totalSims * 100));
    const homePct = d.goal_distribution.home_counts.map((c) => (c / totalSims * 100));
    state.gameSimulator.chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: d.goal_distribution.labels,
        datasets: [
          { label: `${d.away} (away)`, data: awayPct, backgroundColor: "rgba(43, 107, 240, 0.7)" },
          { label: `${d.home} (home)`, data: homePct, backgroundColor: "rgba(232, 160, 32, 0.7)" },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: "#F0F4FF", font: { family: "DM Mono" } } },
          tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` } },
        },
        scales: {
          x: { title: { display: true, text: "Goals scored", color: "#8899BB" },
               ticks: { color: "#C0CCDD" }, grid: { color: "#0E1E35" } },
          y: { title: { display: true, text: "% of simulated games", color: "#8899BB" },
               ticks: { color: "#C0CCDD" }, grid: { color: "#0E1E35" },
               beginAtZero: true },
        },
      },
    });
  }

  // ===========================================================================
  // SUB-TAB 4 — PROJECTIONS
  // ===========================================================================
  function loadProjections() {
    loaded.projections = true;
    fetchProjections(false);
    $("#sim-proj-rerun-btn")?.addEventListener("click", () => {
      const btn = $("#sim-proj-rerun-btn");
      btn.disabled = true;
      const orig = btn.textContent;
      btn.textContent = "Running…";
      btn.classList.add("loading");
      // ?force=true fires a fresh server-side Monte Carlo with a randomized
      // seed; the backend bypasses its 6h cache for this path.
      fetchProjections(true).finally(() => {
        btn.disabled = false;
        btn.classList.remove("loading");
        btn.textContent = orig;
      });
    });
  }

  function fetchProjections(force) {
    const bracketEl = $("#sim-proj-bracket");
    bracketEl.innerHTML = `<div class="leaderboard-loading">Loading projections…</div>`;
    // Cache-bust the browser too with a random query param so successive
    // force-runs aren't served from an HTTP cache.
    const url = force
      ? `/api/season-projections?force=true&_=${Date.now()}`
      : "/api/season-projections";
    return fetch(url)
      .then((r) => r.json())
      .then((d) => {
        state.projections.data = d;
        renderProjections(d);
      })
      .catch((e) => {
        bracketEl.innerHTML = `<div class="leaderboard-blocked">Projections load failed: ${esc(e.message || e)}</div>`;
      });
  }

  function renderProjections(d) {
    const bracketEl = $("#sim-proj-bracket");
    const ts = new Date().toLocaleString("en-US", {
      timeZone: "America/New_York", hour12: false,
      year: "numeric", month: "short", day: "numeric",
      hour: "2-digit", minute: "2-digit",
    });
    $("#sim-proj-last-updated").textContent = `· Last updated ${ts}`;

    // --- Bracket: simple flow of the four remaining series. ---
    const pn = d.playoffs_now;
    if (!pn?.built || !pn.teams?.length) {
      bracketEl.innerHTML = `<div class="leaderboard-blocked">No live playoff projection. Run <code>python3 model/run_season_simulation.py</code>.</div>`;
    } else {
      const alive = pn.teams;
      // Find each team's CF & final probabilities for the bracket boxes.
      const probMap = Object.fromEntries(alive.map((t) => [t.team, t]));
      // Map alive teams to East / West (alignment with simulate.py DIVISIONS)
      const EAST = new Set(["BOS","BUF","DET","FLA","MTL","OTT","TBL","TOR","CAR","CBJ","NJD","NYI","NYR","PHI","PIT","WSH"]);
      const east = alive.filter((t) => EAST.has(t.team));
      const west = alive.filter((t) => !EAST.has(t.team));
      const teamBox = (t, withLogo = true) => {
        if (!t) return `<div class="sim-bracket-box empty">—</div>`;
        return `<div class="sim-bracket-box">
          ${withLogo ? `<img class="sim-bracket-logo" src="${NHL_LOGO(t.team)}" alt="${esc(t.team)}"/>` : ""}
          <div class="sim-bracket-team">${esc(t.team)}</div>
          <div class="sim-bracket-pct">Final ${(t.final_pct * 100).toFixed(1)}%</div>
        </div>`;
      };
      const cup = alive.slice().sort((a,b) => b.cup_pct - a.cup_pct);
      bracketEl.innerHTML = `
        <div class="sim-bracket-flow">
          <div class="sim-bracket-col">
            <div class="sim-bracket-header">East CF</div>
            ${east.map((t) => teamBox(t)).join("")}
          </div>
          <div class="sim-bracket-col sim-bracket-conf-winner">
            <div class="sim-bracket-header">East Cup Finalist</div>
            <div class="sim-bracket-box top-final">
              ${east.length ? `<div class="sim-bracket-team">Most likely:</div>
                <div class="sim-bracket-pct">${esc(east.sort((a,b)=>b.final_pct-a.final_pct)[0].team)} (${(east[0].final_pct*100).toFixed(1)}%)</div>` : ""}
            </div>
          </div>
          <div class="sim-bracket-col sim-bracket-cup-col">
            <div class="sim-bracket-header gold-text">Stanley Cup</div>
            <div class="sim-bracket-box stanley-cup">
              <div class="sim-bracket-cup-icon">🏆</div>
              <div class="sim-bracket-team">${esc(cup[0]?.team || "?")}</div>
              <div class="sim-bracket-pct gold">${(cup[0]?.cup_pct * 100).toFixed(1)}% favorite</div>
            </div>
          </div>
          <div class="sim-bracket-col sim-bracket-conf-winner">
            <div class="sim-bracket-header">West Cup Finalist</div>
            <div class="sim-bracket-box top-final">
              ${west.length ? `<div class="sim-bracket-team">Most likely:</div>
                <div class="sim-bracket-pct">${esc(west.sort((a,b)=>b.final_pct-a.final_pct)[0].team)} (${(west[0].final_pct*100).toFixed(1)}%)</div>` : ""}
            </div>
          </div>
          <div class="sim-bracket-col">
            <div class="sim-bracket-header">West CF</div>
            ${west.map((t) => teamBox(t)).join("")}
          </div>
        </div>`;
    }
  }

  // ===========================================================================
  // SEASON SIMULATOR (Phase 5.1) — sits inside Lineup Editor sub-tab
  // ===========================================================================
  function initSeasonSimulator() {
    const nSimsEl = $("#sim-season-nsims");
    const nSimsVal = $("#sim-season-nsims-val");
    if (nSimsEl) {
      nSimsEl.addEventListener("input", (e) => {
        nSimsVal.textContent = e.target.value;
      });
    }
    $("#sim-season-run-btn")?.addEventListener("click", runSeasonSimulation);
  }

  function buildLineupPayload() {
    // Convert the customLineup forwards/defense/goalies arrays into the
    // payload shape the endpoint expects (player_id + position + names).
    const cl = state.lineupEditor.customLineup;
    if (!cl) return null;
    const mapSlot = (p, defaultKind) => {
      if (!p || !p.player_id) return null;
      const name = p.name || `${p.first_name || ""} ${p.last_name || ""}`.trim();
      const parts = name.split(/\s+/);
      return {
        player_id: p.player_id,
        position: p.position || defaultKind,
        first_name: p.first_name || parts.slice(0, -1).join(" "),
        last_name:  p.last_name  || parts.slice(-1).join(""),
      };
    };
    return {
      forwards: cl.forwards.map((p) => mapSlot(p, "C")),
      defense:  cl.defense.map((p)  => mapSlot(p, "D")),
      goalies:  (cl.goalies || []).slice(0, 2).map((g) => mapSlot(g, "G")),
    };
  }

  function runSeasonSimulation() {
    const team = state.lineupEditor.team;
    if (!team) {
      alert("Pick a team first.");
      return;
    }
    const lineup = buildLineupPayload();
    if (!lineup) { alert("Lineup not ready."); return; }
    const nSims = +$("#sim-season-nsims").value;
    const compare = $("#sim-season-compare").checked;

    const btn = $("#sim-season-run-btn");
    btn.disabled = true;
    const origLabel = btn.textContent;
    btn.textContent = "Simulating…";

    $("#sim-season-loading-n").textContent = nSims;
    $("#sim-season-loading").hidden = false;
    $("#sim-season-results").hidden = true;

    fetch("/api/simulate-season-custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        team, lineup, n_sims: nSims, compare_to_default: compare,
      }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          alert("Sim failed: " + d.error);
          return;
        }
        renderSeasonResults(d);
      })
      .catch((e) => alert("Sim error: " + e.message))
      .finally(() => {
        $("#sim-season-loading").hidden = true;
        btn.disabled = false;
        btn.textContent = origLabel;
      });
  }

  function renderSeasonResults(d) {
    const cust = d.custom;
    const def = d.default;
    const rs = cust.regular_season;
    $("#sim-season-rs-record").textContent =
      `${rs.avg_wins.toFixed(1)} W · ${rs.avg_losses.toFixed(1)} L · ${rs.avg_otl.toFixed(1)} OTL`;
    $("#sim-season-rs-points").textContent = rs.avg_points.toFixed(1);
    $("#sim-season-rs-gf-ga").textContent =
      `${Math.round(rs.avg_gf)} / ${Math.round(rs.avg_ga)}`;
    $("#sim-season-rs-playoff").textContent =
      (rs.playoff_pct * 100).toFixed(1) + "%";

    // Delta indicators (only if compare ran)
    const deltaEl = $("#sim-season-rs-points-delta");
    const deltaP  = $("#sim-season-rs-playoff-delta");
    if (def && d.delta_regular_season) {
      const dpts = d.delta_regular_season.points;
      const dpct = d.delta_regular_season.playoff_pct;
      deltaEl.textContent = `${dpts >= 0 ? "+" : ""}${dpts} vs default (${def.regular_season.avg_points.toFixed(1)})`;
      deltaEl.className = "sim-season-stat-delta " + (dpts > 0.5 ? "green" : dpts < -0.5 ? "red" : "muted");
      deltaP.textContent = `${dpct >= 0 ? "+" : ""}${(dpct * 100).toFixed(1)}pp vs default`;
      deltaP.className = "sim-season-stat-delta " + (dpct > 0.01 ? "green" : dpct < -0.01 ? "red" : "muted");
    } else {
      deltaEl.textContent = "";
      deltaP.textContent = "";
    }

    // Division finish bars
    const divEl = $("#sim-season-division");
    const dist = rs.division_finish || {};
    const order = Object.keys(dist).sort((a, b) => +a.replace("finish_","") - +b.replace("finish_",""));
    const maxPct = Math.max(0.0001, ...order.map((k) => dist[k]));
    divEl.innerHTML = order.map((k) => {
      const rank = k.replace("finish_", "");
      const pct = dist[k];
      const width = ((pct / maxPct) * 100).toFixed(1);
      return `
        <div class="sim-season-div-row">
          <div class="sim-season-div-label">${rank}${ordSuffix(+rank)} place</div>
          <div class="sim-season-div-bar-wrap">
            <div class="sim-season-div-bar" style="width: ${width}%"></div>
          </div>
          <div class="sim-season-div-pct mono">${(pct * 100).toFixed(1)}%</div>
        </div>`;
    }).join("");

    // Player stats table — highlight swapped-in players (those whose player_id
    // is NOT in the default lineup, when compare is on).
    const defPlayerIds = def
      ? new Set(def.player_stats.map((p) => p.player_id))
      : new Set();
    const defStatsById = def
      ? Object.fromEntries(def.player_stats.map((p) => [p.player_id, p]))
      : {};
    const tbody = $("#sim-season-player-tbody");
    tbody.innerHTML = cust.player_stats
      .filter((p) => p.player_id !== null)   // skip empty slots
      .map((p, i) => {
        const lineTag = p.slot_kind === "F"
          ? `Line ${p.line_index + 1}`
          : (p.slot_kind === "D" ? `Pair ${p.line_index + 1}` : "—");
        const a = p.avg_a1 + p.avg_a2;
        const swapped = def && !defPlayerIds.has(p.player_id);
        let deltaCell = "";
        if (def && defStatsById[p.player_id]) {
          const dPts = p.avg_points - defStatsById[p.player_id].avg_points;
          if (Math.abs(dPts) >= 0.5) {
            deltaCell = ` <span class="sim-season-stat-delta-inline ${dPts > 0 ? "green" : "red"}">${dPts > 0 ? "+" : ""}${dPts.toFixed(1)}</span>`;
          }
        }
        return `
          <tr class="${swapped ? "sim-season-row-swapped" : ""}">
            <td class="rank-col">${i + 1}</td>
            <td>${esc(p.name)}${swapped ? " <span class=\"swap-pill\">swap</span>" : ""}</td>
            <td>${esc(p.position)}</td>
            <td>${lineTag}</td>
            <td class="num-cell">${p.expected_gp}</td>
            <td class="num-cell">${p.avg_goals.toFixed(1)}</td>
            <td class="num-cell">${a.toFixed(1)}</td>
            <td class="num-cell gold">${p.avg_points.toFixed(1)}${deltaCell}</td>
            <td class="num-cell">${p.expected_toi}</td>
          </tr>`;
      }).join("");

    // Playoff outcome bars
    const po = cust.playoff_outcome;
    const order2 = [
      ["miss",       "Missed Playoffs", "muted"],
      ["r1_out",     "Eliminated in R1", ""],
      ["r2_out",     "Eliminated in R2", ""],
      ["cf_out",     "Lost Conference Final", ""],
      ["final_out",  "Lost Stanley Cup Final", ""],
      ["champion",   "Won the Cup", "gold"],
    ];
    const maxPo = Math.max(0.0001, ...order2.map(([k]) => po[k] || 0));
    $("#sim-season-playoff-bars").innerHTML = order2.map(([k, label, cls]) => {
      const pct = po[k] || 0;
      const width = ((pct / maxPo) * 100).toFixed(1);
      return `
        <div class="sim-season-po-row">
          <div class="sim-season-po-label ${cls}">${label}</div>
          <div class="sim-season-po-bar-wrap">
            <div class="sim-season-po-bar ${cls === "gold" ? "po-gold" : ""}" style="width: ${width}%"></div>
          </div>
          <div class="sim-season-po-pct mono">${(pct * 100).toFixed(1)}%</div>
        </div>`;
    }).join("");

    $("#sim-season-results").hidden = false;
  }

  function ordSuffix(n) {
    const v = n % 100;
    if (v >= 11 && v <= 13) return "th";
    switch (n % 10) {
      case 1: return "st";
      case 2: return "nd";
      case 3: return "rd";
      default: return "th";
    }
  }

  // ---------------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------------
  function init() {
    initSubTabs();
    // If the user lands directly on the Simulator subpage (e.g. via URL state),
    // trigger the active-tab loader.
    const active = $(".sim-tab.active");
    if (active) lazyLoad(active.dataset.stab);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
