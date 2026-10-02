/* =========================================================
   Morning Brief — 3-zone layout controller
   - Zone 1: games (handled by dashboard.js fetchGames)
   - Zone 2: playoff form table + news feed
   - Zone 3: playoff bracket + next games
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const NEWS_REFRESH_MS = 10 * 60 * 1000; // 10 min

  const state = {
    newsTimer: null,
  };

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  // ---------------------------------------------------------------------------
  // Page header — date + last-updated
  // ---------------------------------------------------------------------------
  function renderDateHeader() {
    const dateEl = $("#mb-page-date");
    if (dateEl) {
      const today = new Date();
      dateEl.textContent = today.toLocaleDateString("en-US", {
        weekday: "long",
        month: "long",
        day: "numeric",
        year: "numeric",
      });
    }
  }

  function setLastUpdated() {
    const el = $("#mb-last-updated");
    if (!el) return;
    const now = new Date();
    el.textContent = now.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      hour12: true,
    });
  }

  // ---------------------------------------------------------------------------
  // Zone 2 — News feed (League Pulse / Ink Report / Trade Block)
  // ---------------------------------------------------------------------------
  function timeAgo(iso) {
    if (!iso) return "";
    const t = new Date(iso).getTime();
    if (isNaN(t)) return "";
    const diff = Math.max(0, Date.now() - t);
    const mins = Math.round(diff / 60000);
    if (mins < 1) return "just now";
    if (mins < 60) return `${mins}m ago`;
    const hours = Math.round(mins / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    return `${days}d ago`;
  }

  // League Pulse — curated big-news headlines (full width)
  function renderPulse(items) {
    const list = $("#pulse-list");
    const empty = $("#pulse-empty");
    if (!list) return;

    if (!items || items.length === 0) {
      list.hidden = true;
      if (empty) empty.hidden = false;
      return;
    }

    if (empty) empty.hidden = true;
    list.innerHTML = items.map((it) => `
      <li class="news-item">
        <a class="news-link" href="${escHtml(it.link)}" target="_blank" rel="noopener">${escHtml(it.title)}</a>
        <div class="news-meta">
          <span class="news-source-tag">${escHtml(it.source || 'NHL')}</span>
          <span class="news-ago">${escHtml(timeAgo(it.pub_date))}</span>
        </div>
      </li>
    `).join("");
    list.hidden = false;
  }

  // Ink Report / Trade Block — type-pure move lists (no per-item badge needed)
  function renderMoveList(listSel, emptySel, items) {
    const list = $(listSel);
    const empty = $(emptySel);
    if (!list) return;

    if (!items || items.length === 0) {
      list.hidden = true;
      if (empty) empty.hidden = false;
      return;
    }

    if (empty) empty.hidden = true;
    list.innerHTML = items.map((it) => `
      <li class="txn-item">
        <a class="txn-title" href="${escHtml(it.link)}" target="_blank" rel="noopener">${escHtml(it.title)}</a>
        <span class="txn-ago">${escHtml(timeAgo(it.pub_date))}</span>
      </li>
    `).join("");
    list.hidden = false;
  }

  async function loadFeed() {
    const spinners = ["#pulse-spinner", "#ink-spinner", "#trade-spinner"].map((s) => $(s));
    const lists = ["#pulse-list", "#ink-list", "#trade-list"].map((s) => $(s));
    const empties = ["#pulse-empty", "#ink-empty", "#trade-empty"].map((s) => $(s));
    const errors = ["#pulse-error", "#ink-error", "#trade-error"].map((s) => $(s));
    const srcLabel = $("#pulse-source");

    [...lists, ...empties, ...errors, srcLabel].forEach((el) => el && (el.hidden = true));
    spinners.forEach((el) => el && (el.hidden = false));

    try {
      const r = await fetch("/api/morning-brief-feed");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      spinners.forEach((el) => el && (el.hidden = true));

      if (data.blocked) {
        errors.forEach((el) => el && (el.hidden = false));
        return;
      }

      renderPulse(data.league_pulse || []);
      renderMoveList("#ink-list", "#ink-empty", data.ink_report || []);
      renderMoveList("#trade-list", "#trade-empty", data.trade_block || []);

      if (srcLabel && data.news_source) {
        srcLabel.textContent = `Source: ${data.news_source}`;
        srcLabel.hidden = false;
      }
    } catch (e) {
      spinners.forEach((el) => el && (el.hidden = true));
      errors.forEach((el) => el && (el.hidden = false));
      console.error("morning-brief-feed fetch failed:", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Zone 3 — Playoff Bracket
  // ---------------------------------------------------------------------------
  function renderSeriesCard(s) {
    if (!s) {
      return `<div class="series-card series-card-tbd"><span>TBD</span></div>`;
    }
    const top = s.top_seed || {};
    const bot = s.bottom_seed || {};
    let statusText = "Not started";
    let statusClass = "status-pending";
    if (s.complete) {
      statusText = "Complete";
      statusClass = "status-complete";
    } else if (s.started) {
      statusText = "In progress";
      statusClass = "status-active";
    }
    const tbd = (seed) => !seed || !seed.abbrev;

    function seedRow(seed) {
      if (tbd(seed)) {
        return `<div class="series-team series-team-tbd"><span class="series-team-abbrev">TBD</span></div>`;
      }
      const wonCls = seed.won ? "series-team-won" : "";
      return `
        <div class="series-team ${wonCls}">
          <img class="series-team-logo" src="${escHtml(seed.logo || '')}" alt="" loading="lazy" />
          <span class="series-team-abbrev">${escHtml(seed.abbrev)}</span>
          <span class="series-team-wins">${seed.wins ?? 0}</span>
        </div>
      `;
    }

    return `
      <div class="series-card ${statusClass}">
        ${seedRow(top)}
        ${seedRow(bot)}
        <div class="series-status">${escHtml(statusText)}</div>
      </div>
    `;
  }

  function groupBracketSeries(rounds) {
    const byRound = {};
    rounds.forEach((r) => { byRound[r.round_number] = r.series; });

    // Split each round into Eastern, Western, Final
    function split(rNum) {
      const all = byRound[rNum] || [];
      return {
        east: all.filter((s) => s.conference === "Eastern"),
        west: all.filter((s) => s.conference === "Western"),
        final: all.filter((s) => s.conference === "Final" || s.round === 4),
      };
    }
    return {
      r1: split(1),
      r2: split(2),
      r3: split(3),
      r4: split(4),
    };
  }

  function renderBracket(data) {
    const content = $("#bracket-content");
    if (!content) return;
    const grouped = groupBracketSeries(data.rounds || []);

    // Pad with nulls so columns have predictable counts (East R1=4, R2=2, R3=1; West mirrored)
    const padTo = (arr, n) => {
      const out = arr.slice(0, n);
      while (out.length < n) out.push(null);
      return out;
    };

    const eastR1 = padTo(grouped.r1.east, 4);
    const eastR2 = padTo(grouped.r2.east, 2);
    const eastR3 = padTo(grouped.r3.east, 1);
    const westR1 = padTo(grouped.r1.west, 4);
    const westR2 = padTo(grouped.r2.west, 2);
    const westR3 = padTo(grouped.r3.west, 1);
    const finals = padTo(grouped.r4.final, 1);

    const col = (label, series) =>
      `<div class="bracket-col">
         <div class="bracket-col-label">${escHtml(label)}</div>
         <div class="bracket-col-series">${series.map(renderSeriesCard).join("")}</div>
       </div>`;

    content.innerHTML = `
      <div class="bracket-grid">
        ${col("East R1", eastR1)}
        ${col("East R2", eastR2)}
        ${col("East Final", eastR3)}
        ${col("Stanley Cup", finals)}
        ${col("West Final", westR3)}
        ${col("West R2", westR2)}
        ${col("West R1", westR1)}
      </div>
    `;
    content.hidden = false;
  }

  function renderNextGames(next) {
    const row = $("#next-games-content");
    if (!row) return;
    if (!next || next.length === 0) {
      row.innerHTML = `<span class="next-games-placeholder">No upcoming playoff games scheduled.</span>`;
      return;
    }
    row.innerHTML = next.map((g) => {
      let timeText = "";
      if (g.start_time_utc) {
        const d = new Date(g.start_time_utc);
        const dateStr = d.toLocaleDateString("en-US", { weekday: "short", month: "short", day: "numeric" });
        const timeStr = d.toLocaleTimeString("en-US", {
          hour: "numeric", minute: "2-digit", hour12: true,
          timeZone: "America/New_York",
        });
        timeText = `${dateStr} · ${timeStr} ET`;
      }
      return `
        <div class="next-game">
          <div class="next-game-teams">
            <img class="next-game-logo" src="${escHtml(g.away.logo || '')}" alt="" loading="lazy" />
            <span class="next-game-abbrev">${escHtml(g.away.abbrev)}</span>
            <span class="next-game-at">@</span>
            <img class="next-game-logo" src="${escHtml(g.home.logo || '')}" alt="" loading="lazy" />
            <span class="next-game-abbrev">${escHtml(g.home.abbrev)}</span>
          </div>
          <div class="next-game-time">${escHtml(timeText)}</div>
        </div>
      `;
    }).join("");
  }

  async function loadBracket() {
    const spinner = $("#bracket-spinner");
    const content = $("#bracket-content");
    const empty = $("#bracket-empty");
    const err = $("#bracket-error");
    [content, empty, err].forEach((el) => el && (el.hidden = true));
    if (spinner) spinner.hidden = false;

    try {
      const r = await fetch("/api/playoff-bracket");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      if (spinner) spinner.hidden = true;

      if (data.empty || !data.rounds || data.rounds.length === 0) {
        if (empty) empty.hidden = false;
        return;
      }

      renderBracket(data);
      renderNextGames(data.next_games || []);
    } catch (e) {
      if (spinner) spinner.hidden = true;
      if (err) err.hidden = false;
      console.error("bracket fetch failed:", e);
    }
  }

  // ---------------------------------------------------------------------------
  // Init + refresh loop
  // ---------------------------------------------------------------------------
  async function loadAll() {
    renderDateHeader();
    await Promise.all([loadFeed(), loadBracket()]);
    setLastUpdated();
  }

  function init() {
    loadAll();
    // Auto-refresh the news feed every 10 minutes
    state.newsTimer = setInterval(async () => {
      await loadFeed();
      setLastUpdated();
    }, NEWS_REFRESH_MS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
