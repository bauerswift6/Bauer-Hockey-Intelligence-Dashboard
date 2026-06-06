/* =========================================================
   Glossary — render cards from GLOSSARY_DATA, handle search,
   filter buttons, related-stat anchor jumps, TOC, copy.
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const data = window.GLOSSARY_DATA;
  if (!data) {
    console.error("GLOSSARY_DATA not loaded");
    return;
  }

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  function statById(id) {
    return data.stats.find((s) => s.id === id);
  }

  // ---------------------------------------------------------------------------
  // Render: benchmark bar (5 segments, red→green) + tier table
  // ---------------------------------------------------------------------------
  function renderBenchmarkBar(tiers) {
    // tiers come ordered Replacement→Elite (5 entries); build colored segments
    const segs = tiers.map((t, i) => {
      const cls = ["bench-replacement", "bench-below", "bench-average", "bench-good", "bench-elite"][i];
      const label = `${t.tier}: ${t.range}`;
      return `<div class="bench-segment ${cls}" title="${escHtml(label)}">
        <span class="bench-tier-label">${escHtml(t.tier)}</span>
      </div>`;
    }).join("");
    return `<div class="benchmark-bar">${segs}</div>`;
  }

  function renderBenchmarkTable(tiers) {
    // Display in Elite → Replacement order (top-down value)
    const rev = tiers.slice().reverse();
    const rows = rev.map((t) =>
      `<tr><td>${escHtml(t.tier)}</td><td>${escHtml(t.range)}</td></tr>`
    ).join("");
    return `<table class="benchmarks-table"><tbody>${rows}</tbody></table>`;
  }

  // ---------------------------------------------------------------------------
  // Render a single stat card
  // ---------------------------------------------------------------------------
  function renderCard(stat) {
    const related = (stat.related || [])
      .map((id) => {
        const s = statById(id);
        if (!s) return "";
        // Use abbreviation or short version of name for the tag
        const tag = s.name.split("—")[0].trim();
        return `<a class="related-tag" href="#stat-${escHtml(s.id)}" data-jump="stat-${escHtml(s.id)}">${escHtml(tag)}</a>`;
      })
      .filter(Boolean)
      .join("");

    const trackingNote = stat.section === "tracking"
      ? `<p class="stat-tracking-note">Tracking data availability varies by source and season.</p>`
      : "";

    return `
      <article class="stat-card" id="stat-${escHtml(stat.id)}" data-stat-id="${escHtml(stat.id)}" data-section="${escHtml(stat.section)}" data-keywords="${escHtml(stat.keywords || "")}">
        <header class="stat-card-head">
          <h4 class="stat-name">${escHtml(stat.name)}</h4>
          <button class="copy-btn" type="button" aria-label="Copy definition">Copy definition</button>
        </header>
        <p class="stat-def">${escHtml(stat.def)}</p>
        <p class="stat-explain">${escHtml(stat.explain)}</p>
        <div class="benchmarks">
          <div class="benchmarks-label">By the numbers</div>
          ${renderBenchmarkBar(stat.tiers)}
          ${renderBenchmarkTable(stat.tiers)}
        </div>
        <div class="stat-meta-row stat-why">
          <span class="stat-meta-label">Why front offices use this</span>
          <span class="stat-meta-text">${escHtml(stat.why)}</span>
        </div>
        <div class="stat-meta-row stat-limit">
          <span class="stat-meta-label">Limitation</span>
          <span class="stat-meta-text">${escHtml(stat.limit)}</span>
        </div>
        ${trackingNote}
        <div class="related-stats">
          <span class="related-label">Related</span>
          ${related}
        </div>
      </article>
    `;
  }

  // ---------------------------------------------------------------------------
  // Render all sections
  // ---------------------------------------------------------------------------
  function renderSections() {
    data.sections.forEach((sec) => {
      const container = document.getElementById(`section-${sec.id}`);
      if (!container) return;
      container.querySelector(".glossary-section-title").textContent = sec.title;
      container.querySelector(".glossary-section-intro").textContent = sec.intro;
      const cards = container.querySelector(".glossary-cards");
      const sectionStats = data.stats.filter((s) => s.section === sec.id);
      cards.innerHTML = sectionStats.map(renderCard).join("");
    });
  }

  // ---------------------------------------------------------------------------
  // TOC — Glossary at a Glance
  // ---------------------------------------------------------------------------
  function renderTOC() {
    const grid = document.getElementById("glossary-toc-grid");
    if (!grid) return;
    const html = data.sections.map((sec) => {
      const stats = data.stats.filter((s) => s.section === sec.id);
      const items = stats.map((s) => {
        const tag = s.name.split("—")[0].trim();
        return `<li><a class="toc-link" href="#stat-${escHtml(s.id)}" data-jump="stat-${escHtml(s.id)}">${escHtml(tag)}</a></li>`;
      }).join("");
      return `
        <div class="glossary-toc-col" data-section="${escHtml(sec.id)}">
          <div class="glossary-toc-col-title">${escHtml(sec.label)}</div>
          <ul>${items}</ul>
        </div>
      `;
    }).join("");
    grid.innerHTML = html;
  }

  // ---------------------------------------------------------------------------
  // Search + Filter
  // ---------------------------------------------------------------------------
  function normalize(s) {
    return (s || "").toLowerCase().replace(/\s+/g, " ").trim();
  }

  let activeFilter = "all";
  let searchQuery = "";

  function applyFilters() {
    const q = normalize(searchQuery);
    const cards = $$(".stat-card");
    let visible = 0;

    cards.forEach((card) => {
      const matchesFilter = activeFilter === "all" || card.dataset.section === activeFilter;
      const text = normalize([
        card.querySelector(".stat-name")?.textContent,
        card.querySelector(".stat-def")?.textContent,
        card.querySelector(".stat-explain")?.textContent,
        card.dataset.keywords,
      ].join(" "));
      const matchesSearch = q === "" || text.includes(q);
      const show = matchesFilter && matchesSearch;
      card.hidden = !show;
      if (show) visible++;
    });

    // Hide sections without visible cards
    $$(".glossary-section").forEach((section) => {
      const has = $$(".stat-card", section).some((c) => !c.hidden);
      section.hidden = !has;
    });

    // TOC also respects filter (but not search)
    $$(".glossary-toc-col").forEach((col) => {
      col.hidden = activeFilter !== "all" && col.dataset.section !== activeFilter;
    });

    const noResults = $("#glossary-no-results");
    if (noResults) noResults.hidden = visible !== 0;

    const total = cards.length;
    const counter = $("#glossary-result-count");
    if (counter) {
      counter.textContent = (q === "" && activeFilter === "all")
        ? `${total} stats`
        : `${visible} of ${total} stats`;
    }
  }

  function setupSearch() {
    const input = $("#glossary-search");
    if (input) {
      input.addEventListener("input", (e) => {
        searchQuery = e.target.value;
        applyFilters();
      });
    }
  }

  function setupFilterButtons() {
    $$(".glossary-filter-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$(".glossary-filter-btn").forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        activeFilter = btn.dataset.filter;
        applyFilters();
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Anchor-jump for related-stat tags & TOC links
  // (Smooth-scroll, brief highlight pulse on target card.)
  // ---------------------------------------------------------------------------
  function setupAnchorJumps() {
    document.addEventListener("click", (e) => {
      const link = e.target.closest("[data-jump]");
      if (!link) return;
      e.preventDefault();
      const targetId = link.dataset.jump;
      const target = document.getElementById(targetId);
      if (!target) return;

      // If filter is hiding it, reset to "all"
      if (target.hidden || target.closest(".glossary-section")?.hidden) {
        $$(".glossary-filter-btn").forEach((b) => {
          b.classList.toggle("active", b.dataset.filter === "all");
          b.setAttribute("aria-selected", b.dataset.filter === "all" ? "true" : "false");
        });
        activeFilter = "all";
        // Also clear search if it's hiding the target
        searchQuery = "";
        const input = $("#glossary-search");
        if (input) input.value = "";
        applyFilters();
      }

      target.scrollIntoView({ behavior: "smooth", block: "start" });
      target.classList.add("stat-card-highlight");
      setTimeout(() => target.classList.remove("stat-card-highlight"), 1800);
    });
  }

  // ---------------------------------------------------------------------------
  // Copy definition
  // ---------------------------------------------------------------------------
  function setupCopyHandlers() {
    document.addEventListener("click", async (e) => {
      const btn = e.target.closest(".copy-btn");
      if (!btn) return;
      const card = btn.closest(".stat-card");
      if (!card) return;
      const name = card.querySelector(".stat-name")?.textContent?.trim() || "";
      const def = card.querySelector(".stat-def")?.textContent?.trim() || "";
      const text = `${name}: ${def}`;

      try {
        await navigator.clipboard.writeText(text);
      } catch (err) {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (_) {}
        document.body.removeChild(ta);
      }

      const original = btn.textContent;
      btn.textContent = "Copied!";
      btn.classList.add("copied");
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove("copied");
      }, 1400);
    });
  }

  // ---------------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------------
  function init() {
    renderSections();
    renderTOC();
    setupSearch();
    setupFilterButtons();
    setupAnchorJumps();
    setupCopyHandlers();
    applyFilters();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
