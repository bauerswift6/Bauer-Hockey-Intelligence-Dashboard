/* =========================================================
   Shared utilities — sortable tables, search filtering,
   escapeHtml, leaderboard table builder.
   ========================================================= */

window.dashUtil = (function () {

  function escHtml(s) {
    return String(s ?? "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
    );
  }

  /**
   * Build a sortable, searchable leaderboard table.
   *
   * @param {object} opts
   *   container: DOM element to render into
   *   id:        unique key (used to scope IDs)
   *   title:     section title (string)
   *   subtitle:  optional one-liner description (Sora)
   *   ehBanner:  legacy no-op opt-in retained for backwards compatibility (deprecated)
   *   sourceLabel: small source line (e.g. "Source: my own models")
   *   columns:   array of { key, label, title?, fmt?, sortKey?, defaultSort?, align? }
   *              defaultSort: "desc" | "asc" — initial sort if column is the active one
   *   activeSort: { key, dir } — initial sort
   *   getRows:   function(state) -> array of plain row objects
   *   state:     optional state passed to getRows
   *   showSearch: if true, render a search box
   *   searchKeys: array of keys to match against the search text
   *   emptyMsg:   message when no rows
   */
  function buildLeaderboard(opts) {
    const {
      container, id, title, subtitle, ehBanner = false,
      sourceLabel = "", columns, activeSort,
      getRows, showSearch = true, searchKeys = [],
      emptyMsg = "No rows match the current filters.",
    } = opts;

    const sortState = { key: activeSort?.key ?? columns[0].key, dir: activeSort?.dir ?? "desc" };
    let searchText = "";

    function render() {
      const rows = (getRows() || []).slice();
      const dirMult = sortState.dir === "asc" ? 1 : -1;
      const sortKey = columns.find((c) => c.key === sortState.key)?.sortKey || sortState.key;
      rows.sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (typeof av === "number" && typeof bv === "number") return (av - bv) * dirMult;
        return String(av).localeCompare(String(bv)) * dirMult;
      });

      let filtered = rows;
      if (searchText) {
        const q = searchText.toLowerCase();
        filtered = rows.filter((r) =>
          searchKeys.some((k) => String(r[k] ?? "").toLowerCase().includes(q))
        );
      }

      const tbody = container.querySelector(`#${id}-tbody`);
      if (!tbody) return;
      if (!filtered.length) {
        tbody.innerHTML = `<tr><td colspan="${columns.length + 1}" class="no-games-msg">${escHtml(emptyMsg)}</td></tr>`;
      } else {
        tbody.innerHTML = filtered.map((r, i) => {
          const cells = columns.map((c) => {
            const v = r[c.key];
            const display = c.fmt ? c.fmt(v, r) : (v ?? "—");
            const cls = c.align === "right" ? "num-cell" : "";
            return `<td class="${cls}">${display}</td>`;
          }).join("");
          return `<tr><td class="rank-col">${i + 1}</td>${cells}</tr>`;
        }).join("");
      }

      // Update sort indicators on column headers
      const ths = container.querySelectorAll("th.sortable");
      ths.forEach((th) => {
        th.classList.remove("active-sort", "asc");
        if (th.dataset.col === sortState.key) {
          th.classList.add("active-sort");
          if (sortState.dir === "asc") th.classList.add("asc");
        }
      });
    }

    function setupSortClicks() {
      container.querySelectorAll("th.sortable").forEach((th) => {
        th.addEventListener("click", () => {
          const key = th.dataset.col;
          if (sortState.key === key) {
            sortState.dir = sortState.dir === "desc" ? "asc" : "desc";
          } else {
            sortState.key = key;
            sortState.dir = "desc";
          }
          render();
        });
      });
    }

    // Initial markup
    const searchHtml = showSearch
      ? `<input class="leaderboard-search" id="${id}-search" type="search" placeholder="Search…" autocomplete="off" />`
      : "";
    const sourceHtml = sourceLabel
      ? `<p class="source-label">${escHtml(sourceLabel)}</p>`
      : "";
    // Legacy `ehBanner` opt-in is now a no-op — every advanced-stat column
    // on the dashboard comes from my own models (composite_war / self-gen
    // xGAR / GSAX / xGF% / Game Score), so the prior third-party-attribution
    // banner no longer applies. Kept as an empty string so existing callers
    // that still pass ehBanner:true don't break.
    const ehHtml = "";
    const subtitleHtml = subtitle
      ? `<p class="leaderboard-subtitle">${escHtml(subtitle)}</p>`
      : "";

    const headerCols = columns.map((c) => {
      const titleAttr = c.title ? `title="${escHtml(c.title)}"` : "";
      const sortCls = "sortable" + (c.key === sortState.key ? " active-sort" + (sortState.dir === "asc" ? " asc" : "") : "");
      const align = c.align === "right" ? " num-th" : "";
      return `<th class="${sortCls}${align}" data-col="${escHtml(c.key)}" ${titleAttr}>${escHtml(c.label)}</th>`;
    }).join("");

    container.innerHTML = `
      <section class="section leaderboard-section">
        <h2 class="section-title">${escHtml(title)}</h2>
        ${ehHtml}
        ${subtitleHtml}
        <div class="section-body">
          <div class="leaderboard-toolbar">
            ${searchHtml}
            ${sourceHtml}
          </div>
          <div class="table-wrap">
            <table class="data-table leaderboard-table">
              <thead>
                <tr>
                  <th class="rank-col">#</th>
                  ${headerCols}
                </tr>
              </thead>
              <tbody id="${id}-tbody"></tbody>
            </table>
          </div>
        </div>
      </section>
    `;

    setupSortClicks();
    if (showSearch) {
      container.querySelector(`#${id}-search`).addEventListener("input", (e) => {
        searchText = e.target.value;
        render();
      });
    }
    render();

    // Return a handle so caller can re-render on state change (e.g., position filter)
    return { rerender: render };
  }

  function formatPct(v, dp = 1) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(dp) + "%";
  }

  function formatNum(v, dp = 2) {
    if (v == null || isNaN(v)) return "—";
    return Number(v).toFixed(dp);
  }

  function classifyPosition(pos) {
    if (!pos) return "F";
    if (pos === "G") return "G";
    if (pos === "D") return "D";
    return "F";
  }

  return {
    escHtml,
    buildLeaderboard,
    formatPct,
    formatNum,
    classifyPosition,
  };
})();
