/* =========================================================
   NHL → Players page: position filter at top of section.
   (Stat Leaders / GAR / xGAR / Top Lines tabs were retired
   in the Players restructure. The Overview and Leaderboards
   tabs each watch the position filter on their own.)
   ========================================================= */

(function () {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function initPositionFilter() {
    const bar = $("#player-pos-filter");
    if (!bar) return;
    bar.addEventListener("click", (e) => {
      const btn = e.target.closest(".pos-btn");
      if (!btn) return;
      $$(".pos-btn", bar).forEach((b) => {
        const on = b === btn;
        b.classList.toggle("active", on);
        b.setAttribute("aria-selected", on ? "true" : "false");
      });
      // Other modules (players_overview.js, players_leaderboards.js,
      // and contract_value.js seg-controls) read this state via the
      // .pos-btn.active selector on their own click handlers.
    });
  }

  function init() {
    initPositionFilter();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
