/* =========================================================
   NHL → Goalies page: extended goalie leaderboard
   (SV% / HDSV% / MDSV% / workload / QS% / GSAX-60).
   ========================================================= */

(function () {
  const util = window.dashUtil;
  const state = { loaded: false };

  const BLURB =
    "Comprehensive goalie evaluation — overall SV%, save percentage broken out by danger zone, workload (games and ice time), GAA, Quality Start % (QS = SV% ≥ .917 OR ≤2 goals allowed), and GSAX rate-adjusted per 60 minutes. Goalies need ≥10 games played to qualify. Click any column to sort.";

  function colorSv(v, threshold) {
    if (v == null) return "—";
    const n = Number(v);
    const cls = n >= threshold + 1 ? "metric-elite" : n >= threshold - 0.5 ? "metric-mid" : "metric-poor";
    return `<span class="metric-pill ${cls}">${n.toFixed(2)}</span>`;
  }

  function colorGsax60(v) {
    if (v == null) return "—";
    const n = Number(v);
    const cls = n >= 0.30 ? "metric-elite" : n >= 0.05 ? "metric-mid" : "metric-poor";
    const sign = n >= 0 ? "+" : "";
    return `<span class="metric-pill ${cls}">${sign}${n.toFixed(3)}</span>`;
  }

  function colorQsPct(v) {
    if (v == null) return "—";
    const n = Number(v);
    const cls = n >= 55 ? "metric-elite" : n >= 50 ? "metric-mid" : "metric-poor";
    return `<span class="metric-pill ${cls}">${n.toFixed(1)}%</span>`;
  }

  async function build() {
    const container = document.getElementById("goalies-extended-container");
    if (!container) return;
    container.innerHTML = `<section class="section leaderboard-section"><h2 class="section-title">Goalie Detail — SV% / Workload / Quality Starts</h2><div class="section-body"><div class="spinner"></div><p class="news-source">Loading per-goalie game logs (first load is slow, then cached 6h)…</p></div></section>`;

    let rows = [];
    try {
      const r = await fetch("/api/goalies-extended");
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      rows = data.goalies || [];
    } catch (e) {
      container.innerHTML = `<section class="section"><div class="section-body"><p class="error-msg">Failed to load goalies-extended.</p></div></section>`;
      return;
    }

    util.buildLeaderboard({
      container,
      id: "goalies-ext",
      title: "Goalie Detail — SV% / Workload / Quality Starts",
      subtitle: BLURB,
      ehBanner: true,
      sourceLabel: "Source: MoneyPuck + NHL API game logs · min 10 GP",
      columns: [
        { key: "name",      label: "Goalie" },
        { key: "team",      label: "Team" },
        { key: "games",     label: "GP",      title: "Games played", align: "right" },
        { key: "starts",    label: "Starts",  title: "Games started this season", align: "right" },
        { key: "toi_min",   label: "TOI",     title: "Total ice time, minutes", align: "right" },
        { key: "sv_pct",    label: "SV%",     title: "Overall save percentage",
          fmt: (v) => colorSv(v, 91), align: "right" },
        { key: "hdsv_pct",  label: "HDSV%",   title: "High-danger save % (inner-slot shots)",
          fmt: (v) => colorSv(v, 80), align: "right" },
        { key: "mdsv_pct",  label: "MDSV%",   title: "Medium-danger save %",
          fmt: (v) => colorSv(v, 90), align: "right" },
        { key: "gaa",       label: "GAA",     title: "Goals against average",
          fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "qs_pct",    label: "QS%",     title: "Quality Start % — SV% ≥ .917 OR ≤2 GA on ≥1 shot, of starts",
          fmt: colorQsPct, align: "right" },
        { key: "gsax",      label: "GSAX",    title: "Goals Saved Above Expected (cumulative)",
          fmt: (v) => util.formatNum(v, 2), align: "right" },
        { key: "gsax_60",   label: "GSAX/60", title: "GSAX rate per 60 minutes of ice time",
          fmt: colorGsax60, align: "right" },
      ],
      activeSort: { key: "gsax", dir: "desc" },
      getRows: () => rows,
      searchKeys: ["name", "team"],
    });
  }

  function init() {
    const tab = document.querySelector('.subpage-tab[data-subpage="goalies"]');
    if (tab) {
      tab.addEventListener("click", () => {
        if (state.loaded) return;
        state.loaded = true;
        build();
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
