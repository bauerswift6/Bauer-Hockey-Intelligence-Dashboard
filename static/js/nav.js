/* =========================================================
   Hub Navigation — Country / League / Sub-page state
   ========================================================= */

const LEAGUE_DEFAULTS = {
  usa: "nhl",
  canada: "ohl",
  international: "khl",
};

const LEAGUE_SUBPAGE_DEFAULTS = {
  nhl: "morning-brief",
};

const state = {
  country: "usa",
  league: "nhl",
  subpage: "morning-brief",
};

function $(sel, root = document) {
  return root.querySelector(sel);
}
function $$(sel, root = document) {
  return Array.from(root.querySelectorAll(sel));
}

function setActiveTab(group, value, attr) {
  $$(`.${group}`).forEach((tab) => {
    const isActive = tab.dataset[attr] === value;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}

function showLeagueGroup(country) {
  $$(".league-group").forEach((group) => {
    group.hidden = group.dataset.country !== country;
  });
}

function showSubpageGroup(league) {
  const subpagesRow = $(".nav-subpages");
  subpagesRow.hidden = false;
  const realGroup = $(`.subpage-group[data-league="${league}"]`);
  const placeholder = $(".subpage-group.subpage-placeholder");
  $$(".subpage-group").forEach((g) => {
    if (g === placeholder) {
      g.hidden = !!realGroup;
    } else {
      g.hidden = g.dataset.league !== league;
    }
  });
}

function showView(viewKey) {
  $$(".page-view").forEach((view) => {
    view.hidden = view.dataset.view !== viewKey;
  });
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}

function resolveViewKey() {
  if (state.league === "nhl") {
    return `nhl/${state.subpage}`;
  }
  return `${state.country}/${state.league}`;
}

function render() {
  setActiveTab("country-tab", state.country, "country");
  showLeagueGroup(state.country);
  setActiveTab("league-tab", state.league, "league");
  showSubpageGroup(state.league);
  if (state.league === "nhl") {
    setActiveTab("subpage-tab", state.subpage, "subpage");
  }
  showView(resolveViewKey());
}

function selectCountry(country) {
  if (state.country === country) return;
  state.country = country;
  state.league = LEAGUE_DEFAULTS[country];
  state.subpage = LEAGUE_SUBPAGE_DEFAULTS[state.league] || null;
  render();
}

function selectLeague(league) {
  if (state.league === league) return;
  state.league = league;
  state.subpage = LEAGUE_SUBPAGE_DEFAULTS[league] || null;
  render();
}

function selectSubpage(subpage) {
  if (state.subpage === subpage) return;
  state.subpage = subpage;
  render();
}

document.addEventListener("DOMContentLoaded", () => {
  $$(".country-tab").forEach((tab) => {
    tab.addEventListener("click", () => selectCountry(tab.dataset.country));
  });
  $$(".league-tab").forEach((tab) => {
    tab.addEventListener("click", () => selectLeague(tab.dataset.league));
  });
  $$(".subpage-tab").forEach((tab) => {
    tab.addEventListener("click", () => selectSubpage(tab.dataset.subpage));
  });
  render();
});
