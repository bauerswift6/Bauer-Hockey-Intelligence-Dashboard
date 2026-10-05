"""
Morning Brief news classification — pure, testable, no network or app imports.

Two jobs:
  1. big_news_score / rank_big_news  → curate the "League Pulse" top feed.
  2. is_real_move                     → keep "Ink Report" / "Trade Block"
                                        to actually-executed transactions,
                                        not columns/opinion/video about them.

Items are plain dicts shaped like the RSS parse output in app.py:
    {"title": str, "link": str, "source": str,
     "pub_date": iso-str|None, "type": str|None}

The feed is title-only (no body, tags, or engagement counts), so every
signal here is derived from the title text, the article/video URL, and the
inferred transaction type. Thresholds and the lexicon are meant to be tuned.
"""
import re

# ---------------------------------------------------------------------------
# Curated marquee lexicon
# ---------------------------------------------------------------------------
# Small hand-maintained set of star surnames. Its only job is to recover
# plainly-worded blockbusters that keyword rules miss — e.g.
# "Blues trade Jordan Kyrou to Capitals" reads like an ordinary trade until
# you know Kyrou is a star. Keep it short; over-stuffing hurts precision.
# Matched on whole words (see _lexicon_score). Names that are also common
# English words or common first names (e.g. "connor", "fox", "point") are
# deliberately excluded — they cause false positives like "Connor Murphy" or
# the "Fox:" byline.
STAR_PLAYERS = {
    "mcdavid", "draisaitl", "matthews", "marner", "crosby", "malkin",
    "ovechkin", "mackinnon", "makar", "rantanen", "kucherov",
    "hedman", "stamkos", "barkov", "reinhart", "tkachuk", "pastrnak",
    "marchand", "eichel", "pietrangelo", "panarin", "shesterkin",
    "hischier", "meier", "bedard", "kaprizov", "hellebuyck",
    "scheifele", "werenski", "sorokin", "tavares", "nylander",
    "kyrou", "binnington", "tuch", "byram", "rielly", "knies", "nemec",
    "eklund", "celebrini", "demidov", "michkov", "fantilli",
}
_LEXICON_RE = re.compile(r"\b(?:" + "|".join(STAR_PLAYERS) + r")\b", re.I)

# ---------------------------------------------------------------------------
# Big-news topic signals (additive weights)
# ---------------------------------------------------------------------------
HIGH_SIGNAL_PATTERNS = [
    # Coaching / front-office changes
    (re.compile(r"\b(fired|relieved of (?:his |their )?dut|steps? down|"
                r"named (?:the )?(?:new )?(?:head |interim )?coach|"
                r"hire[sd]?|as (?:the )?(?:new )?(?:head |interim )?coach|"
                r"behind the bench|new (?:head )?coach|"
                r"general manager|new gm|named gm)\b", re.I), 5.0),
    (re.compile(r"\bsuspend", re.I), 5.0),
    (re.compile(r"\b(retire[sd]?|retirement|hangs? (?:them|it) up)\b", re.I), 4.0),
    (re.compile(r"\b(buyout|bought out|placed on (?:unconditional )?waivers? "
                r"for the purpose)\b", re.I), 3.5),
    (re.compile(r"\bsign-?and-?trade\b", re.I), 3.5),
    (re.compile(r"\bnamed (?:team |new )?captain\b", re.I), 3.0),
    (re.compile(r"\b(requests? (?:a )?trade|trade request|no-?trade|holdout|"
                r"holding out|contract dispute|submits? .{0,20}trade list)\b", re.I), 3.0),
    (re.compile(r"\b(blockbuster|landmark|record[- ](?:deal|contract|extension|"
                r"signing))\b", re.I), 3.5),
]

# Speculation / derivative content — penalised for Pulse and rejected outright
# for the transaction sections. Catches questions, hot-takes, and video clips.
SPECULATION_RE = re.compile(
    r"(\?)"
    r"|^\W*(why|should|could|would|how|what|is it|did|will)\b"
    r"|\b(smart bet|overpay|should be careful|preview|rumou?rs?|report:|"
    r"insider|mailbag|grade[sd]?|takeaways?|analysis|what we learned|"
    r"power rankings|way-too-early|dangle|big board|trade bait|"
    r"could be|might be|linked to|interested in)\b",
    re.I,
)

# Magnitude signals
MONEY_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*(?:m\b|million)", re.I)
BIG_TERM_RE = re.compile(r"\b(eight|seven|six)-year\b", re.I)
PICK_RE = re.compile(r"\b(no\.?\s*\d+\s*(?:overall\s*)?pick|"
                     r"first[- ]round pick|1st[- ]round pick)\b", re.I)
TRADE_VERB_RE = re.compile(r"\b(trade[ds]?|trading|acquir|sign-?and-?trade)", re.I)


def is_speculative(item) -> bool:
    """True for questions, hot-takes, rumour pieces, and video reaction clips."""
    link = (item.get("link") or "").lower()
    if "/video/" in link:
        return True
    return bool(SPECULATION_RE.search(item.get("title") or ""))


def _money_score(title: str) -> float:
    # Capped as a *supporting* signal, not a sole trigger: headlines mix AAV
    # with total contract value, and a big total (e.g. "$20.5M deal") on a
    # mid-tier contract should not by itself reach the big-news threshold.
    best = 0.0
    for m in MONEY_RE.finditer(title):
        try:
            val = float(m.group(1))
        except (TypeError, ValueError):
            continue
        if val >= 9:
            best = max(best, 2.0)
        elif val >= 6:
            best = max(best, 1.5)
        elif val >= 4:
            best = max(best, 1.0)
    if BIG_TERM_RE.search(title):
        best = max(best, 2.0)
    return best


def _lexicon_score(title: str) -> float:
    return 2.0 if _LEXICON_RE.search(title) else 0.0


def big_news_score(item) -> float:
    """Additive title-scoring model. Higher = bigger league-wide story."""
    title = item.get("title") or ""
    t = title.lower()
    score = 0.0

    for pat, weight in HIGH_SIGNAL_PATTERNS:
        if pat.search(t):
            score += weight

    score += _money_score(title)
    score += _lexicon_score(t)

    typ = (item.get("type") or "").upper()
    is_trade = typ == "TRADE" or bool(TRADE_VERB_RE.search(t))
    if is_trade:
        score += 1.0                       # any executed/real trade has baseline interest
        if PICK_RE.search(title):
            score += 2.0                   # picks moving => bigger deal
    if typ == "SUSPENSION":
        score += 4.0
    elif typ == "IR":
        score += 1.0

    if is_speculative(item):
        score -= 4.0

    return score


DEFAULT_THRESHOLD = 3.0


def is_big_news(item, threshold: float = DEFAULT_THRESHOLD) -> bool:
    return big_news_score(item) >= threshold


def rank_big_news(items, limit: int = 8, threshold: float = DEFAULT_THRESHOLD):
    """Score, threshold, de-dupe (within Pulse only), sort by score then recency."""
    scored = []
    seen = set()
    for it in items:
        link = it.get("link")
        if link and link in seen:
            continue
        if link:
            seen.add(link)
        s = big_news_score(it)
        if s >= threshold:
            scored.append((s, it))
    scored.sort(key=lambda pair: (pair[0], pair[1].get("pub_date") or ""),
                reverse=True)
    return [it for _, it in scored[:limit]]


# ---------------------------------------------------------------------------
# Transaction "real move" filter (Ink Report / Trade Block)
# ---------------------------------------------------------------------------
SIGNING_MOVE_RE = re.compile(
    r"\b(signs?|signed|re-?signs?|re-?signed|inks?|"
    r"agree[sd]? to (?:a |terms)|signing|extension|extend[sd]?)\b", re.I)
TRADE_MOVE_RE = re.compile(
    r"\b(trade[sd]?|trading|acquir(?:e|es|ed|ing)|deals?|dealt|"
    r"sign-?and-?trade)\b", re.I)
# Trade-adjacent chatter that is NOT an executed deal — reject from Trade Block
# even though it isn't strictly "speculative".
TRADE_NONMOVE_RE = re.compile(
    r"\b(trade (?:list|calls?|talks?|rumou?rs?|request|deadline|preview|"
    r"market|board|bait|chip|winners?|losers?)|after .{0,30}\btrade\b|"
    r"are more trades|on the (?:trade )?block|trade (?:scenarios?|targets?))\b",
    re.I)


def is_real_move(item) -> bool:
    """True only if the title reads like an executed signing/trade.

    Used to keep Ink Report and Trade Block clean when the upstream feed is the
    Sportsnet general-news fallback (keyword-matched, full of opinion/columns)
    rather than NHL.com's structured transactions feed.
    """
    if is_speculative(item):
        return False
    title = item.get("title") or ""
    typ = (item.get("type") or "").upper()
    if typ == "SIGNING":
        return bool(SIGNING_MOVE_RE.search(title))
    if typ == "TRADE":
        if TRADE_NONMOVE_RE.search(title):
            return False
        return bool(TRADE_MOVE_RE.search(title))
    return False  # non TRADE/SIGNING types are routed to League Pulse instead
