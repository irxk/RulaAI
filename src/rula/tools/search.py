"""Web search and page reading, so Rula can answer about things outside its weights."""
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict

from .base import Tool

log = logging.getLogger("rula.search")

_MAX_PAGE_CHARS = 3500
_MAX_SNIPPET_CHARS = 500
_DEEP_PAGES = 3
_DEEP_PAGE_CHARS = 3000
_MIN_PAGE_CHARS = 400
_TEXTLESS_DOMAINS = ("youtube.com", "youtu.be", "tiktok.com", "twitter.com",
                     "x.com", "instagram.com", "facebook.com", "discord.com",
                     "reddit.com", "pinterest.com")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/126.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en;q=0.9",
}


# Engines tried in turn when a search fails or comes back empty. Measured: the
# same two searches back to back failed outright 2 times in 12 ("No results
# found"), and those turns were answered from the model's memory — which put
# 北岳 in 石川県, 岐阜県 and 長野県. Retrying the default once was not enough.
_BACKENDS = ("auto", "duckduckgo,brave,mojeek", "yahoo,startpage,google")

# A search repeated within ten minutes is answered from here: the same results,
# no wait, and fewer requests counting toward the engines' rate limits.
_CACHE: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_CACHE_TTL, _CACHE_MAX = 600.0, 128
_CACHE_LOCK = threading.Lock()


def _cache_get(query: str) -> str | None:
    with _CACHE_LOCK:
        hit = _CACHE.get(query)
        if hit and time.monotonic() - hit[0] < _CACHE_TTL:
            _CACHE.move_to_end(query)
            return hit[1]
        _CACHE.pop(query, None)
        return None


def _cache_put(query: str, result: str):
    with _CACHE_LOCK:
        _CACHE[query] = (time.monotonic(), result)
        _CACHE.move_to_end(query)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)


# Time allowed for one search, all engine sets together. Measured: a searched
# answer normally takes 2-8 s, yet two in a row of the eval took 70 and 77 s.
# ddgs's "auto" queries its ~10 engines two at a time, waits up to 5 s for each
# pair, and on the way out waits for every request still running; tried three
# times, that adds up to over a minute. Each set runs on a daemon thread that
# is not waited for past this: a late one ends on its own, its result dropped.
_SEARCH_BUDGET = 12.0
# Pages read after a search (deep/smart search), all of them together.
_PAGES_BUDGET = 10.0


def _text(query: str, max_results: int, backend: str) -> list[dict]:
    from ddgs import DDGS
    return list(DDGS().text(query, region="jp-jp", max_results=max_results, backend=backend))


# The next engine set starts when the one before has failed, or has not
# answered within this many seconds; the first results to come back are used.
# Measured: in one 99-question run, 26 of 34 searches waited out the first
# set's full 8 s before the next was tried.
_HEDGE_AFTER = 4.0


def _ddg(query: str, max_results: int) -> tuple[list[dict], str | None]:
    deadline = time.monotonic() + _SEARCH_BUDGET
    cond = threading.Condition()
    state: dict = {"results": None, "errors": [], "finished": 0}

    def attempt(backend: str):
        try:
            results = _text(query, max_results, backend)
            error = None if results else "no results"
        except Exception as e:
            results, error = [], f"{type(e).__name__}: {e}"
        with cond:
            state["finished"] += 1
            if results and state["results"] is None:
                state["results"] = results
            elif error:
                state["errors"].append(error)
            cond.notify_all()

    launched, next_start = 0, time.monotonic()
    with cond:
        while state["results"] is None:
            now = time.monotonic()
            all_failed = state["finished"] == launched
            if launched < len(_BACKENDS) and (now >= next_start or all_failed):
                if launched:
                    log.info("検索を別のエンジンでも試します (%d/%d)", launched + 1, len(_BACKENDS))
                threading.Thread(target=attempt, args=(_BACKENDS[launched],), daemon=True,
                                 name="rula-search").start()
                launched += 1
                next_start = now + _HEDGE_AFTER
                continue
            if now >= deadline or (launched == len(_BACKENDS) and all_failed):
                break
            wake = deadline if launched == len(_BACKENDS) else min(deadline, next_start)
            cond.wait(max(0.01, wake - now))
        if state["results"] is not None:
            return state["results"], None
        if launched > state["finished"]:
            log.info("検索エンジンが時間内に応答しませんでした")
            return [], "timed out"
        return [], state["errors"][-1] if state["errors"] else "no results"


def _read_pages(urls: list[str]) -> list[str]:
    """read_url on each, all at once; a page not in by the deadline is skipped."""
    box: dict[str, str] = {}
    threads = [threading.Thread(target=lambda u=u: box.__setitem__(u, read_url(u)),
                                daemon=True, name="rula-page") for u in urls]
    for th in threads:
        th.start()
    deadline = time.monotonic() + _PAGES_BUDGET
    for th in threads:
        th.join(max(0.0, deadline - time.monotonic()))
    return [box.get(u, "ページの取得に失敗しました: 時間切れ") for u in urls]


# -- Wikipedia ---------------------------------------------------------------
# Asked directly through its API, not through a search engine: for "北岳は何県"
# the article on 北岳 is the authority, and the API does not rate-limit like the
# scraped engines. Only when the question's subject is itself an article title:
# a full-text search for "北岳は何県にありますか" came back with
# "チコちゃんに叱られる!", and a wrong article placed first would mislead.
_WIKI_API = "https://ja.wikipedia.org/w/api.php"
_WIKI_HEADERS = {"User-Agent": "RulaAI/2.0 (local personal assistant)"}
# Only topic markers end the subject. Splitting at で/に/を/が as well turned
# "日本で一番長い川は" into "日本" — the article on Japan, first in the results
# for a question about rivers.
_SUBJECT_END = re.compile(
    r"(?:って|とは|は|の(?:場所|高さ|標高|(?:本来の|正しい|本当の|元々の|もともとの)?意味|"
    r"作者|著者|首都|人口|面積|長さ|読み方|"
    r"作曲者|監督|原作|設立|創業|本社|所在地|誕生日|生年月日|特徴|歴史|由来))")
_FRESH_WORDS = re.compile(r"(最新|最近|今日|今週|今月|今年|昨日|ニュース|速報|株価|為替|天気|予報)")


# Wikimedia answers 429 with a Retry-After when asked too often (measured: after
# about a dozen requests within seconds, every request failed for ~20 s).
# Until then Wikipedia and Wiktionary are not asked at all: retrying early only
# extends the block, and the web engines still answer.
_WIKIMEDIA_PAUSE = {"until": 0.0}


def _wikimedia_get(api: str, params: dict, timeout: float = 5.0):
    if time.monotonic() < _WIKIMEDIA_PAUSE["until"]:
        raise RuntimeError("rate limited")
    url = api + "?" + urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(url, headers=_WIKI_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            try:
                wait = float(e.headers.get("Retry-After", 30))
            except ValueError:
                wait = 30.0
            _WIKIMEDIA_PAUSE["until"] = time.monotonic() + min(max(wait, 5.0), 300.0)
            log.info("Wikimedia の回数制限に当たりました。%.0f 秒は問い合わせません", wait)
        raise


def _wiki_get(params: dict, timeout: float = 5.0):
    return _wikimedia_get(_WIKI_API, params, timeout)


def _subject(query: str) -> str:
    s = re.sub(r"[?？。!！\s]+$", "", query.strip())
    # A quoted word is the subject as written: "「役不足」の本来の意味は" was cut
    # at "の意味" into "役不足」の本来", and "「情けは人の為ならず」" at its own は.
    quoted = re.match(r"[「『](.+?)[」』]", s)
    if quoted:
        return quoted.group(1).strip()
    s = _SUBJECT_END.split(s, maxsplit=1)[0]
    return s.strip(" 「」『』\"'")


def wikipedia_intro(query: str, chars: int = 600) -> str | None:
    """The opening of the Wikipedia article the question is about, or None."""
    if _FRESH_WORDS.search(query):
        return None  # the encyclopedia lags; the news engines do not
    subject = _subject(query)
    if len(subject) < 2:
        return None
    try:
        # The page titled exactly as the subject, in one request. A redirect is
        # followed: "日本の国鳥" is a page of its own that redirects to キジ, and
        # editors made it, so it can be trusted. (An opensearch for the title
        # first cost a second request, and with redirects resolved it listed
        # キジ, not the subject, so the article was missed.)
        pages = _wiki_get({"action": "query", "prop": "extracts", "exintro": 1,
                           "explaintext": 1, "redirects": 1,
                           "titles": subject})["query"]["pages"]
        page = next(iter(pages.values()))
        if "missing" in page or "invalid" in page:
            return None
        title = page.get("title", subject)
        extract = page.get("extract", "").strip()
    except Exception as e:
        log.info("Wikipedia を引けませんでした: %s", type(e).__name__)
        return None
    if not extract or "曖昧さ回避" in extract or re.search(r"(以下を指す|以下のものを指す)", extract[:200]):
        return None
    extract = re.sub(r"\s+", " ", extract)
    if len(extract) > chars:
        extract = extract[:chars] + "…"
    return f"{title} - Wikipedia\n{extract}\n出典: https://ja.wikipedia.org/wiki/{title.replace(' ', '_')}"


# -- Wiktionary --------------------------------------------------------------
# For "what does X mean", the dictionary entry. Measured: asked for すべからく,
# the 9B twice answered with the misused sense ("すべて", "例外なく"), once when
# the scraped engines had failed and it answered from memory. Wiktionary has
# all five of the eval's often-misused words, labels the misused senses as such
# ("（通常、誤用とされる）すべて"), and answers in under a second from the same
# API as Wikipedia, which did not fail when the engines did.
_WIKT_API = "https://ja.wiktionary.org/w/api.php"
_MEANING_Q = re.compile(r"意味|とは|って何|ってなに|語源|由来|誤用|使い方")
_JA_SECTION = re.compile(r"^==\s*(?:\{\{(?:L\|)?ja\}\}|日本語)\s*==\s*$", re.M)
_LEVEL2 = re.compile(r"^==[^=].*==\s*$", re.M)


def _unwiki(text: str) -> str:
    """Wiki markup to plain text, for the few constructs definitions use."""
    text = re.sub(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{\{(?:ふりがな|ruby)\|([^|}]*)\|[^}]*\}\}", r"\1", text)
    text = re.sub(r"\{\{w\|([^|}]*)(?:\|[^}]*)?\}\}", r"\1", text)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    return text.replace("'''", "").replace("''", "").strip()


def wiktionary_meaning(query: str, chars: int = 900) -> str | None:
    """The Japanese senses (and usage notes) of the word the question asks about."""
    word = _subject(query)
    if len(word) < 2:
        return None
    try:
        data = _wikimedia_get(_WIKT_API, {"action": "parse", "page": word,
                                          "prop": "wikitext", "redirects": 1})
        title, wikitext = data["parse"]["title"], data["parse"]["wikitext"]["*"]
    except Exception as e:  # no such entry comes back as an error too
        log.info("ウィクショナリーを引けませんでした: %s", type(e).__name__)
        return None
    start = _JA_SECTION.search(wikitext)
    if not start:
        return None
    end = _LEVEL2.search(wikitext, start.end())
    section = wikitext[start.end():end.start() if end else len(wikitext)]
    senses, notes, in_usage = [], [], False
    for line in section.splitlines():
        if line.startswith("="):
            in_usage = "usage" in line or "語法" in line
        elif line.startswith("#") and line[1:2] not in ("*", ":"):
            senses.append(_unwiki(line.lstrip("# ")))
        elif in_usage and line.startswith("*") and line[1:2] != ":":
            notes.append(_unwiki(line.lstrip("* ")))
    senses = [s for s in senses if s]
    if not senses:
        return None
    body = "\n".join(f"{i}. {s}" for i, s in enumerate(senses, 1))
    if notes:
        body += "\n語法: " + " ".join(n[:200] for n in notes[:2])
    if len(body) > chars:
        body = body[:chars] + "…"
    return (f"{title} - ウィクショナリー(辞書)\n{body}\n"
            f"出典: https://ja.wiktionary.org/wiki/{title.replace(' ', '_')}")


# Ad and click-tracker links some engines slip into results. Measured: a search
# returned a "bing.com/aclick?..." record-shop ad as the first result. These are
# not sources and must never be cited.
_JUNK_URL = re.compile(r"(bing\.com/aclick|/aclk\?|/aclick\?|duckduckgo\.com/y\.js|"
                       r"google\.com/aclk|doubleclick\.net|/adclick)", re.I)


def _dead_wiki_urls(results) -> set:
    """wikipedia.org article URLs in the results that point to no real article.

    Measured: a search for an obscure term returned "[1] よあ帝国 - Wikipedia"
    from the scrapers, whose URL ja.wikipedia.org/wiki/よあ帝国 is a red link
    (404). Cited as a source it looks authoritative but does not exist. One
    existence check per distinct wiki URL; on any error the result is kept
    (fail open), so a rate limit never drops good links.
    """
    dead, seen = set(), {}
    for r in results:
        href = (r.get("href") or "").strip()
        if href in seen:
            if seen[href]:
                dead.add(href)
            continue
        m = re.match(r"https?://([a-z]{2,3})\.wikipedia\.org/wiki/([^?#]+)$", href)
        if not m:
            continue
        title = urllib.parse.unquote(m.group(2)).replace("_", " ")
        try:
            data = _wikimedia_get(f"https://{m.group(1)}.wikipedia.org/w/api.php",
                                  {"action": "query", "titles": title, "redirects": 1})
            page = next(iter(data.get("query", {}).get("pages", {}).values()), {})
            is_dead = "missing" in page or "invalid" in page
        except Exception:
            is_dead = False
        seen[href] = is_dead
        if is_dead:
            dead.add(href)
    return dead


def web_search(query: str, max_results: int = 5) -> str:
    query = query.strip()
    if not query:
        return "検索クエリが空です。"
    cached = _cache_get(query)
    if cached is not None:
        return cached
    try:
        import ddgs  # noqa: F401
    except ImportError:
        return "検索ライブラリが未インストールです。`pip install ddgs` を実行してください。"

    # Alongside the engines, not after. Not in a `with` executor: leaving one
    # waits for the lookup however long it takes, past any timeout given.
    start = time.monotonic()
    found: dict = {}
    lookups = {"intro": wikipedia_intro}
    if _MEANING_Q.search(query):
        lookups["dict"] = wiktionary_meaning

    def lookup(key, fn):
        try:
            found[key] = fn(query)
        except Exception:
            pass
    threads = [threading.Thread(target=lookup, args=item, daemon=True, name="rula-wiki")
               for item in lookups.items()]
    for th in threads:
        th.start()
    results, err = _ddg(query, max_results)
    for th in threads:
        th.join(max(0.5, 8.0 - (time.monotonic() - start)))
    wiki, meaning = found.get("intro"), found.get("dict")

    # The dictionary first: for a meaning question it is the authority, and
    # read first, its labelled misuses are not taken for the meaning.
    blocks = [b for b in (meaning, wiki) if b]
    wiki_url = wiki.rsplit("出典: ", 1)[-1].strip() if wiki else None
    # ddgs's own Wikipedia engine takes the first fuzzy title match, unchecked,
    # and ranks it first: for "日本の国鳥は" that was 日本の国旗, and the answer
    # became トキ under a citation of the flag article. Its results are the
    # wikipedia.org ones whose title is bare (the web engines' end in
    # "- Wikipedia"). Kept last, as a fallback, and dropped when the exact-title
    # lookup above found the article.
    # Drop wikipedia.org results whose article does not actually exist, so a
    # red-link URL is never handed over as a source.
    dead_wiki = _dead_wiki_urls(results)
    fuzzy = []
    for r in results:
        title = (r.get("title") or "").strip()
        body = re.sub(r"\s+", " ", (r.get("body") or "")).strip()
        if len(body) > _MAX_SNIPPET_CHARS:
            body = body[:_MAX_SNIPPET_CHARS] + "…"
        href = (r.get("href") or "").strip()
        if href in dead_wiki or _JUNK_URL.search(href):
            continue
        if wiki_url and urllib.parse.unquote(href).replace(" ", "_") == wiki_url:
            continue  # the same article, already first
        block = f"{title}\n{body}\n出典: {href}"
        if re.match(r"https?://[a-z]+\.wikipedia\.org/wiki/", href) and "Wikipedia" not in title:
            fuzzy.append(block)
        else:
            blocks.append(block)
    if not wiki:
        blocks += fuzzy
    if not blocks:
        return (f"検索に失敗しました: {err}" if err and err != "no results"
                else f"「{query}」の検索結果は見つかりませんでした。")
    out = "\n\n".join(f"[{i}] {b}" for i, b in enumerate(blocks, 1))
    _cache_put(query, out)
    return out


def deep_search(query: str, max_results: int = 8) -> str:
    """Search, then read pages until there is enough text to work from.

    Snippets alone are a sentence or two each — enough to answer "how tall is
    Mt Fuji", nowhere near enough to write anything considered about a topic. For
    questions that ask for a real look, the article text is what makes the
    difference.

    It walks down the results rather than taking the top two: the highest-ranked
    hits are often a link page or a YouTube channel, whose static HTML yields
    barely 100 characters.
    """
    summary = web_search(query, max_results=max_results)
    if summary.startswith(("検索に失敗", "検索クエリ", "検索ライブラリ")) or "見つかりません" in summary:
        return summary

    urls = [url for url in re.findall(r"^出典: (\S+)$", summary, re.M)
            if not any(d in url for d in _TEXTLESS_DOMAINS)][:_DEEP_PAGES + 2]
    if not urls:
        return summary
    sections = [summary]
    for url, body in zip(urls, _read_pages(urls)):
        if len(sections) > _DEEP_PAGES:
            break
        if body.startswith(("ページの取得に失敗", "http:// または")) or len(body) < _MIN_PAGE_CHARS:
            continue
        sections.append(f"--- 本文: {url} ---\n{body[:_DEEP_PAGE_CHARS]}")
    return "\n\n".join(sections)


def read_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^https?://", url):
        return "http:// または https:// で始まるURLを指定してください。"
    try:
        import urllib.request

        from bs4 import BeautifulSoup
    except ImportError:
        return "`pip install beautifulsoup4` を実行してください。"

    # Non-ASCII in the path ("…/wiki/北岳") is percent-encoded here: the HTTP
    # request line must be ASCII, and the readable form is what gets cited.
    url = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%~")
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            html = _decode(raw, resp.headers.get("Content-Type", ""))
    except Exception as e:
        return f"ページの取得に失敗しました: {type(e).__name__}: {e}"

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside",
                     "iframe", "noscript"]):
        tag.decompose()
    content = soup.find("article") or soup.find("main") or soup.find("body") or soup
    text = re.sub(r"\n{3,}", "\n\n", content.get_text("\n")).strip()
    if len(text) > _MAX_PAGE_CHARS:
        text = text[:_MAX_PAGE_CHARS] + "\n…(以降省略)"
    return text or "本文を抽出できませんでした。"


def _decode(raw: bytes, content_type: str) -> str:
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            return raw.decode(charset)
        except (LookupError, UnicodeDecodeError):
            pass
    for enc in ("utf-8", "shift_jis", "euc-jp"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def smart_search(query: str, max_results: int = 5) -> str:
    """Decompose complex queries into sub-queries, search each in parallel,
    deduplicate, and read the best pages — a single tool call that does what
    a human researcher would: look at the topic from multiple angles."""
    from ..quality import decompose_query

    queries = decompose_query(query)
    if queries is None or len(queries) <= 1:
        return deep_search(query, max_results=max_results)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(queries)) as pool:
        raw_results = list(pool.map(
            lambda q: web_search(q, max_results=max_results), queries))

    seen_urls: set[str] = set()
    blocks: list[str] = []
    for result in raw_results:
        for block in result.split("\n\n"):
            url_m = re.search(r"出典: (\S+)", block)
            if url_m:
                url = url_m.group(1)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
            blocks.append(block)

    summary = "\n\n".join(blocks)

    urls = [url for url in re.findall(r"^出典: (\S+)$", summary, re.M)
            if not any(d in url for d in _TEXTLESS_DOMAINS)][:_DEEP_PAGES + 2]
    if not urls:
        return summary

    sections = [summary]
    for url, body in zip(urls, _read_pages(urls)):
        if len(sections) > _DEEP_PAGES:
            break
        if body.startswith(("ページの取得に失敗", "http:// または")) or len(body) < _MIN_PAGE_CHARS:
            continue
        sections.append(f"--- 本文: {url} ---\n{body[:_DEEP_PAGE_CHARS]}")
    return "\n\n".join(sections)


SEARCH_TOOL = Tool(
    name="search",
    description="Web検索して最新情報や事実を調べる。知らないこと・最近の出来事は必ずこれを使う。",
    usage="search: 調べたいキーワード",
    run=web_search,
)

DEEP_SEARCH_TOOL = Tool(
    name="deep_search",
    description="Web検索し、上位ページの本文まで読み込む。詳しい説明や考察が必要なときに使う。",
    usage="deep_search: 調べたいテーマ",
    run=deep_search,
)

READ_URL_TOOL = Tool(
    name="read_url",
    description="指定したURLのページ本文を読み取る。",
    usage="read_url: https://example.com",
    run=read_url,
)

SMART_SEARCH_TOOL = Tool(
    name="smart_search",
    description="複雑な質問を分解して多角的に並列検索する。比較・複数対象の質問に使う。",
    usage="smart_search: AとBの違い",
    run=smart_search,
)
