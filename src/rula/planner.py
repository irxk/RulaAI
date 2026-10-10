"""Decide, in code, when a turn needs a tool.

Qwen-family instruct models carry a strong "I can't access the internet" reflex.
Given tools in the system prompt they still refuse outright a large fraction of
the time, and once one refusal lands in the history the model stays consistent
with it and refuses everything after. Asking the model to choose is therefore
unreliable; this module decides from the user's message instead, so the search
happens whether or not the model was inclined to ask for it.
"""
import re

URL_RE = re.compile(r"https?://[^\s、。」）\)]+")

# Explicit "go look it up" requests.
_SEARCH_VERBS = re.compile(
    r"(検索し|検索して|調べて|調べられ|ググっ|ぐぐっ|探して|調査して|確認して)")

# Topics whose answer changes over time, where stale weights are worse than useless.
_FRESH_TOPICS = re.compile(
    r"(最新|最近|今年|去年|昨年|今月|今週|今日の|現在の|直近|ニュース|速報|"
    r"新作|新製品|新機能|リリース|発売|株価|為替|天気|予報|順位|ランキング|"
    r"おすすめ|オススメ|人気|評判|レビュー|口コミ)")

# An explicit request for depth. Deliberately narrow: "〜について教えて" used to
# match, so an ordinary question pulled three web pages and produced a 3,300
# character report in 156 seconds — long enough for callers to time out.
_WANTS_DEPTH = re.compile(
    r"(詳しく|詳細に|長文で|長めに|まとめて|解説して|説明して|考察|分析して|"
    r"比較して|要約して|レポート|深掘り|徹底的に)")
# The user has said how much they want. Measured: "PythonとRubyの違いを一言で" ran
# a three-way search plus three page reads and came back after 21s as a table.
_WANTS_BRIEF = re.compile(
    r"(一言で|ひとことで|一行で|簡潔に|手短に|短く|端的に|ざっくり)")
# Marks the message as being about an external subject, so "どう思う?" about the
# conversation so far doesn't send Rula off to search the web.
_ABOUT_TOPIC = re.compile(r"(について|に関して|とは)")

# A question with a checkable answer: a name, a place, a date, a measurement. A 4B
# model states these confidently and gets them wrong — asked for Japan's second
# highest mountain it answered 立山, then 富士山; the correct 北岳 only came from a
# search. Anything with an external subject is worth looking up.
_FACT_QUESTION = re.compile(
    r"(どこ|だれ|誰|いつ|何年|何月|何日|何人|何個|何番目|いくつ|いくら|"
    r"どれくらい|どのくらい|何メートル|何キロ|何度|何時間|何分|"
    r"は何|は誰|はどこ|一番|最も|最大|最小|最長|最高|最初|人口|面積|標高|高さ|長さ|"
    r"設立|創業|生まれ|受賞|優勝|"
    r"[0-9０-９]+番目|(世界|日本|国内|全国|県内|市内)一)")
# "枕草子の作者は?" / "日本の国鳥は?": an attribute of a named thing, asked with no
# question word, so _FACT_QUESTION missed it. Measured on questions not used in
# tuning: answered from memory, 枕草子 went to 紫式部 both times and 日本の国鳥
# once to a bird that does not exist.
_ATTRIBUTE_Q = re.compile(
    r"の(作者|著者|筆者|作曲者|作詞者|監督|原作者|脚本家?|主演|首都|国鳥|国花|国歌|通貨|公用語|"
    r"元素記号|化学式|原子番号|創業者|創設者|設立者|社長|会長|メンバー|本名|出身地|生年月日|"
    r"誕生日|所在地|本社|発売日|学名|英語名|別名|正式名称|名前|名称)(は|って)[?？]?\s*$"
    r"|(描いた|書いた|発明した|発見した|建てた|設計した|作曲した|作詞した|監督した|創業した|設立した)"
    r"(の|人)は(誰|だれ)?[?？]?\s*$")
# ...unless it is about the one being asked ("あなたの誕生日は?" is for the persona),
# about the user ("私の誕生日は?" is in the conversation, not on the web), or
# points back into the conversation ("この小説の作者は": a search for "この小説"
# finds nothing about the one meant).
_FIRST_PERSON = re.compile(r"(私|わたし|僕|ぼく|俺|おれ|自分|うち)(は|が|の|も|に|って)")
_YOU = re.compile(r"(あなた|君|きみ|お前|おまえ|Rula|ルーラ)|^\s*(この|その|あの)"
                  r"|" + _FIRST_PERSON.pattern)


# A message that hands over text to work on: "次の文章を50字以内で要約して。\n\n…".
# Read whole, the text inside steered everything: "個人差" in a passage on sleep
# made the request a comparison, and it was sent to a web search; the material
# was already there.
_PASSAGE_TASK = re.compile(
    r"(要約|まとめ|翻訳|訳して|訳す|校正|添削|言い換え|書き直|直して|タイトル|見出し|要点|"
    r"感想|分析|読んで|summar|translat|rewrite|proofread)", re.I)
_PASSAGE_INTRO = re.compile(
    r"^(.{0,80}?(?:次の|以下の|下の|この|後の)(?:文章|文|記事|テキスト|メール|英文|和文|段落)"
    r"[^\n]{0,60}?(?:[。:：]|\n))\s*(\S.{79,})$", re.S)


def split_passage(text: str) -> tuple[str, str | None]:
    """(the instruction, the passage it is about), or (text, None) if none."""
    text = text.strip()
    head, sep, body = text.partition("\n\n")
    if sep and len(body.strip()) >= 80 and _PASSAGE_TASK.search(head):
        return head.strip(), body.strip()
    m = _PASSAGE_INTRO.match(text)
    if m and _PASSAGE_TASK.search(m.group(1)):
        return m.group(1).strip(), m.group(2).strip()
    return text, None


_ENGLISH_ASKED = re.compile(r"英語(で|1文で|一文で|に訳|に翻訳|に直|の文章で)|in English", re.I)


def reply_language(text: str) -> str:
    """"英語" when the message is in English or asks for English, else "日本語".

    Measured: of ten questions asked in English, nine were answered in
    Japanese ("What is the capital of Canada?" -> "カナダの首都はオタワです。"),
    and "英語1文で要約して" got a plan, in Japanese, for writing the summary.
    """
    if _ENGLISH_ASKED.search(text):
        return "英語"
    if re.search(r"日本語(で|に)", text):
        return "日本語"
    # Decided by how the message ends, which is where the sentence's grammar
    # is: "What does いただきます mean?" is English, "OK Google って何?" and
    # "I love you を日本語で" are Japanese. A single English word ("OK",
    # "Thanks") is not enough to switch.
    # A figure can end it too: "What is 15% of 80?".
    tail = re.sub(r"[\s?？!！。．.、,）)」』\"'’%]+$", "", text)
    if tail[-1:].isascii() and tail[-1:].isalnum() and len(re.findall(r"[A-Za-z]+", text)) >= 2:
        return "英語"
    return "日本語"


# A message that opens by pointing back into the conversation. Searched as
# written, "それより高い山は日本にある?" (after 富士山) found other mountains and
# was answered "旭岳"; the conversation already says what "それ" is.
_DEICTIC = re.compile(r"^\s*(それ|これ|あれ|その|この|あの|そこ|ここ|あそこ|そっち|こっち)")


def searchable(text: str) -> bool:
    """Whether a message may be sent to a search engine to check an answer.

    A question about the world, not a statement, not about the user, and not
    pointing back into the conversation. Measured: told "私は1990年生まれです。",
    the fact-check pass searched for exactly that sentence — the user's birth
    year, sent to a search engine.
    """
    return (bool(_IS_QUESTION.search(text)) and not _FIRST_PERSON.search(text)
            and not _DEICTIC.match(text))
# Marks it as a question at all, so statements mentioning these words are left alone.
_IS_QUESTION = re.compile(r"[?？]|(です|ます|だ|の)か[?？]?\s*$|教えて")
# Things the model can work out unaided; searching them adds seconds and nothing else.
_SELF_CONTAINED = re.compile(
    r"^\s*[\d\s+\-*/×÷().]+\s*[?？]?\s*$"          # arithmetic
    r"|(こんにちは|こんばんは|おはよう|ありがとう|よろしく|元気|さようなら)")
# Questions about Rula itself, which the persona answers and the web does not.
_ABOUT_SELF = re.compile(
    r"(あなた|君|きみ|Rula|ルーラ)(の|は|って|に)?"
    r".{0,10}(ベース|基盤|モデル|名前|開発元|作られ|正体|中身|仕組み|できること)"
    r"|自己紹介"
    # "あなたは誰?" fell through to _FACT_QUESTION on "は誰": it was searched,
    # and the Wikipedia article on the pronoun あなた was cited as the source.
    r"|^\s*(あなた|君|きみ|Rula|ルーラ)(は|って)(誰|だれ|何者|なにもの|何|なに)"
    # Asked of an assistant, these mean "yours" even with the subject left out,
    # which is how they actually get typed. "開発元はどこですか?" has no subject at
    # all and used to fall through to _FACT_QUESTION on the "どこ", which sent it to
    # the web and answered with whichever unrelated company the search turned up.
    r"|ベースAI|ベースモデル|基盤モデル"
    r"|開発元|開発者|開発した(の|人|会社)|作ったのは|誰が作っ|どこが作っ")


# A request to change how Rula speaks, e.g. "関西弁で話して" / "辛口で答えて" /
# "冷笑してみて". Captured so the style can be pinned in the system prompt, which
# is the only place this model reliably acts on it.
_TONE_SPEAK_RE = re.compile(
    r"^\s*(.{1,20}?)(?:口調|感じ|調子)?(?:で|に)\s*"
    r"(?:話し|喋っ|答え|返事し|言っ)て(?:ください|くれ|みて|ね)?[。、.!！?？]*\s*$")
_TONE_ACT_RE = re.compile(
    r"^\s*(.{1,12}?)(?:して|になって|ぶって)(?:ください|くれ|みて|ね)?[。、.!！?？]*\s*$")
_TONE_ACT_WORDS = ("冷笑", "嘲笑", "皮肉", "辛口", "毒舌", "煽り", "偉そう", "優しく",
                   "厳しく", "砕け", "タメ口", "敬語", "丁寧")

# "Xとは" at the end of a message → definition question worth searching.
_DEFINITION_Q = re.compile(r".{2,}とは[何なに]?(?:です)?か?[?？。]?\s*$")
# "〜の意味は?" — sayings and words whose popular reading is wrong. Unsearched, the
# 4B answered "情けは人の為ならず" with the common misreading, the opposite of the
# actual meaning; a definition is exactly what a search settles.
# Asked as a question, and about a named thing: "この言葉の意味って何?" points back
# at the conversation, and searching those words literally finds nothing useful.
_MEANING_Q = re.compile(
    r"^(?!\s*(?:この|その|あの|これ|それ|あれ)).{2,}の(?:本来の|正しい|本当の)?意味"
    r"(?:は(?:何|なに|なん)?|って(?:何|なに|なん)?|を教えて)"
    r"(?:ですか|でしょうか|ですか)?[?？。]?\s*$")
# A written-out sum. Left to the model, the 9B answered "17×23を計算して" with 481
# (it is 391) without calling the tool the 4B had called; deciding in code, as with
# search, makes the right answer independent of which model is loaded.
_ARITH_EXPR = re.compile(
    r"[(（]*\s*[0-9０-９][0-9０-９.,]*"
    r"(?:\s*[)）]*\s*[-+*/×÷^＋－＊／]\s*[(（]*\s*[0-9０-９][0-9０-９.,]*)+\s*[)）]*")
# "/" and "-" are often dates and phone numbers ("2026/10/09は何曜日?"), so on
# their own they count only when the numbers cannot be read that way.
_ARITH_OP = re.compile(r"[×÷*＊+＋^]")
_CALC_CUE = re.compile(r"(計算|けいさん|いくつ|いくら|答え|[=＝?？]|は\s*$)")
# A word problem: everything needed is in the question, and what it takes is
# working, not looking up. Measured on the 9B: unsearched and unreasoned it said
# 84 for an average of 70 km/h and 72 for the middle of three integers summing to
# 72; "何匹/いくら/何度" also sent half of them to the web, where one answer was
# copied from a homework site. These get reasoning instead (see Agent.ask).
# 濃度/食塩水: "濃度10%の食塩水200gに水を50g加えると、濃度は何%?" went to a web
# search, and the answer (8%) came last, after the working.
_WORD_PROBLEM_STRONG = re.compile(
    r"(総当たり|連続する|ある数|鶴と亀|つるかめ|長針|短針|何通り|確率|方程式|植木算|旅人算|"
    r"平均時速|平均の速さ|約数|倍数|最大公約数|最小公倍数|濃度|食塩水|仕事算|"
    # Sequences: "1、1、2、3、5、8、13 の次の数は?" was answered with the rule
    # first and the number last, and once as "斐波那契数列" (Chinese).
    r"次の数|数列|規則性|何乗|[nｎxｘ]乗|"
    # Geometry: "正方形の対角線の長さが10cmのとき、面積は?" was searched and
    # answered 25 while its own working said 50.
    r"対角線|半径|直径|一辺|辺の長さ|底辺|円周率)")
# Age from a birth year: "(1990年生まれ…)2026年の誕生日を迎えたら何歳?". Left
# unrouted, the 9B called the date tool and answered 37.
_AGE_SUM = re.compile(r"(誕生日を迎え|生まれ).{0,20}何歳|何歳.{0,10}(誕生日|生まれ)")
_WORD_PROBLEM = re.compile(
    r"(平均|合計|全部で|合わせて|足すと|足して|引くと|引いて|かけると|割ると|和は|差は|"
    r"何歳|年上|年下|倍|割引|[%％]引き|割引き|あたり|何匹|何個|何本|何枚|何人|何回|"
    r"時速|分速|秒速|速さ|距離|かかる時間|何分|何時間|比|割合|残り|余り|お釣り|おつり|"
    r"割り勘|ずつ|分けると|分ける|買うと|買って|払うと|支払)")
_ASKS_AMOUNT = re.compile(r"(いくら|いくつ|何[個人本匹枚回円歳日分秒割]|どれだけ|何時間|何[%％]|何パーセント)")
_NUMBER = re.compile(r"[0-9０-９]+(?:[.,][0-9]+)?|[一二三四五六七八九十百千]+(?=[人個本匹枚回円歳つ])")


def asks_amount(text: str) -> bool:
    """Whether the message asks how many / how much."""
    return bool(_ASKS_AMOUNT.search(text))


def wants_reasoning(user_input: str) -> bool:
    """Whether the message is a word problem to be worked out rather than looked up."""
    text = user_input.strip()
    if _SEARCH_VERBS.search(text) or URL_RE.search(text):
        return False
    numbers = len(_NUMBER.findall(text))
    if _WORD_PROBLEM_STRONG.search(text) and numbers >= 1:
        return True
    if _AGE_SUM.search(text) and numbers >= 1:
        return True
    if _WORD_PROBLEM.search(text) and numbers >= 2:
        return True
    # Three or more figures and a "how much": a sum to work out, whatever its words.
    return numbers >= 3 and bool(_ASKS_AMOUNT.search(text))


# The date and time are in every prompt. "今年は西暦何年?" used to match "今年" as a
# fresh topic, ran a search, and opened its answer with "検索結果がありませんでした".
_CLOCK_Q = re.compile(
    r"^\s*(?:今日|本日|今|現在|今年|今月)(?:は|って)?(?:西暦)?"
    r"(?:何年|何月|何日|何月何日|何曜日?|何時何分|何時|いつ)(?:です|でしょう)?か?[?？。]?\s*$")
# "する方法" / "のやり方" → how-to question, needs deep material.
_HOW_TO = re.compile(r"(する|の|な)(方法|やり方|手順|仕方|し方|コツ|ポイント|手続き)")
_COMPARISON = re.compile(r".{1,20}と.{1,20}(の違い|を比較|の比較|のどちら|はどっち)")
_ENGLISH_SEARCH = re.compile(
    r"(?i)\b(what is|who is|when did|where is|how to|how many|how much|"
    r"latest|newest|current|difference between)\b")
_REASON_Q = re.compile(r"(なぜ|何故|理由|原因|どうして).{2,}.*(のか|の？|のですか|んですか|[?？])")


# A bare noun like "冷笑" is too terse to steer generation; spelled out, it lands.
_TONE_EXPANSIONS = {
    "冷笑": "冷笑的で皮肉っぽい口調。相手を軽くあしらい、突き放した物言いをする",
    "嘲笑": "嘲るような口調。相手の話を茶化して見下した言い方をする",
    "皮肉": "皮肉屋の口調。遠回しに嫌味を混ぜて話す",
    "毒舌": "毒舌な口調。歯に衣着せず容赦なく言う",
    "辛口": "辛口な口調。褒めずに厳しく率直に指摘する",
    "煽り": "挑発的で煽るような口調",
    "偉そう": "偉そうで上から目線の口調",
    "タメ口": "友達に話すようなくだけたタメ口",
    "関西弁": "関西弁。「〜やで」「〜せやな」「〜ちゃう?」のような話し方",
}


# "猫の絵を描いて" / "夕焼けの画像を作って" — a request to draw, not to answer.
_DRAW_RE = re.compile(
    r"^(?P<subject>.+?)(?:の)?(?:絵|画像|イラスト|写真|図)?を?\s*"
    r"(?:描いて|書いて|生成して|作って|つくって)(?:ください|くれ|みて|ね|よ)?[。、.!！?？]*\s*$")
# Without one of these the sentence is probably about writing text, not drawing.
_DRAW_MARKERS = ("絵", "画像", "イラスト", "写真", "図", "描い")


def detect_draw(user_input: str) -> str | None:
    """Return the subject to draw if the message is asking for an image."""
    text = user_input.strip()
    if not any(m in text for m in _DRAW_MARKERS):
        return None
    m = _DRAW_RE.match(text)
    if not m:
        return None
    subject = m.group("subject").strip(" 、。の")
    return subject or None


def wants_depth(user_input: str) -> bool:
    """Whether the message asks to be told about something, rather than asking a
    single fact. Decides how full an answer to write."""
    return bool(_WANTS_DEPTH.search(user_input))


def wants_brief(user_input: str) -> bool:
    """Whether the message asks for a short answer ("一言で", "簡潔に")."""
    return bool(_WANTS_BRIEF.search(user_input))


def detect_tone(user_input: str) -> str | None:
    """Return a style directive if the message is asking Rula to change its tone."""
    text = user_input.strip()
    m = _TONE_SPEAK_RE.match(text)
    if not m:
        m = _TONE_ACT_RE.match(text)
        if not (m and any(w in m.group(1) for w in _TONE_ACT_WORDS)):
            return None
    tone = m.group(1).strip()
    for key, expanded in _TONE_EXPANSIONS.items():
        if key in tone:
            return expanded
    return tone


def plan(user_input: str) -> tuple[str, str] | None:
    """Return (tool_name, argument) to run before answering, or None."""
    planned = _plan(user_input)
    # A short answer needs snippets, not three pages of article text.
    if planned and planned[0] in ("deep_search", "smart_search") and wants_brief(user_input):
        return "search", planned[1]
    return planned


def _plan(user_input: str) -> tuple[str, str] | None:
    text = user_input.strip()

    # The material is in the message: nothing to fetch.
    if split_passage(text)[1] is not None:
        return None

    m = URL_RE.search(text)
    if m:
        return "read_url", m.group(0)

    # "あなたのベースAIは?" is about Rula, not a topic to research. Searching the web
    # for it returns articles about AI in general and buries the actual answer.
    if _ABOUT_SELF.search(text):
        return None

    # "クリスマスまであと何日?" / "3日後は何曜日?" / "今日は何曜日?" — counted in
    # code, not by the model and not from a search result. See tools/dates.py.
    # Before _CLOCK_Q, which would otherwise leave today's weekday to the model.
    from .tools.dates import wants_dates
    if wants_dates(text):
        return "date", text

    if _CLOCK_Q.search(text):
        return None

    expr = _arithmetic(text)
    if expr:
        return "calc", expr

    # Worked out, not searched: see wants_reasoning.
    if wants_reasoning(text):
        return None

    wants_depth = bool(_WANTS_DEPTH.search(text))
    if _SEARCH_VERBS.search(text) or _FRESH_TOPICS.search(text):
        # "調べて考えて" wants a considered answer, so read the pages, not just the
        # search snippets.
        return ("deep_search" if wants_depth else "search"), _clean_query(text)

    # "Xについて解説して" asks about a topic without saying "look it up". A 4B model's
    # own knowledge is thin, so sources are what make the answer worth reading.
    if wants_depth and _ABOUT_TOPIC.search(text):
        return "deep_search", _clean_query(text)

    # "Xとは" — definition question. The model's own recall is thin and often wrong.
    if ((_DEFINITION_Q.search(text) or _MEANING_Q.search(text))
            and not _SELF_CONTAINED.search(text)):
        return "search", _clean_query(text)

    # "する方法" / "のやり方" — how-to questions need comprehensive source material.
    if _HOW_TO.search(text) and not _SELF_CONTAINED.search(text):
        return "deep_search", _clean_query(text)

    # "AとBの違い" — comparison needs facts about both subjects.
    if _COMPARISON.search(text) and not _SELF_CONTAINED.search(text):
        return "smart_search", _clean_query(text)

    # "なぜXなのか" — reasoning questions need factual grounding.
    if _REASON_Q.search(text) and not _SELF_CONTAINED.search(text):
        return "search", _clean_query(text)

    # English questions — small model's English knowledge is thin.
    if _ENGLISH_SEARCH.search(text) and not _SELF_CONTAINED.search(text):
        return "search", _clean_query(text)

    # A checkable fact about the outside world: look it up rather than trust recall.
    if _ATTRIBUTE_Q.search(text) and not _YOU.search(text):
        return "search", _clean_query(text)
    # Not about the user: "私の名前は何でしたっけ?" was sent to a search engine.
    # The answer is in the conversation, and the question is theirs.
    if (_FACT_QUESTION.search(text) and _IS_QUESTION.search(text)
            and not _SELF_CONTAINED.search(text) and not _FIRST_PERSON.search(text)
            and not _DEICTIC.match(text)):
        return "search", _clean_query(text)

    return None


def _arithmetic(text: str) -> str | None:
    """The expression to hand to calc, if the message is asking for a sum."""
    m = _ARITH_EXPR.search(text)
    if not m:
        return None
    expr = m.group().strip()
    bare = text.strip().rstrip("=＝?？ ") == expr
    if not (bare or _ARITH_OP.search(expr) or not _date_or_phone(expr)):
        return None
    if not (bare or _CALC_CUE.search(text)):
        return None
    # Thousands separators: "1,980" means 1980, not a tuple.
    return re.sub(r"(?<=\d)[,，](?=\d{3})", "", expr)


def _date_or_phone(expr: str) -> bool:
    """Whether numbers joined by / or - read naturally as a date or a phone number:
    10/9, 2026/10, 2026-10-09, 03-1234-5678. 100/3 and 100-37 do not."""
    parts = [p.strip() for p in re.split(r"[-/－／]", expr.translate(_FULLWIDTH_DIGITS))]
    if len(parts) >= 3 or any(len(p) > 1 and p.startswith("0") for p in parts):
        return True
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        a, b = int(parts[0]), int(parts[1])
        return (1 <= a <= 12 and 1 <= b <= 31) or (1000 <= a <= 2999 and 1 <= b <= 12)
    return False


_FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def _clean_query(text: str) -> str:
    """Strip the request framing so the query is just the topic."""
    q = re.sub(r"(考えて|考察して|分析して|比較して|要約して|詳しく|詳細に|まとめて|解説して|説明して|"
               r"評価して|どう思う)(ください|くれる|ね|よ)?[。、.!！?？]*", " ", text)
    q = re.sub(r"(について)?(検索して|調べて|ググって|ぐぐって|探して|調査して|確認して)"
               r"(ください|くれる|くれない|くれませんか|ほしい|欲しい)?[。、.!！?？]*", " ", q)
    q = re.sub(r"(教えて|お願い)(ください|します)?[。、.!！?？]*", " ", q)
    q = re.sub(r"を?(一言で|ひとことで|一行で|簡潔に|手短に|短く|端的に|ざっくり)"
               r"(言うと|言えば|まとめて|説明して)?", " ", q)
    q = re.sub(r"とは[何なに]*(?:です)?か?\s*$", "", q)
    q = re.sub(r"(する|の|な)(方法|やり方|手順|仕方|し方|コツ)\s*$", "", q)
    q = re.sub(r"(の違い|を比較して?|の比較|のどちら|はどっち)[。、.!！?？]*\s*$", "", q)
    q = re.sub(r"\s+", " ", q).strip(" 、。.!！?？")
    return q or text
