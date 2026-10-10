"""Question intelligence and answer quality control.

A 4B model needs more external help than a large one: it cannot classify
questions internally, cannot self-correct reliably, and hallucinates facts
at a high rate.  This module compensates by doing the classification,
quality scoring, and context enrichment in deterministic code — fast and
free, running before and after every turn.
"""
import re


# ---------------------------------------------------------------------------
# 1. Question type classification → dynamic answer guidance
# ---------------------------------------------------------------------------

_TYPE_PATTERNS = [
    # Before "explanation": "確信犯の本来の意味" was answered with the misuse
    # ("悪いと知っているのに") folded into the definition.
    ("meaning", re.compile(r"(本来の意味|正しい意味|本当の意味|の意味は|の意味って|誤用)")),
    ("comparison", re.compile(
        r"(比較|違い|差|どっち|どちら|メリット|デメリット|長所|短所|vs|VS|優劣)")),
    ("howto", re.compile(
        r"(方法|やり方|手順|仕方|し方|コツ|手続き|するには|始め方|使い方|作り方|導入)")),
    ("explanation", re.compile(
        r"(とは|仕組み|原理|原因|構造|メカニズム|本質|概念|定義)")),
    ("list", re.compile(
        r"(一覧|リスト|列挙|おすすめ|ランキング|種類|何がある|どんな.+がある|挙げて)")),
    ("opinion", re.compile(
        r"(どう思う|意見|感想|見解|評価して|良い悪い|べき|どう考える)")),
    ("creative", re.compile(
        r"(書いて|作って|小説|詩|歌詞|物語|ストーリー|シナリオ|台本|セリフ|手紙)")),
]

_TYPE_GUIDANCE = {
    # No worked example in here: one used to be ("式: 2500 × 0.88 = 2200"), and
    # asked a different discount problem the 9B answered "売値は2200円です",
    # citing "指示された計算式".
    "problem":
        "\n回答形式の指示: 計算・文章題です。"
        "最初の一文で答え(数値と単位)だけを述べ、その後に考え方を短く示してください。"
        "最後に必ず「式:」で始まる1行に、答えを出した計算を数字・演算子・= と結果だけで"
        "書いてください(この行は自動で検算されます)。"
        "答えを書く前に、求めた値が問題の条件をすべて満たすか確かめてください。",
    "code":
        "\n回答形式の指示: プログラムの質問です。コードは必ず ``` で囲んだコードブロックに入れ、"
        "言語名を付けてください(```python など)。指定された関数名・引数名はそのまま使います。"
        "方法の一覧ではなく、そのまま動くコードを先に示し、説明は後に短く書きます。",
    # Measured, six replies each: "おはよう" was greeted back 3 times of 6, the
    # others opened "Rula2.0-9bです。"; "おやすみ" got "わかりました。" 4 times;
    # "ただいま" once got "おかえりください".
    "greeting":
        "\n回答形式の指示: 挨拶です。それに合う挨拶(おはよう→おはよう、おやすみ→おやすみ、"
        "ただいま→おかえり)で始めて、1〜2文で短く返してください。"
        "名前やモデル名は、聞かれない限り名乗らないこと。",
    # Measured on unseen words: "辞書にある本来の意味" in this text came back as
    # "辞書にある本来の意味は以下の通りです。", once the dictionary entry was
    # pasted whole, and for さわり (触ること, 聞かせ所, …) the first sense was
    # given instead of the one people misuse.
    "meaning":
        "\n回答形式の指示: 言葉の意味を問う質問です。"
        "最初の一文は、その言葉を主語にして本来の意味を述べる1文にしてください"
        "(前置き・見出し・「以下の通り」から始めない)。参考資料の語義の一覧をそのまま写さないこと。"
        "語義がいくつもある言葉で「本来の意味」「正しい意味」を聞かれたら、"
        "誤用が広まっている語義について答えます。"
        "よく見られる誤用があれば、その後に「誤用:」として分けて書きます。"
        "本来の意味の説明に、誤用の意味を混ぜないこと。",
    "comparison":
        "\n回答形式の指示: 比較の質問です。"
        "対象ごとの特徴を整理し、違いを明確にしてください。"
        "結論を最初に述べ、項目ごとに対比する構成にしてください。",
    "howto":
        "\n回答形式の指示: 手順の質問です。"
        "具体的なステップを番号付きで示してください。"
        "各ステップに「何をするか」と「なぜそうするか」を含めてください。",
    "explanation":
        "\n回答形式の指示: 説明・解説の質問です。"
        "まず一文で結論を述べ、その後に仕組み・理由を段階的に解説してください。"
        "専門用語には平易な言い換えを添えてください。",
    "list":
        "\n回答形式の指示: 一覧・列挙の質問です。"
        "項目を網羅的にリストアップし、それぞれ一言で特徴を添えてください。"
        "可能なら5個以上挙げてください。",
    "opinion":
        "\n回答形式の指示: 意見を求める質問です。"
        "まず自分の見解を述べ、根拠を示してください。"
        "反対意見や別の視点にも触れ、バランスを取ってください。",
    "creative":
        "\n回答形式の指示: 創作の依頼です。"
        "メタ的な説明(「〜を書きます」「以下が〜です」)は不要です。"
        "作品本文のみを、指示に忠実かつ具体的に書いてください。",
}


# A request for code: a code noun with a verb, or a language named with one.
_CODE_Q = re.compile(
    r"(関数|コード|プログラム|スクリプト|クラス|メソッド|正規表現|ワンライナー).{0,30}"
    r"(書いて|書け|作って|作成|実装|直して|修正|教えて|示して|ください)"
    r"|(Python|JavaScript|TypeScript|Java|C\+\+|C#|Rust|Ruby|PHP|SQL|bash|PowerShell|シェル)"
    r"で.{0,60}(書いて|作って|実装|する方法|やり方|教えて)", re.I)

# A request to drop the polite form for this answer: "敬語を使わずに、友達に話す
# ような口調で…提案して" came back "…を3つ提案します。" The tone detector only
# reads whole messages that are nothing but a tone request.
_CASUAL = re.compile(r"敬語を使わず|敬語なし|敬語抜き|タメ口|ため口|友達に話す(よう|みたい)|"
                     r"くだけた(口調|感じ|言い方)|カジュアルに")


def style_request(text: str) -> str | None:
    """A style the message asks for in passing, as a note for this turn."""
    if _CASUAL.search(text):
        return ("\n文体の指示: です・ます調を使わず、友達に話すようなくだけた話し言葉で書く"
                "(「〜だよ」「〜しよう」「〜かな」など)。")
    return None


_CODE_LINE = re.compile(
    r"^(\s+\S|(def|class|import|from|for|while|if|elif|else|try|except|finally|with|return|"
    r"print|async|await|@)\b|#|[A-Za-z_][\w.\[\]]*\s*(=|\(|\+=|-=)|\s*$)")


def fence_code(answer: str, language: str = "python") -> str:
    """Put an unfenced block of code inside a code fence.

    Measured: 6 of 12 code answers came back as bare "def factorial(n):…" lines.
    Rendered as Markdown, a bare block loses its indentation, which in Python is
    the program. Only when the answer has no fence at all, and only a run that
    starts with def/class/import.
    """
    if "```" in answer:
        return answer
    lines = answer.split("\n")
    start = next((i for i, line in enumerate(lines)
                  if re.match(r"(def|class|import|from \S+ import|async def)\b", line)), None)
    if start is None:
        return answer
    end = start + 1
    while end < len(lines) and _CODE_LINE.match(lines[end]) and not (
            re.search(r"[぀-ヿ]", lines[end]) and not lines[end].startswith((" ", "\t", "#"))):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    return "\n".join(lines[:start] + [f"```{language}"] + lines[start:end] + ["```"] + lines[end:])


# A message that is only a greeting.
_GREETING = re.compile(
    r"^\s*(おはよう(ございます)?|こんにちは|こんばんは|おやすみ(なさい)?|ただいま|はじめまして|"
    r"よろしく(お願いします)?|やあ|ハロー|hello|hi|hey)[\s!！。、〜ー~♪]*$", re.I)


def classify_question(text: str) -> tuple[str, str | None]:
    """Return (question_type, guidance_prompt_or_None).

    No guidance when the user asked for brevity: "項目ごとに対比" turned
    "違いを一言で" into a table, overriding what was actually asked.
    """
    from .planner import wants_brief, wants_reasoning
    # Code first: "リストの中で2番目に大きい値を返す関数" read as a "list"
    # question, and the answer listed six approaches without writing one; a
    # gcd function read as a word problem.
    if _CODE_Q.search(text):
        return "code", _TYPE_GUIDANCE["code"]
    # Word problems next: "何歳差" would otherwise read as a comparison and be
    # told to lay its answer out as a side-by-side table.
    if wants_reasoning(text):
        return "problem", _TYPE_GUIDANCE["problem"]
    if _GREETING.match(text):
        return "greeting", _TYPE_GUIDANCE["greeting"]
    for qtype, pattern in _TYPE_PATTERNS:
        if pattern.search(text):
            return qtype, (None if wants_brief(text) else _TYPE_GUIDANCE.get(qtype))
    return "general", None


# ---------------------------------------------------------------------------
# 2. Follow-up context enrichment
# ---------------------------------------------------------------------------

_FOLLOWUP_START = re.compile(
    r"^(それ|これ|あれ|その|この|あの|上記|前の|さっきの|先ほどの)")
_FOLLOWUP_MORE = re.compile(
    r"^(もっと|さらに|加えて|続き|もう少し|他に|それ以外|ほかに|深く)")

_TOPIC_RE = re.compile(r"(.{2,25})(?:について|に関して|とは)")
_TOPIC_RE2 = re.compile(r"(.{2,25})(?:の方法|のやり方|の手順|の違い|の比較)")


def detect_followup(text: str) -> bool:
    """Whether the message is a vague follow-up that needs context."""
    text = text.strip()
    if len(text) > 80:
        return False
    return bool(_FOLLOWUP_START.search(text) or _FOLLOWUP_MORE.search(text))


def enrich_followup(text: str, prev_user: str, prev_answer: str) -> str:
    """Rewrite a vague follow-up by injecting the topic from the previous turn."""
    # The subject of "Xの標高は?" is X. Taken whole, "富士山の標高は" became the
    # topic, and "それより高い山は日本にある?" was left with its "それ".
    from .tools.search import _subject
    subject = _subject(prev_user)
    whole = re.sub(r"[?？。!！\s]+$", "", prev_user.strip())
    # Only when a subject was actually cut out of it; "りんごを3個買いました。"
    # has none, and is left to the older extraction.
    topic = subject if 2 <= len(subject) <= 20 and len(subject) < len(whole) else None
    topic = topic or _extract_topic(prev_user)
    if not topic:
        sentences = re.split(r"[。.!！\n]", prev_answer.strip())
        first = next((s for s in sentences if len(s.strip()) > 5), None)
        if first:
            topic = first.strip()[:30]
    if not topic:
        return text

    enriched = re.sub(
        r"^(それ|これ|あれ)(?=に|を|の|は|が|も|で|と|について|より|以上|以外|ほど|くらい|ぐらい)",
        f"「{topic}」", text)
    if enriched != text:
        return enriched

    # "もっと詳しく" asks about the topic; "さらに5個買って、2個食べました。今何個
    # ある?" carries on a story, and became "「りんごを3個買いました」について
    # さらに…" in the history.
    if _FOLLOWUP_MORE.match(text) and len(text) <= 20 and not re.search(r"\d", text):
        return f"「{topic}」について{text}"
    return text


def _extract_topic(text: str) -> str | None:
    m = _TOPIC_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _TOPIC_RE2.search(text)
    if m:
        return m.group(1).strip()
    cleaned = re.sub(
        r"(教えて|調べて|検索して|まとめて|解説して|説明して|比較して).*", "", text)
    cleaned = cleaned.strip("？?。、! \n")
    return cleaned[:30] if 2 <= len(cleaned) <= 30 else None


# ---------------------------------------------------------------------------
# 3. Answer quality scoring
# ---------------------------------------------------------------------------

_HEDGE_RE = re.compile(
    r"(かもしれません|わかりません|不明です|確認できません|定かではない|"
    r"把握していません|存じません|記憶にありません)")
_REPETITION_RE = re.compile(r"(.{12,})\1")


def score_answer(answer: str, had_search: bool) -> tuple[float, list[str]]:
    """Score answer quality 0.0–1.0.  Returns (score, issues)."""
    issues: list[str] = []
    scores: list[float] = []

    length = len(answer)
    if length < 15:
        issues.append("too_short")
        scores.append(0.1)
    elif length < 40:
        scores.append(0.5)
    else:
        scores.append(1.0)

    if _REPETITION_RE.search(answer):
        issues.append("repetitive")
        scores.append(0.3)
    else:
        sentences = [s.strip() for s in re.split(r"[。.!！\n]", answer)
                     if len(s.strip()) > 5]
        if len(sentences) > 2:
            ratio = len(set(sentences)) / len(sentences)
            if ratio < 0.7:
                issues.append("repetitive")
            scores.append(max(ratio, 0.3))
        else:
            scores.append(1.0)

    hedges = len(_HEDGE_RE.findall(answer))
    if hedges >= 3:
        issues.append("too_uncertain")
        scores.append(0.3)
    elif hedges >= 2 and not had_search:
        issues.append("unsupported_uncertainty")
        scores.append(0.5)
    else:
        scores.append(1.0)

    if had_search and not re.search(r"https?://", answer):
        issues.append("missing_citation")
        scores.append(0.7)

    overall = sum(scores) / len(scores) if scores else 0.5
    return overall, issues


# ---------------------------------------------------------------------------
# 4. Auto-verification trigger
# ---------------------------------------------------------------------------

_FACTUAL_CLAIM = re.compile(
    r"(\d{4}年|\d+[万億兆]|[0-9]+%|第[0-9一二三四五六七八九十]+位|"
    r"世界[一最]|日本[一最]|約[0-9]+|[0-9]+メートル|[0-9]+キロ|"
    r"[0-9]+人|[0-9]+か国)")


def needs_verification(answer: str, had_search: bool) -> bool:
    """Whether the answer's factual claims should be checked with a search."""
    if had_search:
        return False
    return len(_FACTUAL_CLAIM.findall(answer)) >= 2


# ---------------------------------------------------------------------------
# 4b. Arithmetic check
# ---------------------------------------------------------------------------

# "2500 × 0.88 = 2168" — a written-out calculation and the value it claims. Units
# may follow a number ("60km/h × 2h = 120km"); they are dropped before evaluating.
_NUM = r"\d[\d,]*(?:\.\d+)?"
_UNIT = r"(?:\s*(?:km/h|km|m|cm|kg|g|円|個|本|匹|枚|人|回|分|秒|時間|h|%|％|度|°)?)"
# The claimed value must end the chain: in "2500 × (1 - 0.12) = 2500 × 0.88 = 2168"
# the "2500" after the first "=" is the start of the next step, not a result.
# Operands may open and close brackets: "(28 - 2 × 10) ÷ (4 - 2) = 4" is how the
# model writes most word-problem lines, and requiring a bare number first left
# them unchecked. Unbalanced brackets ("3) × 2 = 20" out of "(x + 3) × 2") fail
# to evaluate and are skipped.
_OPERAND = rf"(?:[(（]\s*)*{_NUM}{_UNIT}(?:\s*[)）])*"
# Powers too: "7 / (2^3) = 0.421875" went unchecked (it is 0.875).
_OP = r"(?:\*\*|\^|[×x\*÷/＋+−\-])"
_CALC_CLAIM = re.compile(
    rf"(?<![\d.])({_OPERAND}(?:\s*{_OP}\s*{_OPERAND})+)\s*[=＝]\s*({_NUM})"
    rf"(?!\s*{_OP}\s*[(（]?\s*\d)(?![\d.])")


def _plain(text: str) -> str:
    return text.replace("\\times", "×").replace("\\div", "÷").replace("$", "")


def _evaluate(expr_text: str) -> float | None:
    """A written-out expression's value, units dropped; None if it is not one."""
    from .tools.calc import _eval_node, _normalize
    import ast
    # Units out ("60km/h × 2h" -> "60 × 2"); % stays, calc reads 88% as 0.88.
    expr = re.sub(r"km/h|km|cm|kg|円|個|本|匹|枚|人|回|時間|分|秒|°|度", "", expr_text)
    expr = re.sub(r"(?<=\d)\s*[mgh](?![a-z])", "", expr)
    expr = (expr.replace(",", "").replace("x", "×").replace("−", "-")
            .replace("＋", "+").replace("％", "%").replace("（", "(").replace("）", ")"))
    try:
        return float(_eval_node(ast.parse(_normalize(expr), mode="eval")))
    except Exception:
        return None


def _wrong_claims(text: str):
    """(match, claimed as written, claimed, actual) for each wrong calculation."""
    for m in _CALC_CLAIM.finditer(text):
        claimed_text = m.group(2)
        actual = _evaluate(m.group(1))
        if actual is None:
            continue
        claimed = float(claimed_text.replace(",", ""))
        if abs(actual - claimed) > max(0.01, abs(actual) * 0.005):
            yield m, claimed_text, claimed, actual


def check_arithmetic(answer: str) -> list[tuple[str, str, float, float]]:
    """Written-out calculations in the answer whose result is wrong.

    Measured: the 9B set up "12%引き" correctly (pay 88%) and then wrote
    2500 × 0.88 = 2168; asked five times it answered 2364, 2460, 2200, 2200,
    2300. Reasoning does not catch slips like that; recomputing the line does.
    Returns (expression, claimed as written, claimed, actual) for each mismatch.
    """
    return [(m.group(1).strip(), ct, c, a) for m, ct, c, a in _wrong_claims(_plain(answer))]


def _format_like(value: float, original: str) -> str:
    """value written the way the answer wrote the number it replaces."""
    decimals = len(original.split(".")[1]) if "." in original else 0
    if abs(value - round(value)) < 1e-9 and decimals == 0:
        text = str(int(round(value)))
    else:
        text = f"{value:.{max(decimals, 2)}f}".rstrip("0").rstrip(".")
    if "," in original:
        head, _, tail = text.partition(".")
        text = f"{int(head):,}" + (f".{tail}" if tail else "")
    return text


def _first_sentence_end(text: str) -> int:
    m = re.search(r"[。\n]", text.lstrip())
    return (len(text) - len(text.lstrip())) + (m.start() + 1 if m else len(text))


def correct_arithmetic(answer: str, rounds: int = 6, given=()) -> str:
    """Put the right values in place of wrong ones, everywhere they appear.

    Done in code rather than by asking the model to rewrite: the right value is
    known exactly, and a rewrite costs another generation by the same model that
    made the slip. A step that used the wrong value is then wrong itself and is
    caught on the next pass, so the fix carries through a chain of steps to the
    final answer.

    Where the wrong value appears matters. In the first sentence it is the
    stated answer, and after the step it is a later step using it; both are to
    be corrected. Before the step (outside the first sentence) it is an input
    ("2500 × 0.88 = 2500", where 2500 is the price): that one is left alone, as
    replacing it would rewrite the price too.

    Nor is a number from the question (`given`) ever replaced. A line claiming
    one is a check against the problem ("4 × 5 + 2 × 5 = 28" for 28 legs);
    "correcting" it to 30 hid that the answer fails it, and a wrong answer
    passed as checked.
    """
    if not check_arithmetic(answer):
        return answer
    given = set(given)
    text = _plain(answer)
    for _ in range(rounds):
        fixed = False
        head = _first_sentence_end(text)
        for m, claimed_text, claimed, actual in _wrong_claims(text):
            if claimed in given:
                continue
            forms = sorted({claimed_text, claimed_text.replace(",", "")}, key=len, reverse=True)
            pattern = r"(?<![\d.,])(?:" + "|".join(map(re.escape, forms)) + r")(?![\d]|\.\d)"
            if any(head <= o.start() < m.start(2) for o in re.finditer(pattern, text)):
                continue
            text = re.sub(pattern, _format_like(actual, claimed_text), text)
            fixed = True
            break  # positions have moved; look again
        if not fixed:
            break
    return text


_NUM_TOKEN = re.compile(rf"(?<![\d.,]){_NUM}")
# A formula line that is only an expression, maybe after "calc:", maybe with a
# dangling "=". Anything else on the line (a letter, a word) and it is not one.
_BARE_FORMULA = re.compile(
    rf"^\s*(?:calc\s*[:：]\s*)?({_OPERAND}(?:\s*{_OP}\s*{_OPERAND})+)\s*[=＝]?\s*$")


def _value(text: str) -> float:
    return float(text.replace(",", ""))


def given_numbers(question: str) -> set[float]:
    """The numbers a question states."""
    return {_value(n) for n in _NUM_TOKEN.findall(question)}


# Simplified-Chinese characters that Japanese does not use. Conservative: forms
# Japanese shares (条 数 写 没 后 准 点 会 学 号 …) are left out, so ordinary
# Japanese never counts. Measured: 2 answers in 570 slipped into Chinese, one a
# whole paragraph ("整点时分针固定在12点位置…" under "180度").
# Measured again with this set: 4 answers in 570, the rest "短针（时针）",
# "最深处", "总价" — single words in an otherwise Japanese answer.
_SIMPLIFIED, _JAPANESE = (
    "说时个还对为针应钟间问题见过关发经现进动东车长门开给头从样书记边处读买卖"
    "钱专业产员亲认识设计该续总标层转达运选递构织结缩线约级纪纳纸红绍终绝统继绪图种类网话气电队价每",
    "説時個還対為針応鐘間問題見過関発経現進動東車長門開給頭従様書記辺処読買売"
    "銭専業産員親認識設計該続総標層転達運選逓構織結縮線約級紀納紙紅紹終絶統継緒図種類網話気電隊価毎")
_TO_JAPANESE = str.maketrans(_SIMPLIFIED, _JAPANESE)
# Grammar words: no kanji stands in for them, the sentence itself is Chinese.
_CHINESE_GRAMMAR = frozenset("们这么吗呢")
_SIMPLIFIED_ONLY = frozenset(_SIMPLIFIED) | _CHINESE_GRAMMAR
# The user asked about Chinese, so Chinese in the answer is meant.
_ABOUT_CHINESE = re.compile(r"中国語|中文|簡体|繁体|漢語|ピンイン|北京語|普通話|台湾華語")


def chinese_leak(question: str, answer: str) -> int:
    """How many Chinese-only characters slipped into a Japanese answer (0 when
    the question itself is about or in Chinese)."""
    if _ABOUT_CHINESE.search(question) or any(c in _SIMPLIFIED_ONLY for c in question):
        return 0
    return sum(c in _SIMPLIFIED_ONLY for c in answer)


def needs_japanese_rewrite(question: str, answer: str) -> bool:
    """Chinese beyond a stray word or two: a sentence of it, or its grammar."""
    leak = chinese_leak(question, answer)
    return leak >= 3 or (leak > 0 and any(c in _CHINESE_GRAMMAR for c in answer))


# Chinese words no single character swap fixes. Measured: "…過程に我们はいる"
# left after the translation pass, and "斐波那契数列" for フィボナッチ数列.
_CHINESE_WORDS = {
    "斐波那契": "フィボナッチ", "我们": "私たち", "你们": "あなたたち", "他们": "彼ら",
    "她们": "彼女ら", "这个": "この", "那个": "あの", "这些": "これら", "那些": "それら",
    "为什么": "なぜ", "什么": "何", "怎么": "どう", "没有": "ない",
}


def japanese_kanji(question: str, answer: str) -> str:
    """A stray simplified character or two put in their Japanese form: 时针 →
    時針, 最深处 → 最深処, 总价 → 総価; and known Chinese words in Japanese.
    Done in code; a rewrite of the whole answer for one word costs a full
    generation."""
    if _ABOUT_CHINESE.search(question) or any(c in _SIMPLIFIED_ONLY for c in question):
        return answer
    for word, japanese in _CHINESE_WORDS.items():
        answer = answer.replace(word, japanese)
    return answer.translate(_TO_JAPANESE) if chinese_leak(question, answer) else answer


_ITEM = re.compile(r"^ ?(?:[-*・•●◦▪]|\d+\s*[.)．、:]|[①-⑳]|\(\d+\))\s*\S")
_KANSUJI = "一二三四五六七八九十"
_COUNT_ASKED = re.compile(
    rf"(\d+|[{_KANSUJI}])\s*(?:つ|個|案|点|項目|選)(?:の\S{{0,8}})?\s*"
    r"(?:を|に|で|だけ)?\s*(?:挙げ|書|教え|提案|考え|まとめ|紹介|出し|示し|選ん|リスト)")


def items_asked(text: str) -> int | None:
    """The number of items a request asks for: "箇条書きで3つ挙げて" -> 3."""
    found = {m.group(1) for m in _COUNT_ASKED.finditer(text)}
    if len(found) != 1:
        return None
    n = found.pop()
    return int(n) if n.isdigit() else _KANSUJI.index(n) + 1


def count_items(answer: str) -> int:
    """Top-level list items in an answer (indented sub-items not counted)."""
    return sum(1 for line in _plain(answer).replace("**", "").splitlines() if _ITEM.match(line))


_FRACTION_JA = re.compile(r"(\d+)\s*分の\s*(\d+)")
_FRACTION = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])")
_PERCENT = re.compile(rf"(?<![\d.,])({_NUM})\s*(?:%|％|パーセント)")


def stated_value(question: str, answer: str) -> tuple[float, bool] | None:
    """The one value the answer's first sentence states: (value, is_percent).

    Fractions count as one value ("1/6", "6分の1"), and "3.5日（7/2日）" is one
    value written twice. Numbers given in the question are not the answer.
    None when there is no such single value.
    """
    head = _plain(answer).replace("**", "")
    head = re.sub(r"\\d?frac\{(\d+)\}\{(\d+)\}", r"\1/\2", head)
    head = head[:_first_sentence_end(head)]
    head = re.sub(r"(?<=\d),(?=\d{3})", "", head)
    given = given_numbers(question)
    values: list[tuple[float, bool]] = []
    for m in _FRACTION_JA.finditer(head):
        values.append((float(m.group(2)) / float(m.group(1)), False))
    head = _FRACTION_JA.sub(" ", head)
    for m in _FRACTION.finditer(head):
        if float(m.group(2)):
            values.append((float(m.group(1)) / float(m.group(2)), False))
    head = _FRACTION.sub(" ", head)
    for m in _PERCENT.finditer(head):
        values.append((_value(m.group(1)), True))
    head = _PERCENT.sub(" ", head)
    values += [(_value(n), False) for n in _NUM_TOKEN.findall(head) if _value(n) not in given]
    # One value written twice is one value: "171.43 ページ（正確には 1200/7）".
    distinct: list[tuple[float, bool]] = []
    for v, p in values:
        if not any(p == q and abs(v - w) <= max(1e-9, abs(w) * 0.005) for w, q in distinct):
            distinct.append((v, p))
    return distinct[0] if len(distinct) == 1 else None


def agrees(computed: float, stated: tuple[float, bool]) -> bool:
    """Whether a computed answer and a stated one are the same value, allowing
    for rounding in what was written and for a percentage given as a ratio."""
    value, percent = stated
    decimals = len(repr(value).split(".")[1].rstrip("0")) if "." in repr(value) else 0
    slack = max(1e-9, 0.5 * 10 ** -decimals if decimals else 1e-9, abs(computed) * 1e-9)
    candidates = [computed, computed * 100] if percent else [computed]
    if not percent and 0 < computed < 1:
        candidates.append(computed * 100)  # "87.5" for 0.875 without the % sign
    return any(abs(value - c) <= max(slack, abs(c) * 0.005) for c in candidates)


def drop_echoed_question(question: str, answer: str) -> str:
    """Remove a first line that only repeats the question.

    Measured: "弟は何歳ですか？" as the first line, the answer on the lines below.
    Only a short line ending in a question mark whose words are all in the
    question goes; anything that adds to it stays.
    """
    head, sep, rest = answer.lstrip().partition("\n")
    first = head.strip().strip("*#> ").strip()
    if not sep or not rest.strip() or len(first) > 60 or not re.search(r"[?？]$", first):
        return answer
    core = re.sub(r"(ですか|でしょうか|か)?[?？]$", "", first).replace(" ", "")
    if len(core) >= 3 and core in re.sub(r"\s", "", question):
        return rest.lstrip()
    return answer


_FORMULA_LINE = re.compile(r"^(\s*(?:[-*・]\s*)?(?:\*\*)?式(?:\*\*)?\s*[:：]\s*)(.*)$", re.M)


def normalize_formula_lines(answer: str) -> str:
    """Put a "式:" line into the "expression = result" form the checks read.

    Measured: "式: `calc: 3000*(1-0.15)` -> 結果: 2955" under "2955 円" — a
    tool result made up in the text (3000 × 0.85 is 2550). With backticks,
    "calc:" and an arrow, no check recognised it as a calculation. Only 式 lines
    are touched: an arrow elsewhere ("10/21→10/28") is not an equals sign.
    """
    def fix(m):
        body = m.group(2).replace("`", "")
        body = re.sub(r"\b(?:calc|計算)\s*[:：]\s*", "", body)
        body = re.sub(r"\s*(?:->|→|=>)\s*(?:結果\s*[:：]\s*)?", " = ", body)
        return m.group(1) + body
    return _FORMULA_LINE.sub(fix, answer)


_WEEKDAY = re.compile(r"([月火水木金土日])曜")


# A length to write to, not a number of characters talked about ("常用漢字2136字
# について"): it carries 以内/程度/… or "で".
_LENGTH_ASKED = re.compile(r"(\d+)\s*(?:字|文字)\s*(以内|以下|まで|程度|くらい|ぐらい|前後|で)")


def length_asked(text: str) -> tuple[int, str] | None:
    """(n, "max" or "about") for "50字以内" / "200字程度"; None if not asked."""
    found = _LENGTH_ASKED.findall(text)
    if len(found) != 1:
        return None
    n, how = int(found[0][0]), found[0][1]
    if n < 10:
        return None
    return n, ("max" if how in ("以内", "以下", "まで") else "about")


def text_length(answer: str) -> int:
    """Characters as a reader counts them: no spaces, line breaks or Markdown marks."""
    return len(re.sub(r"[\s#*>|`]", "", answer))


def length_ok(answer: str, asked: tuple[int, str]) -> bool:
    n, how = asked
    size = text_length(answer)
    return size <= n if how == "max" else 0.7 * n <= size <= 1.4 * n


def trim_to_length(answer: str, n: int) -> str | None:
    """Cut back at a sentence end to fit n characters, or None if that would
    leave less than 40% of n."""
    best = None
    for m in re.finditer(r"[。！？!?]", answer):
        if text_length(answer[:m.end()]) <= n:
            best = m.end()
    if best is None or text_length(answer[:best]) < n * 0.4:
        return None
    return answer[:best]


def match_tool_weekday(answer: str, date_result: str) -> str:
    """The first sentence's weekday made the date tool's.

    Measured: the tool said "1日が月曜日の場合: 15日は月曜日" and the answer
    said 水曜日, 火曜日 or 木曜日 four times in six. The tool's weekday is
    computed; the answer's is the model's own count. The tool's answer is the
    weekday on its last line that has one, today's line aside when there are
    others ("対象の日: …(金曜日)" then "あと76日").
    """
    lines = [line for line in date_result.strip().splitlines() if _WEEKDAY.search(line)]
    if len(lines) > 1:
        lines = [line for line in lines if not line.startswith("今日")] or lines
    if not lines:
        return answer
    right = _WEEKDAY.findall(lines[-1])[-1]
    end = _first_sentence_end(answer)
    head = answer[:end]
    stated = _WEEKDAY.findall(head)
    if len(set(stated)) != 1 or stated[0] == right:
        return answer
    return _WEEKDAY.sub(right + "曜", head) + answer[end:]


def failed_checks(question: str, answer: str) -> list[tuple[str, str, float, float]]:
    """Checks against the question that do not come out: the answer is wrong.

    Measured: "亀は 3 匹です" closing with "式: 4 * 3 + 2 * 7 = 28". The line
    reproduces the question's 28 legs, but it is 26. The slip is not in the
    arithmetic but in the answer, so correcting the 28 to 26 would only make
    the answer agree with its own mistake; the problem has to be solved again.
    """
    given = given_numbers(question)
    return [w for w in check_arithmetic(answer) if w[2] in given]


def reconcile_answer(question: str, answer: str) -> str:
    """Make the first sentence agree with the answer's own formula line.

    Measured: asked 1 + … + 100, the 9B wrote "2550 です" (50 × 51) and closed
    with "式: 50 * 101 = 5050". Every calculation in it was right, so the
    arithmetic check passes; the stated answer is simply not the one worked out.

    The formula's result replaces the stated number only when the mismatch is
    unambiguous. Measured over 40 saved word-problem answers, 39 formula lines
    ended in the answer, and the one that did not was a check: "7" stated, then
    "(7 + 3) × 2 = 20", reproducing the question's 20. Hence:
    - the formula line computes correctly (or is a bare expression, whose
      value is then its result);
    - its result is not a number from the question (that marks a check);
    - the first sentence states exactly one number not in the question;
    - that number appears nowhere else in the answer: the working does not
      support it.
    """
    text = _plain(answer)
    marks = list(re.finditer(r"式\s*[:：]", text))
    if not marks:
        return answer
    line = text[marks[-1].end():].split("\n")[0]
    claims = list(_CALC_CLAIM.finditer(line))
    if claims:
        if check_arithmetic(line):
            return answer
        result = _value(claims[-1].group(2))
    else:
        # The expression without its result: "式: calc: 8*(8-1)/2" under "36試合",
        # "式: 50 * (50 + 1) / 2" under "2550 です". Its value is the answer the
        # working arrived at.
        bare = _BARE_FORMULA.match(line)
        result = _evaluate(bare.group(1)) if bare else None
        if result is None:
            return answer
    given = {_value(n) for n in _NUM_TOKEN.findall(question)}
    if result in given:
        return answer
    head = _first_sentence_end(text)
    stated = [m for m in _NUM_TOKEN.finditer(text[:head]) if _value(m.group()) not in given]
    if len({_value(m.group()) for m in stated}) != 1:
        return answer
    value = _value(stated[0].group())
    if abs(value - result) < 1e-9:
        return answer
    if any(_value(n) == value for n in _NUM_TOKEN.findall(text[head:])):
        return answer
    for m in reversed(stated):  # right to left keeps the earlier offsets valid
        text = text[:m.start()] + _format_like(result, m.group()) + text[m.end():]
    return text


# ---------------------------------------------------------------------------
# 5. Multi-query decomposition
# ---------------------------------------------------------------------------

def decompose_query(query: str) -> list[str] | None:
    """Break a comparison or compound question into sub-queries.

    Returns None when the question is simple enough for a single search.
    """
    m = re.match(r"(.{2,20})と(.{2,20})(の違い|を比較|の比較|のどちら|はどっち)",
                 query)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        return [a, b, f"{a} {b} 比較"]

    m = re.match(
        r"(.{2,20})と(.{2,20})(のメリット|のデメリット|の特徴|の長所|の短所)",
        query)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        suffix = m.group(3)
        return [f"{a}{suffix}", f"{b}{suffix}", f"{a} {b}"]

    return None


# ---------------------------------------------------------------------------
# 6. Streaming quality monitor
# ---------------------------------------------------------------------------

class StreamMonitor:
    """Stops an answer that has started looping, while it is still being written.

    A looping answer otherwise runs to the full budget — 768 to 1,024 tokens, 11
    to 15 seconds at the measured 70 tok/s — and all of it after the first
    repetition is waste. The text before the loop is usually a sound answer, so
    `salvage` keeps it rather than paying for a regeneration.

    Loops only. Aborting on refusals was tried and removed: it cannot tell a
    reflexive decline from a HARD_LIMITS refusal, so regenerating after one is a
    way around the limits, and it saved only the ~1 second a refusal takes.

    Called with the accumulated text, matching GenConfig.abort_check.
    """

    _EVERY = 40   # characters between checks
    _MIN_PERIOD = 30  # same floor as the reasoning-loop check in ollama_backend

    def __init__(self):
        self._checked = 0
        self.abort_reason: str | None = None
        self.salvage: str | None = None

    def feed(self, text: str) -> bool:
        """Return True if generation should stop."""
        if len(text) - self._checked < self._EVERY:
            return False
        self._checked = len(text)
        tail = text[-900:]
        for period in range(self._MIN_PERIOD, len(tail) // 3 + 1):
            if (tail[-period:] == tail[-2 * period:-period]
                    == tail[-3 * period:-2 * period]):
                self.abort_reason = "content_loop"
                self.salvage = text[:_end_of_first_copy(text, tail[-period:])]
                return True
        return False


def _end_of_first_copy(text: str, block: str) -> int:
    """Where the text would end with the repeated part said only once.

    The detected block can hold several copies of a shorter unit — a 28-character
    sentence is only caught as a 56-character period — so it is reduced to its
    smallest repeating unit before trailing copies are dropped.
    """
    p = len(block)
    unit = next(q for q in range(1, p + 1)
                if p % q == 0 and block == block[:q] * (p // q))
    cut = len(text)
    while cut >= 2 * unit and text[cut - 2 * unit:cut - unit] == text[cut - unit:cut]:
        cut -= unit
    return cut
