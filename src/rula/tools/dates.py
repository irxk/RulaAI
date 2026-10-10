"""Date arithmetic, done in code.

Asked "クリスマスまであと何日?" on 2026-10-09, the 9B searched, read a countdown
site and answered 78 days; it is 77. (Counted by hand while checking that, it came
out as 76 — also wrong.) Counting days across months is exactly the kind of thing
that comes out nearly right, so the planner routes it here.
"""
import re
from datetime import date, datetime, timedelta

from .base import Tool

_WEEKDAYS = "月火水木金土日"
_NAMED = {
    "クリスマスイブ": (12, 24), "クリスマス・イブ": (12, 24), "イブ": (12, 24),
    "クリスマス": (12, 25), "大晦日": (12, 31), "おおみそか": (12, 31),
    "元日": (1, 1), "元旦": (1, 1), "正月": (1, 1), "お正月": (1, 1), "新年": (1, 1),
    "バレンタイン": (2, 14), "ホワイトデー": (3, 14), "ハロウィン": (10, 31),
    "七夕": (7, 7), "ひな祭り": (3, 3), "こどもの日": (5, 5),
}
_FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")
_YMD = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SLASH = re.compile(r"(?:(\d{4})/)?(\d{1,2})/(\d{1,2})")
_OFFSET = re.compile(r"(\d+)\s*(日|週間|週|か月|ヶ月|カ月|ヵ月)\s*(後|前)")
_RELATIVE = {"明後日": 2, "あさって": 2, "明日": 1, "あした": 1, "今日": 0,
             "昨日": -1, "きのう": -1, "一昨日": -2, "おととい": -2}
# "今日が水曜日だとすると、100日後は何曜日?": a supposed today, not the real one.
# Answered from the real date, the reply was "今日が土曜日なら…月曜日" — right
# arithmetic on the wrong day.
_SUPPOSING = re.compile(r"(だと|なら|とする|とした|と仮定|とすれば)")
_SUPPOSE_DATE = re.compile(
    r"(?:今日|本日)が\s*(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SUPPOSE_WEEKDAY = re.compile(r"(?:今日|本日)が\s*([月火水木金土日])曜")
# "ある月の1日が月曜日のとき、その月の15日は何曜日?": a day of some month, its
# weekday given. Searched, this was answered 水曜日 after a page of second-guessing.
_MONTH_DAY_WEEKDAY = re.compile(
    r"(\d{1,2})\s*日が\s*([月火水木金土日])曜(?:日)?(?:の(?:とき|時|場合)|なら|だと|とすると|とした)"
    r".*?(\d{1,2})\s*日は\s*何曜")


def today() -> date:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("Asia/Tokyo")).date()
    except (ImportError, KeyError):
        return datetime.now().date()


def _fmt(d: date) -> str:
    # The weekday spelled out: given "(土)", the 9B answered "土です。"
    return f"{d.year}年{d.month}月{d.day}日({_WEEKDAYS[d.weekday()]}曜日)"


def _target(text: str, base: date) -> date | None:
    """The date the message refers to, if it names one."""
    for name in sorted(_NAMED, key=len, reverse=True):
        if name in text:
            m, d = _NAMED[name]
            t = date(base.year, m, d)
            return t if t >= base else date(base.year + 1, m, d)
    for pat in (_YMD, _SLASH):
        m = pat.search(text)
        if m:
            y = int(m.group(1)) if m.group(1) else base.year
            try:
                t = date(y, int(m.group(2)), int(m.group(3)))
            except ValueError:
                return None
            if not m.group(1) and t < base and "まで" in text:
                t = date(y + 1, t.month, t.day)  # "12月25日まで" in late December
            return t
    return None


def wants_dates(text: str) -> bool:
    """Whether a message asks for date arithmetic this tool can do."""
    t = text.translate(_FULLWIDTH)
    if _MONTH_DAY_WEEKDAY.search(t):
        return True
    if re.search(r"まで(?:あと)?\s*(?:何日|なんにち|何週間|いくつ)", t) and _target(t, today()):
        return True
    if _OFFSET.search(t) and re.search(r"(何日|何曜|いつ|何月)", t):
        return True
    if any(w in t for w in _RELATIVE if w != "今日") and re.search(r"(何日|何曜|何月)", t):
        return True
    if (_YMD.search(t) or _SLASH.search(t)) and re.search(r"何曜", t):
        return True
    # Today's own date and weekday. They are in the system prompt, but the 9B
    # read "(金)" there and answered 木曜日 — then claimed a tool had checked it.
    if re.search(r"(今日|本日)", t) and re.search(r"(何曜|何日|何月)", t):
        return True
    return False


def _supposed_weekday(t: str, w: int) -> str:
    """Weekday arithmetic from a supposed weekday alone: no real date involved."""
    name = _WEEKDAYS[w] + "曜日"
    m = _OFFSET.search(t)
    if m:
        n, unit, sign = int(m.group(1)), m.group(2), -1 if m.group(3) == "前" else 1
        if unit == "日":
            days = n
        elif unit.startswith("週"):
            days = 7 * n
        else:
            return f"今日を{name}とした場合、月単位の先の曜日は、実際の日付がないと決まりません。"
        after = _WEEKDAYS[(w + sign * days) % 7] + "曜日"
        weeks, rest = divmod(days, 7)
        return (f"今日を{name}とした場合(仮定): {n}{unit}{m.group(3)}は{after}"
                f"({days}日 = {weeks}週と{rest}日)")
    for word, off in sorted(_RELATIVE.items(), key=lambda kv: -len(kv[0])):
        if word in t and word != "今日":
            return f"今日を{name}とした場合(仮定): {word}は{_WEEKDAYS[(w + off) % 7]}曜日"
    return f"今日を{name}とした場合(仮定)。"


def calc_date(text: str) -> str:
    t = text.translate(_FULLWIDTH)
    m = _MONTH_DAY_WEEKDAY.search(t)
    if m:
        day, w, target = int(m.group(1)), _WEEKDAYS.index(m.group(2)), int(m.group(3))
        if not (1 <= day <= 31 and 1 <= target <= 31):
            return "日付を読み取れませんでした。"
        after = _WEEKDAYS[(w + target - day) % 7]
        return (f"{day}日が{m.group(2)}曜日の場合: {target}日は{after}曜日"
                f"({target}日は{day}日の{abs(target - day)}日{'後' if target >= day else '前'})")
    supposing = _SUPPOSING.search(t)
    weekday = _SUPPOSE_WEEKDAY.search(t) if supposing else None
    if weekday:
        return _supposed_weekday(t, _WEEKDAYS.index(weekday.group(1)))
    supposed = _SUPPOSE_DATE.search(t) if supposing else None
    if supposed:
        try:
            base = date(int(supposed.group(1) or today().year), int(supposed.group(2)),
                        int(supposed.group(3)))
        except ValueError:
            return "日付を読み取れませんでした。"
        t = t[:supposed.start()] + t[supposed.end():]  # not the target date as well
        lines = [f"今日(仮定): {_fmt(base)}"]
    else:
        base = today()
        lines = [f"今日: {_fmt(base)}"]
    m = _OFFSET.search(t)
    if m:
        n, unit, sign = int(m.group(1)), m.group(2), -1 if m.group(3) == "前" else 1
        if unit == "日":
            d = base + timedelta(days=sign * n)
        elif unit.startswith("週"):
            d = base + timedelta(weeks=sign * n)
        else:  # months: same day of month, clamped to the month's end
            mo = base.month - 1 + sign * n
            y, mo = base.year + mo // 12, mo % 12 + 1
            last = (date(y + (mo == 12), mo % 12 + 1, 1) - timedelta(days=1)).day
            d = date(y, mo, min(base.day, last))
        lines.append(f"{n}{unit}{m.group(3)}: {_fmt(d)}")
        return "\n".join(lines)
    for word, off in sorted(_RELATIVE.items(), key=lambda kv: -len(kv[0])):
        if word in t and word != "今日":
            lines.append(f"{word}: {_fmt(base + timedelta(days=off))}")
            return "\n".join(lines)
    target = _target(t, base)
    if target is None:
        if re.search(r"(今日|本日)", t):
            return "\n".join(lines)  # the question was about today itself
        return "日付を読み取れませんでした。"
    delta = (target - base).days
    lines.append(f"対象の日: {_fmt(target)}")
    if delta > 0:
        lines.append(f"今日から{delta}日後(あと{delta}日)")
    elif delta < 0:
        lines.append(f"{-delta}日前(すでに過ぎています)")
    else:
        lines.append("今日です")
    return "\n".join(lines)


DATE_TOOL = Tool(
    name="date",
    description="日付の計算をする(〜まであと何日、N日後の日付、ある日の曜日)。数えず必ずこれを使う。",
    usage="date: クリスマスまであと何日",
    run=calc_date,
)
