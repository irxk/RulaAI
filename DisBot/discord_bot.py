"""Discord bot for Rula.

Talks to the local API rather than holding its own Engine. The API already
serialises generation behind a lock — one GPU, one answer at a time — and owns
the load/unload dance the vision model needs to fit on an 8GB card. A second
process with its own Engine would race it for VRAM, so the bot is an ordinary
client: it needs the API running and an API key, nothing more.

Mention it, reply to it, or DM it. The answer streams into the message as it is
produced, because with reasoning switched on a turn takes tens of seconds and
nearly all of that is thinking — worth watching, unlike a spinner.
"""
import asyncio
import json
import logging
import os
import re
import sys
import time

import aiohttp
import discord

log = logging.getLogger("rula.discord")

# Defaults only. The environment is read in main(), after .env has been loaded —
# reading it here would be too early, and RULA_API_BASE written in .env would be
# silently ignored in favour of this value.
DEFAULT_API_BASE = "http://127.0.0.1:8000"
DEFAULT_MODEL = "rula-1.2-4b"

# Everything is sent as an embed, so this is the limit that binds: a description
# holds 4096 characters against a plain message's 2000. Embeds also cannot ping
# anyone, whatever the model wrote — a second guard behind AllowedMentions.none().
MAX_EMBED_CHARS = 4096
# Reserve for a code fence reopened across a split (see _balance_fences).
SPLIT_LIMIT = MAX_EMBED_CHARS - 16
# A plain message holds far less. Only used where Embed Links is missing, so the
# fallback splits to fit rather than losing the tail of the answer.
PLAIN_SPLIT_LIMIT = 2000 - 16
# The API's own cap. Trimming here gives a plain answer instead of a 422.
MAX_INPUT_CHARS = 4000
# How often the live message is rewritten. Discord allows roughly five edits per
# five seconds per channel; discord.py waits out the limit rather than failing,
# so editing faster would not error — it would silently fall behind the model.
EDIT_INTERVAL = 1.5
# How much reasoning to show while it is being written. The tail, not the head:
# what is being thought right now is the part that reads as alive. Roomier than
# the plain-text version could afford, now that 4096 characters are available.
THINKING_TAIL = 800

# Left edge of the embed. Colour is the fastest way to tell at a glance whether
# Rula is still working, finished, or broken.
COLOR_ANSWER = 0xE58FB1   # the reply
COLOR_WORKING = 0x9AA0A6  # thinking, or waiting for a turn
COLOR_ERROR = 0xE05252    # something went wrong
# Longer than the API's own 300s generation timeout, so a slow turn ends with the
# API's error rather than the socket giving up first and hiding it.
READ_TIMEOUT = 330
USER_COOLDOWN = 5.0
# Paused before exiting on a startup failure, so a supervisor that restarts on
# exit cannot turn one into a login flood. See _fatal.
RESTART_COOLDOWN = int(os.environ.get("DISCORD_RESTART_COOLDOWN", "30"))

# When DISCORD_THINK=auto, the questions worth spending ~20 extra seconds on:
# ones with a derivation behind the answer. A heuristic, and a cheap one — the
# cost of a wrong guess either way is latency, not a wrong answer, and "on" and
# "off" are there for anyone who would rather not have it guessed at all.
_NEEDS_THOUGHT = re.compile(
    r"[0-9０-９]\s*[+\-*/×÷=]\s*[0-9０-９]"
    r"|(何分|何時間|何日|何個|何人|何円|何倍|何%|何パーセント|いくつ|いくら)"
    r"|(なぜ|なぜか|どうして|理由|根拠)"
    r"|(比較|違い|どっち|どちら|メリット|デメリット|向いてる|選ぶ)"
    r"|(計算|求めて|見積|証明|論理|矛盾|検証|考えて|検討)")


# Shown when the API cannot be reached at all. By far the usual cause is the bot
# running somewhere other than the machine the API is on — a hosting service, a
# VPS — where 127.0.0.1 is that host, not the PC with the GPU on it.
_UNREACHABLE_HINT = (
    "別のホストでBotを動かしている場合、127.0.0.1 はそのホスト自身を指します。\n"
    "  トンネルの公開URLを指定してください: RULA_API_BASE=https://api.example.com\n"
    "  同じPCで動かしている場合は、run_api.ps1 が起動しているか確認してください。")


class ApiError(RuntimeError):
    """A non-200 from the API, kept with its status so it can be explained."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status

    def as_japanese(self) -> str:
        return {
            401: "APIキーが正しくありません。.env の RULA_API_KEY を確認してください。",
            429: "リクエストが多すぎます。少し待ってからもう一度どうぞ。",
            503: "モデルを読み込み中です。少し待ってからもう一度どうぞ。",
            504: "生成が時間切れになりました。もう少し短い質問だと通るかもしれません。",
        }.get(self.status, f"APIがエラーを返しました({self.status})。")


class RulaClient:
    """The OpenAI-compatible endpoint, as an async generator of deltas."""

    def __init__(self, base: str, key: str, model: str = DEFAULT_MODEL):
        self.base = base.rstrip("/")
        self.key = key
        self.model = model
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.key}"},
            # No total timeout: a thinking turn on a busy card legitimately runs
            # for minutes. sock_read is the real guard — it fires when the server
            # has stopped sending, which is the failure worth catching.
            timeout=aiohttp.ClientTimeout(total=None, sock_read=READ_TIMEOUT))
        return self

    async def __aexit__(self, *_exc):
        await self._session.close()

    async def preflight(self) -> dict | None:
        """Check the server and the key before connecting to Discord.

        A rejected key is fatal: it will never start working on its own, and
        every mention would be answered with an error until someone noticed.

        An unreachable API is not fatal. The bot may well be running somewhere
        the API is not — on a hosting service, with the API behind a tunnel on a
        PC that is currently asleep — and that machine will come back. Warn,
        start anyway, and let each turn report the problem if it persists.
        """
        try:
            async with self._session.get(f"{self.base}/health") as resp:
                # content_type=None: an error page from a tunnel or a proxy comes
                # back as HTML, and the ContentTypeError for it says far less than
                # the check below does.
                health = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("APIに接続できません: %s (%s)", self.base, e)
            log.warning("%s", _UNREACHABLE_HINT)
            return None
        except ValueError:
            raise RuntimeError(
                f"{self.base}/health がJSONを返しませんでした。\n"
                "RULA_API_BASE がRula APIを指しているか確認してください。") from None

        if not isinstance(health, dict) or "status" not in health:
            raise RuntimeError(
                f"{self.base} はRula APIではなさそうです(応答: {str(health)[:80]})。")
        if health.get("status") != "ok":
            log.warning("APIがまだモデルを読み込んでいます。復帰すれば答えられるようになります。")
            return None

        try:
            async with self._session.get(f"{self.base}/v1/models") as resp:
                if resp.status == 401:
                    raise RuntimeError(
                        "APIキーが正しくありません。RULA_API_KEY を確認してください。\n"
                        "キーが手元にない場合は python make_api_key.py で再発行できます"
                        "(古いキーは無効になります)。")
                if resp.status != 200:
                    raise RuntimeError(f"APIの確認に失敗しました({resp.status})。")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            log.warning("APIキーを確認できませんでした: %s", e)
            return None
        return health

    async def stream(self, message: str, conversation: str,
                     image: str | None = None, think: bool = False):
        """Yield ("thinking" | "answer", chunk) as the API produces them."""
        content: list[dict] = [{"type": "text", "text": message}]
        if image:
            content.append({"type": "image_url", "image_url": {"url": image}})
        payload = {
            "model": self.model,
            # Only the newest turn: history lives on the server, keyed by `user`.
            "messages": [{"role": "user", "content": content}],
            "user": conversation,
            "stream": True,
            "think": think,
        }
        async with self._session.post(f"{self.base}/v1/chat/completions",
                                      json=payload) as resp:
            if resp.status != 200:
                raise ApiError(resp.status, await _detail(resp))
            async for raw in resp.content:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    delta = (json.loads(data).get("choices")
                             or [{}])[0].get("delta") or {}
                except (ValueError, IndexError, AttributeError):
                    continue  # a malformed frame is not worth losing the turn over
                # Reasoning arrives on its own field, so it can be shown as
                # working rather than pasted into the answer.
                if delta.get("reasoning_content"):
                    yield "thinking", delta["reasoning_content"]
                if delta.get("content"):
                    yield "answer", delta["content"]


async def _detail(resp) -> str:
    try:
        body = await resp.json()
        return str(body.get("detail") or body)[:200]
    except Exception:
        return (await resp.text())[:200]


def split_message(text: str, limit: int = SPLIT_LIMIT) -> list[str]:
    """Break an answer into Discord-sized pieces, preferring paragraph breaks."""
    parts, rest = [], text.strip()
    while len(rest) > limit:
        cut = _cut_point(rest, limit)
        parts.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest:
        parts.append(rest)
    return _balance_fences(parts)


def _cut_point(text: str, limit: int) -> int:
    window = text[:limit]
    for sep in ("\n\n", "\n", "。", "、", " "):
        idx = window.rfind(sep)
        # Not in the opening fifth: cutting there leaves a stub and shoves almost
        # the whole part into the next message.
        if idx > limit // 5:
            return idx + len(sep)
    return limit


def _balance_fences(parts: list[str]) -> list[str]:
    """Close and reopen code fences across a split.

    A fence left hanging at a split point turns the remainder of the answer —
    and whatever anyone posts next — into one long code block.
    """
    out, language = [], ""
    for part in parts:
        if language:
            part = f"```{language}\n{part}"
        fences = re.findall(r"^```(\w*)", part, re.M)
        if len(fences) % 2:
            language = fences[-1] or ""
            part = part.rstrip() + "\n```"
        else:
            language = ""
        out.append(part)
    return out


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _live_view(thinking: str, answer: str) -> tuple[str | None, str, int]:
    """(title, description, colour) for the message while it is being written.

    Returned as plain values rather than an Embed so the caller can tell whether
    anything actually changed since the last edit — two Embeds built from the
    same text are different objects, and rewriting on every token would put the
    bot straight into Discord's edit rate limit.
    """
    if answer:
        # Once the answer starts it replaces the reasoning outright: it is what
        # was actually asked for, and it is seconds from being final.
        return None, _clip(answer, MAX_EMBED_CHARS - 4) + " ▌", COLOR_ANSWER
    if thinking:
        tail = thinking[-THINKING_TAIL:].replace("\n", " ").strip()
        lead = "…" if len(thinking) > THINKING_TAIL else ""
        return "💭 考えています…", f"> {lead}{tail}", COLOR_WORKING
    return "💭 考えています…", "", COLOR_WORKING


def _as_text(embed: discord.Embed) -> str:
    """The same content as a plain message, for channels that refuse embeds."""
    parts = [p for p in (embed.title, embed.description) if p]
    return _clip("\n".join(parts), 2000) or "(空の応答)"


def _embed(description: str, title: str | None = None,
           color: int = COLOR_ANSWER) -> discord.Embed:
    """One place where an Embed is built, so every reply looks the same.

    The description is clipped rather than trusted: Discord rejects the whole
    message over the limit, which would lose the answer instead of trimming it.
    """
    return discord.Embed(title=title, color=color,
                         description=_clip(description, MAX_EMBED_CHARS) or None)


def _first_image(message: discord.Message) -> str | None:
    for attachment in message.attachments:
        if (attachment.content_type or "").startswith("image/"):
            return attachment.url
    return None


class RulaBot(discord.Client):
    def __init__(self, rula: RulaClient, think_mode: str,
                 allowed_guilds: set[int] | None):
        intents = discord.Intents.default()
        # Privileged, and the whole point of the bot: without it every message
        # arrives with empty content and there is nothing to answer.
        intents.message_content = True
        super().__init__(
            intents=intents,
            # The model's output is untrusted text. Without this, an answer that
            # happens to contain @everyone would ping the server.
            allowed_mentions=discord.AllowedMentions.none())
        self.rula = rula
        self.think_mode = think_mode
        self.allowed_guilds = allowed_guilds
        # One turn at a time. The API serialises anyway; queueing here as well is
        # what lets the wait be shown to the person waiting.
        self._turn = asyncio.Semaphore(1)
        # Said once, not once per message, when Embed Links turns out to be missing.
        self._warned_embed = False
        self._dm_cooldown: dict[int, float] = {}
        self._user_cooldown: dict[int, float] = {}

    async def on_ready(self):
        log.info("接続しました: %s (%d サーバー)", self.user, len(self.guilds))
        # Named with their ids, so the right value for DISCORD_GUILDS can simply
        # be copied out of the log instead of hunted for in the Discord client.
        for guild in self.guilds:
            log.info("  参加中: %s (%d)", guild.name, guild.id)
        if self.allowed_guilds is None:
            return
        log.info("許可サーバー: %s", ", ".join(str(g) for g in sorted(self.allowed_guilds)))
        # An allowlist that matches nothing is silent by design: every mention is
        # dropped with no error anywhere. Said out loud here, because otherwise
        # the only symptom is a bot that looks online and ignores everyone.
        if not self.allowed_guilds & {g.id for g in self.guilds}:
            log.warning("DISCORD_GUILDS のIDが参加中のサーバーと一致しません。"
                        "このままでは全てのメンションを無視します。")
            log.warning("上の「参加中」行のIDに直すか、DISCORD_GUILDS を外してください。")

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if message.guild is None:
            await self._decline_dm(message)
            return
        if not self._addressed(message):
            return
        text = self._strip_mention(message)[:MAX_INPUT_CHARS]
        image = _first_image(message)
        if not text and not image:
            await self._say(
                _embed("聞きたいことを続けて書いてください。", "呼びましたか?",
                       COLOR_WORKING), message)
            return
        await self._answer(message, text or "この画像について説明してください。", image)

    def _addressed(self, message: discord.Message) -> bool:
        if message.author.bot:
            return False
        # DMs are ignored outright, without so much as a refusal: anyone who
        # shares a server with the bot can open one, and a reply there is a
        # conversation nobody else in the server can see.
        if message.guild is None:
            return False
        if self.allowed_guilds is not None and message.guild.id not in self.allowed_guilds:
            return False
        # @everyone technically mentions this bot too. Answering it would mean
        # replying to every announcement in the server.
        if message.mention_everyone:
            return False
        if self.user in message.mentions:
            return True
        # A reply to something Rula said continues that exchange; making people
        # re-mention it every turn is friction nobody expects from a chat bot.
        replied = message.reference.resolved if message.reference else None
        return (isinstance(replied, discord.Message)
                and replied.author.id == self.user.id)

    def _strip_mention(self, message: discord.Message) -> str:
        text = re.sub(rf"<@!?{self.user.id}>", " ", message.content)
        return re.sub(r"[ \t]+", " ", text).strip()

    def _wants_thinking(self, text: str) -> bool:
        if self.think_mode == "on":
            return True
        if self.think_mode == "off":
            return False
        return bool(_NEEDS_THOUGHT.search(text))

    async def _answer(self, message: discord.Message, text: str, image: str | None):
        now = time.monotonic()
        uid = message.author.id
        if now - self._user_cooldown.get(uid, 0) < USER_COOLDOWN:
            return
        self._user_cooldown[uid] = now
        queued = self._turn.locked()
        placeholder = await self._say(_embed(
            "", "⏳ 順番待ちです…" if queued else "💭 考えています…", COLOR_WORKING),
            message)
        if placeholder is None:
            return
        async with self._turn:
            if queued:
                await self._edit(placeholder,
                                 _embed("", "💭 考えています…", COLOR_WORKING))
            try:
                async with message.channel.typing():
                    await self._stream_into(placeholder, message, text, image)
            except ApiError as e:
                log.warning("API %s: %s", e.status, e)
                await self._fail(placeholder, e.as_japanese())
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("APIに接続できません: %s", e)
                await self._fail(placeholder,
                                 "APIサーバーに接続できません。起動しているか確認してください。")
            except Exception:
                log.exception("回答に失敗しました")
                await self._fail(placeholder, "回答の生成に失敗しました。")

    async def _fail(self, placeholder: discord.Message, reason: str):
        await self._edit(placeholder, _embed(reason, "⚠️ エラー", COLOR_ERROR))

    async def _decline_dm(self, message: discord.Message):
        now = time.monotonic()
        if now - self._dm_cooldown.get(message.author.id, 0) < 300:
            return
        self._dm_cooldown[message.author.id] = now
        try:
            await message.reply(
                "DMには対応していません。サーバーのチャンネルでメンションしてください。",
                mention_author=False)
        except discord.HTTPException:
            pass

    async def _stream_into(self, placeholder: discord.Message,
                           message: discord.Message, text: str, image: str | None):
        think = self._wants_thinking(text)
        conversation = f"discord-{message.channel.id}"
        started = time.monotonic()
        thinking, answer = [], []
        shown, last_edit = None, 0.0

        async for kind, chunk in self.rula.stream(text, conversation, image, think):
            (thinking if kind == "thinking" else answer).append(chunk)
            now = time.monotonic()
            if now - last_edit < EDIT_INTERVAL:
                continue
            view = _live_view("".join(thinking), "".join(answer))
            if view != shown:
                shown, last_edit = view, now
                title, description, color = view
                await self._edit(placeholder, _embed(description, title, color))

        elapsed = time.monotonic() - started
        await self._finish(placeholder, "".join(answer), "".join(thinking),
                           elapsed, think)
        log.info("answer ch=%s %.1fs think=%s %d字", message.channel.id,
                 elapsed, think, len("".join(answer)))

    async def _finish(self, placeholder: discord.Message, answer: str, thinking: str,
                      elapsed: float = 0, think: bool = False):
        answer = answer.strip()
        if not answer:
            log.warning("空の回答 (thinking=%d字)", len(thinking))
            answer = "うまく答えられませんでした。もう一度聞いてもらえますか。"
        parts = split_message(answer, SPLIT_LIMIT
                              if self._can_embed(placeholder.channel)
                              else PLAIN_SPLIT_LIMIT)
        embed = _embed(parts[0])
        if elapsed > 0:
            footer = f"Rula1.2-4b • {elapsed:.1f}s"
            if think:
                footer += " • 🧠"
            embed.set_footer(text=footer)
        await self._edit(placeholder, embed)
        for part in parts[1:]:
            try:
                await placeholder.channel.send(
                    **self._payload(placeholder.channel, _embed(part)))
            except discord.HTTPException:
                log.warning("続きを送れませんでした")
                return

    def _can_embed(self, channel) -> bool:
        """Whether this channel will accept an embed.

        Checked rather than attempted. Without Embed Links every send comes back
        403, and the bot goes silent with the reason visible only in the log —
        which is exactly how it looks when it is simply broken.
        """
        try:
            return channel.permissions_for(channel.guild.me).embed_links
        except AttributeError:
            return True  # no guild context to judge by; assume it is fine

    def _payload(self, channel, embed: discord.Embed) -> dict:
        """`embed=` where allowed, `content=` where not.

        Both keys are always set, so an edit cannot leave the other one behind.
        """
        if self._can_embed(channel):
            return {"content": None, "embed": embed}
        if not self._warned_embed:
            self._warned_embed = True
            log.warning("このチャンネルで埋め込みを送れません(Embed Links 権限が不足)。"
                        "平文で応答します。")
            log.warning("サーバー設定 → 連携サービス → RulaAI、または"
                        "チャンネルの権限で「埋め込みリンク」を許可してください。")
        return {"content": _as_text(embed), "embed": None}

    async def _say(self, embed: discord.Embed,
                   message: discord.Message) -> discord.Message | None:
        try:
            return await message.reply(
                **self._payload(message.channel, embed), mention_author=False)
        except discord.HTTPException as e:
            # Deleted message, or no permission to post here. Nothing to recover.
            log.warning("返信できませんでした: %s", e)
            return None

    async def _edit(self, placeholder: discord.Message, embed: discord.Embed):
        try:
            await placeholder.edit(**self._payload(placeholder.channel, embed))
        except discord.HTTPException as e:
            log.warning("編集できませんでした: %s", e)


def _load_env():
    """Read .env beside this file.

    Deliberately standalone rather than imported from api.py: that module pulls
    in FastAPI and the engine, which the bot has no use for.

    Reads this folder first, then RulaAI's — which is where the shared .env
    actually lives, holding the API key the bot authenticates with. A DisBot/.env
    is optional and wins where the two overlap, since setdefault keeps whichever
    value was seen first.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in (here, os.path.dirname(here)):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    name, value = line.split("=", 1)
                    os.environ.setdefault(name.strip(), value.strip().strip("\"'"))


def _allowed_guilds() -> set[int] | None:
    raw = os.environ.get("DISCORD_GUILDS", "").strip()
    return {int(x) for x in re.findall(r"\d+", raw)} if raw else None


SETUP_HELP = """\
DISCORD_TOKEN が未設定です。RulaAI\\.env(このフォルダの1つ上)に追記してください。

  1. https://discord.com/developers/applications で New Application
  2. Bot タブ → Reset Token → 出てきた文字列を控える
  3. 同じ Bot タブの Privileged Gateway Intents で
     MESSAGE CONTENT INTENT を ON にする(これが無いと本文を読めません)
  4. .env に  DISCORD_TOKEN=<控えた文字列>
  5. OAuth2 → URL Generator で scopes=bot、権限は
     Send Messages / Read Message History を選び、出たURLでサーバーに招待
"""


async def _run(token: str, key: str, base: str, model: str, think_mode: str,
               guilds: set[int] | None):
    async with RulaClient(base, key, model) as rula:
        log.info("接続先API: %s", base)
        health = await rula.preflight()
        if health:
            log.info("API: %s (%s) thinking=%s think_mode=%s", health.get("weights"),
                     health.get("backend"), health.get("thinking"), think_mode)
        else:
            # Connected to Discord regardless, so the bot is at least present and
            # can say what is wrong when someone asks, rather than being absent.
            log.warning("APIが応答しない状態で起動します(復帰すれば自動的に答えます)。")
        bot = RulaBot(rula, think_mode, guilds)
        try:
            await bot.start(token)
        except discord.PrivilegedIntentsRequired:
            raise RuntimeError(
                "MESSAGE CONTENT INTENT が無効です。Discord Developer Portal の "
                "Bot → Privileged Gateway Intents で有効にしてください。") from None
        except discord.HTTPException as e:
            if e.status != 429:
                raise
            # Cloudflare 1015, not Discord's own per-route limit: the host's IP is
            # shut out of discord.com for logging in too often. Restarting is what
            # causes it and what extends it, so the only move is to stop.
            raise RuntimeError(
                "Discord がこのホストからのログインを一時的に制限しています "
                "(Cloudflare Error 1015)。\n"
                "  短時間に何度も起動したのが原因です。数分〜1時間ほどで解けます。\n"
                "  再起動を繰り返すと制限が延びます。しばらく待ってから起動して"
                "ください。") from None
        except discord.LoginFailure:
            raise RuntimeError("DISCORD_TOKEN が正しくありません。") from None
        finally:
            await bot.close()


def _force_utf8():
    """Make both streams UTF-8, where that is possible at all.

    A Windows console left at cp932 turns every Japanese log line into mojibake,
    and logging writes to stderr rather than stdout, so both need it.

    Guarded because neither stream is guaranteed to be a real file: a hosting
    panel that captures output can substitute an object with no reconfigure() on
    it, and crashing on startup over an encoding preference would be a poor
    trade — the bot runs fine either way.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _fatal(message: str) -> int:
    """Report, wait, then exit non-zero.

    The wait is the point. A hosting service that restarts the process whenever it
    exits turns a startup failure into a login attempt every few seconds, and
    Discord answers a stream of those by locking the whole host out of
    discord.com (Cloudflare Error 1015) — which is how a missing environment
    variable becomes an hour of downtime. It costs nothing when the failure is
    permanent, and breaks the spin when the supervisor is eager.
    """
    print(f"\n{message}")
    print(f"({RESTART_COOLDOWN}秒待ってから終了します。"
          "再起動の連打でDiscordから締め出されるのを防ぐためです)")
    time.sleep(RESTART_COOLDOWN)
    return 1


def main():
    _force_utf8()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _load_env()

    token = os.environ.get("DISCORD_TOKEN", "").strip()
    if not token:
        return _fatal(SETUP_HELP)
    key = os.environ.get("RULA_API_KEY", "").strip()
    if not key:
        return _fatal(
            "RULA_API_KEY が未設定です。RulaAI\\.env に APIキーを書いてください。\n"
            "(.env の SALT / HASH は照合用で、キーそのものではありません。\n"
            " 手元に無ければ python make_api_key.py で再発行できます)")
    think_mode = os.environ.get("DISCORD_THINK", "auto").strip().lower()
    if think_mode not in ("on", "off", "auto"):
        return _fatal(
            f"DISCORD_THINK は on / off / auto のどれかです(受け取った値: {think_mode})")
    # Read here rather than at import, so a value in .env is actually used.
    base = os.environ.get("RULA_API_BASE", DEFAULT_API_BASE).strip() or DEFAULT_API_BASE
    model = os.environ.get("RULA_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    try:
        asyncio.run(_run(token, key, base, model, think_mode, _allowed_guilds()))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        return _fatal(str(e))
    except aiohttp.ClientError as e:
        # Anything the preflight did not already turn into a warning. A bare
        # traceback here says nothing useful about a misconfigured address.
        return _fatal(f"APIに接続できません: {base}\n  {e}\n{_UNREACHABLE_HINT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
