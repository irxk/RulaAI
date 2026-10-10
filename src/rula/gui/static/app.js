// Rula desktop app — the window's logic. Talks only to its own backend
// (src/rula/gui/server.py) on 127.0.0.1, with the token the launcher put in the URL.
(function () {
  "use strict";

  const TOKEN = new URLSearchParams(location.search).get("t") || "";
  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  };
  const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const state = { chatId: null, busy: false, image: null, thumb: null, think: false, info: null };

  async function api(path, opts = {}) {
    opts.headers = Object.assign({ "X-Rula-Token": TOKEN }, opts.headers || {});
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.status + "";
      try { msg = (await r.json()).detail || msg; } catch (_) { /* not json */ }
      throw new Error(msg);
    }
    return r;
  }
  const post = (path, body) => api(path, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });

  // ---------------------------------------------------------------- status
  async function pollInfo() {
    try {
      const info = await (await api("/api/info")).json();
      state.info = info;
      showInfo(info);
      setTimeout(pollInfo, info.status === "ready" ? 30000 : 1000);
    } catch (e) {
      setTimeout(pollInfo, 2000);
    }
  }

  function showInfo(info) {
    const dot = $("model-dot");
    dot.className = "dot" + (info.status === "loading" ? " loading" : info.status === "error" ? " error" : "");
    if (info.status === "loading") {
      $("model-name").textContent = "モデルを読み込んでいます…";
      $("hint").textContent = "モデルを読み込んでいます (10秒ほど)";
    } else if (info.status === "error") {
      $("model-name").textContent = "読み込みに失敗しました";
      $("hint").textContent = "起動に失敗: " + (info.error || "") + "  (ログ: %APPDATA%\\Rula\\rula.log)";
    } else {
      $("model-name").textContent = info.model + (info.backend ? " · " + info.backend : "");
      if (!state.busy) $("hint").textContent = "";
    }
    const badges = $("badges");
    badges.innerHTML = "";
    if (info.status === "ready") {
      if (info.mtp) badges.append(el("span", "badge", "MTP"));
      if (info.vision) badges.append(el("span", "badge", "画像OK"));
      if (info.vram_free != null && info.vram_free < 400) {
        const b = el("span", "badge warn", "VRAM残り " + info.vram_free + "MB");
        b.title = "空きが少ないと生成が極端に遅くなります。GPUを使う他のアプリを閉じてください。";
        badges.append(b);
      }
    }
    $("attach").disabled = !(info.status === "ready" && info.vision);
    $("attach").title = info.vision ? "画像を添付" : "このモデルでは画像を読めません";
    $("think").disabled = !(info.status === "ready" && info.thinking);
    updateSend();
  }

  // ---------------------------------------------------------------- chat list
  async function refreshList() {
    let chats = [];
    try { chats = await (await api("/api/chats")).json(); } catch (_) { return; }
    const list = $("chat-list");
    list.innerHTML = "";
    const now = Date.now() / 1000;
    let group = null;
    for (const c of chats) {
      const age = now - c.updated;
      const g = age < 86400 ? "今日" : age < 7 * 86400 ? "過去7日間" : "それ以前";
      if (g !== group) { list.append(el("div", "side-group", g)); group = g; }
      const item = el("div", "chat-item" + (c.id === state.chatId ? " active" : ""));
      const t = el("span", "t");
      t.textContent = c.title;
      const del = el("button", "del", "削除");
      del.type = "button";
      del.title = "このチャットを削除";
      del.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (!del.classList.contains("confirm")) {
          del.classList.add("confirm");
          del.textContent = "本当に削除";
          setTimeout(() => { del.classList.remove("confirm"); del.textContent = "削除"; }, 3000);
          return;
        }
        await api("/api/chats/" + c.id, { method: "DELETE" });
        if (c.id === state.chatId) newChat();
        refreshList();
      });
      item.append(t, del);
      item.addEventListener("click", () => openChat(c.id));
      list.append(item);
    }
  }

  function newChat() {
    if (state.busy) return;
    state.chatId = null;
    $("messages").innerHTML = "";
    $("messages").append(emptyState);
    $("chat-title").textContent = "";
    refreshList();
    $("input").focus();
  }

  async function openChat(id) {
    if (state.busy) return;
    let chat;
    try { chat = await (await api("/api/chats/" + id)).json(); } catch (_) { return; }
    state.chatId = id;
    $("chat-title").textContent = chat.title || "";
    const box = $("messages");
    box.innerHTML = "";
    for (const m of chat.messages) {
      if (m.role === "user") addUser(m.content, m.thumb);
      else addAssistant().final(m);
    }
    refreshList();
    scrollToEnd(true);
  }

  // ---------------------------------------------------------------- messages
  const emptyState = $("empty");

  function addUser(text, thumb) {
    emptyState.remove();
    const m = el("div", "msg user");
    if (thumb) {
      const img = el("img", "thumb");
      img.src = thumb;
      m.append(img);
    }
    if (text) {
      const b = el("div", "bubble");
      b.textContent = text;
      m.append(b);
    }
    $("messages").append(m);
    scrollToEnd(true);
  }

  function addAssistant() {
    emptyState.remove();
    const m = el("div", "msg assistant");
    const status = el("div", "status", '<span class="dots"><i></i><i></i><i></i></span><span class="st"></span>');
    const body = el("div", "body");
    m.append(status, body);
    $("messages").append(m);
    let thinkBox = null;
    let pending = null;

    const setThinking = (text, live) => {
      if (!text) return;
      if (!thinkBox) {
        thinkBox = el("details", "thinking");
        thinkBox.innerHTML = "<summary></summary><div class=\"tbody\"></div>";
        m.insertBefore(thinkBox, status.nextSibling);
        if (live) thinkBox.open = true;
      }
      thinkBox.querySelector("summary").textContent =
        (live ? "考えています… " : "思考 ") + "(" + text.length + "字)";
      const tb = thinkBox.querySelector(".tbody");
      tb.textContent = text;
      tb.scrollTop = tb.scrollHeight;
    };

    return {
      status(text) { status.hidden = false; status.querySelector(".st").textContent = text || ""; scrollToEnd(); },
      thinking(text) { setThinking(text, true); scrollToEnd(); },
      stream(text) {
        // Rendered at most once a frame: tokens arrive faster than layout.
        pending = text;
        requestAnimationFrame(() => {
          if (pending == null) return;
          status.hidden = true;
          body.innerHTML = window.renderMarkdown(pending);
          pending = null;
          scrollToEnd();
        });
      },
      final(ev) {
        pending = null;
        status.hidden = true;
        if (ev.thinking) {
          setThinking(ev.thinking, false);
          thinkBox.open = false;
        }
        body.innerHTML = window.renderMarkdown(ev.answer != null ? ev.answer : ev.content || "");
        const meta = el("div", "meta");
        const bits = [];
        if (ev.elapsed != null) bits.push(ev.elapsed + "秒");
        for (const t of ev.tools || []) {
          const label = { search: "検索", deep_search: "検索", smart_search: "検索", read_url: "ページ",
                          calc: "計算", date: "日付計算" }[t.name] || t.name;
          bits.push(label + (/search/.test(t.name) ? ": " + t.arg : ""));
        }
        if (ev.stopped) bits.push("停止しました");
        meta.append(el("span", null, escapeHtml(bits.join(" · "))));
        const copy = el("button", "act", "コピー");
        copy.type = "button";
        copy.addEventListener("click", () => {
          navigator.clipboard.writeText(ev.answer != null ? ev.answer : ev.content || "");
          copy.textContent = "コピーしました";
          setTimeout(() => { copy.textContent = "コピー"; }, 1500);
        });
        meta.append(copy);
        m.append(meta);
        scrollToEnd();
      },
      error(text) {
        status.hidden = true;
        const e = el("div", "error");
        e.textContent = "エラー: " + text;
        body.append(e);
        scrollToEnd();
      },
    };
  }

  function scrollToEnd(force) {
    const s = $("scroller");
    const near = s.scrollHeight - s.scrollTop - s.clientHeight < 160;
    if (force || near) requestAnimationFrame(() => { s.scrollTop = s.scrollHeight; });
  }

  // ---------------------------------------------------------------- sending
  function updateSend() {
    const ready = state.info && state.info.status === "ready";
    const hasInput = $("input").value.trim() || state.image;
    $("send").disabled = !(state.busy || (ready && hasInput));
    $("send").title = state.busy ? "停止" : "送信";
  }

  function setBusy(b) {
    state.busy = b;
    document.body.classList.toggle("busy", b);
    $("new-chat").disabled = b;
    updateSend();
  }

  async function send(textOverride) {
    if (state.busy) return;
    const text = (textOverride != null ? textOverride : $("input").value).trim();
    if (!text && !state.image) return;
    if (!state.info || state.info.status !== "ready") return;

    const image = state.image, thumb = state.thumb;
    addUser(text, thumb);
    const a = addAssistant();
    a.status(image ? "画像を読んでいます" : state.think ? "考えています" : "");
    $("input").value = "";
    autosize();
    clearAttachment();
    setBusy(true);

    let answer = "", thinking = "";
    const handle = (ev) => {
      switch (ev.type) {
        case "chat":
          if (!state.chatId) {
            state.chatId = ev.id;
            $("chat-title").textContent = ev.title || "";
            refreshList();
          }
          break;
        case "status": a.status(ev.text); break;
        case "thinking": thinking += ev.text; a.thinking(thinking); break;
        case "token": answer += ev.text; a.stream(answer); break;
        case "reset": answer = ""; break;
        case "done": a.final(ev); break;
        case "error": a.error(ev.text); break;
      }
    };
    try {
      const r = await post("/api/chat", { chat_id: state.chatId, message: text, image, thumb,
                                          think: state.think });
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let cut;
        while ((cut = buf.indexOf("\n\n")) >= 0) {
          const chunk = buf.slice(0, cut);
          buf = buf.slice(cut + 2);
          const line = chunk.split("\n").find((l) => l.startsWith("data:"));
          if (line) handle(JSON.parse(line.slice(5)));
        }
      }
    } catch (e) {
      a.error(e.message);
    } finally {
      setBusy(false);
      refreshList();
      $("input").focus();
    }
  }

  // ---------------------------------------------------------------- images
  async function attachFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    if (!state.info || !state.info.vision) {
      $("hint").textContent = "このモデルでは画像を読めません";
      return;
    }
    const bmp = await createImageBitmap(file);
    // Resized here: the model reads at most ~1024px anyway, and a 12MP photo as a
    // data URI would be tens of megabytes on the wire for nothing.
    state.image = toDataUrl(bmp, 1024, 0.9);
    state.thumb = toDataUrl(bmp, 320, 0.8);
    $("attach-img").src = state.thumb;
    $("attach-preview").hidden = false;
    updateSend();
    $("input").focus();
  }

  function toDataUrl(bmp, max, quality) {
    const s = Math.min(1, max / Math.max(bmp.width, bmp.height));
    const c = document.createElement("canvas");
    c.width = Math.max(1, Math.round(bmp.width * s));
    c.height = Math.max(1, Math.round(bmp.height * s));
    const g = c.getContext("2d");
    g.fillStyle = "#fff";  // transparent PNGs onto white, as JPEG has no alpha
    g.fillRect(0, 0, c.width, c.height);
    g.drawImage(bmp, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", quality);
  }

  function clearAttachment() {
    state.image = state.thumb = null;
    $("attach-preview").hidden = true;
    $("file").value = "";
    updateSend();
  }

  // ---------------------------------------------------------------- input
  function autosize() {
    const t = $("input");
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 220) + "px";
  }

  $("input").addEventListener("input", () => { autosize(); updateSend(); });
  $("input").addEventListener("keydown", (e) => {
    // isComposing / 229: Enter that confirms a Japanese IME conversion must not
    // send the message.
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
      e.preventDefault();
      send();
    }
  });
  $("input").addEventListener("paste", (e) => {
    const item = [...(e.clipboardData || {}).items || []].find((i) => i.type.startsWith("image/"));
    if (item) { e.preventDefault(); attachFile(item.getAsFile()); }
  });
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    if (state.busy) post("/api/stop").catch(() => {});
    else send();
  });
  $("attach").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => attachFile(e.target.files[0]));
  $("attach-remove").addEventListener("click", clearAttachment);
  $("think").addEventListener("click", () => {
    state.think = !state.think;
    $("think").setAttribute("aria-pressed", String(state.think));
  });
  $("new-chat").addEventListener("click", newChat);
  $("toggle-side").addEventListener("click", () => document.body.classList.toggle("side-closed"));
  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => send(chip.textContent));
  }

  let dragDepth = 0;
  window.addEventListener("dragenter", (e) => {
    if ([...e.dataTransfer.types].includes("Files")) { dragDepth++; $("drop").hidden = false; }
  });
  window.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("drop").hidden = true; } });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    $("drop").hidden = true;
    attachFile(e.dataTransfer.files[0]);
  });

  // Links open in the normal browser; code blocks copy.
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (a) {
      e.preventDefault();
      if (/^https?:\/\//.test(a.getAttribute("href"))) post("/api/open", { url: a.getAttribute("href") });
      return;
    }
    const copy = e.target.closest(".code .copy");
    if (copy) {
      navigator.clipboard.writeText(copy.closest(".code").querySelector("code").textContent);
      copy.textContent = "コピーしました";
      setTimeout(() => { copy.textContent = "コピー"; }, 1500);
    }
  });

  // The backend exits when these stop: closing the window is how the app quits.
  const ping = () => post("/api/ping").catch(() => {});
  ping();
  setInterval(ping, 10000);
  // Closing the window is how the app quits; this is the signal for it.
  window.addEventListener("pagehide", () => {
    navigator.sendBeacon("/api/bye?t=" + encodeURIComponent(TOKEN));
  });

  if (window.innerWidth < 760) document.body.classList.add("side-closed");
  pollInfo();
  refreshList();
  $("input").focus();
})();
