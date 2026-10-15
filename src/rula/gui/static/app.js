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

  const state = { chatId: null, busy: false, image: null, thumb: null, think: false, info: null,
                  workspace: localStorage.getItem("rula-workspace") || "", interaction: null,
                  autoConfirmedWorkspaces: new Set() };

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
    const version = $("app-version");
    version.textContent = "ver." + info.version;
    version.hidden = false;
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
      if (info.size) {
        const s = el("span", "badge size", info.size);
        s.title = "読み込み中のモデルの大きさ (" + info.model + ")";
        badges.append(s);
      }
      if (info.mtp) badges.append(el("span", "badge", "MTP"));
      if (info.vision) badges.append(el("span", "badge", "画像OK"));
      if (info.vram_free != null && info.vram_free < 400) {
        const b = el("span", "badge warn", "VRAM残り " + info.vram_free + "MB");
        b.title = "空きが少ないと生成が極端に遅くなります。GPUを使う他のアプリを閉じてください。";
        badges.append(b);
      }
    }
    $("attach").disabled = !(info.status === "ready" && info.vision) || $("profile").value === "coder";
    $("attach").title = info.vision ? "画像を添付" : "このモデルでは画像を読めません";
    $("think").disabled = !(info.status === "ready" && info.thinking);
    updateSend();
  }

  function updateProfile() {
    const coder = $("profile").value === "coder";
    $("coder-mode").hidden = !coder;
    $("coder-effort").hidden = !coder;
    $("workspace-select").hidden = !coder;
    $("workspace-name").hidden = !coder;
    $("workspace-name").textContent = state.workspace || "未選択";
    $("coder-effort").title = {
      small: "小: 最大8回のツール利用。速度優先。",
      medium: "中: 最大14回のツール利用。調査・実装・検証の標準。",
      large: "大: 最大32回のツール利用、最大6,144トークン。思考・調査・回答量を拡張。9Bモデルは同一。",
    }[$("coder-effort").value];
    $("attach").disabled = coder || !(state.info && state.info.status === "ready" && state.info.vision);
    if (coder) clearAttachment();
    updateSend();
  }

  async function selectWorkspace() {
    $("workspace-select").disabled = true;
    try {
      const data = await (await post("/api/workspace/select")).json();
      if (data.workspace) {
        state.workspace = data.workspace;
        localStorage.setItem("rula-workspace", state.workspace);
      }
      updateProfile();
    } catch (e) {
      $("hint").textContent = "フォルダを選べませんでした: " + e.message;
    } finally {
      $("workspace-select").disabled = false;
    }
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
    $("profile").value = chat.profile || (chat.coder_workspace ? "coder" : "general");
    $("coder-mode").value = chat.coder_mode || "chat";
    $("coder-effort").value = chat.coder_effort || "medium";
    if (chat.coder_workspace) {
      state.workspace = chat.coder_workspace;
      localStorage.setItem("rula-workspace", state.workspace);
    }
    updateProfile();
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
    const plan = el("div", "work-plan");
    plan.hidden = true;
    const activity = el("details", "activity-log");
    activity.hidden = true;
    activity.append(el("summary", null, "作業ログ"));
    const activityItems = el("ol", "activity-list");
    activity.append(activityItems);
    const body = el("div", "body");
    m.append(status, plan, activity, body);
    $("messages").append(m);
    let thinkBox = null;
    let pending = null;
    const activityRows = new Map();

    const renderActivity = (entry) => {
      if (!entry || entry.index == null) return;
      const firstEntry = activity.hidden;
      activity.hidden = false;
      if (firstEntry) activity.open = true;
      let row = activityRows.get(entry.index);
      if (!row) {
        row = el("li", "activity-running");
        activityRows.set(entry.index, row);
        activityItems.append(row);
      }
      row.replaceChildren();
      const heading = el("span", "activity-heading");
      heading.textContent = entry.label || entry.name || "作業";
      row.append(heading);
      if (entry.detail) {
        const detail = el("code", "activity-detail");
        detail.textContent = entry.detail;
        row.append(detail);
      }
      if (entry.result) {
        const result = el("span", "activity-result");
        result.textContent = entry.result;
        row.append(result);
      } else if (entry.phase === "running") {
        row.append(el("span", "activity-progress", "実行中"));
      }
      row.classList.toggle("activity-complete", entry.phase === "done");
      scrollToEnd();
    };

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
      plan(tasks) {
        if (!Array.isArray(tasks) || !tasks.length) {
          plan.hidden = true;
          return;
        }
        plan.hidden = false;
        plan.replaceChildren();
        const heading = el("div", "work-plan-title", "作業計画");
        const list = el("ol", "work-plan-list");
        for (const task of tasks) {
          const item = el("li", "work-plan-" + task.status);
          item.textContent = task.step;
          const stateLabel = el("span", "work-plan-state",
            { pending: "待機", in_progress: "作業中", completed: "完了" }[task.status] || task.status);
          item.append(stateLabel);
          list.append(item);
        }
        plan.append(heading, list);
        scrollToEnd();
      },
      activity(entry) { renderActivity(entry); },
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
        this.plan(ev.coder_plan || ev.plan);
        if (Array.isArray(ev.activity_log)) {
          activityRows.clear();
          activityItems.replaceChildren();
          activity.hidden = !ev.activity_log.length;
          for (const entry of ev.activity_log) renderActivity(entry);
        }
        if (ev.thinking) {
          setThinking(ev.thinking, false);
          thinkBox.open = false;
        }
        body.innerHTML = window.renderMarkdown(ev.answer != null ? ev.answer : ev.content || "");
        const filesChanged = [...new Set(ev.files_changed || [])];
        const tools = ev.tools || [];
        if (ev.elapsed != null || filesChanged.length || tools.length || ev.stopped) {
          const labels = {
            search: "検索", deep_search: "検索", smart_search: "検索", read_url: "ページ閲覧",
            calc: "計算", date: "日付計算", list_files: "ファイル一覧",
            read_file: "ファイル読み込み", search_code: "コード検索",
            project_info: "プロジェクト情報", update_plan: "作業計画更新",
            git_status: "Git状態確認", git_diff: "Git差分確認", ask_user: "質問",
            write_file: "ファイル保存", edit_file: "ファイル編集",
            apply_patch: "パッチ適用", review_changes: "差分レビュー",
            run_command: "コマンド実行",
          };
          const summaryBits = [];
          if (ev.elapsed != null) summaryBits.push("処理時間 " + ev.elapsed + "秒");
          if (filesChanged.length) summaryBits.push(filesChanged.length + "ファイルを編集");
          if (tools.length) summaryBits.push("操作 " + tools.length + "回");
          if (ev.stopped) summaryBits.push("停止");

          const details = el("details", "turn-details");
          details.append(el("summary", null, summaryBits.join(" · ") || "回答の詳細"));
          const content = el("div", "turn-details-content");
          if (ev.elapsed != null) {
            const section = el("div", "turn-detail-section");
            section.append(el("strong", null, "処理時間"));
            section.append(el("span", null, ev.elapsed + "秒"));
            content.append(section);
          }
          if (filesChanged.length) {
            const section = el("div", "turn-detail-section");
            section.append(el("strong", null, "編集したファイル"));
            const list = el("ul");
            for (const path of filesChanged) {
              const item = el("li");
              item.textContent = path;
              list.append(item);
            }
            section.append(list);
            content.append(section);
          }
          if (tools.length) {
            const section = el("div", "turn-detail-section");
            section.append(el("strong", null, "使用した操作"));
            const counts = new Map();
            for (const tool of tools) {
              const label = labels[tool.name] || tool.name;
              counts.set(label, (counts.get(label) || 0) + 1);
            }
            const list = el("ul");
            for (const [label, count] of counts) {
              const item = el("li");
              item.textContent = label + (count > 1 ? " (" + count + "回)" : "");
              list.append(item);
            }
            section.append(list);
            content.append(section);
          }
          if (ev.stopped) {
            const section = el("div", "turn-detail-section");
            section.append(el("strong", null, "状態"), el("span", null, "停止しました"));
            content.append(section);
          }
          details.append(content);
          m.append(details);
        }
        const meta = el("div", "meta");
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
    const profile = $("profile").value;
    const coderMode = $("coder-mode").value;
    const coderEffort = $("coder-effort").value;
    if (profile === "coder" && !state.workspace) {
      $("hint").textContent = "先にCoderの作業フォルダを選んでください。";
      return;
    }
    if (profile === "coder" && coderMode === "auto" &&
        !state.autoConfirmedWorkspaces.has(state.workspace)) {
      if (!window.confirm("オートモードの注意\n\nCoderは選択した作業フォルダ内のファイルを変更し、"
          + "必要なコマンドを個別確認なしで実行します。作業内容は下の作業ログに表示されます。"
          + "信頼できるプロジェクトだけを選んでください。変更を元に戻せない場合があります。\n\n"
          + "作業フォルダ: " + state.workspace + "\n\nこの作業フォルダで続行しますか？")) return;
      state.autoConfirmedWorkspaces.add(state.workspace);
    }

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
        case "interaction": showInteraction(ev); break;
        case "activity": a.activity(ev.entry); break;
        case "plan": a.plan(ev.tasks); break;
        case "done": a.final(ev); break;
        case "error": a.error(ev.text); break;
      }
    };
    try {
      const r = await post("/api/chat", { chat_id: state.chatId, message: text, image, thumb,
                                          think: state.think, profile, coder_mode: coderMode,
                                          coder_effort: coderEffort,
                                          auto_approved: profile === "coder" && coderMode === "auto"
                                            && state.autoConfirmedWorkspaces.has(state.workspace),
                                          workspace: state.workspace || null });
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
      // A dropped connection to the backend (closed or crashed) rejects fetch
      // with a TypeError ("Failed to fetch"); say what to do, not the raw text.
      // HTTP errors from api() are plain Error with their own message, so those
      // still show as-is.
      a.error(e instanceof TypeError
        ? "Rula が終了しているようです。このウィンドウを閉じて、開き直してください。"
        : e.message);
    } finally {
      setBusy(false);
      refreshList();
      $("input").focus();
    }
  }

  function showInteraction(ev) {
    state.interaction = ev;
    const command = ev.kind === "command";
    $("interaction-title").textContent = command ? "コマンド実行の確認" : "Coderからの質問";
    $("interaction-message").textContent = command
      ? "以下のコマンドを作業フォルダ内で実行してよいですか？"
      : ev.message || "追加情報を入力してください。";
    $("interaction-command").hidden = !command;
    $("interaction-command").textContent = command ? ev.command : "";
    $("interaction-answer").hidden = command;
    $("interaction-answer").value = "";
    $("interaction-cancel").hidden = false;
    $("interaction-deny").hidden = !command;
    $("interaction-submit").hidden = command;
    $("interaction-approve").hidden = !command;
    $("interaction").hidden = false;
    if (!command) $("interaction-answer").focus();
  }

  async function replyInteraction(response) {
    if (!state.interaction) return;
    const current = state.interaction;
    state.interaction = null;
    $("interaction").hidden = true;
    try {
      await post("/api/interaction", { interaction_id: current.interaction_id, response });
    } catch (e) {
      $("hint").textContent = "確認への応答に失敗しました: " + e.message;
    }
  }

  // ---------------------------------------------------------------- images
  async function attachFile(file) {
    if (!file || !/^image\//.test(file.type)) return;
    if ($("profile").value === "coder") {
      $("hint").textContent = "Coderでは画像添付できません";
      return;
    }
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
    if (state.busy) {
      state.interaction = null;
      $("interaction").hidden = true;
      post("/api/stop").catch(() => {});
    }
    else send();
  });
  $("attach").addEventListener("click", () => $("file").click());
  $("file").addEventListener("change", (e) => attachFile(e.target.files[0]));
  $("attach-remove").addEventListener("click", clearAttachment);
  $("think").addEventListener("click", () => {
    state.think = !state.think;
    $("think").setAttribute("aria-pressed", String(state.think));
  });
  $("profile").addEventListener("change", updateProfile);
  $("coder-mode").addEventListener("change", () => updateProfile());
  $("workspace-select").addEventListener("click", selectWorkspace);
  $("interaction-submit").addEventListener("click", () =>
    replyInteraction($("interaction-answer").value.trim()));
  $("interaction-approve").addEventListener("click", () => replyInteraction("approved"));
  $("interaction-deny").addEventListener("click", () => replyInteraction("denied"));
  $("interaction-cancel").addEventListener("click", () =>
    replyInteraction(state.interaction && state.interaction.kind === "command" ? "denied" : ""));
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
  // A minimized or background window has this timer throttled by the browser to
  // about once a minute, so ping again the moment the window is shown or woken
  // from sleep — that refreshes the backend's heartbeat before it times out.
  document.addEventListener("visibilitychange", () => { if (!document.hidden) ping(); });
  window.addEventListener("pageshow", ping);
  // Closing the window is how the app quits; this is the signal for it.
  window.addEventListener("pagehide", () => {
    navigator.sendBeacon("/api/bye?t=" + encodeURIComponent(TOKEN));
  });

  if (window.innerWidth < 760) document.body.classList.add("side-closed");
  pollInfo();
  refreshList();
  updateProfile();
  $("input").focus();
})();
