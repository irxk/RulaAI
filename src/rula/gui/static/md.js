// Markdown → HTML for Rula's answers.
//
// Everything is escaped first; only the constructs below become markup, so an
// answer can never inject script or styles into the window.
(function () {
  "use strict";

  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");

  // Answers sometimes carry TeX ("$17 \times 23 = 391$"). Shown as plain text with
  // the common symbols translated, rather than as raw backslashes.
  const TEX = [[/\\times/g, "×"], [/\\div/g, "÷"], [/\\cdot/g, "·"], [/\\pm/g, "±"],
    [/\\leq?/g, "≤"], [/\\geq?/g, "≥"], [/\\neq/g, "≠"], [/\\approx/g, "≈"],
    [/\\infty/g, "∞"], [/\\pi/g, "π"], [/\\sqrt\{([^}]*)\}/g, "√($1)"],
    [/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1)/($2)"], [/\\text\{([^}]*)\}/g, "$1"],
    [/\\[,;!]/g, " "], [/\\quad/g, "  "], [/[{}]/g, ""]];
  const tex = (s) => TEX.reduce((acc, [re, to]) => acc.replace(re, to), s);

  // Trailing punctuation is not part of a URL, but a closing paren is when the URL
  // opened one: https://ja.wikipedia.org/wiki/日本の山一覧_(高さ順)
  function trimUrl(u) {
    for (;;) {
      const last = u.slice(-1);
      if ("。、，,.!！?？」』】".includes(last)) { u = u.slice(0, -1); continue; }
      if ((last === ")" || last === "）") &&
          (u.split("(").length + u.split("（").length) < (u.split(")").length + u.split("）").length)) {
        u = u.slice(0, -1); continue;
      }
      return u;
    }
  }

  function inline(raw) {
    const slots = [];
    const keep = (html) => { slots.push(html); return "\u0000" + (slots.length - 1) + "\u0000"; };
    let s = raw;
    s = s.replace(/`([^`\n]+)`/g, (_, c) => keep("<code>" + esc(c) + "</code>"));
    s = s.replace(/\$\$([^$]+)\$\$|\$([^$\n]+)\$/g,
      (_, a, b) => keep('<span class="math">' + esc(tex(a || b)) + "</span>"));
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+(?:\([^\s)]*\))?[^\s)]*)\)/g,
      (_, t, u) => keep('<a href="' + esc(u) + '">' + esc(t) + "</a>"));
    s = s.replace(/https?:\/\/[^\s<>"'`]+/g, (m) => {
      const u = trimUrl(m);
      return keep('<a href="' + esc(u) + '">' + esc(u) + "</a>") + m.slice(u.length);
    });
    s = esc(s);
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*\w])\*([^*\s][^*\n]*?)\*(?!\*)/g, "$1<em>$2</em>");
    s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => slots[+i]);
  }

  function codeBlock(code, lang) {
    return '<div class="code"><div class="code-head"><span>' + esc(lang || "code") +
      '</span><button class="copy" type="button">コピー</button></div><pre><code>' +
      esc(code) + "</code></pre></div>";
  }

  const isTableRow = (l) => /^\s*\|.*\|\s*$/.test(l);
  const isTableSep = (l) => /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(l);
  const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
  const LIST = /^(\s*)([-*+•]|\d+[.)])\s+(.*)$/;
  const BLOCK_START = /^\s*(```|#{1,6}\s|>|([-*+•]|\d+[.)])\s+|(-{3,}|\*{3,}|_{3,})\s*$)/;

  function render(md) {
    const lines = (md || "").replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      let m = line.match(/^\s*```\s*([\w+#.-]*)\s*$/);
      if (m) {
        const buf = [];
        i++;
        while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) buf.push(lines[i++]);
        i++;  // the closing fence (or the end, while still streaming)
        out.push(codeBlock(buf.join("\n"), m[1]));
        continue;
      }
      if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
        const head = cells(line);
        i += 2;
        const rows = [];
        while (i < lines.length && isTableRow(lines[i])) rows.push(cells(lines[i++]));
        out.push('<div class="table"><table><thead><tr>' +
          head.map((c) => "<th>" + inline(c) + "</th>").join("") + "</tr></thead><tbody>" +
          rows.map((r) => "<tr>" + r.map((c) => "<td>" + inline(c) + "</td>").join("") + "</tr>").join("") +
          "</tbody></table></div>");
        continue;
      }
      m = line.match(/^(#{1,6})\s+(.*)$/);
      if (m) {
        const level = Math.min(m[1].length + 1, 5);
        out.push("<h" + level + ">" + inline(m[2]) + "</h" + level + ">");
        i++;
        continue;
      }
      if (/^\s*(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) { out.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        const buf = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*>\s?/, ""));
        out.push("<blockquote>" + render(buf.join("\n")) + "</blockquote>");
        continue;
      }
      if (LIST.test(line)) {
        out.push(list(lines, i, (next) => { i = next; }));
        continue;
      }
      if (!line.trim()) { i++; continue; }
      // The first line is always taken: a line that looks like a block start but
      // matched none of the rules above ("```python title") would otherwise be
      // left where it is, and the loop would never move past it.
      const para = [inline(lines[i++].replace(/\s+$/, ""))];
      while (i < lines.length && lines[i].trim() && !BLOCK_START.test(lines[i]) &&
             !(isTableRow(lines[i]) && i + 1 < lines.length && isTableSep(lines[i + 1]))) {
        para.push(inline(lines[i++].replace(/\s+$/, "")));
      }
      out.push("<p>" + para.join("<br>") + "</p>");
    }
    return out.join("");
  }

  // Lists, one level of nesting by indentation; continuation lines join the item.
  function list(lines, start, done) {
    let i = start;
    const base = lines[i].match(LIST)[1].length;
    const ordered = /\d/.test(lines[i].match(LIST)[2]);
    const items = [];
    while (i < lines.length) {
      const m = lines[i].match(LIST);
      if (m && m[1].length <= base + 1) {
        items.push({ text: [m[3]], sub: [] });
        i++;
      } else if (m && items.length) {
        items[items.length - 1].sub.push(lines[i]);
        i++;
      } else if (lines[i].trim() && items.length && /^\s{2,}\S/.test(lines[i])) {
        items[items.length - 1].text.push(lines[i].trim());
        i++;
      } else {
        break;
      }
    }
    done(i);
    const tag = ordered ? "ol" : "ul";
    return "<" + tag + ">" + items.map((it) =>
      "<li>" + it.text.map(inline).join("<br>") +
      (it.sub.length ? render(it.sub.map((l) => l.replace(/^\s+/, "")).join("\n")) : "") +
      "</li>").join("") + "</" + tag + ">";
  }

  window.renderMarkdown = render;
})();
