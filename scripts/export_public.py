"""Copy the files that are meant to be public into a fresh folder, and check them.

  python scripts/export_public.py <empty-or-new-folder | the public git checkout>

What goes out is an allowlist, not "everything but .gitignore": a file added
to this folder later (a key, a tunnel config, a test transcript) stays private
until it is named here. After copying, every text file is scanned for:

  - a path inside a Windows user profile (drive, Users, then a name),
  - anything listed in .private-patterns.txt next to this project (one regular
    expression per line; that file is never exported).

Any hit stops the export with the file and line. PowerShell scripts are written
with a UTF-8 BOM: Windows PowerShell 5.1, which runs Rula-Setup.cmd, reads a
script without one in the system code page and garbles the Japanese in it.
"""
import fnmatch
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ALLOW = [
    "src/**", "launcher/RulaLauncher.cs", "launcher/rula.ico",
    "scripts/graft_mtp.py", "scripts/gguf_header.py", "scripts/make_icon.py",
    "scripts/export_public.py",
    "DisBot/discord_bot.py", "DisBot/run_bot.ps1",
    "setup.ps1", "Rula-Setup.cmd", "install.ps1", "setup_llama.ps1", "build.ps1",
    "api.py", "client_example.py", "make_api_key.py", "run_api.ps1", "setup_tunnel.ps1",
    "install_autostart.ps1", "check_identity.py",
    "Modelfile.rula", "setup_ollama.py", "setup_model.py",
    "requirements.txt", "requirements.lock", "requirements-torch.txt",
    ".env.example", ".gitignore", ".gitattributes", "README.md", "NOTICE", "LICENSE",
]
SKIP_PARTS = {"__pycache__"}
# Run with "iwr ... | iex": a BOM there breaks the first line (measured in both
# Windows PowerShell 5.1 and PowerShell 7), so it is kept ASCII without one.
NO_BOM = {"setup.ps1"}
TEXT_EXT = {".py", ".ps1", ".cmd", ".md", ".txt", ".lock", ".cs", ".html", ".css", ".js",
            ".example", ".gitignore", ".gitattributes", ""}
USER_PATH = re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b)[^\\/\s\"'<>%]+[\\/]", re.I)


def allowed(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    if any(part in SKIP_PARTS for part in rel.split("/")):
        return False
    return any(fnmatch.fnmatch(rel, pat) for pat in ALLOW)


def private_patterns() -> list[re.Pattern]:
    path = os.path.join(ROOT, ".private-patterns.txt")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [re.compile(line.strip(), re.I) for line in f
                if line.strip() and not line.startswith("#")]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = os.path.abspath(sys.argv[1])
    # Empty, or the public repository itself: everything but .git is replaced,
    # so a file dropped from the allowlist is also dropped from the next commit.
    if os.path.exists(out):
        for name in os.listdir(out):
            if name == ".git":
                continue
            if not os.path.isfile(os.path.join(out, ".git", "HEAD")):
                sys.exit(f"出力先が空ではありません: {out}")
            path = os.path.join(out, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    copied = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_PARTS and d != ".git"]
        for name in filenames:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, ROOT)
            if not allowed(rel):
                continue
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if name.endswith(".ps1"):
                with open(src, encoding="utf-8-sig") as f:
                    text = f.read()
                if rel in NO_BOM:
                    if not text.isascii():
                        sys.exit(f"{rel} は ASCII だけで書く必要があります (iwr | iex で実行するため)")
                    with open(dst, "w", encoding="ascii", newline="") as f:
                        f.write(text)
                else:
                    with open(dst, "w", encoding="utf-8-sig", newline="") as f:
                        f.write(text)
            else:
                shutil.copy2(src, dst)
            copied.append(rel)
    missing = [p for p in ALLOW if "*" not in p and not os.path.isfile(os.path.join(ROOT, p))]

    patterns = [USER_PATH] + private_patterns()
    hits = []
    for rel in copied:
        if os.path.splitext(rel)[1].lower() not in TEXT_EXT:
            continue
        with open(os.path.join(out, rel), encoding="utf-8-sig", errors="replace") as f:
            for i, line in enumerate(f, 1):
                for pat in patterns:
                    if pat.search(line):
                        hits.append(f"{rel}:{i}: {line.strip()[:100]}")
    print(f"{len(copied)} files -> {out}")
    if missing:
        print("見つからない(公開に含まれない)ファイル:", ", ".join(missing))
    if hits:
        print("個人情報の可能性がある箇所が見つかりました。公開を中止してください:")
        print("\n".join(hits))
        sys.exit(1)
    print("個人情報の検査: 問題なし")


if __name__ == "__main__":
    main()
