"""Generate an API key and store it in .env.

The key does not expire — revoking means running this again, which replaces the
old one. Keys are held as a salted SHA-256 hash so a leaked .env does not hand
over a usable credential; the plaintext is shown once, here, and never stored.
"""
import hashlib
import os
import re
import secrets
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
KEY_PREFIX = "rula_"


def hash_key(key: str, salt: str) -> str:
    return hashlib.sha256((salt + key).encode()).hexdigest()


def _env_value(name: str) -> str | None:
    """A value already in .env, if there is one."""
    if not os.path.isfile(ENV_PATH):
        return None
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            m = re.match(rf"\s*{name}\s*=\s*(\S+)", line)
            if m:
                return m.group(1)
    return None


def _write_env(values: dict[str, str]):
    """Update keys in place, leaving anything else in the file untouched."""
    lines: list[str] = []
    if os.path.isfile(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()

    remaining = dict(values)
    out = []
    for line in lines:
        m = re.match(r"\s*([A-Z_]+)\s*=", line)
        if m and m.group(1) in remaining:
            out.append(f"{m.group(1)}={remaining.pop(m.group(1))}")
        else:
            out.append(line)
    if remaining and not any(l.startswith("# SALT") for l in out):
        # Says plainly that neither value is the key: mistaking the hash for the
        # credential is the obvious reading of a file that contains only these two.
        out.append("")
        out.append("# SALT と HASH は照合用です。APIキーそのものではありません。")
        out.append("# (キーは発行時に一度だけ表示されます。ここからは復元できません)")
    for name, value in remaining.items():
        out.append(f"{name}={value}")

    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(out).strip() + "\n")


def main():
    key = KEY_PREFIX + secrets.token_urlsafe(32)
    salt = secrets.token_hex(16)
    _write_env({"RULA_API_KEY_SALT": salt, "RULA_API_KEY_HASH": hash_key(key, salt)})

    bar = "=" * 60
    print(f"\n{bar}\n  APIキー(この行の文字列をそのまま使います)\n{bar}\n")
    print(f"  {key}\n")
    print(f"{bar}")
    print("\nこの文字列は二度と表示されません。今すぐ控えてください。")
    print(f".env に書かれた SALT / HASH は照合用で、APIキーではありません。")
    print(f"(保存先: {ENV_PATH})")
    print("\n使い方(OpenAI互換クライアント):")
    print("  from openai import OpenAI")
    # The hostname from .env (set by setup_tunnel.ps1), else a placeholder.
    host = os.environ.get("RULA_HOSTNAME") or _env_value("RULA_HOSTNAME") or "api.example.com"
    print(f'  client = OpenAI(base_url="https://{host}/v1",')
    print(f'                  api_key="{key}")')
    print("\n古いキーは無効になりました。")


if __name__ == "__main__":
    main()
