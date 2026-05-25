#!/usr/bin/env python3
"""Create an MCP API key and print ready-to-paste configs for every major MCP client."""

from __future__ import annotations

import json
import platform
import shutil
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx is required. Run: uv run python scripts/generate_mcp_config.py")
    sys.exit(1)


def _prompt(msg: str, default: str) -> str:
    value = input(f"{msg} [{default}]: ").strip()
    return value or default


def _node_available() -> bool:
    return shutil.which("node") is not None


def _claude_desktop_config_path() -> Path:
    system = platform.system()
    if system == "Darwin":
        return Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    if system == "Windows":
        return Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
    return Path.home() / ".config/Claude/claude_desktop_config.json"


def _cursor_config_path() -> Path:
    system = platform.system()
    if system == "Windows":
        return Path.home() / ".cursor/mcp.json"
    return Path.home() / ".cursor/mcp.json"


def _print_all_configs(base_url: str, raw_key: str, key_data: dict) -> None:
    mcp_url = f"{base_url}/mcp"
    bearer = f"Bearer {raw_key}"

    sep = "─" * 60

    # ── 1. Claude Desktop — stdio via mcp-remote ─────────────────────────────
    stdio_config = {
        "mcpServers": {
            "knowledge-platform": {
                "command": "npx",
                "args": ["-y", "mcp-remote", mcp_url, "--header", f"Authorization: {bearer}"],
            }
        }
    }
    print(f"\n{sep}")
    print("1.  Claude Desktop  (this build — stdio via mcp-remote)")
    print(sep)
    if not _node_available():
        print("    ⚠  Node.js not found — install from https://nodejs.org before proceeding")
    print(json.dumps(stdio_config, indent=2))
    print(f"\n    Paste into: {_claude_desktop_config_path()}")
    print("    Then restart Claude Desktop.")

    # ── 2. Cursor / newer Claude Desktop — native HTTP ───────────────────────
    http_config = {
        "mcpServers": {
            "knowledge-platform": {
                "url": mcp_url,
                "headers": {"Authorization": bearer},
            }
        }
    }
    print(f"\n{sep}")
    print("2.  Cursor / Claude Desktop (newer builds)  — native HTTP")
    print(sep)
    print(json.dumps(http_config, indent=2))
    print(f"\n    Cursor config path: {_cursor_config_path()}")
    print(f"    Claude Desktop config path: {_claude_desktop_config_path()}")
    print("    Then restart the client.")

    # ── 3. Claude Code CLI ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("3.  Claude Code CLI  — run one of these commands in your terminal")
    print(sep)
    print("    Streamable HTTP (preferred, recent builds):")
    print(f'      claude mcp add --transport http \\')
    print(f'        --header "Authorization: {bearer}" \\')
    print(f'        knowledge-platform {mcp_url}')
    print()
    print("    SSE fallback (older builds):")
    print(f'      claude mcp add --transport sse \\')
    print(f'        --header "Authorization: {bearer}" \\')
    print(f'        knowledge-platform {mcp_url}')
    print()
    print('    Verify with: claude mcp list')

    # ── 4. Claude.ai Projects ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("4.  Claude.ai Projects  — paste in the web UI")
    print(sep)
    print(f"    URL:         {mcp_url}")
    print(f"    Auth header: Authorization: {bearer}")
    print("    Path: claude.ai → Settings → Integrations → Add MCP server")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"API key  prefix={key_data['prefix']}  name={key_data['name']!r}")
    print("Keep this key safe — it cannot be retrieved again.")
    print(sep)


def main() -> None:
    base_url = _prompt("App base URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")
    key_name = _prompt("API key name", "Claude Desktop")

    action = _prompt("Login or register? (login/register)", "login")
    email = input("Email: ").strip()
    password = input("Password: ").strip()

    with httpx.Client(base_url=base_url) as client:
        if action == "register":
            display = input("Display name (optional): ").strip() or None
            resp = client.post(
                "/api/v1/auth/register",
                json={"email": email, "password": password, "display_name": display},
            )
            if resp.status_code not in (200, 201):
                print(f"Registration failed: {resp.text}")
                sys.exit(1)

        resp = client.post(
            "/api/v1/auth/login",
            json={"email": email, "password": password},
        )
        if resp.status_code != 200:
            print(f"Login failed: {resp.text}")
            sys.exit(1)

        access_token = resp.json()["access_token"]

        resp = client.post(
            "/api/v1/auth/api-keys",
            json={"name": key_name},
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if resp.status_code not in (200, 201):
            print(f"API key creation failed: {resp.text}")
            sys.exit(1)

        key_data = resp.json()
        raw_key = key_data["key"]

    _print_all_configs(base_url, raw_key, key_data)


if __name__ == "__main__":
    main()
