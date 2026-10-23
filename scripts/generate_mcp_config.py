#!/usr/bin/env python3
"""Create an MCP API key and print how to connect a client to it.

The server speaks only the handshake-free MCP protocol revisions (2026-07-28 and
later); clients on the older handshake revisions, and bridges such as mcp-remote
that make them look modern, are refused. Only clients verified end to end against
a running stack are listed.
"""

from __future__ import annotations

import sys

try:
    import httpx
except ImportError:
    print("httpx is required. Run: uv run python scripts/generate_mcp_config.py")
    sys.exit(1)


def _prompt(msg: str, default: str) -> str:
    value = input(f"{msg} [{default}]: ").strip()
    return value or default


def _print_all_configs(base_url: str, raw_key: str, key_data: dict) -> None:
    mcp_url = f"{base_url}/mcp"
    bearer = f"Bearer {raw_key}"

    sep = "─" * 60

    # ── Claude Code ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Claude Code  — run in your terminal")
    print(sep)
    # Name and URL before --header: it takes several values and would swallow them.
    print(f"    claude mcp add --transport http knowledge-platform {mcp_url} \\")
    print(f'      --header "Authorization: {bearer}"')
    print()
    print("    Verify with: claude mcp list")

    # ── Any other client ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("Other clients  — must support MCP protocol 2026-07-28 or later over")
    print("Streamable HTTP; clients on older protocol versions are refused")
    print(sep)
    print(f"    URL:     {mcp_url}")
    print(f"    Header:  Authorization: {bearer}")

    # ── Footer ────────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"API key  prefix={key_data['prefix']}  name={key_data['name']!r}")
    print("Keep this key safe — it cannot be retrieved again.")
    print(sep)


def main() -> None:
    base_url = _prompt("App base URL", "http://localhost:8000")
    base_url = base_url.rstrip("/")
    key_name = _prompt("API key name", "MCP client")

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
