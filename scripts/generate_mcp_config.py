#!/usr/bin/env python3
"""Interactive script to create an MCP API key and print the Claude Desktop config."""

from __future__ import annotations

import json
import platform
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


def _print_config_path() -> None:
    system = platform.system()
    if system == "Darwin":
        path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
    elif system == "Windows":
        path = Path.home() / "AppData/Roaming/Claude/claude_desktop_config.json"
    else:
        path = Path.home() / ".config/Claude/claude_desktop_config.json"
    print(f"\nPaste the config into: {path}")


def main() -> None:
    base_url = _prompt("App base URL", "http://localhost:8001")
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

    config = {
        "mcpServers": {
            "knowledge-platform": {
                "url": f"{base_url}/mcp",
                "headers": {"Authorization": f"Bearer {raw_key}"},
            }
        }
    }

    print("\n=== MCP Config ===")
    print(json.dumps(config, indent=2))
    print(f"\nAPI key prefix: {key_data['prefix']}  (name: {key_data['name']})")
    _print_config_path()
    print("\nRestart Claude Desktop after updating the config file.")


if __name__ == "__main__":
    main()
