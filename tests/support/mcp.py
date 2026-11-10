"""A raw MCP request as the SDK client sends it, for tests below the SDK client."""

from __future__ import annotations

from typing import Any

# The only protocol revision we support: every request is self-contained (no
# initialize handshake, no session id) and names its version in a header and _meta.
PROTOCOL_VERSION = "2026-07-28"


def rpc_request(
    method: str, params: dict[str, Any] | None = None, *, name: str | None = None
) -> tuple[dict[str, Any], dict[str, str]]:
    """A valid request's JSON body and headers: the version and method in headers
    (and the tool's ``name`` for a tools/call), the version and client details in
    ``params._meta`` (without them the SDK answers 400)."""
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": PROTOCOL_VERSION,
                "io.modelcontextprotocol/clientCapabilities": {},
                "io.modelcontextprotocol/clientInfo": {"name": "test", "version": "1"},
            },
        },
    }
    headers = {
        # Without this Accept the SDK answers 406 whatever else is wrong.
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": PROTOCOL_VERSION,
        "MCP-Method": method,
    }
    if name is not None:
        headers["MCP-Name"] = name
    return body, headers
