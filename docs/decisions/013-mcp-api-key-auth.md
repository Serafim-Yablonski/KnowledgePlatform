# 013. MCP API Key Authentication and Session-Scoped Workspace State
Status: Accepted | Date: 2026-05-25

## Context
The MCP server previously supported JWT Bearer tokens (short-lived, issued by the login endpoint) and a static env-var API key pair (`MCP_API_KEY` + `MCP_API_KEY_USER_EMAIL`). Long-running desktop clients like Claude Desktop need a credential that survives token expiry without user intervention. Additionally, every existing MCP tool required an explicit `workspace_id` UUID parameter, forcing the calling agent to track and pass this on every call.

## Decision
Introduce database-backed API keys (SHA-256 hash stored, raw key shown once) as a first-class auth method alongside the existing JWT path. Detection is transparent: the middleware tries JWT decode first; on `JWTError` it falls through to a hash lookup. This means both auth methods accept `Authorization: Bearer <token>` with zero client-side negotiation.

Add session-scoped workspace state (`McpSessionState`) stored in a per-process dict keyed by the `mcp-session-id` HTTP header. Two new tools — `list_workspaces` and `set_active_workspace` — let an agent discover and select a workspace once per session, replacing the need to pass `workspace_id` on every call. Existing tools retain their explicit parameter for backward compatibility.

## Consequences
API keys are the right credential for long-running desktop clients and self-contained scripts: they don't expire, are easy to revoke per-device, and require no OAuth redirect flow. OAuth 2.1 is the correct pattern when delegating to an external IdP or when the calling application cannot securely store a secret — neither condition applies here. The static env-var key pair is kept to avoid breaking existing setups; it should be deprecated once all deployments migrate to database keys.

Session state is in-memory and per-process. A server restart or horizontal scaling requires the client to call `list_workspaces` + `set_active_workspace` again, which is acceptable for an interactive desktop client. A persistent alternative (e.g. Redis-backed session store) was rejected as disproportionate complexity for a portfolio project that runs as a single process.
