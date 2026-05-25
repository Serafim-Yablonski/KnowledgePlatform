"""Trace demo: seed data, run AI queries, and surface trace IDs.

Usage:
    uv run python scripts/trace_demo.py

Requires the app to be running: make run
"""

from __future__ import annotations

import sys
import time

import httpx

BASE_URL = "http://localhost:8000"
DEMO_EMAIL = "trace-demo@example.com"
DEMO_PASSWORD = "trace-demo-password-123"


def _post(client: httpx.Client, path: str, **kwargs: object) -> dict[object, object]:
    resp = client.post(f"{BASE_URL}{path}", **kwargs)  # type: ignore[arg-type]
    if resp.status_code not in (200, 201):
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)
    return resp.json()  # type: ignore[return-value]


def _get(client: httpx.Client, path: str, **kwargs: object) -> dict[object, object]:
    resp = client.get(f"{BASE_URL}{path}", **kwargs)  # type: ignore[arg-type]
    if resp.status_code not in (200, 201):
        print(f"  ERROR {resp.status_code}: {resp.text[:200]}")
        sys.exit(1)
    return resp.json()  # type: ignore[return-value]


def main() -> None:
    print("=== Knowledge Platform trace demo ===\n")

    with httpx.Client(timeout=60.0) as client:
        # ── 1. Health check ─────────────────────────────────────────────────
        print("1. Checking app health...")
        resp = client.get(f"{BASE_URL}/health")
        if resp.status_code != 200:
            print(f"   App is not running at {BASE_URL}. Run `make run` first.")
            sys.exit(1)
        print("   OK\n")

        # ── 2. Register user ─────────────────────────────────────────────────
        print("2. Registering demo user...")
        _post(
            client,
            "/api/v1/auth/register",
            json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
        )
        print(f"   Created user: {DEMO_EMAIL}\n")

        # ── 3. Obtain token ──────────────────────────────────────────────────
        print("3. Obtaining access token...")
        token_data = _post(
            client,
            "/api/v1/auth/token",
            data={"username": DEMO_EMAIL, "password": DEMO_PASSWORD},
        )
        token = token_data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("   Token obtained\n")

        # ── 4. Create workspace ──────────────────────────────────────────────
        print("4. Creating demo workspace...")
        ws = _post(
            client,
            "/api/v1/workspaces",
            json={"name": "Trace Demo Workspace"},
            headers=headers,
        )
        workspace_id = ws["id"]
        print(f"   Workspace ID: {workspace_id}\n")

        # ── 5. Upload 3 documents ────────────────────────────────────────────
        print("5. Uploading 3 demo documents...")
        doc_contents = [
            (
                "architecture.txt",
                (
                    "The system uses a layered architecture with FastAPI routes, "
                    "service layer, repositories, and PostgreSQL with pgvector for "
                    "semantic search. All queries are async using asyncpg."
                ),
            ),
            (
                "ai-stack.txt",
                (
                    "PydanticAI handles agent logic with typed deps and output_type. "
                    "LangGraph orchestrates multi-step research workflows with "
                    "PostgreSQL checkpointing for crash recovery."
                ),
            ),
            (
                "observability.txt",
                (
                    "OpenTelemetry traces are exported to Pydantic Logfire. "
                    "Each span records model, token_count, latency_ms, and cache_hit. "
                    "structlog emits structured JSON with trace_id on every log line."
                ),
            ),
        ]

        for filename, content in doc_contents:
            files = {"file": (filename, content.encode(), "text/plain")}
            resp = client.post(
                f"{BASE_URL}/api/v1/workspaces/{workspace_id}/documents",
                files=files,
                headers=headers,
            )
            if resp.status_code not in (200, 201):
                print(f"   Upload failed for {filename}: {resp.text[:200]}")
                sys.exit(1)
            print(f"   Uploaded {filename}")

        # ── 6. Wait for embedding (Celery worker) ─────────────────────────────
        print("\n6. Waiting 5s for Celery worker to embed documents...")
        time.sleep(5)
        print("   Done waiting\n")

        # ── 7. Ask a question ────────────────────────────────────────────────
        print("7. Asking a question (generates ai_ask + search + embed spans)...")
        answer = _post(
            client,
            f"/api/v1/workspaces/{workspace_id}/ai/ask",
            json={"question": "How does the system handle observability and tracing?"},
            headers=headers,
        )
        print(f"   Answer confidence: {answer.get('confidence', 'n/a')}")
        print(f"   Sources: {len(answer.get('sources', []))}\n")

        # ── 8. Start research workflow ────────────────────────────────────────
        print("8. Starting research workflow (generates LangGraph spans)...")
        research = _post(
            client,
            f"/api/v1/workspaces/{workspace_id}/research",
            json={
                "topic": "What AI frameworks are used and why?",
                "max_iterations": 2,
            },
            headers=headers,
        )
        thread_id = research.get("thread_id", "unknown")
        print(f"   Research thread ID: {thread_id}\n")

    # ── 9. Done ──────────────────────────────────────────────────────────────
    print("=" * 50)
    print("Trace demo complete.")
    print()
    if True:  # always print Logfire note
        print("Check Logfire dashboard for traces:")
        print("  https://logfire.pydantic.dev/")
        print()
        print("If LOGFIRE_TOKEN is not set, traces were printed to the console above.")
    print()
    print(f"Workspace ID: {workspace_id}")
    print(f"Research thread: {thread_id}")
    print()
    print("Expected spans in the /ask trace:")
    print("  FastAPI middleware → auth → rate_limit → ai_ask")
    print("    → pydantic_ai.agent.run → search_documents tool")
    print("       → search (query_hash, result_count, top_score)")
    print("          → embed_texts (model, cache_hits, cache_misses, api_latency_ms)")
    print("          → asyncpg.execute (SQL query)")
    print("    → LLM completion (model, prompt_tokens, completion_tokens)")


if __name__ == "__main__":
    main()
