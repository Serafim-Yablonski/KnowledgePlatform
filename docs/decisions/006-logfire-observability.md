# 006. Pydantic Logfire over Self-Hosted Jaeger
Status: Accepted | Date: 2026-05-04

## Context
The platform needs distributed tracing to correlate a single `/ask` request across FastAPI middleware, the PydanticAI agent, RAG retrieval, embedding, and asyncpg query execution. The choices were a self-hosted OTel collector + Jaeger/Tempo backend or a managed SaaS with native PydanticAI integration.

## Decision
Use Pydantic Logfire as the OTel export destination. All instrumentation code is standard OpenTelemetry (`opentelemetry-api`, `opentelemetry-sdk`) — Logfire is only the exporter. `logfire.instrument_fastapi()`, `logfire.instrument_asyncpg()`, and `logfire.instrument_celery()` auto-instrument those layers. PydanticAI agents emit spans automatically to whatever OTel exporter is configured. When `LOGFIRE_TOKEN` is absent the code calls `logfire.configure(send_to_logfire=False)` which activates a console exporter — the app never crashes for missing config.

## Consequences
No Kubernetes or Docker infrastructure to maintain for the observability backend; the free tier (10M spans/month) covers portfolio traffic. Because instrumentation is standard OTel, migrating to Grafana Tempo or self-hosted Jaeger requires only swapping the exporter — zero application code changes.

Rejected alternative: self-hosted Jaeger via `opentelemetry-exporter-jaeger`. This would require an additional Docker container, persistent storage config, and a Helm chart or Docker Compose service. The operational overhead is a DevOps signal, not a backend signal — the portfolio goal is demonstrating correct instrumentation patterns, not infrastructure management.
