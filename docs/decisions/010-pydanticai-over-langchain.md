# 009. PydanticAI over LangChain for Agent Logic
Status: Accepted | Date: 2026-05-19

## Context
We need an agent framework for the question-answering layer. The two main options were LangChain (the ecosystem default) and PydanticAI (newer, built by the Pydantic team). We also needed a model-agnostic design so the same code runs with Gemini in development, Claude in production, and a test double in CI.

## Decision
Use PydanticAI for all agent logic. The model string comes from `LLM_MODEL` / `LLM_STRONG_MODEL` env vars — never hardcoded. Two tiers: `LLM_MODEL` (default: `google-gla:gemini-2.0-flash`) for routine Q&A where cost matters; `LLM_STRONG_MODEL` (default: `anthropic:claude-sonnet-4-5`) for eval judging where accuracy matters. Swapping providers requires only a one-line env change with no code modifications.

The dependency injection model (`deps_type` dataclass injected via `RunContext`) mirrors FastAPI's `Depends` pattern exactly. This makes agents unit-testable the same way services are: swap the real `SearchService` for an `AsyncMock` and override the model with `TestModel` for deterministic, zero-LLM-call tests. LangChain has no equivalent first-class testing primitive — tests require either real API calls or fragile monkey-patching of internal chain state.

## Consequences
`TestModel` with `custom_output_args` produces deterministic, schema-validated output without any network calls — six agent tests run in milliseconds. `output_type` validation with `retries=2` means the framework handles malformed LLM output automatically; we write no parse/retry logic. The `pydantic-ai-slim` package pulls in ~8 direct dependencies vs. LangChain's ~60+, eliminating version-conflict risk with our OTel and FastAPI stack.

Rejected alternatives: raw `anthropic`/`google-generativeai` SDKs (no structured output validation, no test double), LangChain (heavy dependency footprint, no typed deps injection, verbose testing story), LlamaIndex (optimised for document indexing, not general agent logic). For multi-step workflows with branching and persistence we use LangGraph — PydanticAI agents become nodes inside LangGraph graphs, each responsible only for the LLM call and output validation.
