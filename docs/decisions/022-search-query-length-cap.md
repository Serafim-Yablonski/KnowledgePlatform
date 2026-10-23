# 022. Search Queries Are Capped at 2,000 Characters, Derived from the Embedding Model's Input Limit
Status: Accepted | Date: 2026-10-22

## Context
A search query is embedded by `gemini-embedding-001` before it is compared with chunks, and that model reads at most 2,048 tokens. A probe against the live API (2026-10-22) showed what happens beyond that: the call succeeds and the rest is **silently dropped**. Two inputs that differed only after token ~2,100 got identical embeddings (cosine 1.0), while the same two endings after a short prefix scored 0.74. A too-long query therefore returns results for a different query, with no error. The caps were hand-picked and disagreed (REST 1,000, MCP 2,000). Pydantic's `max_length` counts characters (code points), but the model's limit is in tokens, and the ratio depends on the text. Tokens per character, from the model's own `countTokens`:

| Input | Tokens / char | Chars that fit in 2,048 tokens |
|---|---|---|
| English / Russian / stack trace | 0.20–0.27 | ~7,500+ |
| Code / Japanese | 0.42–0.44 | ~4,650 |
| JSON log / base64 | 0.65–0.67 | ~3,050 |
| SHA-256 hashes / UUIDs | 0.87–0.90 | ~2,270 |
| Emoji | 1.00 | 2,048 |
| Math-alphanumeric Unicode (𝔄𝔅ℭ) | 2.44 | ~840 |

## Decision
One cap for REST and MCP, shared as a single `SearchQuery` type: **2,000 characters**, after stripping surrounding whitespace. The cap is 2,048 tokens ÷ 1 token per character, rounded down. It is set by the densest text our clients realistically send. Those clients are code agents, which paste logs, UUIDs and hashes (0.65–0.90 tokens per character), and emoji also reach 1.0. Below the cap, nothing measured short of exotic Unicode is truncated. Above it, the caller gets a validation error naming the limit, which a model reads and acts on. The other possible reasons for a cap don't bind here. Cost doesn't: a query is at most ~2k tokens, at 20 searches per minute per user. Capacity doesn't: a 2 KB string is trivial to parse, and the query is hashed, never logged. Retrieval quality is a soft bound, not a limit: 2,000 English characters is ~400 tokens, about one 512-token chunk. Advice to keep queries short and specific belongs in the tool description, not in validation.

## Consequences
REST's 1,000 cap rose to 2,000; it rejected valid queries for no stated reason. Text above 1 token per character can still be truncated: math-alphanumeric Unicode longer than ~840 characters. This is accepted because it only degrades the sender's own results and has no security or cost effect. The cap belongs to the model: changing the embedding provider means re-deriving it, just as it means re-embedding every chunk. **Rejected: 4,000 characters** (sized to natural text): JSON logs and ID dumps, what code agents paste most, would be truncated silently. **Rejected: ~800 characters** (guarantees fit for every input): it rejects ordinary 1,000-character English queries to defend against text nobody searches with. **Rejected: counting tokens exactly with `countTokens` before each search**: a second network round trip per search, to catch only the rare exotic input. **Rejected: an opt-in live test of the ratio**: the measurements are recorded here instead; nothing in CI depends on the network.
