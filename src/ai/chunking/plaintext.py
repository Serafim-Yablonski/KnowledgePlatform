from __future__ import annotations

import re

import tiktoken

from src.ai.chunking import ChunkData

# Same rationale as markdown.py: cl100k_base as a provider-agnostic approximation.
_ENC = tiktoken.get_encoding("cl100k_base")

_PARA_SEP = re.compile(r"\n\n+")
_SENTENCE_SEP = re.compile(r"(?<=[.!?])\s+")
_WORD_SEP = re.compile(r"\s+")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _make_chunks(
    segments: list[str],
    max_tokens: int,
    overlap_tokens: int,
    metadata: dict,  # type: ignore[type-arg]
) -> list[ChunkData]:
    """Pack segments into token-bounded chunks with overlap."""
    chunks: list[ChunkData] = []
    current: list[str] = []
    current_tokens = 0
    overlap_text = ""

    for seg in segments:
        seg_tokens = _count_tokens(seg)
        if current_tokens + seg_tokens > max_tokens and current:
            body = " ".join(current)
            full = (overlap_text + " " + body).strip() if overlap_text else body
            chunks.append(
                ChunkData(text=full, metadata=metadata, token_count=_count_tokens(full))
            )
            # Build overlap from the tail of the current window.
            tail = body
            tail_tokens = _count_tokens(tail)
            if tail_tokens > overlap_tokens:
                words = _WORD_SEP.split(tail)
                overlap_parts: list[str] = []
                ot = 0
                for w in reversed(words):
                    wt = _count_tokens(w)
                    if ot + wt > overlap_tokens:
                        break
                    overlap_parts.insert(0, w)
                    ot += wt
                overlap_text = " ".join(overlap_parts)
            else:
                overlap_text = tail
            current = [seg]
            current_tokens = seg_tokens
        else:
            current.append(seg)
            current_tokens += seg_tokens

    if current:
        body = " ".join(current)
        full = (overlap_text + " " + body).strip() if overlap_text else body
        chunks.append(
            ChunkData(text=full, metadata=metadata, token_count=_count_tokens(full))
        )

    return chunks


def _recursive_split(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    metadata: dict,  # type: ignore[type-arg]
) -> list[ChunkData]:
    """Recursively split text: paragraphs → sentences → words."""
    if _count_tokens(text) <= max_tokens:
        stripped = text.strip()
        return [
            ChunkData(
                text=stripped, metadata=metadata, token_count=_count_tokens(stripped)
            )
        ]

    # Try paragraph splits first.
    paragraphs = [p.strip() for p in _PARA_SEP.split(text) if p.strip()]
    if len(paragraphs) > 1:
        return _make_chunks(paragraphs, max_tokens, overlap_tokens, metadata)

    # Fall back to sentence splits.
    sentences = [s.strip() for s in _SENTENCE_SEP.split(text) if s.strip()]
    if len(sentences) > 1:
        return _make_chunks(sentences, max_tokens, overlap_tokens, metadata)

    # Last resort: word splits.
    words = [w for w in _WORD_SEP.split(text) if w]
    return _make_chunks(words, max_tokens, overlap_tokens, metadata)


class PlainTextChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: dict) -> list[ChunkData]:  # type: ignore[type-arg]
        if not text.strip():
            return []
        return _recursive_split(
            text.strip(), self._max_tokens, self._overlap_tokens, metadata
        )
