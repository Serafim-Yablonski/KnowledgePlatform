from __future__ import annotations

import re

import tiktoken

from src.ai.chunking import ChunkData

# cl100k_base is OpenAI's tokenizer, used here as a provider-agnostic approximation.
# The exact count doesn't need to match the embedding model's tokenizer — what matters
# is consistent, reproducible chunk size control across all document types.
_ENC = tiktoken.get_encoding("cl100k_base")

_HEADER_RE = re.compile(r"^(#{2,3})\s+(.+)$", re.MULTILINE)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_END_RE.split(text.strip())
    return [p for p in parts if p.strip()]


def _overlap_prefix(sentences: list[str], overlap_tokens: int) -> str:
    """Return the tail sentences of a chunk that fit within overlap_tokens."""
    prefix_sentences: list[str] = []
    tokens = 0
    for sentence in reversed(sentences):
        t = _count_tokens(sentence)
        if tokens + t > overlap_tokens:
            break
        prefix_sentences.insert(0, sentence)
        tokens += t
    return " ".join(prefix_sentences)


def _split_section(
    text: str,
    section_header: str,
    max_tokens: int,
    overlap_tokens: int,
) -> list[ChunkData]:
    """Split a single markdown section by sentence boundaries if it exceeds max_tokens."""  # noqa: E501
    if _count_tokens(text) <= max_tokens:
        return [
            ChunkData(
                text=text.strip(),
                metadata={"section": section_header},
                token_count=_count_tokens(text.strip()),
            )
        ]

    sentences = _split_sentences(text)
    chunks: list[ChunkData] = []
    current_sentences: list[str] = []
    current_tokens = 0
    prev_overlap = ""

    for sentence in sentences:
        st = _count_tokens(sentence)
        if current_tokens + st > max_tokens and current_sentences:
            body = " ".join(current_sentences)
            full_text = (prev_overlap + " " + body).strip() if prev_overlap else body
            chunks.append(
                ChunkData(
                    text=full_text,
                    metadata={"section": section_header},
                    token_count=_count_tokens(full_text),
                )
            )
            prev_overlap = _overlap_prefix(current_sentences, overlap_tokens)
            current_sentences = [sentence]
            current_tokens = st
        else:
            current_sentences.append(sentence)
            current_tokens += st

    if current_sentences:
        body = " ".join(current_sentences)
        full_text = (prev_overlap + " " + body).strip() if prev_overlap else body
        chunks.append(
            ChunkData(
                text=full_text,
                metadata={"section": section_header},
                token_count=_count_tokens(full_text),
            )
        )

    return chunks


class MarkdownChunker:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: dict) -> list[ChunkData]:  # type: ignore[type-arg]
        if not text.strip():
            return []

        # Split on ## and ### header boundaries, preserving header text.
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return _split_section(text, "", self._max_tokens, self._overlap_tokens)

        sections: list[tuple[str, str]] = []  # (header_path, body)
        h2_current = ""
        h3_current = ""

        for i, match in enumerate(matches):
            level = len(match.group(1))
            header_text = match.group(2).strip()
            body_start = match.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            if level == 2:
                h2_current = f"## {header_text}"
                h3_current = ""
                section_path = h2_current
            else:
                h3_current = f"### {header_text}"
                sep = f"{h2_current} > {h3_current}"
                section_path = sep if h2_current else h3_current

            if body:
                sections.append((section_path, body))

        # Handle text before the first header
        first_start = matches[0].start()
        if first_start > 0:
            preamble = text[:first_start].strip()
            if preamble:
                sections.insert(0, ("", preamble))

        chunks: list[ChunkData] = []
        for section_path, body in sections:
            chunks.extend(
                _split_section(
                    body, section_path, self._max_tokens, self._overlap_tokens
                )
            )

        return chunks
