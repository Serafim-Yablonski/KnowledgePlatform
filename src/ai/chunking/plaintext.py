from typing import Any

from semantic_text_splitter import TextSplitter

from src.ai.chunking import TIKTOKEN_MODEL, ChunkData, count_tokens


class PlainTextChunker:
    """Plain text and PDF text (ADR-031): every chunk, overlap included, is at most
    ``max_tokens``, split at the largest boundary that fits — paragraph, line,
    sentence, word, then characters for a run with none."""

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        self._splitter = TextSplitter.from_tiktoken_model(
            TIKTOKEN_MODEL, max_tokens, overlap_tokens
        )

    def spans(self, text: str) -> list[tuple[int, str]]:
        """Each chunk with its offset in ``text``: the input's own characters, trimmed
        of surrounding whitespace, in order."""
        return self._splitter.chunk_indices(text)

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[ChunkData]:
        return [
            ChunkData(text=chunk, metadata=metadata, token_count=count_tokens(chunk))
            for _, chunk in self.spans(text)
        ]
