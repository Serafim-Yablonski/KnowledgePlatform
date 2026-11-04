from dataclasses import dataclass
from typing import Any, Protocol

import tiktoken

# cl100k_base is OpenAI's tokenizer, used here as a provider-agnostic approximation.
# The exact count doesn't need to match the embedding model's tokenizer — what matters
# is consistent, reproducible chunk size control across all document types.
_ENC = tiktoken.get_encoding("cl100k_base")
# semantic-text-splitter names a tokenizer by model, not by encoding; gpt-4's is
# cl100k_base, so the splitters bound chunks in the tokens count_tokens counts.
TIKTOKEN_MODEL = "gpt-4"


def count_tokens(text: str) -> int:
    # Ordinary: a document quoting "<|endoftext|>" holds text, not a special token
    # (``encode`` raises on it), and the splitters count it the same way.
    return len(_ENC.encode_ordinary(text))


@dataclass
class ChunkData:
    text: str
    metadata: dict[str, Any]
    token_count: int


class ChunkingStrategy(Protocol):
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[ChunkData]: ...


__all__ = ["TIKTOKEN_MODEL", "ChunkData", "ChunkingStrategy", "count_tokens"]
