from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ChunkData:
    text: str
    metadata: dict[str, Any]
    token_count: int


class ChunkingStrategy(Protocol):
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[ChunkData]: ...


__all__ = ["ChunkData", "ChunkingStrategy"]
