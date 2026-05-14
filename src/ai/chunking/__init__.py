from dataclasses import dataclass
from typing import Protocol


@dataclass
class ChunkData:
    text: str
    metadata: dict  # type: ignore[type-arg]
    token_count: int


class ChunkingStrategy(Protocol):
    def chunk(self, text: str, metadata: dict) -> list[ChunkData]:  # type: ignore[type-arg]
        ...


__all__ = ["ChunkData", "ChunkingStrategy"]
