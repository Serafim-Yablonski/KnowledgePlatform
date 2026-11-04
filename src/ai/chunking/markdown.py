import re
from dataclasses import dataclass
from typing import Any

from semantic_text_splitter import MarkdownSplitter

from src.ai.chunking import TIKTOKEN_MODEL, ChunkData, count_tokens

# CommonMark ATX headings of level 2 and 3, and code fences: a backtick fence's info
# string has no backtick, and a fence closes on a line of the same character, at
# least as long, with nothing after it.
_HEADING = re.compile(r" {0,3}(#{2,3})[ \t]+(\S.*?)[ \t]*")
_FENCE_OPEN = re.compile(r" {0,3}(?:(`{3,})[^`]*|(~{3,}).*)")
_FENCE_CLOSE = re.compile(r" {0,3}(`{3,}|~{3,})[ \t]*")


@dataclass(frozen=True)
class _Section:
    start: int
    end: int
    path: str  # "## A > ### B"; "" before the first heading


def _headings(text: str) -> list[tuple[int, int, str, int]]:
    """(line start, level, heading, line end) of each heading outside code fences."""
    found: list[tuple[int, int, str, int]] = []
    fence: str | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if fence is not None:
            close = _FENCE_CLOSE.fullmatch(bare)
            if close and close[1][0] == fence[0] and len(close[1]) >= len(fence):
                fence = None
        elif opening := _FENCE_OPEN.fullmatch(bare):
            fence = opening[1] or opening[2]
        elif heading := _HEADING.fullmatch(bare):
            found.append((offset, len(heading[1]), heading[2], offset + len(line)))
        offset += len(line)
    return found


def _sections(text: str) -> list[_Section]:
    """The text between headings, each heading with what follows it. A heading with
    nothing under it joins the next section, which keeps the deeper path."""
    headings = _headings(text)
    sections: list[_Section] = []
    first = headings[0][0] if headings else len(text)
    if text[:first].strip():
        sections.append(_Section(0, first, ""))
    h2 = ""
    pending: int | None = None
    for i, (start, level, title, line_end) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        if level == 2:
            h2 = path = f"## {title}"
        else:
            path = f"{h2} > ### {title}" if h2 else f"### {title}"
        if not text[line_end:end].strip() and i + 1 < len(headings):
            pending = start if pending is None else pending
            continue
        sections.append(_Section(start if pending is None else pending, end, path))
        pending = None
    return sections


class MarkdownChunker:
    """Markdown (ADR-031): sections start at ``##`` and ``###`` headings outside code
    fences, and a chunk never spans two, so its section path is exact. Each section,
    heading included, is split like plain text but at markdown's own boundaries
    first (blocks, code blocks, list items); every chunk, overlap included, is at
    most ``max_tokens``."""

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50) -> None:
        self._splitter = MarkdownSplitter.from_tiktoken_model(
            TIKTOKEN_MODEL, max_tokens, overlap_tokens
        )

    def _chunks(self, text: str) -> list[tuple[int, str, str]]:
        return [
            (section.start + offset, chunk, section.path)
            for section in _sections(text)
            for offset, chunk in self._splitter.chunk_indices(
                text[section.start : section.end]
            )
        ]

    def spans(self, text: str) -> list[tuple[int, str]]:
        """Each chunk with its offset in ``text``: the input's own characters, trimmed
        of surrounding whitespace, in order."""
        return [(start, chunk) for start, chunk, _ in self._chunks(text)]

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[ChunkData]:
        return [
            ChunkData(
                text=chunk, metadata={"section": path}, token_count=count_tokens(chunk)
            )
            for _, chunk, path in self._chunks(text)
        ]
