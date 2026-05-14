"""Unit tests for MarkdownChunker and PlainTextChunker."""

from __future__ import annotations

from src.ai.chunking.markdown import MarkdownChunker
from src.ai.chunking.plaintext import PlainTextChunker

# ---------------------------------------------------------------------------
# MarkdownChunker
# ---------------------------------------------------------------------------


class TestMarkdownChunker:
    def test_splits_on_h2_and_h3_headers(self) -> None:
        text = """\
## Setup
Some setup text here.

### Prerequisites
You need Python and Docker installed.

## Usage
Run the application.
"""
        chunks = MarkdownChunker().chunk(text, {})
        assert len(chunks) >= 3
        sections = [c.metadata["section"] for c in chunks]
        assert any("## Setup" in s for s in sections)
        assert any("### Prerequisites" in s for s in sections)
        assert any("## Usage" in s for s in sections)

    def test_metadata_contains_section_path(self) -> None:
        text = """\
## Setup

### Prerequisites
Install dependencies.
"""
        chunks = MarkdownChunker().chunk(text, {})
        prereq_chunk = next(
            c for c in chunks if "Prerequisites" in c.metadata.get("section", "")
        )
        assert "## Setup" in prereq_chunk.metadata["section"]
        assert "### Prerequisites" in prereq_chunk.metadata["section"]

    def test_large_section_splits_by_sentences(self) -> None:
        # Build a section that exceeds max_tokens=64 (small limit for the test).
        sentences = [f"This is sentence number {i}." for i in range(30)]
        text = "## Big Section\n" + " ".join(sentences)
        # overlap_tokens=0 so chunk size is bounded purely by max_tokens + one sentence.
        chunks = MarkdownChunker(max_tokens=64, overlap_tokens=0).chunk(text, {})
        assert len(chunks) > 1
        # Each chunk body is ≤ max_tokens; one extra sentence may push it slightly over.
        for c in chunks:
            assert c.token_count <= 64 + 15

    def test_overlap_carries_tail_sentences(self) -> None:
        sentences = [f"Sentence {i} ends here." for i in range(20)]
        text = "## Section\n" + " ".join(sentences)
        chunks = MarkdownChunker(max_tokens=32, overlap_tokens=15).chunk(text, {})
        assert len(chunks) >= 2
        # The start of chunk[1] should contain text from chunk[0]'s tail.
        tail_words = chunks[0].text.split()[-5:]
        head_words = chunks[1].text.split()[:10]
        assert any(w in head_words for w in tail_words)

    def test_empty_text_returns_empty(self) -> None:
        assert MarkdownChunker().chunk("", {}) == []
        assert MarkdownChunker().chunk("   ", {}) == []

    def test_text_shorter_than_one_chunk(self) -> None:
        text = "## Intro\nJust one sentence."
        chunks = MarkdownChunker().chunk(text, {})
        assert len(chunks) == 1
        assert "Just one sentence." in chunks[0].text

    def test_no_headers_treated_as_single_section(self) -> None:
        text = "Plain paragraph without any headers at all."
        chunks = MarkdownChunker().chunk(text, {})
        assert len(chunks) == 1
        assert chunks[0].metadata["section"] == ""

    def test_token_count_populated(self) -> None:
        text = "## Section\nHello world."
        chunks = MarkdownChunker().chunk(text, {})
        assert all(c.token_count > 0 for c in chunks)


# ---------------------------------------------------------------------------
# PlainTextChunker
# ---------------------------------------------------------------------------


class TestPlainTextChunker:
    def test_short_text_is_one_chunk(self) -> None:
        text = "Short document."
        chunks = PlainTextChunker().chunk(text, {})
        assert len(chunks) == 1
        assert chunks[0].text == "Short document."

    def test_chunk_sizes_within_max_tokens(self) -> None:
        # Use overlap_tokens=0 so body size is bounded purely by max_tokens.
        # With overlap > 0, chunks legitimately reach max_tokens + overlap_tokens.
        words = ["word"] * 600
        text = " ".join(words)
        chunks = PlainTextChunker(max_tokens=64, overlap_tokens=0).chunk(text, {})
        assert len(chunks) > 1
        for c in chunks:
            # One extra word may push a chunk past max_tokens at a word boundary.
            assert c.token_count <= 64 + 5

    def test_paragraph_splits_preferred_over_sentence_splits(self) -> None:
        paragraphs = [f"Paragraph {i}. It has some words." for i in range(10)]
        text = "\n\n".join(paragraphs)
        chunks = PlainTextChunker(max_tokens=30).chunk(text, {})
        # Each chunk should be at a paragraph boundary, not mid-paragraph.
        for c in chunks:
            assert "Paragraph" in c.text

    def test_overlap_present_between_chunks(self) -> None:
        words = ["unique_word"] + [f"w{i}" for i in range(200)]
        text = " ".join(words)
        chunks = PlainTextChunker(max_tokens=40, overlap_tokens=10).chunk(text, {})
        assert len(chunks) >= 2
        # Overlap words from chunk[0] tail should appear in chunk[1] head.
        tail = set(chunks[0].text.split()[-5:])
        head = set(chunks[1].text.split()[:15])
        assert tail & head

    def test_empty_text_returns_empty(self) -> None:
        assert PlainTextChunker().chunk("", {}) == []
        assert PlainTextChunker().chunk("\n\n", {}) == []

    def test_text_with_no_sentence_boundaries(self) -> None:
        # A wall of words with no punctuation — falls back to word splitting.
        text = " ".join([f"longword{i}" for i in range(200)])
        chunks = PlainTextChunker(max_tokens=32).chunk(text, {})
        assert len(chunks) > 1

    def test_token_count_populated(self) -> None:
        chunks = PlainTextChunker().chunk("Hello world.", {})
        assert all(c.token_count > 0 for c in chunks)

    def test_metadata_passed_through(self) -> None:
        meta = {"source": "test"}
        chunks = PlainTextChunker().chunk("Some text.", meta)
        assert all(c.metadata == meta for c in chunks)
