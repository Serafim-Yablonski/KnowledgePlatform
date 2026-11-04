"""Chunking (ADR-031): every chunk fits the token bound with its overlap, chunks are
the input's own text in order with none of it lost, and markdown sections follow the
headings outside code fences. Properties are checked on generated documents built
from what broke the old chunkers: a paragraph longer than a chunk among short ones,
runs with no break at all, '## ' inside fences, tables and CJK."""

from __future__ import annotations

import pytest
import tiktoken
from hypothesis import given, settings
from hypothesis import strategies as st

from src.ai.chunking import ChunkingStrategy
from src.ai.chunking.markdown import MarkdownChunker
from src.ai.chunking.plaintext import PlainTextChunker

# The bound is counted in cl100k tokens, as the chunkers count it.
_ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_ENC.encode_ordinary(text))


# Strings the tokenizer reserves (<|endoftext|>…): in a document they are text.
_SPECIAL = sorted(_ENC.special_tokens_set)


_word = st.text(
    alphabet=st.characters(categories=("L", "N", "P", "S")), min_size=1, max_size=12
)
_blocks = st.one_of(
    st.text(max_size=200),
    st.lists(_word, max_size=150).map(" ".join),
    st.text(alphabet="ABCDEFabcdef0123456789+/=", min_size=100, max_size=2000),
    st.lists(_word, max_size=40).map(lambda ws: "```\n## " + " ".join(ws) + "\n```"),
    st.lists(_word, min_size=1, max_size=6).map(
        lambda ws: "| " + " | ".join(ws) + " |"
    ),
    st.text(alphabet=st.characters(min_codepoint=0x4E00, max_codepoint=0x9FFF)),
    st.sampled_from(["## Setup", "### Install", "# Title", "#### Deep", "```", "~~~"]),
    st.sampled_from(_SPECIAL),
)
documents = st.lists(
    st.tuples(_blocks, st.sampled_from(["\n", "\n\n", " ", ""])), max_size=12
).map(lambda parts: "".join(block + sep for block, sep in parts))

# A small bound splits small documents; the production one runs on the same inputs.
CHUNKERS = [
    pytest.param(PlainTextChunker(max_tokens=32, overlap_tokens=8), 32, id="text-32"),
    pytest.param(MarkdownChunker(max_tokens=32, overlap_tokens=8), 32, id="md-32"),
    pytest.param(PlainTextChunker(), 512, id="text-512"),
    pytest.param(MarkdownChunker(), 512, id="md-512"),
]


@pytest.mark.parametrize(("chunker", "bound"), CHUNKERS)
@settings(deadline=None)
@given(text=documents)
def test_every_chunk_fits_the_bound_overlap_included(
    chunker: ChunkingStrategy, bound: int, text: str
) -> None:
    for chunk in chunker.chunk(text, {}):
        assert _tokens(chunk.text) <= bound
        assert chunk.token_count == _tokens(chunk.text)


@pytest.mark.parametrize(("chunker", "bound"), CHUNKERS)
@settings(deadline=None)
@given(text=documents)
def test_chunks_are_the_input_in_order_and_lose_no_text(
    chunker: PlainTextChunker | MarkdownChunker, bound: int, text: str
) -> None:
    spans = chunker.spans(text)

    covered: set[int] = set()
    for start, chunk in spans:
        # Never whitespace alone; markdown keeps a chunk's leading indentation.
        assert chunk.strip()
        assert text[start : start + len(chunk)] == chunk
        covered.update(range(start, start + len(chunk)))
    starts = [start for start, _ in spans]
    assert starts == sorted(set(starts))
    lost = [i for i, char in enumerate(text) if not char.isspace() and i not in covered]
    assert lost == []
    assert [c.text for c in chunker.chunk(text, {})] == [chunk for _, chunk in spans]


@pytest.mark.parametrize("chunker", [PlainTextChunker(), MarkdownChunker()])
@pytest.mark.parametrize("special", _SPECIAL)
def test_a_tokenizers_reserved_string_is_ordinary_text(
    chunker: ChunkingStrategy, special: str
) -> None:
    # Notes about LLMs quote them; counting them as specials raised, and the
    # document failed for good as internal_error.
    text = f"GPT-2 ends each document with {special} in training data."

    [chunk] = chunker.chunk(text, {})

    assert chunk.text == text
    assert chunk.token_count == _tokens(text)


@pytest.mark.parametrize("chunker", [PlainTextChunker(), MarkdownChunker()])
def test_whitespace_alone_is_no_chunk(chunker: ChunkingStrategy) -> None:
    # A scanned PDF extracts to page separators only: no chunk, so FAILED(no_text).
    assert chunker.chunk("", {}) == []
    assert chunker.chunk("\n\n\n \t\f", {}) == []


@pytest.mark.parametrize(
    "chunker",
    [
        PlainTextChunker(max_tokens=40, overlap_tokens=10),
        MarkdownChunker(max_tokens=40, overlap_tokens=10),
    ],
)
def test_neighbouring_chunks_repeat_a_little_text(chunker: ChunkingStrategy) -> None:
    # read_document's description promises it: a seam never cuts a thought in two.
    text = " ".join(f"Sentence number {i} ends here." for i in range(60))
    spans = chunker.spans(text)  # type: ignore[attr-defined]

    assert len(spans) > 2
    for (start, chunk), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert next_start < start + len(chunk)


def test_plaintext_chunks_carry_the_metadata_given() -> None:
    meta = {"document_id": "d"}
    chunks = PlainTextChunker(max_tokens=8, overlap_tokens=2).chunk(
        "Some words. " * 20, meta
    )
    assert len(chunks) > 1 and all(c.metadata == meta for c in chunks)


# ---------------------------------------------------------------------------
# Markdown sections
# ---------------------------------------------------------------------------


def _sections(text: str, **kwargs: int) -> list[tuple[str, str]]:
    return [
        (c.metadata["section"], c.text)
        for c in MarkdownChunker(**kwargs).chunk(text, {})
    ]


def test_each_section_is_its_own_chunk_with_its_heading_path() -> None:
    # Small sections are not merged: a chunk's section is exact, and its heading is
    # part of its text, so reading the chunks in order reads the document.
    text = "## Setup\nA.\n\n### Prerequisites\nB.\n\n## Usage\nC.\n\n### Flags\nD.\n"

    assert _sections(text) == [
        ("## Setup", "## Setup\nA."),
        ("## Setup > ### Prerequisites", "### Prerequisites\nB."),
        ("## Usage", "## Usage\nC."),
        ("## Usage > ### Flags", "### Flags\nD."),
    ]


def test_a_heading_with_nothing_under_it_joins_the_next_section() -> None:
    text = "## Setup\n\n### Prerequisites\nInstall.\n"

    assert _sections(text) == [
        ("## Setup > ### Prerequisites", "## Setup\n\n### Prerequisites\nInstall.")
    ]


@pytest.mark.parametrize("fence", ["```", "~~~", "````"])
def test_a_heading_inside_a_code_fence_is_code(fence: str) -> None:
    text = (
        f"## Deploy\n{fence}bash\n## build the image\ndocker build .\n{fence}\nafter\n"
    )

    assert _sections(text) == [("## Deploy", text.strip())]


@pytest.mark.parametrize("inner", ["```", "~~~~", "```` info"])
def test_a_fence_closes_only_on_its_own_kind_at_least_as_long(inner: str) -> None:
    text = f"## Deploy\n````\n{inner}\n## still code\n````\nafter\n"

    assert _sections(text) == [("## Deploy", text.strip())]


def test_an_unclosed_fence_runs_to_the_end() -> None:
    text = "## Deploy\n```\n## not a heading\n"

    assert _sections(text) == [("## Deploy", text.strip())]


def test_only_level_two_and_three_headings_start_sections() -> None:
    text = "# Title\nIntro.\n\n#### Detail\nMore.\n\n### Alone\nX.\n"

    assert _sections(text) == [
        ("", "# Title\nIntro.\n\n#### Detail\nMore."),
        ("### Alone", "### Alone\nX."),
    ]


def test_a_long_section_splits_and_keeps_its_path() -> None:
    body = " ".join(f"Sentence {i} is here." for i in range(80))
    sections = _sections(f"## A\n### B\n{body}\n", max_tokens=32, overlap_tokens=8)

    assert len(sections) > 2
    assert {section for section, _ in sections} == {"## A > ### B"}


def test_a_code_block_that_fits_is_never_cut() -> None:
    # Markdown's own boundaries first: plain text splitting would cut this block at
    # its blank line, as a paragraph break.
    code = (
        "```python\ndef build():\n    run('docker build .')\n\n\n"
        "def push():\n    run('docker push')\n```"
    )
    text = (
        "Build the image first, then push it to the registry used by the cluster."
        f"\n\n{code}\n\nAfter pushing, deploy with the usual command."
    )

    chunks = [c for _, c in _sections(text, max_tokens=40, overlap_tokens=0)]

    assert len(chunks) > 1
    assert any(code in chunk for chunk in chunks)
