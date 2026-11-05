"""The PDF child process: its exit contract (how the extract job reads each exit:
tests/integration/test_jobs.py)."""

import io
import logging
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from pypdf import PageObject

from src.workers import pdf_text
from tests.pdfs import AWKWARD_TO_UNICODE, pdf_showing, pdf_with_text

_DAMAGED_PDF = b"%PDF-1.4\nnot a pdf at all\n"


@pytest.fixture(autouse=True)
def _restore_logging() -> Generator[None]:
    # The child silences logging for its whole process; in-process, undo it.
    yield
    logging.disable(logging.NOTSET)


_NO_CAP = 1 << 40


def _run_main(
    monkeypatch: pytest.MonkeyPatch, data: bytes, max_bytes: int = _NO_CAP
) -> tuple[int, bytes, str]:
    stdout = io.TextIOWrapper(io.BytesIO())
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = pdf_text.main(max_bytes)
    stdout.flush()
    return code, stdout.buffer.getvalue(), stderr.getvalue()  # type: ignore[attr-defined]


def _run_child(
    data: bytes, cwd: Path, max_bytes: int = _NO_CAP
) -> subprocess.CompletedProcess[bytes]:
    # Exactly as extract_text runs it: by path, isolated (-I: no PYTHONPATH, no
    # working directory or script directory on sys.path), the cap as its argument.
    return subprocess.run(
        [sys.executable, "-I", str(Path(pdf_text.__file__)), str(max_bytes)],
        input=data,
        capture_output=True,
        cwd=cwd,
        timeout=30,
        check=False,
    )


# ---------------------------------------------------------------------------
# The child
# ---------------------------------------------------------------------------


def test_the_child_runs_by_path_in_isolation_and_writes_utf8_text(
    tmp_path: Path,
) -> None:
    done = _run_child(pdf_with_text("Hello from the child", "Page two"), tmp_path)

    assert done.returncode == 0, done.stderr
    assert done.stdout.decode("utf-8") == "Hello from the child\nPage two"


def test_text_past_the_cap_exits_too_large_with_nothing_written(
    tmp_path: Path,
) -> None:
    # A small compressed PDF can extract to far more text than any upload (R4).
    pdf = pdf_with_text("x" * 100, "y" * 100)
    size = len(_run_child(pdf, tmp_path).stdout)

    at_cap = _run_child(pdf, tmp_path, max_bytes=size)
    past_cap = _run_child(pdf, tmp_path, max_bytes=size - 1)

    assert (at_cap.returncode, len(at_cap.stdout)) == (0, size)
    assert (past_cap.returncode, past_cap.stdout) == (pdf_text.TOO_LARGE, b"")


def test_the_child_stops_parsing_at_the_page_that_passes_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cap bounds the work, not just the output: later pages are never parsed.
    parsed: list[str] = []
    extract = PageObject.extract_text

    def _counting(page: PageObject, *args: object, **kwargs: object) -> str:
        text = extract(page, *args, **kwargs)  # type: ignore[arg-type]
        parsed.append(text)
        return text

    monkeypatch.setattr(PageObject, "extract_text", _counting)

    code, stdout, _ = _run_main(monkeypatch, pdf_with_text("a" * 50, "b", "c"), 10)

    assert (code, stdout, len(parsed)) == (pdf_text.TOO_LARGE, b"", 1)


def test_a_damaged_pdf_exits_parse_failed_and_stderr_names_only_the_type(
    tmp_path: Path,
) -> None:
    # pypdf logs warnings that quote the file's bytes; stderr stays ours.
    done = _run_child(_DAMAGED_PDF, tmp_path)

    assert done.returncode == pdf_text.PARSE_FAILED
    assert (done.stdout, done.stderr) == (b"", b"PdfStreamError")


def test_any_exception_from_pypdf_is_the_files_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # pypdf raises more than PdfReadError on damaged files (S6-3); none is listed.
    def _pypdf_bug(_stream: object) -> None:
        raise TypeError("'NoneType' object is not subscriptable")

    monkeypatch.setattr(pdf_text, "PdfReader", _pypdf_bug)

    code, stdout, stderr = _run_main(monkeypatch, pdf_with_text("x"))

    assert (code, stdout, stderr) == (pdf_text.PARSE_FAILED, b"", "TypeError")


@pytest.mark.parametrize(
    ("extracted", "stored"),
    [
        ("A\x00B", "A\ufffdB"),  # PostgreSQL text can't hold NUL
        ("x\ud800y", "x\ufffdy"),  # a lone high surrogate
        ("\udc00", "\ufffd"),  # a lone low surrogate
        # A valid astral character pypdf returned as two code points (S6-12): joined,
        # not replaced; replacing each code point would lose it.
        ("\ud83d\ude00", "\U0001f600"),
        ("Grüße \U0001f600 漢字", "Grüße \U0001f600 漢字"),
        ("", ""),
    ],
    ids=["nul", "lone-high", "lone-low", "split-pair", "ordinary", "empty"],
)
def test_text_becomes_storable(extracted: str, stored: str) -> None:
    assert pdf_text.storable(extracted) == stored


@pytest.mark.parametrize(
    ("pdf", "text"),
    [
        (pdf_showing(rb"A\000B"), "A\ufffdB"),
        (
            pdf_showing(b"ABCDE", to_unicode=AWKWARD_TO_UNICODE),
            "\U0001f600\U0001f600\ufffd\ufffd",
        ),
    ],
    ids=["nul-in-a-string", "awkward-to-unicode"],
)
def test_the_child_writes_only_what_postgres_can_store(
    tmp_path: Path, pdf: bytes, text: str
) -> None:
    done = _run_child(pdf, tmp_path)

    assert done.returncode == 0, done.stderr
    assert done.stdout.decode("utf-8") == text
