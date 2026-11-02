"""The PDF child process: its exit contract, and how extract_text reads each exit."""

import io
import logging
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path

import pytest

from src.core.config import settings
from src.domain.documents import FailureReason
from src.workers import pdf_text
from src.workers.pipeline import ParseKilled, PermanentFailure
from tests.pdfs import AWKWARD_TO_UNICODE, pdf_showing, pdf_with_text

_DAMAGED_PDF = b"%PDF-1.4\nnot a pdf at all\n"


@pytest.fixture(autouse=True)
def _restore_logging() -> Generator[None]:
    # The child silences logging for its whole process; in-process, undo it.
    yield
    logging.disable(logging.NOTSET)


def _run_main(monkeypatch: pytest.MonkeyPatch, data: bytes) -> tuple[int, bytes, str]:
    stdout = io.TextIOWrapper(io.BytesIO())
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(data)))
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    code = pdf_text.main()
    stdout.flush()
    return code, stdout.buffer.getvalue(), stderr.getvalue()  # type: ignore[attr-defined]


def _run_child(data: bytes, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    # Exactly as extract_text runs it: by path, isolated (-I: no PYTHONPATH, no
    # working directory or script directory on sys.path).
    return subprocess.run(
        [sys.executable, "-I", str(Path(pdf_text.__file__))],
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


# ---------------------------------------------------------------------------
# How extract_text reads the child's exit
# ---------------------------------------------------------------------------


def _parse_with_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> str:
    from src.workers.tasks import extract_text as extract_mod

    child = tmp_path / "child.py"
    child.write_text(body)
    monkeypatch.setattr(sys.modules[extract_mod.__name__], "_PDF_CHILD", child)
    parse = sys.modules[extract_mod.__name__]._parse_pdf
    return parse(b"%PDF-1.4 stand-in")  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    "body",
    [
        f"import sys; sys.exit({pdf_text.PARSE_FAILED})",
        # A crash while parsing: the same bytes crash it again.
        "import os, signal; os.kill(os.getpid(), signal.SIGSEGV)",
        "import os, signal; os.kill(os.getpid(), signal.SIGABRT)",
    ],
    ids=["parse-failed", "segfault", "abort"],
)
def test_a_child_that_failed_on_the_file_means_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, body: str
) -> None:
    with pytest.raises(PermanentFailure) as exc_info:
        _parse_with_child(monkeypatch, tmp_path, body)

    assert exc_info.value.reason is FailureReason.UNREADABLE_FILE


def test_a_child_killed_by_sigkill_is_an_attempt_not_a_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Our own timeout kill is TimeoutExpired; any other SIGKILL is the OOM killer
    under the extract container's cap, whose victim depends on what else is parsing
    (S8-18). Retried; only a parse killed on every attempt ends extraction_too_large."""
    with pytest.raises(ParseKilled):
        _parse_with_child(
            monkeypatch, tmp_path, "import os, signal; os.kill(os.getpid(), 9)"
        )


def test_a_child_that_died_for_our_reasons_is_retried(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Exit 1: an uncaught exception before the parse (a broken deploy, a missing
    # module). Not the user's file: not permanent, so autoretry retries it.
    with pytest.raises(Exception) as exc_info:
        _parse_with_child(monkeypatch, tmp_path, "raise ImportError('no pypdf')")

    assert not isinstance(exc_info.value, PermanentFailure)
    assert "exited with 1" in str(exc_info.value)


def test_a_parse_past_its_timeout_is_killed_as_extraction_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings_override: pytest.MonkeyPatch,
) -> None:
    settings_override.setattr(settings, "CELERY_EXTRACT_PARSE_TIMEOUT", 1)
    started = time.monotonic()

    with pytest.raises(PermanentFailure) as exc_info:
        _parse_with_child(monkeypatch, tmp_path, "import time; time.sleep(30)")

    assert exc_info.value.reason is FailureReason.EXTRACTION_TIMEOUT
    assert time.monotonic() - started < 10  # killed, not waited for


def test_the_child_s_text_is_returned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    text = _parse_with_child(
        monkeypatch,
        tmp_path,
        "import sys\n"
        "sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write('Grüße'.encode())",
    )

    assert text == "Grüße"
