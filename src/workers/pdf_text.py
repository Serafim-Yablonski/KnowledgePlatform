"""Extract a PDF's text in a child process: the file's bytes on stdin, UTF-8 text on
stdout, and the exit code saying whose fault a failure is.

extract_text runs this file by path, as ``python -I pdf_text.py <max_bytes>``.
pypdf parses untrusted input: it can swallow an exception meant to stop it and
return partial text (under Celery, the soft time limit: S6-10), loop, or crash. In
a child, a timeout is a kill nothing can swallow, and a crash takes only the child.
Imports only pypdf and the standard library: under -I neither PYTHONPATH, the
working directory nor this file's directory is on sys.path.
"""

from __future__ import annotations

import io
import logging
import sys

from pypdf import PdfReader

# pypdf raised: the file's fault. Any other non-zero exit is ours (e.g. 1, an uncaught
# exception such as a failed import).
PARSE_FAILED = 3
# The text passed ``max_bytes``: a small compressed PDF can extract to far more text
# than any upload, so the cap bounds the work done, not just the bytes received.
TOO_LARGE = 4


def storable(text: str) -> str:
    """``text`` with what PostgreSQL text can't hold replaced by U+FFFD: NUL, and
    surrogates that pair with nothing. Both are glyph-mapping artifacts; one of them
    shouldn't cost the document. pypdf can return a valid astral character as two
    separate surrogates (a ToUnicode map split across codes): the UTF-16 round trip
    joins those instead of replacing them."""
    joined = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return joined.replace("\x00", "\ufffd")


def main(max_bytes: int) -> int:
    """The pages' text, joined by newlines; the parse stops at the page whose text
    passes ``max_bytes`` (UTF-8), and nothing is written."""
    # pypdf's warnings quote the file's bytes; stderr carries only our one line.
    logging.disable(logging.CRITICAL)
    data = sys.stdin.buffer.read()
    pages: list[bytes] = []
    size = 0
    try:
        for page in PdfReader(io.BytesIO(data)).pages:
            # Page by page: the newline between pages keeps a surrogate half at a
            # page's end apart from the next page's anyway.
            text = storable(page.extract_text() or "").encode("utf-8")
            size += len(text) + (1 if pages else 0)
            if size > max_bytes:
                sys.stderr.write(f"text past {max_bytes} bytes")
                return TOO_LARGE
            pages.append(text)
    except Exception as exc:  # noqa: BLE001 — pypdf's exceptions are an open set (S6-3)
        sys.stderr.write(type(exc).__name__)
        return PARSE_FAILED
    sys.stdout.buffer.write(b"\n".join(pages))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1])))
