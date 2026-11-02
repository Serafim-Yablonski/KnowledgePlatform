"""Extract a PDF's text in a child process: the file's bytes on stdin, UTF-8 text on
stdout, and the exit code saying whose fault a failure is.

extract_text runs this file by path, as ``python -I pdf_text.py``. pypdf parses
untrusted input: it can catch the soft time limit and return partial text (S6-10),
loop, or crash. In a child, a timeout is a kill nothing can swallow, and a crash takes
only the child. Imports only pypdf and the standard library: under -I neither
PYTHONPATH, the working directory nor this file's directory is on sys.path.
"""

from __future__ import annotations

import io
import logging
import sys

from pypdf import PdfReader

# pypdf raised: the file's fault. Any other non-zero exit is ours (e.g. 1, an uncaught
# exception such as a failed import).
PARSE_FAILED = 3


def storable(text: str) -> str:
    """``text`` with what PostgreSQL text can't hold replaced by U+FFFD: NUL, and
    surrogates that pair with nothing. Both are glyph-mapping artifacts; one of them
    shouldn't cost the document. pypdf can return a valid astral character as two
    separate surrogates (a ToUnicode map split across codes): the UTF-16 round trip
    joins those instead of replacing them."""
    joined = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return joined.replace("\x00", "\ufffd")


def main() -> int:
    # pypdf's warnings quote the file's bytes; stderr carries only our one line.
    logging.disable(logging.CRITICAL)
    data = sys.stdin.buffer.read()
    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # noqa: BLE001 — pypdf's exceptions are an open set (S6-3)
        sys.stderr.write(type(exc).__name__)
        return PARSE_FAILED
    sys.stdout.buffer.write(storable(text).encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
