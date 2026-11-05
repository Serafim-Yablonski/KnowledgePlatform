"""Real PDFs with a text layer, built with pypdf (no fixture files to keep in sync)."""

import io

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_with_text(*pages: str) -> bytes:
    """One page per argument, each showing its text in Helvetica (ASCII only)."""
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in pages:
        page = writer.add_blank_page(612, 792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# A ToUnicode map as broken PDFs carry them: one valid astral character split across
# two codes (A, B), the same character whole (C), a lone surrogate (D) and NUL (E).
AWKWARD_TO_UNICODE = b"""/CIDInit /ProcSet findresource begin 12 dict begin begincmap
/CMapName /Awkward def 1 begincodespacerange <00> <FF> endcodespacerange
5 beginbfchar <41> <D83D> <42> <DE00> <43> <D83DDE00> <44> <D800> <45> <0000>
endbfchar endcmap CMapName currentdict /CMap defineresource pop end end"""


def pdf_showing(string: bytes, *, to_unicode: bytes | None = None) -> bytes:
    """One page showing ``string``, a PDF literal string body such as rb"A\\000B",
    in Helvetica; with ``to_unicode``, through that ToUnicode CMap."""
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    if to_unicode is not None:
        cmap = DecodedStreamObject()
        cmap.set_data(to_unicode)
        font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    page = writer.add_blank_page(612, 792)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (" + string + b") Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
