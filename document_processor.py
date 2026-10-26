"""Loading, validating and normalising judgment text (pasted, TXT or PDF).

Nothing here writes uploaded content to disk: files are processed from
in-memory bytes only.
"""
from __future__ import annotations

import bisect
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import config

logger = logging.getLogger(__name__)


# --- Exceptions (messages are written to be shown directly to the user) ------
class DocumentError(Exception):
    """Base class for input problems. ``str(exc)`` is safe to display."""


class UnsupportedFileTypeError(DocumentError):
    pass


class EmptyDocumentError(DocumentError):
    pass


class ScannedPdfError(DocumentError):
    pass


class CorruptFileError(DocumentError):
    pass


class FileTooLargeError(DocumentError):
    pass


# --- Data model ---------------------------------------------------------------
@dataclass
class Document:
    """Extracted text plus the metadata needed to cite where things came from."""

    text: str
    source_name: str
    source_type: str  # "Pasted text" | "TXT file" | "PDF file"
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)  # (page, start, end)
    page_count: int | None = None
    truncated: bool = False
    original_chars: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        """Short content hash used to detect when the input has changed."""
        return hashlib.sha256(self.text.encode("utf-8", "ignore")).hexdigest()[:12]

    def locate(self, start: int, end: int | None = None) -> str:
        """Human-readable location of a character range in the extracted text."""
        end = start if end is None else end
        span = f"characters {start:,}–{end:,} of extracted text"
        page = self._page_at(start)
        return f"page {page}, {span}" if page else span

    def _page_at(self, offset: int) -> int | None:
        if not self.page_spans:
            return None
        starts = [s for _, s, _ in self.page_spans]
        idx = bisect.bisect_right(starts, offset) - 1
        return self.page_spans[idx][0] if idx >= 0 else None


# --- Text normalisation ---------------------------------------------------------
_HSPACE_RE = re.compile(r"[ \t\f\v]+")
_MANY_NL_RE = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    """Tidy line endings and whitespace without altering the wording."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "").replace(" ", " ")
    text = _HSPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NL_RE.sub("\n\n", text).strip()


def _apply_char_limit(doc: Document, max_chars: int) -> Document:
    """Cut the text to *max_chars* - and say so loudly (never silently)."""
    doc.original_chars = len(doc.text)
    if len(doc.text) <= max_chars:
        return doc
    doc.text = doc.text[:max_chars].rstrip()
    doc.truncated = True
    doc.page_spans = [
        (p, s, min(e, len(doc.text))) for p, s, e in doc.page_spans if s < len(doc.text)
    ]
    doc.notes.append(
        f"The document has {doc.original_chars:,} characters; only the first "
        f"{len(doc.text):,} were analysed (limit: {max_chars:,}). "
        "Content after that point was NOT analysed."
    )
    return doc


# --- Public loaders ---------------------------------------------------------------
def load_pasted_text(text: str, max_chars: int | None = None) -> Document:
    """Build a Document from text pasted into the UI."""
    if text is None or not text.strip():
        raise EmptyDocumentError("No text was provided. Paste a judgment or upload a file.")
    doc = Document(text=normalise_text(text), source_name="pasted text", source_type="Pasted text")
    if not doc.text:
        raise EmptyDocumentError("The pasted text contains no readable characters.")
    return _apply_char_limit(doc, max_chars or config.MAX_TEXT_CHARS)


def load_uploaded_file(
    filename: str,
    data: bytes,
    max_bytes: int | None = None,
    max_chars: int | None = None,
    max_pages: int | None = None,
) -> Document:
    """Validate and read an uploaded PDF or TXT file from memory."""
    max_bytes = max_bytes or config.MAX_FILE_BYTES
    max_chars = max_chars or config.MAX_TEXT_CHARS
    max_pages = max_pages or config.MAX_PDF_PAGES

    extension = Path(filename or "").suffix.lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Please upload one of: {', '.join(config.ALLOWED_EXTENSIONS)}."
        )
    if not data:
        raise EmptyDocumentError(f"'{filename}' is empty (0 bytes).")
    if len(data) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(data) / 1_048_576:.1f} MB; the limit is {max_bytes / 1_048_576:.0f} MB."
        )

    if extension == ".pdf":
        doc = _read_pdf(filename, data, max_pages)
    else:
        doc = _read_txt(filename, data)
    return _apply_char_limit(doc, max_chars)


# --- TXT --------------------------------------------------------------------------
def _decode_text_bytes(data: bytes) -> tuple[str, str | None]:
    """Decode bytes to text, returning (text, optional note about the encoding)."""
    if data.startswith(b"%PDF-"):
        raise UnsupportedFileTypeError(
            "This file has a .txt name but contains PDF data. Rename it to .pdf and upload it again."
        )
    has_utf16_bom = data.startswith((b"\xff\xfe", b"\xfe\xff"))
    if not has_utf16_bom and data.count(b"\x00") > max(1, len(data) // 100):
        raise CorruptFileError("This file looks like binary data, not plain text.")
    if has_utf16_bom:
        return data.decode("utf-16", errors="replace"), None
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "File is not valid UTF-8; decoded as Windows-1252. Some characters may differ."
    except UnicodeDecodeError:
        return data.decode("latin-1"), "File is not valid UTF-8; decoded as Latin-1. Some characters may differ."


def _read_txt(filename: str, data: bytes) -> Document:
    text, note = _decode_text_bytes(data)
    text = normalise_text(text)
    if not text:
        raise EmptyDocumentError(f"'{filename}' contains no readable text.")
    doc = Document(text=text, source_name=filename, source_type="TXT file")
    if note:
        doc.notes.append(note)
    return doc


# --- PDF --------------------------------------------------------------------------
def _read_pdf(filename: str, data: bytes, max_pages: int) -> Document:
    if b"%PDF-" not in data[:1024]:
        raise CorruptFileError(f"'{filename}' does not look like a valid PDF file.")
    try:
        import pymupdf  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DocumentError("PDF support requires PyMuPDF. Install it with: pip install pymupdf") from exc

    try:
        pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("PDF open failed: %s", type(exc).__name__)
        raise CorruptFileError(f"'{filename}' could not be opened. It may be corrupted.") from exc

    with pdf:
        if pdf.needs_pass:
            raise CorruptFileError(f"'{filename}' is password-protected. Remove the password and try again.")
        total_pages = pdf.page_count
        if total_pages == 0:
            raise CorruptFileError(f"'{filename}' has no pages.")

        pages_to_read = min(total_pages, max_pages)
        parts: list[str] = []
        spans: list[tuple[int, int, int]] = []
        empty_pages: list[int] = []
        cursor = 0
        for index in range(pages_to_read):
            try:
                page_text = normalise_text(pdf[index].get_text("text"))
            except Exception as exc:
                logger.warning("Page %d could not be read: %s", index + 1, type(exc).__name__)
                page_text = ""
            if not page_text:
                empty_pages.append(index + 1)
                continue
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            spans.append((index + 1, start, cursor))
            cursor += 2  # the "\n\n" separator inserted below

    text = "\n\n".join(parts)
    if len(text.strip()) < config.MIN_PDF_TEXT_CHARS:
        raise ScannedPdfError(
            f"No extractable text was found in '{filename}'. It is probably a scanned (image-only) PDF. "
            "OCR is not available in this basic version. Please use a text-based PDF, run the file "
            "through an OCR tool first, or paste the text directly."
        )

    doc = Document(
        text=text,
        source_name=filename,
        source_type="PDF file",
        page_spans=spans,
        page_count=total_pages,
    )
    if total_pages > pages_to_read:
        doc.truncated = True
        doc.notes.append(
            f"The PDF has {total_pages} pages; only the first {pages_to_read} were read "
            f"(limit: {max_pages}). Later pages were NOT analysed."
        )
    if empty_pages:
        shown = ", ".join(map(str, empty_pages[:15])) + ("…" if len(empty_pages) > 15 else "")
        doc.notes.append(
            f"{len(empty_pages)} page(s) had no extractable text (possibly scanned images): {shown}. "
            "Their content is NOT included in the analysis."
        )
    return doc
