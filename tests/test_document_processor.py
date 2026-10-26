import pytest

import config
from document_processor import (
    CorruptFileError,
    EmptyDocumentError,
    FileTooLargeError,
    ScannedPdfError,
    UnsupportedFileTypeError,
    load_pasted_text,
    load_uploaded_file,
    normalise_text,
)


# --- Empty input ---------------------------------------------------------------
@pytest.mark.parametrize("value", ["", "   \n\t  ", None])
def test_empty_pasted_text_is_rejected(value):
    with pytest.raises(EmptyDocumentError):
        load_pasted_text(value)


def test_empty_file_is_rejected():
    with pytest.raises(EmptyDocumentError):
        load_uploaded_file("empty.txt", b"")


def test_whitespace_only_txt_is_rejected():
    with pytest.raises(EmptyDocumentError):
        load_uploaded_file("blank.txt", b"   \n \n  ")


# --- Valid input ---------------------------------------------------------------
def test_valid_pasted_text_is_normalised():
    doc = load_pasted_text("Line one.\r\n\r\n\r\n\r\nLine   two. End.")
    assert doc.text == "Line one.\n\nLine two. End."
    assert not doc.truncated and doc.source_type == "Pasted text"


def test_txt_upload_utf8_and_fallback_encoding():
    assert load_uploaded_file("a.txt", "Café judgment text.".encode("utf-8")).text == "Café judgment text."
    doc = load_uploaded_file("b.TXT", "Caf\xe9 text".encode("cp1252"))
    assert "Café" in doc.text
    assert any("Windows-1252" in n for n in doc.notes)  # the decoding fallback is disclosed


# --- Invalid input -------------------------------------------------------------
@pytest.mark.parametrize("name", ["judgment.docx", "image.png", "noextension", ""])
def test_unsupported_extension(name):
    with pytest.raises(UnsupportedFileTypeError):
        load_uploaded_file(name, b"some bytes")


def test_file_too_large():
    with pytest.raises(FileTooLargeError):
        load_uploaded_file("big.txt", b"a" * 2048, max_bytes=1024)


def test_binary_data_named_txt_is_rejected():
    with pytest.raises(CorruptFileError):
        load_uploaded_file("fake.txt", b"\x00\x01\x02\x00" * 500)


def test_pdf_bytes_named_txt_is_rejected(pdf_factory):
    with pytest.raises(UnsupportedFileTypeError):
        load_uploaded_file("really_a_pdf.txt", pdf_factory([["Hello world"]]))


@pytest.mark.parametrize("payload", [b"this is not a pdf at all", b"%PDF-1.7\ngarbage garbage garbage"])
def test_corrupted_pdf_does_not_crash(payload):
    with pytest.raises(CorruptFileError):
        load_uploaded_file("broken.pdf", payload)


# --- PDF extraction ------------------------------------------------------------
def test_pdf_text_extraction_with_page_mapping(pdf_factory):
    data = pdf_factory([
        ["IN THE HIGH COURT OF SAMPLE STATE", "This is the first page of a synthetic test document."],
        ["SECOND PAGE MARKER", "More synthetic content appears on the second page of the document."],
    ])
    doc = load_uploaded_file("synthetic.pdf", data)
    assert doc.page_count == 2
    assert "first page" in doc.text and "SECOND PAGE MARKER" in doc.text
    offset = doc.text.index("SECOND PAGE MARKER")
    assert "page 2" in doc.locate(offset, offset + 10)
    assert "page 1" in doc.locate(doc.text.index("HIGH COURT"))


def test_scanned_pdf_gives_helpful_message(pdf_factory):
    blank = pdf_factory([[], []])  # pages with no text layer, like a scan
    with pytest.raises(ScannedPdfError) as info:
        load_uploaded_file("scan.pdf", blank)
    assert "OCR is not available" in str(info.value)


def test_partially_empty_pdf_discloses_missing_pages(pdf_factory):
    text_page = ["A synthetic page with enough readable text to pass the minimum length check easily."]
    doc = load_uploaded_file("mixed.pdf", pdf_factory([text_page, [], text_page]))
    assert any("no extractable text" in n for n in doc.notes)


# --- Long documents / limits ---------------------------------------------------
def test_long_text_truncation_is_disclosed_not_silent():
    doc = load_pasted_text("Sentence number one. " * 500, max_chars=1000)
    assert doc.truncated and len(doc.text) <= 1000
    assert doc.original_chars > 1000
    assert any("NOT analysed" in n for n in doc.notes)


def test_pdf_page_limit_is_disclosed(pdf_factory):
    page = ["Synthetic page text that is long enough to be accepted as real extractable content."]
    doc = load_uploaded_file("many.pdf", pdf_factory([page] * 4), max_pages=2)
    assert doc.truncated and doc.page_count == 4
    assert any("first 2" in n for n in doc.notes)


def test_normalise_text_preserves_wording():
    assert normalise_text("Section 302 IPC\x00 and\r\nSection 34.") == "Section 302 IPC and\nSection 34."


def test_config_limits_are_positive():
    assert config.MAX_FILE_BYTES > 0 and config.MAX_TEXT_CHARS > 0
