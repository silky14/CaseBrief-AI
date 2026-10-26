"""Shared fixtures. All sample texts are SYNTHETIC and fictional - they are not real judgments."""
from pathlib import Path

import pytest

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "sample_data" / "synthetic_sample_judgment.txt"


@pytest.fixture(scope="session")
def sample_text() -> str:
    return SAMPLE_PATH.read_text(encoding="utf-8")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Build a small text PDF in memory. *pages* is a list of pages, each a list of lines."""
    import pymupdf

    pdf = pymupdf.open()
    for lines in pages:
        page = pdf.new_page()
        for i, line in enumerate(lines):
            page.insert_text((50, 60 + 14 * i), line, fontsize=10)
    data = pdf.tobytes()
    pdf.close()
    return data


@pytest.fixture
def pdf_factory():
    return make_pdf
