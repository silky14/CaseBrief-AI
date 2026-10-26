import re

import pytest

import summarizer
from document_processor import load_pasted_text
from information_extractor import extract_all
from summarizer import ORDER_NOT_FOUND, map_reduce_summarise, summarise
from text_utils import chunk_text


def words(text: str) -> int:
    return len(text.split())


def test_summary_fallback_is_labelled_extractive(sample_text):
    doc = load_pasted_text(sample_text)
    result = summarise(doc, extract_all(doc), use_model=False)
    assert not result.is_abstractive
    assert "Extractive fallback" in result.method and "NOT an AI-generated abstractive summary" in result.method
    assert result.fallback_reason is None  # the model was never requested
    assert 60 <= result.word_count <= 175


def test_extractive_sentences_are_verbatim_from_the_document(sample_text):
    doc = load_pasted_text(sample_text)
    result = summarise(doc, extract_all(doc))
    flat_doc = " ".join(doc.text.split())
    for segment in result.segments:
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", segment["text"]):
            assert sentence in flat_doc, f"not verbatim: {sentence!r}"  # nothing invented


def test_final_order_is_included_when_found_and_flagged_when_not(sample_text):
    doc = load_pasted_text(sample_text)
    found = summarise(doc, extract_all(doc))
    assert found.order_extracted and "allowed" in found.order_statement

    plain = "The parties argued at length about many things. " * 40
    doc2 = load_pasted_text(plain)
    missing = summarise(doc2, extract_all(doc2))
    assert not missing.order_extracted and missing.order_statement == ORDER_NOT_FOUND


def test_model_unavailable_falls_back_and_says_so(monkeypatch, sample_text):
    def boom(*a, **k):
        raise ImportError("transformers not installed")

    monkeypatch.setattr(summarizer, "_load_seq2seq", boom)
    doc = load_pasted_text(sample_text)
    result = summarise(doc, extract_all(doc), use_model=True)
    assert not result.is_abstractive and "Extractive fallback" in result.method
    assert result.fallback_reason and "could not be loaded" in result.fallback_reason


def test_insufficient_text_gets_no_invented_summary():
    doc = load_pasted_text("Too short to summarise.")
    result = summarise(doc, extract_all(doc))
    assert result.insufficient_text and "not enough text" in result.text.lower()
    assert result.word_count == 0


# --- Long-document handling -----------------------------------------------------------------
def make_long_text(n_paragraphs: int = 120) -> str:
    blocks = []
    for i in range(n_paragraphs):
        blocks.append(
            f"Paragraph {i}: the petitioner contended in this synthetic passage number {i} that the impugned action "
            f"was arbitrary and the respondent submitted that the record disclosed no illegality in item {i}. "
            f"The Court considered the material on record for point {i} and noted the relevant facts carefully."
        )
    return "\n\n".join(blocks) + "\n\nThe petition is accordingly dismissed."


def test_chunking_covers_every_character_of_a_long_document():
    text = make_long_text()
    chunks = chunk_text(text, 2500)
    assert len(chunks) > 5
    assert all(len(c) <= 2500 for c in chunks)
    assert "".join("".join(c.split()) for c in chunks) == "".join(text.split())  # nothing dropped


def test_map_reduce_passes_every_chunk_to_the_model():
    text = make_long_text()
    chunks = chunk_text(text, 2500)
    seen_inputs: list[tuple[str, bool]] = []

    def fake_generate(chunk: str, final: bool) -> str:
        seen_inputs.append((chunk, final))
        return "Summary of a chunk." if not final else "Final merged summary."

    summary, processed, calls = map_reduce_summarise(chunks, fake_generate, chunk_chars=2500)
    first_pass = [c for c, final in seen_inputs[: len(chunks)]]
    assert first_pass == chunks  # all chunks, in order - not just the beginning
    assert processed == len(chunks) and calls == len(seen_inputs)
    assert summary == "Final merged summary." and seen_inputs[-1][1] is True


def test_extractive_summary_draws_from_across_a_long_document():
    text = make_long_text()
    doc = load_pasted_text(text)
    result = summarise(doc, extract_all(doc))
    # Every synthetic sentence carries its paragraph number ("Paragraph N", "point N", "item N"),
    # so we can see which part of the document each selected sentence came from.
    positions = [
        int(m.group(1))
        for segment in result.segments
        for m in re.finditer(r"(?:Paragraph|point|item|number) (\d+)", segment["text"])
    ]
    assert positions and max(positions) > 30  # not just the opening paragraphs
    reasoning = [s for s in result.segments if s["label"] == "Court's reasoning"]
    assert reasoning, "expected a reasoning segment from the later part of the document"
    reasoning_numbers = [int(n) for n in re.findall(r"(?:point|item) (\d+)", reasoning[0]["text"])]
    assert reasoning_numbers and min(reasoning_numbers) > 30
    assert "whole document" in result.coverage_note
    assert result.order_extracted and "dismissed" in result.order_statement


def test_truncated_documents_are_disclosed_in_the_summary():
    doc = load_pasted_text(make_long_text(), max_chars=5000)
    result = summarise(doc, extract_all(doc))
    assert doc.truncated and not result.full_text_used
    assert "truncated" in result.coverage_note
