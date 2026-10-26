import json

from document_processor import load_pasted_text, load_uploaded_file
from pipeline import AnalysisOptions, analyze
from report import DISCLAIMER, to_dict, to_json, to_text


def test_end_to_end_on_synthetic_sample(sample_text):
    result = analyze(load_pasted_text(sample_text))
    assert result.classification.category == "Bail"
    assert "Keyword baseline" in result.classification.method
    assert result.extraction.get("parties").found
    assert result.summary.text and not result.summary.is_abstractive
    assert result.full_document_processed
    assert not result.stage_errors


def test_end_to_end_on_pdf(pdf_factory, sample_text):
    lines = [l for l in sample_text.splitlines() if l.strip()][:40]
    data = pdf_factory([lines[:20], lines[20:]])
    result = analyze(load_uploaded_file("synthetic.pdf", data))
    assert result.document.page_count == 2
    assert result.extraction.get("court").found


def test_pipeline_with_optional_models_requested_but_unavailable(sample_text, monkeypatch):
    import classifier, summarizer

    monkeypatch.setattr(classifier, "_load_zero_shot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    monkeypatch.setattr(summarizer, "_load_seq2seq", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    result = analyze(load_pasted_text(sample_text), AnalysisOptions(use_zero_shot=True, use_summary_model=True))
    assert "Keyword baseline" in result.classification.method
    assert not result.summary.is_abstractive and result.summary.fallback_reason


def test_progress_callback_receives_stage_messages(sample_text):
    messages: list[str] = []
    analyze(load_pasted_text(sample_text), progress=messages.append)
    assert len(messages) == 3


def test_text_report_has_all_sections_and_disclaimer(sample_text):
    text = to_text(analyze(load_pasted_text(sample_text)))
    for heading in ("CASEBRIEF AI — ANALYSIS REPORT", "1. CLASSIFICATION", "2. CASE INFORMATION",
                    "3. CASE BRIEF", "4. VERIFICATION", "5. DISCLAIMER"):
        assert heading in text
    for line in DISCLAIMER:
        assert line in text


def test_json_report_is_valid_and_complete(sample_text):
    result = analyze(load_pasted_text(sample_text))
    payload = json.loads(to_json(result))
    assert payload["classification"]["confidence"] is None
    assert payload["case_information"]["court"]["found"] is True
    assert payload["verification"]["full_document_processed"] is True
    assert payload["disclaimer"] == DISCLAIMER
    assert sample_text[:200] not in json.dumps(payload)  # full text is not embedded


def test_report_for_empty_extraction_says_not_found():
    result = analyze(load_pasted_text("A short synthetic note without any structure or legal provisions in it."))
    text = to_text(result)
    assert "Not found in document" in text
    assert "could not be reliably extracted" in text
    assert to_dict(result)["verification"]["missing_fields"]


def test_truncated_input_is_reported_as_not_fully_processed(sample_text):
    result = analyze(load_pasted_text(sample_text, max_chars=1200))
    assert not result.full_document_processed
    assert "NO - see details below" in to_text(result)
