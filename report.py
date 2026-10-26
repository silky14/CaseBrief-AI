"""Builds the downloadable analysis report (JSON and plain text).

The full judgment text is deliberately NOT included - only short supporting
quotations - to keep reports small and avoid re-distributing the document.
"""
from __future__ import annotations

import json
import textwrap

from information_extractor import NOT_FOUND
from pipeline import AnalysisResult

REPORT_TITLE = "CASEBRIEF AI — ANALYSIS REPORT"

DISCLAIMER = [
    "This is an AI-assisted research prototype.",
    "Results may be incomplete or inaccurate.",
    "Verify all information against the original judgment.",
    "This tool does not provide legal advice.",
]


def _processing_statement(result: AnalysisResult) -> list[str]:
    doc, cls, summ = result.document, result.classification, result.summary
    lines = [
        f"Full document processed: {'Yes' if result.full_document_processed else 'NO - see details below'}",
        f"Input: {doc.source_type if doc.source_type == 'Pasted text' else f'{doc.source_type} ({doc.source_name})'}; "
        f"{len(doc.text):,} characters analysed"
        + (f" of {doc.original_chars:,}" if doc.truncated else "")
        + (f"; {doc.page_count} page(s)" if doc.page_count else ""),
        f"Classification coverage: {cls.coverage}",
        f"Extraction coverage: {result.extraction.coverage}",
        f"Summary coverage: {summ.coverage_note or 'n/a'}",
    ]
    lines.extend(f"Input note: {note}" for note in doc.notes)
    return lines


def to_dict(result: AnalysisResult) -> dict:
    """JSON-serialisable representation of the analysis."""
    cls, summ, ext = result.classification, result.summary, result.extraction
    return {
        "report": REPORT_TITLE,
        "generated_at": result.generated_at,
        "source": {
            "name": result.document.source_name,
            "type": result.document.source_type,
            "characters_analysed": len(result.document.text),
            "original_characters": result.document.original_chars,
            "pages": result.document.page_count,
            "truncated": result.document.truncated,
            "content_fingerprint": result.document.fingerprint,
        },
        "classification": {
            "category": cls.category,
            "method": cls.method,
            "confidence": cls.confidence,
            "confidence_note": cls.confidence_note,
            "scores": cls.scores,
            "matched_evidence": cls.evidence,
            "notes": cls.notes,
            "coverage": cls.coverage,
        },
        "case_information": {
            key: {
                "label": f.label,
                "value": f.display_value,
                "found": f.found,
                "extraction_method": f.method,
                "needs_manual_verification": f.needs_verification,
                "note": f.note,
                "evidence": [
                    {"value": e.value, "quote": e.quote, "location": e.location} for e in f.entries
                ],
            }
            for key, f in ext.fields.items()
        },
        "case_brief": {
            "text": summ.text,
            "method": summ.method,
            "is_abstractive_ai_summary": summ.is_abstractive,
            "word_count": summ.word_count,
            "final_order_statement": summ.order_statement,
            "final_order_extracted": summ.order_extracted,
            "segments": summ.segments,
            "coverage": summ.coverage_note,
            "chunks_total": summ.chunks_total,
            "chunks_processed": summ.chunks_processed,
            "fallback_reason": summ.fallback_reason,
            "warnings": summ.warnings,
        },
        "verification": {
            "full_document_processed": result.full_document_processed,
            "processing_details": _processing_statement(result),
            "missing_fields": ext.missing(),
            "fields_needing_manual_verification": ext.needing_verification(),
            "stage_errors": result.stage_errors,
            "warnings": result.warnings,
        },
        "disclaimer": DISCLAIMER,
    }


def to_json(result: AnalysisResult) -> str:
    return json.dumps(to_dict(result), indent=2, ensure_ascii=False)


def _wrap(text: str, indent: str = "   ", width: int = 96) -> str:
    """Wrap *text*; continuation lines are aligned under the text, not repeated bullets."""
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=" " * len(indent))


def to_text(result: AnalysisResult) -> str:
    """Plain-text report following the structure in the specification."""
    cls, summ, ext = result.classification, result.summary, result.extraction
    out: list[str] = [REPORT_TITLE, "=" * len(REPORT_TITLE), f"Generated: {result.generated_at}", ""]

    out += ["1. CLASSIFICATION", "-----------------"]
    out.append(f"   Category: {cls.category}")
    out.append(f"   Classification method: {cls.method}")
    if cls.confidence is not None:
        out.append(f"   Confidence: {cls.confidence:.2f}")
        out.append(_wrap(cls.confidence_note, "   ("))
    else:
        out.append("   Confidence: not provided (this method gives no meaningful probability)")
    if cls.evidence:
        out.append("   Matched keyword evidence (keyword counts, not probabilities):")
        for category in dict.fromkeys(e["category"] for e in cls.evidence):
            terms = ", ".join(f"{e['term']} x{e['count']}" for e in cls.evidence if e["category"] == category)
            out.append(_wrap(f"{category}: {terms}", "     - "))
    for note in cls.notes:
        out.append(_wrap(f"Note: {note}"))
    out.append("")

    out += ["2. CASE INFORMATION", "-------------------"]
    for key in ("case_title", "court", "case_number", "parties", "judgment_date", "statutes", "final_order",
                "judges", "key_dates"):
        f = ext.fields[key]
        flag = "  [VERIFY MANUALLY]" if f.found and f.needs_verification else ""
        out.append(f"   {f.label}:{flag}")
        if f.found:
            for entry in f.entries:
                out.append(_wrap(entry.value, "      - "))
        else:
            out.append(f"      {NOT_FOUND}")
    out.append("")

    out += ["3. CASE BRIEF", "-------------"]
    out.append(f"   Method: {summ.method}")
    if summ.fallback_reason:
        out.append(_wrap(f"Note: {summ.fallback_reason}"))
    out.append("")
    for paragraph in summ.text.split("\n\n"):
        out.append(_wrap(paragraph))
        out.append("")
    out.append(_wrap(f"Final order: {summ.order_statement}"))
    for warning in summ.warnings:
        out.append(_wrap(f"Warning: {warning}"))
    out.append("")

    out += ["4. VERIFICATION", "---------------"]
    for line in _processing_statement(result):
        out.append(_wrap(line))
    missing = ext.missing()
    out.append(_wrap("Fields not found: " + (", ".join(missing) if missing else "none")))
    uncertain = ext.needing_verification()
    out.append(_wrap("Fields requiring manual verification: " + (", ".join(uncertain) if uncertain else "none")))
    for error in result.stage_errors:
        out.append(_wrap(f"Error: {error}"))
    out.append("   Supporting source text:")
    for f in ext.fields.values():
        shown: set[tuple[str, str]] = set()
        for entry in f.entries:
            if (entry.quote, entry.location) in shown or len(shown) >= 3:
                continue
            shown.add((entry.quote, entry.location))
            out.append(_wrap(f'{f.label} — "{entry.quote}" [{entry.location}]', "     * "))
    out.append("")

    out += ["5. DISCLAIMER", "-------------"]
    out += [f"   - {line}" for line in DISCLAIMER]
    return "\n".join(out) + "\n"
