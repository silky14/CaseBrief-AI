"""Orchestrates classification, extraction and summarisation for one Document."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import config
from classifier import ClassificationResult, UNCERTAIN, METHOD_KEYWORD, classify
from document_processor import Document
from information_extractor import ExtractionResult, extract_all
from summarizer import SummaryResult, summarise

logger = logging.getLogger(__name__)


@dataclass
class AnalysisOptions:
    use_zero_shot: bool = False
    use_summary_model: bool = False


@dataclass
class AnalysisResult:
    document: Document
    classification: ClassificationResult
    extraction: ExtractionResult
    summary: SummaryResult
    options: AnalysisOptions
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    stage_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fingerprint(self) -> str:
        return self.document.fingerprint

    @property
    def full_document_processed(self) -> bool:
        """True only if nothing was truncated and every stage looked at the whole text."""
        return (
            not self.document.truncated
            and self.classification.full_text_used
            and self.summary.full_text_used
        )


def analyze(
    doc: Document,
    options: AnalysisOptions | None = None,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Run all three stages. A failing stage degrades to an honest placeholder."""
    options = options or AnalysisOptions()
    notify = progress or (lambda _msg: None)
    stage_errors: list[str] = []
    warnings: list[str] = list(doc.notes)

    if len(doc.text.strip()) < config.MIN_CHARS_FOR_ANALYSIS:
        warnings.append(
            f"The text is very short (under {config.MIN_CHARS_FOR_ANALYSIS} characters); results are likely to be incomplete."
        )

    notify("Classifying judgment…")
    try:
        classification = classify(doc.text, use_zero_shot=options.use_zero_shot)
    except Exception as exc:
        logger.exception("Classification stage failed")
        stage_errors.append(f"Classification failed ({type(exc).__name__}).")
        classification = ClassificationResult(category=UNCERTAIN, method=METHOD_KEYWORD, notes=["Classification failed."])

    notify("Extracting case information…")
    extraction = extract_all(doc)

    notify("Generating case brief…")
    try:
        summary = summarise(doc, extraction, use_model=options.use_summary_model)
    except Exception as exc:
        logger.exception("Summary stage failed")
        stage_errors.append(f"Summary generation failed ({type(exc).__name__}).")
        summary = SummaryResult(text="Summary generation failed.", method="None (error)")

    return AnalysisResult(
        document=doc,
        classification=classification,
        extraction=extraction,
        summary=summary,
        options=options,
        stage_errors=stage_errors,
        warnings=warnings,
    )
