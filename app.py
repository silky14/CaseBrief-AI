"""CaseBrief AI - Streamlit interface.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import streamlit as st

import config
from document_processor import (
    Document,
    DocumentError,
    load_pasted_text,
    load_uploaded_file,
)
from information_extractor import NOT_FOUND
from pipeline import AnalysisOptions, AnalysisResult, analyze
from report import DISCLAIMER, to_json, to_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("casebrief.app")

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "synthetic_sample_judgment.txt"
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None

st.set_page_config(page_title="CaseBrief AI", page_icon="⚖️", layout="wide")


# --------------------------------------------------------------------------------
# State helpers
# --------------------------------------------------------------------------------
def _init_state() -> None:
    st.session_state.setdefault("gen", 0)  # bumping this resets all input widgets
    st.session_state.setdefault("result", None)


def _text_key() -> str:
    return f"pasted_text_{st.session_state['gen']}"


def _file_key() -> str:
    return f"uploaded_file_{st.session_state['gen']}"


def reset_all() -> None:
    """Clear inputs and results. Uploaded files only ever lived in memory."""
    st.session_state["gen"] += 1
    st.session_state["result"] = None


def load_sample() -> None:
    st.session_state[_text_key()] = SAMPLE_PATH.read_text(encoding="utf-8")
    st.session_state["result"] = None


# --------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------
def read_input() -> tuple[Document | None, str | None]:
    """Turn whatever the user supplied into a Document (or a user-facing error)."""
    uploaded = st.session_state.get(_file_key())
    pasted = st.session_state.get(_text_key(), "")
    try:
        if uploaded is not None:
            return load_uploaded_file(uploaded.name, uploaded.getvalue()), None
        if pasted and pasted.strip():
            return load_pasted_text(pasted), None
    except DocumentError as exc:
        return None, str(exc)
    except Exception:  # last-resort guard: never crash the page on odd input
        logger.exception("Unexpected error while reading input")
        return None, "The input could not be read. Try a different file or paste the text instead."
    return None, None


def render_sidebar() -> AnalysisOptions:
    with st.sidebar:
        st.header("About")
        st.write(
            "CaseBrief AI is an independent academic prototype that classifies Indian court "
            "judgments, extracts key case information and drafts a short case brief."
        )
        st.caption("It is not an official system, is not validated for legal decision-making, and gives no legal advice.")

        st.header("Optional AI models")
        if not TRANSFORMERS_AVAILABLE:
            st.caption(
                "Hugging Face Transformers is not installed, so the built-in rule-based methods are used. "
                "See the README to enable the optional models."
            )
        use_zero_shot = st.checkbox(
            "Zero-shot classifier",
            value=False,
            disabled=not TRANSFORMERS_AVAILABLE,
            help="Pretrained NLI model; downloads ~1.6 GB on first use. Falls back to keywords if it cannot load.",
        )
        use_summary_model = st.checkbox(
            "Pretrained summarisation model",
            value=False,
            disabled=not TRANSFORMERS_AVAILABLE,
            help="Abstractive model; downloads ~1.2 GB on first use. Falls back to an extractive brief if it cannot load.",
        )

        st.header("Limits")
        st.caption(
            f"Files: PDF or TXT, up to {config.MAX_FILE_MB} MB. Text: up to {config.MAX_TEXT_CHARS:,} characters. "
            "Anything beyond a limit is reported, never silently dropped."
        )
        st.caption("Uploaded files are processed in memory and are not saved by this app.")
    return AnalysisOptions(use_zero_shot=use_zero_shot, use_summary_model=use_summary_model)


# --------------------------------------------------------------------------------
# Result rendering
# --------------------------------------------------------------------------------
def render_classification(result: AnalysisResult) -> None:
    cls = result.classification
    with st.container(border=True):
        st.subheader("1. Classification")
        st.markdown(f"### {cls.category}")
        st.markdown(f"**Method:** {cls.method}")
        if cls.confidence is not None:
            st.markdown(f"**Model score:** {cls.confidence:.2f}")
            st.caption(cls.confidence_note)
        else:
            st.markdown("**Confidence:** not provided")
            st.caption("This method does not produce a meaningful probability, so none is shown.")
        for note in cls.notes:
            st.caption(f"• {note}")
        if cls.evidence:
            with st.expander("Matched keyword evidence (counts, not probabilities)"):
                for category in dict.fromkeys(e["category"] for e in cls.evidence):
                    terms = ", ".join(f"{e['term']} ×{e['count']}" for e in cls.evidence if e["category"] == category)
                    st.markdown(f"**{category}:** {terms}")


def render_verification_summary(result: AnalysisResult) -> None:
    ext = result.extraction
    with st.container(border=True):
        st.subheader("Verification summary")
        if result.full_document_processed:
            st.markdown("**Full document processed:** yes")
        else:
            st.markdown("**Full document processed:** NO — see notes below")
        st.caption(f"Classification: {result.classification.coverage}")
        st.caption(f"Extraction: {ext.coverage}")
        st.caption(f"Brief: {result.summary.coverage_note or 'n/a'}")
        missing = ext.missing()
        st.markdown("**Not found:** " + (", ".join(missing) if missing else "none"))
        uncertain = ext.needing_verification()
        st.markdown("**Verify manually:** " + (", ".join(uncertain) if uncertain else "none flagged"))
        for note in result.warnings:
            st.warning(note)
        for error in result.stage_errors:
            st.error(error)


def render_case_information(result: AnalysisResult) -> None:
    st.subheader("2. Case information")
    order = ["case_title", "court", "case_number", "parties", "judgment_date", "statutes", "final_order", "judges", "key_dates"]
    left, right = st.columns(2)
    for position, key in enumerate(order):
        f = result.extraction.fields[key]
        column = left if position % 2 == 0 else right
        with column, st.container(border=True):
            badge = "  ⚠️ *verify manually*" if f.found and f.needs_verification else ""
            st.markdown(f"**{f.label}**{badge}")
            if not f.found:
                st.markdown(f"_{NOT_FOUND}_")
                if f.note:
                    st.caption(f.note)
                continue
            for entry in f.entries[:12]:
                st.markdown(f"- {entry.value}")
            st.caption(f"{f.method}. {f.note}".strip())
            with st.expander("Source text"):
                shown: set[tuple[str, str]] = set()
                for entry in f.entries:
                    if (entry.quote, entry.location) in shown:
                        continue
                    shown.add((entry.quote, entry.location))
                    st.markdown(f"> {entry.quote}")
                    st.caption(entry.location)


def render_brief(result: AnalysisResult) -> None:
    summ = result.summary
    with st.container(border=True):
        st.subheader("3. Case brief")
        label = "AI-generated (abstractive)" if summ.is_abstractive else "Extractive fallback"
        st.markdown(f"**Type:** {label}")
        st.caption(summ.method)
        if summ.fallback_reason:
            st.info(summ.fallback_reason)
        if summ.insufficient_text:
            st.info(summ.text)
        else:
            if summ.is_abstractive:
                st.write(summ.text)
            else:
                for segment in summ.segments or [{"label": "", "text": summ.text, "location": ""}]:
                    prefix = f"**{segment['label']}:** " if segment["label"] else ""
                    st.markdown(f"{prefix}{segment['text']}")
                    if segment["location"]:
                        st.caption(f"Source: {segment['location']}")
            st.caption(f"{summ.word_count} words. {summ.coverage_note}")
        st.markdown("**Final order**")
        if summ.order_extracted:
            st.markdown(summ.order_statement)
        else:
            st.warning(summ.order_statement)
        for warning in summ.warnings:
            st.caption(f"⚠️ {warning}")
        with st.expander("Review / copy the brief"):
            st.text_area(
                "Brief text for manual review",
                value=f"{summ.text}\n\nFinal order: {summ.order_statement}",
                height=220,
                key=f"review_{result.fingerprint}",
            )


def render_results(result: AnalysisResult, doc: Document) -> None:
    st.divider()
    st.warning(
        "**Verify before relying on this output.** " + " ".join(DISCLAIMER),
        icon="⚠️",
    )
    top_left, top_right = st.columns([3, 2])
    with top_left:
        render_classification(result)
    with top_right:
        render_verification_summary(result)
    render_case_information(result)
    render_brief(result)

    with st.expander("Extracted document text"):
        st.text_area("Full extracted text", value=doc.text, height=320, disabled=True, key=f"fulltext_{result.fingerprint}")

    st.subheader("Download report")
    d1, d2, _ = st.columns([1, 1, 3])
    stamp = result.generated_at.replace(":", "").replace("-", "")
    d1.download_button("Download TXT report", to_text(result), file_name=f"casebrief_report_{stamp}.txt", mime="text/plain")
    d2.download_button("Download JSON report", to_json(result), file_name=f"casebrief_report_{stamp}.json", mime="application/json")
    st.caption("The report contains short supporting quotations only, not the full judgment text.")


# --------------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------------
def main() -> None:
    _init_state()
    options = render_sidebar()

    st.title("⚖️ CaseBrief AI")
    st.markdown("**Indian Court Judgment Classifier and Key-Information Extractor** — independent academic prototype")
    st.caption(
        "Paste a judgment or upload a text-based PDF/TXT. The app classifies it, extracts key case details "
        "with source quotations, and drafts a short case brief. All output requires verification against the original."
    )

    tab_paste, tab_upload = st.tabs(["Paste text", "Upload PDF / TXT"])
    with tab_paste:
        st.text_area(
            "Judgment text",
            key=_text_key(),
            height=240,
            placeholder="Paste the text of a judgment here…",
        )
        st.button("Load synthetic sample", on_click=load_sample, help="A fictional demonstration text, not a real judgment.")
    with tab_upload:
        st.file_uploader(
            "Upload a judgment (text-based PDF or TXT)",
            type=[ext.lstrip(".") for ext in config.ALLOWED_EXTENSIONS],
            key=_file_key(),
            help=f"Maximum {config.MAX_FILE_MB} MB. Scanned (image-only) PDFs are not supported in this version.",
        )
        st.caption("If both a file and pasted text are provided, the uploaded file is used.")

    doc, error = read_input()
    if error:
        st.error(error)
    if doc is not None:
        with st.container(border=True):
            st.markdown(f"**Input ready:** {doc.source_type} · {len(doc.text):,} characters" + (f" · {doc.page_count} pages" if doc.page_count else ""))
            for note in doc.notes:
                st.warning(note)
            with st.expander(f"Preview (first {config.PREVIEW_CHARS:,} characters)"):
                st.text(doc.text[: config.PREVIEW_CHARS] + ("\n…" if len(doc.text) > config.PREVIEW_CHARS else ""))

    analyze_col, reset_col, _ = st.columns([1, 1, 4])
    run = analyze_col.button("Analyze Judgment", type="primary", disabled=doc is None)
    reset_col.button("Clear / Reset", on_click=reset_all)

    if run and doc is not None:
        with st.status("Analysing judgment…", expanded=True) as status:
            try:
                st.session_state["result"] = analyze(doc, options, progress=lambda msg: st.write(msg))
                status.update(label="Analysis complete", state="complete", expanded=False)
            except Exception:
                logger.exception("Analysis failed")
                st.session_state["result"] = None
                status.update(label="Analysis failed", state="error")
                st.error("The analysis failed unexpectedly. Try shorter text or a different file.")

    result: AnalysisResult | None = st.session_state.get("result")
    if result is not None:
        if doc is None or result.fingerprint != doc.fingerprint:
            st.info("The input has changed since the last analysis. Click **Analyze Judgment** to refresh the results.")
        else:
            render_results(result, doc)
    elif doc is None and not error:
        st.info("Provide a judgment above to begin. You can load the synthetic sample to try the app.")


main()
