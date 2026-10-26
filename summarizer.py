"""Case-brief generation.

Two clearly separated methods:

* **Abstractive** (optional): a pretrained sequence-to-sequence model via Hugging
  Face Transformers, applied map-reduce style so the WHOLE document is covered
  (every chunk is summarised, then the partial summaries are combined).
* **Extractive fallback** (always available): sentences copied verbatim from the
  document. It is labelled as extractive everywhere and never presented as an
  AI-generated abstractive summary.

The final order is never produced by the model: it is taken from the extraction
step (or reported as not reliably extracted) and shown separately.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import config
from document_processor import Document
from information_extractor import ExtractionResult
from text_utils import chunk_text, clip, collapse_ws, sentence_spans

logger = logging.getLogger(__name__)

METHOD_EXTRACTIVE = "Extractive fallback (sentences copied from the judgment; NOT an AI-generated abstractive summary)"
METHOD_ABSTRACTIVE = "Abstractive AI summary (pretrained model: {model}; chunked map-reduce over the full text)"

ORDER_NOT_FOUND = "The final order could not be reliably extracted from the document. Please read the end of the original judgment."


@dataclass
class SummaryResult:
    text: str
    method: str
    is_abstractive: bool = False
    word_count: int = 0
    order_statement: str = ORDER_NOT_FOUND
    order_extracted: bool = False
    segments: list[dict] = field(default_factory=list)  # extractive: [{"label", "text", "location"}]
    coverage_note: str = ""
    full_text_used: bool = True
    chunks_total: int = 0
    chunks_processed: int = 0
    warnings: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    insufficient_text: bool = False


# --------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------
def _word_count(text: str) -> int:
    return len(text.split())


_STOPWORDS = frozenset(
    "the and that this with from have been were which their there would could should shall "
    "will your what when where whom whose also into than then them they these those such some "
    "other only more most very being under upon about after before between because while "
    "court learned counsel".split()
)


def _word_frequencies(text: str) -> Counter:
    words = [w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOPWORDS]
    return Counter(words)


def _looks_like_prose(sentence: str, max_words: int) -> bool:
    words = sentence.split()
    if not (8 <= len(words) <= max_words):
        return False
    letters = [c for c in sentence if c.isalpha()]
    if not letters or sum(c.isupper() for c in letters) / len(letters) > 0.5:
        return False  # headings, party blocks, signatures
    if re.search(r"\.{4,}|_{3,}", sentence) or re.match(r"(?i)^\s*page\s+\d+", sentence):
        return False
    return True


_BACKGROUND_CUES = re.compile(
    r"(?i)\b(filed|alleg\w*|FIR|registered|complain\w*|prosecution|petition\w*|appeal\w*|suit|plaintiff|"
    r"facts|dispute|agreement|purchased|arrested|lodged|charge\w*|case)\b"
)
_ISSUE_CUES = re.compile(
    r"(?i)\b(whether|question|issue|point\s+for\s+determination|contend\w*|submit\w*|argu\w*|"
    r"learned\s+counsel|prayer|seeks?|challenge\w*)\b"
)
_REASONING_CUES = re.compile(
    r"(?i)\b(held|we\s+find|finds?|observed|noted|considered|therefore|accordingly|in\s+view\s+of|"
    r"having\s+regard|thus|opined|satisf\w+|do\s+not\s+find|no\s+merit|merit|reasons?)\b"
)


# --------------------------------------------------------------------------------
# Extractive fallback
# --------------------------------------------------------------------------------
def extractive_summary(doc: Document, extraction: ExtractionResult | None = None) -> SummaryResult:
    """Select sentences from across the whole document for each part of a brief."""
    text = doc.text
    low, high = config.SUMMARY_TARGET_WORDS
    max_sentence_words = 40
    freq = _word_frequencies(text)
    top_freq = max(freq.values()) if freq else 1

    # Final order evidence (so that sentence is not reused as "reasoning").
    order_entry = None
    if extraction is not None:
        order_field = extraction.fields.get("final_order")
        if order_field and order_field.found:
            order_entry = order_field.entries[0]

    all_spans = sentence_spans(text)
    prose = []  # (position index, start, end, text)
    for s, e in all_spans:
        sentence = collapse_ws(text[s:e])
        if order_entry and order_entry.start is not None and s >= order_entry.start - 1 and e <= (order_entry.end or 0) + 1:
            continue
        if _looks_like_prose(sentence, max_sentence_words):
            prose.append((s, e, sentence))

    n = len(prose)
    used: set[int] = set()

    def score(idx: int, cues: re.Pattern) -> float:
        sentence = prose[idx][2]
        words = re.findall(r"[a-z]{4,}", sentence.lower())
        freq_score = sum(freq.get(w, 0) / top_freq for w in words) / (len(words) ** 0.5 + 1)
        return 2.0 * len(cues.findall(sentence)) + freq_score

    def pick(label: str, cues: re.Pattern, lo: float, hi: float, k: int, require_cue: bool) -> list[tuple[int, str]]:
        """Pick up to k sentences in the [lo, hi) fraction of the document."""
        candidates = [
            i for i in range(n)
            if lo * n <= i < max(hi * n, lo * n + 1) and i not in used and (not require_cue or cues.search(prose[i][2]))
        ]
        candidates.sort(key=lambda i: score(i, cues), reverse=True)
        chosen = sorted(candidates[:k])
        used.update(chosen)
        return [(i, label) for i in chosen]

    background = pick("Background", _BACKGROUND_CUES, 0.0, 0.4, 2, require_cue=False)
    issue = pick("Issue and contentions", _ISSUE_CUES, 0.0, 0.75, 1, require_cue=True)
    reasoning = pick("Court's reasoning", _REASONING_CUES, 0.3, 1.0, 2, require_cue=True)

    parts: dict[str, list[int]] = {"Background": [], "Issue and contentions": [], "Court's reasoning": []}
    for i, label in background + issue + reasoning:
        parts[label].append(i)

    def total_words() -> int:
        return sum(_word_count(prose[i][2]) for idxs in parts.values() for i in idxs)

    # Too long: drop the weakest extra sentence from the largest part.
    while total_words() > high:
        largest = max(parts, key=lambda lbl: len(parts[lbl]))
        if len(parts[largest]) <= 1:
            break
        parts[largest].pop()  # drop the last-picked (latest) extra sentence
    # Too short: add more cue-bearing sentences until in range or no candidates remain.
    attempts = 0
    while total_words() < low and attempts < 6:
        attempts += 1
        extra = pick("Court's reasoning", _REASONING_CUES, 0.0, 1.0, 1, require_cue=False)
        if not extra:
            break
        candidate_words = _word_count(prose[extra[0][0]][2])
        if total_words() + candidate_words > high:
            used.discard(extra[0][0])
            break
        parts["Court's reasoning"].append(extra[0][0])
        parts["Court's reasoning"].sort()

    segments, lines = [], []
    for label, idxs in parts.items():
        if not idxs:
            continue
        joined = " ".join(prose[i][2] for i in sorted(idxs))
        first_s, first_e = prose[sorted(idxs)[0]][0], prose[sorted(idxs)[0]][1]
        segments.append({"label": label, "text": joined, "location": doc.locate(first_s, first_e)})
        lines.append(f"{label}: {joined}")
    brief_text = "\n\n".join(lines)

    result = SummaryResult(
        text=brief_text,
        method=METHOD_EXTRACTIVE,
        is_abstractive=False,
        word_count=_word_count(brief_text),
        segments=segments,
        coverage_note=(
            f"Sentences were selected from across the whole document ({len(all_spans)} sentences scanned)."
            if not doc.truncated else
            "Sentences were selected from the analysed portion of the document only; the document was truncated at the configured limit."
        ),
        full_text_used=not doc.truncated,
    )
    if not segments:
        result.text = "No suitable sentences could be selected from this text for an extractive brief."
        result.word_count = 0
        result.warnings.append("Extractive selection found no usable prose sentences.")
    elif not (low <= result.word_count <= high):
        result.warnings.append(
            f"The brief has {result.word_count} words, outside the ~{low}-{high} word target (extractive selection is not able to tune length precisely)."
        )
    if not any(seg["label"] == "Court's reasoning" for seg in segments):
        result.warnings.append("No sentence clearly stating the court's reasoning was identified.")
    return result


# --------------------------------------------------------------------------------
# Abstractive (optional) - map-reduce over chunks
# --------------------------------------------------------------------------------
def map_reduce_summarise(
    chunks: list[str],
    generate: Callable[[str, bool], str],
    chunk_chars: int | None = None,
    max_rounds: int = 3,
) -> tuple[str, int, int]:
    """Summarise every chunk, merge, and summarise the merge.

    *generate(text, final)* produces a summary; ``final=True`` asks for the
    ~100-150 word final version. Returns (summary, chunks_processed, model_calls).
    Every chunk is passed to the model - none are skipped.
    """
    chunk_chars = chunk_chars or config.CHUNK_CHARS
    calls = 0
    partials = []
    for chunk in chunks:
        partials.append(generate(chunk, False))
        calls += 1
    combined = " ".join(p.strip() for p in partials if p.strip())

    rounds = 1
    while len(combined) > chunk_chars and rounds < max_rounds:
        sub_chunks = chunk_text(combined, chunk_chars)
        if len(sub_chunks) <= 1:
            break
        partials = []
        for sub in sub_chunks:
            partials.append(generate(sub, False))
            calls += 1
        combined = " ".join(p.strip() for p in partials if p.strip())
        rounds += 1

    final = generate(combined, True)
    calls += 1
    return final.strip(), len(chunks), calls


@lru_cache(maxsize=1)
def _load_seq2seq(model_name: str):
    """Load tokenizer + model (cached). Raises if transformers/torch/model are unavailable."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # optional dependency

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _make_generate_fn(model_name: str, state: dict) -> Callable[[str, bool], str]:
    """Return generate(text, final) backed by the pretrained model."""
    tokenizer, model = _load_seq2seq(model_name)
    max_input = min(int(getattr(tokenizer, "model_max_length", 1024) or 1024), 1024)
    low, high = config.SUMMARY_TARGET_WORDS

    def generate(text: str, final: bool) -> str:
        full_len = len(tokenizer(text, add_special_tokens=True)["input_ids"])
        if full_len > max_input:
            state["truncated_inputs"] += 1  # reported to the user, never silent
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input)
        input_len = inputs["input_ids"].shape[1]
        max_new = int(high * 1.4) if final else 130
        min_new = min(int(low * 1.2), max(10, input_len // 2)) if final else min(30, max(5, input_len // 4))
        output = model.generate(
            **inputs, num_beams=4, max_new_tokens=max_new, min_new_tokens=min_new,
            no_repeat_ngram_size=3, early_stopping=True,
        )
        return tokenizer.decode(output[0], skip_special_tokens=True)

    return generate


def _trim_to_sentence_limit(text: str, max_words: int) -> str:
    """If over *max_words*, keep whole sentences only (never cut mid-sentence)."""
    if _word_count(text) <= max_words:
        return text
    kept: list[str] = []
    for s, e in sentence_spans(text):
        sentence = text[s:e].strip()
        if _word_count(" ".join(kept + [sentence])) > max_words:
            break
        kept.append(sentence)
    return " ".join(kept) if kept else text


def abstractive_summary(doc: Document, model_name: str | None = None) -> SummaryResult:
    """Summarise the full document with the pretrained model. Raises on any failure."""
    model_name = model_name or config.SUMMARY_MODEL
    state = {"truncated_inputs": 0}
    generate = _make_generate_fn(model_name, state)
    chunks = chunk_text(doc.text, config.CHUNK_CHARS)
    raw, processed, calls = map_reduce_summarise(chunks, generate)
    text = _trim_to_sentence_limit(raw, config.SUMMARY_TARGET_WORDS[1])

    result = SummaryResult(
        text=text,
        method=METHOD_ABSTRACTIVE.format(model=model_name),
        is_abstractive=True,
        word_count=_word_count(text),
        coverage_note=f"All {processed} chunk(s) of the analysed text were summarised ({calls} model calls), then merged.",
        full_text_used=not doc.truncated,
        chunks_total=len(chunks),
        chunks_processed=processed,
    )
    if doc.truncated:
        result.coverage_note += " The document itself was truncated at the configured limit."
        result.warnings.append("Only the analysed portion of the document was summarised (see input notes).")
    if state["truncated_inputs"]:
        result.warnings.append(
            f"{state['truncated_inputs']} model input(s) exceeded the model's maximum length and were truncated internally."
        )
    result.warnings.append(
        "Abstractive models can produce statements that are not in the judgment. "
        "Compare this brief with the source text before relying on it."
    )
    return result


# --------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------
def summarise(doc: Document, extraction: ExtractionResult | None = None, use_model: bool = False) -> SummaryResult:
    """Create the case brief, falling back to the labelled extractive method."""
    order_text, order_found = ORDER_NOT_FOUND, False
    if extraction is not None:
        order_field = extraction.fields.get("final_order")
        if order_field and order_field.found:
            order_text = f"{order_field.entries[0].value}  (extracted from the document text; verify manually)"
            order_found = True

    if len(doc.text.strip()) < config.MIN_CHARS_FOR_BRIEF or len(sentence_spans(doc.text)) < 4:
        return SummaryResult(
            text="There is not enough text to generate a reliable case brief.",
            method="None (insufficient text)",
            order_statement=order_text,
            order_extracted=order_found,
            coverage_note="No summary attempted.",
            insufficient_text=True,
        )

    result: SummaryResult | None = None
    fallback_reason: str | None = None
    if use_model:
        try:
            result = abstractive_summary(doc)
        except Exception as exc:
            logger.warning("Summarisation model unavailable: %s", type(exc).__name__)
            fallback_reason = (
                f"The pretrained summarisation model could not be loaded or run ({type(exc).__name__}). "
                "An extractive fallback was used instead."
            )
    if result is None:
        result = extractive_summary(doc, extraction)
        result.fallback_reason = fallback_reason

    result.order_statement = order_text
    result.order_extracted = order_found
    return result
