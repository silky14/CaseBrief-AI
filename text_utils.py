"""Small text helpers shared by several modules."""
from __future__ import annotations

import re

_WS_RE = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    """Collapse all whitespace runs (including newlines) to single spaces."""
    return _WS_RE.sub(" ", text).strip()


def clip(text: str, max_chars: int = 240) -> str:
    """Whitespace-collapse *text* and shorten it with an ellipsis if needed."""
    text = collapse_ws(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


# Sentence boundary: terminal punctuation followed by whitespace and a capital,
# digit or opening quote/bracket - or a blank line.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(\[]?[A-Z0-9])|\n{2,}")

# Sentence fragments ending in these are abbreviations, not sentence ends.
_ABBREV_END_RE = re.compile(
    r"(?:\b(?:No|Nos|Sec|Secs|Dr|Mr|Mrs|Ms|Smt|Vs|Ltd|Co|Pvt|Art|Ors|Anr|Addl|Govt|"
    r"Hon|Rs|Cr|Crl|etc|viz|cf|Sr|Jr|St|Dy|Adv|Vol|Para|Paras|Ex|Exh)\.|\b[A-Za-z]\.)$"
)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans of sentences in *text*.

    A deliberately simple splitter: it merges fragments that end in common
    legal abbreviations ("No.", "Cr.P.C.", "v.") but is not perfect.
    """
    raw: list[list[int]] = []
    pos = 0
    for match in _SPLIT_RE.finditer(text):
        if match.start() > pos:
            raw.append([pos, match.start()])
        pos = match.end()
    if pos < len(text):
        raw.append([pos, len(text)])

    merged: list[list[int]] = []
    prev_end = 0
    for start, end in raw:
        gap = text[prev_end:start]
        if (
            merged
            and "\n\n" not in gap
            and _ABBREV_END_RE.search(text[merged[-1][0]:merged[-1][1]].rstrip())
        ):
            merged[-1][1] = end
        else:
            merged.append([start, end])
        prev_end = end
    return [(s, e) for s, e in merged if text[s:e].strip()]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into chunks of at most *max_chars*, on sentence boundaries.

    Every character of the input ends up in exactly one chunk (apart from
    whitespace at sentence boundaries), so nothing is silently dropped.
    Sentences longer than *max_chars* are hard-split.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for start, end in sentence_spans(text):
        sentence = text[start:end].strip()
        pieces = [sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars)] or [sentence]
        for piece in pieces:
            if current and current_len + len(piece) + 1 > max_chars:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
