"""Rule-based extraction of key case information.

Every field is extracted with regular expressions and light heuristics - there is
no learned model here, and the patterns will NOT work on every Indian court
judgment. Each result records:

* the exact text it came from (a short quotation) and where it was found,
* how it was obtained (explicit pattern match vs. heuristic), and
* whether it needs manual verification.

Nothing is inferred: if the evidence is not in the text, the field is reported as
"Not found in document". Legal provisions are kept exactly as written - no
IPC->BNS or CrPC->BNSS conversion is attempted.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import config
from document_processor import Document
from text_utils import clip, collapse_ws, sentence_spans

logger = logging.getLogger(__name__)

NOT_FOUND = "Not found in document"

METHOD_EXPLICIT = "Explicit pattern match (regex)"
METHOD_HEURISTIC = "Heuristic pattern match (regex) - interpretation may be wrong"


@dataclass
class Entry:
    """One extracted value with its supporting evidence."""

    value: str
    quote: str = ""
    location: str = ""
    start: int | None = None
    end: int | None = None


@dataclass
class ExtractedField:
    key: str
    label: str
    entries: list[Entry] = field(default_factory=list)
    method: str = METHOD_EXPLICIT
    needs_verification: bool = False
    note: str = ""

    @property
    def found(self) -> bool:
        return bool(self.entries)

    @property
    def display_value(self) -> str:
        return "; ".join(e.value for e in self.entries) if self.entries else NOT_FOUND


@dataclass
class ExtractionResult:
    fields: dict[str, ExtractedField]
    coverage: str = ""

    def get(self, key: str) -> ExtractedField:
        return self.fields[key]

    def missing(self) -> list[str]:
        return [f.label for f in self.fields.values() if not f.found]

    def needing_verification(self) -> list[str]:
        return [f.label for f in self.fields.values() if f.found and f.needs_verification]


FIELD_LABELS = {
    "case_title": "Case title",
    "court": "Court",
    "case_number": "Case number",
    "parties": "Parties",
    "judgment_date": "Judgment date",
    "statutes": "Statutes and sections",
    "final_order": "Final order",
    "judges": "Judges (as named in header)",
    "key_dates": "Other dates mentioned",
}


def _new_field(key: str, **kwargs) -> ExtractedField:
    return ExtractedField(key=key, label=FIELD_LABELS[key], **kwargs)


def _entry(doc: Document, start: int, end: int, value: str | None = None, pad: int = 0) -> Entry:
    """Build an Entry whose quote is the text span (plus optional context padding)."""
    text = doc.text
    quote_start, quote_end = max(0, start - pad), min(len(text), end + pad)
    # Padding can cut a word in half; move the quote edges to whitespace.
    if quote_start < start and quote_start > 0 and not text[quote_start - 1].isspace():
        shift = text.find(" ", quote_start, start)
        quote_start = shift + 1 if shift != -1 else start
    if quote_end > end and quote_end < len(text) and not text[quote_end].isspace():
        shift = text.rfind(" ", end, quote_end)
        quote_end = shift if shift != -1 else end
    return Entry(
        value=value if value is not None else collapse_ws(text[start:end]),
        quote=clip(text[quote_start:quote_end]),
        location=doc.locate(start, end),
        start=start,
        end=end,
    )


def _header(doc: Document) -> str:
    return doc.text[: config.HEADER_CHARS]


def _lines_with_offsets(text: str) -> list[tuple[int, int, str]]:
    """(start, end, line) for every line, including blank ones."""
    out, pos = [], 0
    for line in text.split("\n"):
        out.append((pos, pos + len(line), line))
        pos += len(line) + 1
    return out


# =====================================================================================
# Court
# =====================================================================================
_COURT_KEYWORD_RE = re.compile(
    r"(?i)\b(supreme\s+court|high\s+court|district\s+court|sessions?\s+court|court\s+of|"
    r"magistrate|tribunal|commission|family\s+court|civil\s+court)\b"
)
_STRONG_COURT_RE = re.compile(
    r"(?i)\b(supreme\s+court\s+of\s+india|high\s+court\s+(?:of|at)\s+[a-z]|district\s+court|sessions?\s+court)\b"
)
_COURT_LOCATION_LINE_RE = re.compile(r"(?i)^\s*at\s+[A-Za-z][A-Za-z .]{1,40}$")
_BODY_TEXT_RE = re.compile(r"(?i)\b(?:judgment|order|decision)\s+of\s+the\b|\bhon'?ble\b|\bversus\b|\bvs\b|\bv\.")


def extract_court(doc: Document) -> ExtractedField:
    """Court name from the header lines only (never guessed from the body)."""
    result = _new_field("court")
    lines = _lines_with_offsets(_header(doc))
    for idx, (start, end, raw) in enumerate(lines):
        line = raw.strip()
        if not line or len(line) > 110 or len(line.split()) > 16:
            continue
        if not _COURT_KEYWORD_RE.search(line) or _BODY_TEXT_RE.search(line):
            continue
        value = re.sub(r"(?i)^in\s+the\s+", "", line).strip()
        span_end = end
        if idx + 1 < len(lines) and _COURT_LOCATION_LINE_RE.match(lines[idx + 1][2]):
            value += " " + lines[idx + 1][2].strip()  # e.g. "AT NEW DELHI"
            span_end = lines[idx + 1][1]
        result.entries.append(_entry(doc, start, span_end, value=value))
        result.needs_verification = not _STRONG_COURT_RE.search(line)
        if result.needs_verification:
            result.method = METHOD_HEURISTIC
        return result
    result.note = "No court name found in the header of the document."
    return result


# =====================================================================================
# Case number
# =====================================================================================
# "Criminal Appeal No. 123 of 2020", "C.A. No. 1234/2020", "Civil Appeal Nos. 100-101 of 2021"
_CASE_NO_A_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z.()]*[ ]){1,5}(?i:nos?)\.?[ ]*\d+[A-Za-z]?(?:[ ]*(?:-|–|&|,|(?i:to))[ ]*\d+)*"
    r"[ ]*(?:/|(?i:of))[ ]*\d{4})"
)
# "W.P.(C) 1234/2021", "CRL.M.C. 123/2020", "BAIL APPLN. 456/2022"
_CASE_NO_B_RE = re.compile(
    r"\b((?:[A-Z][A-Z.]*[ ]){0,2}[A-Z][A-Z.]{1,12}(?:[ ]?\([A-Za-z.]{1,6}\))?[ ]*\d{1,6}[ ]*/[ ]*\d{4})\b"
)


# Numbers that look like case numbers but are FIRs, crime numbers or section references.
_NOT_A_CASE_NO_RE = re.compile(r"(?i)^(?:fir|f\.i\.r\.?|crime|cr|police|section|sections|sec|article)\b")


def extract_case_number(doc: Document) -> ExtractedField:
    """Case/petition/appeal numbers from the header (first few are kept)."""
    result = _new_field("case_number", method=METHOD_EXPLICIT)
    header = doc.text[: config.HEADER_CHARS + 1000]
    seen: set[str] = set()
    hits: list[re.Match] = []
    for pattern in (_CASE_NO_A_RE, _CASE_NO_B_RE):
        hits.extend(pattern.finditer(header))
    for match in sorted(hits, key=lambda m: m.start()):
        value = collapse_ws(match.group(1))
        key = value.lower()
        if key in seen or _NOT_A_CASE_NO_RE.match(value):
            continue  # e.g. an FIR number or a section reference is not the case number
        seen.add(key)
        result.entries.append(_entry(doc, match.start(1), match.end(1), value=value))
        if len(result.entries) >= 5:
            break
    if result.entries:
        result.needs_verification = True  # pattern-based; check against the original
        result.note = "Matched by number-pattern; several numbers may belong to connected matters."
    else:
        result.note = "No case number pattern found in the header."
    return result


# =====================================================================================
# Parties and case title
# =====================================================================================
_ROLE_RE = re.compile(
    r"(?i)\b(petitioners?|appellants?|applicants?|plaintiffs?|complainants?|revisionists?|"
    r"respondents?|defendants?|accused|opposite\s+party|non-?applicants?)\b"
)
_VS_LINE_RE = re.compile(r"(?i)^\s*(?:versus|vs\.?|v\.?|v/s)\s*$")
_INLINE_VS_RE = re.compile(
    r"^(?P<a>[A-Z][^\n]{2,120}?)\s+(?i:versus|vs\.?|v\.|v/s)\s+(?P<b>[A-Z][^\n]{2,120})$"
)
_STOP_LINE_RE = re.compile(r"(?i)\b(court|tribunal)\b|\bno\.?\s*\d|\d+\s*/\s*\d{4}|\bof\s+\d{4}\b")
_COUNSEL_LINE_RE = re.compile(r"(?i)^\s*(through|represented\s+by|counsel|advocates?|mr\.|ms\.|mrs\.|shri|smt)\b")


def _clean_party(raw: str) -> tuple[str, str]:
    """Return (party name, role) from a raw header line(s); role may be ''."""
    role_match = _ROLE_RE.search(raw)
    role = role_match.group(1).title() if role_match else ""
    name = raw[: role_match.start()] if role_match else raw
    name = re.sub(r"[.…_\-–]{2,}", " ", name)
    name = re.sub(r"^\s*\(?\d+[.)]\s*", "", name)
    name = collapse_ws(name).strip(" ,;:-–()")
    return name, role


def _find_parties(doc: Document) -> dict | None:
    """Locate 'A versus B' in the header. Returns names/roles/evidence span, or None."""
    header = _header(doc)
    lines = _lines_with_offsets(header)

    # Layout 1: "versus" alone on its own line, parties above and below.
    for idx, (_, _, line) in enumerate(lines):
        if not _VS_LINE_RE.match(line):
            continue
        before: list[tuple[int, int, str]] = []
        j = idx - 1
        while j >= 0 and not lines[j][2].strip():
            j -= 1
        while j >= 0 and len(before) < 4:
            text_j = lines[j][2].strip()
            if not text_j or _STOP_LINE_RE.search(text_j):
                break
            if not _COUNSEL_LINE_RE.match(text_j):
                before.insert(0, lines[j])
            j -= 1
        after: list[tuple[int, int, str]] = []
        k = idx + 1
        while k < len(lines) and not lines[k][2].strip():
            k += 1
        while k < len(lines) and len(after) < 4:
            text_k = lines[k][2].strip()
            if not text_k:
                break
            if not _COUNSEL_LINE_RE.match(text_k):
                after.append(lines[k])
            k += 1
            if _ROLE_RE.search(text_k):
                break
        if before and after:
            name1, role1 = _clean_party(" ".join(l[2] for l in before))
            name2, role2 = _clean_party(" ".join(l[2] for l in after))
            if name1 and name2 and len(name1) <= 150 and len(name2) <= 150:
                return {
                    "p1": name1, "role1": role1, "p2": name2, "role2": role2,
                    "start": before[0][0], "end": after[-1][1], "layout": "separate lines",
                }

    # Layout 2: "A v. B" on a single line.
    for start, end, raw in lines:
        match = _INLINE_VS_RE.match(raw.strip())
        if match and not _STOP_LINE_RE.search(raw):
            name1, role1 = _clean_party(match.group("a"))
            name2, role2 = _clean_party(match.group("b"))
            if name1 and name2:
                return {
                    "p1": name1, "role1": role1, "p2": name2, "role2": role2,
                    "start": start, "end": end, "layout": "single line",
                }
    return None


def extract_parties_and_title(doc: Document) -> tuple[ExtractedField, ExtractedField]:
    """Parties (with roles) and a case title composed from them."""
    parties = _new_field("parties", method=METHOD_HEURISTIC, needs_verification=True)
    title = _new_field("case_title", method=METHOD_HEURISTIC, needs_verification=True)
    found = _find_parties(doc)
    if not found:
        parties.note = "No 'X versus Y' party block found in the header."
        title.note = "No title could be built because the parties were not identified."
        return parties, title

    for name, role in ((found["p1"], found["role1"]), (found["p2"], found["role2"])):
        value = f"{name} ({role})" if role else name
        parties.entries.append(_entry(doc, found["start"], found["end"], value=value))
    parties.note = (
        "Parsed from the header block. Header names may be abbreviated (e.g. '& Ors.'), "
        "so the full list of parties may differ."
    )
    title.entries.append(_entry(doc, found["start"], found["end"], value=f"{found['p1']} v. {found['p2']}"))
    title.note = "Composed from the first-named parties detected in the header; it is not a quoted official title."
    return parties, title


# =====================================================================================
# Judgment date
# =====================================================================================
_MONTH = (
    r"(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_DATE_PATTERN = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?[ ]+(?:day[ ]+of[ ]+)?{_MONTH}\b\.?,?[ ]+\d{{4}}"
    rf"|{_MONTH}\b\.?[ ]+\d{{1,2}}(?:st|nd|rd|th)?,?[ ]+\d{{4}}"
    rf"|\d{{1,2}}[./-]\d{{1,2}}[./-]\d{{2,4}})"
)
_DATE_RE = re.compile(_DATE_PATTERN)
_LABELLED_DATE_RE = re.compile(
    r"(?i:date[ ]+of[ ]+(?:judg(?:e)?ment|decision|order|pronouncement)"
    r"|judg(?:e)?ment[ ]+(?:pronounced|delivered|dated)(?:[ ]+on)?"
    r"|(?:pronounced|decided|delivered)[ ]+on"
    r"|date[ ]+of[ ]+decision)[ ]*[:\-–]?[ ]*(?P<date>" + _DATE_PATTERN + ")"
)
_DATED_RE = re.compile(r"(?i:\bdated?\b)[ ]*[:\-–]?[ ]*(?P<date>" + _DATE_PATTERN + ")")


def extract_judgment_date(doc: Document) -> ExtractedField:
    """Judgment date, only when introduced by a recognised label."""
    result = _new_field("judgment_date")
    match = _LABELLED_DATE_RE.search(doc.text)
    if match:
        label_start, label_end = match.start(), match.end()
        result.entries.append(_entry(doc, label_start, label_end, value=collapse_ws(match.group("date")), pad=0))
        result.method = METHOD_EXPLICIT
        result.note = "Date appeared next to a label such as 'Date of Judgment' / 'Pronounced on'."
        return result

    tail_start = max(0, len(doc.text) - 1500)
    tail_matches = list(_DATED_RE.finditer(doc.text, tail_start))
    if tail_matches:
        last = tail_matches[-1]
        result.entries.append(_entry(doc, last.start(), last.end(), value=collapse_ws(last.group("date"))))
        result.method = METHOD_HEURISTIC
        result.needs_verification = True
        result.note = (
            "No explicit 'Date of Judgment' label. This is the last 'Dated' entry near the end of the "
            "document and may be an order or signing date rather than the judgment date."
        )
        return result

    result.note = "No labelled judgment date found. Other dates in the text are listed separately."
    return result


def extract_key_dates(doc: Document, limit: int = 12) -> ExtractedField:
    """Dates exactly as written, with context. Their meaning is not interpreted."""
    result = _new_field("key_dates")
    seen: set[str] = set()
    for match in _DATE_RE.finditer(doc.text):
        value = collapse_ws(match.group(0))
        if value.lower() in seen:
            continue
        seen.add(value.lower())
        result.entries.append(_entry(doc, match.start(), match.end(), value=value, pad=50))
        if len(result.entries) >= limit:
            result.note = f"Showing the first {limit} distinct dates. "
            break
    if result.entries:
        result.note += "Dates are shown as written; whether each is a filing, hearing or order date is not interpreted."
    else:
        result.note = "No date expressions found."
    return result


# =====================================================================================
# Statutes and sections
# =====================================================================================
_ABBREV = (
    r"(?:IPC|I\.P\.C\.?|BNSS|BNS|BSA|Cr\.?\s?P\.?\s?C\.?|CPC|C\.P\.C\.?|NDPS(?:[ ]Act)?|"
    r"POCSO(?:[ ]Act)?|PMLA|IT[ ]Act|NI[ ]Act|N\.I\.[ ]Act|SARFAESI[ ]Act|RTI[ ]Act)"
)
_ACT_NAME = (
    r"(?:Code[ ]of[ ](?:Criminal|Civil)[ ]Procedure(?:,?[ ]*(?:of[ ]+)?\d{4})?"
    r"|Constitution(?:[ ]of[ ]India)?"
    r"|(?:[A-Z][A-Za-z'’\-]*(?:[ ]+(?:of|and|for|the|on|in)[ ]+|[ ]+)){0,8}?"
    r"(?:Act|Code|Sanhita|Adhiniyam)(?:,?[ ]*(?:of[ ]+)?\d{4})?)"
)
_NUM = r"\d{1,4}[A-Z]{0,2}(?:\([A-Za-z0-9]{1,4}\))*"
_NUMS = rf"{_NUM}(?:[ ]*(?:,|and|&|/|to|-|–|or)[ ]*{_NUM})*"
_SECTION_RE = re.compile(
    rf"(?<![A-Za-z])(?i:sections?|secs?\.?|u/s\.?|ss?\.)[ ]*(?P<nums>{_NUMS})"
    rf"(?P<tail>[ ,]*(?:(?i:of|under|read[ ]with)[ ]+)?(?:(?i:the)[ ]+)?(?:{_ACT_NAME}|{_ABBREV})\b)?"
)
_ARTICLE_RE = re.compile(
    r"(?<![A-Za-z])(?i:articles?)[ ]*\d{1,3}[A-Z]?(?:\(\d+\))?"
    r"(?:[ ]*(?:,|and|&|to|-)[ ]*\d{1,3}[A-Z]?(?:\(\d+\))?)*"
    r"(?P<tail>[ ]+(?i:of[ ]+the)[ ]+Constitution(?:[ ]of[ ]India)?)?"
)
_ORDER_RULE_RE = re.compile(
    r"\bOrder[ ]+[IVXLC\d]+[ ]+(?i:rules?)[ ]+\d+[A-Za-z]?"
    r"(?:[ ]+(?:(?i:of)[ ]+(?:the[ ]+)?)?(?:CPC|C\.P\.C\.?|Code[ ]of[ ]Civil[ ]Procedure(?:,?[ ]*1908)?))?"
)


def extract_statutes(doc: Document, limit: int = 40) -> ExtractedField:
    """Sections, articles and order/rule references exactly as written."""
    result = _new_field("statutes")
    text = doc.text
    constitution_mentioned = "constitution" in text.lower()
    candidates: list[re.Match] = list(_SECTION_RE.finditer(text)) + list(_ORDER_RULE_RE.finditer(text))
    for match in _ARTICLE_RE.finditer(text):
        # "Article 5" alone could be a contract clause; keep only constitutional-looking ones.
        if match.group("tail") or constitution_mentioned:
            candidates.append(match)

    seen: set[str] = set()
    total_distinct = 0
    for match in sorted(candidates, key=lambda m: m.start()):
        value = collapse_ws(match.group(0)).rstrip(" ,")
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        total_distinct += 1
        if len(result.entries) < limit:
            result.entries.append(_entry(doc, match.start(), match.end(), value=value, pad=60))

    if result.entries:
        result.note = (
            "Provisions are shown exactly as written in the document. No mapping between old and new "
            "codes (e.g. IPC to BNS, CrPC to BNSS) is performed. A section shown without an Act name "
            "was written that way in the text."
        )
        if total_distinct > limit:
            result.note += f" Showing {limit} of {total_distinct} distinct references."
    else:
        result.note = "No explicit 'Section N' / 'Article N' references found."
    return result


# =====================================================================================
# Final order / disposition
# =====================================================================================
_DISPOSITION_RE = re.compile(
    r"(?i)\b(allowed|dismissed|disposed[ ]of|partly[ ]allowed|partially[ ]allowed|rejected|decreed|"
    r"set[ ]aside|quashed|acquitted|convicted|granted|admitted|remanded|upheld|affirmed|confirmed)\b"
)
_ORDER_SUBJECT_RE = re.compile(
    r"(?i)\b(appeals?|petitions?|applications?|suits?|writ|revisions?|bail|conviction|sentence|"
    r"decree|accused|appellants?|petitioners?|impugned|prayer|order|judgment)\b"
)
_PENDING_RE = re.compile(r"(?i)\bpending\b|\bno\s+(?:order\s+as\s+to\s+)?costs?\b")
_FOLLOWUP_RE = re.compile(r"(?i)\b(costs?|directed?|directions?|released?|ordered|furnish|shall|liberty|surrender|pay)\b")


def extract_final_order(doc: Document) -> ExtractedField:
    """Operative order, searched only in the closing part of the document."""
    result = _new_field("final_order", method=METHOD_HEURISTIC, needs_verification=True)
    text = doc.text
    n = len(text)
    tail_start = max(0, n - max(2500, int(n * 0.25)))
    spans = [(s + tail_start, e + tail_start) for s, e in sentence_spans(text[tail_start:])]

    primary_idx = None
    for idx, (s, e) in enumerate(spans):
        sentence = text[s:e]
        if len(sentence) > 600:
            continue
        if _DISPOSITION_RE.search(sentence) and _ORDER_SUBJECT_RE.search(sentence) and not _PENDING_RE.search(sentence):
            primary_idx = idx  # keep the LAST qualifying sentence

    if primary_idx is None:
        result.note = "A final order could not be reliably extracted. Check the end of the original judgment."
        return result

    start, end = spans[primary_idx]
    for s, e in spans[primary_idx + 1: primary_idx + 3]:
        sentence = text[s:e]
        if _FOLLOWUP_RE.search(sentence) or _DISPOSITION_RE.search(sentence):
            if e - start <= 700:
                end = e
    result.entries.append(_entry(doc, start, end, value=clip(text[start:end], 600)))
    result.note = (
        "Taken from the closing part of the text by looking for disposition words (allowed, dismissed, "
        "disposed of, ...). The operative order may contain further directions; read the original."
    )
    return result


# =====================================================================================
# Judges named in header
# =====================================================================================
_JUDGE_RE = re.compile(
    r"(?i:\bjustice)[ ]+(?P<name>[A-Z][A-Za-z.'’]*(?:[ ]+[A-Z][A-Za-z.'’]*){0,3})"
)
_JUDGE_STOPWORDS = {"JUDGMENT", "ORDER", "DATE", "DATED", "JJ", "AND", "JUDGEMENT"}


def extract_judges(doc: Document, limit: int = 5) -> ExtractedField:
    """Judge names that follow the word 'Justice' in the header."""
    result = _new_field("judges", method=METHOD_HEURISTIC, needs_verification=True)
    seen: set[str] = set()
    for match in _JUDGE_RE.finditer(_header(doc)):
        tokens = match.group("name").split()
        while tokens and tokens[-1].upper().strip(".") in _JUDGE_STOPWORDS:
            tokens.pop()
        if not tokens:
            continue
        name = " ".join(tokens)
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        result.entries.append(_entry(doc, match.start(), match.end(), value=name))
        if len(result.entries) >= limit:
            break
    result.note = (
        "Names following 'Justice' in the header." if result.entries else "No judge names found in the header."
    )
    return result


# =====================================================================================
# Orchestration
# =====================================================================================
def extract_all(doc: Document) -> ExtractionResult:
    """Run every extractor; a failure in one never prevents the others from running."""
    fields: dict[str, ExtractedField] = {}

    def run(key: str, func):
        try:
            return func(doc)
        except Exception as exc:  # defensive: regexes on odd input should never crash the app
            logger.exception("Extractor '%s' failed", key)
            failed = _new_field(key)
            failed.note = f"Extraction error ({type(exc).__name__}); treat this field as not extracted."
            failed.needs_verification = True
            return failed

    parties, title = (None, None)
    try:
        parties, title = extract_parties_and_title(doc)
    except Exception as exc:
        logger.exception("Party extraction failed")
        parties = _new_field("parties")
        title = _new_field("case_title")
        parties.note = title.note = f"Extraction error ({type(exc).__name__}); treat as not extracted."

    fields["case_title"] = title
    fields["court"] = run("court", extract_court)
    fields["case_number"] = run("case_number", extract_case_number)
    fields["parties"] = parties
    fields["judgment_date"] = run("judgment_date", extract_judgment_date)
    fields["statutes"] = run("statutes", extract_statutes)
    fields["final_order"] = run("final_order", extract_final_order)
    fields["judges"] = run("judges", extract_judges)
    fields["key_dates"] = run("key_dates", extract_key_dates)

    coverage = (
        "Statutes, dates and final order: searched the full text (final order: closing portion only). "
        f"Court, case number, parties, title and judges: searched only the first {config.HEADER_CHARS:,} characters."
    )
    if doc.truncated:
        coverage += " NOTE: the document was truncated at the configured limit."
    return ExtractionResult(fields=fields, coverage=coverage)
