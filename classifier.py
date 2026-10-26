"""Judgment classification.

Three methods, tried in this order of priority:

1. **Trained model** - a TF-IDF + Logistic Regression pipeline, used ONLY if the
   user has trained one on their own labelled dataset (see ``train_classifier``
   and the README). No trained model ships with this project.
2. **Zero-shot model** - optional pretrained NLI model via Hugging Face
   Transformers; used only when the user turns it on and it can be loaded.
3. **Keyword baseline** - transparent, rule-based, always available. Its scores
   are keyword-evidence totals, NOT probabilities, so no confidence is reported.

The result always records which method produced it.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import config

logger = logging.getLogger(__name__)

CATEGORIES = ["Criminal", "Civil", "Constitutional", "Bail", "Property", "Other / Uncertain"]
UNCERTAIN = "Other / Uncertain"

METHOD_KEYWORD = "Keyword baseline (rule-based; not a trained model)"
METHOD_ZERO_SHOT = "Zero-shot classifier (pretrained NLI model: {model})"
METHOD_TRAINED = "TF-IDF + Logistic Regression (trained on a user-supplied dataset)"


@dataclass
class ClassificationResult:
    category: str
    method: str
    confidence: float | None = None  # only set when the method provides a score
    confidence_note: str = "Not provided: keyword scores are not calibrated probabilities."
    evidence: list[dict] = field(default_factory=list)  # [{"category", "term", "count"}]
    scores: dict[str, float] = field(default_factory=dict)  # keyword evidence totals / model scores
    notes: list[str] = field(default_factory=list)
    coverage: str = "Full document text"
    full_text_used: bool = True

    @property
    def is_uncertain(self) -> bool:
        return self.category == UNCERTAIN


# --------------------------------------------------------------------------------
# 3. Keyword baseline
# --------------------------------------------------------------------------------
# (label, regex, weight). Each rule contributes weight * min(occurrences, 5).
# These are hand-written heuristics, not learned from data.
_KEYWORD_RULES: dict[str, list[tuple[str, str, int]]] = {
    "Bail": [
        ("bail", r"(?i:\bbail\b)", 3),
        ("anticipatory bail", r"(?i:\banticipatory\s+bail\b)", 4),
        ("regular bail", r"(?i:\bregular\s+bail\b)", 4),
        ("bail application/bond/order", r"(?i:\bbail\s+(?:application|bond|order)s?\b)", 3),
        ("surety", r"(?i:\bsureties|\bsurety\b)", 2),
        ("Section 437/438/439", r"(?i:\bsections?\s+(?:437|438|439)\b)", 2),
        ("Section 480/482/483 BNSS", r"(?i:\bsections?\s+(?:480|482|483)\b[^.\n]{0,60}\b(?:BNSS|Bharatiya\s+Nagarik))", 2),
        ("custody", r"(?i:\b(?:judicial|police)\s+custody\b)", 1),
    ],
    "Criminal": [
        ("FIR", r"\bFIR\b", 2),
        ("accused", r"(?i:\baccused\b)", 2),
        ("prosecution", r"(?i:\bprosecution\b)", 2),
        ("Indian Penal Code / IPC", r"(?:\bIndian\s+Penal\s+Code\b|\bIPC\b|\bI\.P\.C\b)", 3),
        ("Bharatiya Nyaya Sanhita / BNS", r"(?:\bBharatiya\s+Nyaya\s+Sanhita\b|\bBNS\b)", 3),
        ("Cr.P.C. / CrPC / BNSS", r"(?:\bCr\.?\s?P\.?\s?C\b|\bBNSS\b|Code\s+of\s+Criminal\s+Procedure)", 3),
        ("conviction/acquittal", r"(?i:\b(?:convict\w*|acquit\w*)\b)", 2),
        ("sentence", r"(?i:\bsentenc\w+)", 1),
        ("charge-sheet", r"(?i:\bcharge[\s-]?sheet\b)", 2),
        ("offence", r"(?i:\boffences?\b)", 1),
        ("murder / NDPS / POCSO", r"(?:\b[Mm]urder\b|\bNDPS\b|\bPOCSO\b)", 2),
        ("police", r"(?i:\bpolice\b)", 1),
    ],
    "Civil": [
        ("plaintiff", r"(?i:\bplaintiffs?\b)", 3),
        ("defendant", r"(?i:\bdefendants?\b)", 3),
        ("suit", r"(?i:\bsuits?\b)", 2),
        ("decree", r"(?i:\bdecree[ds]?\b)", 2),
        ("Code of Civil Procedure / CPC", r"(?:Code\s+of\s+Civil\s+Procedure|\bCPC\b|\bC\.P\.C\b)", 3),
        ("injunction", r"(?i:\binjunctions?\b)", 2),
        ("damages", r"(?i:\bdamages\b)", 1),
        ("specific performance", r"(?i:\bspecific\s+performance\b)", 3),
        ("contract/agreement", r"(?i:\b(?:contract|agreement|breach)\b)", 1),
        ("written statement", r"(?i:\bwritten\s+statement\b)", 2),
        ("Limitation Act", r"(?i:\blimitation\s+act\b)", 1),
    ],
    "Constitutional": [
        ("Article 14/19/21/32/226", r"(?i:\barticles?\s+(?:14|19|21|32|226|227)\b)", 3),
        ("Constitution of India", r"(?i:\bconstitution\s+of\s+india\b)", 2),
        ("fundamental right", r"(?i:\bfundamental\s+rights?\b)", 3),
        ("writ of mandamus/certiorari/habeas corpus", r"(?i:\b(?:mandamus|certiorari|habeas\s+corpus|prohibition|quo\s+warranto)\b)", 3),
        ("writ petition", r"(?i:\bwrit\s+petitions?\b)", 2),
        ("constitutional validity", r"(?i:\bconstitutional(?:ity|\s+validity)?\b)", 2),
        ("ultra vires", r"(?i:\bultra\s+vires\b)", 2),
        ("basic structure", r"(?i:\bbasic\s+structure\b)", 3),
        ("directive principles", r"(?i:\bdirective\s+principles?\b)", 2),
    ],
    "Property": [
        ("property", r"(?i:\bpropert(?:y|ies)\b)", 2),
        ("title / possession", r"(?i:\b(?:title|possession)\b)", 1),
        ("sale deed / gift deed", r"(?i:\b(?:sale|gift|lease|conveyance)\s+deed\b)", 3),
        ("partition", r"(?i:\bpartition\b)", 2),
        ("landlord / tenant / eviction", r"(?i:\b(?:landlord|tenant|tenancy|eviction|ejectment)\b)", 3),
        ("Transfer of Property Act", r"(?i:\btransfer\s+of\s+property\s+act\b)", 3),
        ("land acquisition", r"(?i:\bland\s+acquisition\b)", 3),
        ("mutation / revenue records", r"(?i:\b(?:mutation|revenue\s+records?|khasra|khatauni|jamabandi)\b)", 2),
        ("encroachment / easement", r"(?i:\b(?:encroach\w*|easement)\b)", 2),
        ("immovable property", r"(?i:\bimmovable\b)", 2),
    ],
}

KEYWORD_MIN_SCORE = 8      # below this total there is not enough evidence
KEYWORD_MARGIN = 1.25      # winner must beat the runner-up by at least 25%
BAIL_VS_CRIMINAL = 0.5     # bail counts as the more specific label if it has >= 50% of criminal's score


def classify_keywords(text: str) -> ClassificationResult:
    """Rule-based classification. Returns Other / Uncertain when evidence is weak."""
    scores: dict[str, float] = {}
    evidence: list[dict] = []
    for category, rules in _KEYWORD_RULES.items():
        total = 0
        for label, pattern, weight in rules:
            count = len(re.findall(pattern, text))
            if count:
                total += weight * min(count, 5)
                evidence.append({"category": category, "term": label, "count": count})
        scores[category] = float(total)

    result = ClassificationResult(
        category=UNCERTAIN,
        method=METHOD_KEYWORD,
        confidence=None,
        evidence=sorted(evidence, key=lambda e: (e["category"], -e["count"])),
        scores=scores,
    )

    # "Bail" is a sub-type of criminal matters: when bail evidence is solid, rank it
    # as the more specific label instead of letting Criminal swamp it.
    ranking = dict(scores)
    if scores["Bail"] >= KEYWORD_MIN_SCORE and scores["Bail"] >= BAIL_VS_CRIMINAL * scores["Criminal"]:
        ranking["Criminal"] = 0.0
        result.notes.append("Bail evidence was ranked above general Criminal evidence (bail is a sub-type of criminal matters).")

    ordered = sorted(ranking.items(), key=lambda kv: kv[1], reverse=True)
    (top, top_score), (second, second_score) = ordered[0], ordered[1]

    if top_score < KEYWORD_MIN_SCORE:
        result.notes.append(
            f"Insufficient keyword evidence (best score {top_score:.0f}, minimum {KEYWORD_MIN_SCORE}); reported as uncertain."
        )
    elif second_score > 0 and top_score < KEYWORD_MARGIN * second_score:
        result.notes.append(
            f"Evidence is too close between {top} and {second}; reported as uncertain. "
            "Review the keyword evidence below."
        )
    else:
        result.category = top

    result.notes.append(
        "Scores are weighted keyword-match totals, not probabilities. Check the matched terms against the judgment."
    )
    return result


# --------------------------------------------------------------------------------
# 2. Optional zero-shot classifier
# --------------------------------------------------------------------------------
_ZERO_SHOT_LABELS = {
    "Criminal": "criminal case",
    "Civil": "civil dispute",
    "Constitutional": "constitutional matter",
    "Bail": "bail application",
    "Property": "property dispute",
}


@lru_cache(maxsize=1)
def _load_zero_shot(model_name: str):
    """Load the zero-shot pipeline (cached). Raises if transformers/model unavailable."""
    from transformers import pipeline  # imported lazily: optional dependency

    return pipeline("zero-shot-classification", model=model_name)


def classify_zero_shot(text: str, model_name: str | None = None) -> ClassificationResult:
    """Classify with a pretrained zero-shot model. Raises on any loading/runtime failure."""
    model_name = model_name or config.ZERO_SHOT_MODEL
    classifier = _load_zero_shot(model_name)
    snippet = text[: config.ZERO_SHOT_INPUT_CHARS]
    output = classifier(
        snippet,
        candidate_labels=list(_ZERO_SHOT_LABELS.values()),
        hypothesis_template="This legal judgment concerns a {}.",
        multi_label=False,
    )
    reverse = {v: k for k, v in _ZERO_SHOT_LABELS.items()}
    scores = {reverse[label]: float(score) for label, score in zip(output["labels"], output["scores"])}
    top = max(scores, key=scores.get)
    top_score = scores[top]

    result = ClassificationResult(
        category=top if top_score >= config.ZERO_SHOT_MIN_SCORE else UNCERTAIN,
        method=METHOD_ZERO_SHOT.format(model=model_name),
        confidence=top_score,
        confidence_note=(
            "Relative model score across the five candidate labels. Not calibrated, and the model has "
            "not been validated on Indian court judgments."
        ),
        scores=scores,
        coverage=f"First {config.ZERO_SHOT_INPUT_CHARS:,} characters only",
        full_text_used=len(text) <= config.ZERO_SHOT_INPUT_CHARS,
    )
    if top_score < config.ZERO_SHOT_MIN_SCORE:
        result.notes.append(
            f"Top model score {top_score:.2f} is below the {config.ZERO_SHOT_MIN_SCORE:.2f} threshold; reported as uncertain."
        )
    if not result.full_text_used:
        result.notes.append(
            f"Only the first {config.ZERO_SHOT_INPUT_CHARS:,} characters were given to the model."
        )
    return result


# --------------------------------------------------------------------------------
# 1. Trained TF-IDF + Logistic Regression
# --------------------------------------------------------------------------------
def train_classifier(
    csv_path: str | Path,
    out_path: str | Path | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    min_per_class: int = 5,
) -> dict:
    """Train and evaluate a TF-IDF + Logistic Regression pipeline on a labelled CSV.

    The CSV needs two columns: ``text`` (the judgment text) and ``label`` (one of
    ``CATEGORIES``). The model is trained on a stratified train split and
    evaluated on the held-out split; the saved model is exactly the one that was
    evaluated. Returns a dict with the evaluation report - nothing is invented.
    """
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    out_path = Path(out_path or config.TRAINED_MODEL_PATH)
    texts: list[str] = []
    labels: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"text", "label"} <= set(reader.fieldnames):
            raise ValueError("CSV must have 'text' and 'label' columns.")
        for row in reader:
            text, label = (row.get("text") or "").strip(), (row.get("label") or "").strip()
            if not text:
                continue
            if label not in CATEGORIES:
                raise ValueError(f"Unknown label '{label}'. Allowed labels: {CATEGORIES}")
            texts.append(text)
            labels.append(label)

    counts = {label: labels.count(label) for label in set(labels)}
    if len(counts) < 2:
        raise ValueError("Need at least two different labels to train a classifier.")
    too_small = {label: n for label, n in counts.items() if n < min_per_class}
    if too_small:
        raise ValueError(f"Each label needs at least {min_per_class} examples; too few for: {too_small}")

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])
    pipeline.fit(x_train, y_train)
    report_text = classification_report(y_test, pipeline.predict(x_test), zero_division=0)
    report_dict = classification_report(y_test, pipeline.predict(x_test), zero_division=0, output_dict=True)

    metadata = {
        "dataset": Path(csv_path).name,
        "n_train": len(x_train),
        "n_test": len(x_test),
        "label_counts": counts,
        "seed": seed,
        "test_size": test_size,
        "held_out_report": report_text,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, out_path)
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"metadata": metadata, "report": report_dict, "report_text": report_text, "model_path": str(out_path)}


def classify_trained(text: str, model_path: str | Path) -> ClassificationResult:
    """Classify with a model produced by ``train_classifier``. Raises on failure.

    Only load model files you trained yourself: joblib files can execute code.
    """
    import joblib

    bundle = joblib.load(model_path)
    pipeline, metadata = bundle["pipeline"], bundle.get("metadata", {})
    probabilities = pipeline.predict_proba([text])[0]
    scores = {str(c): float(p) for c, p in zip(pipeline.classes_, probabilities)}
    top = max(scores, key=scores.get)
    result = ClassificationResult(
        category=top if scores[top] >= config.TRAINED_MIN_PROBABILITY else UNCERTAIN,
        method=METHOD_TRAINED,
        confidence=scores[top],
        confidence_note=(
            "Model probability from Logistic Regression. Treat as uncalibrated unless you have checked "
            f"calibration on held-out data (model trained on {metadata.get('n_train', '?')} examples)."
        ),
        scores=scores,
    )
    if scores[top] < config.TRAINED_MIN_PROBABILITY:
        result.notes.append(
            f"Top probability {scores[top]:.2f} is below {config.TRAINED_MIN_PROBABILITY:.2f}; reported as uncertain."
        )
    return result


# --------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------
def classify(
    text: str,
    use_zero_shot: bool = False,
    model_path: str | Path | None = None,
) -> ClassificationResult:
    """Classify *text*, choosing the best available method and falling back honestly."""
    fallback_notes: list[str] = []

    if not text or len(text.strip()) < config.MIN_CHARS_FOR_ANALYSIS:
        result = classify_keywords(text or "")
        result.category = UNCERTAIN
        result.notes.insert(0, "Text is too short for a reliable classification.")
        return result

    model_file = Path(model_path or config.TRAINED_MODEL_PATH)
    if model_file.is_file():
        try:
            return classify_trained(text, model_file)
        except Exception as exc:  # corrupt/incompatible model file
            logger.warning("Trained model could not be used: %s", type(exc).__name__)
            fallback_notes.append(f"A trained model file was found but could not be used ({type(exc).__name__}); fell back.")

    if use_zero_shot:
        try:
            return classify_zero_shot(text)
        except Exception as exc:
            logger.warning("Zero-shot classifier unavailable: %s", type(exc).__name__)
            fallback_notes.append(
                f"The zero-shot model could not be loaded or run ({type(exc).__name__}); "
                "the keyword baseline was used instead."
            )

    result = classify_keywords(text)
    result.notes = fallback_notes + result.notes
    return result


# --------------------------------------------------------------------------------
# CLI: python classifier.py train --data path/to/labelled.csv
# --------------------------------------------------------------------------------
def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CaseBrief AI classifier tools")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train TF-IDF + Logistic Regression on a labelled CSV (columns: text,label)")
    train.add_argument("--data", required=True, help="path to the labelled CSV")
    train.add_argument("--out", default=config.TRAINED_MODEL_PATH, help="where to save the model")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        outcome = train_classifier(args.data, args.out, args.test_size, args.seed)
    except (ValueError, OSError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    print(outcome["report_text"])
    print(f"Model saved to {outcome['model_path']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(_main())
