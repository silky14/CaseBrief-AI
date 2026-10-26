import json

import pytest

import classifier
from classifier import CATEGORIES, UNCERTAIN, classify, classify_keywords, train_classifier

# Synthetic snippets - short, fictional, for keyword mechanics only.
BAIL = ("The petitioner seeks regular bail under Section 439 of the Code of Criminal Procedure. "
        "The bail application is opposed. The petitioner offers sureties and a bail bond. "
        "Bail is granted subject to conditions. The accused has been in custody. ") * 2
CRIMINAL = ("The prosecution alleges that the accused committed offences punishable under the Indian Penal Code. "
            "The FIR was registered and the charge-sheet filed. The accused was convicted and the sentence imposed. ") * 2
CIVIL = ("The plaintiff filed a suit against the defendant seeking a decree of specific performance of the contract. "
         "The defendant filed a written statement under the Code of Civil Procedure. An injunction was sought. ") * 2
CONSTITUTIONAL = ("The writ petition under Article 226 challenges the constitutional validity of the rule as violating "
                  "the fundamental rights under Article 14 and Article 21 of the Constitution of India. Mandamus is sought. ") * 2
PROPERTY = ("The dispute concerns immovable property. The sale deed and mutation in the revenue records are challenged. "
            "The tenant resists eviction by the landlord and alleges encroachment. Partition of the property is sought. ") * 2


@pytest.mark.parametrize("text,expected", [
    (BAIL, "Bail"), (CRIMINAL, "Criminal"), (CONSTITUTIONAL, "Constitutional"), (PROPERTY, "Property"), (CIVIL, "Civil"),
])
def test_keyword_baseline_categories(text, expected):
    result = classify(text)
    assert result.category == expected
    assert result.category in CATEGORIES


def test_keyword_method_is_labelled_and_has_no_confidence():
    result = classify(BAIL)
    assert "Keyword baseline" in result.method and "not a trained model" in result.method
    assert result.confidence is None
    assert "not calibrated" in result.confidence_note


def test_fallback_to_uncertain_when_evidence_is_weak():
    filler = "This paragraph discusses nothing in particular and uses everyday words only. " * 10
    result = classify(filler)
    assert result.category == UNCERTAIN
    assert any("Insufficient" in n for n in result.notes)


def test_short_or_empty_text_is_uncertain():
    assert classify("").category == UNCERTAIN
    assert classify("bail").category == UNCERTAIN
    assert any("too short" in n for n in classify("bail").notes)


def test_mixed_evidence_is_reported_uncertain():
    mixed = CIVIL + PROPERTY  # civil and property vocabularies overlap heavily
    result = classify_keywords(mixed)
    assert result.category in (UNCERTAIN, "Civil", "Property")  # never a false-precision answer
    if result.category == UNCERTAIN:
        assert any("too close" in n for n in result.notes)


def test_zero_shot_failure_falls_back_to_keywords(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(classifier, "_load_zero_shot", boom)
    result = classify(BAIL, use_zero_shot=True, model_path="does/not/exist.joblib")
    assert result.category == "Bail"
    assert "Keyword baseline" in result.method
    assert any("zero-shot model could not be loaded" in n for n in result.notes)


def test_missing_trained_model_falls_back_to_keywords(tmp_path):
    result = classify(CRIMINAL, model_path=tmp_path / "absent.joblib")
    assert "Keyword baseline" in result.method


def test_corrupt_trained_model_falls_back_with_note(tmp_path):
    bad = tmp_path / "model.joblib"
    bad.write_bytes(b"not a real model")
    result = classify(CRIMINAL, model_path=bad)
    assert "Keyword baseline" in result.method
    assert any("trained model file was found but could not be used" in n for n in result.notes)


# --- Training pipeline mechanics ------------------------------------------------
# The rows below are generated purely to test that the training code RUNS and saves
# a model. They are not a dataset, and no accuracy claim is made from them.
def _write_mechanics_csv(path, per_class=10):
    import csv

    rows = []
    for i in range(per_class):
        rows.append((f"accused prosecution offence FIR police charge-sheet case {i}", "Criminal"))
        rows.append((f"plaintiff defendant suit decree injunction contract case {i}", "Civil"))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)


def test_training_pipeline_runs_saves_and_loads(tmp_path):
    data, out = tmp_path / "labelled.csv", tmp_path / "model.joblib"
    _write_mechanics_csv(data)
    outcome = train_classifier(data, out, seed=7)
    assert out.is_file() and out.with_suffix(".json").is_file()
    assert json.loads(out.with_suffix(".json").read_text())["seed"] == 7
    result = classify(CRIMINAL + " accused prosecution FIR police", model_path=out)
    assert "TF-IDF + Logistic Regression" in result.method
    assert result.confidence is not None and 0.0 <= result.confidence <= 1.0
    assert "report_text" in outcome


def test_training_rejects_tiny_or_invalid_data(tmp_path):
    data = tmp_path / "tiny.csv"
    _write_mechanics_csv(data, per_class=2)
    with pytest.raises(ValueError, match="at least"):
        train_classifier(data, tmp_path / "m.joblib")

    bad = tmp_path / "bad.csv"
    bad.write_text("text,label\nsome text,NotACategory\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown label"):
        train_classifier(bad, tmp_path / "m.joblib")

    wrong_cols = tmp_path / "cols.csv"
    wrong_cols.write_text("body,tag\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'text' and 'label'"):
        train_classifier(wrong_cols, tmp_path / "m.joblib")
