from document_processor import load_pasted_text
from information_extractor import NOT_FOUND, extract_all, extract_statutes


def run(text: str):
    return extract_all(load_pasted_text(text))


# --- Valid judgment text (synthetic sample) --------------------------------------
def test_sample_judgment_fields(sample_text):
    ext = run(sample_text)
    assert "HIGH COURT OF SAMPLE STATE" in ext.get("court").display_value
    assert "BAIL APPLICATION NO. 123 OF 2024" in ext.get("case_number").display_value
    assert "FIR" not in ext.get("case_number").display_value  # FIR number is not the case number
    assert "ASHOK VERMA (Petitioner)" in ext.get("parties").display_value
    assert "STATE OF SAMPLE STATE (Respondent)" in ext.get("parties").display_value
    assert ext.get("case_title").display_value == "ASHOK VERMA v. STATE OF SAMPLE STATE"
    assert ext.get("judgment_date").display_value == "15th March, 2024"
    assert "allowed" in ext.get("final_order").display_value
    assert "EXAMPLE" in ext.get("judges").display_value


def test_every_found_field_has_quote_and_location(sample_text):
    ext = run(sample_text)
    for f in ext.fields.values():
        for entry in f.entries:
            assert entry.quote and entry.location, f"{f.key} lacks evidence"


def test_heuristic_fields_are_flagged_for_manual_verification(sample_text):
    ext = run(sample_text)
    assert "Parties" in ext.needing_verification()
    assert "Final order" in ext.needing_verification()


# --- Missing fields ----------------------------------------------------------------
def test_missing_fields_are_reported_honestly():
    ext = run("This is a short paragraph with no court, no parties, no dates and no legal provisions at all.")
    for key in ("case_title", "court", "case_number", "parties", "judgment_date", "statutes", "final_order"):
        assert ext.get(key).display_value == NOT_FOUND
        assert not ext.get(key).found
    assert "A final order could not be reliably extracted" in ext.get("final_order").note
    assert len(ext.missing()) >= 7


def test_court_is_not_inferred_from_body_text():
    # The court is only *mentioned* in the body (as the trial court's superior); it is not the deciding court.
    text = "The appellant approached the matter on facts.\n" * 3 + "The order of the High Court of Delhi was challenged before us."
    assert run(text).get("court").display_value == NOT_FOUND


def test_unlabelled_dates_are_not_called_judgment_date():
    ext = run("The incident occurred on 12.05.2021 and the complaint was filed on 14.05.2021 according to the record.")
    assert ext.get("judgment_date").display_value == NOT_FOUND
    assert "12.05.2021" in ext.get("key_dates").display_value  # still listed, but uninterpreted


def test_dated_line_near_end_is_flagged_as_uncertain():
    text = "Some synthetic body text. " * 20 + "\nDated: 01.02.2023"
    field = run(text).get("judgment_date")
    assert field.found and field.needs_verification and "may be an order or signing date" in field.note


# --- Legal sections -------------------------------------------------------------------
def test_legal_section_extraction_preserves_wording():
    doc = load_pasted_text(
        "The accused is charged under Sections 302 and 34 of the Indian Penal Code, 1860 and "
        "Section 438 Cr.P.C. Reliance is placed on Article 21 of the Constitution of India and "
        "Order VII Rule 11 CPC. The offence under u/s 420 IPC is also alleged."
    )
    values = extract_statutes(doc).display_value
    assert "Sections 302 and 34 of the Indian Penal Code, 1860" in values
    assert "Section 438 Cr.P.C" in values
    assert "Article 21 of the Constitution of India" in values
    assert "Order VII Rule 11 CPC" in values
    assert "u/s 420 IPC" in values


def test_no_automatic_ipc_to_bns_conversion():
    doc = load_pasted_text("The accused was charged under Section 302 IPC and Section 439 CrPC in the FIR.")
    field = extract_statutes(doc)
    joined = field.display_value + " " + field.note
    assert "Section 302 IPC" in field.display_value and "Section 439 CrPC" in field.display_value
    assert "BNS" not in field.display_value  # nothing is converted or invented
    assert "No mapping" in field.note


def test_new_codes_are_kept_exactly_as_written():
    doc = load_pasted_text("Proceedings were initiated under Section 103 of the Bharatiya Nyaya Sanhita, 2023.")
    assert "Section 103 of the Bharatiya Nyaya Sanhita, 2023" in extract_statutes(doc).display_value


def test_article_without_constitution_context_is_ignored():
    doc = load_pasted_text("Under Article 5 of the supply agreement the buyer must pay within thirty days.")
    assert extract_statutes(doc).display_value == NOT_FOUND


# --- Final order ------------------------------------------------------------------------
def test_final_order_ignores_pending_applications_line():
    text = ("Body text of the synthetic judgment. " * 30
            + "\nThe appeal is dismissed. No order as to costs.\nPending applications, if any, stand disposed of.")
    value = run(text).get("final_order").display_value
    assert "appeal is dismissed" in value.lower()
    assert not value.lower().startswith("pending")


def test_final_order_not_taken_from_early_text():
    text = "The earlier appeal was dismissed in 2015 by the lower court.\n" + "Neutral filler sentence here. " * 400
    assert run(text).get("final_order").display_value == NOT_FOUND


# --- Robustness --------------------------------------------------------------------------
def test_extractors_survive_odd_input():
    for text in ["!!!", "." * 500, "versus\nversus\nversus", "Section", "Article 21", "\n\n\n a \n\n"]:
        extract_all(load_pasted_text(text))  # must not raise
