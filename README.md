# CaseBrief AI — Indian Court Judgment Classifier and Key-Information Extractor

CaseBrief AI is an **independent academic prototype**, inspired by the problem of automated
judicial-document summarisation. It is **not** an official system of any police force, court or
government body, and it is **not** a validated legal decision-making tool.

Given the text of an Indian court judgment (pasted, or from a text-based PDF/TXT file), it:

1. classifies the judgment into a broad legal category,
2. extracts key case information, each item with a supporting quotation and location,
3. drafts a short case brief (~100–150 words), and
4. lets you download an analysis report (TXT or JSON).

> **Everything it outputs must be verified against the original judgment.** It does not give legal advice.

---

## Features

- **Three input routes:** paste text, upload PDF, upload TXT.
- **Input safety:** file-type and size validation, empty-input handling, corrupted-file handling,
  password-protected PDF detection, and a clear message for scanned (image-only) PDFs.
- **Configurable limits** for file size, extracted text length and PDF pages. Anything beyond a limit
  is *disclosed*, never silently dropped.
- **Classification** into Criminal, Civil, Constitutional, Bail, Property or Other / Uncertain, with the
  method named and confidence shown only when the method provides one.
- **Key-information extraction:** case title, court, case number, parties, judgment date, statutes and
  sections (exactly as written), final order, judges named in the header, and other dates.
- **Case brief:** optional pretrained abstractive model, with a clearly labelled extractive fallback.
- **Honest output:** missing fields read *"Not found in document"*; heuristic fields are flagged
  *verify manually*; the report states whether the whole document was processed.
- **Privacy by default:** uploads are processed in memory and never written to disk by the app.

## Technologies used

| Purpose | Technology |
|---|---|
| Web interface | Streamlit |
| PDF text extraction | PyMuPDF |
| Field extraction | Python `re` (regular expressions) and plain Python |
| Trainable classifier (only if you supply data) | scikit-learn (TF-IDF + Logistic Regression) |
| Optional pretrained models | Hugging Face Transformers + PyTorch |
| Tests | pytest |

## Folder structure

```
CaseBrief-AI/
├── app.py                      # Streamlit interface
├── config.py                   # Limits and model names (env-overridable)
├── document_processor.py       # Paste/TXT/PDF loading, validation, page mapping
├── classifier.py               # Keyword baseline, optional zero-shot, TF-IDF+LogReg training
├── information_extractor.py    # Regex-based field extraction with evidence
├── summarizer.py               # Abstractive (optional) and extractive (fallback) briefs
├── pipeline.py                 # Runs classification + extraction + summary
├── report.py                   # TXT / JSON report builder
├── text_utils.py               # Sentence splitting, chunking helpers
├── sample_data/
│   └── synthetic_sample_judgment.txt   # FICTIONAL demo text, clearly labelled
├── tests/                      # pytest suite (synthetic text only)
├── .streamlit/config.toml      # Upload limit, usage-stats off
├── requirements.txt            # Core dependencies
├── requirements-optional.txt   # Optional Transformers + PyTorch
├── pytest.ini
├── .gitignore
└── README.md
```

## Installation

Requires **Python 3.10 or later**. The commands below create an isolated environment.

**Linux / macOS**

```bash
cd CaseBrief-AI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
cd CaseBrief-AI
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional: pretrained models

The app is fully functional without these. To enable the optional zero-shot classifier and the
abstractive summariser (large downloads — roughly 1.6 GB and 1.2 GB on first use):

```bash
pip install -r requirements-optional.txt
```

Models are fetched from the Hugging Face Hub the first time they are used. If a model cannot be
downloaded or loaded, the app tells you and falls back to the built-in methods.
Defaults can be changed with `CASEBRIEF_ZERO_SHOT_MODEL` and `CASEBRIEF_SUMMARY_MODEL`.

## Running the application

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (normally <http://localhost:8501>).

Optional limit overrides (examples):

```bash
CASEBRIEF_MAX_FILE_MB=20 CASEBRIEF_MAX_TEXT_CHARS=800000 streamlit run app.py
```

If you raise the file-size limit, also raise `maxUploadSize` in `.streamlit/config.toml`.

## How to use

1. Paste judgment text in the **Paste text** tab, or upload a PDF/TXT in the **Upload PDF / TXT** tab.
   (If both are provided, the uploaded file is used.) *New here?* Click **Load synthetic sample**.
2. Check the **Preview** to confirm the text was read correctly, and read any warnings (truncation,
   empty pages, encoding fallbacks).
3. Click **Analyze Judgment**.
4. Review the classification, case information (expand **Source text** on any field), the case brief
   and the verification summary. Open **Extracted document text** to compare with the original.
5. Download the **TXT** or **JSON** report. Use **Clear / Reset** to start over.

## Methodology

### Classification

| Method | When it is used | Confidence shown? |
|---|---|---|
| **Trained TF-IDF + Logistic Regression** | Only if you trained a model and saved it at `models/judgment_classifier.joblib` | Yes — the model probability, labelled as not guaranteed to be calibrated |
| **Zero-shot classifier** (pretrained NLI model) | Only if you tick the option *and* the model loads. Uses the first 3,000 characters | Yes — a relative model score, labelled uncalibrated and unvalidated on Indian judgments |
| **Keyword baseline** | Default, always available | **No.** Scores are weighted keyword-match totals, not probabilities |

**No trained classifier ships with this project**, because no labelled dataset is bundled and none
has been fabricated. The keyword baseline is a set of hand-written rules. It returns
**Other / Uncertain** when the total evidence is weak or when the top two categories are too close.
"Bail" is treated as a more specific sub-type of criminal matters when its evidence is solid. The
matched keywords are shown so you can judge the result yourself.

#### Training a real classifier on a labelled dataset

1. Obtain a labelled corpus of Indian court judgments (for example from a research dataset you are
   licensed to use). Check its licence and privacy terms first.
2. Save it as a UTF-8 CSV with two columns, `text` and `label`, where `label` is one of
   `Criminal`, `Civil`, `Constitutional`, `Bail`, `Property`, `Other / Uncertain`.
3. Train and evaluate:

   ```bash
   python classifier.py train --data data/labelled_judgments.csv
   ```

   This performs a stratified train/held-out split (`--seed` makes it reproducible), fits TF-IDF
   (1–2-grams) + Logistic Regression, prints a per-class precision/recall/F1 report **on the held-out
   split**, and saves the evaluated model to `models/judgment_classifier.joblib` plus a `.json`
   file with the dataset name, sizes and report. The app then uses it automatically.
4. Before relying on it, go further than a single split: use cross-validation, inspect the
   confusion matrix, check for duplicate or near-duplicate judgments across splits (leakage), and
   test on judgments from courts and years that were *not* in training. Consider probability
   calibration if you want to present confidence values.
5. Only load `.joblib` files you created yourself — they can execute code when loaded.

### Key-information extraction

Fields are found with regular expressions and simple heuristics; there is no learned model.

- **Court, case number, parties, title, judges** are searched only in the first 4,000 characters
  (the usual location of a judgment header). The court is never guessed from the body text.
- **Parties** are parsed from an "X versus Y" block; the **case title** is *composed* from them and is
  labelled as such.
- **Judgment date** is accepted only next to a label such as "Date of Judgment" or "Pronounced on".
  A bare "Dated:" near the end is shown but flagged as uncertain. Other dates are listed separately,
  uninterpreted.
- **Statutes and sections** (Section/Sec./u/s, Articles, Order–Rule references) are reproduced
  *exactly as written*. There is **no** automatic IPC→BNS or CrPC→BNSS conversion.
- **Final order** is looked for in the closing part of the text using disposition words (allowed,
  dismissed, disposed of …). If none is reliably found, the app says so.
- Each found value carries a short supporting quote and its location (page, when available, and
  character range of the extracted text). Heuristic fields show a **verify manually** flag.

### Case brief

- **Abstractive (optional):** a pretrained sequence-to-sequence model (default
  `sshleifer/distilbart-cnn-12-6`). The text is split into chunks on sentence boundaries; **every
  chunk is summarised**, the partial summaries are merged (recursively if needed) and condensed to the
  target length. Inputs that would exceed the model's limit are reported. Because such models can
  state things that are not in the judgment, the brief carries an explicit warning.
- **Extractive fallback (default):** selects sentences *copied from the judgment* for background,
  issue/contentions and the court's reasoning, drawing from across the whole text. It is labelled
  "Extractive fallback — NOT an AI-generated abstractive summary" everywhere it appears.
- **The final order is never produced by a model.** It is shown separately, taken from the
  extraction step, or reported as not reliably extractable.
- Very short texts are not summarised.

## Limitations

- Regex extraction is **not** reliable across all Indian court judgments. Formats vary widely by
  court, year and language; unusual headers, multi-party cases, tabular headers and non-English text
  will often be missed or mis-parsed.
- **No OCR.** Scanned PDFs are rejected with an explanatory message. Multi-column layouts, headers/
  footers and footnotes in PDFs can interleave with the body text.
- The keyword classifier is a rough heuristic; Civil/Property overlap is common, and it only
  recognises English terms. No accuracy figures are claimed because none have been measured.
- The extractive brief is a selection of existing sentences and may read abruptly; it can miss the
  true issue or reasoning. The abstractive model is a general-purpose news summariser, not trained on
  Indian law.
- The final-order finder can pick a wrong sentence in judgments with complex or multi-part directions.
- The optional Transformers paths were **not** exercised in the development environment used to
  build this version (see "Testing").
- The app handles one document at a time and does not support Indian-language judgments.

## Testing

```bash
python -m pytest -q
```

The suite uses only **synthetic, fictional** text (clearly labelled) and PDFs generated in memory. It
covers empty input, valid text, missing fields, section extraction (including "no automatic
IPC→BNS conversion"), PDF extraction and page mapping, scanned/corrupted/oversized/wrong-type files,
classification fallback, summary fallback, truncation disclosure, long-document chunking and
map-reduce coverage, and the report formats. The abstractive path is tested with a stand-in generator
to verify chunk coverage; it is not tested against a real downloaded model.

## Future improvements

- Train and publicly evaluate a classifier on a properly licensed, labelled Indian judgment dataset.
- OCR support (e.g. Tesseract) for scanned PDFs.
- NER-based extraction for parties, judges and statutes; handling of multi-party headers.
- Support for Hindi and other Indian-language judgments.
- Legal-domain summarisation models, with evaluation (e.g. against expert-written headnotes).
- Side-by-side source highlighting inside a PDF viewer.
- Batch processing and a structured export for research use.

## Privacy considerations

- Uploaded files and pasted text are processed **in memory** and are **not saved** by the app.
  Streamlit session state holds them only while your session is open; **Clear / Reset** discards
  them from the session.
- The app does not log document contents.
- Reports contain only short supporting quotes, not the full judgment. They still include case
  details, so handle downloaded reports with the same care as the source.
- The built-in methods run locally. If you enable the optional models, the *model files* are
  downloaded from the Hugging Face Hub; your document text is processed locally and is not sent to
  that service.
- Judgments can contain sensitive personal information (victims, minors, witnesses). Make sure you
  are permitted to process what you load, and run the app on a machine you trust.
- Streamlit usage-statistics collection is disabled in `.streamlit/config.toml`.

## Disclaimer

- This is an AI-assisted research prototype.
- Results may be incomplete or inaccurate.
- Verify all information against the original judgment.
- This tool does not provide legal advice.
- It is not affiliated with, endorsed by, or an official system of the Delhi Police or any court,
  government body or legal authority.
