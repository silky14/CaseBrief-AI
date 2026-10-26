"""Central, configurable settings for CaseBrief AI.

Every limit can be overridden with an environment variable, e.g.

    CASEBRIEF_MAX_FILE_MB=20 streamlit run app.py
"""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to *default*."""
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


# --- Input limits -----------------------------------------------------------
MAX_FILE_MB = _env_int("CASEBRIEF_MAX_FILE_MB", 10)
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
MAX_TEXT_CHARS = _env_int("CASEBRIEF_MAX_TEXT_CHARS", 400_000)  # roughly 100+ pages
MAX_PDF_PAGES = _env_int("CASEBRIEF_MAX_PDF_PAGES", 300)
ALLOWED_EXTENSIONS = (".pdf", ".txt")

# A PDF yielding fewer characters than this is treated as "no extractable text".
MIN_PDF_TEXT_CHARS = 50
# Below this length the analysis is run but flagged as unreliable.
MIN_CHARS_FOR_ANALYSIS = 200
MIN_CHARS_FOR_BRIEF = 400

PREVIEW_CHARS = 3000

# --- Extraction -------------------------------------------------------------
# Court, parties, case number and judges are looked for only in this leading
# portion of the document (where judgment headers normally appear).
HEADER_CHARS = 4000

# --- Summarisation ----------------------------------------------------------
SUMMARY_TARGET_WORDS = (100, 150)
CHUNK_CHARS = 2500  # characters per chunk sent to the optional summarisation model
SUMMARY_MODEL = os.environ.get("CASEBRIEF_SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")

# --- Classification ---------------------------------------------------------
ZERO_SHOT_MODEL = os.environ.get("CASEBRIEF_ZERO_SHOT_MODEL", "facebook/bart-large-mnli")
ZERO_SHOT_INPUT_CHARS = 3000
ZERO_SHOT_MIN_SCORE = 0.5  # below this the result is reported as Other / Uncertain
TRAINED_MODEL_PATH = os.environ.get("CASEBRIEF_MODEL_PATH", "models/judgment_classifier.joblib")
TRAINED_MIN_PROBABILITY = 0.5
