"""
Centralized path configuration for the Hermes pipeline.
Forked from Goliath - configured for Qwen/Nous provider.
"""

import os
from pathlib import Path

# ── Repo layout (auto-detected) ──────────────────────────────
REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
QUIZ_DIR = DATA_DIR / "quiz"
QUIZ_SHIPPABLE_DIR = DATA_DIR / "quiz_shippable"
QUIZ_REVIEW_DIR = DATA_DIR / "quiz_review"
BATCH_DIR = DATA_DIR / "batch"
SCRIPTS_DIR = REPO_ROOT / "scripts"
LOG_DIR = REPO_ROOT / "logs"

# ── Internal data (inside hermes) ────────────────────────────
CONCEPT_VOCAB_DIR = DATA_DIR / "concept_vocab"
ANCHOR_BRIEFS_DIR = DATA_DIR / "anchor_briefs"
DOMAIN_VOCAB_DIR = DATA_DIR / "domain_vocab"

# ── Source CSVs ─────────────────────────────────────────────
MASTER_CSV_DIR = REPO_ROOT / "csvs"
ANCHOR_POINTS_CSV = MASTER_CSV_DIR / "anchor_points.csv"
CHAPTER_SCHEMA_CSV = MASTER_CSV_DIR / "chapter_schema_v3.csv"
ANCHOR_PASSAGES_CSV = MASTER_CSV_DIR / "anchor_passages_v3_pure_textbook_1081.csv"

# ── Distribution output ─────────────────────────────────────
ONEDRIVE_DISTRIBUTION_DIR = Path(
    os.getenv("HERMES_ONEDRIVE_DIR")
    or Path("/mnt/c/Users/Admin/OneDrive/Master CSVs")
)
ENRICHMENT_CSV = DATA_DIR / "enrichment_all_questions.csv"
ENRICHMENT_CSV_MASTER = ONEDRIVE_DISTRIBUTION_DIR / "enrichment_all_questions.csv"

# ── Output targets ──────────────────────────────────────────
PASSEPPP_DIR = Path(
    os.getenv("HERMES_PASSEPPP_DIR")
    or (Path.home() / "PassEPPP-website")
)
ENRICHMENT_BUNDLE_DIR = PASSEPPP_DIR / "content" / "enrichment"

# ── Checkpoint files ────────────────────────────────────────
QUIZ_CHECKPOINT = SCRIPTS_DIR / "quiz_checkpoint.json"
CONCEPT_VOCAB_CHECKPOINT = SCRIPTS_DIR / "concept_vocab_checkpoint.json"
BATCH_MANIFEST = BATCH_DIR / "manifest.json"

# ── API Configuration ───────────────────────────────────────
# Default to Nous/Qwen, but support Anthropic too
API_PROVIDER = os.getenv("API_PROVIDER", "nous")  # "nous" or "anthropic"

if API_PROVIDER == "nous":
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen/qwen3.5-plus-02-15")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    API_KEY_ENV = "NOUS_API_KEY"
else:
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-opus-4-7")
    API_BASE_URL = None  # Anthropic uses direct API
    API_KEY_ENV = "ANTHROPIC_API_KEY"
