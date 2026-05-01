"""Goliath Pipeline Configuration (Hermes fork - Qwen support)"""
from pathlib import Path
import os

# Repository root
REPO_ROOT = Path(__file__).parent

# Source data directory (CSVs in repo)
MASTER_CSV_DIR = REPO_ROOT / "csvs"

# OneDrive distribution directory (for generated outputs)
ONEDRIVE_DISTRIBUTION_DIR = Path(os.getenv(
    "ONEDRIVE_DISTRIBUTION_DIR",
    "/mnt/c/Users/mcdan/OneDrive - PassEPPP/Master CSVs"
))

# Data directories (gitignored, generated output)
DATA_ROOT = REPO_ROOT / "data"
ANCHOR_BRIEFS_DIR = DATA_ROOT / "anchor_briefs"
CONCEPT_VOCAB_DIR = DATA_ROOT / "concept_vocab"
DOMAIN_VOCAB_DIR = DATA_ROOT / "domain_vocab"
QUIZ_DIR = DATA_ROOT / "quiz"

# Pipeline output
OUTPUT_DIR = DATA_ROOT / "output"

# API Configuration
# Default to Nous/Qwen, but support Anthropic too
API_PROVIDER = os.getenv("API_PROVIDER", "nous")  # "nous" or "anthropic"

if API_PROVIDER == "nous":
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen/qwen3.5-plus-02-15")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://inference-api.nousresearch.com/v1")
    API_KEY_ENV = "NOUS_API_KEY"
else:
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-opus-4-7")
    API_BASE_URL = None  # Anthropic uses direct API
    API_KEY_ENV = "ANTHROPIC_API_KEY"
