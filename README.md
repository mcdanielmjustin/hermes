# Hermes - EPPP Question Generation Pipeline

A fork of the Goliath pipeline, configured to use **Qwen via Nous Research** instead of Anthropic Claude.

## Overview

Hermes generates 4-tier multiple-choice questions for EPPP exam preparation from research-backed anchor points across 9 psychology domains.

**Key Difference from Goliath:** Uses OpenAI-compatible API (Qwen models via Nous) instead of Anthropic Claude.

## Quick Start

### 1. Install Dependencies

```bash
cd hermes
pip install -r requirements.txt
```

### 2. Configure API Key

Set your Nous API key:

```bash
export NOUS_API_KEY="your-key-here"
export API_BASE_URL="https://inference-api.nousresearch.com/v1"
```

Or create a `.env` file:
```
NOUS_API_KEY=your-key-here
API_BASE_URL=https://inference-api.nousresearch.com/v1
```

### 3. Generate Anchor Briefs (Optional but Recommended)

```bash
cd scripts
python generate_anchor_briefs.py --domain BPSY
```

### 4. Generate Questions

**Generate for a single anchor:**
```bash
python generate_quiz_questions.py --anchor D7-PHY-021-b323a513 --domain BPSY
```

**Generate for all BPSY anchors:**
```bash
python generate_quiz_questions.py --domain BPSY --workers 5
```

**Generate for all domains:**
```bash
python generate_quiz_questions.py --all --workers 10
```

**Dry run (preview without API calls):**
```bash
python generate_quiz_questions.py --domain BPSY --anchor D7-PHY-021-b323a513 --dry-run
```

## Configuration

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--provider` | `nous` | API provider: `nous` or `anthropic` |
| `--model` | `qwen/qwen3.5-plus-02-15` | Model to use |
| `--base-url` | Nous API | Custom API base URL |
| `--domain` | - | Domain code (BPSY, CASS, CPAT, etc.) |
| `--anchor` | - | Single anchor UID |
| `--workers` | 5 | Concurrent API workers |
| `--difficulty` | - | Single tier (1-4) |
| `--count` | 5 | Variants per anchor per tier |
| `--dry-run` | False | Preview without API calls |
| `--resume` | False | Skip already-generated questions |

### Environment Variables

```bash
# Required for Nous/Qwen
export NOUS_API_KEY="your-key"
export API_BASE_URL="https://inference-api.nousresearch.com/v1"

# Optional
export DEFAULT_MODEL="qwen/qwen3.5-plus-02-15"
export API_PROVIDER="nous"  # or "anthropic"
```

## Domain Codes

| Code | Domain |
|------|--------|
| PMET | Psychometrics & Research Methods |
| LDEV | Lifespan & Developmental Stages |
| CPAT | Clinical Psychopathology |
| PTHE | Psychotherapy Models & Interventions |
| SOCu | Social & Cultural Psychology |
| WDEV | Workforce Development & Leadership |
| BPSY | Biopsychology |
| CASS | Clinical Assessment & Interpretation |
| PETH | Psychopharmacology & Ethics |

## Project Structure

```
hermes/
├── config.py              # Configuration paths and API settings
├── pipeline/              # Core pipeline code
│   ├── api_client.py      # OpenAI/Anthropic client wrapper
│   ├── orchestrator.py    # Main orchestration logic
│   ├── agents.py          # Agent classes
│   ├── gates.py           # Validation gates
│   └── prompts.py         # Prompt templates
├── scripts/               # Generation scripts
│   ├── generate_quiz_questions.py
│   ├── generate_anchor_briefs.py
│   └── ...
├── csvs/                  # Source data (committed)
│   ├── anchor_points.csv
│   ├── anchor_passages_v3.csv
│   └── chapter_schema_v3.csv
├── data/                  # Generated output (gitignored)
│   ├── anchor_briefs/
│   ├── concept_vocab/
│   └── quiz/
└── tests/                 # Test suite
```

## Model Comparison

| Provider | Model | Cost | Notes |
|----------|-------|------|-------|
| Nous | qwen/qwen3.5-plus-02-15 | ~50% less than Claude | Default, recommended |
| Anthropic | claude-opus-4-7 | Higher | Original Goliath default |

## Migration from Goliath

1. Clone this repo instead of Goliath
2. Set `NOUS_API_KEY` instead of `ANTHROPIC_API_KEY`
3. Run the same commands - everything else is identical!

To use Anthropic instead:
```bash
python generate_quiz_questions.py --provider anthropic --domain BPSY
```

## Testing

```bash
cd tests
python -m pytest -v
# or
python -m unittest discover -v
```

## License

MIT License (same as Goliath)

## Original Project

This is a fork of [Goliath](https://github.com/mcdanielmjustin/goliath) by mcdanielmjustin.
