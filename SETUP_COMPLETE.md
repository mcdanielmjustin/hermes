# Hermes Pipeline - Setup Complete ✓

## Repository Created

**URL:** https://github.com/mcdanielmjustin/hermes

## What Was Done

### 1. Forked Goliath Codebase
- Copied entire Goliath pipeline to new `hermes` repository
- Preserved all original functionality

### 2. Added Qwen/Nous Support
- Created `pipeline/api_client.py` - OpenAI-compatible wrapper
- Updated `config.py` - Support for multiple providers
- Modified `scripts/generate_quiz_questions.py` - Added `--provider`, `--model`, `--base-url` flags
- Updated `pipeline/orchestrator.py` - Compatible with new client interface

### 3. Configuration
- Default provider: `nous` (Qwen via OpenRouter)
- Default model: `qwen/qwen3.5-plus-02-15`
- API base URL: `https://openrouter.ai/api/v1`

### 4. Documentation
- Comprehensive README.md
- `.env.example` template
- `quickstart.sh` script
- `test_setup.py` verification script

## How to Use

### Clone the Repository
```bash
git clone https://github.com/mcdanielmjustin/hermes.git
cd hermes
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configure API Key
The `.env` file is already configured with your OpenRouter key.

Or set manually:
```bash
export NOUS_API_KEY="sk-or-your-key"
export API_BASE_URL="https://openrouter.ai/api/v1"
export DEFAULT_MODEL="qwen/qwen3.5-plus-02-15"
```

### Generate Questions

**Dry run (test without API calls):**
```bash
cd scripts
python generate_quiz_questions.py --anchor D7-PHY-021-b323a513 --domain BPSY --dry-run
```

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

### Use Anthropic Instead (optional)
```bash
python generate_quiz_questions.py --provider anthropic --domain BPSY
```

## Next Steps - Generate Questions from BPSY Anchor

Now you're ready to generate questions! The anchor point we identified:

**Anchor:** `D7-PHY-021-b323a513`
**Domain:** BPSY (Biopsychology)
**Chapter:** D7-Ch11 - The Biology of Memory and Sleep
**Content:** Nondeclarative/implicit memory, basal ganglia, cerebellum, hippocampal system

Run this command:
```bash
cd /home/justin/hermes/scripts
python generate_quiz_questions.py --anchor D7-PHY-021-b323a513 --domain BPSY
```

This will generate:
- 4 tiers (difficulty levels)
- 5 variants per tier
- Total: 20 questions

Output will be saved to: `/home/justin/hermes/data/quiz/BPSY/`

## Cost Estimate

Using OpenRouter with Qwen:
- ~50% less expensive than Anthropic Claude
- Estimated cost per question: ~$0.10-0.15
- 20 questions (1 anchor): ~$2-3
- Full BPSY domain (192 anchors × 4 tiers × 5 variants = 3,840 questions): ~$384-576

## Key Files Modified

- `pipeline/api_client.py` (NEW) - Universal API wrapper
- `config.py` - Added multi-provider support
- `scripts/generate_quiz_questions.py` - Added CLI args
- `pipeline/orchestrator.py` - Updated client interface
- `requirements.txt` - Added openai package

## Support

If you encounter issues:
1. Run `python test_setup.py` to verify configuration
2. Use `--dry-run` to test without API charges
3. Check the README.md for detailed documentation
