#!/bin/bash
# Quick Start Script for Hermes Pipeline

set -e

echo "=== Hermes Pipeline Quick Start ==="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "✏️  Please edit .env and add your NOUS_API_KEY"
    echo ""
    exit 1
fi

# Load environment
source .env

echo "✓ Configuration loaded"
echo "  Provider: ${API_PROVIDER:-nous}"
echo "  Model: ${DEFAULT_MODEL:-qwen/qwen3.5-plus-02-15}"
echo ""

# Check API key
if [ -z "$NOUS_API_KEY" ] && [ "$API_PROVIDER" = "nous" ]; then
    echo "✗ Error: NOUS_API_KEY not set"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt -q
echo "✓ Dependencies installed"
echo ""

# Show usage
echo "=== Quick Start Commands ==="
echo ""
echo "1. Generate a test question (dry run):"
echo "   cd scripts"
echo "   python generate_quiz_questions.py --anchor D7-PHY-021-b323a513 --domain BPSY --dry-run"
echo ""
echo "2. Generate questions for BPSY domain:"
echo "   python generate_quiz_questions.py --domain BPSY --workers 5"
echo ""
echo "3. Generate anchor briefs first (recommended):"
echo "   python generate_anchor_briefs.py --domain BPSY"
echo ""
echo "=== Ready! ==="
