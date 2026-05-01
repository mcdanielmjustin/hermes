"""
Test script to verify Hermes pipeline setup with Qwen
"""
import os
import sys
from pathlib import Path

# Add scripts to path
SCRIPT_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

# Test 1: Import config
print("Test 1: Importing config...")
try:
    from config import API_PROVIDER, DEFAULT_MODEL, API_BASE_URL
    print(f"  ✓ Config loaded: {API_PROVIDER} / {DEFAULT_MODEL}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import API client
print("\nTest 2: Importing API client...")
try:
    from pipeline.api_client import create_client
    print("  ✓ API client imported")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 3: Check API key
print("\nTest 3: Checking API key...")
api_key = os.getenv("NOUS_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
if api_key:
    print(f"  ✓ API key found: {api_key[:10]}...")
else:
    print("  ⚠ No API key found - set NOUS_API_KEY or ANTHROPIC_API_KEY")

# Test 4: Create client (without calling API)
print("\nTest 4: Creating client...")
try:
    if api_key:
        client = create_client(
            provider=API_PROVIDER,
            api_key=api_key,
            base_url=API_BASE_URL,
            model=DEFAULT_MODEL
        )
        print(f"  ✓ Client created successfully")
    else:
        print("  ⊘ Skipped (no API key)")
except Exception as e:
    print(f"  ✗ Failed: {e}")

# Test 5: Load source data
print("\nTest 5: Loading source data...")
try:
    import csv
    csv_path = Path(__file__).parent / "csvs" / "anchor_points.csv"
    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
        print(f"  ✓ Loaded {count} anchor points")
    else:
        print("  ✗ CSV not found")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\n=== All tests completed ===")
