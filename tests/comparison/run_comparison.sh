#!/bin/bash
# Run full comparison test and generate plots: Python vs C++ CURE

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "CURE Algorithm: Python vs C++ Comparison"
echo "=========================================="
echo ""

# Step 1: Generate Python test data
echo "[Step 1] Generating test data with Python CURE..."
python3 tests/comparison/generate_test_data.py
python3 tests/comparison/generate_challenging_tests.py 2>/dev/null || true

echo ""
echo "[Step 2] Building C++ tests..."
cd build && make test_comparison -j4 && cd ..

echo ""
echo "[Step 3] Running C++ comparison (writes cpp_results.json; must run from project root)..."
./build/tests/test_comparison tests/comparison/test_data

echo ""
echo "[Step 4] Generating comparison plots..."
if command -v python3 &>/dev/null; then
  (source .venv/bin/activate 2>/dev/null || true; python3 tests/comparison/plot_comparison.py)
  echo "Plots saved to: tests/comparison/plots/"
else
  echo "Skipped (python3 not found). Run: python3 tests/comparison/plot_comparison.py"
fi

echo ""
echo "=========================================="
echo "Comparison complete! Check tests/comparison/plots/"
echo "=========================================="
