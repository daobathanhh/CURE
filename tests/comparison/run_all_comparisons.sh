#!/bin/bash
# Run comparison for all 4 CURE types: euclidean, pearson, scalable_euclidean, scalable_pearson.
# Generates same test cases for each, runs C++ comparison, then plots. Results in plots/<variant>/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

VARIANTS="euclidean pearson scalable_euclidean scalable_pearson"

echo "=============================================="
echo "CURE comparison: 4 types (same test cases)"
echo "=============================================="
echo ""

echo "[1] Generating test data for all 4 variants..."
source .venv/bin/activate 2>/dev/null || true
python3 tests/comparison/generate_all_variants.py

echo ""
echo "[2] Building C++ comparison test..."
cd build && make test_comparison -j4 && cd ..

echo ""
echo "[3] Running C++ comparison for each variant..."
for v in $VARIANTS; do
  echo "  -> $v"
  ./build/tests/test_comparison "tests/comparison/test_data/$v" || true
done

echo ""
echo "[4] Generating plots per variant (plots/<variant>/)..."
for v in $VARIANTS; do
  echo "  -> plots/$v/"
  python3 tests/comparison/plot_comparison.py \
    --test_data "tests/comparison/test_data/$v" \
    --out "tests/comparison/plots/$v" || true
done

echo ""
echo "=============================================="
echo "Done. Plots: tests/comparison/plots/<variant>/"
echo "  - euclidean/"
echo "  - pearson/"
echo "  - scalable_euclidean/"
echo "  - scalable_pearson/"
echo "=============================================="
