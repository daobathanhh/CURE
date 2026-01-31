#!/bin/bash
# Generate test data, run C++ vs true labels, then plot (True | C++ Euclidean | C++ Pearson).
# Same test case names as comparison tests. Run from project root.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

echo "=============================================="
echo "C++ vs true labels (Euclidean + Pearson)"
echo "=============================================="
echo ""

echo "[1] Generating test data (same cases as comparison)..."
python3 tests/cpp_vs_true/generate_test_data.py

echo ""
echo "[2] Building and running test_cpp_vs_true..."
cd build && make test_cpp_vs_true -j4 2>/dev/null || (cmake .. && make test_cpp_vs_true -j4)
cd ..
./build/tests/test_cpp_vs_true tests/cpp_vs_true/test_data

echo ""
echo "[3] Plotting (True | C++ Euclidean | C++ Pearson)..."
python3 tests/cpp_vs_true/plot_cpp_vs_true.py --test_data tests/cpp_vs_true/test_data --out tests/cpp_vs_true/plots

echo ""
echo "=============================================="
echo "Done. Plots: tests/cpp_vs_true/plots/"
echo "  Add --show to plot_cpp_vs_true.py to display."
echo "=============================================="
