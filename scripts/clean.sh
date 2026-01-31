#!/bin/bash
# Remove build artifacts, Python caches, and optionally generated data and venvs.
# Run from project root: ./scripts/clean.sh [--all]

set -e
cd "$(dirname "$0")/.."

echo "Removing build/..."
rm -rf build

echo "Removing Python __pycache__/ and *.pyc in python_code/..."
find python_code -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find python_code -type f -name "*.pyc" -delete 2>/dev/null || true

if [[ "$1" == "--all" ]]; then
  echo "Removing tests/comparison/test_data/..."
  rm -rf tests/comparison/test_data
  echo "Removing tests/comparison/plots/..."
  rm -rf tests/comparison/plots
  echo "Removing python_code/venv/..."
  rm -rf python_code/venv
  echo "Clean (all) done. (.venv/ is kept)"
else
  echo "Clean done. Use ./scripts/clean.sh --all to also remove test_data, plots, and python_code/venv"
fi
