# CURE C++ Library

A C++ implementation of the CURE (Clustering Using REpresentatives) algorithm with Euclidean and Pearson metrics, base and scalable variants.

## Prerequisites

- **CMake** 3.14+
- **C++ compiler** with C++17 (GCC, Clang, or MSVC)

## Workflow

All commands are from the **project root**.

### 1. Build

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
cd ..
```

Release (faster):

```bash
mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && cd ..
```

### 2. Run examples

```bash
./build/examples/cure_example      # Base CURE (Euclidean / Pearson)
./build/examples/kdtree_example    # KD-tree demo
./build/examples/cure_benchmark    # Performance benchmark
```

### 3. Run tests

```bash
cd build && ctest --output-on-failure && cd ..
```

Or run executables directly:

```bash
./build/tests/test_cure
./build/tests/test_distance
./build/tests/test_kdtree
```

`test_comparison` is run by CTest with no arguments; it looks for `tests/comparison/test_data` and **skips** if missing (exit 0), so CTest always passes even without comparison data.

### 4. CLI usage

| Command | Usage |
|--------|--------|
| **Examples** | No arguments. `./build/examples/cure_example`, `./build/examples/kdtree_example`, `./build/examples/cure_benchmark` |
| **test_comparison** | `./build/tests/test_comparison [test_data_dir]` — default: `tests/comparison/test_data`. Use e.g. `tests/comparison/test_data/euclidean` to run one variant. |
| **test_cpp_vs_true** | `./build/tests/test_cpp_vs_true [test_data_dir]` — default: `tests/cpp_vs_true/test_data`. C++ vs true labels only (Euclidean + Pearson). Writes `cpp_results_euclidean.json` and `cpp_results_pearson.json` per case for plotting. |
| **plot_cpp_vs_true.py** | `python3 tests/cpp_vs_true/plot_cpp_vs_true.py [--test_data DIR] [--out DIR] [--show]` — plots True \| C++ Euclidean \| C++ Pearson (same case names as comparison). |
| **run_cpp_vs_true.sh** | `./tests/cpp_vs_true/run_cpp_vs_true.sh` — generate data, run C++ test, then plot (one-shot). |
| **plot_comparison.py** | `python3 tests/comparison/plot_comparison.py --test_data DIR --out DIR [--show] [--tests name1,name2] [--format pdf]` |
| **generate_all_variants.py** | `python3 tests/comparison/generate_all_variants.py` — writes under `tests/comparison/test_data/<variant>/`. |
| **generate_test_data.py** (cpp_vs_true) | `python3 tests/cpp_vs_true/generate_test_data.py` — writes `tests/cpp_vs_true/test_data/`. |
| **clean.sh** | `./scripts/clean.sh` or `./scripts/clean.sh --all` |

Run from **project root** unless noted.

### 5. Python vs C++ comparison (optional)

To compare Python and C++ CURE and generate plots for all four variants (Euclidean, Pearson, Scalable Euclidean, Scalable Pearson):

**One-shot (recommended):**

```bash
./tests/comparison/run_all_comparisons.sh
```

This script (from project root):

1. Activates `.venv` and runs `generate_all_variants.py` (creates test data for all 4 variants).
2. Builds `test_comparison`.
3. Runs `test_comparison` for each variant (`test_data/euclidean`, `pearson`, `scalable_euclidean`, `scalable_pearson`).
4. Runs `plot_comparison.py` per variant; plots go to `tests/comparison/plots/<variant>/`.

**Step-by-step:**

1. **Virtualenv and dependencies** (once):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install numpy scipy scikit-learn matplotlib
   ```

2. **Generate test data** (all 4 variants, same logical test cases):

   ```bash
   source .venv/bin/activate
   python3 tests/comparison/generate_all_variants.py
   ```

3. **Run C++ comparison** (from project root; one per variant):

   ```bash
   ./build/tests/test_comparison tests/comparison/test_data/euclidean
   ./build/tests/test_comparison tests/comparison/test_data/pearson
   ./build/tests/test_comparison tests/comparison/test_data/scalable_euclidean
   ./build/tests/test_comparison tests/comparison/test_data/scalable_pearson
   ```

4. **Plot** (per variant):

   ```bash
   python3 tests/comparison/plot_comparison.py --test_data tests/comparison/test_data/euclidean --out tests/comparison/plots/euclidean
   python3 tests/comparison/plot_comparison.py --test_data tests/comparison/test_data/pearson --out tests/comparison/plots/pearson
   # same for scalable_euclidean, scalable_pearson
   ```

   Add `--show` to also display the matplotlib window.

See `tests/comparison/README.md` for variant layout, plot options, and interpretation.

### 6. Clean

```bash
./scripts/clean.sh       # Remove build/, python_code __pycache__
./scripts/clean.sh --all # Also remove test_data, plots, python_code/venv (.venv is kept)
```

---

## Algorithm options

**Base CURE – outlier-resistant sampling:**  
Use only a fraction of points (closest to global centroid) for the merge phase, then assign all points to the nearest cluster. Set `config.outlier_sample_fraction` (e.g. `0.5`, `0.2`); `0` = disabled. Works with Euclidean and Pearson.

**Scalable CURE – sample size and sampling:**  
- **Auto sample size** (`sample_size_auto = true`): sublinear formula so that 1M points → ~2k–5k, 1B → ~20k–50k, with min/max bounds.  
- **Centroid-based sampling** (`use_centroid_sampling = true`): sample = points nearest to global centroid (outlier-resistant) instead of random.

---

## Project structure

| Path | Description |
|------|-------------|
| `include/cure/` | Public headers (use `#include "cure/cure/cure.hpp"` etc.) |
| `src/` | Library source |
| `examples/` | Example apps |
| `tests/` | Unit tests; `tests/comparison/` = Python vs C++ and plots; `tests/cpp_vs_true/` = C++ vs true labels only (Euclidean + Pearson) |
| `python_code/` | Reference Python CURE implementations |
| `scripts/clean.sh` | Remove build and generated files |
