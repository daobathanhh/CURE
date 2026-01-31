# C++ vs true labels only

This test runs **C++ CURE only** against ground-truth labels and can **show plots** (True | C++ Euclidean | C++ Pearson). It does **not** use or overwrite the Python comparison test data.

**Test cases** are the same as in `tests/comparison/generate_all_variants.py`: `test1_small_2d` … `test7_pattern_15d`, so you can compare results and plots with the comparison workflow.

## One-shot (generate + run C++ + plot)

From project root:

```bash
./tests/cpp_vs_true/run_cpp_vs_true.sh
```

This: generates test data, runs `test_cpp_vs_true`, then runs `plot_cpp_vs_true.py`. Plots go to `tests/cpp_vs_true/plots/<name>.png`.

## Step-by-step

1. **Generate test data** (same cases as comparison):

   ```bash
   python3 tests/cpp_vs_true/generate_test_data.py
   ```

   Creates `tests/cpp_vs_true/test_data/<name>/` with `data.csv`, `true_labels.csv`, `params.json` (no Python CURE).

2. **Run C++ test** (writes `cpp_results_euclidean.json` and `cpp_results_pearson.json` per case):

   ```bash
   ./build/tests/test_cpp_vs_true
   # or: ./build/tests/test_cpp_vs_true tests/cpp_vs_true/test_data
   ```

3. **Plot** (True | C++ Euclidean | C++ Pearson):

   ```bash
   python3 tests/cpp_vs_true/plot_cpp_vs_true.py --out tests/cpp_vs_true/plots
   python3 tests/cpp_vs_true/plot_cpp_vs_true.py --out tests/cpp_vs_true/plots --show   # also display
   ```

   Options: `--test_data DIR`, `--out DIR`, `--tests name1,name2`, `--format pdf`, `--show`.

## CTest

```bash
cd build && ctest -R test_cpp_vs_true
```

If test data is missing, the test skips (exit 0) so CTest still passes.
