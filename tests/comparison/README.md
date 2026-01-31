# CURE: Python vs C++ Comparison and Plots

This folder contains tests and **visual comparison plots** so you can verify that the C++ CURE implementation is equal or better than the Python one.

## Four test types (same test cases, separate storage and plots)

| Type                | Storage                          | Plots                    |
|---------------------|----------------------------------|---------------------------|
| Euclidean (base)    | `test_data/euclidean/<name>/`    | `plots/euclidean/<name>.png` |
| Pearson (base)      | `test_data/pearson/<name>/`      | `plots/pearson/<name>.png`   |
| Scalable Euclidean  | `test_data/scalable_euclidean/<name>/` | `plots/scalable_euclidean/<name>.png` |
| Scalable Pearson    | `test_data/scalable_pearson/<name>/`   | `plots/scalable_pearson/<name>.png`   |

**One-shot (all 4 types):**
```bash
./tests/comparison/run_all_comparisons.sh
```
This: generates data for all 4 variants, runs C++ comparison for each, then plots into `plots/euclidean/`, `plots/pearson/`, `plots/scalable_euclidean/`, `plots/scalable_pearson/`.

**Step-by-step (single variant, e.g. euclidean):**

1. **Generate** (all 4 variants, same test cases):
   ```bash
   source .venv/bin/activate
   python3 tests/comparison/generate_all_variants.py
   ```

2. **Run C++** (from project root; repeat for each variant):
   ```bash
   ./build/tests/test_comparison tests/comparison/test_data/euclidean
   ./build/tests/test_comparison tests/comparison/test_data/pearson
   ./build/tests/test_comparison tests/comparison/test_data/scalable_euclidean
   ./build/tests/test_comparison tests/comparison/test_data/scalable_pearson
   ```

3. **Plot** (per variant):
   ```bash
   python3 tests/comparison/plot_comparison.py --test_data tests/comparison/test_data/euclidean --out tests/comparison/plots/euclidean
   python3 tests/comparison/plot_comparison.py --test_data tests/comparison/test_data/pearson --out tests/comparison/plots/pearson
   # same for scalable_euclidean, scalable_pearson
   ```

**Legacy (single flat test_data):** `run_comparison.sh` and `generate_test_data.py` / `generate_challenging_tests.py` still work for Euclidean/Pearson-only comparison under `test_data/<name>/` and `plots/`.

## Performance note (Pearson)

Python base Pearson CURE used to run very slowly on larger cases (e.g. test5_large_2d) because it scanned all clusters to find the closest one. It now uses a Euclidean KD-tree (scipy) to get candidate clusters, then computes Pearson distance only for those candidates, so the full comparison run finishes in under a minute.

## Euclidean vs Pearson vs Scalable

- **Euclidean / Pearson**: Base CURE. `params.json` has `"metric": "euclidean"` or `"metric": "pearson"`, `"scalable": false`.
- **Scalable Euclidean / Scalable Pearson**: Scalable CURE. `params.json` has `"scalable": true` and the same `"metric"`. C++ uses `ScalableCURE` when `"scalable": true`.

`generate_all_variants.py` produces the same logical test cases (e.g. `test1_small_2d`, `test4_pattern_10d`) for all 4 types, so you can compare base vs scalable and Euclidean vs Pearson on identical data.

## Plot interpretation

- **Left**: Ground truth (true labels).
- **Middle**: Python CURE result (ARI in title).
- **Right**: C++ CURE result (ARI in title).

If C++ is correct, the middle and right panels should look **identical** (same cluster colors/assignments). The footer shows whether C++ matches or is better than Python (ARI diff).

## Options

- `plot_comparison.py --out DIR` — save plots to `DIR`.
- `plot_comparison.py --tests test1_small_2d,test5_large_2d` — plot only selected cases.
- `plot_comparison.py --format pdf` — output PDF instead of PNG.
