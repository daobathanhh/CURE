#!/usr/bin/env python3
"""
Plot CURE clustering results: True labels vs Python CURE vs C++ CURE.

Uses matplotlib to draw side-by-side scatter plots so you can visually
verify that C++ produces equal or better clustering than Python.

Usage:
  1. Generate test data:    python3 generate_test_data.py
                            python3 generate_challenging_tests.py
  2. Run C++ to write cpp_results.json:  ./build/tests/test_comparison tests/comparison/test_data
  3. Plot:                  python3 plot_comparison.py [--show] [--out DIR] [--tests NAME,...]
"""

import os
import sys
import json
import argparse
import numpy as np

# Parse --show early so we can choose matplotlib backend before importing pyplot
def _parse_show():
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true")
    return p.parse_known_args()[0].show

try:
    import matplotlib
    if not _parse_show():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None


def load_test_case(test_dir):
    """Load data, true labels, Python results, and C++ results for a test case."""
    data_path = os.path.join(test_dir, 'data.csv')
    true_path = os.path.join(test_dir, 'true_labels.csv')
    params_path = os.path.join(test_dir, 'params.json')
    py_path = os.path.join(test_dir, 'python_results.json')
    cpp_path = os.path.join(test_dir, 'cpp_results.json')

    data = np.loadtxt(data_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    true_labels = np.atleast_1d(np.loadtxt(true_path, delimiter=',', dtype=int))
    if len(true_labels) != len(data):
        true_labels = np.atleast_1d(np.loadtxt(true_path, dtype=int))

    with open(params_path) as f:
        params = json.load(f)

    with open(py_path) as f:
        py_results = json.load(f)

    # Result key: "result" (single variant) or "euclidean"/"pearson"
    result_key = None
    for k in ('result', 'euclidean', 'pearson'):
        if k in py_results and isinstance(py_results.get(k), dict) and 'labels' in py_results[k]:
            result_key = k
            break
    if result_key is None:
        return None, None, None, None, None, params, "No result/euclidean/pearson labels in python_results.json"

    py_labels = np.array(py_results[result_key]['labels'])
    py_ari = py_results[result_key].get('ari_vs_true', None)

    if not os.path.exists(cpp_path):
        return None, None, None, None, None, params, f"Missing {cpp_path} — run: ./build/tests/test_comparison tests/comparison/test_data"

    with open(cpp_path) as f:
        cpp_results = json.load(f)
    cpp_labels = np.array(cpp_results['labels'])
    cpp_ari = cpp_results.get('ari_vs_true', None)

    if len(cpp_labels) != len(data) or len(py_labels) != len(data):
        return None, None, None, None, None, params, "Label length mismatch"

    return data, true_labels, py_labels, cpp_labels, (py_ari, cpp_ari), params, None


def get_2d_coords(data):
    """Return (X, Y) for plotting: use first 2 dims if 2D, else PCA."""
    if data.shape[1] == 2:
        return data[:, 0], data[:, 1]
    if data.shape[1] == 1:
        return data[:, 0], np.zeros_like(data[:, 0])
    if PCA is not None:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(data)
        return xy[:, 0], xy[:, 1]
    return data[:, 0], data[:, 1]  # fallback: first two features


def scatter_ax(ax, x, y, labels, title, ari=None):
    """Single scatter subplot with cluster colors."""
    uniq = np.unique(labels)
    n_clusters = len(uniq)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        ax.scatter(x[mask], y[mask], c=[colors[i % len(colors)]], label=f'C{lab}', s=18, alpha=0.8, edgecolors='none')
    ax.set_title(title + (f"  (ARI={ari:.3f})" if ari is not None else ""), fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=6)


def plot_one_case(test_name, test_dir, out_dir, ext='png', dpi=120, show=False):
    """Create one figure with matplotlib: True | Python CURE | C++ CURE."""
    result = load_test_case(test_dir)
    data, true_labels, py_labels, cpp_labels, aris, params, err = result[0], result[1], result[2], result[3], result[4], result[5], result[6]
    if err:
        print(f"  Skip {test_name}: {err}")
        return False

    x, y = get_2d_coords(data)
    py_ari, cpp_ari = aris if aris else (None, None)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    fig.suptitle(f"{test_name}  (k={params['k']}, c={params['c']}, α={params['alpha']})", fontsize=11)

    scatter_ax(axes[0], x, y, true_labels, "True labels", ari=None)
    scatter_ax(axes[1], x, y, py_labels, "Python CURE", ari=py_ari)
    scatter_ax(axes[2], x, y, cpp_labels, "C++ CURE", ari=cpp_ari)

    # Emphasize equality or superiority
    if py_ari is not None and cpp_ari is not None:
        diff = cpp_ari - py_ari
        if abs(diff) < 1e-5:
            fig.text(0.5, 0.02, "C++ matches Python (same ARI).", ha='center', fontsize=9, style='italic')
        elif diff > 0:
            fig.text(0.5, 0.02, f"C++ better than Python (ARI diff = +{diff:.4f}).", ha='center', fontsize=9, style='italic')
        else:
            fig.text(0.5, 0.02, f"C++ vs Python ARI diff = {diff:.4f}.", ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if out_dir:
        out_path = os.path.join(out_dir, f"{test_name}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {out_path}")

    if show:
        plt.show()
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(description="Plot CURE with matplotlib: True vs Python vs C++")
    ap.add_argument("--test_data", default=None, help="Test data directory (default: tests/comparison/test_data)")
    ap.add_argument("--out", default=None, help="Output directory for plots (default: tests/comparison/plots)")
    ap.add_argument("--tests", default=None, help="Comma-separated test names (default: all)")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Image format")
    ap.add_argument("--dpi", type=int, default=120, help="DPI for raster formats")
    ap.add_argument("--show", action="store_true", help="Display plots with matplotlib (interactive window)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = args.test_data or os.path.join(project_dir, "tests", "comparison", "test_data")
    out_dir = args.out or os.path.join(project_dir, "tests", "comparison", "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(test_data_dir):
        print(f"Test data directory not found: {test_data_dir}")
        sys.exit(1)

    names = sorted(d for d in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, d)))
    if args.tests:
        requested = set(t.strip() for t in args.tests.split(","))
        names = [n for n in names if n in requested]
        if not names:
            print("No matching test names found.")
            sys.exit(1)

    print(f"Plotting {len(names)} test cases -> {out_dir}" + (" (and displaying)" if args.show else ""))
    print("=" * 60)
    count = 0
    for name in names:
        if plot_one_case(name, os.path.join(test_data_dir, name), out_dir, ext=args.format, dpi=args.dpi, show=args.show):
            count += 1
    print("=" * 60)
    print(f"Done. {count}/{len(names)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
