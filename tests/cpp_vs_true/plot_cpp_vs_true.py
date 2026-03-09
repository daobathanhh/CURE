#!/usr/bin/env python3
"""
Plot CURE vs true labels.

Default: True | C++ Euclidean | C++ Pearson.

You can also use pyclustering's CURE implementation instead of the C++ results:
  - With --pycure, plots True | pyclustering CURE (Euclidean).

Expects tests/cpp_vs_true/test_data/<name>/ with:
  - data.csv, true_labels.csv, params.json
  - and for C++ plots: cpp_results_euclidean.json, cpp_results_pearson.json
    (written by test_cpp_vs_true after running). Same test case names as comparison tests.

Usage:
  python3 tests/cpp_vs_true/plot_cpp_vs_true.py [--test_data DIR] [--out DIR] [--show] [--pycure]
"""

import os
import sys
import json
import argparse

def _parse_show():
    p = argparse.ArgumentParser()
    p.add_argument("--show", action="store_true")
    return p.parse_known_args()[0].show

try:
    import matplotlib
    if not _parse_show():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError as e:
    print("Install matplotlib: pip install matplotlib", e)
    sys.exit(1)

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = None

try:
    from sklearn.metrics import adjusted_rand_score
except ImportError:
    adjusted_rand_score = None

import numpy as np


def load_test_case(test_dir):
    """Load data, true labels, C++ Euclidean and Pearson results."""
    data_path = os.path.join(test_dir, 'data.csv')
    true_path = os.path.join(test_dir, 'true_labels.csv')
    params_path = os.path.join(test_dir, 'params.json')
    euc_path = os.path.join(test_dir, 'cpp_results_euclidean.json')
    pearson_path = os.path.join(test_dir, 'cpp_results_pearson.json')

    if not os.path.exists(data_path) or not os.path.exists(true_path):
        return None, None, None, None, None, None, "Missing data.csv or true_labels.csv"
    data = np.loadtxt(data_path, delimiter=',')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    true_labels = np.atleast_1d(np.loadtxt(true_path, delimiter=',', dtype=int))
    if len(true_labels) != len(data):
        true_labels = np.atleast_1d(np.loadtxt(true_path, dtype=int))

    with open(params_path) as f:
        params = json.load(f)

    if not os.path.exists(euc_path):
        return None, None, None, None, None, params, f"Missing {euc_path} — run: ./build/tests/test_cpp_vs_true"
    if not os.path.exists(pearson_path):
        return None, None, None, None, None, params, f"Missing {pearson_path} — run: ./build/tests/test_cpp_vs_true"

    with open(euc_path) as f:
        euc_results = json.load(f)
    with open(pearson_path) as f:
        pearson_results = json.load(f)

    euc_labels = np.array(euc_results['labels'])
    pearson_labels = np.array(pearson_results['labels'])
    euc_ari = euc_results.get('ari_vs_true')
    pearson_ari = pearson_results.get('ari_vs_true')

    if len(euc_labels) != len(data) or len(pearson_labels) != len(data):
        return None, None, None, None, None, params, "Label length mismatch"

    return data, true_labels, euc_labels, pearson_labels, (euc_ari, pearson_ari), params, None


def _import_pyclust_cure():
    """
    Import the local pyclustering CURE implementation from python_code/pyclustering_cure.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    cure_dir = os.path.join(project_dir, "python_code", "pyclustering_cure")
    if cure_dir not in sys.path:
        sys.path.insert(0, cure_dir)
    try:
        # Module file is pyclustering_cure.py and defines class `cure`.
        from pyclustering_cure import cure  # type: ignore
    except Exception as e:
        print(
            "Failed to import local pyclustering CURE implementation. "
            "Expected python_code/pyclustering_cure/pyclustering_cure.py",
            e,
        )
        sys.exit(1)
    return cure


def load_test_case_pycure(test_dir):
    """
    Load data, true labels, params.json and run pyclustering.cure (Euclidean) instead of using C++ results.
    """
    data_path = os.path.join(test_dir, "data.csv")
    true_path = os.path.join(test_dir, "true_labels.csv")
    params_path = os.path.join(test_dir, "params.json")

    if not os.path.exists(data_path) or not os.path.exists(true_path):
        return None, None, None, None, None, "Missing data.csv or true_labels.csv"

    data = np.loadtxt(data_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    true_labels = np.atleast_1d(np.loadtxt(true_path, delimiter=",", dtype=int))
    if len(true_labels) != len(data):
        true_labels = np.atleast_1d(np.loadtxt(true_path, dtype=int))

    with open(params_path) as f:
        params = json.load(f)

    PYCLUST_CURE = _import_pyclust_cure()
    k = int(params["k"])
    c = int(params["c"])
    alpha = float(params["alpha"])

    cure_instance = PYCLUST_CURE(
        data.tolist(), k, number_represent_points=c, compression=alpha, ccore=True
    )
    cure_instance.process()
    clusters = cure_instance.get_clusters()

    labels = np.empty(len(data), dtype=int)
    for cid, idxs in enumerate(clusters):
        for idx in idxs:
            labels[idx] = cid

    if adjusted_rand_score is not None:
        ari = adjusted_rand_score(true_labels, labels)
    else:
        ari = None

    return data, true_labels, labels, ari, params, None


def get_2d_coords(data):
    """Return (X, Y) for plotting: first 2 dims or PCA."""
    if data.shape[1] == 2:
        return data[:, 0], data[:, 1]
    if data.shape[1] == 1:
        return data[:, 0], np.zeros_like(data[:, 0])
    if PCA is not None:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(data)
        return xy[:, 0], xy[:, 1]
    return data[:, 0], data[:, 1]


def scatter_ax(ax, x, y, labels, title, ari=None):
    """Single 2D scatter subplot with cluster colors."""
    uniq = np.unique(labels)
    n_clusters = len(uniq)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        ax.scatter(x[mask], y[mask], c=[colors[i % len(colors)]], label=f'C{lab}', s=18, alpha=0.8, edgecolors='none')
    ax.set_title(title + (f"  (ARI={ari:.3f})" if ari is not None else ""), fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='upper right', fontsize=6)


def scatter_ax_3d(ax, x, y, z, labels, title, ari=None):
    """Single 3D scatter subplot with cluster colors."""
    uniq = np.unique(labels)
    n_clusters = len(uniq)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    for i, lab in enumerate(uniq):
        mask = labels == lab
        ax.scatter(x[mask], y[mask], z[mask], c=[colors[i % len(colors)]], label=f'C{lab}', s=18, alpha=0.8, edgecolors='none')
    ax.set_title(title + (f"  (ARI={ari:.3f})" if ari is not None else ""), fontsize=10)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    ax.legend(loc='upper right', fontsize=6)


def plot_one_case(test_name, test_dir, out_dir, ext='png', dpi=120, show=False, use_pycure=False):
    """
    Create one figure:
      - Default: True labels | C++ Euclidean | C++ Pearson.
      - With use_pycure=True: True labels | pyclustering CURE (Euclidean).
    Uses 3D axes when data is 3D.
    """
    if use_pycure:
        data, true_labels, cure_labels, cure_ari, params, err = load_test_case_pycure(test_dir)
        if err:
            print(f"  Skip {test_name}: {err}")
            return False
    else:
        result = load_test_case(test_dir)
        data, true_labels, euc_labels, pearson_labels, aris, params, err = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
        )
        if err:
            print(f"  Skip {test_name}: {err}")
            return False
        euc_ari, pearson_ari = aris if aris else (None, None)
    is_3d = data.shape[1] == 3

    if is_3d:
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        if use_pycure:
            fig = plt.figure(figsize=(10, 4.5))
            fig.suptitle(
                f"{test_name}  (k={params['k']}, c={params['c']}, α={params['alpha']})  [3D, pyclustering]",
                fontsize=11,
            )
            ax0 = fig.add_subplot(1, 2, 1, projection='3d')
            ax1 = fig.add_subplot(1, 2, 2, projection='3d')
            scatter_ax_3d(ax0, x, y, z, true_labels, "True labels", ari=None)
            scatter_ax_3d(ax1, x, y, z, cure_labels, "pyclustering CURE", ari=cure_ari)
        else:
            fig = plt.figure(figsize=(14, 5))
            fig.suptitle(f"{test_name}  (k={params['k']}, c={params['c']}, α={params['alpha']})  [3D]", fontsize=11)
            ax0 = fig.add_subplot(1, 3, 1, projection='3d')
            ax1 = fig.add_subplot(1, 3, 2, projection='3d')
            ax2 = fig.add_subplot(1, 3, 3, projection='3d')
            scatter_ax_3d(ax0, x, y, z, true_labels, "True labels", ari=None)
            scatter_ax_3d(ax1, x, y, z, euc_labels, "C++ Euclidean", ari=euc_ari)
            scatter_ax_3d(ax2, x, y, z, pearson_labels, "C++ Pearson", ari=pearson_ari)
    else:
        x, y = get_2d_coords(data)
        if use_pycure:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
            fig.suptitle(
                f"{test_name}  (k={params['k']}, c={params['c']}, α={params['alpha']})  [pyclustering]",
                fontsize=11,
            )
            scatter_ax(axes[0], x, y, true_labels, "True labels", ari=None)
            scatter_ax(axes[1], x, y, cure_labels, "pyclustering CURE", ari=cure_ari)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
            fig.suptitle(f"{test_name}  (k={params['k']}, c={params['c']}, α={params['alpha']})", fontsize=11)
            scatter_ax(axes[0], x, y, true_labels, "True labels", ari=None)
            scatter_ax(axes[1], x, y, euc_labels, "C++ Euclidean", ari=euc_ari)
            scatter_ax(axes[2], x, y, pearson_labels, "C++ Pearson", ari=pearson_ari)

    if (not use_pycure) and (euc_ari is not None and pearson_ari is not None):
        diff = pearson_ari - euc_ari
        if abs(diff) < 1e-5:
            fig.text(0.5, 0.02, "Euclidean and Pearson same ARI.", ha='center', fontsize=9, style='italic')
        elif diff > 0:
            fig.text(0.5, 0.02, f"Pearson better (ARI diff = +{diff:.4f}).", ha='center', fontsize=9, style='italic')
        else:
            fig.text(0.5, 0.02, f"Euclidean better (ARI diff = {diff:.4f}).", ha='center', fontsize=9, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if out_dir:
        out_path = os.path.join(out_dir, f"{test_name}.{ext}")
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"  Saved: {out_path}")

    if show:
        # Only show if backend is interactive (has display); else skip to avoid warning
        backend = matplotlib.get_backend().lower()
        if backend not in ('agg', 'svg', 'pdf', 'ps', 'cairo'):
            plt.show()
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Plot CURE vs true labels: C++ Euclidean/Pearson or pyclustering CURE."
    )
    ap.add_argument("--test_data", default=None, help="Test data directory (default: tests/cpp_vs_true/test_data)")
    ap.add_argument("--out", default=None, help="Output directory for plots (default: tests/cpp_vs_true/plots)")
    ap.add_argument("--tests", default=None, help="Comma-separated test names (default: all)")
    ap.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Image format")
    ap.add_argument("--dpi", type=int, default=120, help="DPI for raster formats")
    ap.add_argument("--show", action="store_true", help="Display plots (interactive)")
    ap.add_argument(
        "--pycure",
        action="store_true",
        help="Use pyclustering.cure instead of C++ CURE results (plots True | pyclustering CURE).",
    )
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(script_dir))
    test_data_dir = args.test_data or os.path.join(script_dir, "test_data")
    out_dir = args.out or os.path.join(script_dir, "plots")
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
        if plot_one_case(
            name,
            os.path.join(test_data_dir, name),
            out_dir,
            ext=args.format,
            dpi=args.dpi,
            show=args.show,
            use_pycure=args.pycure,
        ):
            count += 1
    print("=" * 60)
    print(f"Done. {count}/{len(names)} plots saved to {out_dir}")


if __name__ == "__main__":
    main()
