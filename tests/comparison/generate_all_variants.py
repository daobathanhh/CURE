#!/usr/bin/env python3
"""
Generate the same test cases for all 4 CURE types and save under separate dirs.

Variants:
  - euclidean          (base CURE, Euclidean)
  - pearson            (base CURE, Pearson)
  - scalable_euclidean (Scalable CURE, Euclidean)
  - scalable_pearson   (Scalable CURE, Pearson)

Storage: test_data/<variant>/<test_name>/  (data.csv, true_labels.csv, params.json, python_results.json)
Plots:  plots/<variant>/<test_name>.png    (when running plot_comparison with --out plots/<variant>)
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python_code'))

from euclidean_cure import CURE as EuclideanCURE
from euclidean_cure import Scalable_CURE as ScalableEuclideanCURE
from pearson_cure import CURE as PearsonCURE
from pearson_cure import Scalable_CURE as ScalablePearsonCURE
from sklearn.metrics import adjusted_rand_score

VARIANTS = ['euclidean', 'pearson', 'scalable_euclidean', 'scalable_pearson']


def generate_clustered_2d(n_per_cluster, n_clusters, cluster_std=1.0, seed=42):
    np.random.seed(seed)
    data, true_labels = [], []
    for c in range(n_clusters):
        cx, cy = (c % 3) * 10.0, (c // 3) * 10.0
        for _ in range(n_per_cluster):
            data.append([cx + np.random.randn() * cluster_std, cy + np.random.randn() * cluster_std])
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def generate_pattern_data(n_per_cluster, n_clusters, n_features=10, noise=0.2, seed=42):
    np.random.seed(seed)
    patterns = []
    for c in range(n_clusters):
        p = np.zeros(n_features)
        for i in range(n_features):
            if c == 0: p[i] = np.sin(2.0 * np.pi * i / n_features)
            elif c == 1: p[i] = np.cos(2.0 * np.pi * i / n_features)
            else: p[i] = float(i) / n_features
        patterns.append(p)
    data, true_labels = [], []
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            data.append((patterns[c] + np.random.randn(n_features) * noise).tolist())
            true_labels.append(c)
    return np.array(data), np.array(true_labels)


def run_variant(data, true_labels, variant, k, c, alpha):
    """Run one of the 4 CURE variants; return dict with labels, ari_vs_true, n_clusters, cluster_sizes."""
    if variant == 'euclidean':
        model = EuclideanCURE(k=k, c=c, alpha=alpha)
    elif variant == 'pearson':
        model = PearsonCURE(k=k, c=c, alpha=alpha, standardize=False)
    elif variant == 'scalable_euclidean':
        model = ScalableEuclideanCURE(k=k, c=c, alpha=alpha, sample_size=0.2, n_partitions=5, reduce_factor=3, random_state=42)
    elif variant == 'scalable_pearson':
        model = ScalablePearsonCURE(k=k, c=c, alpha=alpha, standardize=False, sample_size=0.2, n_partitions=5, reduce_factor=3, random_state=42)
    else:
        raise ValueError('unknown variant: ' + variant)
    labels = model.fit_predict(data)
    ari = adjusted_rand_score(true_labels, labels)
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = {int(u): int(v) for u, v in zip(unique, counts)}
    return {
        'labels': labels.tolist(),
        'ari_vs_true': float(ari),
        'n_clusters': len(unique),
        'cluster_sizes': cluster_sizes,
    }


def save_case(base_dir, variant, name, data, true_labels, params, result):
    out_dir = os.path.join(base_dir, variant, name)
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, 'data.csv'), data, delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(out_dir, 'true_labels.csv'), true_labels, delimiter=',', fmt='%d')
    with open(os.path.join(out_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    python_results = {
        'true_labels': true_labels.tolist(),
        'params': params,
        'result': result,
    }
    with open(os.path.join(out_dir, 'python_results.json'), 'w') as f:
        json.dump(python_results, f, indent=2)


def main():
    base_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    metric_map = {'euclidean': 'euclidean', 'pearson': 'pearson',
                  'scalable_euclidean': 'euclidean', 'scalable_pearson': 'pearson'}
    scalable_map = {'euclidean': False, 'pearson': False,
                   'scalable_euclidean': True, 'scalable_pearson': True}

    # Define test cases: (name, data, true_labels, params)
    cases = []

    # 1. Small 2D
    data, true_labels = generate_clustered_2d(30, 3, cluster_std=0.5, seed=42)
    cases.append(('test1_small_2d', data, true_labels, {'k': 3, 'c': 3, 'alpha': 0.3}))

    # 2. Medium 2D
    data, true_labels = generate_clustered_2d(50, 4, cluster_std=1.0, seed=123)
    cases.append(('test2_medium_2d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    # 3. Alpha 0.5
    data, true_labels = generate_clustered_2d(40, 3, cluster_std=0.8, seed=456)
    cases.append(('test3_alpha05', data, true_labels, {'k': 3, 'c': 4, 'alpha': 0.5}))

    # 4. Pearson-style pattern 10D
    data, true_labels = generate_pattern_data(50, 3, n_features=10, noise=0.2, seed=789)
    cases.append(('test4_pattern_10d', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # 5. Larger 2D
    data, true_labels = generate_clustered_2d(100, 5, cluster_std=1.5, seed=999)
    cases.append(('test5_large_2d', data, true_labels, {'k': 5, 'c': 5, 'alpha': 0.3}))

    # 6. Overlapping (spread centers)
    np.random.seed(1234)
    data, true_labels = [], []
    for c, (cx, cy) in enumerate([(0, 0), (6, 0), (3, 5)]):
        for _ in range(50):
            data.append([cx + np.random.randn() * 1.3, cy + np.random.randn() * 1.3])
            true_labels.append(c)
    data, true_labels = np.array(data), np.array(true_labels)
    cases.append(('test6_overlapping', data, true_labels, {'k': 3, 'c': 5, 'alpha': 0.3}))

    # 7. Pattern 15D, 4 clusters
    data, true_labels = generate_pattern_data(40, 4, n_features=15, noise=0.15, seed=888)
    cases.append(('test7_pattern_15d', data, true_labels, {'k': 4, 'c': 5, 'alpha': 0.3}))

    print('Generating 4 variants for each test case...')
    print('Variants:', VARIANTS)
    print('=' * 60)
    for name, data, true_labels, params in cases:
        print(f'\n[{name}]')
        for variant in VARIANTS:
            p = dict(params)
            p['metric'] = metric_map[variant]
            p['scalable'] = scalable_map[variant]
            result = run_variant(data, true_labels, variant, **{k: p[k] for k in ('k', 'c', 'alpha')})
            save_case(base_dir, variant, name, data, true_labels, p, result)
            print(f'  {variant}: ARI={result["ari_vs_true"]:.4f}')
    print('=' * 60)
    print(f'Saved to {base_dir}/<variant>/<name>/')
    print('Run C++: ./build/tests/test_comparison tests/comparison/test_data/euclidean  (and pearson, scalable_euclidean, scalable_pearson)')
    print('Plot:    python3 plot_comparison.py --test_data test_data/euclidean --out plots/euclidean  (and for each variant)')


if __name__ == '__main__':
    main()
