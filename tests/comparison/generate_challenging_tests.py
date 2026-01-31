#!/usr/bin/env python3
"""
Generate more challenging test cases for thorough comparison.
"""

import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python_code'))

from euclidean_cure import CURE as EuclideanCURE
from pearson_cure import CURE as PearsonCURE
from sklearn.metrics import adjusted_rand_score


def save_test_case(output_dir, name, data, true_labels, params, result, metric='euclidean'):
    """Save a test case."""
    case_dir = os.path.join(output_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    
    np.savetxt(os.path.join(case_dir, 'data.csv'), data, delimiter=',', fmt='%.10f')
    np.savetxt(os.path.join(case_dir, 'true_labels.csv'), true_labels, delimiter=',', fmt='%d')
    
    params_save = dict(params)
    if 'metric' not in params_save:
        params_save['metric'] = metric  # 'euclidean' or 'pearson'
    with open(os.path.join(case_dir, 'params.json'), 'w') as f:
        json.dump(params_save, f, indent=2)
    
    results = {
        'true_labels': true_labels.tolist(),
        'params': params,
        metric: {
            'labels': result['labels'],
            'ari_vs_true': result['ari'],
            'n_clusters': result['n_clusters']
        }
    }
    
    with open(os.path.join(case_dir, 'python_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved: {name} (ARI={result['ari']:.4f})")


def run_cure(data, k, c, alpha, metric='euclidean'):
    """Run CURE and return results."""
    if metric == 'pearson':
        cure = PearsonCURE(k=k, c=c, alpha=alpha, standardize=False)
    else:
        cure = EuclideanCURE(k=k, c=c, alpha=alpha)
    
    labels = cure.fit_predict(data)
    return {
        'labels': labels.tolist(),
        'n_clusters': len(cure.clusters_),
        'ari': 0.0  # Will be computed separately
    }


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating challenging comparison tests...")
    print("=" * 60)
    
    # Test 9: Overlapping but visually separable clusters in 2D
    # Centers spaced so three blobs are visible in 2D; true labels = which blob.
    print("\n[Test 9] Overlapping clusters")
    np.random.seed(1234)
    data = []
    true_labels = []
    centers = [(0, 0), (6, 0), (3, 5)]  # Spread so 2D blobs are distinguishable
    std = 1.3  # Overlap at boundaries but not one big blob
    for c, (cx, cy) in enumerate(centers):
        for _ in range(50):
            x = cx + np.random.randn() * std
            y = cy + np.random.randn() * std
            data.append([x, y])
            true_labels.append(c)
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 3, 'c': 5, 'alpha': 0.3}
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test9_overlapping', data, true_labels, params, result)
    
    # Test 10: Non-spherical clusters (elongated)
    print("\n[Test 10] Elongated clusters")
    np.random.seed(2345)
    data = []
    true_labels = []
    # Cluster 1: horizontal line
    for i in range(50):
        data.append([i * 0.5, np.random.randn() * 0.3])
        true_labels.append(0)
    # Cluster 2: vertical line
    for i in range(50):
        data.append([30 + np.random.randn() * 0.3, i * 0.5])
        true_labels.append(1)
    # Cluster 3: diagonal line
    for i in range(50):
        data.append([i * 0.4 + 10, i * 0.4 + np.random.randn() * 0.3])
        true_labels.append(2)
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 3, 'c': 8, 'alpha': 0.2}  # More reps, less shrink for non-spherical
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test10_elongated', data, true_labels, params, result)
    
    # Test 11: Varying cluster sizes
    print("\n[Test 11] Varying cluster sizes")
    np.random.seed(3456)
    data = []
    true_labels = []
    sizes = [20, 50, 100, 30]
    centers = [(0, 0), (15, 0), (7.5, 13), (22, 10)]
    for c, (size, (cx, cy)) in enumerate(zip(sizes, centers)):
        for _ in range(size):
            data.append([cx + np.random.randn() * 1.5, cy + np.random.randn() * 1.5])
            true_labels.append(c)
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 4, 'c': 5, 'alpha': 0.3}
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test11_varying_sizes', data, true_labels, params, result)
    
    # Test 12: Noisy data with outliers
    print("\n[Test 12] Noisy data with outliers")
    np.random.seed(4567)
    data = []
    true_labels = []
    for c in range(3):
        cx, cy = c * 12, 0
        for _ in range(40):
            data.append([cx + np.random.randn() * 1.5, cy + np.random.randn() * 1.5])
            true_labels.append(c)
    # Add 15 outliers
    for _ in range(15):
        data.append([np.random.uniform(-20, 50), np.random.uniform(-20, 20)])
        true_labels.append(3)  # Outlier label
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 4, 'c': 5, 'alpha': 0.3}
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test12_noisy_outliers', data, true_labels, params, result)
    
    # Test 13: Many clusters
    print("\n[Test 13] Many clusters (k=8)")
    np.random.seed(5678)
    data = []
    true_labels = []
    n_clusters = 8
    for c in range(n_clusters):
        cx = (c % 4) * 8
        cy = (c // 4) * 8
        for _ in range(25):
            data.append([cx + np.random.randn() * 0.8, cy + np.random.randn() * 0.8])
            true_labels.append(c)
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 8, 'c': 4, 'alpha': 0.3}
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test13_many_clusters', data, true_labels, params, result)
    
    # Test 14: Concentric circles (challenging for distance-based)
    print("\n[Test 14] Concentric patterns")
    np.random.seed(6789)
    data = []
    true_labels = []
    # Inner circle
    for _ in range(50):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 2 + np.random.randn() * 0.3
        data.append([r * np.cos(angle), r * np.sin(angle)])
        true_labels.append(0)
    # Outer circle
    for _ in range(80):
        angle = np.random.uniform(0, 2 * np.pi)
        r = 5 + np.random.randn() * 0.3
        data.append([r * np.cos(angle), r * np.sin(angle)])
        true_labels.append(1)
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 2, 'c': 10, 'alpha': 0.1}  # Low alpha for non-spherical
    result = run_cure(data, **params)
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test14_concentric', data, true_labels, params, result)
    
    # Test 15: High-dimensional Pearson with correlated features
    print("\n[Test 15] High-dim Pearson with correlations")
    np.random.seed(7890)
    n_features = 20
    data = []
    true_labels = []
    
    # Generate 4 distinct patterns
    patterns = []
    for c in range(4):
        base = np.zeros(n_features)
        for i in range(n_features):
            if c == 0:
                base[i] = np.sin(2 * np.pi * i / n_features)
            elif c == 1:
                base[i] = np.cos(2 * np.pi * i / n_features)
            elif c == 2:
                base[i] = np.sin(4 * np.pi * i / n_features)
            else:
                base[i] = float(i) / n_features - 0.5
        patterns.append(base)
    
    for c in range(4):
        for _ in range(35):
            point = patterns[c] + np.random.randn(n_features) * 0.15
            data.append(point)
            true_labels.append(c)
    
    data = np.array(data)
    true_labels = np.array(true_labels)
    
    params = {'k': 4, 'c': 5, 'alpha': 0.3, 'metric': 'pearson'}
    result = run_cure(data, **{k: params[k] for k in ('k', 'c', 'alpha')}, metric='pearson')
    result['ari'] = adjusted_rand_score(true_labels, result['labels'])
    save_test_case(output_dir, 'test15_pearson_corr', data, true_labels, params, result, metric='pearson')
    
    print("\n" + "=" * 60)
    print("Challenging tests generated!")
    print("Run: ./build/tests/test_comparison tests/comparison/test_data")


if __name__ == '__main__':
    main()
