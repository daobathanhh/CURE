#!/usr/bin/env python3
"""
Generate test datasets and compute Python CURE results for comparison with C++.

This script:
1. Generates various test datasets
2. Runs Python CURE implementations (Euclidean and Pearson)
3. Saves data and results to files for C++ comparison
"""

import numpy as np
import json
import os
import sys

# Add parent directory to path to import CURE modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python_code'))

from euclidean_cure import CURE as EuclideanCURE
from pearson_cure import CURE as PearsonCURE


def generate_clustered_2d(n_per_cluster: int, n_clusters: int, 
                          cluster_std: float = 1.0, seed: int = 42) -> tuple:
    """Generate 2D clustered data."""
    np.random.seed(seed)
    
    data = []
    true_labels = []
    
    # Generate cluster centers spread out
    centers = []
    for c in range(n_clusters):
        cx = (c % 3) * 10.0
        cy = (c // 3) * 10.0
        centers.append((cx, cy))
    
    # Generate points around each center
    for c in range(n_clusters):
        cx, cy = centers[c]
        for _ in range(n_per_cluster):
            x = cx + np.random.randn() * cluster_std
            y = cy + np.random.randn() * cluster_std
            data.append([x, y])
            true_labels.append(c)
    
    return np.array(data), np.array(true_labels)


def generate_pattern_data(n_per_cluster: int, n_clusters: int,
                          n_features: int = 10, noise: float = 0.2, 
                          seed: int = 42) -> tuple:
    """Generate high-dimensional pattern data for Pearson distance testing."""
    np.random.seed(seed)
    
    data = []
    true_labels = []
    
    # Generate base patterns
    patterns = []
    for c in range(n_clusters):
        pattern = np.zeros(n_features)
        for i in range(n_features):
            if c == 0:
                pattern[i] = np.sin(2.0 * np.pi * i / n_features)
            elif c == 1:
                pattern[i] = np.cos(2.0 * np.pi * i / n_features)
            else:
                pattern[i] = float(i) / n_features
        patterns.append(pattern)
    
    # Generate points with noise
    for c in range(n_clusters):
        for _ in range(n_per_cluster):
            point = patterns[c] + np.random.randn(n_features) * noise
            data.append(point.tolist())
            true_labels.append(c)
    
    return np.array(data), np.array(true_labels)


def compute_ari(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Compute Adjusted Rand Index between two labelings."""
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(labels1, labels2)


def compute_cluster_sizes(labels: np.ndarray) -> dict:
    """Compute cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


def run_euclidean_cure(data: np.ndarray, k: int, c: int, alpha: float) -> dict:
    """Run Euclidean CURE and return results."""
    cure = EuclideanCURE(k=k, c=c, alpha=alpha)
    labels = cure.fit_predict(data)
    
    # Get cluster representatives
    reps = []
    for cluster in cure.clusters_:
        cluster_reps = [r.tolist() for r in cluster.reps]
        reps.append(cluster_reps)
    
    return {
        'labels': labels.tolist(),
        'n_clusters': len(cure.clusters_),
        'cluster_sizes': compute_cluster_sizes(labels),
        'representatives': reps
    }


def run_pearson_cure(data: np.ndarray, k: int, c: int, alpha: float) -> dict:
    """Run Pearson CURE and return results."""
    cure = PearsonCURE(k=k, c=c, alpha=alpha, standardize=False)
    labels = cure.fit_predict(data)
    
    # Get cluster representatives
    reps = []
    for cluster in cure.clusters_:
        cluster_reps = [r.tolist() for r in cluster.reps]
        reps.append(cluster_reps)
    
    return {
        'labels': labels.tolist(),
        'n_clusters': len(cure.clusters_),
        'cluster_sizes': compute_cluster_sizes(labels),
        'representatives': reps
    }


def save_test_case(output_dir: str, name: str, data: np.ndarray, 
                   true_labels: np.ndarray, params: dict, 
                   euclidean_result: dict = None, pearson_result: dict = None):
    """Save a test case to files."""
    case_dir = os.path.join(output_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    
    # Save data as CSV (easy to read in C++)
    np.savetxt(os.path.join(case_dir, 'data.csv'), data, delimiter=',', fmt='%.10f')
    
    # Save true labels
    np.savetxt(os.path.join(case_dir, 'true_labels.csv'), true_labels, delimiter=',', fmt='%d')
    
    # Ensure metric is in params for C++ (euclidean or pearson)
    params_save = dict(params)
    if 'metric' not in params_save:
        params_save['metric'] = 'pearson' if pearson_result else 'euclidean'
    with open(os.path.join(case_dir, 'params.json'), 'w') as f:
        json.dump(params_save, f, indent=2)
    
    # Save Python results
    results = {
        'true_labels': true_labels.tolist(),
        'params': params
    }
    
    if euclidean_result:
        results['euclidean'] = euclidean_result
        # Compute ARI with true labels
        results['euclidean']['ari_vs_true'] = compute_ari(
            true_labels, np.array(euclidean_result['labels'])
        )
    
    if pearson_result:
        results['pearson'] = pearson_result
        results['pearson']['ari_vs_true'] = compute_ari(
            true_labels, np.array(pearson_result['labels'])
        )
    
    with open(os.path.join(case_dir, 'python_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved test case: {name}")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'test_data')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating comparison test data...")
    print("=" * 60)
    
    # Test Case 1: Small 2D clustered data (Euclidean)
    print("\n[Test 1] Small 2D clustered data")
    data, true_labels = generate_clustered_2d(30, 3, cluster_std=0.5, seed=42)
    params = {'k': 3, 'c': 3, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test1_small_2d', data, true_labels, params, 
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    # Test Case 2: Medium 2D clustered data
    print("\n[Test 2] Medium 2D clustered data")
    data, true_labels = generate_clustered_2d(50, 4, cluster_std=1.0, seed=123)
    params = {'k': 4, 'c': 5, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test2_medium_2d', data, true_labels, params,
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    # Test Case 3: Different alpha values (shrinking factor 0.1 to 0.9 only)
    print("\n[Test 3] Alpha variation test")
    data, true_labels = generate_clustered_2d(40, 3, cluster_std=0.8, seed=456)
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        params = {'k': 3, 'c': 4, 'alpha': alpha}
        euclidean_result = run_euclidean_cure(data, **params)
        save_test_case(output_dir, f'test3_alpha_{alpha:.1f}', data, true_labels, params,
                       euclidean_result=euclidean_result)
        ari = compute_ari(true_labels, np.array(euclidean_result['labels']))
        print(f"    Alpha={alpha:.1f}: ARI={ari:.4f}")
    
    # Test Case 4: High-dimensional pattern data (Pearson)
    print("\n[Test 4] High-dimensional pattern data (Pearson)")
    data, true_labels = generate_pattern_data(50, 3, n_features=10, noise=0.2, seed=789)
    params = {'k': 3, 'c': 5, 'alpha': 0.3, 'metric': 'pearson'}
    pearson_result = run_pearson_cure(data, **{k: params[k] for k in ('k', 'c', 'alpha')})
    save_test_case(output_dir, 'test4_pearson_10d', data, true_labels, params,
                   pearson_result=pearson_result)
    print(f"    Pearson ARI vs true: {pearson_result['ari_vs_true'] if 'ari_vs_true' in pearson_result else 'N/A':.4f}")
    
    # Test Case 4b: Pearson, 4 clusters, 15D
    print("\n[Test 4b] Pearson 4 clusters 15D")
    data, true_labels = generate_pattern_data(40, 4, n_features=15, noise=0.15, seed=888)
    params = {'k': 4, 'c': 5, 'alpha': 0.3, 'metric': 'pearson'}
    pearson_result = run_pearson_cure(data, **{k: params[k] for k in ('k', 'c', 'alpha')})
    save_test_case(output_dir, 'test4b_pearson_4clusters_15d', data, true_labels, params,
                   pearson_result=pearson_result)
    print(f"    Pearson ARI vs true: {pearson_result['ari_vs_true'] if 'ari_vs_true' in pearson_result else 'N/A':.4f}")
    
    # Test Case 4c: Pearson, alpha variation
    print("\n[Test 4c] Pearson alpha variation")
    data, true_labels = generate_pattern_data(45, 3, n_features=12, noise=0.2, seed=777)
    params = {'k': 3, 'c': 5, 'alpha': 0.5, 'metric': 'pearson'}
    pearson_result = run_pearson_cure(data, **{k: params[k] for k in ('k', 'c', 'alpha')})
    save_test_case(output_dir, 'test4c_pearson_alpha05', data, true_labels, params,
                   pearson_result=pearson_result)
    print(f"    Pearson ARI vs true: {pearson_result['ari_vs_true'] if 'ari_vs_true' in pearson_result else 'N/A':.4f}")
    
    # Test Case 5: Larger dataset
    print("\n[Test 5] Larger 2D dataset")
    data, true_labels = generate_clustered_2d(100, 5, cluster_std=1.5, seed=999)
    params = {'k': 5, 'c': 5, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test5_large_2d', data, true_labels, params,
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    # Test Case 6: Single representative (c=1)
    print("\n[Test 6] Single representative (c=1)")
    data, true_labels = generate_clustered_2d(30, 3, cluster_std=0.5, seed=42)
    params = {'k': 3, 'c': 1, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test6_single_rep', data, true_labels, params,
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    # Test Case 7: Many representatives
    print("\n[Test 7] Many representatives (c=10)")
    data, true_labels = generate_clustered_2d(50, 3, cluster_std=0.5, seed=42)
    params = {'k': 3, 'c': 10, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test7_many_reps', data, true_labels, params,
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    # Test Case 8: High-dimensional Euclidean
    print("\n[Test 8] High-dimensional Euclidean (20D)")
    np.random.seed(111)
    n_features = 20
    data = []
    true_labels = []
    for c in range(3):
        center = np.random.randn(n_features) * 5
        for _ in range(40):
            point = center + np.random.randn(n_features) * 0.5
            data.append(point)
            true_labels.append(c)
    data = np.array(data)
    true_labels = np.array(true_labels)
    params = {'k': 3, 'c': 5, 'alpha': 0.3}
    euclidean_result = run_euclidean_cure(data, **params)
    save_test_case(output_dir, 'test8_euclidean_20d', data, true_labels, params,
                   euclidean_result=euclidean_result)
    print(f"    Euclidean ARI vs true: {euclidean_result['ari_vs_true'] if 'ari_vs_true' in euclidean_result else 'N/A':.4f}")
    
    print("\n" + "=" * 60)
    print(f"Test data saved to: {output_dir}")
    print("Run C++ comparison tests with: ./build/tests/test_comparison")


if __name__ == '__main__':
    main()
