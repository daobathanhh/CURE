"""
CURE Algorithm 3D Test and Visualization

This script tests both Euclidean and Pearson CURE implementations
on 3D data and provides comprehensive 3D visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
import time

# Import CURE implementations
from euclidean_cure import CURE as EuclideanCURE
from euclidean_cure import Scalable_CURE as EuclideanScalableCURE
from pearson_cure import CURE as PearsonCURE
from pearson_cure import Scalable_CURE as PearsonScalableCURE


def generate_3d_test_data(n_samples=200, n_clusters=4, random_state=42):
    """
    Generate 3D test data with clear cluster structure.
    
    Args:
        n_samples: Total number of samples
        n_clusters: Number of clusters to generate
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (data array, true labels)
    """
    np.random.seed(random_state)
    
    samples_per_cluster = n_samples // n_clusters
    
    # Define cluster centers in 3D space
    centers = [
        [20, 20, 20],   # Bottom-front-left
        [80, 20, 80],   # Bottom-front-right-high
        [20, 80, 50],   # Top-front-left-mid
        [80, 80, 20],   # Top-back-right-low
        [50, 50, 90],   # Center-high
        [50, 50, 10],   # Center-low
    ][:n_clusters]
    
    # Generate clusters
    data = []
    labels = []
    
    for i, center in enumerate(centers):
        # Add some variance to cluster spread
        spread = np.random.uniform(8, 15)
        cluster_points = np.random.randn(samples_per_cluster, 3) * spread + center
        data.append(cluster_points)
        labels.extend([i] * samples_per_cluster)
    
    # Add remaining samples to first cluster if needed
    remaining = n_samples - len(labels)
    if remaining > 0:
        extra_points = np.random.randn(remaining, 3) * 10 + centers[0]
        data.append(extra_points)
        labels.extend([0] * remaining)
    
    return np.vstack(data), np.array(labels)


def generate_3d_complex_data(n_samples=300, random_state=42):
    """
    Generate more complex 3D test data with non-spherical clusters.
    
    Args:
        n_samples: Total number of samples
        random_state: Random seed
        
    Returns:
        Tuple of (data array, true labels)
    """
    np.random.seed(random_state)
    
    n_per_cluster = n_samples // 4
    
    # Cluster 1: Elongated along Z-axis
    theta1 = np.random.uniform(0, 2*np.pi, n_per_cluster)
    z1 = np.random.uniform(0, 50, n_per_cluster)
    r1 = np.random.uniform(5, 15, n_per_cluster)
    cluster1 = np.column_stack([
        20 + r1 * np.cos(theta1),
        20 + r1 * np.sin(theta1),
        z1
    ])
    
    # Cluster 2: Spherical cluster
    phi2 = np.random.uniform(0, np.pi, n_per_cluster)
    theta2 = np.random.uniform(0, 2*np.pi, n_per_cluster)
    r2 = np.random.uniform(0, 15, n_per_cluster)
    cluster2 = np.column_stack([
        80 + r2 * np.sin(phi2) * np.cos(theta2),
        30 + r2 * np.sin(phi2) * np.sin(theta2),
        70 + r2 * np.cos(phi2)
    ])
    
    # Cluster 3: Flat disk in XY plane
    theta3 = np.random.uniform(0, 2*np.pi, n_per_cluster)
    r3 = np.random.uniform(0, 20, n_per_cluster)
    cluster3 = np.column_stack([
        50 + r3 * np.cos(theta3),
        80 + r3 * np.sin(theta3),
        25 + np.random.uniform(-3, 3, n_per_cluster)
    ])
    
    # Cluster 4: Spiral/helix shape
    t4 = np.linspace(0, 4*np.pi, n_per_cluster)
    noise4 = np.random.randn(n_per_cluster, 3) * 3
    cluster4 = np.column_stack([
        30 + 15 * np.cos(t4),
        60 + 15 * np.sin(t4),
        10 + t4 * 3
    ]) + noise4
    
    data = np.vstack([cluster1, cluster2, cluster3, cluster4])
    labels = np.array([0]*n_per_cluster + [1]*n_per_cluster + 
                      [2]*n_per_cluster + [3]*n_per_cluster)
    
    # Handle remaining samples
    remaining = n_samples - len(labels)
    if remaining > 0:
        extra = np.random.randn(remaining, 3) * 10 + [50, 50, 50]
        data = np.vstack([data, extra])
        labels = np.append(labels, [0] * remaining)
    
    return data, labels


# Pre-generated 3D test dataset (similar to S in 2D)
def generate_S_3d(n_samples=200, random_state=42):
    """Generate a 3D version similar to the 2D S array."""
    np.random.seed(random_state)
    return np.random.uniform(0, 100, size=(n_samples, 3))


# Create the 3D S array
S_3d = generate_S_3d(200, random_state=42)


def plot_clusters_3d(X, labels, representatives, title, ax, show_representatives=True, 
                     elev=20, azim=45):
    """
    Plot clustered 3D data with optional representatives.
    
    Args:
        X: Data points (n_samples, 3)
        labels: Cluster labels for each point
        representatives: List of representative arrays for each cluster
        title: Plot title
        ax: Matplotlib 3D axis
        show_representatives: Whether to show representative points
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                  c=[colors[i % len(colors)]], 
                  label=f'Cluster {label}',
                  alpha=0.6, s=50, edgecolors='white', linewidth=0.3)
    
    # Plot representatives
    if show_representatives and representatives:
        for i, reps in enumerate(representatives):
            if reps is not None and len(reps) > 0:
                ax.scatter(reps[:, 0], reps[:, 1], reps[:, 2],
                          c=[colors[i % len(colors)]], 
                          marker='*', s=300, edgecolors='black', linewidth=1.5)
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)


def visualize_comparison_3d(X, results_dict, suptitle="CURE Algorithm Comparison - 3D"):
    """
    Create a comparison visualization of multiple CURE results in 3D.
    
    Args:
        X: Original data
        results_dict: Dictionary with algorithm names as keys and (labels, representatives) tuples
        suptitle: Super title for the figure
    """
    n_algorithms = len(results_dict)
    fig = plt.figure(figsize=(6 * n_algorithms, 5))
    
    for idx, (name, (labels, representatives)) in enumerate(results_dict.items()):
        ax = fig.add_subplot(1, n_algorithms, idx + 1, projection='3d')
        plot_clusters_3d(X, labels, representatives, name, ax)
    
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def print_cluster_statistics(labels, algorithm_name):
    """Print statistics about clustering results."""
    unique_labels = np.unique(labels)
    print(f"\n{algorithm_name}:")
    print(f"  Number of clusters: {len(unique_labels)}")
    for label in unique_labels:
        count = np.sum(labels == label)
        print(f"  Cluster {label}: {count} points ({100*count/len(labels):.1f}%)")


def run_tests_3d(data, data_name="3D Data", n_clusters=4):
    """
    Run comprehensive tests on 3D dataset.
    
    Args:
        data: 3D data array
        data_name: Name of the dataset for display
        n_clusters: Number of clusters to form
    """
    print("=" * 70)
    print(f"CURE Algorithm 3D Test Suite - {data_name}")
    print("=" * 70)
    print(f"\nDataset shape: {data.shape}")
    print(f"Target clusters: {n_clusters}")
    
    results = {}
    
    # Test 1: Euclidean CURE (base)
    print("\n" + "-" * 50)
    print("Test 1: Euclidean CURE (Base)")
    print("-" * 50)
    
    start_time = time.time()
    euclidean_cure = EuclideanCURE(
        n_clusters=n_clusters,
        n_representatives=5,
        shrink_factor=0.3,
        random_state=42
    )
    euclidean_labels = euclidean_cure.fit_predict(data)
    euclidean_time = time.time() - start_time
    
    print(f"  Time: {euclidean_time:.4f} seconds")
    print_cluster_statistics(euclidean_labels, "Euclidean CURE")
    results["Euclidean CURE"] = (euclidean_labels, euclidean_cure.representatives_)
    
    # Test 2: Euclidean Scalable CURE
    print("\n" + "-" * 50)
    print("Test 2: Euclidean Scalable CURE")
    print("-" * 50)
    
    start_time = time.time()
    euclidean_scalable = EuclideanScalableCURE(
        n_clusters=n_clusters,
        n_representatives=5,
        shrink_factor=0.3,
        sample_size=0.5,
        n_partitions=3,
        exclude_outliers_from_sample=True,
        random_state=42
    )
    euclidean_scalable_labels = euclidean_scalable.fit_predict(data)
    euclidean_scalable_time = time.time() - start_time
    
    print(f"  Time: {euclidean_scalable_time:.4f} seconds")
    print(f"  Sample size used: {len(euclidean_scalable.sample_indices_)}")
    if euclidean_scalable.outlier_indices_ is not None:
        print(f"  Outliers detected: {len(euclidean_scalable.outlier_indices_)}")
    print_cluster_statistics(euclidean_scalable_labels, "Euclidean Scalable CURE")
    results["Euclidean Scalable CURE"] = (euclidean_scalable_labels, euclidean_scalable.representatives_)
    
    # Test 3: Pearson CURE (base)
    print("\n" + "-" * 50)
    print("Test 3: Pearson CURE (Base)")
    print("-" * 50)
    print("  Note: Pearson correlation measures linear relationship between dimensions.")
    
    start_time = time.time()
    pearson_cure = PearsonCURE(
        n_clusters=n_clusters,
        n_representatives=5,
        shrink_factor=0.3,
        random_state=42
    )
    pearson_labels = pearson_cure.fit_predict(data)
    pearson_time = time.time() - start_time
    
    print(f"  Time: {pearson_time:.4f} seconds")
    print_cluster_statistics(pearson_labels, "Pearson CURE")
    results["Pearson CURE"] = (pearson_labels, pearson_cure.representatives_)
    
    # Test 4: Pearson Scalable CURE
    print("\n" + "-" * 50)
    print("Test 4: Pearson Scalable CURE")
    print("-" * 50)
    
    start_time = time.time()
    pearson_scalable = PearsonScalableCURE(
        n_clusters=n_clusters,
        n_representatives=5,
        shrink_factor=0.3,
        sample_size=0.5,
        n_partitions=3,
        exclude_outliers_from_sample=True,
        random_state=42
    )
    pearson_scalable_labels = pearson_scalable.fit_predict(data)
    pearson_scalable_time = time.time() - start_time
    
    print(f"  Time: {pearson_scalable_time:.4f} seconds")
    print(f"  Sample size used: {len(pearson_scalable.sample_indices_)}")
    if pearson_scalable.outlier_indices_ is not None:
        print(f"  Outliers detected: {len(pearson_scalable.outlier_indices_)}")
    print_cluster_statistics(pearson_scalable_labels, "Pearson Scalable CURE")
    results["Pearson Scalable CURE"] = (pearson_scalable_labels, pearson_scalable.representatives_)
    
    return results


def visualize_all_results_3d(data, results, data_name="3D Data", save_path=None):
    """
    Create comprehensive 3D visualization of all results.
    
    Args:
        data: 3D data array
        results: Dictionary of results from run_tests_3d()
        data_name: Name of the dataset
        save_path: Optional path to save the figure
    """
    # Main comparison figure
    fig1 = visualize_comparison_3d(data, results, f"CURE Algorithm Comparison - {data_name}")
    
    # Detailed figure with different viewing angles
    fig2 = plt.figure(figsize=(16, 12))
    
    angles = [(20, 45), (20, 135), (60, 45), (0, 90)]  # Different viewing angles
    
    for row, (name, (labels, representatives)) in enumerate(results.items()):
        for col, (elev, azim) in enumerate(angles):
            ax = fig2.add_subplot(len(results), len(angles), row * len(angles) + col + 1, 
                                 projection='3d')
            plot_clusters_3d(data, labels, representatives, 
                           f"{name}\n(elev={elev}, azim={azim})" if col == 0 else f"View {col+1}",
                           ax, elev=elev, azim=azim)
    
    plt.suptitle(f"CURE Algorithm Comparison - Multiple Views ({data_name})", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(save_path.replace('.png', '_comparison.png'), dpi=150, bbox_inches='tight')
        fig2.savefig(save_path.replace('.png', '_detailed.png'), dpi=150, bbox_inches='tight')
        print(f"\nFigures saved to {save_path}")
    
    return fig1, fig2


def test_different_cluster_counts_3d(data, data_name="3D Data"):
    """Test with different numbers of clusters in 3D."""
    print("\n" + "=" * 70)
    print(f"Testing Different Cluster Counts - {data_name}")
    print("=" * 70)
    
    cluster_counts = [2, 3, 4, 5]
    
    fig = plt.figure(figsize=(5 * len(cluster_counts), 10))
    
    for col, k in enumerate(cluster_counts):
        # Euclidean CURE
        cure_euc = EuclideanCURE(n_clusters=k, n_representatives=5, shrink_factor=0.3, random_state=42)
        labels_euc = cure_euc.fit_predict(data)
        
        ax1 = fig.add_subplot(2, len(cluster_counts), col + 1, projection='3d')
        plot_clusters_3d(data, labels_euc, cure_euc.representatives_, 
                        f"Euclidean (k={k})", ax1)
        
        # Pearson CURE
        cure_prs = PearsonCURE(n_clusters=k, n_representatives=5, shrink_factor=0.3, random_state=42)
        labels_prs = cure_prs.fit_predict(data)
        
        ax2 = fig.add_subplot(2, len(cluster_counts), len(cluster_counts) + col + 1, projection='3d')
        plot_clusters_3d(data, labels_prs, cure_prs.representatives_, 
                        f"Pearson (k={k})", ax2)
    
    plt.suptitle(f"Effect of Cluster Count on CURE Results - {data_name}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def test_shrink_factors_3d(data, data_name="3D Data"):
    """Test effect of different shrink factors in 3D."""
    print("\n" + "=" * 70)
    print(f"Testing Different Shrink Factors - {data_name}")
    print("=" * 70)
    
    shrink_factors = [0.0, 0.2, 0.5, 0.8]
    n_clusters = 4
    
    fig = plt.figure(figsize=(5 * len(shrink_factors), 5))
    
    for col, alpha in enumerate(shrink_factors):
        cure = EuclideanCURE(n_clusters=n_clusters, n_representatives=5, 
                            shrink_factor=alpha, random_state=42)
        labels = cure.fit_predict(data)
        
        ax = fig.add_subplot(1, len(shrink_factors), col + 1, projection='3d')
        plot_clusters_3d(data, labels, cure.representatives_, 
                        f"α = {alpha}", ax)
    
    plt.suptitle(f"Effect of Shrink Factor (α) on Representative Points - {data_name}", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_interactive_3d_plot(data, labels, representatives, title="CURE Clustering"):
    """
    Create an interactive 3D plot that can be rotated.
    
    Args:
        data: 3D data array
        labels: Cluster labels
        representatives: List of representative arrays
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    plot_clusters_3d(data, labels, representatives, title, ax, show_representatives=True)
    
    # Add colorbar legend
    unique_labels = np.unique(labels)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=plt.cm.tab10(i / 10), markersize=10,
                                   label=f'Cluster {i}: {np.sum(labels == i)} pts')
                      for i in unique_labels]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return fig


def compare_with_ground_truth_3d(data, true_labels, predicted_labels, algorithm_name):
    """
    Compare predicted clustering with ground truth.
    
    Args:
        data: 3D data array
        true_labels: Ground truth labels
        predicted_labels: Predicted cluster labels
        algorithm_name: Name of the algorithm
    """
    fig = plt.figure(figsize=(12, 5))
    
    # Ground truth
    ax1 = fig.add_subplot(121, projection='3d')
    unique_true = np.unique(true_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_true))))
    for i, label in enumerate(unique_true):
        mask = true_labels == label
        ax1.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                   c=[colors[i]], alpha=0.6, s=50)
    ax1.set_title("Ground Truth", fontsize=12, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Predicted
    ax2 = fig.add_subplot(122, projection='3d')
    unique_pred = np.unique(predicted_labels)
    for i, label in enumerate(unique_pred):
        mask = predicted_labels == label
        ax2.scatter(data[mask, 0], data[mask, 1], data[mask, 2],
                   c=[colors[i % len(colors)]], alpha=0.6, s=50)
    ax2.set_title(f"{algorithm_name} Prediction", fontsize=12, fontweight='bold')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Calculate accuracy metrics
    try:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        plt.suptitle(f"Ground Truth vs {algorithm_name}\nARI: {ari:.4f}, NMI: {nmi:.4f}", 
                    fontsize=14, fontweight='bold')
    except ImportError:
        plt.suptitle(f"Ground Truth vs {algorithm_name}", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test 1: Random uniform 3D data (similar to S array)
    print("\n" + "=" * 70)
    print("Test Suite 1: Random Uniform 3D Data (S_3d)")
    print("=" * 70)
    
    results_uniform = run_tests_3d(S_3d, "Random Uniform 3D", n_clusters=4)
    
    # Test 2: Structured 3D data with clear clusters
    print("\n" + "=" * 70)
    print("Test Suite 2: Structured 3D Data with Clear Clusters")
    print("=" * 70)
    
    structured_data, true_labels = generate_3d_test_data(n_samples=200, n_clusters=4)
    results_structured = run_tests_3d(structured_data, "Structured 3D", n_clusters=4)
    
    # Test 3: Complex 3D data with non-spherical clusters
    print("\n" + "=" * 70)
    print("Test Suite 3: Complex 3D Data (Non-spherical Clusters)")
    print("=" * 70)
    
    complex_data, complex_labels = generate_3d_complex_data(n_samples=300)
    results_complex = run_tests_3d(complex_data, "Complex 3D", n_clusters=4)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating 3D Visualizations...")
    print("=" * 70)
    
    # Visualize all results
    fig1, fig2 = visualize_all_results_3d(S_3d, results_uniform, "Random Uniform 3D")
    fig3, fig4 = visualize_all_results_3d(structured_data, results_structured, "Structured 3D")
    fig5, fig6 = visualize_all_results_3d(complex_data, results_complex, "Complex 3D")
    
    # Test different cluster counts
    fig7 = test_different_cluster_counts_3d(structured_data, "Structured 3D")
    
    # Test shrink factors
    fig8 = test_shrink_factors_3d(structured_data, "Structured 3D")
    
    # Compare with ground truth for structured data
    euclidean_cure = EuclideanCURE(n_clusters=4, n_representatives=5, shrink_factor=0.3, random_state=42)
    euclidean_labels = euclidean_cure.fit_predict(structured_data)
    fig9 = compare_with_ground_truth_3d(structured_data, true_labels, euclidean_labels, "Euclidean CURE")
    
    # Show all figures
    plt.show()
    
    print("\n" + "=" * 70)
    print("All 3D tests completed!")
    print("=" * 70)
