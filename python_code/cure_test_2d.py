"""
CURE Algorithm 2D Test and Visualization

This script tests both Euclidean and Pearson CURE implementations
on 2D data and provides comprehensive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import time

# Import CURE implementations
from euclidean_cure import CURE as EuclideanCURE
from euclidean_cure import Scalable_CURE as EuclideanScalableCURE
from pearson_cure import CURE as PearsonCURE
from pearson_cure import Scalable_CURE as PearsonScalableCURE


# Test data from new_cure.py
S = np.array([[82.68456706,  4.45922012],
       [15.86935568, 40.86335396],
       [61.91947832, 83.62464105],
       [78.5321037 , 55.72687319],
       [35.22380287, 69.52947118],
       [14.31433367, 84.51472094],
       [45.56225679, 86.70398922],
       [38.17813064, 36.24561558],
       [25.12518629, 48.06771104],
       [38.79577979, 69.8433773 ],
       [53.89180384,  8.4574877 ],
       [57.96463358,  2.51607491],
       [19.87958796, 25.00048336],
       [29.60111534, 63.85470859],
       [96.26315396, 75.7944596 ],
       [45.6781928 , 48.01700787],
       [47.36043981, 65.33336647],
       [14.79387894, 53.93647558],
       [ 0.5349268 , 73.8408741 ],
       [ 1.9565076 , 38.83379104],
       [ 7.22259619,  7.46329912],
       [83.70908182, 87.19282854],
       [99.46010373, 88.06434118],
       [10.58629585, 58.17182678],
       [91.16103317, 81.97991079],
       [52.33079582, 36.95667037],
       [16.39406002, 59.02514162],
       [91.25682157, 47.886654  ],
       [59.75783353, 99.76065577],
       [30.12172653, 51.64006229],
       [87.04887465, 19.54581896],
       [35.4454102 , 46.78774647],
       [32.11390573, 81.20586971],
       [ 4.71182292, 36.74864897],
       [ 3.56895639, 52.79824126],
       [20.91179372, 28.06152283],
       [26.71986957, 88.80691215],
       [ 9.83678778, 67.92188701],
       [54.93751982, 43.50173492],
       [89.75902282, 61.76181973],
       [68.22378666, 60.52227261],
       [82.83370095, 52.96478617],
       [81.74331837, 93.72689515],
       [16.90881705, 52.60839024],
       [73.53944709, 69.68580462],
       [69.43734034, 21.50686882],
       [15.41377231, 52.82975414],
       [46.35854535, 53.46012437],
       [15.79851879, 51.07157605],
       [16.33483227, 15.27623169],
       [95.62494752, 65.59904866],
       [60.98646791, 12.5120598 ],
       [97.87091774, 60.25955038],
       [23.25268963, 47.88283382],
       [66.98653963, 45.82274375],
       [24.50001518, 12.27595334],
       [90.62482297, 26.10163377],
       [26.70519833, 29.97363842],
       [17.50240288, 16.19866068],
       [14.62325911, 55.85259812],
       [53.93148459, 99.97245606],
       [64.37868097, 93.79043934],
       [35.21903809, 32.54456791],
       [56.98641416, 14.63091507],
       [47.58180387, 82.16518798],
       [56.53993328, 21.09958963],
       [89.26654502, 44.93145293],
       [46.45184173, 37.40672241],
       [66.12699612, 25.57541456],
       [29.00718138,  1.33773216],
       [87.0515256 , 93.26402434],
       [32.15622127, 78.43082708],
       [94.28719189,  7.52011228],
       [14.20634369,  6.98549156],
       [53.46566368, 75.99886882],
       [ 9.95331317,  2.71253542],
       [58.7819291 , 29.15052768],
       [79.06911821, 15.86612444],
       [43.31852003, 50.16360299],
       [58.52516723, 73.98061493],
       [84.60479974, 47.95630807],
       [ 4.40048228, 44.45135353],
       [73.48815099, 21.85374176],
       [21.2640658 , 15.84101504],
       [55.26823867, 82.68339427],
       [86.39822125, 64.03112123],
       [14.0346749 , 72.51956147],
       [69.67670801, 17.69164319],
       [99.25664713, 70.12692082],
       [35.05569725, 69.25629184],
       [ 7.65597509, 65.68552145],
       [51.19585098, 95.57447799],
       [47.75689988, 19.49903901],
       [68.53747516,  9.1824247 ],
       [18.96192323, 18.01509825],
       [63.17996916, 10.61488992],
       [68.00907032, 62.11248417],
       [36.62704183, 72.57986598],
       [88.28433514, 30.50844673],
       [84.55027487, 61.67768895],
       [59.94312233, 12.63362876],
       [31.69575909, 90.27180413],
       [65.23166435,  6.20209849],
       [18.04401365, 67.39479239],
       [ 6.95013218, 11.01505727],
       [69.2557347 , 22.88141242],
       [29.04451722,  8.95596388],
       [37.3877622 , 37.53720104],
       [75.32597049, 93.23091484],
       [28.76642235, 87.88358978],
       [82.06950295, 98.32890762],
       [46.51349181, 90.51103712],
       [61.20806614, 80.32981976],
       [68.73248806,  2.26841179],
       [36.30248833, 52.77997883],
       [32.80198452, 89.83846476],
       [81.1819152 , 63.65975624],
       [83.95030926, 12.06409136],
       [91.51060964, 86.11196053],
       [16.98574816, 42.20737922],
       [18.45941299, 91.4280678 ],
       [ 7.99552013, 20.51448468],
       [68.05039452, 92.56722853],
       [26.06795327, 81.69248808],
       [92.38489181, 21.74713364],
       [69.94215954, 14.56188632],
       [19.69490657, 17.37566407],
       [89.10269625, 39.23804332],
       [70.92497779, 28.64422886],
       [91.43405098,  2.5601724 ],
       [39.32875024, 13.52380743],
       [88.57709428, 36.94695247],
       [20.23036818, 98.86408233],
       [43.84770104, 21.74923664],
       [46.96982643, 37.32634408],
       [14.08860497, 36.36306382],
       [49.39713726, 13.88859656],
       [33.55110921, 74.07535433],
       [19.21161401, 84.95056794],
       [ 2.05857404, 21.26401099],
       [65.56571167, 37.06532122],
       [44.74177903, 72.74861354],
       [66.62568178, 75.35017042],
       [46.97446318, 33.20796734],
       [88.25275129, 47.57305316],
       [37.16209226, 99.4801409 ],
       [96.48200633, 99.33198113],
       [16.24682867, 81.87817515],
       [60.93794619, 67.17168511],
       [33.34482629, 99.24595876],
       [75.50400476,  7.44998757],
       [41.36792183,  9.77071218],
       [45.18217333, 43.04981032],
       [19.9093177 , 79.63131394],
       [14.14785596, 43.49740298],
       [30.87305779, 92.15495235],
       [35.38365041, 58.00158456],
       [91.65797048, 24.12955327],
       [62.92176766, 54.11590603],
       [81.17907883, 81.70157175],
       [30.16774745, 99.80564807],
       [95.65703792, 75.65790447],
       [ 5.15771472, 74.01281953],
       [21.65343839, 97.08906101],
       [52.49049315, 86.72663471],
       [ 6.82667805,  3.75871534],
       [11.95860806, 86.02196542],
       [ 4.89411092, 43.47584336],
       [67.62348882, 11.7502263 ],
       [46.39875405, 68.02804694],
       [53.20598591, 34.0089533 ],
       [95.52395479, 57.24704101],
       [55.4862418 , 10.95895732],
       [81.15931526, 46.88664817],
       [56.69246413, 74.17423445],
       [98.54556888, 31.02732811],
       [20.89407524, 22.02683976],
       [58.11174018, 65.28731387],
       [31.83236563, 49.23416373],
       [74.77591598, 47.17096975],
       [12.02816917, 52.77834219],
       [ 6.19772266, 29.79945563],
       [ 2.61194772, 47.8387975 ],
       [30.94038661, 31.94677585],
       [76.18204171, 79.27624062],
       [53.33680115, 74.06711373],
       [89.87049934, 58.61213401],
       [87.58609509, 94.56424317],
       [24.12137522, 37.42993941],
       [75.24281553,  5.22281284],
       [15.34360843, 71.91140591],
       [54.05147419, 92.24689086],
       [78.74153397, 92.90997195],
       [63.00926935, 95.19819928],
       [70.00561018, 78.93404605],
       [48.7853465 , 58.83323314],
       [ 4.09693836, 62.35609454],
       [66.50375609, 40.8561185 ]])


def plot_clusters(X, labels, representatives, title, ax, show_representatives=True):
    """
    Plot clustered data with optional representatives.
    
    Args:
        X: Data points (n_samples, 2)
        labels: Cluster labels for each point
        representatives: List of representative arrays for each cluster
        title: Plot title
        ax: Matplotlib axis
        show_representatives: Whether to show representative points
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_clusters)))
    
    # Plot each cluster
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=[colors[i % len(colors)]], 
                  label=f'Cluster {label}',
                  alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    
    # Plot representatives
    if show_representatives and representatives:
        for i, reps in enumerate(representatives):
            if reps is not None and len(reps) > 0:
                ax.scatter(reps[:, 0], reps[:, 1], 
                          c=[colors[i % len(colors)]], 
                          marker='*', s=200, edgecolors='black', linewidth=1.5,
                          label=f'Reps {i}' if i == 0 else None)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)


def visualize_comparison(X, results_dict, suptitle="CURE Algorithm Comparison"):
    """
    Create a comparison visualization of multiple CURE results.
    
    Args:
        X: Original data
        results_dict: Dictionary with algorithm names as keys and (labels, representatives) tuples
        suptitle: Super title for the figure
    """
    n_algorithms = len(results_dict)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6 * n_algorithms, 5))
    
    if n_algorithms == 1:
        axes = [axes]
    
    for ax, (name, (labels, representatives)) in zip(axes, results_dict.items()):
        plot_clusters(X, labels, representatives, name, ax)
    
    plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def print_cluster_statistics(labels, algorithm_name):
    """Print statistics about clustering results."""
    unique_labels = np.unique(labels)
    print(f"\n{algorithm_name}:")
    print(f"  Number of clusters: {len(unique_labels)}")
    singleton_count = 0
    for label in unique_labels:
        count = np.sum(labels == label)
        if count == 1:
            singleton_count += 1
        print(f"  Cluster {label}: {count} points ({100*count/len(labels):.1f}%)")
    
    if singleton_count > 0:
        print(f"  WARNING: {singleton_count} singleton cluster(s) detected!")


def run_tests(n_clusters=3):
    """
    Run comprehensive tests on the S dataset.
    
    Args:
        n_clusters: Number of clusters to form
    """
    print("=" * 70)
    print("CURE Algorithm 2D Test Suite")
    print("=" * 70)
    print(f"\nDataset: S array ({S.shape[0]} samples, {S.shape[1]} dimensions)")
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
    euclidean_labels = euclidean_cure.fit_predict(S)
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
        sample_size=0.5,  # Use 50% of data for this small dataset
        n_partitions=3,
        exclude_outliers_from_sample=True,
        random_state=42
    )
    euclidean_scalable_labels = euclidean_scalable.fit_predict(S)
    euclidean_scalable_time = time.time() - start_time
    
    print(f"  Time: {euclidean_scalable_time:.4f} seconds")
    print(f"  Sample size used: {len(euclidean_scalable.sample_indices_)}")
    if euclidean_scalable.outlier_indices_ is not None:
        print(f"  Outliers detected: {len(euclidean_scalable.outlier_indices_)}")
    print_cluster_statistics(euclidean_scalable_labels, "Euclidean Scalable CURE")
    results["Euclidean Scalable CURE"] = (euclidean_scalable_labels, euclidean_scalable.representatives_)
    
    # Test 3: Pearson CURE (base)
    # Note: Pearson correlation works better with higher dimensional data
    # For 2D data, results may differ from Euclidean
    print("\n" + "-" * 50)
    print("Test 3: Pearson CURE (Base)")
    print("-" * 50)
    print("  Note: Pearson correlation is typically used for higher-dimensional data.")
    print("  For 2D data, it measures the linear relationship between coordinates.")
    print("  Using standardization (z-score per feature) for better stability.")
    
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore the low-dimensional warning
        
        start_time = time.time()
        pearson_cure = PearsonCURE(
            n_clusters=n_clusters,
            n_representatives=5,
            shrink_factor=0.3,
            standardize=True,  # Enable standardization for better results
            random_state=42
        )
        pearson_labels = pearson_cure.fit_predict(S)
        pearson_time = time.time() - start_time
    
    print(f"  Time: {pearson_time:.4f} seconds")
    print_cluster_statistics(pearson_labels, "Pearson CURE")
    results["Pearson CURE"] = (pearson_labels, pearson_cure.representatives_)
    
    # Test 4: Pearson Scalable CURE
    print("\n" + "-" * 50)
    print("Test 4: Pearson Scalable CURE")
    print("-" * 50)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Ignore the low-dimensional warning
        
        start_time = time.time()
        pearson_scalable = PearsonScalableCURE(
            n_clusters=n_clusters,
            n_representatives=5,
            shrink_factor=0.3,
            sample_size=0.5,
            n_partitions=3,
            standardize=True,  # Enable standardization for better results
            exclude_outliers_from_sample=True,
            random_state=42
        )
        pearson_scalable_labels = pearson_scalable.fit_predict(S)
        pearson_scalable_time = time.time() - start_time
    
    print(f"  Time: {pearson_scalable_time:.4f} seconds")
    print(f"  Sample size used: {len(pearson_scalable.sample_indices_)}")
    if pearson_scalable.outlier_indices_ is not None:
        print(f"  Outliers detected: {len(pearson_scalable.outlier_indices_)}")
    print_cluster_statistics(pearson_scalable_labels, "Pearson Scalable CURE")
    results["Pearson Scalable CURE"] = (pearson_scalable_labels, pearson_scalable.representatives_)
    
    return results


def visualize_all_results(results, save_path=None):
    """
    Create comprehensive visualization of all results.
    
    Args:
        results: Dictionary of results from run_tests()
        save_path: Optional path to save the figure
    """
    # Main comparison figure
    fig1 = visualize_comparison(S, results, "CURE Algorithm Comparison - 2D Data")
    
    # Additional detailed figure
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    for ax, (name, (labels, representatives)) in zip(axes.flat, results.items()):
        plot_clusters(S, labels, representatives, name, ax, show_representatives=True)
        
        # Add legend
        unique_labels = np.unique(labels)
        legend_elements = [Patch(facecolor=plt.cm.tab10(i / 10), 
                                 label=f'Cluster {i}: {np.sum(labels == i)} pts')
                         for i in unique_labels]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.suptitle("CURE Algorithm Comparison - 2D Test Data (S Array)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(save_path.replace('.png', '_comparison.png'), dpi=150, bbox_inches='tight')
        fig2.savefig(save_path.replace('.png', '_detailed.png'), dpi=150, bbox_inches='tight')
        print(f"\nFigures saved to {save_path}")
    
    return fig1, fig2


def test_different_cluster_counts():
    """Test with different numbers of clusters."""
    print("\n" + "=" * 70)
    print("Testing Different Cluster Counts")
    print("=" * 70)
    
    cluster_counts = [2, 3, 4, 5]
    
    fig, axes = plt.subplots(2, len(cluster_counts), figsize=(5 * len(cluster_counts), 10))
    
    for col, k in enumerate(cluster_counts):
        # Euclidean CURE
        cure_euc = EuclideanCURE(n_clusters=k, n_representatives=5, shrink_factor=0.3, random_state=42)
        labels_euc = cure_euc.fit_predict(S)
        plot_clusters(S, labels_euc, cure_euc.representatives_, 
                     f"Euclidean (k={k})", axes[0, col])
        
        # Pearson CURE
        cure_prs = PearsonCURE(n_clusters=k, n_representatives=5, shrink_factor=0.3, random_state=42)
        labels_prs = cure_prs.fit_predict(S)
        plot_clusters(S, labels_prs, cure_prs.representatives_, 
                     f"Pearson (k={k})", axes[1, col])
    
    axes[0, 0].set_ylabel('Euclidean CURE', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Pearson CURE', fontsize=12, fontweight='bold')
    
    plt.suptitle("Effect of Cluster Count on CURE Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def test_shrink_factors():
    """Test effect of different shrink factors."""
    print("\n" + "=" * 70)
    print("Testing Different Shrink Factors")
    print("=" * 70)
    
    shrink_factors = [0.0, 0.2, 0.5, 0.8]
    n_clusters = 3
    
    fig, axes = plt.subplots(1, len(shrink_factors), figsize=(5 * len(shrink_factors), 5))
    
    for col, alpha in enumerate(shrink_factors):
        cure = EuclideanCURE(n_clusters=n_clusters, n_representatives=5, 
                            shrink_factor=alpha, random_state=42)
        labels = cure.fit_predict(S)
        plot_clusters(S, labels, cure.representatives_, 
                     f"α = {alpha}", axes[col])
    
    plt.suptitle("Effect of Shrink Factor (α) on Representative Points", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Run main tests
    results = run_tests(n_clusters=3)
    
    # Create visualizations
    print("\n" + "=" * 70)
    print("Creating Visualizations...")
    print("=" * 70)
    
    # Main results
    fig1, fig2 = visualize_all_results(results)
    
    # Additional tests
    fig3 = test_different_cluster_counts()
    fig4 = test_shrink_factors()
    
    # Show all figures
    plt.show()
    
    print("\n" + "=" * 70)
    print("All tests completed!")
    print("=" * 70)
