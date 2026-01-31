/**
 * @file main.cpp
 * @brief Example usage of CURE clustering algorithm
 * 
 * This example demonstrates:
 * 1. Basic CURE clustering with Euclidean distance
 * 2. CURE with Pearson correlation distance
 * 3. Scalable CURE for larger datasets
 */

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

#include "cure/cure/cure.hpp"

using namespace cure;

/**
 * @brief Generate synthetic clustered data
 * 
 * Creates n_clusters gaussian clusters in 2D space
 */
Matrix generateClusteredData(int n_points_per_cluster, int n_clusters, 
                             double cluster_std = 2.0, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, cluster_std);
    
    Matrix data;
    
    // Generate cluster centers spread out in space
    std::vector<std::pair<double, double>> centers;
    for (int i = 0; i < n_clusters; ++i) {
        double x = (i % 3) * 15.0;
        double y = (i / 3) * 15.0;
        centers.emplace_back(x, y);
    }
    
    // Generate points around each center
    for (int c = 0; c < n_clusters; ++c) {
        for (int i = 0; i < n_points_per_cluster; ++i) {
            Point p = {
                centers[c].first + normal(rng),
                centers[c].second + normal(rng)
            };
            data.push_back(p);
        }
    }
    
    return data;
}

/**
 * @brief Generate high-dimensional data with correlated patterns
 * 
 * Useful for demonstrating Pearson correlation distance
 */
Matrix generatePatternData(int n_points_per_cluster, int n_clusters,
                           int n_features = 10, double noise = 0.2, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, noise);
    
    Matrix data;
    
    // Generate base patterns
    std::vector<Point> patterns(n_clusters);
    for (int c = 0; c < n_clusters; ++c) {
        patterns[c].resize(n_features);
        for (int i = 0; i < n_features; ++i) {
            // Different pattern types for each cluster
            if (c == 0) {
                patterns[c][i] = std::sin(2.0 * M_PI * i / n_features);
            } else if (c == 1) {
                patterns[c][i] = std::cos(2.0 * M_PI * i / n_features);
            } else {
                patterns[c][i] = static_cast<double>(i) / n_features;
            }
        }
    }
    
    // Generate points with noise
    for (int c = 0; c < n_clusters; ++c) {
        for (int i = 0; i < n_points_per_cluster; ++i) {
            Point p(n_features);
            for (int j = 0; j < n_features; ++j) {
                p[j] = patterns[c][j] + normal(rng);
            }
            data.push_back(p);
        }
    }
    
    return data;
}

/**
 * @brief Compute simple clustering accuracy (when true labels are known)
 */
double computeAccuracy(const std::vector<int>& labels, 
                       const std::vector<int>& true_labels,
                       int n_clusters) {
    // Create confusion matrix
    std::vector<std::vector<int>> confusion(n_clusters, std::vector<int>(n_clusters, 0));
    
    for (size_t i = 0; i < labels.size(); ++i) {
        int true_label = true_labels[i];
        int pred_label = labels[i];
        if (true_label < n_clusters && pred_label < n_clusters) {
            confusion[true_label][pred_label]++;
        }
    }
    
    // Find best assignment (greedy)
    int correct = 0;
    std::vector<bool> used(n_clusters, false);
    
    for (int iter = 0; iter < n_clusters; ++iter) {
        int best_true = -1, best_pred = -1, best_count = -1;
        for (int t = 0; t < n_clusters; ++t) {
            for (int p = 0; p < n_clusters; ++p) {
                if (!used[p] && confusion[t][p] > best_count) {
                    best_count = confusion[t][p];
                    best_true = t;
                    best_pred = p;
                }
            }
        }
        if (best_pred >= 0) {
            used[best_pred] = true;
            correct += best_count;
            // Clear this row to avoid counting again
            for (int p = 0; p < n_clusters; ++p) {
                confusion[best_true][p] = -1;
            }
        }
    }
    
    return static_cast<double>(correct) / labels.size();
}

void printSeparator() {
    std::cout << "\n" << std::string(60, '=') << "\n\n";
}

int main() {
    std::cout << "CURE Clustering Algorithm - C++ Implementation\n";
    std::cout << "================================================\n\n";
    
    // =========================================================
    // Example 1: Basic CURE with 2D data
    // =========================================================
    {
        std::cout << "Example 1: Basic CURE Clustering (2D Euclidean)\n";
        std::cout << "-----------------------------------------------\n";
        
        const int n_points = 100;
        const int n_clusters = 3;
        
        // Generate data
        Matrix data = generateClusteredData(n_points, n_clusters);
        
        // True labels
        std::vector<int> true_labels;
        for (int c = 0; c < n_clusters; ++c) {
            for (int i = 0; i < n_points; ++i) {
                true_labels.push_back(c);
            }
        }
        
        std::cout << "Generated " << data.size() << " points in " << n_clusters << " clusters\n";
        
        // Run CURE
        auto start = std::chrono::high_resolution_clock::now();
        
        CureConfig config;
        config.k = n_clusters;
        config.c = 5;
        config.alpha = 0.3;
        config.verbose = true;
        
        CURE cure(config);
        auto labels = cure.fit_predict(data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nTime: " << duration.count() << " ms\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4) 
                  << computeAccuracy(labels, true_labels, n_clusters) << "\n";
        
        // Print cluster info
        std::cout << "\nCluster sizes: ";
        for (const auto& cluster : cure.clusters()) {
            std::cout << cluster.size() << " ";
        }
        std::cout << "\n";
    }
    
    printSeparator();
    
    // =========================================================
    // Example 2: CURE with Pearson distance (high-dimensional)
    // =========================================================
    {
        std::cout << "Example 2: CURE with Pearson Distance (10D patterns)\n";
        std::cout << "----------------------------------------------------\n";
        
        const int n_points = 100;
        const int n_clusters = 3;
        const int n_features = 10;
        
        // Generate pattern data
        Matrix data = generatePatternData(n_points, n_clusters, n_features);
        
        // True labels
        std::vector<int> true_labels;
        for (int c = 0; c < n_clusters; ++c) {
            for (int i = 0; i < n_points; ++i) {
                true_labels.push_back(c);
            }
        }
        
        std::cout << "Generated " << data.size() << " points with " 
                  << n_features << " features\n";
        
        // Run CURE with Pearson distance
        auto start = std::chrono::high_resolution_clock::now();
        
        CureConfig config;
        config.k = n_clusters;
        config.c = 5;
        config.alpha = 0.3;
        config.verbose = true;
        
        CURE cure(config);
        cure.setMetric(DistanceMetric::Pearson);
        auto labels = cure.fit_predict(data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nTime: " << duration.count() << " ms\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4) 
                  << computeAccuracy(labels, true_labels, n_clusters) << "\n";
    }
    
    printSeparator();
    
    // =========================================================
    // Example 3: Scalable CURE for larger datasets
    // =========================================================
    {
        std::cout << "Example 3: Scalable CURE (1000 points)\n";
        std::cout << "--------------------------------------\n";
        
        const int n_points = 334;  // ~1000 total with 3 clusters
        const int n_clusters = 3;
        
        // Generate larger dataset
        Matrix data = generateClusteredData(n_points, n_clusters);
        
        // True labels
        std::vector<int> true_labels;
        for (int c = 0; c < n_clusters; ++c) {
            for (int i = 0; i < n_points; ++i) {
                true_labels.push_back(c);
            }
        }
        
        std::cout << "Generated " << data.size() << " points\n";
        
        // Run Scalable CURE
        auto start = std::chrono::high_resolution_clock::now();
        
        ScalableCureConfig config;
        config.k = n_clusters;
        config.c = 5;
        config.alpha = 0.3;
        config.sample_size = 0.3;  // Use 30% of data
        config.n_partitions = 3;
        config.verbose = true;
        
        ScalableCURE cure(config);
        auto labels = cure.fit_predict(data);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nTime: " << duration.count() << " ms\n";
        std::cout << "Accuracy: " << std::fixed << std::setprecision(4) 
                  << computeAccuracy(labels, true_labels, n_clusters) << "\n";
    }
    
    printSeparator();
    
    // =========================================================
    // Example 4: Using convenience function
    // =========================================================
    {
        std::cout << "Example 4: Using convenience function cure_clustering()\n";
        std::cout << "-------------------------------------------------------\n";
        
        Matrix data = generateClusteredData(50, 4);
        
        std::cout << "Generated " << data.size() << " points\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto labels = cure_clustering(data, 4, 5, 0.3, false, true);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nTime: " << duration.count() << " ms\n";
        
        // Count cluster assignments
        std::vector<int> counts(4, 0);
        for (int label : labels) {
            if (label >= 0 && label < 4) {
                counts[label]++;
            }
        }
        
        std::cout << "Cluster distribution: ";
        for (int i = 0; i < 4; ++i) {
            std::cout << "C" << i << "=" << counts[i] << " ";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "All examples completed successfully!\n";
    
    return 0;
}
