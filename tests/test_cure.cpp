/**
 * @file test_cure.cpp
 * @brief Unit tests for CURE clustering algorithm
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>
#include <set>
#include <map>

#include "cure/cure/cure.hpp"

using namespace cure;

// Simple test framework
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    test_##name(); \
    std::cout << "PASSED\n"; \
} while(0)

#define ASSERT_EQ(a, b) assert((a) == (b))
#define ASSERT_NEAR(a, b, eps) assert(std::abs((a) - (b)) < (eps))
#define ASSERT_TRUE(x) assert(x)
#define ASSERT_FALSE(x) assert(!(x))

// Helper: Generate clustered 2D data
Matrix generateClusteredData(int n_per_cluster, int n_clusters, 
                             double cluster_std = 1.0, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, cluster_std);
    
    Matrix data;
    
    for (int c = 0; c < n_clusters; ++c) {
        double cx = (c % 3) * 10.0;
        double cy = (c / 3) * 10.0;
        
        for (int i = 0; i < n_per_cluster; ++i) {
            data.push_back({cx + normal(rng), cy + normal(rng)});
        }
    }
    
    return data;
}

// Helper: Compute accuracy
double computeAccuracy(const std::vector<int>& labels, 
                       int n_per_cluster, int n_clusters) {
    // Compute majority label for each true cluster
    std::vector<std::vector<int>> cluster_labels(n_clusters);
    
    for (size_t i = 0; i < labels.size(); ++i) {
        int true_cluster = static_cast<int>(i) / n_per_cluster;
        if (true_cluster < n_clusters) {
            cluster_labels[true_cluster].push_back(labels[i]);
        }
    }
    
    int correct = 0;
    std::set<int> used_labels;
    
    for (int c = 0; c < n_clusters; ++c) {
        // Find most common label in this cluster
        std::map<int, int> counts;
        for (int l : cluster_labels[c]) {
            counts[l]++;
        }
        
        int best_label = -1;
        int best_count = 0;
        for (const auto& [label, count] : counts) {
            if (count > best_count && used_labels.find(label) == used_labels.end()) {
                best_count = count;
                best_label = label;
            }
        }
        
        if (best_label >= 0) {
            correct += best_count;
            used_labels.insert(best_label);
        }
    }
    
    return static_cast<double>(correct) / labels.size();
}

// ============================================================
// Tests
// ============================================================

TEST(cluster_creation) {
    Cluster c(0, 0, Point{1.0, 2.0});
    
    ASSERT_EQ(c.id, 0);
    ASSERT_EQ(c.size(), 1);
    ASSERT_TRUE(c.alive);
    ASSERT_EQ(c.closest, -1);
}

TEST(cluster_distance_euclidean) {
    Cluster u(0, 0, Point{0.0, 0.0});
    Cluster v(1, 1, Point{3.0, 4.0});
    
    double dist = cluster_distance_euclidean(u, v);
    ASSERT_NEAR(dist, 5.0, 1e-10);
}

TEST(basic_cure_small) {
    // Very small test case: 6 points, 2 clusters
    Matrix data = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.5, 0.5},
        {10.0, 10.0},
        {11.0, 10.0},
        {10.5, 10.5}
    };
    
    CureConfig config;
    config.k = 2;
    config.c = 2;
    config.alpha = 0.3;
    
    CURE cure(config);
    auto labels = cure.fit_predict(data);
    
    ASSERT_EQ(labels.size(), 6);
    ASSERT_EQ(cure.clusters().size(), 2);
    
    // First 3 should be in same cluster, last 3 in another
    ASSERT_EQ(labels[0], labels[1]);
    ASSERT_EQ(labels[1], labels[2]);
    ASSERT_EQ(labels[3], labels[4]);
    ASSERT_EQ(labels[4], labels[5]);
    ASSERT_TRUE(labels[0] != labels[3]);
}

TEST(cure_three_clusters) {
    const int n_per_cluster = 30;
    const int n_clusters = 3;
    
    Matrix data = generateClusteredData(n_per_cluster, n_clusters, 0.5);
    
    CureConfig config;
    config.k = n_clusters;
    config.c = 3;
    config.alpha = 0.3;
    
    CURE cure(config);
    auto labels = cure.fit_predict(data);
    
    ASSERT_EQ(labels.size(), n_per_cluster * n_clusters);
    ASSERT_EQ(cure.clusters().size(), n_clusters);
    
    // Check accuracy
    double accuracy = computeAccuracy(labels, n_per_cluster, n_clusters);
    ASSERT_TRUE(accuracy > 0.8);  // Should achieve at least 80% accuracy
}

TEST(cure_different_alpha) {
    const int n = 50;
    Matrix data = generateClusteredData(n / 2, 2, 1.0);
    
    // Test with different alpha values
    for (double alpha : {0.0, 0.3, 0.7, 1.0}) {
        CureConfig config;
        config.k = 2;
        config.c = 3;
        config.alpha = alpha;
        
        CURE cure(config);
        auto labels = cure.fit_predict(data);
        
        ASSERT_EQ(labels.size(), n);
        ASSERT_EQ(cure.clusters().size(), 2);
    }
}

TEST(cure_predict) {
    Matrix train = generateClusteredData(20, 2, 0.5);
    
    CureConfig config;
    config.k = 2;
    config.c = 3;
    config.alpha = 0.3;
    
    CURE cure(config);
    cure.fit(train);
    
    // Test prediction on new points
    Matrix test = {
        {0.5, 0.5},   // Should be cluster 0
        {10.5, 0.5}   // Should be cluster 1
    };
    
    auto predictions = cure.predict(test);
    ASSERT_EQ(predictions.size(), 2);
    ASSERT_TRUE(predictions[0] != predictions[1]);
}

TEST(scalable_cure_basic) {
    const int n_per_cluster = 100;
    const int n_clusters = 3;
    
    Matrix data = generateClusteredData(n_per_cluster, n_clusters, 1.0);
    
    ScalableCureConfig config;
    config.k = n_clusters;
    config.c = 3;
    config.alpha = 0.3;
    config.sample_size = 0.5;
    config.n_partitions = 3;
    config.random_seed = 42;
    
    ScalableCURE cure(config);
    auto labels = cure.fit_predict(data);
    
    ASSERT_EQ(labels.size(), n_per_cluster * n_clusters);
    
    // Should have at most n_clusters clusters (might be fewer due to outlier removal)
    ASSERT_TRUE(cure.clusters().size() <= static_cast<size_t>(n_clusters + 1));
}

TEST(convenience_function) {
    Matrix data = generateClusteredData(30, 3, 0.5);
    
    auto labels = cure_clustering(data, 3, 3, 0.3, false, false);
    
    ASSERT_EQ(labels.size(), 90);
    
    // Check that we have 3 distinct labels
    std::set<int> unique_labels(labels.begin(), labels.end());
    ASSERT_EQ(unique_labels.size(), 3);
}

TEST(pearson_distance_metric) {
    // Generate pattern data
    const int n_features = 10;
    Matrix data;
    
    // Pattern 1: sine wave (3 points)
    for (int i = 0; i < 3; ++i) {
        Point p(n_features);
        for (int j = 0; j < n_features; ++j) {
            p[j] = std::sin(2.0 * M_PI * j / n_features) + 0.1 * i;
        }
        data.push_back(p);
    }
    
    // Pattern 2: cosine wave (3 points)
    for (int i = 0; i < 3; ++i) {
        Point p(n_features);
        for (int j = 0; j < n_features; ++j) {
            p[j] = std::cos(2.0 * M_PI * j / n_features) + 0.1 * i;
        }
        data.push_back(p);
    }
    
    CureConfig config;
    config.k = 2;
    config.c = 2;
    config.alpha = 0.3;
    
    CURE cure(config);
    cure.setMetric(DistanceMetric::Pearson);
    auto labels = cure.fit_predict(data);
    
    ASSERT_EQ(labels.size(), 6);
    
    // First 3 should be in same cluster
    ASSERT_EQ(labels[0], labels[1]);
    ASSERT_EQ(labels[1], labels[2]);
    
    // Last 3 should be in same cluster
    ASSERT_EQ(labels[3], labels[4]);
    ASSERT_EQ(labels[4], labels[5]);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "CURE Algorithm Unit Tests\n";
    std::cout << std::string(50, '=') << "\n";
    
    RUN_TEST(cluster_creation);
    RUN_TEST(cluster_distance_euclidean);
    RUN_TEST(basic_cure_small);
    RUN_TEST(cure_three_clusters);
    RUN_TEST(cure_different_alpha);
    RUN_TEST(cure_predict);
    RUN_TEST(scalable_cure_basic);
    RUN_TEST(convenience_function);
    RUN_TEST(pearson_distance_metric);
    
    std::cout << std::string(50, '=') << "\n";
    std::cout << "All CURE tests PASSED!\n";
    
    return 0;
}
