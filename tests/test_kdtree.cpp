/**
 * @file test_kdtree.cpp
 * @brief Unit tests for KD-Tree implementation
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <random>

#include "cure/cure/kd_tree.hpp"
#include "cure/cure/distance.hpp"

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

// Helper function: brute force nearest neighbor
std::pair<double, Index> bruteForceNN(const Matrix& data, const Point& query) {
    double min_dist = INF;
    Index best_idx = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        double dist = euclidean_distance(query, data[i]);
        if (dist < min_dist) {
            min_dist = dist;
            best_idx = i;
        }
    }
    
    return {min_dist, best_idx};
}

// ============================================================
// Tests
// ============================================================

TEST(empty_tree) {
    KDTree tree;
    ASSERT_TRUE(tree.empty());
    ASSERT_EQ(tree.size(), 0);
}

TEST(single_point) {
    Matrix data = {{5.0, 3.0}};
    KDTree tree(data);
    
    ASSERT_FALSE(tree.empty());
    ASSERT_EQ(tree.size(), 1);
    
    Point query = {4.0, 2.0};
    auto results = tree.query(query, 1);
    
    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].index, 0);
    ASSERT_NEAR(results[0].distance, std::sqrt(2.0), 1e-10);
}

TEST(multiple_points_2d) {
    Matrix data = {
        {2.0, 3.0},
        {5.0, 4.0},
        {9.0, 6.0},
        {4.0, 7.0},
        {8.0, 1.0},
        {7.0, 2.0}
    };
    
    KDTree tree(data);
    ASSERT_EQ(tree.size(), 6);
    
    // Query for nearest to (6, 3) - should be (7, 2) at index 5
    Point query = {6.0, 3.0};
    auto results = tree.query(query, 1);
    
    ASSERT_EQ(results.size(), 1);
    ASSERT_EQ(results[0].index, 5);  // (7, 2)
}

TEST(knn_query) {
    Matrix data = {
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0},
        {10.0, 10.0},
        {11.0, 11.0}
    };
    
    KDTree tree(data);
    
    Point query = {2.5, 2.5};
    auto results = tree.query(query, 3);
    
    ASSERT_EQ(results.size(), 3);
    
    // Should return indices 1, 2, 0 (in order of distance)
    // Closest: (2,2) or (3,3), then the other, then (1,1)
    ASSERT_TRUE(results[0].index == 1 || results[0].index == 2);
}

TEST(distance_upper_bound) {
    Matrix data = {
        {0.0, 0.0},
        {10.0, 10.0},
        {100.0, 100.0}
    };
    
    KDTree tree(data);
    
    Point query = {0.0, 0.0};
    
    // With no bound, should find all
    auto all_results = tree.query(query, 3);
    ASSERT_EQ(all_results.size(), 3);
    
    // With bound of 20, should only find first two
    auto bounded_results = tree.query(query, 3, 20.0);
    ASSERT_TRUE(bounded_results.size() <= 2);
    
    for (const auto& r : bounded_results) {
        ASSERT_TRUE(r.distance < 20.0);
    }
}

TEST(ball_query) {
    Matrix data = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {10.0, 10.0}
    };
    
    KDTree tree(data);
    
    Point center = {0.5, 0.5};
    auto indices = tree.queryBallPoint(center, 1.0);
    
    // Should find first 4 points (all within sqrt(0.5) < 1.0 from center)
    ASSERT_EQ(indices.size(), 4);
    
    // Check that point at (10, 10) is not included
    for (Index idx : indices) {
        ASSERT_TRUE(idx != 4);
    }
}

TEST(correctness_random) {
    // Generate random data
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    const size_t n_points = 1000;
    const size_t n_dims = 3;
    const size_t n_queries = 100;
    
    Matrix data(n_points, Point(n_dims));
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j < n_dims; ++j) {
            data[i][j] = dist(rng);
        }
    }
    
    KDTree tree(data);
    
    // Test queries
    for (size_t q = 0; q < n_queries; ++q) {
        Point query(n_dims);
        for (size_t j = 0; j < n_dims; ++j) {
            query[j] = dist(rng);
        }
        
        auto kdtree_result = tree.query(query, 1);
        auto bf_result = bruteForceNN(data, query);
        
        // Distance should match
        ASSERT_NEAR(kdtree_result[0].distance, bf_result.first, 1e-10);
    }
}

TEST(high_dimensional) {
    // Test with 10 dimensions
    const size_t n_dims = 10;
    const size_t n_points = 500;
    
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    Matrix data(n_points, Point(n_dims));
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j < n_dims; ++j) {
            data[i][j] = dist(rng);
        }
    }
    
    KDTree tree(data);
    ASSERT_EQ(tree.size(), n_points);
    
    // Query
    Point query(n_dims);
    for (size_t j = 0; j < n_dims; ++j) {
        query[j] = 0.5;
    }
    
    auto results = tree.query(query, 5);
    ASSERT_EQ(results.size(), 5);
    
    // Results should be sorted by distance
    for (size_t i = 1; i < results.size(); ++i) {
        ASSERT_TRUE(results[i-1].distance <= results[i].distance);
    }
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "KD-Tree Unit Tests\n";
    std::cout << std::string(50, '=') << "\n";
    
    RUN_TEST(empty_tree);
    RUN_TEST(single_point);
    RUN_TEST(multiple_points_2d);
    RUN_TEST(knn_query);
    RUN_TEST(distance_upper_bound);
    RUN_TEST(ball_query);
    RUN_TEST(correctness_random);
    RUN_TEST(high_dimensional);
    
    std::cout << std::string(50, '=') << "\n";
    std::cout << "All KD-tree tests PASSED!\n";
    
    return 0;
}
