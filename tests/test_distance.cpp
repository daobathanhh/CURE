/**
 * @file test_distance.cpp
 * @brief Unit tests for distance functions
 */

#include <iostream>
#include <cassert>
#include <cmath>

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

// ============================================================
// Tests
// ============================================================

TEST(euclidean_distance_2d) {
    Point p = {0.0, 0.0};
    Point q = {3.0, 4.0};
    
    double dist = euclidean_distance(p, q);
    ASSERT_NEAR(dist, 5.0, 1e-10);
}

TEST(euclidean_distance_same_point) {
    Point p = {5.0, 3.0, 2.0};
    
    double dist = euclidean_distance(p, p);
    ASSERT_NEAR(dist, 0.0, 1e-10);
}

TEST(euclidean_distance_high_dim) {
    Point p = {1.0, 2.0, 3.0, 4.0, 5.0};
    Point q = {2.0, 3.0, 4.0, 5.0, 6.0};
    
    // Each dimension differs by 1, so dist = sqrt(5)
    double dist = euclidean_distance(p, q);
    ASSERT_NEAR(dist, std::sqrt(5.0), 1e-10);
}

TEST(euclidean_distance_squared) {
    Point p = {0.0, 0.0};
    Point q = {3.0, 4.0};
    
    double dist_sq = euclidean_distance_squared(p, q);
    ASSERT_NEAR(dist_sq, 25.0, 1e-10);
}

TEST(pearson_distance_identical) {
    Point p = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    // Identical vectors should have distance 0
    double dist = pearson_distance(p, p);
    ASSERT_NEAR(dist, 0.0, 1e-10);
}

TEST(pearson_distance_proportional) {
    Point p = {1.0, 2.0, 3.0, 4.0, 5.0};
    Point q = {2.0, 4.0, 6.0, 8.0, 10.0};  // q = 2*p
    
    // Proportional vectors have correlation 1, distance 0
    double dist = pearson_distance(p, q);
    ASSERT_NEAR(dist, 0.0, 1e-10);
}

TEST(pearson_distance_opposite) {
    Point p = {1.0, 2.0, 3.0, 4.0, 5.0};
    Point q = {5.0, 4.0, 3.0, 2.0, 1.0};  // Reversed
    
    // Opposite patterns have correlation -1, distance 2
    double dist = pearson_distance(p, q);
    ASSERT_NEAR(dist, 2.0, 1e-10);
}

TEST(pearson_distance_uncorrelated) {
    // Two uncorrelated patterns
    Point p = {1.0, -1.0, 1.0, -1.0};
    Point q = {1.0, 1.0, -1.0, -1.0};
    
    // Should have correlation ~0, distance ~1
    double dist = pearson_distance(p, q);
    ASSERT_NEAR(dist, 1.0, 1e-10);
}

TEST(point_arithmetic) {
    Point p = {1.0, 2.0, 3.0};
    Point q = {4.0, 5.0, 6.0};
    
    // Addition
    Point sum = point_add(p, q);
    ASSERT_NEAR(sum[0], 5.0, 1e-10);
    ASSERT_NEAR(sum[1], 7.0, 1e-10);
    ASSERT_NEAR(sum[2], 9.0, 1e-10);
    
    // Subtraction
    Point diff = point_subtract(q, p);
    ASSERT_NEAR(diff[0], 3.0, 1e-10);
    ASSERT_NEAR(diff[1], 3.0, 1e-10);
    ASSERT_NEAR(diff[2], 3.0, 1e-10);
    
    // Scale
    Point scaled = point_scale(p, 2.0);
    ASSERT_NEAR(scaled[0], 2.0, 1e-10);
    ASSERT_NEAR(scaled[1], 4.0, 1e-10);
    ASSERT_NEAR(scaled[2], 6.0, 1e-10);
}

TEST(compute_mean) {
    std::vector<Point> points = {
        {1.0, 2.0},
        {3.0, 4.0},
        {5.0, 6.0}
    };
    
    Point mean = compute_mean(points);
    ASSERT_NEAR(mean[0], 3.0, 1e-10);
    ASSERT_NEAR(mean[1], 4.0, 1e-10);
}

TEST(compute_mean_indexed) {
    Matrix data = {
        {1.0, 1.0},
        {2.0, 2.0},
        {3.0, 3.0},
        {10.0, 10.0}  // Not included
    };
    
    IndexList indices = {0, 1, 2};
    Point mean = compute_mean(data, indices);
    
    ASSERT_NEAR(mean[0], 2.0, 1e-10);
    ASSERT_NEAR(mean[1], 2.0, 1e-10);
}

TEST(norm) {
    Point p = {3.0, 4.0};
    double n = norm(p);
    ASSERT_NEAR(n, 5.0, 1e-10);
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "Distance Functions Unit Tests\n";
    std::cout << std::string(50, '=') << "\n";
    
    RUN_TEST(euclidean_distance_2d);
    RUN_TEST(euclidean_distance_same_point);
    RUN_TEST(euclidean_distance_high_dim);
    RUN_TEST(euclidean_distance_squared);
    RUN_TEST(pearson_distance_identical);
    RUN_TEST(pearson_distance_proportional);
    RUN_TEST(pearson_distance_opposite);
    RUN_TEST(pearson_distance_uncorrelated);
    RUN_TEST(point_arithmetic);
    RUN_TEST(compute_mean);
    RUN_TEST(compute_mean_indexed);
    RUN_TEST(norm);
    
    std::cout << std::string(50, '=') << "\n";
    std::cout << "All distance tests PASSED!\n";
    
    return 0;
}
