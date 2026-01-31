/**
 * @file kdtree_example.cpp
 * @brief Example usage of KD-Tree for nearest neighbor search
 */

#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>

#include "cure/cure/kd_tree.hpp"
#include "cure/cure/distance.hpp"

using namespace cure;
using namespace std::chrono;

/**
 * @brief Generate random 2D data
 */
Matrix generateData(size_t n_points, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    Matrix data(n_points, Point(2));
    for (size_t i = 0; i < n_points; ++i) {
        data[i][0] = dist(rng);
        data[i][1] = dist(rng);
    }
    return data;
}

/**
 * @brief Brute force nearest neighbor (for comparison)
 */
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

int main() {
    std::cout << "KD-Tree Example\n";
    std::cout << std::string(50, '=') << "\n\n";
    
    // =========================================================
    // Example 1: Basic nearest neighbor search
    // =========================================================
    {
        std::cout << "Example 1: Basic Nearest Neighbor Search\n";
        std::cout << std::string(50, '-') << "\n";
        
        // Create some 2D points
        Matrix data = {
            {2.0, 3.0},
            {5.0, 4.0},
            {9.0, 6.0},
            {4.0, 7.0},
            {8.0, 1.0},
            {7.0, 2.0}
        };
        
        std::cout << "Data points:\n";
        for (size_t i = 0; i < data.size(); ++i) {
            std::cout << "  [" << i << "] (" << data[i][0] << ", " << data[i][1] << ")\n";
        }
        
        // Build KD-tree
        KDTree tree(data);
        
        // Query point
        Point query = {6.0, 3.0};
        std::cout << "\nQuery point: (" << query[0] << ", " << query[1] << ")\n";
        
        // Find nearest neighbor
        auto results = tree.query(query, 1);
        
        if (!results.empty()) {
            std::cout << "Nearest neighbor: [" << results[0].index << "] ";
            std::cout << "(" << data[results[0].index][0] << ", " 
                      << data[results[0].index][1] << ")";
            std::cout << " distance=" << std::fixed << std::setprecision(3) 
                      << results[0].distance << "\n";
        }
        
        // Find 3 nearest neighbors
        std::cout << "\n3 nearest neighbors:\n";
        results = tree.query(query, 3);
        for (const auto& r : results) {
            std::cout << "  [" << r.index << "] ";
            std::cout << "(" << data[r.index][0] << ", " << data[r.index][1] << ")";
            std::cout << " distance=" << std::fixed << std::setprecision(3) 
                      << r.distance << "\n";
        }
    }
    
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    
    // =========================================================
    // Example 2: Ball query (all points within radius)
    // =========================================================
    {
        std::cout << "Example 2: Ball Query (radius search)\n";
        std::cout << std::string(50, '-') << "\n";
        
        Matrix data = generateData(100);
        KDTree tree(data);
        
        Point center = {50.0, 50.0};
        double radius = 15.0;
        
        std::cout << "Center: (" << center[0] << ", " << center[1] << ")\n";
        std::cout << "Radius: " << radius << "\n";
        
        auto indices = tree.queryBallPoint(center, radius);
        
        std::cout << "Found " << indices.size() << " points within radius:\n";
        for (size_t i = 0; i < std::min(indices.size(), size_t(5)); ++i) {
            Index idx = indices[i];
            double dist = euclidean_distance(center, data[idx]);
            std::cout << "  [" << idx << "] (" << std::fixed << std::setprecision(1)
                      << data[idx][0] << ", " << data[idx][1] << ")";
            std::cout << " distance=" << dist << "\n";
        }
        if (indices.size() > 5) {
            std::cout << "  ... and " << (indices.size() - 5) << " more\n";
        }
    }
    
    std::cout << "\n" << std::string(50, '=') << "\n\n";
    
    // =========================================================
    // Example 3: Performance comparison with brute force
    // =========================================================
    {
        std::cout << "Example 3: Performance Comparison\n";
        std::cout << std::string(50, '-') << "\n";
        
        const size_t n_points = 10000;
        const size_t n_queries = 1000;
        
        Matrix data = generateData(n_points);
        Matrix queries = generateData(n_queries, 123);
        
        std::cout << "Data points: " << n_points << "\n";
        std::cout << "Queries: " << n_queries << "\n\n";
        
        // Build KD-tree
        auto build_start = high_resolution_clock::now();
        KDTree tree(data);
        auto build_end = high_resolution_clock::now();
        auto build_time = duration_cast<microseconds>(build_end - build_start).count();
        
        std::cout << "KD-tree build time: " << build_time << " us\n";
        
        // KD-tree queries
        auto kdtree_start = high_resolution_clock::now();
        for (const auto& query : queries) {
            tree.query(query, 1);
        }
        auto kdtree_end = high_resolution_clock::now();
        auto kdtree_time = duration_cast<microseconds>(kdtree_end - kdtree_start).count();
        
        std::cout << "KD-tree " << n_queries << " queries: " << kdtree_time << " us ";
        std::cout << "(" << std::fixed << std::setprecision(2) 
                  << (double)kdtree_time / n_queries << " us/query)\n";
        
        // Brute force queries (only first 100 for speed)
        size_t bf_queries = std::min(n_queries, size_t(100));
        auto bf_start = high_resolution_clock::now();
        for (size_t i = 0; i < bf_queries; ++i) {
            bruteForceNN(data, queries[i]);
        }
        auto bf_end = high_resolution_clock::now();
        auto bf_time = duration_cast<microseconds>(bf_end - bf_start).count();
        
        std::cout << "Brute force " << bf_queries << " queries: " << bf_time << " us ";
        std::cout << "(" << std::fixed << std::setprecision(2) 
                  << (double)bf_time / bf_queries << " us/query)\n";
        
        // Speedup
        double speedup = ((double)bf_time / bf_queries) / ((double)kdtree_time / n_queries);
        std::cout << "\nKD-tree speedup: " << std::fixed << std::setprecision(1) 
                  << speedup << "x\n";
        
        // Verify correctness
        std::cout << "\nVerifying correctness (first 10 queries):\n";
        bool all_correct = true;
        for (size_t i = 0; i < 10; ++i) {
            auto kdtree_result = tree.query(queries[i], 1);
            auto bf_result = bruteForceNN(data, queries[i]);
            
            bool correct = (std::abs(kdtree_result[0].distance - bf_result.first) < 1e-10);
            if (!correct) all_correct = false;
            
            std::cout << "  Query " << i << ": KD=" << kdtree_result[0].index 
                      << " BF=" << bf_result.second 
                      << (correct ? " OK" : " MISMATCH") << "\n";
        }
        
        std::cout << "\nAll results " << (all_correct ? "CORRECT" : "HAVE ERRORS") << "\n";
    }
    
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "KD-tree examples completed!\n";
    
    return 0;
}
