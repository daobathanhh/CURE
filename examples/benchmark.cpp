/**
 * @file benchmark.cpp
 * @brief Performance benchmark for CURE algorithm
 * 
 * Compares execution time for different dataset sizes and configurations
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

#include "cure/cure/cure.hpp"

using namespace cure;
using namespace std::chrono;

/**
 * @brief Generate random data
 */
Matrix generateRandomData(size_t n_points, size_t n_dims, unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    
    Matrix data(n_points, Point(n_dims));
    for (size_t i = 0; i < n_points; ++i) {
        for (size_t j = 0; j < n_dims; ++j) {
            data[i][j] = dist(rng);
        }
    }
    return data;
}

/**
 * @brief Run benchmark for a specific configuration
 */
void runBenchmark(const std::string& name, size_t n_points, size_t n_dims,
                  int k, int c, double alpha, bool scalable) {
    std::cout << std::setw(30) << name << " | ";
    std::cout << std::setw(8) << n_points << " | ";
    std::cout << std::setw(5) << n_dims << " | ";
    
    // Generate data
    Matrix data = generateRandomData(n_points, n_dims);
    
    // Run clustering
    auto start = high_resolution_clock::now();
    
    if (scalable) {
        ScalableCureConfig config;
        config.k = k;
        config.c = c;
        config.alpha = alpha;
        config.sample_size = 0.2;
        config.n_partitions = 5;
        
        ScalableCURE cure(config);
        cure.fit(data);
    } else {
        CureConfig config;
        config.k = k;
        config.c = c;
        config.alpha = alpha;
        
        CURE cure(config);
        cure.fit(data);
    }
    
    auto end = high_resolution_clock::now();
    auto duration_ms = duration_cast<milliseconds>(end - start).count();
    auto duration_us = duration_cast<microseconds>(end - start).count();
    
    if (duration_ms > 0) {
        std::cout << std::setw(10) << duration_ms << " ms\n";
    } else {
        std::cout << std::setw(10) << duration_us << " us\n";
    }
}

int main() {
    std::cout << "CURE Algorithm Benchmark\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << std::setw(30) << "Configuration" << " | ";
    std::cout << std::setw(8) << "Points" << " | ";
    std::cout << std::setw(5) << "Dims" << " | ";
    std::cout << std::setw(14) << "Time\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Small datasets - Base CURE
    runBenchmark("Base CURE (small)", 100, 2, 5, 5, 0.3, false);
    runBenchmark("Base CURE (small)", 200, 2, 5, 5, 0.3, false);
    runBenchmark("Base CURE (small)", 300, 2, 5, 5, 0.3, false);
    
    std::cout << std::string(70, '-') << "\n";
    
    // Medium datasets - Base CURE
    runBenchmark("Base CURE (medium)", 500, 2, 10, 5, 0.3, false);
    runBenchmark("Base CURE (medium)", 500, 5, 10, 5, 0.3, false);
    runBenchmark("Base CURE (medium)", 500, 10, 10, 5, 0.3, false);
    
    std::cout << std::string(70, '-') << "\n";
    
    // Scalable CURE comparison
    runBenchmark("Scalable CURE", 1000, 2, 10, 5, 0.3, true);
    runBenchmark("Scalable CURE", 2000, 2, 10, 5, 0.3, true);
    runBenchmark("Scalable CURE", 5000, 2, 10, 5, 0.3, true);
    
    std::cout << std::string(70, '-') << "\n";
    
    // High dimensional
    runBenchmark("High-dim (Scalable)", 1000, 20, 10, 5, 0.3, true);
    runBenchmark("High-dim (Scalable)", 1000, 50, 10, 5, 0.3, true);
    runBenchmark("High-dim (Scalable)", 1000, 100, 10, 5, 0.3, true);
    
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Benchmark completed!\n";
    
    return 0;
}
