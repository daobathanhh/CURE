#ifndef CURE_CPP_TYPES_HPP
#define CURE_CPP_TYPES_HPP

#include <vector>
#include <cstddef>
#include <limits>

namespace cure {

// Type aliases for clarity
using Point = std::vector<double>;
using Matrix = std::vector<Point>;
using Index = std::size_t;
using IndexList = std::vector<Index>;

// Constants
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 1e-10;

/**
 * @brief Distance metric types (used by CURE and MapReduce CURE)
 */
enum class DistanceMetric {
    Euclidean,
    Pearson
};

/**
 * @brief Configuration for CURE algorithm
 */
struct CureConfig {
    int k = 5;              // Number of clusters
    int c = 5;              // Number of representative points per cluster
    double alpha = 0.3;     // Shrink factor (0.1 to 0.9 only)
    bool verbose = false;   // Print progress
    /** Distance metric for all CURE and merge steps (Euclidean or Pearson). */
    DistanceMetric metric = DistanceMetric::Euclidean;
    /** Outlier-resistant sampling: fraction of points closest to global centroid
     *  used for merge phase (0 = disabled). E.g. 0.5 = 1/2, 0.2 = 1/5. Then all
     *  points are assigned to nearest cluster. */
    double outlier_sample_fraction = 0.0;
    
    CureConfig() = default;
    CureConfig(int k_, int c_, double alpha_) : k(k_), c(c_), alpha(alpha_) {}
};

/**
 * @brief Configuration for Scalable CURE algorithm
 */
struct ScalableCureConfig : public CureConfig {
    /** Fraction (0-1) or absolute count; ignored if sample_size_auto is true */
    double sample_size = 0.1;
    int n_partitions = 5;
    int reduce_factor = 3;
    int outlier_threshold = 5;
    int random_seed = -1;
    /** If true: sample_n = clamp(C * n^exponent, min, min(max, n)). E.g. 1M->2k-5k, 1B->20k-50k. */
    bool sample_size_auto = false;
    double sample_size_exponent = 0.33;   // n^0.33: sublinear growth
    double sample_size_multiplier = 50.0;  // C in C * n^exponent
    int sample_size_min = 500;             // Don't under-sample (e.g. not 100 for 100k)
    int sample_size_max = 15000;           // Cap so Step 4 (base CURE on reps) stays tractable for large n
    /** If true: sample = points nearest to global centroid (outlier-resistant); else random. */
    bool use_centroid_sampling = false;
    
    ScalableCureConfig() = default;
    ScalableCureConfig(int k_, int c_, double alpha_)
        : CureConfig(k_, c_, alpha_) {}
};

} // namespace cure

#endif // CURE_CPP_TYPES_HPP
