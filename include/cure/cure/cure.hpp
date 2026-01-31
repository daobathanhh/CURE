#ifndef CURE_CPP_CURE_HPP
#define CURE_CPP_CURE_HPP

#include "../common/types.hpp"
#include "cluster.hpp"
#include "kd_tree.hpp"

#include <vector>
#include <unordered_map>

namespace cure {

/**
 * @brief CURE (Clustering Using REpresentatives) Algorithm
 * 
 * Based on the paper:
 * "CURE: An Efficient Clustering Algorithm for Large Databases"
 * by Sudipto Guha, Rajeev Rastogi, and Kyuseok Shim (SIGMOD 1998)
 * 
 * The algorithm:
 * 1. Each point starts as its own cluster
 * 2. Build KD-tree T with all representative points
 * 3. Build heap Q ordered by distance to closest cluster
 * 4. While more than k clusters remain:
 *    - Extract cluster u with minimum distance to its closest
 *    - Merge u with v (u's closest cluster)
 *    - Update KD-tree and heap
 *    - Update closest pointers for affected clusters
 */
class CURE {
public:
    /**
     * @brief Construct CURE algorithm
     * @param config Configuration parameters
     */
    explicit CURE(const CureConfig& config = CureConfig());
    
    /**
     * @brief Construct CURE with individual parameters
     */
    CURE(int k, int c, double alpha);
    
    /**
     * @brief Set distance metric
     */
    void setMetric(DistanceMetric metric);
    
    /**
     * @brief Set verbose mode
     */
    void setVerbose(bool verbose);
    
    /**
     * @brief Set outlier-resistant sample fraction (0 = off, 0.2 = 1/5, 0.5 = 1/2 closest to centroid)
     */
    void setOutlierSampleFraction(double fraction);
    
    /**
     * @brief Fit the CURE algorithm to data
     * @param data Input data (n_samples, n_features)
     * @return Reference to this
     */
    CURE& fit(const Matrix& data);
    
    /**
     * @brief Fit and return labels
     */
    std::vector<int> fit_predict(const Matrix& data);
    
    /**
     * @brief Predict cluster labels for new data points
     * Each point is assigned to the cluster with the nearest representative
     */
    std::vector<int> predict(const Matrix& data) const;
    
    /**
     * @brief Get cluster labels
     */
    const std::vector<int>& labels() const;
    
    /**
     * @brief Get final clusters
     */
    const std::vector<Cluster>& clusters() const;
    
    /**
     * @brief Get the data
     */
    const Matrix& data() const;

private:
    int k_;                     // Number of clusters
    int c_;                     // Number of representatives per cluster
    double alpha_;              // Shrink factor
    bool verbose_;              // Print progress
    DistanceMetric metric_;     // Distance metric
    double outlier_sample_fraction_;  // Fraction of points closest to centroid for merge (0 = off)

    Matrix data_;               // Input data
    size_t n_ = 0;              // Number of points
    size_t d_ = 0;              // Number of dimensions
    
    std::vector<Cluster> clusters_;  // Final clusters
    std::vector<int> labels_;        // Cluster labels for each point
    
    /**
     * @brief Build KD-tree from representative points
     */
    std::pair<KDTree, std::vector<int>> buildKDTree(
        const std::unordered_map<int, Cluster>& clusters) const;
    
    /**
     * @brief Find closest cluster using KD-tree
     */
    std::pair<int, double> findClosestKDTree(
        const Cluster& query,
        const KDTree& kdtree,
        const std::vector<int>& rep_map,
        double threshold = INF) const;
    
    /**
     * @brief Find closest cluster using brute force
     */
    std::pair<int, double> findClosestBruteForce(
        const Cluster& query,
        const std::unordered_map<int, Cluster>& clusters) const;
    
    /**
     * @brief Recover original point from shrunk representative
     */
    Point unshrinkPoint(const Point& shrunk, const Point& mean) const;
    
    /**
     * @brief Merge two clusters
     */
    Cluster mergeClusters(const Cluster& u, const Cluster& v, int new_id) const;

    /**
     * @brief Assign every point to the cluster with nearest representative (for use after outlier sampling)
     */
    void assignAllPointsByNearestRep();
};

/**
 * @brief Scalable CURE for large datasets
 * 
 * Uses random sampling and partitioning as described in Section 4 of the paper:
 * 1. Draw random sample from data
 * 2. Partition sample into p partitions
 * 3. Partially cluster each partition
 * 4. Cluster the partial clusters in second pass
 * 5. Eliminate outliers
 * 6. Label all data points using representatives
 */
class ScalableCURE {
public:
    /**
     * @brief Construct Scalable CURE
     */
    explicit ScalableCURE(const ScalableCureConfig& config = ScalableCureConfig());
    
    /**
     * @brief Set distance metric
     */
    void setMetric(DistanceMetric metric);
    
    /**
     * @brief Set verbose mode
     */
    void setVerbose(bool verbose);
    
    /**
     * @brief Fit to data
     */
    ScalableCURE& fit(const Matrix& data);
    
    /**
     * @brief Fit and return labels
     */
    std::vector<int> fit_predict(const Matrix& data);
    
    /**
     * @brief Get labels
     */
    const std::vector<int>& labels() const;
    
    /**
     * @brief Get clusters
     */
    const std::vector<Cluster>& clusters() const;

private:
    int k_;
    int c_;
    double alpha_;
    double sample_size_;
    int n_partitions_;
    int reduce_factor_;
    int outlier_threshold_;
    int random_seed_;
    bool sample_size_auto_;
    double sample_size_exponent_;
    double sample_size_multiplier_;
    int sample_size_min_;
    int sample_size_max_;
    bool use_centroid_sampling_;
    bool verbose_;
    DistanceMetric metric_;
    
    Matrix data_;
    size_t n_ = 0;
    size_t d_ = 0;
    
    std::vector<Cluster> clusters_;
    std::vector<int> labels_;
    
    /**
     * @brief Compute actual sample size
     */
    size_t computeSampleSize() const;
    
    /**
     * @brief Partially cluster a partition
     */
    std::vector<Cluster> partialCluster(
        const Matrix& points,
        const IndexList& indices,
        int target_clusters) const;
    
    /**
     * @brief Assign labels using nearest representative
     */
    void assignLabels();
};

/**
 * @brief Convenience function for CURE clustering
 */
std::vector<int> cure_clustering(
    const Matrix& data,
    int k = 5,
    int c = 5,
    double alpha = 0.3,
    bool scalable = false,
    bool verbose = false);

} // namespace cure

#endif // CURE_CPP_CURE_HPP
