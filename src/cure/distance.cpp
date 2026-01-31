#include "cure/cure/distance.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace cure {

double euclidean_distance(const Point& p, const Point& q) {
    if (p.size() != q.size()) {
        return INF;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        double diff = p[i] - q[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

double euclidean_distance_squared(const Point& p, const Point& q) {
    if (p.size() != q.size()) {
        return INF;
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < p.size(); ++i) {
        double diff = p[i] - q[i];
        sum += diff * diff;
    }
    return sum;
}

double pearson_distance(const Point& p, const Point& q) {
    if (p.size() != q.size() || p.size() < 2) {
        // Fallback for very low dimensional data
        double diff = euclidean_distance(p, q);
        double max_norm = std::max({
            std::sqrt(std::inner_product(p.begin(), p.end(), p.begin(), 0.0)),
            std::sqrt(std::inner_product(q.begin(), q.end(), q.begin(), 0.0)),
            EPS
        });
        return std::min(diff / max_norm, 2.0);
    }
    
    const size_t n = p.size();
    
    // Compute means
    double mean_p = std::accumulate(p.begin(), p.end(), 0.0) / n;
    double mean_q = std::accumulate(q.begin(), q.end(), 0.0) / n;
    
    // Compute centered dot product and norms
    double dot = 0.0, norm_p = 0.0, norm_q = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double dp = p[i] - mean_p;
        double dq = q[i] - mean_q;
        dot += dp * dq;
        norm_p += dp * dp;
        norm_q += dq * dq;
    }
    
    norm_p = std::sqrt(norm_p);
    norm_q = std::sqrt(norm_q);
    
    // Handle zero variance
    if (norm_p < EPS || norm_q < EPS) {
        return 1.0;  // Undefined correlation -> neutral distance
    }
    
    // Pearson correlation
    double correlation = dot / (norm_p * norm_q);
    
    // Clamp for numerical stability
    correlation = std::clamp(correlation, -1.0, 1.0);
    
    return 1.0 - correlation;
}

double norm(const Point& p) {
    return std::sqrt(std::inner_product(p.begin(), p.end(), p.begin(), 0.0));
}

Point point_add(const Point& p, const Point& q) {
    Point result(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        result[i] = p[i] + q[i];
    }
    return result;
}

Point point_subtract(const Point& p, const Point& q) {
    Point result(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        result[i] = p[i] - q[i];
    }
    return result;
}

Point point_scale(const Point& p, double scalar) {
    Point result(p.size());
    for (size_t i = 0; i < p.size(); ++i) {
        result[i] = p[i] * scalar;
    }
    return result;
}

Point compute_mean(const std::vector<Point>& points) {
    if (points.empty()) {
        return {};
    }
    
    const size_t dim = points[0].size();
    Point mean(dim, 0.0);
    
    for (const auto& p : points) {
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += p[i];
        }
    }
    
    for (size_t i = 0; i < dim; ++i) {
        mean[i] /= points.size();
    }
    
    return mean;
}

Point compute_mean(const Matrix& data, const IndexList& indices) {
    if (indices.empty() || data.empty()) {
        return {};
    }
    
    const size_t dim = data[0].size();
    Point mean(dim, 0.0);
    
    for (Index idx : indices) {
        for (size_t i = 0; i < dim; ++i) {
            mean[i] += data[idx][i];
        }
    }
    
    for (size_t i = 0; i < dim; ++i) {
        mean[i] /= indices.size();
    }
    
    return mean;
}

} // namespace cure
