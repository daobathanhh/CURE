#ifndef CURE_CPP_DISTANCE_HPP
#define CURE_CPP_DISTANCE_HPP

#include "../common/types.hpp"
#include <vector>

namespace cure {

/**
 * @brief Compute Euclidean distance between two points
 * @param p First point
 * @param q Second point
 * @return Euclidean distance
 */
double euclidean_distance(const Point& p, const Point& q);

/**
 * @brief Compute squared Euclidean distance (faster, avoids sqrt)
 * @param p First point
 * @param q Second point
 * @return Squared Euclidean distance
 */
double euclidean_distance_squared(const Point& p, const Point& q);

/**
 * @brief Compute Pearson correlation distance: d = 1 - correlation(p, q)
 * 
 * Returns value in [0, 2]:
 *   0 = identical patterns (correlation = 1)
 *   1 = no correlation (correlation = 0)
 *   2 = opposite patterns (correlation = -1)
 * 
 * @param p First point
 * @param q Second point
 * @return Pearson distance
 */
double pearson_distance(const Point& p, const Point& q);

/**
 * @brief Compute L2 norm of a point
 */
double norm(const Point& p);

/**
 * @brief Point addition: result = p + q
 */
Point point_add(const Point& p, const Point& q);

/**
 * @brief Point subtraction: result = p - q
 */
Point point_subtract(const Point& p, const Point& q);

/**
 * @brief Point scaling: result = p * scalar
 */
Point point_scale(const Point& p, double scalar);

/**
 * @brief Compute mean of multiple points
 */
Point compute_mean(const std::vector<Point>& points);

/**
 * @brief Compute mean from indexed points
 */
Point compute_mean(const Matrix& data, const IndexList& indices);

} // namespace cure

#endif // CURE_CPP_DISTANCE_HPP
