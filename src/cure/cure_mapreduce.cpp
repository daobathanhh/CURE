#include "cure/cure/cure_mapreduce.hpp"
#include "cure/cure/cure.hpp"
#include "cure/cure/distance.hpp"

#include <sstream>
#include <algorithm>
#include <set>
#include <cmath>
#include <random>
#include <numeric>

namespace cure {

namespace {

// Merge two clusters without raw data: weighted mean + select c scattered from union of reps, then shrink.
Cluster merge_two_cluster_summaries(
    const Cluster& u,
    const Cluster& v,
    int new_id,
    int c,
    double alpha,
    DistanceMetric metric)
{
    size_t nu = u.points_idx.size();
    size_t nv = v.points_idx.size();
    size_t nw = nu + nv;
    if (nw == 0) return Cluster();

    // Weighted mean
    Point w_mean(u.mean.size());
    for (size_t i = 0; i < u.mean.size(); ++i) {
        w_mean[i] = (u.mean[i] * static_cast<double>(nu) + v.mean[i] * static_cast<double>(nv)) / static_cast<double>(nw);
    }

    // Union of reps as candidates (use set to dedupe by coordinate)
    std::set<Point> candidates_set;
    for (const auto& p : u.reps) candidates_set.insert(p);
    for (const auto& p : v.reps) candidates_set.insert(p);
    std::vector<Point> candidates(candidates_set.begin(), candidates_set.end());

    if (static_cast<int>(candidates.size()) < c) {
        candidates.clear();
        candidates.push_back(u.mean);
        candidates.push_back(v.mean);
        for (const auto& p : u.reps) candidates.push_back(p);
        for (const auto& p : v.reps) candidates.push_back(p);
    }

    // Select c scattered: first = farthest from mean, then iteratively farthest from selected
    std::vector<Point> selected;
    auto dist_fn = [metric](const Point& a, const Point& b) {
        return metric == DistanceMetric::Euclidean ? euclidean_distance(a, b) : pearson_distance(a, b);
    };

    double max_dist = -1;
    size_t best_idx = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
        double d = dist_fn(candidates[i], w_mean);
        if (d > max_dist) { max_dist = d; best_idx = i; }
    }
    if (!candidates.empty()) selected.push_back(candidates[best_idx]);

    for (int i = 1; i < c && !selected.empty(); ++i) {
        double max_min_dist = -1;
        size_t best_candidate = 0;
        bool found = false;
        for (size_t j = 0; j < candidates.size(); ++j) {
            bool is_selected = false;
            for (const auto& s : selected) {
                if (s == candidates[j]) { is_selected = true; break; }
            }
            if (is_selected) continue;
            double min_dist = INF;
            for (const auto& s : selected)
                min_dist = std::min(min_dist, dist_fn(candidates[j], s));
            if (min_dist > max_min_dist) { max_min_dist = min_dist; best_candidate = j; found = true; }
        }
        if (found) selected.push_back(candidates[best_candidate]);
        else break;
    }

    // Shrink toward mean
    std::vector<Point> shrunk_reps;
    for (const auto& p : selected) {
        Point shrunk(p.size());
        for (size_t i = 0; i < p.size(); ++i)
            shrunk[i] = p[i] + alpha * (w_mean[i] - p[i]);
        shrunk_reps.push_back(shrunk);
    }

    IndexList w_points_idx = u.points_idx;
    w_points_idx.insert(w_points_idx.end(), v.points_idx.begin(), v.points_idx.end());
    return Cluster(new_id, std::move(w_points_idx), std::move(shrunk_reps), std::move(w_mean));
}

} // namespace

// -----------------------------------------------------------------------------
// Serialization (tab-separated: id  n_points  n_reps  dim  mean_0 ...  rep0_0 ...)
// -----------------------------------------------------------------------------

std::string cluster_serialize(const Cluster& c) {
    std::ostringstream os;
    os << c.id << "\t" << c.points_idx.size() << "\t" << c.reps.size() << "\t" << c.mean.size();
    for (double x : c.mean) os << "\t" << x;
    for (const auto& rep : c.reps)
        for (double x : rep) os << "\t" << x;
    return os.str();
}

Cluster cluster_deserialize(const std::string& line) {
    std::istringstream is(line);
    int id, n_points, n_reps, dim;
    if (!(is >> id >> n_points >> n_reps >> dim)) return Cluster();

    Point mean(static_cast<size_t>(dim));
    for (int i = 0; i < dim && is; ++i) is >> mean[i];

    std::vector<Point> reps(static_cast<size_t>(n_reps), Point(static_cast<size_t>(dim)));
    for (int r = 0; r < n_reps && is; ++r)
        for (int i = 0; i < dim && is; ++i) is >> reps[r][i];

    // Use dummy indices so points_idx.size() == n_points for weighted merge in reduce
    IndexList points_idx(static_cast<size_t>(std::max(0, n_points)), 0);
    return Cluster(id, std::move(points_idx), std::move(reps), std::move(mean));
}

// -----------------------------------------------------------------------------
// Map phase: partial CURE on one split
// -----------------------------------------------------------------------------

std::vector<Cluster> partial_cluster_map(
    const Matrix& points,
    int target_clusters,
    const CureConfig& config,
    int max_sample,
    int random_seed)
{
    if (points.empty()) return {};
    if (static_cast<int>(points.size()) <= target_clusters) {
        std::vector<Cluster> result;
        for (size_t i = 0; i < points.size(); ++i)
            result.emplace_back(static_cast<int>(i), static_cast<Index>(i), points[i]);
        return result;
    }

    // Subsample if partition is larger than max_sample to keep CURE tractable.
    // If outlier_sample_fraction > 0 (centroid-sampling mode), use centroid-biased
    // subsampling here too — take the max_sample points closest to the partition mean.
    // This is consistent with how CURE's outlier_sample_fraction works internally
    // (merge phase uses only central points), so both layers filter toward the center.
    const Matrix* data_ptr = &points;
    Matrix sampled;
    if (max_sample > 0 && static_cast<int>(points.size()) > max_sample) {
        size_t keep = static_cast<size_t>(max_sample);
        if (config.outlier_sample_fraction > 0.0) {
            // Centroid-biased: take the keep points nearest to partition mean (same metric as CURE)
            IndexList all_idx(points.size());
            std::iota(all_idx.begin(), all_idx.end(), 0);
            Point centroid = compute_mean(points, all_idx);
            std::vector<std::pair<double, size_t>> dist_idx;
            dist_idx.reserve(points.size());
            auto point_dist = [&config](const Point& a, const Point& b) {
                return config.metric == DistanceMetric::Euclidean ? euclidean_distance(a, b) : pearson_distance(a, b);
            };
            for (size_t i = 0; i < points.size(); ++i)
                dist_idx.emplace_back(point_dist(points[i], centroid), i);
            std::sort(dist_idx.begin(), dist_idx.end());
            sampled.reserve(keep);
            for (size_t i = 0; i < keep; ++i)
                sampled.push_back(points[dist_idx[i].second]);
        } else {
            // Random subsampling (original behaviour)
            std::mt19937 rng(random_seed < 0
                ? static_cast<unsigned>(std::random_device{}())
                : static_cast<unsigned>(random_seed));
            std::vector<size_t> idx(points.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            idx.resize(keep);
            sampled.reserve(keep);
            for (size_t i : idx) sampled.push_back(points[i]);
        }
        data_ptr = &sampled;
    }

    CureConfig cfg = config;
    cfg.k = target_clusters;
    CURE cure(cfg);
    cure.setMetric(config.metric);
    cure.fit(*data_ptr);

    std::vector<Cluster> result;
    for (size_t label = 0; label < cure.clusters().size(); ++label) {
        const Cluster& cl = cure.clusters()[label];
        IndexList orig_indices;
        for (Index i : cl.points_idx)
            orig_indices.push_back(i);
        result.emplace_back(static_cast<int>(label), std::move(orig_indices), cl.reps, cl.mean);
    }
    return result;
}

// -----------------------------------------------------------------------------
// Reduce phase: merge partial clusters to k final clusters
// -----------------------------------------------------------------------------

std::vector<Cluster> merge_partial_clusters_reduce(
    std::vector<Cluster> partial_clusters,
    int k,
    const CureConfig& config)
{
    if (partial_clusters.empty()) return {};
    if (static_cast<int>(partial_clusters.size()) <= k) return partial_clusters;

    // Build synthetic dataset: one row per representative, track which partial cluster each rep belongs to
    Matrix rep_data;
    std::vector<int> rep_to_partial_idx;
    for (size_t i = 0; i < partial_clusters.size(); ++i) {
        for (const auto& rep : partial_clusters[i].reps) {
            rep_data.push_back(rep);
            rep_to_partial_idx.push_back(static_cast<int>(i));
        }
    }

    CureConfig cfg = config;
    cfg.k = k;
    CURE cure(cfg);
    cure.setMetric(config.metric);
    cure.fit(rep_data);

    // Each partial cluster must go to exactly one final cluster (avoid double-counting points).
    // Use majority vote: for each partial cluster, assign it to the final cluster that received
    // the majority of its representatives.
    int n_partial = static_cast<int>(partial_clusters.size());
    std::vector<int> partial_to_final(n_partial, -1);
    for (int part_idx = 0; part_idx < n_partial; ++part_idx) {
        std::vector<int> label_votes(static_cast<size_t>(k), 0);
        for (size_t rep_idx = 0; rep_idx < rep_to_partial_idx.size(); ++rep_idx) {
            if (rep_to_partial_idx[rep_idx] == part_idx)
                label_votes[static_cast<size_t>(cure.labels()[rep_idx])]++;
        }
        int best = 0;
        for (int l = 1; l < k; ++l)
            if (label_votes[static_cast<size_t>(l)] > label_votes[static_cast<size_t>(best)])
                best = l;
        partial_to_final[static_cast<size_t>(part_idx)] = best;
    }
    std::vector<std::vector<int>> label_to_partials(static_cast<size_t>(k));
    for (int part_idx = 0; part_idx < n_partial; ++part_idx)
        label_to_partials[static_cast<size_t>(partial_to_final[static_cast<size_t>(part_idx)])].push_back(part_idx);

    // Merge partial clusters that share the same final label (use same metric as map/reduce)
    std::vector<Cluster> final_clusters;
    int c = config.c;
    double alpha = std::max(0.1, std::min(0.9, config.alpha));
    DistanceMetric metric = config.metric;

    for (int label = 0; label < k; ++label) {
        const auto& partial_ids = label_to_partials[static_cast<size_t>(label)];
        if (partial_ids.empty()) continue;

        Cluster merged = partial_clusters[static_cast<size_t>(partial_ids[0])];
        merged.id = label;
        for (size_t j = 1; j < partial_ids.size(); ++j) {
            merged = merge_two_cluster_summaries(
                merged, partial_clusters[static_cast<size_t>(partial_ids[j])],
                label, c, alpha, metric);
            merged.id = label;
        }
        final_clusters.push_back(std::move(merged));
    }
    return final_clusters;
}

// -----------------------------------------------------------------------------
// Assign points to final clusters (second pass: local partition points)
// -----------------------------------------------------------------------------

std::vector<int> assign_points_to_clusters(
    const Matrix& points,
    const std::vector<Cluster>& final_clusters,
    DistanceMetric metric)
{
    std::vector<int> labels(points.size(), 0);
    if (final_clusters.empty()) return labels;

    auto dist_fn = [metric](const Point& a, const Point& b) {
        return metric == DistanceMetric::Euclidean ? euclidean_distance(a, b) : pearson_distance(a, b);
    };

    for (size_t i = 0; i < points.size(); ++i) {
        double min_dist = INF;
        int best = 0;
        for (size_t c = 0; c < final_clusters.size(); ++c) {
            for (const auto& rep : final_clusters[c].reps) {
                double d = dist_fn(points[i], rep);
                if (d < min_dist) {
                    min_dist = d;
                    best = static_cast<int>(c);
                }
            }
        }
        labels[i] = best;
    }
    return labels;
}

// -----------------------------------------------------------------------------
// Stream I/O for Hadoop (one line per cluster)
// -----------------------------------------------------------------------------

std::vector<Cluster> clusters_deserialize_stream(std::istream& in) {
    std::vector<Cluster> out;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        out.push_back(cluster_deserialize(line));
    }
    return out;
}

void clusters_serialize_stream(std::ostream& out, const std::vector<Cluster>& clusters) {
    for (const auto& c : clusters)
        out << cluster_serialize(c) << "\n";
}

} // namespace cure
