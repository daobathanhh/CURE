/**
 * @file test_cpp_vs_true.cpp
 * @brief C++ CURE vs true labels only (no Python).
 *
 * For each test case, runs C++ base CURE with Euclidean and with Pearson,
 * reports ARI vs ground-truth labels so you can compare how well each metric
 * recovers the true clusters. Does not use or overwrite comparison test data.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <algorithm>
#include <filesystem>
#include <iomanip>

#include "cure/cure/cure.hpp"
#include "cure/cure/distance.hpp"

namespace fs = std::filesystem;
using namespace cure;

// ============================================================
// Utilities (same format as comparison test)
// ============================================================

static Matrix readCSV(const std::string& filename) {
    Matrix data;
    std::ifstream file(filename);
    if (!file.is_open()) return data;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        Point row;
        std::stringstream ss(line);
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        if (!row.empty()) data.push_back(row);
    }
    return data;
}

static std::vector<int> readLabels(const std::string& filename) {
    std::vector<int> labels;
    std::ifstream file(filename);
    if (!file.is_open()) return labels;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        labels.push_back(std::stoi(line));
    }
    return labels;
}

struct TestParams {
    int k = 3;
    int c = 5;
    double alpha = 0.3;
    double outlier_sample_fraction = 0.0;

    static TestParams parse(const std::string& filename) {
        TestParams params;
        std::ifstream file(filename);
        if (!file.is_open()) return params;
        std::string content((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
        auto findValue = [&content](const std::string& key) -> std::string {
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return "";
            pos = content.find(":", pos);
            if (pos == std::string::npos) return "";
            pos++;
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
            size_t end = content.find_first_of(",}\n", pos);
            std::string s = content.substr(pos, end - pos);
            if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
                s = s.substr(1, s.size() - 2);
            return s;
        };
        std::string val;
        if (!(val = findValue("k")).empty()) params.k = std::stoi(val);
        if (!(val = findValue("c")).empty()) params.c = std::stoi(val);
        if (!(val = findValue("alpha")).empty()) params.alpha = std::stod(val);
        if (!(val = findValue("outlier_sample_fraction")).empty())
            params.outlier_sample_fraction = std::stod(val);
        return params;
    }
};

static double computeARI(const std::vector<int>& labels1, const std::vector<int>& labels2) {
    if (labels1.size() != labels2.size() || labels1.empty()) return 0.0;
    size_t n = labels1.size();
    std::map<std::pair<int,int>, int> contingency;
    std::map<int, int> sum_a, sum_b;
    for (size_t i = 0; i < n; ++i) {
        contingency[{labels1[i], labels2[i]}]++;
        sum_a[labels1[i]]++;
        sum_b[labels2[i]]++;
    }
    auto comb2 = [](long long x) -> long long { return x * (x - 1) / 2; };
    long long sum_comb_nij = 0;
    for (const auto& kv : contingency) sum_comb_nij += comb2(kv.second);
    long long sum_comb_a = 0;
    for (const auto& kv : sum_a) sum_comb_a += comb2(kv.second);
    long long sum_comb_b = 0;
    for (const auto& kv : sum_b) sum_comb_b += comb2(kv.second);
    long long comb_n = comb2(n);
    if (comb_n == 0) return 1.0;
    double expected = (double)sum_comb_a * sum_comb_b / comb_n;
    double max_index = 0.5 * (sum_comb_a + sum_comb_b);
    if (max_index == expected) return 1.0;
    return (sum_comb_nij - expected) / (max_index - expected);
}

static std::map<int, int> clusterSizes(const std::vector<int>& labels) {
    std::map<int, int> m;
    for (int L : labels) m[L]++;
    return m;
}

static std::vector<int> sortedSizes(const std::map<int, int>& sizes) {
    std::vector<int> v;
    for (const auto& kv : sizes) v.push_back(kv.second);
    std::sort(v.begin(), v.end(), std::greater<int>());
    return v;
}

static void writeCppResultsJson(const std::string& test_dir, const std::string& suffix,
                               const std::vector<int>& labels, double ari_vs_true,
                               const std::vector<int>& cluster_sizes) {
    std::string path = test_dir + "/cpp_results_" + suffix + ".json";
    std::ofstream out(path);
    if (!out.is_open()) return;
    out << "{\n  \"labels\": [";
    for (size_t i = 0; i < labels.size(); ++i) {
        if (i > 0) out << ", ";
        out << labels[i];
    }
    out << "],\n  \"ari_vs_true\": " << std::fixed << std::setprecision(6) << ari_vs_true
        << ",\n  \"n_clusters\": " << cluster_sizes.size()
        << ",\n  \"cluster_sizes\": [";
    for (size_t i = 0; i < cluster_sizes.size(); ++i) {
        if (i > 0) out << ", ";
        out << cluster_sizes[i];
    }
    out << "]\n}\n";
}

// ============================================================
// Run one case: Euclidean and Pearson ARI vs true
// ============================================================

struct CppVsTrueResult {
    std::string name;
    double ari_euclidean;
    double ari_pearson;
};

static CppVsTrueResult runCase(const std::string& test_dir, const std::string& name) {
    CppVsTrueResult out;
    out.name = name;
    out.ari_euclidean = 0.0;
    out.ari_pearson = 0.0;

    Matrix data = readCSV(test_dir + "/data.csv");
    std::vector<int> true_labels = readLabels(test_dir + "/true_labels.csv");
    TestParams params = TestParams::parse(test_dir + "/params.json");

    if (data.empty() || true_labels.size() != data.size()) {
        std::cerr << "  Skip " << name << ": bad data\n";
        return out;
    }

    CureConfig config;
    config.k = params.k;
    config.c = params.c;
    config.alpha = params.alpha;
    config.outlier_sample_fraction = params.outlier_sample_fraction;

    // Euclidean
    CURE cure_euc(config);
    cure_euc.setMetric(DistanceMetric::Euclidean);
    std::vector<int> labels_euc = cure_euc.fit_predict(data);
    out.ari_euclidean = computeARI(labels_euc, true_labels);
    writeCppResultsJson(test_dir, "euclidean", labels_euc, out.ari_euclidean,
                        sortedSizes(clusterSizes(labels_euc)));

    // Pearson
    CURE cure_pearson(config);
    cure_pearson.setMetric(DistanceMetric::Pearson);
    std::vector<int> labels_pearson = cure_pearson.fit_predict(data);
    out.ari_pearson = computeARI(labels_pearson, true_labels);
    writeCppResultsJson(test_dir, "pearson", labels_pearson, out.ari_pearson,
                        sortedSizes(clusterSizes(labels_pearson)));

    return out;
}

int main(int argc, char* argv[]) {
    std::cout << "C++ CURE vs true labels only (Euclidean + Pearson)\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::string test_data_dir;
    std::vector<std::string> possible_paths = {
        "tests/cpp_vs_true/test_data",
        "../tests/cpp_vs_true/test_data",
        "../../tests/cpp_vs_true/test_data",
        "./test_data"
    };
    if (argc > 1) {
        possible_paths.insert(possible_paths.begin(), argv[1]);
    }
    for (const auto& path : possible_paths) {
        if (fs::exists(path) && fs::is_directory(path)) {
            test_data_dir = path;
            break;
        }
    }
    if (test_data_dir.empty()) {
        std::cout << "test_cpp_vs_true: Skipped (no test data).\n";
        std::cout << "Generate with: python3 tests/cpp_vs_true/generate_test_data.py\n";
        std::cout << "Or run: ./build/tests/test_cpp_vs_true <path_to_test_data>\n";
        return 0;
    }

    std::cout << "Test data: " << test_data_dir << "\n\n";

    std::vector<std::string> test_cases;
    for (const auto& entry : fs::directory_iterator(test_data_dir)) {
        if (entry.is_directory()) {
            auto data_csv = entry.path() / "data.csv";
            if (fs::exists(data_csv))
                test_cases.push_back(entry.path().filename().string());
        }
    }
    std::sort(test_cases.begin(), test_cases.end());

    if (test_cases.empty()) {
        std::cout << "No test cases found.\n";
        return 0;
    }

    std::vector<CppVsTrueResult> results;
    for (const auto& name : test_cases) {
        std::cout << "  " << name << " ... ";
        std::cout.flush();
        CppVsTrueResult r = runCase(test_data_dir + "/" + name, name);
        results.push_back(r);
        std::cout << "Euc=" << std::fixed << std::setprecision(3) << r.ari_euclidean
                  << " Pearson=" << r.ari_pearson << "\n";
    }

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY (ARI vs true labels)\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << std::left << std::setw(22) << "Test"
              << std::setw(14) << "Euclidean ARI"
              << std::setw(14) << "Pearson ARI"
              << "Note\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(22) << r.name.substr(0, 21)
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << r.ari_euclidean
                  << std::setw(14) << r.ari_pearson;
        if (r.ari_pearson >= r.ari_euclidean - 0.05)
            std::cout << "Pearson OK\n";
        else if (r.ari_pearson < 0.5)
            std::cout << "Pearson low (check metric fit)\n";
        else
            std::cout << "\n";
    }
    std::cout << std::string(60, '-') << "\n";
    return 0;
}
