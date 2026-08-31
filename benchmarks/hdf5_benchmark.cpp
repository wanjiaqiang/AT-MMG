#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ATMMG/ATMMG.hpp"
#include "ATMMG/utils/space.hpp"

namespace {

struct SweepPoint {
    size_t ef_search = 0;
    size_t ignored_rerank_candidates = 0;
    size_t graph_search_neighbor_cap = 0;
};

std::vector<SweepPoint> parse_sweep_points(const char* raw) {
    std::vector<SweepPoint> points;
    if (raw == nullptr || *raw == '\0') {
        return points;
    }
    const char* p = raw;
    while (*p != '\0') {
        while (*p == ',' || *p == ';' || *p == ' ' || *p == '\t') {
            ++p;
        }
        if (*p == '\0') {
            break;
        }
        char* end = nullptr;
        size_t ef = static_cast<size_t>(std::strtoull(p, &end, 10));
        if (end == p || ef == 0) {
            break;
        }
        p = end;

        size_t rerank = 0;
        size_t cap = 0;
        if (*p == ':') {
            ++p;
            rerank = static_cast<size_t>(std::strtoull(p, &end, 10));
            if (end == p) {
                break;
            }
            p = end;
            if (*p == ':') {
                ++p;
                cap = static_cast<size_t>(std::strtoull(p, &end, 10));
                if (end == p) {
                    break;
                }
                p = end;
            }
        }
        points.push_back({ef, rerank, cap});
        while (*p != '\0' && *p != ',' && *p != ';') {
            ++p;
        }
    }
    return points;
}

template <typename T>
struct Matrix {
    size_t rows = 0;
    size_t cols = 0;
    std::vector<T> data;

    const T* row(size_t r) const { return data.data() + (r * cols); }
    T* row(size_t r) { return data.data() + (r * cols); }
    const T& operator()(size_t r, size_t c) const { return data[(r * cols) + c]; }
};

template <typename T>
Matrix<T> read_2d_dataset(
    H5::H5File& file, const std::string& name, const H5::PredType& native_type
) {
    H5::DataSet dataset = file.openDataSet(name);
    H5::DataSpace space = dataset.getSpace();
    if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("dataset " + name + " is not 2D");
    }

    hsize_t dims[2] = {0, 0};
    space.getSimpleExtentDims(dims);

    Matrix<T> matrix;
    matrix.rows = static_cast<size_t>(dims[0]);
    matrix.cols = static_cast<size_t>(dims[1]);
    matrix.data.resize(matrix.rows * matrix.cols);
    dataset.read(matrix.data.data(), native_type);
    return matrix;
}

std::vector<ATMMG::PID> exact_topk(
    const float* base,
    size_t base_size,
    size_t dim,
    const float* query,
    size_t k,
    float* kth_distance_sqr = nullptr
) {
    struct Candidate {
        float dist;
        ATMMG::PID id;
        bool operator<(const Candidate& other) const { return dist < other.dist; }
    };

    std::vector<Candidate> scored(base_size);
    for (size_t i = 0; i < base_size; ++i) {
        scored[i] = {
            ATMMG::euclidean_sqr<float>(query, base + (i * dim), dim),
            static_cast<ATMMG::PID>(i)
        };
    }
    if (k < scored.size()) {
        std::nth_element(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(k), scored.end());
        scored.resize(k);
    }
    std::sort(scored.begin(), scored.end());

    std::vector<ATMMG::PID> result;
    result.reserve(std::min(k, scored.size()));
    for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
        result.push_back(scored[i].id);
    }
    if (kth_distance_sqr != nullptr) {
        *kth_distance_sqr = result.empty() ? 0.0F : scored[result.size() - 1].dist;
    }
    return result;
}

size_t overlap_count(
    const std::vector<ATMMG::PID>& results, const std::vector<ATMMG::PID>& truth
) {
    std::unordered_set<ATMMG::PID> truth_set(truth.begin(), truth.end());
    size_t overlap = 0;
    for (ATMMG::PID id : results) {
        overlap += static_cast<size_t>(truth_set.find(id) != truth_set.end());
    }
    return overlap;
}

size_t annb_recall_count_by_threshold_d2(
    const std::vector<ATMMG::PID>& results,
    const float* base,
    size_t dim,
    const float* query,
    float threshold_d2
) {
    size_t count = 0;
    for (ATMMG::PID id : results) {
        float dist2 = ATMMG::euclidean_sqr<float>(
            query, base + (static_cast<size_t>(id) * dim), dim
        );
        count += static_cast<size_t>(dist2 <= threshold_d2 * (1.0F + 1e-5F));
    }
    return count;
}

size_t annb_recall_count_by_distance(
    const std::vector<ATMMG::PID>& results,
    const float* base,
    size_t dim,
    const float* query,
    const Matrix<float>& distances,
    size_t query_id,
    size_t k
) {
    size_t truth_k = std::min(k, distances.cols);
    if (truth_k == 0) {
        return 0;
    }
    float kth_dist = distances(query_id, truth_k - 1);
    float threshold_d2 = kth_dist * kth_dist;
    return annb_recall_count_by_threshold_d2(results, base, dim, query, threshold_d2);
}

void print_stat_avg_max(const char* name, const ATMMG::StatSummary& summary) {
    std::cout << name << "_avg=" << summary.avg() << '\n';
    std::cout << name << "_max=" << summary.max << '\n';
}

void print_stat_avg_max_if_nonzero(
    const char* name, const ATMMG::StatSummary& summary
) {
    if (summary.sum != 0) {
        print_stat_avg_max(name, summary);
    }
}

void print_time_avg_max(const char* name, const ATMMG::TimeSummary& summary) {
    std::cout << name << "_ms_avg=" << summary.avg() << '\n';
    std::cout << name << "_ms_max=" << summary.max << '\n';
}

const char* center_entry_mode_name(ATMMG::CenterEntryMode mode) {
    return mode == ATMMG::CenterEntryMode::TreeOnly ? "tree_only" : "disabled";
}

void force_mainline_only(ATMMG::Config& config) {
    config.center_entry_mode = ATMMG::CenterEntryMode::TreeOnly;
    config.center_flat_exact_scan = false;
    config.center_super_level_scan = false;
    config.center_coarse_cascade_scan = false;
    config.use_hash_neighborhood_spectrum = false;
    config.use_residual_hash_spectrum = false;
    config.use_residual_hash_bucket_graph = false;
    config.graph_search_use_quant = false;
    config.graph_search_full_quant = false;
    config.graph_early_stop = false;
    config.graph_result_margin_stop = false;
    config.graph_admission_bound = false;
    config.graph_exact_l2_fast_path = false;
    config.graph_lazy_center_distance = false;
    config.graph_distance_use_norm_dot = false;
    config.graph_build_mode = 0;
    config.graph_build_bridge_edges = false;
    config.graph_query_adjacency_order = false;
    config.graph_query_front_prune = false;
    config.graph_dual_scale_search = false;
    config.graph_reorder_by_center = false;
    config.graph_query_adaptive_center_margin = false;
    config.center_adaptive_refine = false;
    config.hard_query_fallback = false;
    config.graph_adaptive_ef_min = 0;
    config.graph_query_adaptive_ef_min = 0;
    config.graph_query_adaptive_ef_mid = 0;
    config.graph_hot_neighbor_count = 0;
    config.graph_cold_neighbor_count = 0;
    config.graph_cold_max_expansions = 0;
}

}  // namespace

int run_ATMMG_hdf5(int argc, char** argv) {
    std::string hdf5_path = argc > 1 ? argv[1] : "sift-128-euclidean.hdf5";
    size_t base_limit = argc > 2 ? static_cast<size_t>(std::strtoull(argv[2], nullptr, 10)) : 10000;
    size_t eval_queries = argc > 3 ? static_cast<size_t>(std::strtoull(argv[3], nullptr, 10)) : 10;
    size_t random_seed = argc > 4 ? static_cast<size_t>(std::strtoull(argv[4], nullptr, 10)) : 42;
    constexpr size_t k_eval = 10;
    bool use_fast_query = true;
    bool print_details = false;

    H5::Exception::dontPrint();

    H5::H5File file(hdf5_path, H5F_ACC_RDONLY);
    Matrix<float> train = read_2d_dataset<float>(file, "train", H5::PredType::NATIVE_FLOAT);
    Matrix<float> test = read_2d_dataset<float>(file, "test", H5::PredType::NATIVE_FLOAT);
    Matrix<uint32_t> neighbors =
        read_2d_dataset<uint32_t>(file, "neighbors", H5::PredType::NATIVE_UINT32);
    Matrix<float> distances =
        read_2d_dataset<float>(file, "distances", H5::PredType::NATIVE_FLOAT);

    if (train.cols != test.cols) {
        throw std::runtime_error("train/test dimensions do not match");
    }
    if (neighbors.rows != test.rows || distances.rows != test.rows) {
        throw std::runtime_error("groundtruth rows do not match test rows");
    }

    base_limit = std::min(base_limit, train.rows);
    eval_queries = std::min(eval_queries, test.rows);

    std::vector<size_t> query_ids;
    if (const char* query_ids_path = std::getenv("ATMMG_QUERY_IDS_PATH")) {
        std::ifstream ids_file(query_ids_path);
        if (!ids_file) {
            throw std::runtime_error(
                std::string("failed to open ATMMG_QUERY_IDS_PATH=") + query_ids_path
            );
        }
        query_ids.reserve(eval_queries);
        size_t qid = 0;
        while (ids_file >> qid && query_ids.size() < eval_queries) {
            if (qid >= test.rows) {
                throw std::runtime_error("query id out of range in ATMMG_QUERY_IDS_PATH");
            }
            query_ids.push_back(qid);
        }
        if (query_ids.size() < eval_queries) {
            throw std::runtime_error("ATMMG_QUERY_IDS_PATH contains fewer ids than eval_queries");
        }
    } else {
        query_ids.resize(test.rows);
        std::iota(query_ids.begin(), query_ids.end(), static_cast<size_t>(0));
        std::mt19937 rng(static_cast<uint32_t>(random_seed));
        std::shuffle(query_ids.begin(), query_ids.end(), rng);
        query_ids.resize(eval_queries);
    }

    ATMMG::Config config;
    config.n_centers = 128;
    config.center_leaf_min_size = 16;
    config.center_scan_keep = 16;
    config.exact_center_keep = 6;
    config.center_refine_neighbor_scan = 0;
    config.center_coarse_projection_dims = 24;
    config.center_coarse_keep = 512;
    config.center_coarse_prefilter_dims = 0;
    config.center_coarse_prefilter_keep = 0;
    config.center_neighbor_prefilter_dims = 0;
    config.center_neighbor_prefilter_keep = 0;
    config.center_neighbor_global_guard_keep = 0;
    config.center_flat_exact_scan = false;
    config.center_super_level_scan = false;
    config.center_super_count = 128;
    config.center_super_probe = 3;
    config.center_super_overlap = 2;
    config.center_coarse_cascade_scan = false;
    config.center_cascade_low_dims = 0;
    config.center_cascade_low_keep = 0;
    config.center_cascade_mid_keep = 0;
    config.center_cascade_use_nth = false;
    config.center_topn_scan = 2000;
    config.center_topn_probe = 0;
    config.center_topn_coarse_keep = 96;
    config.init_keep = 32;
    config.use_hash_neighborhood_spectrum = false;
    config.hash_spectrum_size = 0;
    config.hash_spectrum_pool_scan = 1024;
    config.hash_spectrum_bits = 64;
    config.hash_spectrum_min_hamming = 8;
    config.hash_spectrum_segment_bits = 16;
    config.hash_spectrum_segment_radius = 2;
    config.hash_spectrum_candidates = 64;
    config.hash_spectrum_entry_take = 16;
    config.hash_spectrum_center_keep = 1;
    config.use_residual_hash_spectrum = false;
    config.residual_hash_bits = 256;
    config.residual_hash_segments = 4;
    config.residual_radius_full = 32;
    config.residual_radius_segment = 8;
    config.normalize_residual_before_hash = false;
    config.residual_hash_seed = 12345;
    config.residual_hash_filter_after = 0;
    config.use_residual_hash_bucket_graph = false;
    config.residual_hash_hot_count = 12;
    config.residual_hash_bucket_count = 16;
    config.residual_hash_bucket_take = 3;
    config.residual_hash_bucket_probe = 1;
    config.residual_hash_fallback_min = 4;
    config.residual_hash_cold_count = 8;
    config.residual_hash_build_query_buckets = false;
    config.residual_hash_node_local_bucket = false;
    config.graph_degree = 48;
    config.ef_search = 64;
    config.graph_adaptive_ef_min = 0;
    config.graph_adaptive_ef_slack = 1.0F;
    config.graph_adaptive_ef_check_interval = 4;
    config.graph_build_intra_candidates = 384;
    config.graph_build_cross_candidates = 256;
    config.graph_build_projection_dims = 6;
    config.graph_build_center_neighbors = 8;
    config.graph_insert_new_degree = 0;
    config.graph_search_use_quant = false;
    config.graph_search_full_quant = false;
    config.graph_early_stop = false;
    config.graph_search_neighbor_cap = 0;
    config.graph_late_neighbor_cap = 0;
    config.graph_late_neighbor_after = 0;
    config.graph_neighbor_prefilter_dims = 0;
    config.graph_neighbor_prefilter_keep = 0;
    config.graph_result_margin_stop = false;
    config.graph_result_margin_min_expansions = 0;
    config.graph_result_margin = 0.05F;
    config.graph_result_margin_check_interval = 4;
    config.graph_admission_bound = false;
    config.graph_admission_slack = 1.0F;
    config.graph_neighbor_prefetch = 0;
    config.graph_exact_l2_fast_path = false;
    config.graph_lazy_center_distance = false;
    config.graph_distance_use_norm_dot = false;
    config.graph_build_mode = 0;
    config.graph_vamana_alpha = 1.2F;
    config.graph_vamana_candidate_limit = 0;
    config.graph_build_bridge_edges = false;
    config.graph_bridge_center_neighbors = 2;
    config.graph_bridge_points_per_center = 2;
    config.graph_bridge_candidate_scan = 64;
    config.graph_query_adjacency_order = false;
    config.graph_query_front_prune = false;
    config.graph_reorder_by_center = false;
    config.graph_portal_pool_size = 0;
    config.graph_hot_neighbor_count = 0;
    config.graph_cold_neighbor_count = 0;
    config.graph_cold_max_expansions = 0;
    config.graph_cold_search_slack = 1.0F;
    config.graph_query_adaptive_center_margin = false;
    config.graph_query_adaptive_easy_margin = 0.08F;
    config.graph_query_adaptive_hard_margin = 0.025F;
    config.center_adaptive_refine = false;
    config.center_adaptive_coarse_keep_easy = 0;
    config.center_adaptive_coarse_keep_mid = 0;
    config.center_adaptive_scan_keep_easy = 0;
    config.center_adaptive_scan_keep_mid = 0;
    config.center_adaptive_easy_margin = 0.12F;
    config.center_adaptive_hard_margin = 0.04F;
    config.hard_query_fallback = false;
    config.hard_query_center_margin = 0.025F;
    config.hard_query_init_keep = 0;
    config.hard_query_neighbor_cap = 0;
    config.hard_query_late_neighbor_cap = 0;
    config.hard_query_late_neighbor_after = 0;
    config.hard_query_rerank_candidates = 0;
    config.random_seed = random_seed;
    config.center_entry_mode = ATMMG::CenterEntryMode::TreeOnly;

    if (argc > 5) {
        config.ef_search = static_cast<size_t>(std::strtoull(argv[5], nullptr, 10));
    }
    if (argc > 6) {
        config.center_topn_coarse_keep =
            static_cast<size_t>(std::strtoull(argv[6], nullptr, 10));
    }
    // argv[7] used to configure graph reranking; rerank count now equals query k.
    // argv[8] used to toggle a removed legacy search branch.
    if (argc > 9) {
        config.center_topn_scan =
            static_cast<size_t>(std::strtoull(argv[9], nullptr, 10));
    }
    if (argc > 10) {
        config.graph_early_stop = std::strtoull(argv[10], nullptr, 10) != 0;
    }
    if (argc > 11) {
        config.graph_search_neighbor_cap =
            static_cast<size_t>(std::strtoull(argv[11], nullptr, 10));
    }
    if (argc > 12) {
        config.center_topn_probe =
            static_cast<size_t>(std::strtoull(argv[12], nullptr, 10));
    }
    if (argc > 13) {
        config.graph_lazy_center_distance = std::strtoull(argv[13], nullptr, 10) != 0;
    }
    if (argc > 14) {
        config.n_centers = static_cast<size_t>(std::strtoull(argv[14], nullptr, 10));
    }
    if (argc > 15) {
        config.center_leaf_min_size =
            static_cast<size_t>(std::strtoull(argv[15], nullptr, 10));
    }
    if (argc > 16) {
        config.center_scan_keep =
            static_cast<size_t>(std::strtoull(argv[16], nullptr, 10));
    }
    if (argc > 17) {
        config.exact_center_keep =
            static_cast<size_t>(std::strtoull(argv[17], nullptr, 10));
    }
    if (argc > 18) {
        config.center_refine_neighbor_scan =
            static_cast<size_t>(std::strtoull(argv[18], nullptr, 10));
    }
    // argv[19] and argv[20] used to toggle removed legacy center/search branches.
    if (argc > 21) {
        config.graph_distance_use_norm_dot = std::strtoull(argv[21], nullptr, 10) != 0;
    }
    // argv[22]..argv[26] used to configure removed center-anchor seed paths.
    if (argc > 27) {
        use_fast_query = std::strtoull(argv[27], nullptr, 10) != 0;
    }
    // argv[28]..argv[32] used to configure removed micro-entry seed paths.
    if (argc > 33) {
        config.graph_build_bridge_edges = std::strtoull(argv[33], nullptr, 10) != 0;
    }
    if (argc > 34) {
        config.graph_bridge_center_neighbors =
            static_cast<size_t>(std::strtoull(argv[34], nullptr, 10));
    }
    if (argc > 35) {
        config.graph_bridge_points_per_center =
            static_cast<size_t>(std::strtoull(argv[35], nullptr, 10));
    }
    if (argc > 36) {
        config.graph_bridge_candidate_scan =
            static_cast<size_t>(std::strtoull(argv[36], nullptr, 10));
    }
    if (argc > 37) {
        config.graph_query_adjacency_order =
            std::strtoull(argv[37], nullptr, 10) != 0;
    }
    if (argc > 38) {
        config.graph_reorder_by_center = std::strtoull(argv[38], nullptr, 10) != 0;
    }
    if (argc > 39) {
        config.center_coarse_projection_dims =
            static_cast<size_t>(std::strtoull(argv[39], nullptr, 10));
    }
    if (argc > 40) {
        config.center_coarse_keep =
            static_cast<size_t>(std::strtoull(argv[40], nullptr, 10));
    }
    if (argc > 41) {
        config.graph_early_stop_min_expansions =
            static_cast<size_t>(std::strtoull(argv[41], nullptr, 10));
    }
    if (argc > 42) {
        config.graph_early_stop_slack = static_cast<float>(std::atof(argv[42]));
    }
    if (argc > 44) {
        config.graph_query_front_prune = std::strtoull(argv[44], nullptr, 10) != 0;
    }
    // argv[45] used to configure removed micro-entry seed rerank behavior.
    if (argc > 46) {
        config.graph_portal_pool_size =
            static_cast<size_t>(std::strtoull(argv[46], nullptr, 10));
    }
    if (argc > 47) {
        config.graph_hot_neighbor_count =
            static_cast<size_t>(std::strtoull(argv[47], nullptr, 10));
    }
    if (argc > 48) {
        config.graph_cold_neighbor_count =
            static_cast<size_t>(std::strtoull(argv[48], nullptr, 10));
    }
    if (argc > 49) {
        config.graph_cold_search_slack = static_cast<float>(std::atof(argv[49]));
    }
    if (argc > 50) {
        config.graph_cold_max_expansions =
            static_cast<size_t>(std::strtoull(argv[50], nullptr, 10));
    }
    if (argc > 51) {
        config.graph_neighbor_prefilter_dims =
            static_cast<size_t>(std::strtoull(argv[51], nullptr, 10));
    }
    if (argc > 52) {
        config.graph_neighbor_prefilter_keep =
            static_cast<size_t>(std::strtoull(argv[52], nullptr, 10));
    }
    if (argc > 53) {
        config.graph_admission_bound = std::strtoull(argv[53], nullptr, 10) != 0;
    }
    if (argc > 54) {
        config.graph_admission_slack = static_cast<float>(std::atof(argv[54]));
    }
    if (argc > 55) {
        config.graph_neighbor_prefetch =
            static_cast<size_t>(std::strtoull(argv[55], nullptr, 10));
    }
    if (argc > 56) {
        config.graph_build_mode =
            static_cast<size_t>(std::strtoull(argv[56], nullptr, 10));
    }
    if (argc > 57) {
        config.graph_adaptive_ef_min =
            static_cast<size_t>(std::strtoull(argv[57], nullptr, 10));
    }
    if (argc > 58) {
        config.graph_adaptive_ef_slack = static_cast<float>(std::atof(argv[58]));
    }
    if (argc > 59) {
        config.graph_adaptive_ef_check_interval =
            static_cast<size_t>(std::strtoull(argv[59], nullptr, 10));
    }
    if (argc > 60) {
        config.graph_degree =
            static_cast<size_t>(std::strtoull(argv[60], nullptr, 10));
    }
    if (argc > 61) {
        config.graph_build_intra_candidates =
            static_cast<size_t>(std::strtoull(argv[61], nullptr, 10));
    }
    if (argc > 62) {
        config.graph_build_cross_candidates =
            static_cast<size_t>(std::strtoull(argv[62], nullptr, 10));
    }
    if (argc > 63) {
        config.graph_build_center_neighbors =
            static_cast<size_t>(std::strtoull(argv[63], nullptr, 10));
    }
    if (argc > 64) {
        config.graph_build_projection_dims =
            static_cast<size_t>(std::strtoull(argv[64], nullptr, 10));
    }
    if (argc > 65) {
        config.graph_vamana_alpha = static_cast<float>(std::atof(argv[65]));
    }
    if (argc > 66) {
        config.graph_vamana_candidate_limit =
            static_cast<size_t>(std::strtoull(argv[66], nullptr, 10));
    }
    if (argc > 68) {
        config.graph_query_adaptive_ef_min =
            static_cast<size_t>(std::strtoull(argv[68], nullptr, 10));
    }
    if (argc > 69) {
        config.graph_query_adaptive_ef_mid =
            static_cast<size_t>(std::strtoull(argv[69], nullptr, 10));
    }
    if (argc > 70) {
        config.graph_query_adaptive_l1_low_quantile =
            static_cast<float>(std::atof(argv[70]));
    }
    if (argc > 71) {
        config.graph_query_adaptive_l1_high_quantile =
            static_cast<float>(std::atof(argv[71]));
    }
    if (argc > 72) {
        config.graph_insert_new_degree =
            static_cast<size_t>(std::strtoull(argv[72], nullptr, 10));
    }
    if (argc > 73) {
        config.graph_query_adaptive_center_margin =
            std::strtoull(argv[73], nullptr, 10) != 0;
    }
    if (argc > 74) {
        config.graph_query_adaptive_easy_margin = static_cast<float>(std::atof(argv[74]));
    }
    if (argc > 75) {
        config.graph_query_adaptive_hard_margin = static_cast<float>(std::atof(argv[75]));
    }
    if (argc > 76) {
        config.graph_late_neighbor_cap =
            static_cast<size_t>(std::strtoull(argv[76], nullptr, 10));
    }
    if (argc > 77) {
        config.graph_late_neighbor_after =
            static_cast<size_t>(std::strtoull(argv[77], nullptr, 10));
    }
    if (argc > 78) {
        config.center_adaptive_refine = std::strtoull(argv[78], nullptr, 10) != 0;
    }
    if (argc > 79) {
        config.center_adaptive_coarse_keep_easy =
            static_cast<size_t>(std::strtoull(argv[79], nullptr, 10));
    }
    if (argc > 80) {
        config.center_adaptive_coarse_keep_mid =
            static_cast<size_t>(std::strtoull(argv[80], nullptr, 10));
    }
    if (argc > 81) {
        config.center_adaptive_scan_keep_easy =
            static_cast<size_t>(std::strtoull(argv[81], nullptr, 10));
    }
    if (argc > 82) {
        config.center_adaptive_scan_keep_mid =
            static_cast<size_t>(std::strtoull(argv[82], nullptr, 10));
    }
    if (argc > 83) {
        config.center_adaptive_easy_margin = static_cast<float>(std::atof(argv[83]));
    }
    if (argc > 84) {
        config.center_adaptive_hard_margin = static_cast<float>(std::atof(argv[84]));
    }
    if (argc > 85) {
        config.center_coarse_prefilter_dims =
            static_cast<size_t>(std::strtoull(argv[85], nullptr, 10));
    }
    if (argc > 86) {
        config.center_coarse_prefilter_keep =
            static_cast<size_t>(std::strtoull(argv[86], nullptr, 10));
    }
    if (argc > 87) {
        config.center_neighbor_prefilter_dims =
            static_cast<size_t>(std::strtoull(argv[87], nullptr, 10));
    }
    if (argc > 88) {
        config.center_neighbor_prefilter_keep =
            static_cast<size_t>(std::strtoull(argv[88], nullptr, 10));
    }
    if (argc > 89) {
        config.center_neighbor_global_guard_keep =
            static_cast<size_t>(std::strtoull(argv[89], nullptr, 10));
    }
    if (argc > 90) {
        config.graph_result_margin_stop = std::strtoull(argv[90], nullptr, 10) != 0;
    }
    if (argc > 91) {
        config.graph_result_margin_min_expansions =
            static_cast<size_t>(std::strtoull(argv[91], nullptr, 10));
    }
    if (argc > 92) {
        config.graph_result_margin = static_cast<float>(std::atof(argv[92]));
    }
    if (argc > 93) {
        config.graph_result_margin_check_interval =
            static_cast<size_t>(std::strtoull(argv[93], nullptr, 10));
    }
    if (argc > 94) {
        config.graph_exact_l2_fast_path = std::strtoull(argv[94], nullptr, 10) != 0;
    }
    if (argc > 95) {
        config.center_flat_exact_scan = std::strtoull(argv[95], nullptr, 10) != 0;
    }
    if (argc > 96) {
        config.center_coarse_cascade_scan = std::strtoull(argv[96], nullptr, 10) != 0;
    }
    if (argc > 97) {
        config.center_cascade_low_dims =
            static_cast<size_t>(std::strtoull(argv[97], nullptr, 10));
    }
    if (argc > 98) {
        config.center_cascade_low_keep =
            static_cast<size_t>(std::strtoull(argv[98], nullptr, 10));
    }
    if (argc > 99) {
        config.center_cascade_mid_keep =
            static_cast<size_t>(std::strtoull(argv[99], nullptr, 10));
    }
    if (argc > 100) {
        config.center_cascade_use_nth = std::strtoull(argv[100], nullptr, 10) != 0;
    }
    if (argc > 101) {
        print_details = std::strtoull(argv[101], nullptr, 10) != 0;
    }
    if (argc > 102) {
        config.hard_query_fallback = std::strtoull(argv[102], nullptr, 10) != 0;
    }
    if (argc > 103) {
        config.hard_query_center_margin = static_cast<float>(std::atof(argv[103]));
    }
    // argv[104] and argv[105] used to configure removed hard-query micro seeds.
    if (argc > 106) {
        config.hard_query_init_keep =
            static_cast<size_t>(std::strtoull(argv[106], nullptr, 10));
    }
    if (argc > 107) {
        config.hard_query_neighbor_cap =
            static_cast<size_t>(std::strtoull(argv[107], nullptr, 10));
    }
    if (argc > 108) {
        config.hard_query_late_neighbor_cap =
            static_cast<size_t>(std::strtoull(argv[108], nullptr, 10));
    }
    if (argc > 109) {
        config.hard_query_late_neighbor_after =
            static_cast<size_t>(std::strtoull(argv[109], nullptr, 10));
    }
    if (argc > 110) {
        config.hard_query_rerank_candidates =
            static_cast<size_t>(std::strtoull(argv[110], nullptr, 10));
    }
    if (argc > 111) {
        config.use_hash_neighborhood_spectrum =
            std::strtoull(argv[111], nullptr, 10) != 0;
    }
    if (argc > 112) {
        config.hash_spectrum_size =
            static_cast<size_t>(std::strtoull(argv[112], nullptr, 10));
    }
    if (argc > 113) {
        config.hash_spectrum_pool_scan =
            static_cast<size_t>(std::strtoull(argv[113], nullptr, 10));
    }
    if (argc > 114) {
        config.hash_spectrum_bits =
            static_cast<size_t>(std::strtoull(argv[114], nullptr, 10));
    }
    if (argc > 115) {
        config.hash_spectrum_min_hamming =
            static_cast<size_t>(std::strtoull(argv[115], nullptr, 10));
    }
    if (argc > 116) {
        config.hash_spectrum_segment_bits =
            static_cast<size_t>(std::strtoull(argv[116], nullptr, 10));
    }
    if (argc > 117) {
        config.hash_spectrum_segment_radius =
            static_cast<size_t>(std::strtoull(argv[117], nullptr, 10));
    }
    if (argc > 118) {
        config.hash_spectrum_candidates =
            static_cast<size_t>(std::strtoull(argv[118], nullptr, 10));
    }
    if (argc > 119) {
        config.hash_spectrum_entry_take =
            static_cast<size_t>(std::strtoull(argv[119], nullptr, 10));
    }
    if (argc > 120) {
        config.hash_spectrum_center_keep =
            static_cast<size_t>(std::strtoull(argv[120], nullptr, 10));
    }
    if (argc > 121) {
        config.use_residual_hash_spectrum =
            std::strtoull(argv[121], nullptr, 10) != 0;
    }
    if (argc > 122) {
        config.residual_hash_bits =
            static_cast<size_t>(std::strtoull(argv[122], nullptr, 10));
    }
    if (argc > 123) {
        config.residual_hash_segments =
            static_cast<size_t>(std::strtoull(argv[123], nullptr, 10));
    }
    if (argc > 124) {
        config.residual_radius_full =
            static_cast<size_t>(std::strtoull(argv[124], nullptr, 10));
    }
    if (argc > 125) {
        config.residual_radius_segment =
            static_cast<size_t>(std::strtoull(argv[125], nullptr, 10));
    }
    if (argc > 126) {
        config.normalize_residual_before_hash =
            std::strtoull(argv[126], nullptr, 10) != 0;
    }
    if (argc > 127) {
        config.residual_hash_seed =
            static_cast<size_t>(std::strtoull(argv[127], nullptr, 10));
    }
    if (argc > 128) {
        config.residual_hash_filter_after =
            static_cast<size_t>(std::strtoull(argv[128], nullptr, 10));
    }
    if (argc > 129) {
        config.use_residual_hash_bucket_graph =
            std::strtoull(argv[129], nullptr, 10) != 0;
    }
    if (argc > 130) {
        config.residual_hash_hot_count =
            static_cast<size_t>(std::strtoull(argv[130], nullptr, 10));
    }
    if (argc > 131) {
        config.residual_hash_bucket_count =
            static_cast<size_t>(std::strtoull(argv[131], nullptr, 10));
    }
    if (argc > 132) {
        config.residual_hash_bucket_take =
            static_cast<size_t>(std::strtoull(argv[132], nullptr, 10));
    }
    if (argc > 133) {
        config.residual_hash_bucket_probe =
            static_cast<size_t>(std::strtoull(argv[133], nullptr, 10));
    }
    if (argc > 134) {
        config.residual_hash_fallback_min =
            static_cast<size_t>(std::strtoull(argv[134], nullptr, 10));
    }
    if (argc > 135) {
        config.residual_hash_cold_count =
            static_cast<size_t>(std::strtoull(argv[135], nullptr, 10));
    }
    if (argc > 136) {
        config.residual_hash_build_query_buckets =
            std::strtoull(argv[136], nullptr, 10) != 0;
    }
    if (argc > 137) {
        config.center_super_level_scan = std::strtoull(argv[137], nullptr, 10) != 0;
    }
    if (argc > 138) {
        config.center_super_count =
            static_cast<size_t>(std::strtoull(argv[138], nullptr, 10));
    }
    if (argc > 139) {
        config.center_super_probe =
            static_cast<size_t>(std::strtoull(argv[139], nullptr, 10));
    }
    if (argc > 140) {
        config.center_super_overlap =
            static_cast<size_t>(std::strtoull(argv[140], nullptr, 10));
    }
    if (argc > 141) {
        config.residual_hash_node_local_bucket =
            std::strtoull(argv[141], nullptr, 10) != 0;
    }
    if (argc > 142) {
        config.graph_dual_scale_search = std::strtoull(argv[142], nullptr, 10) != 0;
    }
    if (argc > 143) {
        config.graph_dual_short_count =
            static_cast<size_t>(std::strtoull(argv[143], nullptr, 10));
    }
    if (argc > 144) {
        config.graph_dual_long_count =
            static_cast<size_t>(std::strtoull(argv[144], nullptr, 10));
    }
    if (argc > 145) {
        config.graph_dual_long_alpha = std::strtof(argv[145], nullptr);
    }
    if (argc > 146) {
        config.graph_dual_query_beta = std::strtof(argv[146], nullptr);
    }
    if (argc > 147) {
        config.graph_dual_long_no_improve_limit =
            static_cast<size_t>(std::strtoull(argv[147], nullptr, 10));
    }
    auto env_u64 = [](const char* name, size_t& value) {
        if (const char* raw = std::getenv(name)) {
            value = static_cast<size_t>(std::strtoull(raw, nullptr, 10));
        }
    };
    auto env_bool = [](const char* name, bool& value) {
        if (const char* raw = std::getenv(name)) {
            value = std::strtoull(raw, nullptr, 10) != 0;
        }
    };
    auto env_float = [](const char* name, float& value) {
        if (const char* raw = std::getenv(name)) {
            value = std::strtof(raw, nullptr);
        }
    };
    env_bool("ATMMG_PRINT_DETAILS", print_details);
    env_bool("ATMMG_GRAPH_DUAL_SCALE_SEARCH", config.graph_dual_scale_search);
    env_u64("ATMMG_GRAPH_DUAL_SHORT_COUNT", config.graph_dual_short_count);
    env_u64("ATMMG_GRAPH_DUAL_LONG_COUNT", config.graph_dual_long_count);
    env_float("ATMMG_GRAPH_DUAL_LONG_ALPHA", config.graph_dual_long_alpha);
    env_float("ATMMG_GRAPH_DUAL_QUERY_BETA", config.graph_dual_query_beta);
    env_u64(
        "ATMMG_GRAPH_DUAL_LONG_NO_IMPROVE_LIMIT",
        config.graph_dual_long_no_improve_limit
    );
    env_bool("ATMMG_USE_FAST_QUERY", use_fast_query);
    env_u64("ATMMG_N_CENTERS", config.n_centers);
    env_u64("ATMMG_GRAPH_DEGREE", config.graph_degree);
    env_u64("ATMMG_EF_SEARCH", config.ef_search);
    env_u64("ATMMG_GRAPH_SEARCH_NEIGHBOR_CAP", config.graph_search_neighbor_cap);
    env_u64("ATMMG_CENTER_TOPN_PROBE", config.center_topn_probe);
    env_u64("ATMMG_INIT_KEEP", config.init_keep);
    env_bool("ATMMG_GRAPH_BUILD_BRIDGE_EDGES", config.graph_build_bridge_edges);
    env_u64("ATMMG_GRAPH_BRIDGE_CENTER_NEIGHBORS", config.graph_bridge_center_neighbors);
    env_u64("ATMMG_GRAPH_BRIDGE_POINTS_PER_CENTER", config.graph_bridge_points_per_center);
    env_u64("ATMMG_GRAPH_BRIDGE_CANDIDATE_SCAN", config.graph_bridge_candidate_scan);
    env_bool("ATMMG_GRAPH_REORDER_BY_CENTER", config.graph_reorder_by_center);
    env_bool("ATMMG_GRAPH_POST_NND_REFINE", config.graph_post_nnd_refine);
    env_u64("ATMMG_GRAPH_POST_NND_ITERATIONS", config.graph_post_nnd_iterations);
    env_u64(
        "ATMMG_GRAPH_POST_NND_CANDIDATE_LIMIT",
        config.graph_post_nnd_candidate_limit
    );
    env_float("ATMMG_GRAPH_POST_NND_ALPHA", config.graph_post_nnd_alpha);
    env_bool(
        "ATMMG_GRAPH_POST_NND_PRESERVE_DEGREE",
        config.graph_post_nnd_preserve_degree
    );
    env_u64("ATMMG_GRAPH_QUERY_ADAPTIVE_EF_MIN", config.graph_query_adaptive_ef_min);
    env_u64("ATMMG_GRAPH_QUERY_ADAPTIVE_EF_MID", config.graph_query_adaptive_ef_mid);
    env_bool("ATMMG_HARD_QUERY_FALLBACK", config.hard_query_fallback);
    env_float("ATMMG_HARD_QUERY_CENTER_MARGIN", config.hard_query_center_margin);
    env_u64("ATMMG_HARD_QUERY_INIT_KEEP", config.hard_query_init_keep);
    env_u64("ATMMG_HARD_QUERY_NEIGHBOR_CAP", config.hard_query_neighbor_cap);
    env_u64("ATMMG_HARD_QUERY_LATE_NEIGHBOR_CAP", config.hard_query_late_neighbor_cap);
    env_u64("ATMMG_HARD_QUERY_LATE_NEIGHBOR_AFTER", config.hard_query_late_neighbor_after);
    env_u64("ATMMG_HARD_QUERY_RERANK_CANDIDATES", config.hard_query_rerank_candidates);
    env_bool("ATMMG_CENTER_SUPER_LEVEL_SCAN", config.center_super_level_scan);
    env_u64("ATMMG_CENTER_SUPER_COUNT", config.center_super_count);
    env_u64("ATMMG_CENTER_SUPER_PROBE", config.center_super_probe);
    env_u64("ATMMG_CENTER_SUPER_OVERLAP", config.center_super_overlap);
    env_u64("ATMMG_CENTER_TOPN_SCAN", config.center_topn_scan);
    env_u64("ATMMG_CENTER_TOPN_COARSE_KEEP", config.center_topn_coarse_keep);
    env_u64("ATMMG_CENTER_SCAN_KEEP", config.center_scan_keep);
    env_u64("ATMMG_EXACT_CENTER_KEEP", config.exact_center_keep);
    env_u64("ATMMG_GRAPH_BUILD_INTRA_CANDIDATES", config.graph_build_intra_candidates);
    env_u64("ATMMG_GRAPH_BUILD_CROSS_CANDIDATES", config.graph_build_cross_candidates);
    env_u64("ATMMG_GRAPH_LATE_NEIGHBOR_CAP", config.graph_late_neighbor_cap);
    env_u64("ATMMG_GRAPH_LATE_NEIGHBOR_AFTER", config.graph_late_neighbor_after);
    env_bool("ATMMG_GRAPH_EARLY_STOP", config.graph_early_stop);
    env_u64("ATMMG_GRAPH_EARLY_STOP_MIN_EXPANSIONS", config.graph_early_stop_min_expansions);
    env_float("ATMMG_GRAPH_EARLY_STOP_SLACK", config.graph_early_stop_slack);
    force_mainline_only(config);
    using Clock = std::chrono::high_resolution_clock;
    ATMMG::Index index(config);
    const auto& effective_config = index.config();

    auto create_begin = Clock::now();
    index.construct(train.data.data(), base_limit, train.cols);
    auto create_end = Clock::now();

    ATMMG::graph::BatchStats batch_stats;
    size_t annb_correct = 0;
    size_t annb_oracle_correct = 0;
    size_t overlap_correct = 0;

    std::vector<std::vector<ATMMG::PID>> exact_results(query_ids.size());
    std::vector<float> exact_threshold_d2(query_ids.size(), 0.0F);
    bool recompute_groundtruth = base_limit < train.rows;
    env_bool("ATMMG_RECOMPUTE_GT", recompute_groundtruth);
    auto eval_begin = Clock::now();
    for (size_t qi = 0; qi < query_ids.size(); ++qi) {
        size_t qid = query_ids[qi];
        bool use_exact_for_query = recompute_groundtruth;
        if (!use_exact_for_query) {
            for (size_t j = 0; j < std::min(k_eval, neighbors.cols); ++j) {
                if (static_cast<size_t>(neighbors(qid, j)) >= base_limit) {
                    use_exact_for_query = true;
                    break;
                }
            }
        }

        if (use_exact_for_query) {
            exact_results[qi] = exact_topk(
                train.data.data(),
                base_limit,
                train.cols,
                test.row(qid),
                k_eval,
                &exact_threshold_d2[qi]
            );
        } else {
            exact_results[qi].reserve(k_eval);
            size_t truth_k = std::min(k_eval, neighbors.cols);
            for (size_t j = 0; j < truth_k; ++j) {
                exact_results[qi].push_back(static_cast<ATMMG::PID>(neighbors(qid, j)));
            }
            float kth_dist = distances(qid, std::min(k_eval, distances.cols) - 1);
            exact_threshold_d2[qi] = kth_dist * kth_dist;
        }
        annb_oracle_correct += annb_recall_count_by_threshold_d2(
            exact_results[qi],
            train.data.data(),
            train.cols,
            test.row(qid),
            exact_threshold_d2[qi]
        );
    }
    auto eval_end = Clock::now();

    std::vector<SweepPoint> sweep_points =
        parse_sweep_points(std::getenv("ATMMG_SWEEP"));
    if (!sweep_points.empty()) {
        double create_ms =
            std::chrono::duration<double, std::milli>(create_end - create_begin)
                .count();
        double eval_exact_ms =
            std::chrono::duration<double, std::milli>(eval_end - eval_begin)
                .count();
        double eval_exact_avg_ms =
            eval_queries == 0 ? 0.0 : eval_exact_ms / static_cast<double>(eval_queries);
        double denom = static_cast<double>(eval_queries * k_eval);
        double annb_oracle_recall =
            denom == 0 ? 0.0 : static_cast<double>(annb_oracle_correct) / denom;

        std::cout << "run info\n";
        std::cout << "train_rows=" << train.rows << '\n';
        std::cout << "test_rows=" << test.rows << '\n';
        std::cout << "base_limit=" << base_limit << '\n';
        std::cout << "eval_queries=" << eval_queries << '\n';
        std::cout << "recompute_groundtruth=" << (recompute_groundtruth ? 1 : 0)
                  << '\n';
        std::cout << "query_ids_first=";
        for (size_t i = 0; i < std::min<size_t>(query_ids.size(), 8); ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << query_ids[i];
        }
        std::cout << '\n';
        std::cout << "create=" << create_ms << " ms\n";
        std::cout << "eval_exact=" << eval_exact_ms << " ms\n";
        std::cout << "eval_exact_avg=" << eval_exact_avg_ms << " ms\n";
        std::cout << "query_fast_path=" << (use_fast_query ? 1 : 0) << '\n';
        std::cout << "sweep=1\n";
        std::cout
            << "sweep_csv=ef_search,effective_rerank_candidates,"
               "graph_search_neighbor_cap,query_ms,query_avg_ms,qps,"
               "annb_recall,overlap_recall,annb_oracle_recall\n";

        for (const SweepPoint& point : sweep_points) {
            size_t cap = point.graph_search_neighbor_cap;
            index.set_search_params(point.ef_search, k_eval, cap);

            ATMMG::graph::BatchStats sweep_stats;
            size_t sweep_annb_correct = 0;
            size_t sweep_overlap_correct = 0;
            std::vector<ATMMG::PID> fast_result_buffer(k_eval);
            double sweep_search_ms = 0.0;
            auto query_begin = Clock::now();
            for (size_t qi = 0; qi < query_ids.size(); ++qi) {
                size_t qid = query_ids[qi];
                if (use_fast_query) {
                    fast_result_buffer.resize(k_eval);
                    auto search_begin = Clock::now();
                    size_t got =
                        index.search_into(test.row(qid), k_eval, fast_result_buffer.data());
                    auto search_end = Clock::now();
                    sweep_search_ms +=
                        std::chrono::duration<double, std::milli>(
                            search_end - search_begin
                        )
                            .count();
                    fast_result_buffer.resize(got);
                    sweep_annb_correct += annb_recall_count_by_threshold_d2(
                        fast_result_buffer, train.data.data(), train.cols, test.row(qid),
                        exact_threshold_d2[qi]
                    );
                    sweep_overlap_correct +=
                        overlap_count(fast_result_buffer, exact_results[qi]);
                } else {
                    ATMMG::QueryStats stats;
                    auto search_begin = Clock::now();
                    std::vector<ATMMG::PID> result =
                        index.search(test.row(qid), k_eval, &stats);
                    auto search_end = Clock::now();
                    sweep_search_ms +=
                        std::chrono::duration<double, std::milli>(
                            search_end - search_begin
                        )
                            .count();
                    sweep_stats.add(stats);
                    sweep_annb_correct += annb_recall_count_by_threshold_d2(
                        result,
                        train.data.data(),
                        train.cols,
                        test.row(qid),
                        exact_threshold_d2[qi]
                    );
                    sweep_overlap_correct += overlap_count(result, exact_results[qi]);
                }
            }
            auto query_end = Clock::now();
            double query_ms =
                std::chrono::duration<double, std::milli>(query_end - query_begin)
                    .count();
            double query_avg_ms =
                eval_queries == 0 ? 0.0 : query_ms / static_cast<double>(eval_queries);
            double annb_recall =
                denom == 0 ? 0.0 : static_cast<double>(sweep_annb_correct) / denom;
            double overlap_recall =
                denom == 0 ? 0.0 : static_cast<double>(sweep_overlap_correct) / denom;

            const auto& after = index.config();
            std::cout << after.ef_search << ',' << k_eval << ','
                      << after.graph_search_neighbor_cap << ',' << query_ms << ','
                      << query_avg_ms << ','
                      << (query_ms == 0.0 ? 0.0
                                           : eval_queries * 1000.0 / query_ms)
                      << ',' << annb_recall << ',' << overlap_recall << ','
                      << annb_oracle_recall << '\n';

            if (print_details && !use_fast_query) {
                std::cout << "sweep_detail_ef=" << after.ef_search << '\n';
                std::cout << "search_only_ms=" << sweep_search_ms << '\n';
                std::cout << "search_only_avg="
                          << (eval_queries == 0
                                  ? 0.0
                                  : sweep_search_ms / static_cast<double>(eval_queries))
                          << '\n';
                std::cout << "search_only_qps="
                          << (sweep_search_ms == 0.0
                                  ? 0.0
                                  : eval_queries * 1000.0 / sweep_search_ms)
                          << '\n';
                print_stat_avg_max("heap_pushes", sweep_stats.heap_pushes);
                print_stat_avg_max("heap_pops", sweep_stats.heap_pops);
                print_stat_avg_max("jumps", sweep_stats.jumps);
                print_stat_avg_max("visited_nodes", sweep_stats.visited_nodes);
                print_stat_avg_max("node_expansions", sweep_stats.node_expansions);
                print_stat_avg_max("edges_scanned", sweep_stats.edges_scanned);
                print_stat_avg_max(
                    "exact_distance_evals", sweep_stats.exact_distance_evals
                );
                print_stat_avg_max(
                    "graph_distance_evals", sweep_stats.graph_distance_evals
                );
                print_stat_avg_max("final_candidates", sweep_stats.final_candidates);
            } else if (print_details) {
                std::cout << "sweep_detail_ef=" << after.ef_search << '\n';
                std::cout << "search_only_ms=" << sweep_search_ms << '\n';
                std::cout << "search_only_avg="
                          << (eval_queries == 0
                                  ? 0.0
                                  : sweep_search_ms / static_cast<double>(eval_queries))
                          << '\n';
                std::cout << "search_only_qps="
                          << (sweep_search_ms == 0.0
                                  ? 0.0
                                  : eval_queries * 1000.0 / sweep_search_ms)
                          << '\n';
            }
        }
        return 0;
    }

    std::vector<ATMMG::PID> fast_result_buffer(k_eval);
    double search_only_ms = 0.0;
    auto query_begin = Clock::now();
    for (size_t qi = 0; qi < query_ids.size(); ++qi) {
        size_t qid = query_ids[qi];
        if (use_fast_query) {
            fast_result_buffer.resize(k_eval);
            auto search_begin = Clock::now();
            size_t got =
                index.search_into(test.row(qid), k_eval, fast_result_buffer.data());
            auto search_end = Clock::now();
            search_only_ms +=
                std::chrono::duration<double, std::milli>(search_end - search_begin)
                    .count();
            fast_result_buffer.resize(got);
            annb_correct += annb_recall_count_by_threshold_d2(
                fast_result_buffer, train.data.data(), train.cols, test.row(qid),
                exact_threshold_d2[qi]
            );
            overlap_correct += overlap_count(fast_result_buffer, exact_results[qi]);
        } else {
            ATMMG::QueryStats stats;
            auto search_begin = Clock::now();
            std::vector<ATMMG::PID> result = index.search(test.row(qid), k_eval, &stats);
            auto search_end = Clock::now();
            search_only_ms +=
                std::chrono::duration<double, std::milli>(search_end - search_begin)
                    .count();
            batch_stats.add(stats);
            annb_correct += annb_recall_count_by_threshold_d2(
                result, train.data.data(), train.cols, test.row(qid), exact_threshold_d2[qi]
            );
            overlap_correct += overlap_count(result, exact_results[qi]);
        }
    }
    auto query_end = Clock::now();

    double create_ms =
        std::chrono::duration<double, std::milli>(create_end - create_begin).count();
    double eval_exact_ms =
        std::chrono::duration<double, std::milli>(eval_end - eval_begin).count();
    double query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    double query_avg_ms =
        eval_queries == 0 ? 0.0 : query_ms / static_cast<double>(eval_queries);
    double eval_exact_avg_ms =
        eval_queries == 0 ? 0.0 : eval_exact_ms / static_cast<double>(eval_queries);

    double denom = static_cast<double>(eval_queries * k_eval);
    double annb_recall = denom == 0 ? 0.0 : static_cast<double>(annb_correct) / denom;
    double annb_oracle_recall =
        denom == 0 ? 0.0 : static_cast<double>(annb_oracle_correct) / denom;
    double overlap_recall = denom == 0 ? 0.0 : static_cast<double>(overlap_correct) / denom;

    std::cout << "run info\n";
    std::cout << "train_rows=" << train.rows << '\n';
    std::cout << "test_rows=" << test.rows << '\n';
    std::cout << "base_limit=" << base_limit << '\n';
    std::cout << "eval_queries=" << eval_queries << '\n';
    std::cout << "recompute_groundtruth=" << (recompute_groundtruth ? 1 : 0) << '\n';
    std::cout << "create=" << create_ms << " ms\n";
    std::cout << "query=" << query_ms << " ms\n";
    std::cout << "query_avg=" << query_avg_ms << " ms\n";
    std::cout << "search_only=" << search_only_ms << " ms\n";
    std::cout << "search_only_avg="
              << (eval_queries == 0 ? 0.0
                                     : search_only_ms / static_cast<double>(eval_queries))
              << " ms\n";
    std::cout << "search_only_qps="
              << (search_only_ms == 0.0 ? 0.0 : eval_queries * 1000.0 / search_only_ms)
              << '\n';
    std::cout << "eval_exact=" << eval_exact_ms << " ms\n";
    std::cout << "eval_exact_avg=" << eval_exact_avg_ms << " ms\n";
    std::cout << "qps=" << (query_ms == 0.0 ? 0.0 : eval_queries * 1000.0 / query_ms) << '\n';
    std::cout << "query_fast_path=" << (use_fast_query ? 1 : 0) << '\n';

    if (print_details && !use_fast_query) {
        std::cout << "search summary\n";
        print_stat_avg_max("entry_points_raw", batch_stats.entry_points_raw);
        print_stat_avg_max("entry_points_before_cap", batch_stats.entry_points_before_cap);
        print_stat_avg_max("entry_points_after_cap", batch_stats.entry_points_after_cap);
        print_stat_avg_max("init_size", batch_stats.init_sizes);
        print_stat_avg_max("heap_pushes", batch_stats.heap_pushes);
        print_stat_avg_max("heap_pops", batch_stats.heap_pops);
        print_stat_avg_max("visited_nodes", batch_stats.visited_nodes);
        print_stat_avg_max("node_expansions", batch_stats.node_expansions);
        print_stat_avg_max("edges_scanned", batch_stats.edges_scanned);
        print_stat_avg_max("exact_distance_evals", batch_stats.exact_distance_evals);
        print_stat_avg_max("graph_distance_evals", batch_stats.graph_distance_evals);
        print_stat_avg_max("final_candidates", batch_stats.final_candidates);
        print_stat_avg_max_if_nonzero("graph_adaptive_ef", batch_stats.graph_adaptive_ef);
        print_stat_avg_max_if_nonzero(
            "graph_adaptive_ef_stop", batch_stats.graph_adaptive_ef_stop
        );
        print_stat_avg_max_if_nonzero(
            "graph_query_ef_budget", batch_stats.graph_query_ef_budget
        );
        print_stat_avg_max_if_nonzero("graph_prefilter_evals", batch_stats.graph_prefilter_evals);
        print_stat_avg_max_if_nonzero(
            "graph_result_margin_stop_count", batch_stats.graph_result_margin_stop_count
        );
        print_stat_avg_max_if_nonzero(
            "graph_admission_rejects", batch_stats.graph_admission_rejects
        );
        print_stat_avg_max_if_nonzero("graph_cold_expansions", batch_stats.graph_cold_expansions);
        print_stat_avg_max_if_nonzero(
            "graph_cold_edges_scanned", batch_stats.graph_cold_edges_scanned
        );
        std::cout << "trigger_pass_count=" << batch_stats.trigger_pass_count << '\n';
        std::cout << "time breakdown\n";
        print_time_avg_max("query_total", batch_stats.query_total_ms);
        print_time_avg_max("center_refine", batch_stats.center_refine_ms);
        print_time_avg_max("seed_select", batch_stats.seed_select_ms);
        print_time_avg_max("graph_search", batch_stats.graph_search_ms);
        print_time_avg_max("final_select", batch_stats.final_select_ms);
    }

    if (print_details) {
        std::cout << "index summary\n";
        std::cout << "dim=" << index.dim() << '\n';
        std::cout << "n_centers=" << index.num_centers() << '\n';
        std::cout << "center_cascade_low_keep="
                  << effective_config.center_cascade_low_keep << '\n';
        std::cout << "center_cascade_mid_keep="
                  << effective_config.center_cascade_mid_keep << '\n';
        std::cout << "center_entry_mode_name="
                  << center_entry_mode_name(effective_config.center_entry_mode) << '\n';
        std::cout << "center_topn_scan="
                  << effective_config.center_topn_scan << '\n';
        std::cout << "center_topn_probe="
                  << effective_config.center_topn_probe << '\n';
        std::cout << "center_topn_coarse_keep="
                  << effective_config.center_topn_coarse_keep << '\n';
        std::cout << "init_keep=" << effective_config.init_keep << '\n';
        std::cout << "use_hash_neighborhood_spectrum="
                  << (effective_config.use_hash_neighborhood_spectrum ? 1 : 0) << '\n';
        std::cout << "hash_spectrum_size="
                  << effective_config.hash_spectrum_size << '\n';
        std::cout << "hash_spectrum_pool_scan="
                  << effective_config.hash_spectrum_pool_scan << '\n';
        std::cout << "hash_spectrum_bits="
                  << effective_config.hash_spectrum_bits << '\n';
        std::cout << "hash_spectrum_min_hamming="
                  << effective_config.hash_spectrum_min_hamming << '\n';
        std::cout << "hash_spectrum_candidates="
                  << effective_config.hash_spectrum_candidates << '\n';
        std::cout << "hash_spectrum_entry_take="
                  << effective_config.hash_spectrum_entry_take << '\n';
        std::cout << "hash_spectrum_center_keep="
                  << effective_config.hash_spectrum_center_keep << '\n';
        std::cout << "use_residual_hash_spectrum="
                  << (effective_config.use_residual_hash_spectrum ? 1 : 0) << '\n';
        std::cout << "residual_hash_bits="
                  << effective_config.residual_hash_bits << '\n';
        std::cout << "residual_hash_segments="
                  << effective_config.residual_hash_segments << '\n';
        std::cout << "residual_radius_full="
                  << effective_config.residual_radius_full << '\n';
        std::cout << "residual_radius_segment="
                  << effective_config.residual_radius_segment << '\n';
        std::cout << "residual_hash_filter_after="
                  << effective_config.residual_hash_filter_after << '\n';
        std::cout << "use_residual_hash_bucket_graph="
                  << (effective_config.use_residual_hash_bucket_graph ? 1 : 0)
                  << '\n';
        std::cout << "residual_hash_hot_count="
                  << effective_config.residual_hash_hot_count << '\n';
        std::cout << "residual_hash_bucket_count="
                  << effective_config.residual_hash_bucket_count << '\n';
        std::cout << "residual_hash_bucket_take="
                  << effective_config.residual_hash_bucket_take << '\n';
        std::cout << "residual_hash_bucket_probe="
                  << effective_config.residual_hash_bucket_probe << '\n';
        std::cout << "residual_hash_fallback_min="
                  << effective_config.residual_hash_fallback_min << '\n';
        std::cout << "residual_hash_cold_count="
                  << effective_config.residual_hash_cold_count << '\n';
        std::cout << "residual_hash_build_query_buckets="
                  << (effective_config.residual_hash_build_query_buckets ? 1 : 0)
                  << '\n';
        std::cout << "residual_hash_node_local_bucket="
                  << (effective_config.residual_hash_node_local_bucket ? 1 : 0)
                  << '\n';
        std::cout << "residual_hash_bucket_edges="
                  << index.residual_hash_bucket_edges() << '\n';
        std::cout << "center_residual_spectrum_compat=1" << '\n';
        std::cout << "center_residual_spectrum_note=uses_existing_residual_hash_bucket_args" << '\n';
        std::cout << "center_super_level_scan="
                  << (effective_config.center_super_level_scan ? 1 : 0) << '\n';
        std::cout << "center_super_count="
                  << effective_config.center_super_count << '\n';
        std::cout << "center_super_probe="
                  << effective_config.center_super_probe << '\n';
        std::cout << "center_super_overlap="
                  << effective_config.center_super_overlap << '\n';
        std::cout << "graph_degree=" << effective_config.graph_degree << '\n';
        std::cout << "graph_query_front_prune="
                  << (effective_config.graph_query_front_prune ? 1 : 0) << '\n';
        std::cout << "graph_edges=" << index.graph_edges() << '\n';
        std::cout << "avg_graph_degree=" << index.avg_graph_degree() << '\n';
        std::cout << "graph_post_nnd_refine="
                  << (effective_config.graph_post_nnd_refine ? 1 : 0) << '\n';
        std::cout << "graph_post_nnd_iterations="
                  << effective_config.graph_post_nnd_iterations << '\n';
        std::cout << "graph_post_nnd_candidate_limit="
                  << effective_config.graph_post_nnd_candidate_limit << '\n';
        std::cout << "graph_post_nnd_alpha="
                  << effective_config.graph_post_nnd_alpha << '\n';
        std::cout << "graph_post_nnd_preserve_degree="
                  << (effective_config.graph_post_nnd_preserve_degree ? 1 : 0)
                  << '\n';
        std::cout << "graph_post_nnd_edges_before="
                  << index.graph_post_nnd_edges_before() << '\n';
        std::cout << "graph_post_nnd_edges_after="
                  << index.graph_post_nnd_edges_after() << '\n';
        std::cout << "graph_post_nnd_candidate_total="
                  << index.graph_post_nnd_candidate_total() << '\n';
        std::cout << "graph_post_nnd_candidate_sources="
                  << index.graph_post_nnd_candidate_sources() << '\n';
        std::cout << "ef_search=" << effective_config.ef_search << '\n';
        std::cout << "graph_rerank_candidates=" << k_eval << '\n';
        std::cout << "graph_search_neighbor_cap="
                  << effective_config.graph_search_neighbor_cap << '\n';
        std::cout << "graph_dual_scale_search="
                  << (effective_config.graph_dual_scale_search ? 1 : 0) << '\n';
        std::cout << "graph_dual_short_count="
                  << effective_config.graph_dual_short_count << '\n';
        std::cout << "graph_dual_long_count="
                  << effective_config.graph_dual_long_count << '\n';
        std::cout << "graph_dual_long_alpha="
                  << effective_config.graph_dual_long_alpha << '\n';
        std::cout << "graph_dual_query_beta="
                  << effective_config.graph_dual_query_beta << '\n';
        std::cout << "graph_dual_long_no_improve_limit="
                  << effective_config.graph_dual_long_no_improve_limit << '\n';
        std::cout << "graph_late_neighbor_cap="
                  << effective_config.graph_late_neighbor_cap << '\n';
        std::cout << "graph_late_neighbor_after="
                  << effective_config.graph_late_neighbor_after << '\n';
        std::cout << "graph_query_adaptive_ef_min="
                  << effective_config.graph_query_adaptive_ef_min << '\n';
        std::cout << "graph_query_adaptive_ef_mid="
                  << effective_config.graph_query_adaptive_ef_mid << '\n';
        std::cout << "center_adaptive_refine="
                  << (effective_config.center_adaptive_refine ? 1 : 0) << '\n';
        std::cout << "center_adaptive_coarse_keep_easy="
                  << effective_config.center_adaptive_coarse_keep_easy << '\n';
        std::cout << "center_adaptive_coarse_keep_mid="
                  << effective_config.center_adaptive_coarse_keep_mid << '\n';
        std::cout << "center_adaptive_scan_keep_easy="
                  << effective_config.center_adaptive_scan_keep_easy << '\n';
        std::cout << "center_adaptive_scan_keep_mid="
                  << effective_config.center_adaptive_scan_keep_mid << '\n';
        std::cout << "hard_query_fallback="
                  << (effective_config.hard_query_fallback ? 1 : 0) << '\n';
        std::cout << "hard_query_center_margin="
                  << effective_config.hard_query_center_margin << '\n';
        std::cout << "hard_query_init_keep="
                  << effective_config.hard_query_init_keep << '\n';
        std::cout << "hard_query_neighbor_cap="
                  << effective_config.hard_query_neighbor_cap << '\n';
        std::cout << "hard_query_late_neighbor_cap="
                  << effective_config.hard_query_late_neighbor_cap << '\n';
        std::cout << "hard_query_late_neighbor_after="
                  << effective_config.hard_query_late_neighbor_after << '\n';
        std::cout << "hard_query_rerank_candidates="
                  << effective_config.hard_query_rerank_candidates << '\n';
    }

    std::cout << "ANNB recall@" << k_eval << "=" << annb_recall << '\n';
    std::cout << "ANNB oracle@" << k_eval << "=" << annb_oracle_recall << '\n';
    std::cout << "Overlap recall@" << k_eval << "=" << overlap_recall << '\n';

    return 0;
}

int main(int argc, char** argv) {
    try {
        return run_ATMMG_hdf5(argc, argv);
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getDetailMsg() << '\n';
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
    }
    return 1;
}
