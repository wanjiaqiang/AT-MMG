#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "ATMMG/ATMMG.hpp"
#include "ATMMG/defines.hpp"
#include "ATMMG/utils/io.hpp"

namespace {

using data_type = ATMMG::RowMajorArray<float>;
using gt_type = ATMMG::RowMajorArray<int32_t>;

struct SweepPoint {
    size_t ef_search = 0;
    size_t ignored_rerank_candidates = 0;
    size_t graph_search_neighbor_cap = 0;
};

struct RunMetrics {
    double query_ms = 0.0;
    double query_avg_ms = 0.0;
    double qps = 0.0;
    double recall_at_10 = 0.0;
    bool has_recall = false;
};

bool parse_size_arg(const char* raw, size_t& value) {
    if (raw == nullptr || *raw == '\0') {
        return false;
    }
    char* end = nullptr;
    size_t parsed = static_cast<size_t>(std::strtoull(raw, &end, 10));
    if (end == raw || *end != '\0') {
        return false;
    }
    value = parsed;
    return true;
}

size_t read_optional_size_arg(int argc, char** argv, int& arg, size_t default_value) {
    size_t value = 0;
    if (arg < argc && parse_size_arg(argv[arg], value)) {
        ++arg;
        return value;
    }
    return default_value;
}

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

bool csv_disabled(const std::string& value) {
    return value.empty() || value == "0" || value == "off" || value == "none";
}

std::string output_csv_path() {
    const char* raw = std::getenv("ATMMG_OUTPUT_CSV");
    if (raw == nullptr) {
        return {};
    }
    std::string path(raw);
    return csv_disabled(path) ? std::string{} : path;
}

size_t count_recall_hits(
    const gt_type& groundtruth,
    size_t qid,
    const ATMMG::PID* result,
    size_t result_count,
    size_t k
) {
    size_t hits = 0;
    for (size_t gi = 0; gi < k; ++gi) {
        int32_t gt_id = groundtruth(static_cast<long>(qid), static_cast<long>(gi));
        if (gt_id < 0) {
            continue;
        }
        for (size_t ri = 0; ri < result_count; ++ri) {
            if (static_cast<int64_t>(result[ri]) == static_cast<int64_t>(gt_id)) {
                ++hits;
                break;
            }
        }
    }
    return hits;
}

RunMetrics run_queries(
    const ATMMG::Index& index,
    const data_type& queries,
    const std::vector<size_t>& query_ids,
    const gt_type* groundtruth,
    bool use_fast_query,
    size_t k_eval
) {
    using Clock = std::chrono::high_resolution_clock;
    RunMetrics metrics;
    metrics.has_recall = groundtruth != nullptr;

    std::vector<ATMMG::PID> result(k_eval);
    size_t recall_hits = 0;
    const size_t dim = static_cast<size_t>(queries.cols());

    auto query_begin = Clock::now();
    for (size_t qi = 0; qi < query_ids.size(); ++qi) {
        size_t qid = query_ids[qi];
        const float* query = queries.data() + (qid * dim);
        size_t result_count = 0;
        if (use_fast_query) {
            result_count = index.search_into(query, k_eval, result.data());
        } else {
            auto ids = index.search(query, k_eval, nullptr);
            result_count = ids.size();
            result.assign(ids.begin(), ids.end());
        }
        if (groundtruth != nullptr) {
            recall_hits +=
                count_recall_hits(*groundtruth, qid, result.data(), result_count, k_eval);
        }
    }
    auto query_end = Clock::now();

    metrics.query_ms =
        std::chrono::duration<double, std::milli>(query_end - query_begin).count();
    metrics.query_avg_ms = query_ids.empty()
                               ? 0.0
                               : metrics.query_ms / static_cast<double>(query_ids.size());
    metrics.qps =
        metrics.query_ms == 0.0 ? 0.0 : query_ids.size() * 1000.0 / metrics.query_ms;
    if (groundtruth != nullptr) {
        double denom = static_cast<double>(query_ids.size() * k_eval);
        metrics.recall_at_10 = denom == 0.0 ? 0.0 : static_cast<double>(recall_hits) / denom;
    }
    return metrics;
}

void write_csv_header(std::ostream& out) {
    out << "ef_search,effective_rerank_candidates,graph_search_neighbor_cap,"
           "recall@10,qps,query_ms,query_avg_ms\n";
}

void write_csv_row(
    std::ostream& out,
    const ATMMG::Config& config,
    const RunMetrics& metrics,
    size_t k_eval
) {
    out << config.ef_search << ',' << k_eval << ','
        << config.graph_search_neighbor_cap << ',' << std::fixed << std::setprecision(8)
        << metrics.recall_at_10 << ',' << std::setprecision(3) << metrics.qps << ','
        << metrics.query_ms << ',' << metrics.query_avg_ms << '\n';
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

void apply_env_overrides(ATMMG::Config& config, bool& print_details) {
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
    env_u64("ATMMG_N_CENTERS", config.n_centers);
    env_u64("ATMMG_GRAPH_DEGREE", config.graph_degree);
    env_u64("ATMMG_EF_SEARCH", config.ef_search);
    env_u64("ATMMG_GRAPH_SEARCH_NEIGHBOR_CAP", config.graph_search_neighbor_cap);
    env_u64("ATMMG_CENTER_TOPN_SCAN", config.center_topn_scan);
    env_u64("ATMMG_CENTER_TOPN_PROBE", config.center_topn_probe);
    env_u64("ATMMG_CENTER_TOPN_COARSE_KEEP", config.center_topn_coarse_keep);
    env_u64("ATMMG_INIT_KEEP", config.init_keep);
    env_bool("ATMMG_GRAPH_BUILD_BRIDGE_EDGES", config.graph_build_bridge_edges);
    env_u64("ATMMG_GRAPH_BRIDGE_CENTER_NEIGHBORS", config.graph_bridge_center_neighbors);
    env_u64("ATMMG_GRAPH_BRIDGE_POINTS_PER_CENTER", config.graph_bridge_points_per_center);
    env_u64("ATMMG_GRAPH_BRIDGE_CANDIDATE_SCAN", config.graph_bridge_candidate_scan);
    env_bool("ATMMG_GRAPH_QUERY_ADJACENCY_ORDER", config.graph_query_adjacency_order);
    env_bool("ATMMG_GRAPH_QUERY_FRONT_PRUNE", config.graph_query_front_prune);
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
    config.center_topn_scan = 0;
}

void apply_common_args(
    int argc, char** argv, ATMMG::Config& config, bool& use_fast_query
) {
    if (argc > 5) {
        config.ef_search = static_cast<size_t>(std::strtoull(argv[5], nullptr, 10));
    }
    if (argc > 6) {
        config.center_topn_coarse_keep = static_cast<size_t>(std::strtoull(argv[6], nullptr, 10));
    }
    // argv[7] used to configure graph reranking; rerank count now equals query k.
    // argv[8] used to toggle a removed legacy search branch.
    if (argc > 9) {
        config.center_topn_scan = static_cast<size_t>(std::strtoull(argv[9], nullptr, 10));
    }
    if (argc > 10) {
        config.graph_early_stop = std::strtoull(argv[10], nullptr, 10) != 0;
    }
    if (argc > 11) {
        config.graph_search_neighbor_cap = static_cast<size_t>(std::strtoull(argv[11], nullptr, 10));
    }
    if (argc > 12) {
        config.center_topn_probe = static_cast<size_t>(std::strtoull(argv[12], nullptr, 10));
    }
    if (argc > 13) {
        config.graph_lazy_center_distance = std::strtoull(argv[13], nullptr, 10) != 0;
    }
    if (argc > 14) {
        config.n_centers = static_cast<size_t>(std::strtoull(argv[14], nullptr, 10));
    }
    if (argc > 15) {
        config.center_leaf_min_size = static_cast<size_t>(std::strtoull(argv[15], nullptr, 10));
    }
    if (argc > 16) {
        config.center_scan_keep = static_cast<size_t>(std::strtoull(argv[16], nullptr, 10));
    }
    if (argc > 17) {
        config.exact_center_keep = static_cast<size_t>(std::strtoull(argv[17], nullptr, 10));
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
        config.graph_query_adjacency_order = std::strtoull(argv[37], nullptr, 10) != 0;
    }
    if (argc > 38) {
        config.graph_reorder_by_center = std::strtoull(argv[38], nullptr, 10) != 0;
    }
    if (argc > 39) {
        config.center_coarse_projection_dims =
            static_cast<size_t>(std::strtoull(argv[39], nullptr, 10));
    }
    if (argc > 40) {
        config.center_coarse_keep = static_cast<size_t>(std::strtoull(argv[40], nullptr, 10));
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
    if (argc > 51) {
        config.graph_neighbor_prefilter_dims =
            static_cast<size_t>(std::strtoull(argv[51], nullptr, 10));
    }
    if (argc > 52) {
        config.graph_neighbor_prefilter_keep =
            static_cast<size_t>(std::strtoull(argv[52], nullptr, 10));
    }
}

}  // namespace

int main(int argc, char** argv) {
    int arg = 1;
    std::string base_path = arg < argc ? argv[arg++] : "msong_base.fvecs";
    std::string query_path = arg < argc ? argv[arg++] : "msong_query.fvecs";
    std::string groundtruth_path;
    if (const char* env_gt = std::getenv("ATMMG_GROUNDTRUTH")) {
        groundtruth_path = env_gt;
    }

    size_t ignored = 0;
    if (arg < argc && !parse_size_arg(argv[arg], ignored)) {
        groundtruth_path = argv[arg++];
    }
    size_t base_limit = read_optional_size_arg(argc, argv, arg, 0);
    size_t eval_queries = read_optional_size_arg(argc, argv, arg, 1000);
    size_t random_seed = read_optional_size_arg(argc, argv, arg, 42);
    if (arg < argc && !parse_size_arg(argv[arg], ignored)) {
        groundtruth_path = argv[arg++];
    }
    constexpr size_t k_eval = 10;

    data_type base;
    data_type queries;
    gt_type groundtruth;
    ATMMG::load_vecs<float, data_type>(base_path.c_str(), base);
    ATMMG::load_vecs<float, data_type>(query_path.c_str(), queries);
    bool has_groundtruth = !groundtruth_path.empty();
    if (has_groundtruth) {
        ATMMG::load_vecs<int32_t, gt_type>(groundtruth_path.c_str(), groundtruth);
        if (groundtruth.rows() < queries.rows()) {
            throw std::runtime_error("groundtruth rows must cover query rows");
        }
        if (groundtruth.cols() < static_cast<long>(k_eval)) {
            throw std::runtime_error("groundtruth must contain at least 10 neighbors per query");
        }
    }
    if (base.cols() != queries.cols()) {
        throw std::runtime_error("base/query dimensions do not match");
    }
    if (base_limit == 0 || base_limit > static_cast<size_t>(base.rows())) {
        base_limit = static_cast<size_t>(base.rows());
    }
    eval_queries = std::min(eval_queries, static_cast<size_t>(queries.rows()));

    std::vector<size_t> query_ids(static_cast<size_t>(queries.rows()));
    std::iota(query_ids.begin(), query_ids.end(), static_cast<size_t>(0));
    std::mt19937 rng(static_cast<uint32_t>(random_seed));
    std::shuffle(query_ids.begin(), query_ids.end(), rng);
    query_ids.resize(eval_queries);

    ATMMG::Config config;
    config.n_centers = 2048;
    config.center_leaf_min_size = 16;
    config.center_scan_keep = 32;
    config.exact_center_keep = 6;
    config.center_refine_neighbor_scan = 768;
    config.center_coarse_projection_dims = 24;
    config.center_coarse_keep = 512;
    config.center_entry_mode = ATMMG::CenterEntryMode::TreeOnly;
    config.center_topn_scan = 0;
    config.center_topn_coarse_keep = 64;
    config.init_keep = 32;
    config.graph_degree = 48;
    config.ef_search = 48;
    config.graph_search_neighbor_cap = 24;
    config.graph_build_bridge_edges = true;
    config.graph_bridge_center_neighbors = 4;
    config.graph_bridge_points_per_center = 4;
    config.graph_bridge_candidate_scan = 128;
    config.graph_query_adjacency_order = true;
    config.graph_query_front_prune = true;
    config.graph_reorder_by_center = true;
    config.graph_insert_new_degree = 24;
    config.graph_build_intra_candidates = 384;
    config.graph_build_cross_candidates = 256;
    config.graph_build_projection_dims = 6;
    config.graph_build_center_neighbors = 8;
    config.random_seed = random_seed;

    bool use_fast_query = true;
    bool print_details = false;
    // Reuse the HDF5 sample's positional parameter layout after the dataset path.
    // fvecs has two paths, plus optional GT, so align the first remaining knob to argv[5].
    if (arg < argc) {
        int common_shift = std::max(0, arg - 5);
        apply_common_args(argc - common_shift, argv + common_shift, config, use_fast_query);
    }
    apply_env_overrides(config, print_details);
    force_mainline_only(config);

    using Clock = std::chrono::high_resolution_clock;
    ATMMG::Index index(config);

    auto create_begin = Clock::now();
    index.construct(base.data(), base_limit, static_cast<size_t>(base.cols()));
    auto create_end = Clock::now();

    double create_ms =
        std::chrono::duration<double, std::milli>(create_end - create_begin).count();

    std::vector<SweepPoint> sweep_points =
        parse_sweep_points(std::getenv("ATMMG_SWEEP"));
    std::string csv_path = output_csv_path();
    std::ofstream csv_file;
    if (!csv_path.empty()) {
        csv_file.open(csv_path);
        if (!csv_file) {
            throw std::runtime_error("failed to open ATMMG_OUTPUT_CSV path");
        }
        write_csv_header(csv_file);
    }

    std::cout << "run info\n";
    std::cout << "base_rows=" << base.rows() << '\n';
    std::cout << "query_rows=" << queries.rows() << '\n';
    std::cout << "base_limit=" << base_limit << '\n';
    std::cout << "eval_queries=" << eval_queries << '\n';
    std::cout << "dim=" << base.cols() << '\n';
    std::cout << "groundtruth_path=" << (has_groundtruth ? groundtruth_path : "(none)") << '\n';
    std::cout << "create=" << create_ms << " ms\n";
    std::cout << "query_fast_path=" << (use_fast_query ? 1 : 0) << '\n';
    if (!csv_path.empty()) {
        std::cout << "output_csv=" << csv_path << '\n';
    }

    if (!sweep_points.empty()) {
        std::cout << "sweep=1\n";
        std::cout
            << "sweep_csv=ef_search,effective_rerank_candidates,"
               "graph_search_neighbor_cap,recall@10,qps,query_ms,query_avg_ms\n";

        for (const SweepPoint& point : sweep_points) {
            const auto& before = index.config();
            size_t cap = point.graph_search_neighbor_cap == 0
                             ? before.graph_search_neighbor_cap
                             : point.graph_search_neighbor_cap;
            index.set_search_params(point.ef_search, k_eval, cap);

            RunMetrics metrics = run_queries(
                index,
                queries,
                query_ids,
                has_groundtruth ? &groundtruth : nullptr,
                use_fast_query,
                k_eval
            );
            const auto& after = index.config();
            std::cout << after.ef_search << ',' << k_eval << ','
                      << after.graph_search_neighbor_cap << ',' << std::fixed
                      << std::setprecision(8) << metrics.recall_at_10 << ','
                      << std::setprecision(3) << metrics.qps << ',' << metrics.query_ms
                      << ',' << metrics.query_avg_ms << '\n';
            if (csv_file) {
                write_csv_row(csv_file, after, metrics, k_eval);
            }
        }
        return 0;
    }

    RunMetrics metrics = run_queries(
        index,
        queries,
        query_ids,
        has_groundtruth ? &groundtruth : nullptr,
        use_fast_query,
        k_eval
    );
    std::cout << "query_ms=" << metrics.query_ms << '\n';
    std::cout << "query=" << metrics.query_ms << " ms\n";
    std::cout << "query_avg_ms=" << metrics.query_avg_ms << '\n';
    std::cout << "query_avg=" << metrics.query_avg_ms << " ms\n";
    std::cout << "qps=" << metrics.qps << '\n';
    if (metrics.has_recall) {
        std::cout << "recall@10=" << metrics.recall_at_10 << '\n';
    }
    if (csv_file) {
        write_csv_row(csv_file, index.config(), metrics, k_eval);
    }

    if (print_details) {
        const auto& effective_config = index.config();
        std::cout << "index summary\n";
        std::cout << "n_centers=" << index.num_centers() << '\n';
        std::cout << "center_entry_mode_name="
                  << center_entry_mode_name(effective_config.center_entry_mode) << '\n';
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
        std::cout << "graph_search_neighbor_cap="
                  << effective_config.graph_search_neighbor_cap << '\n';
    }

    return 0;
}
