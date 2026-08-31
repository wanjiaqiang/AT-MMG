#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#endif

#include "ATMMG/defines.hpp"
#include "ATMMG/index/estimator.hpp"
#include "ATMMG/index/query.hpp"
#include "ATMMG/quantization/data_layout.hpp"
#include "ATMMG/quantization/scalar_quant.hpp"
#include "ATMMG/utils/rotator.hpp"
#include "ATMMG/utils/space.hpp"
#include "ATMMG/utils/tools.hpp"

namespace ATMMG::graph {

enum class CenterEntryMode : uint8_t {
    TreeThenQuant = 0,
    TreeOnly = 1,
    QuantOnly = 2,
    AnnoyTreeOnly = 3,
    ClusterTreeOnly = 4
};

struct ATMMGGraphConfig {
    size_t n_centers = 128;
    size_t center_leaf_min_size = 16;

    size_t center_scan_keep = 16;
    size_t exact_center_keep = 6;
    size_t center_refine_neighbor_scan = 0;
    size_t center_coarse_projection_dims = 24;
    size_t center_coarse_keep = 512;
    size_t center_coarse_prefilter_dims = 0;
    size_t center_coarse_prefilter_keep = 0;
    size_t center_neighbor_prefilter_dims = 0;
    size_t center_neighbor_prefilter_keep = 0;
    size_t center_neighbor_global_guard_keep = 0;
    bool center_flat_exact_scan = false;
    bool center_super_level_scan = false;
    size_t center_super_count = 128;
    size_t center_super_probe = 3;
    size_t center_super_overlap = 2;
    bool center_coarse_cascade_scan = false;
    size_t center_cascade_low_dims = 0;
    size_t center_cascade_low_keep = 0;
    size_t center_cascade_mid_keep = 0;
    bool center_cascade_use_nth = false;
    CenterEntryMode center_entry_mode = CenterEntryMode::TreeOnly;
    size_t annoy_route_trees = 8;
    size_t annoy_route_leaf_size = 8;
    size_t cluster_route_leaf_size = 8;
    size_t cluster_route_iters = 4;

    size_t center_real_pool_size = 256;
    size_t center_real_pool_take = 48;
    size_t center_real_pool_trigger_topk = 100;
    bool center_real_pool_monotonic = true;
    bool center_real_pool_force = false;

    size_t center_topn_scan = 2000;
    size_t center_topn_probe = 0;
    size_t center_topn_coarse_keep = 96;
    size_t init_keep = 32;
    bool use_hash_neighborhood_spectrum = false;
    size_t hash_spectrum_size = 0;
    size_t hash_spectrum_pool_scan = 1024;
    size_t hash_spectrum_bits = 64;
    size_t hash_spectrum_min_hamming = 8;
    size_t hash_spectrum_segment_bits = 16;
    size_t hash_spectrum_segment_radius = 2;
    size_t hash_spectrum_candidates = 64;
    size_t hash_spectrum_entry_take = 16;
    size_t hash_spectrum_center_keep = 1;
    bool use_residual_hash_spectrum = false;
    size_t residual_hash_bits = 256;
    size_t residual_hash_segments = 4;
    size_t residual_radius_full = 32;
    size_t residual_radius_segment = 8;
    bool normalize_residual_before_hash = false;
    size_t residual_hash_seed = 12345;
    size_t residual_hash_filter_after = 0;
    bool use_residual_hash_bucket_graph = false;
    size_t residual_hash_hot_count = 12;
    size_t residual_hash_bucket_count = 16;
    size_t residual_hash_bucket_take = 3;
    size_t residual_hash_bucket_probe = 1;
    size_t residual_hash_fallback_min = 4;
    size_t residual_hash_cold_count = 8;
    bool residual_hash_build_query_buckets = false;
    bool residual_hash_node_local_bucket = false;

    size_t graph_degree = 48;
    size_t ef_search = 64;
    size_t graph_adaptive_ef_min = 0;
    float graph_adaptive_ef_slack = 1.0F;
    size_t graph_adaptive_ef_check_interval = 4;
    size_t graph_query_adaptive_ef_min = 0;
    size_t graph_query_adaptive_ef_mid = 0;
    float graph_query_adaptive_l1_low_quantile = 0.50F;
    float graph_query_adaptive_l1_high_quantile = 0.85F;
    size_t graph_build_intra_candidates = 384;
    size_t graph_build_cross_candidates = 256;
    size_t graph_build_projection_dims = 6;
    size_t graph_build_center_neighbors = 8;
    size_t graph_insert_new_degree = 0;
    bool graph_search_use_quant = false;
    bool graph_search_full_quant = false;
    bool graph_search_use_u8_l2 = false;
    float graph_u8_clip_low_percentile = 0.0F;
    float graph_u8_clip_high_percentile = 100.0F;
    size_t graph_u8_clip_sample_size = 1000000;
    // Compatibility placeholder. Effective rerank count is always the query's k.
    size_t graph_rerank_candidates = 0;
    bool graph_early_stop = false;
    size_t graph_early_stop_min_expansions = 0;
    float graph_early_stop_slack = 1.0F;
    size_t graph_search_neighbor_cap = 0;
    size_t graph_late_neighbor_cap = 0;
    size_t graph_late_neighbor_after = 0;
    size_t graph_neighbor_prefilter_dims = 0;
    size_t graph_neighbor_prefilter_keep = 0;
    size_t graph_final_prefilter_dims = 0;
    size_t graph_final_prefilter_keep = 0;
    bool graph_result_margin_stop = false;
    size_t graph_result_margin_min_expansions = 0;
    float graph_result_margin = 0.05F;
    size_t graph_result_margin_check_interval = 4;
    bool graph_admission_bound = false;
    float graph_admission_slack = 1.0F;
    size_t graph_neighbor_prefetch = 0;
    bool graph_exact_l2_fast_path = false;
    bool graph_lazy_center_distance = false;
    bool graph_distance_use_norm_dot = false;
    size_t graph_build_mode = 0;  // 0=insertion, 1=nsg, 2=vamana
    bool graph_build_use_u8_l2 = false;
    float graph_vamana_alpha = 1.2F;
    size_t graph_vamana_candidate_limit = 0;
    bool graph_post_nnd_refine = false;
    size_t graph_post_nnd_iterations = 1;
    size_t graph_post_nnd_candidate_limit = 256;
    float graph_post_nnd_alpha = 1.0F;
    bool graph_post_nnd_preserve_degree = true;
    bool graph_build_bridge_edges = false;
    size_t graph_bridge_center_neighbors = 2;
    size_t graph_bridge_points_per_center = 2;
    size_t graph_bridge_candidate_scan = 64;
    bool graph_query_adjacency_order = false;
    bool graph_query_front_prune = false;
    bool graph_dual_scale_search = false;
    size_t graph_dual_short_count = 24;
    size_t graph_dual_long_count = 8;
    float graph_dual_long_alpha = 1.5F;
    float graph_dual_query_beta = 1.5F;
    size_t graph_dual_long_no_improve_limit = 1;
    size_t graph_portal_pool_size = 0;
    size_t graph_hot_neighbor_count = 0;
    size_t graph_cold_neighbor_count = 0;
    size_t graph_cold_max_expansions = 0;
    float graph_cold_search_slack = 1.0F;
    bool graph_reorder_by_center = false;
    bool graph_query_adaptive_center_margin = false;
    float graph_query_adaptive_easy_margin = 0.08F;
    float graph_query_adaptive_hard_margin = 0.025F;
    bool center_adaptive_refine = false;
    size_t center_adaptive_coarse_keep_easy = 0;
    size_t center_adaptive_coarse_keep_mid = 0;
    size_t center_adaptive_scan_keep_easy = 0;
    size_t center_adaptive_scan_keep_mid = 0;
    float center_adaptive_easy_margin = 0.12F;
    float center_adaptive_hard_margin = 0.04F;
    bool hard_query_fallback = false;
    float hard_query_center_margin = 0.025F;
    size_t hard_query_init_keep = 0;
    size_t hard_query_neighbor_cap = 0;
    size_t hard_query_late_neighbor_cap = 0;
    size_t hard_query_late_neighbor_after = 0;
    // Compatibility placeholder. Hard queries use the same query k.
    size_t hard_query_rerank_candidates = 0;
    float hard_query_result_margin = 0.0F;

    size_t quant_total_bits = 4;
    size_t random_seed = 42;
};

enum class SeedMode : uint8_t {
    CenterRealPool,
    Fallback
};

struct QueryStats {
    size_t entry_points_raw = 0;
    size_t entry_points_before_cap = 0;
    size_t entry_points_after_cap = 0;
    size_t init_size = 0;
    size_t seed_pushes = 0;
    size_t heap_pushes = 0;
    size_t heap_pops = 0;
    size_t visited_nodes = 0;
    size_t node_expansions = 0;
    size_t edges_scanned = 0;
    size_t final_candidates = 0;
    size_t topk = 0;
    size_t jumps = 0;
    size_t quant_center_est_evals = 0;
    size_t quant_topn_coarse_evals = 0;
    size_t quant_topn_refine_evals = 0;
    size_t graph_est_evals = 0;
    size_t graph_scan_evals = 0;
    size_t graph_scan_batches = 0;
    size_t exact_distance_evals = 0;
    SeedMode init_mode = SeedMode::Fallback;
    bool trigger_pass = false;
    bool fallback = false;

    // Legacy fields kept for older sample/debug output while the prototype evolves.
    size_t visited = 0;
    size_t center_estimates = 0;
    size_t center_exact_scores = 0;
    size_t topn_coarse_estimates = 0;
    size_t topn_full_estimates = 0;
    size_t graph_center_exact_scores = 0;
    size_t graph_distance_evals = 0;
    size_t exact_rerank_scores = 0;
    size_t graph_early_stop_count = 0;
    size_t graph_adaptive_ef = 0;
    size_t graph_adaptive_ef_stop = 0;
    size_t graph_query_ef_budget = 0;
    size_t graph_prefilter_evals = 0;
    size_t graph_result_margin_stop_count = 0;
    size_t graph_admission_rejects = 0;
    size_t graph_cold_expansions = 0;
    size_t graph_cold_edges_scanned = 0;
    size_t fallback_count = 0;
    size_t trigger_pass_count = 0;
    SeedMode mode = SeedMode::Fallback;
    PID routed_center = 0;
    PID best_center = 0;

    double query_total_ms = 0.0;
    double query_prepare_ms = 0.0;
    double route_ms = 0.0;
    double center_refine_ms = 0.0;
    double seed_select_ms = 0.0;
    double graph_prepare_ms = 0.0;
    double graph_search_ms = 0.0;
    double final_select_ms = 0.0;
};

using ATMMGGraphQueryStats = QueryStats;

struct StatSummary {
    size_t sum = 0;
    size_t max = 0;
    size_t count = 0;

    void add(size_t value) {
        sum += value;
        max = std::max(max, value);
        ++count;
    }

    [[nodiscard]] double avg() const {
        return count == 0 ? 0.0 : static_cast<double>(sum) / static_cast<double>(count);
    }
};

struct TimeSummary {
    double sum = 0.0;
    double max = 0.0;
    size_t count = 0;

    void add(double value) {
        sum += value;
        max = std::max(max, value);
        ++count;
    }

    [[nodiscard]] double avg() const {
        return count == 0 ? 0.0 : sum / static_cast<double>(count);
    }
};

struct BatchStats {
    StatSummary entry_points_raw;
    StatSummary entry_points_before_cap;
    StatSummary entry_points_after_cap;
    StatSummary init_sizes;
    StatSummary seed_pushes;
    StatSummary heap_pushes;
    StatSummary heap_pops;
    StatSummary visited_nodes;
    StatSummary node_expansions;
    StatSummary edges_scanned;
    StatSummary final_candidates;
    StatSummary topk;
    StatSummary jumps;
    StatSummary quant_center_est_evals;
    StatSummary quant_topn_coarse_evals;
    StatSummary quant_topn_refine_evals;
    StatSummary graph_est_evals;
    StatSummary graph_scan_evals;
    StatSummary graph_scan_batches;
    StatSummary exact_distance_evals;
    StatSummary distance_evals;
    StatSummary graph_center_exact_scores;
    StatSummary graph_distance_evals;
    StatSummary exact_rerank_scores;
    StatSummary graph_early_stop_count;
    StatSummary graph_adaptive_ef;
    StatSummary graph_adaptive_ef_stop;
    StatSummary graph_query_ef_budget;
    StatSummary graph_prefilter_evals;
    StatSummary graph_result_margin_stop_count;
    StatSummary graph_admission_rejects;
    StatSummary graph_cold_expansions;
    StatSummary graph_cold_edges_scanned;
    StatSummary trigger_pass;
    StatSummary fallback;
    TimeSummary query_total_ms;
    TimeSummary query_prepare_ms;
    TimeSummary route_ms;
    TimeSummary center_refine_ms;
    TimeSummary seed_select_ms;
    TimeSummary graph_prepare_ms;
    TimeSummary graph_search_ms;
    TimeSummary final_select_ms;
    size_t fallback_count = 0;
    size_t center_real_pool = 0;
    size_t trigger_pass_count = 0;

    void add(const QueryStats& stats) {
        entry_points_raw.add(stats.entry_points_raw);
        entry_points_before_cap.add(stats.entry_points_before_cap);
        entry_points_after_cap.add(stats.entry_points_after_cap);
        init_sizes.add(stats.init_size);
        seed_pushes.add(stats.seed_pushes);
        heap_pushes.add(stats.heap_pushes);
        heap_pops.add(stats.heap_pops);
        visited_nodes.add(stats.visited_nodes);
        node_expansions.add(stats.node_expansions);
        edges_scanned.add(stats.edges_scanned);
        final_candidates.add(stats.final_candidates);
        topk.add(stats.topk);
        jumps.add(stats.jumps);
        quant_center_est_evals.add(stats.quant_center_est_evals);
        quant_topn_coarse_evals.add(stats.quant_topn_coarse_evals);
        quant_topn_refine_evals.add(stats.quant_topn_refine_evals);
        graph_est_evals.add(stats.graph_est_evals);
        graph_scan_evals.add(stats.graph_scan_evals);
        graph_scan_batches.add(stats.graph_scan_batches);
        exact_distance_evals.add(stats.exact_distance_evals);
        distance_evals.add(stats.exact_distance_evals);
        graph_center_exact_scores.add(stats.graph_center_exact_scores);
        graph_distance_evals.add(stats.graph_distance_evals);
        exact_rerank_scores.add(stats.exact_rerank_scores);
        graph_early_stop_count.add(stats.graph_early_stop_count);
        graph_adaptive_ef.add(stats.graph_adaptive_ef);
        graph_adaptive_ef_stop.add(stats.graph_adaptive_ef_stop);
        graph_query_ef_budget.add(stats.graph_query_ef_budget);
        graph_prefilter_evals.add(stats.graph_prefilter_evals);
        graph_result_margin_stop_count.add(stats.graph_result_margin_stop_count);
        graph_admission_rejects.add(stats.graph_admission_rejects);
        graph_cold_expansions.add(stats.graph_cold_expansions);
        graph_cold_edges_scanned.add(stats.graph_cold_edges_scanned);
        trigger_pass.add(stats.trigger_pass ? 1 : 0);
        fallback.add(stats.fallback ? 1 : 0);
        query_total_ms.add(stats.query_total_ms);
        query_prepare_ms.add(stats.query_prepare_ms);
        route_ms.add(stats.route_ms);
        center_refine_ms.add(stats.center_refine_ms);
        seed_select_ms.add(stats.seed_select_ms);
        graph_prepare_ms.add(stats.graph_prepare_ms);
        graph_search_ms.add(stats.graph_search_ms);
        final_select_ms.add(stats.final_select_ms);
        fallback_count += stats.fallback ? 1 : stats.fallback_count;
        trigger_pass_count += stats.trigger_pass ? 1 : stats.trigger_pass_count;
        if (stats.init_mode == SeedMode::CenterRealPool) {
            ++center_real_pool;
        }
    }
};

class ATMMGGraphIndex {
   public:
    explicit ATMMGGraphIndex(
        ATMMGGraphConfig config = ATMMGGraphConfig(),
        RotatorType rotator_type = RotatorType::FhtKacRotator
    )
        : config_(config), rotator_type_(rotator_type) {
        if (config_.quant_total_bits < 1 || config_.quant_total_bits > 9) {
            throw std::invalid_argument("quant_total_bits must be in [1, 9]");
        }
        config_.n_centers = std::max<size_t>(1, config_.n_centers);
        config_.center_leaf_min_size =
            std::max<size_t>(1, config_.center_leaf_min_size);
        config_.graph_degree = std::max<size_t>(1, config_.graph_degree);
        config_.ef_search = std::max<size_t>(1, config_.ef_search);
        config_.graph_adaptive_ef_min =
            std::min(config_.graph_adaptive_ef_min, config_.ef_search);
        config_.graph_adaptive_ef_slack =
            std::max(config_.graph_adaptive_ef_slack, 0.1F);
        config_.graph_adaptive_ef_check_interval =
            std::max<size_t>(1, config_.graph_adaptive_ef_check_interval);
        config_.graph_query_adaptive_ef_min =
            std::min(config_.graph_query_adaptive_ef_min, config_.ef_search);
        config_.graph_query_adaptive_ef_mid =
            std::min(config_.graph_query_adaptive_ef_mid, config_.ef_search);
        config_.graph_query_adaptive_l1_low_quantile =
            std::max(0.0F, std::min(1.0F, config_.graph_query_adaptive_l1_low_quantile));
        config_.graph_query_adaptive_l1_high_quantile =
            std::max(0.0F, std::min(1.0F, config_.graph_query_adaptive_l1_high_quantile));
        if (config_.graph_query_adaptive_l1_high_quantile <
            config_.graph_query_adaptive_l1_low_quantile) {
            std::swap(
                config_.graph_query_adaptive_l1_high_quantile,
                config_.graph_query_adaptive_l1_low_quantile
            );
        }
        config_.center_topn_coarse_keep =
            std::max<size_t>(config_.center_topn_coarse_keep, config_.init_keep);
        config_.center_coarse_keep =
            std::max<size_t>(config_.center_coarse_keep, config_.center_scan_keep);
        config_.center_coarse_prefilter_dims =
            std::min(config_.center_coarse_prefilter_dims, config_.center_coarse_projection_dims);
        if (config_.center_coarse_prefilter_keep > 0) {
            config_.center_coarse_prefilter_keep =
                std::max(config_.center_coarse_prefilter_keep, config_.center_coarse_keep);
        }
        config_.center_neighbor_prefilter_dims =
            std::min(config_.center_neighbor_prefilter_dims, config_.center_coarse_projection_dims);
        if (config_.center_neighbor_prefilter_keep > 0) {
            config_.center_neighbor_prefilter_keep =
                std::max<size_t>(1, std::min(
                    config_.center_neighbor_prefilter_keep,
                    config_.center_refine_neighbor_scan + 1
                ));
        }
        config_.center_neighbor_global_guard_keep =
            std::min(config_.center_neighbor_global_guard_keep, config_.center_coarse_keep);
        config_.center_cascade_low_dims =
            std::min(config_.center_cascade_low_dims, config_.center_coarse_projection_dims);
        if (config_.center_cascade_low_keep > 0) {
            config_.center_cascade_low_keep =
                std::max(
                    {config_.center_cascade_low_keep,
                     config_.center_scan_keep,
                     config_.exact_center_keep}
                );
        }
        if (config_.center_cascade_mid_keep > 0) {
            config_.center_cascade_mid_keep =
                std::max(
                    {config_.center_cascade_mid_keep,
                     config_.center_scan_keep,
                     config_.exact_center_keep}
                );
        }
        config_.annoy_route_trees = std::max<size_t>(1, config_.annoy_route_trees);
        config_.annoy_route_leaf_size =
            std::max<size_t>(1, config_.annoy_route_leaf_size);
        config_.cluster_route_leaf_size =
            std::max<size_t>(1, config_.cluster_route_leaf_size);
        config_.cluster_route_iters = std::max<size_t>(1, config_.cluster_route_iters);
        config_.hash_spectrum_bits =
            std::max<size_t>(1, std::min<size_t>(64, config_.hash_spectrum_bits));
        config_.hash_spectrum_pool_scan =
            std::max<size_t>(1, config_.hash_spectrum_pool_scan);
        config_.hash_spectrum_min_hamming =
            std::min(config_.hash_spectrum_min_hamming, config_.hash_spectrum_bits);
        config_.hash_spectrum_segment_bits = std::max<size_t>(
            1, std::min(config_.hash_spectrum_segment_bits, config_.hash_spectrum_bits)
        );
        config_.hash_spectrum_segment_radius =
            std::min(config_.hash_spectrum_segment_radius, config_.hash_spectrum_segment_bits);
        config_.hash_spectrum_candidates =
            std::max<size_t>(1, config_.hash_spectrum_candidates);
        config_.hash_spectrum_entry_take =
            std::max<size_t>(1, config_.hash_spectrum_entry_take);
        config_.hash_spectrum_center_keep =
            std::max<size_t>(1, config_.hash_spectrum_center_keep);
        config_.residual_hash_bits =
            std::max<size_t>(64, std::min<size_t>(512, config_.residual_hash_bits));
        config_.residual_hash_bits =
            ((config_.residual_hash_bits + 63) / 64) * 64;
        config_.residual_hash_segments =
            std::max<size_t>(
                1, std::min(config_.residual_hash_segments, config_.residual_hash_bits)
            );
        while ((config_.residual_hash_bits % config_.residual_hash_segments) != 0) {
            --config_.residual_hash_segments;
        }
        config_.residual_radius_full =
            std::min(config_.residual_radius_full, config_.residual_hash_bits);
        size_t residual_segment_bits =
            config_.residual_hash_bits / config_.residual_hash_segments;
        config_.residual_radius_segment =
            std::min(config_.residual_radius_segment, residual_segment_bits);
        config_.residual_hash_hot_count =
            std::min(config_.residual_hash_hot_count, config_.graph_degree);
        config_.residual_hash_bucket_count =
            std::max<size_t>(1, config_.residual_hash_bucket_count);
        config_.residual_hash_bucket_take =
            std::max<size_t>(1, config_.residual_hash_bucket_take);
        config_.residual_hash_bucket_probe =
            std::max<size_t>(1, config_.residual_hash_bucket_probe);
        config_.residual_hash_fallback_min =
            std::max<size_t>(1, config_.residual_hash_fallback_min);
        config_.residual_hash_cold_count =
            std::min(config_.residual_hash_cold_count, config_.graph_degree);
        config_.graph_build_intra_candidates =
            std::max(config_.graph_build_intra_candidates, config_.graph_degree);
        config_.graph_build_cross_candidates =
            std::max<size_t>(1, config_.graph_build_cross_candidates);
        config_.graph_build_projection_dims =
            std::max<size_t>(1, config_.graph_build_projection_dims);
        config_.graph_build_center_neighbors =
            std::max<size_t>(1, config_.graph_build_center_neighbors);
        if (config_.graph_insert_new_degree > 0) {
            config_.graph_insert_new_degree =
                std::max<size_t>(1, std::min(config_.graph_insert_new_degree, config_.graph_degree));
        }
        config_.graph_rerank_candidates = 0;
        config_.hard_query_rerank_candidates = 0;
        config_.graph_early_stop_slack =
            std::max(config_.graph_early_stop_slack, 0.1F);
        config_.graph_admission_slack =
            std::max(config_.graph_admission_slack, 0.1F);
        config_.graph_search_neighbor_cap =
            std::min(config_.graph_search_neighbor_cap, config_.graph_degree);
        config_.graph_late_neighbor_cap =
            std::min(config_.graph_late_neighbor_cap, config_.graph_degree);
        if (config_.graph_search_neighbor_cap > 0 && config_.graph_late_neighbor_cap > 0) {
            config_.graph_late_neighbor_cap =
                std::min(config_.graph_late_neighbor_cap, config_.graph_search_neighbor_cap);
        }
        config_.graph_late_neighbor_after =
            std::min(config_.graph_late_neighbor_after, config_.ef_search);
        config_.graph_neighbor_prefilter_keep =
            std::min(config_.graph_neighbor_prefilter_keep, config_.graph_degree);
        config_.graph_result_margin_min_expansions =
            std::min(config_.graph_result_margin_min_expansions, config_.ef_search);
        config_.graph_result_margin =
            std::max(config_.graph_result_margin, 0.0F);
        config_.graph_result_margin_check_interval =
            std::max<size_t>(1, config_.graph_result_margin_check_interval);
        config_.graph_bridge_center_neighbors =
            std::max<size_t>(1, config_.graph_bridge_center_neighbors);
        config_.graph_bridge_points_per_center =
            std::max<size_t>(1, config_.graph_bridge_points_per_center);
        config_.graph_bridge_candidate_scan =
            std::max<size_t>(1, config_.graph_bridge_candidate_scan);
        config_.graph_portal_pool_size =
            std::min(config_.graph_portal_pool_size, config_.center_topn_scan);
        config_.graph_dual_short_count =
            std::min(config_.graph_dual_short_count, config_.graph_degree);
        config_.graph_dual_long_count =
            std::min(config_.graph_dual_long_count, config_.graph_degree);
        config_.graph_dual_long_alpha =
            std::max(config_.graph_dual_long_alpha, 0.0F);
        config_.graph_dual_query_beta =
            std::max(config_.graph_dual_query_beta, 0.0F);
        config_.graph_hot_neighbor_count =
            std::min(config_.graph_hot_neighbor_count, config_.graph_degree);
        config_.graph_cold_neighbor_count =
            std::min(config_.graph_cold_neighbor_count, config_.graph_degree);
        if (config_.graph_cold_neighbor_count > 0 &&
            config_.graph_cold_max_expansions == 0) {
            config_.graph_cold_max_expansions =
                std::max<size_t>(1, std::min<size_t>(8, config_.ef_search / 8));
        }
        config_.graph_cold_max_expansions =
            std::min(config_.graph_cold_max_expansions, config_.ef_search);
        config_.graph_cold_search_slack =
            std::max(config_.graph_cold_search_slack, 0.1F);
        config_.graph_query_adaptive_easy_margin =
            std::max(config_.graph_query_adaptive_easy_margin, 0.0F);
        config_.graph_query_adaptive_hard_margin =
            std::max(config_.graph_query_adaptive_hard_margin, 0.0F);
        if (config_.graph_query_adaptive_easy_margin <
            config_.graph_query_adaptive_hard_margin) {
            std::swap(
                config_.graph_query_adaptive_easy_margin,
                config_.graph_query_adaptive_hard_margin
            );
        }
        config_.center_adaptive_coarse_keep_easy =
            std::min(config_.center_adaptive_coarse_keep_easy, config_.center_coarse_keep);
        config_.center_adaptive_coarse_keep_mid =
            std::min(config_.center_adaptive_coarse_keep_mid, config_.center_coarse_keep);
        config_.center_adaptive_scan_keep_easy =
            std::min(config_.center_adaptive_scan_keep_easy, config_.center_scan_keep);
        config_.center_adaptive_scan_keep_mid =
            std::min(config_.center_adaptive_scan_keep_mid, config_.center_scan_keep);
        config_.center_adaptive_easy_margin =
            std::max(config_.center_adaptive_easy_margin, 0.0F);
        config_.center_adaptive_hard_margin =
            std::max(config_.center_adaptive_hard_margin, 0.0F);
        if (config_.center_adaptive_easy_margin <
            config_.center_adaptive_hard_margin) {
            std::swap(
                config_.center_adaptive_easy_margin,
                config_.center_adaptive_hard_margin
            );
        }
        config_.hard_query_center_margin =
            std::max(config_.hard_query_center_margin, 0.0F);
        config_.hard_query_result_margin =
            std::max(config_.hard_query_result_margin, 0.0F);
        config_.hard_query_neighbor_cap =
            std::min(config_.hard_query_neighbor_cap, config_.graph_degree);
        config_.hard_query_late_neighbor_cap =
            std::min(config_.hard_query_late_neighbor_cap, config_.graph_degree);
        size_t hard_base_cap = config_.hard_query_neighbor_cap == 0
                                   ? config_.graph_search_neighbor_cap
                                   : config_.hard_query_neighbor_cap;
        if (hard_base_cap > 0 && config_.hard_query_late_neighbor_cap > 0) {
            config_.hard_query_late_neighbor_cap =
                std::min(config_.hard_query_late_neighbor_cap, hard_base_cap);
        }
        config_.hard_query_late_neighbor_after =
            std::min(config_.hard_query_late_neighbor_after, config_.ef_search);
        config_.graph_build_mode = std::min<size_t>(config_.graph_build_mode, 2);
        config_.graph_vamana_alpha = std::max(config_.graph_vamana_alpha, 1.0F);
        if (config_.graph_vamana_candidate_limit > 0) {
            config_.graph_vamana_candidate_limit =
                std::max(config_.graph_vamana_candidate_limit, config_.graph_degree);
        }
        config_.graph_post_nnd_iterations =
            std::max<size_t>(1, config_.graph_post_nnd_iterations);
        if (config_.graph_post_nnd_candidate_limit > 0) {
            config_.graph_post_nnd_candidate_limit =
                std::max(config_.graph_post_nnd_candidate_limit, config_.graph_degree);
        }
        config_.graph_post_nnd_alpha =
            std::max(config_.graph_post_nnd_alpha, 1.0F);
    }

    void construct(const float* base, size_t num, size_t dim);

    std::vector<PID> search(
        const float* query, size_t k, ATMMGGraphQueryStats* stats = nullptr
    ) const;
    size_t search_into(const float* query, size_t k, PID* out_ids) const;

    void set_search_params(
        size_t ef_search,
        size_t graph_rerank_candidates,
        size_t graph_search_neighbor_cap
    ) {
        set_query_runtime_params(
            ef_search,
            graph_search_neighbor_cap,
            graph_rerank_candidates,
            config_.hard_query_init_keep,
            config_.hard_query_neighbor_cap,
            config_.hard_query_late_neighbor_cap,
            config_.hard_query_late_neighbor_after,
            config_.hard_query_rerank_candidates,
            config_.graph_final_prefilter_dims,
            config_.graph_final_prefilter_keep,
            config_.hard_query_center_margin,
            config_.hard_query_result_margin
        );
    }

    void set_query_runtime_params(
        size_t ef_search,
        size_t graph_search_neighbor_cap,
        size_t graph_rerank_candidates,
        size_t hard_query_init_keep,
        size_t hard_query_neighbor_cap,
        size_t hard_query_late_neighbor_cap,
        size_t hard_query_late_neighbor_after,
        size_t hard_query_rerank_candidates,
        size_t graph_final_prefilter_dims = 0,
        size_t graph_final_prefilter_keep = 0,
        float hard_query_center_margin = -1.0F,
        float hard_query_result_margin = -1.0F
    ) {
        (void)graph_rerank_candidates;
        (void)hard_query_rerank_candidates;
        config_.ef_search = std::max<size_t>(1, ef_search);
        config_.graph_rerank_candidates = 0;
        config_.graph_search_neighbor_cap =
            std::min(graph_search_neighbor_cap, config_.graph_degree);
        config_.hard_query_init_keep = hard_query_init_keep;
        config_.hard_query_neighbor_cap =
            std::min(hard_query_neighbor_cap, config_.graph_degree);
        config_.hard_query_late_neighbor_cap =
            std::min(hard_query_late_neighbor_cap, config_.graph_degree);
        size_t hard_base_cap = config_.hard_query_neighbor_cap == 0
                                   ? config_.graph_search_neighbor_cap
                                   : config_.hard_query_neighbor_cap;
        if (hard_base_cap > 0 && config_.hard_query_late_neighbor_cap > 0) {
            config_.hard_query_late_neighbor_cap =
                std::min(config_.hard_query_late_neighbor_cap, hard_base_cap);
        }
        config_.hard_query_late_neighbor_after =
            std::min(hard_query_late_neighbor_after, config_.ef_search);
        config_.hard_query_rerank_candidates = 0;
        if (hard_query_center_margin >= 0.0F) {
            config_.hard_query_center_margin = hard_query_center_margin;
        }
        if (hard_query_result_margin >= 0.0F) {
            config_.hard_query_result_margin = hard_query_result_margin;
        }
        bool final_prefilter_changed =
            config_.graph_final_prefilter_dims != graph_final_prefilter_dims ||
            config_.graph_final_prefilter_keep != graph_final_prefilter_keep;
        config_.graph_final_prefilter_dims = graph_final_prefilter_dims;
        config_.graph_final_prefilter_keep = graph_final_prefilter_keep;
        if (final_prefilter_changed) {
            build_graph_prefilter_projection();
        }
        size_t fixed_need = config_.graph_search_neighbor_cap;
        fixed_need = std::max(fixed_need, config_.hard_query_neighbor_cap);
        fixed_need = std::max(fixed_need, config_.graph_late_neighbor_cap);
        fixed_need = std::max(fixed_need, config_.hard_query_late_neighbor_cap);
        rebuild_fixed_search_neighbors(fixed_need);
        quantize_graph_edge_batch_codes();
    }

    [[nodiscard]] const ATMMGGraphConfig& config() const { return config_; }
    void set_center_entry_mode(CenterEntryMode mode) { config_.center_entry_mode = mode; }
    [[nodiscard]] PID route_center_for_query(const float* query) const {
        return route_entry_center(query);
    }
    [[nodiscard]] float center_l2_for_query(const float* query, PID center) const {
        return l2_to_center(query, center);
    }
    [[nodiscard]] PID exact_nearest_center_for_query(const float* query) const {
        size_t centers_count = num_centers();
        PID best = 0;
        float best_d2 = std::numeric_limits<float>::infinity();
        for (size_t c = 0; c < centers_count; ++c) {
            float d2 = l2_to_center(query, static_cast<PID>(c));
            if (d2 < best_d2) {
                best_d2 = d2;
                best = static_cast<PID>(c);
            }
        }
        return best;
    }
    [[nodiscard]] size_t exact_center_rank_for_query(
        const float* query, PID center
    ) const {
        float target = l2_to_center(query, center);
        size_t rank = 1;
        size_t centers_count = num_centers();
        for (size_t c = 0; c < centers_count; ++c) {
            if (static_cast<PID>(c) == center) {
                continue;
            }
            if (l2_to_center(query, static_cast<PID>(c)) < target) {
                ++rank;
            }
        }
        return rank;
    }
    [[nodiscard]] size_t num_points() const { return num_; }
    [[nodiscard]] size_t dim() const { return dim_; }
    [[nodiscard]] size_t padded_dim() const { return padded_dim_; }
    [[nodiscard]] size_t num_centers() const { return centers_.size() / dim_; }
    [[nodiscard]] size_t graph_edges() const { return graph_indices_.size(); }
    [[nodiscard]] size_t graph_scan_edges() const {
        return graph_edge_batch_codes_.count;
    }
    [[nodiscard]] size_t graph_scan_batches() const {
        return graph_edge_batch_codes_.batch_count;
    }
    [[nodiscard]] size_t graph_hot_edges() const { return graph_hot_indices_.size(); }
    [[nodiscard]] size_t residual_hash_bucket_edges() const {
        size_t total = 0;
        for (uint16_t count : residual_bucket_counts_) {
            total += static_cast<size_t>(count);
        }
        return total;
    }
    [[nodiscard]] bool center_quant_refine_enabled() const {
        return uses_center_quant_refine();
    }
    [[nodiscard]] size_t graph_bridge_edges_added() const {
        return graph_bridge_edges_added_;
    }
    [[nodiscard]] size_t graph_post_nnd_edges_before() const {
        return graph_post_nnd_edges_before_;
    }
    [[nodiscard]] size_t graph_post_nnd_edges_after() const {
        return graph_post_nnd_edges_after_;
    }
    [[nodiscard]] size_t graph_post_nnd_candidate_total() const {
        return graph_post_nnd_candidate_total_;
    }
    [[nodiscard]] size_t graph_post_nnd_candidate_sources() const {
        return graph_post_nnd_candidate_sources_;
    }
    [[nodiscard]] bool graph_reordered_by_center() const {
        return graph_reordered_by_center_;
    }
    [[nodiscard]] double avg_graph_degree() const {
        return num_ == 0 ? 0.0
                         : static_cast<double>(graph_indices_.size()) /
                               static_cast<double>(num_);
    }

   private:
    struct TreeNode {
        bool leaf = false;
        size_t left = 0;
        size_t right = 0;
        size_t split_dim = 0;
        float split_value = 0;
        PID center = 0;
    };

    struct AnnoyRouteNode {
        bool leaf = false;
        size_t left = 0;
        size_t right = 0;
        PID pivot_a = 0;
        PID pivot_b = 0;
        float threshold = 0.0F;
        std::vector<PID> centers;
    };

    struct ClusterRouteNode {
        bool leaf = false;
        size_t left = 0;
        size_t right = 0;
        std::vector<float> left_centroid;
        std::vector<float> right_centroid;
        std::vector<PID> centers;
    };

    struct CodeBank {
        size_t count = 0;
        size_t bin_bytes = 0;
        size_t ex_bytes = 0;
        std::vector<uint64_t> bin_storage;
        std::vector<uint64_t> ex_storage;

        static size_t words_for(size_t bytes) {
            return div_round_up(bytes, sizeof(uint64_t));
        }

        void reset(size_t new_count, size_t new_bin_bytes, size_t new_ex_bytes) {
            count = new_count;
            bin_bytes = new_bin_bytes;
            ex_bytes = new_ex_bytes;
            bin_storage.assign(words_for(count * bin_bytes), 0);
            ex_storage.assign(words_for(count * ex_bytes), 0);
        }

        char* bin(size_t i) {
            return reinterpret_cast<char*>(bin_storage.data()) + (i * bin_bytes);
        }

        const char* bin(size_t i) const {
            return reinterpret_cast<const char*>(bin_storage.data()) + (i * bin_bytes);
        }

        char* ex(size_t i) {
            if (ex_bytes == 0) {
                return nullptr;
            }
            return reinterpret_cast<char*>(ex_storage.data()) + (i * ex_bytes);
        }

        const char* ex(size_t i) const {
            if (ex_bytes == 0) {
                return nullptr;
            }
            return reinterpret_cast<const char*>(ex_storage.data()) + (i * ex_bytes);
        }
    };

    struct BatchCodeBank {
        size_t count = 0;
        size_t batch_count = 0;
        size_t batch_bytes = 0;
        std::vector<uint64_t> storage;

        static size_t words_for(size_t bytes) {
            return div_round_up(bytes, sizeof(uint64_t));
        }

        void reset(size_t new_count, size_t new_batch_bytes) {
            count = new_count;
            batch_count = div_round_up(count, scan::kBatchSize);
            batch_bytes = new_batch_bytes;
            storage.assign(words_for(batch_count * batch_bytes), 0);
        }

        char* batch(size_t i) {
            return reinterpret_cast<char*>(storage.data()) + (i * batch_bytes);
        }

        const char* batch(size_t i) const {
            return reinterpret_cast<const char*>(storage.data()) + (i * batch_bytes);
        }
    };

    struct ScoredPid {
        float distance = std::numeric_limits<float>::max();
        PID id = 0;

        friend bool operator<(const ScoredPid& a, const ScoredPid& b) {
            return a.distance < b.distance;
        }
        friend bool operator>(const ScoredPid& a, const ScoredPid& b) {
            return a.distance > b.distance;
        }
    };

    struct IdCount {
        PID id = 0;
        uint32_t count = 0;
    };

    struct IdValueCount {
        PID id = 0;
        uint32_t count = 0;
        float score = 0.0F;
    };

    ATMMGGraphConfig config_;
    RotatorType rotator_type_;
    size_t num_ = 0;
    size_t dim_ = 0;
    size_t padded_dim_ = 0;
    size_t ex_bits_ = 0;
    size_t root_ = 0;

    std::vector<float> base_;
    std::vector<float> base_sq_norms_;
    std::vector<uint8_t> base_u8_;
    std::vector<float> base_u8_min_;
    std::vector<float> base_u8_scale_;
    bool base_u8_identity_quantization_ = false;
    std::vector<float> rotated_base_;
    std::vector<float> centers_;
    std::vector<float> center_sq_norms_;
    std::vector<float> center_super_centers_;
    std::vector<size_t> center_super_offsets_;
    std::vector<PID> center_super_fine_centers_;
    std::vector<float> rotated_centers_;
    std::vector<size_t> center_coarse_dims_;
    std::vector<float> center_coarse_values_;
    std::vector<size_t> graph_prefilter_dims_;
    std::vector<float> center_pool_trigger_d2_;
    std::vector<float> point_center_residual_norms_;
    std::vector<PID> external_ids_;
    float query_adaptive_l1_low_ = 0.0F;
    float query_adaptive_l1_high_ = 0.0F;
    bool query_adaptive_l1_ready_ = false;

    std::vector<TreeNode> tree_;
    std::vector<AnnoyRouteNode> annoy_route_nodes_;
    std::vector<size_t> annoy_route_roots_;
    std::vector<ClusterRouteNode> cluster_route_nodes_;
    size_t cluster_route_root_ = 0;
    std::vector<std::vector<PID>> leaf_ids_;
    std::vector<PID> point_center_;

    std::vector<std::vector<PID>> center_real_pool_;
    std::vector<std::vector<PID>> center_topn_;
    std::vector<std::vector<PID>> center_neighbors_;
    std::vector<std::vector<PID>> center_portal_pool_;
    std::vector<uint8_t> point_is_portal_;
    std::vector<std::vector<PID>> hash_spectrum_ids_;
    std::vector<std::vector<uint64_t>> hash_spectrum_codes_;
    std::vector<uint16_t> hash_spectrum_dims_;
    std::vector<uint16_t> residual_hash_dims_;
    std::vector<int8_t> residual_hash_signs_;
    std::vector<uint64_t> residual_hash_codes_;
    size_t residual_hash_words_ = 0;
    std::vector<PID> residual_bucket_indices_;
    std::vector<uint16_t> residual_bucket_counts_;
    std::vector<float> residual_bucket_center_thresholds_;
    size_t residual_bucket_code_bits_ = 0;
    std::vector<PID> residual_query_bucket_indices_;
    std::vector<uint16_t> residual_query_bucket_counts_;
    size_t residual_query_bucket_width_ = 0;
    size_t residual_query_bucket_stride_ = 0;
    size_t residual_bucket_count_ = 0;
    size_t residual_bucket_take_ = 0;
    size_t residual_bucket_stride_ = 0;
    std::vector<CodeBank> topn_codes_;
    std::vector<BatchCodeBank> topn_batch_codes_;
    CodeBank center_codes_;
    CodeBank point_codes_;
    BatchCodeBank graph_edge_batch_codes_;
    std::vector<size_t> graph_edge_batch_offsets_;

    std::vector<std::vector<PID>> graph_;
    std::vector<size_t> graph_offsets_;
    std::vector<PID> graph_indices_;
    std::vector<size_t> graph_hot_offsets_;
    std::vector<PID> graph_hot_indices_;
    std::vector<size_t> graph_cold_offsets_;
    std::vector<PID> graph_cold_indices_;
    std::vector<PID> graph_fixed_indices_;
    std::vector<uint16_t> graph_fixed_counts_;
    size_t graph_fixed_neighbor_count_ = 0;
    std::vector<PID> graph_dual_short_indices_;
    std::vector<uint16_t> graph_dual_short_counts_;
    std::vector<PID> graph_dual_long_indices_;
    std::vector<uint16_t> graph_dual_long_counts_;
    std::vector<float> graph_dual_short_radius_;
    std::vector<float> graph_dual_short_u8_radius_;
    size_t graph_dual_short_neighbor_count_ = 0;
    size_t graph_dual_long_neighbor_count_ = 0;
    size_t graph_bridge_edges_added_ = 0;
    size_t graph_post_nnd_edges_before_ = 0;
    size_t graph_post_nnd_edges_after_ = 0;
    size_t graph_post_nnd_candidate_total_ = 0;
    size_t graph_post_nnd_candidate_sources_ = 0;
    bool graph_reordered_by_center_ = false;
    mutable std::vector<uint32_t> visit_marks_;
    mutable uint32_t visit_epoch_ = 0;

    std::unique_ptr<Rotator<float>> rotator_;
    quant::QuantConfig index_config_;
    quant::QuantConfig query_config_;
    ex_ipfunc ip_func_ = nullptr;

    size_t build_tree(std::vector<PID> ids, size_t target_leaves);
    PID make_leaf(const std::vector<PID>& ids);
    size_t choose_split_dim(const std::vector<PID>& ids) const;
    PID route_to_center(const float* query) const;
    void build_center_sq_norms();
    void build_annoy_route_forest();
    size_t build_annoy_route_tree(std::vector<PID> centers, std::mt19937& rng);
    PID route_annoy_to_center(const float* query) const;
    void build_cluster_route_tree();
    size_t build_cluster_route_tree(std::vector<PID> centers);
    PID route_cluster_to_center(const float* query) const;
    PID route_entry_center(const float* query) const;

    void rotate_base_and_centers();
    void build_center_pools();
    void build_center_neighbors();
    void build_point_center_residual_norms();
    void build_center_coarse_projection();
    void build_center_super_layer();
    void build_graph_prefilter_projection();
    void build_query_adaptive_l1_thresholds();
    void build_center_portals();
    void rebuild_portal_marks();
    void build_hash_neighborhood_spectrum();
    void build_residual_hash_spectrum();
    void build_residual_hash_bucket_graph();
    void quantize_center_codes();
    void quantize_topn_codes();
    void quantize_point_codes();
    void quantize_graph_edge_batch_codes();
    void build_u8_codes();
    void build_graph();
    void build_graph_insertion();
    void build_graph_nsg();
    void build_graph_vamana();
    void apply_graph_post_nnd_refine();
    bool append_graph_edge(PID a, PID b);
    void add_bridge_edges();
    void order_graph_for_query();
    void reorder_graph_by_center();
    void finalize_graph_csr();
    void rebuild_fixed_search_neighbors(size_t min_neighbor_count);
    void rebuild_dual_scale_neighbors();
    std::vector<PID> search_fast(const float* query, size_t k) const;
    size_t search_fast_into(const float* query, size_t k, PID* out_ids) const;
    bool uses_center_quant_refine() const;
    bool can_use_exact_l2_light_fast_path() const;
    bool can_use_u8_l2_light_fast_path() const;
    std::vector<PID> search_exact_l2_light_fast(
        const float* query,
        size_t k,
        const std::vector<PID>& seeds,
        PID best_center
    ) const;
    size_t search_exact_l2_light_fast_into(
        const float* query,
        size_t k,
        const std::vector<PID>& seeds,
        PID best_center,
        PID* out_ids
    ) const;
    std::vector<PID> search_exact_l2_light_with_stats(
        const float* query,
        size_t k,
        const std::vector<PID>& seeds,
        PID best_center,
        ATMMGGraphQueryStats& local_stats
    ) const;
    std::vector<PID> search_u8_l2_light_fast(
        const float* query,
        float query_sq_norm,
        size_t k,
        const std::vector<PID>& seeds,
        PID best_center,
        float center_margin,
        size_t ef_override = 0
    ) const;
    size_t search_u8_l2_light_fast_into(
        const float* query,
        float query_sq_norm,
        size_t k,
        const std::vector<PID>& seeds,
        PID best_center,
        float center_margin,
        PID* out_ids,
        size_t ef_override = 0
    ) const;
    size_t adaptive_center_budget(
        size_t default_budget,
        size_t easy_budget,
        size_t mid_budget,
        float margin
    ) const;
    bool is_hard_query(float center_margin) const;
    std::vector<ScoredPid> coarse_center_candidates(
        const float* query,
        size_t centers_count,
        size_t coarse_keep,
        float* coarse_margin
    ) const;

    void append_hash_spectrum_seeds(
        const float* query,
        float query_sq_norm,
        const float* rotated_query,
        PID best_center,
        const std::vector<PID>& candidate_centers,
        std::vector<PID>& seeds,
        ATMMGGraphQueryStats* stats = nullptr
    ) const;

    PID refine_center(
        const float* query,
        const SplitSingleQuery<float>& query_wrapper,
        PID routed_center,
        bool use_routed_center,
        ATMMGGraphQueryStats& stats,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    PID refine_center_exact_scan(
        const float* query,
        ATMMGGraphQueryStats* stats = nullptr,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    PID refine_center_super_level_scan(
        const float* query,
        PID routed_center,
        bool use_routed_center,
        ATMMGGraphQueryStats* stats = nullptr,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    PID refine_center_coarse_cascade_scan(
        const float* query,
        PID routed_center,
        bool use_routed_center,
        ATMMGGraphQueryStats* stats = nullptr,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    PID refine_center_fast(
        const float* query,
        const SplitSingleQuery<float>& query_wrapper,
        PID routed_center,
        bool use_routed_center,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    PID refine_center_fast_without_quant(
        const float* query,
        PID routed_center,
        bool use_routed_center,
        std::vector<PID>* exact_centers = nullptr,
        float* center_margin = nullptr
    ) const;

    float estimate_code(
        const char* bin_data,
        const char* ex_data,
        const SplitSingleQuery<float>& query_wrapper,
        float g_add,
        float g_error
    ) const;

    float estimate_code_onebit(
        const char* bin_data,
        const SplitSingleQuery<float>& query_wrapper,
        float g_add,
        float g_error
    ) const;

    bool graph_scan_ready() const;

    uint32_t next_visit_epoch() const;

    static float squared_norm(const float* data, size_t dim) {
        float sum = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            sum += data[i] * data[i];
        }
        return sum;
    }

    static float dot_product(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
        if (dim == 128) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            for (size_t i = 0; i < 128; i += 16) {
                __m256 av0 = _mm256_loadu_ps(a + i);
                __m256 bv0 = _mm256_loadu_ps(b + i);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(av0, bv0));

                __m256 av1 = _mm256_loadu_ps(a + i + 8);
                __m256 bv1 = _mm256_loadu_ps(b + i + 8);
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(av1, bv1));
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }
#else
        (void)a;
        (void)b;
#endif
        float sum = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static float euclidean_sqr_projected_fast(
        const float* a, const float* b, size_t dim
    ) {
#if defined(__AVX2__)
        if (dim > 0 && dim <= 64 && (dim % 8) == 0) {
            __m256 acc = _mm256_setzero_ps();
            for (size_t i = 0; i < dim; i += 8) {
                __m256 av = _mm256_loadu_ps(a + i);
                __m256 bv = _mm256_loadu_ps(b + i);
                __m256 diff = _mm256_sub_ps(av, bv);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            return _mm_cvtss_f32(sum);
        }
#else
        (void)a;
        (void)b;
#endif
        float sum = 0.0F;
        for (size_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    static float euclidean_sqr_fast(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
        if (dim >= 8) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 15 < dim; i += 16) {
                __m256 av0 = _mm256_loadu_ps(a + i);
                __m256 bv0 = _mm256_loadu_ps(b + i);
                __m256 diff0 = _mm256_sub_ps(av0, bv0);
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(diff0, diff0));

                __m256 av1 = _mm256_loadu_ps(a + i + 8);
                __m256 bv1 = _mm256_loadu_ps(b + i + 8);
                __m256 diff1 = _mm256_sub_ps(av1, bv1);
                acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(diff1, diff1));
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            for (; i + 7 < dim; i += 8) {
                __m256 av = _mm256_loadu_ps(a + i);
                __m256 bv = _mm256_loadu_ps(b + i);
                __m256 diff = _mm256_sub_ps(av, bv);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(diff, diff));
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            float scalar_sum = _mm_cvtss_f32(sum);
            for (; i < dim; ++i) {
                float diff = a[i] - b[i];
                scalar_sum += diff * diff;
            }
            return scalar_sum;
        }
#else
        (void)a;
        (void)b;
#endif
        return euclidean_sqr<float>(a, b, dim);
    }

    float l2_to_point(const float* query, PID id) const {
        return euclidean_sqr_fast(query, base_.data() + (static_cast<size_t>(id) * dim_), dim_);
    }

    float l2_to_point(const float* query, float query_sq_norm, PID id) const {
        if (!config_.graph_distance_use_norm_dot) {
            return l2_to_point(query, id);
        }
        const size_t offset = static_cast<size_t>(id) * dim_;
        float dist = query_sq_norm + base_sq_norms_[id] -
                     (2.0F * dot_product(query, base_.data() + offset, dim_));
        return std::max(dist, 0.0F);
    }

    void encode_query_u8_to_buffer(const float* query, uint8_t* query_u8) const {
        if (base_u8_identity_quantization_) {
            for (size_t d = 0; d < dim_; ++d) {
                int value = static_cast<int>(query[d] + 0.5F);
                if (static_cast<unsigned int>(value) > 255U) {
                    value = value < 0 ? 0 : 255;
                }
                query_u8[d] = static_cast<uint8_t>(value);
            }
            return;
        }
        if (base_u8_scale_.empty()) {
            for (size_t d = 0; d < dim_; ++d) {
                float value = std::round(std::max(0.0F, std::min(255.0F, query[d])));
                query_u8[d] = static_cast<uint8_t>(value);
            }
            return;
        }
        for (size_t d = 0; d < dim_; ++d) {
            float encoded = (query[d] - base_u8_min_[d]) * base_u8_scale_[d];
            encoded = std::round(std::max(0.0F, std::min(255.0F, encoded)));
            query_u8[d] = static_cast<uint8_t>(encoded);
        }
    }

    void encode_query_u8(const float* query, std::vector<uint8_t>& query_u8) const {
        query_u8.resize(dim_);
        encode_query_u8_to_buffer(query, query_u8.data());
    }

    float u8_l2_raw(const uint8_t* a, const uint8_t* b) const {
#if defined(__AVX2__)
        if (dim_ == 128) {
            __m256i acc = _mm256_setzero_si256();
            for (size_t i = 0; i < 128; i += 32) {
                __m256i q = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a + i)
                );
                __m256i x = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(b + i)
                );

                __m128i q_lo8 = _mm256_castsi256_si128(q);
                __m128i q_hi8 = _mm256_extracti128_si256(q, 1);
                __m128i x_lo8 = _mm256_castsi256_si128(x);
                __m128i x_hi8 = _mm256_extracti128_si256(x, 1);

                __m256i q_lo = _mm256_cvtepu8_epi16(q_lo8);
                __m256i q_hi = _mm256_cvtepu8_epi16(q_hi8);
                __m256i x_lo = _mm256_cvtepu8_epi16(x_lo8);
                __m256i x_hi = _mm256_cvtepu8_epi16(x_hi8);

                __m256i diff_lo = _mm256_sub_epi16(q_lo, x_lo);
                __m256i diff_hi = _mm256_sub_epi16(q_hi, x_hi);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_lo, diff_lo));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(diff_hi, diff_hi));
            }
            __m128i lo = _mm256_castsi256_si128(acc);
            __m128i hi = _mm256_extracti128_si256(acc, 1);
            __m128i sum = _mm_add_epi32(lo, hi);
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
            sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
            return static_cast<float>(_mm_cvtsi128_si32(sum));
        }
#endif
        uint32_t sum = 0;
        for (size_t d = 0; d < dim_; ++d) {
            int diff = static_cast<int>(a[d]) - static_cast<int>(b[d]);
            sum += static_cast<uint32_t>(diff * diff);
        }
        return static_cast<float>(sum);
    }

    float u8_l2_to_point(const uint8_t* query_u8, PID id) const {
        const uint8_t* point = base_u8_.data() + (static_cast<size_t>(id) * dim_);
        return u8_l2_raw(query_u8, point);
    }

    float l2_to_center(const float* query, PID center) const {
        return euclidean_sqr_fast(
            query, centers_.data() + (static_cast<size_t>(center) * dim_), dim_
        );
    }

    float coarse_l2_to_center(const float* query, PID center) const {
        return coarse_l2_to_center(query, center, center_coarse_dims_.size());
    }

    float coarse_l2_to_center(const float* query, PID center, size_t dims_count) const {
        if (center_coarse_dims_.empty()) {
            return l2_to_center(query, center);
        }
        dims_count = std::min(dims_count, center_coarse_dims_.size());
        if (dims_count == 0) {
            return 0.0F;
        }
        bool use_projected = !center_coarse_values_.empty();
        const float* center_vec =
            use_projected
                ? center_coarse_values_.data() +
                      (static_cast<size_t>(center) * center_coarse_dims_.size())
                : centers_.data() + (static_cast<size_t>(center) * dim_);
        float sum = 0.0F;
        for (size_t i = 0; i < dims_count; ++i) {
            size_t d = center_coarse_dims_[i];
            float center_value = use_projected ? center_vec[i] : center_vec[d];
            float diff = query[d] - center_value;
            sum += diff * diff;
        }
        return sum;
    }

    float graph_prefilter_l2_to_point(
        const float* query,
        PID id,
        size_t dims_limit = std::numeric_limits<size_t>::max()
    ) const {
        const float* point = base_.data() + (static_cast<size_t>(id) * dim_);
        float sum = 0.0F;
        size_t dims_count = std::min(dims_limit, graph_prefilter_dims_.size());
        for (size_t i = 0; i < dims_count; ++i) {
            size_t d = graph_prefilter_dims_[i];
            float diff = query[d] - point[d];
            sum += diff * diff;
        }
        return sum;
    }

    size_t apply_final_prefilter(
        const float* query,
        size_t k,
        std::vector<ScoredPid>& candidates,
        size_t rerank_keep
    ) const {
        if (config_.graph_final_prefilter_dims == 0 ||
            config_.graph_final_prefilter_keep == 0 ||
            graph_prefilter_dims_.empty() || rerank_keep <= k) {
            return rerank_keep;
        }
        size_t final_keep =
            std::min(rerank_keep, std::max(k, config_.graph_final_prefilter_keep));
        if (final_keep >= rerank_keep) {
            return rerank_keep;
        }
        size_t dims =
            std::min(config_.graph_final_prefilter_dims, graph_prefilter_dims_.size());
        if (dims == 0) {
            return rerank_keep;
        }
        for (size_t i = 0; i < rerank_keep; ++i) {
            candidates[i].distance =
                graph_prefilter_l2_to_point(query, candidates[i].id, dims);
        }
        keep_smallest(candidates, final_keep);
        return final_keep;
    }

    void prefetch_point(PID id) const {
#if defined(__GNUC__) || defined(__clang__)
        const char* ptr = reinterpret_cast<const char*>(
            base_.data() + (static_cast<size_t>(id) * dim_)
        );
        __builtin_prefetch(ptr, 0, 1);
        if (dim_ * sizeof(float) > 256) {
            __builtin_prefetch(ptr + 256, 0, 1);
        }
#elif defined(_MSC_VER)
        const char* ptr = reinterpret_cast<const char*>(
            base_.data() + (static_cast<size_t>(id) * dim_)
        );
        _mm_prefetch(ptr, _MM_HINT_T0);
        if (dim_ * sizeof(float) > 256) {
            _mm_prefetch(ptr + 256, _MM_HINT_T0);
        }
#endif
    }

    void prefetch_u8_point(PID id) const {
#if defined(__GNUC__) || defined(__clang__)
        const char* ptr = reinterpret_cast<const char*>(
            base_u8_.data() + (static_cast<size_t>(id) * dim_)
        );
        __builtin_prefetch(ptr, 0, 1);
        if (dim_ > 64) {
            __builtin_prefetch(ptr + 64, 0, 1);
        }
#elif defined(_MSC_VER)
        const char* ptr = reinterpret_cast<const char*>(
            base_u8_.data() + (static_cast<size_t>(id) * dim_)
        );
        _mm_prefetch(ptr, _MM_HINT_T0);
        if (dim_ > 64) {
            _mm_prefetch(ptr + 64, _MM_HINT_T0);
        }
#else
        (void)id;
#endif
    }

    uint64_t hash_spectrum_code(PID center, const float* rotated_vec) const;
    static uint32_t popcount64(uint64_t value);
    uint32_t hash_spectrum_segment_min(uint64_t a, uint64_t b) const;
    bool residual_hash_ready() const;
    bool residual_hash_codes_ready() const;
    bool residual_hash_bucket_ready() const;
    bool residual_hash_query_bucket_ready() const;
    size_t residual_hash_bucket_id(const uint64_t* code) const;
    size_t residual_hash_bucket_id_for_node(const float* query, PID node_id) const;
    size_t residual_hash_bucket_id_for_center(
        const float* query,
        PID center_id
    ) const;
    void residual_hash_code(const float* vec, const float* center, uint64_t* out) const;
    void residual_hash_prepare_query(const float* query, float* out) const;
    void residual_hash_code_from_prepared_query(
        const float* center,
        const float* prepared_query,
        uint64_t* out
    ) const;
    bool residual_hash_pass(const uint64_t* query_code, const uint64_t* edge_code) const;

    float point_point_l2(PID a, PID b) const {
        if (config_.graph_build_use_u8_l2 && !base_u8_.empty()) {
            const uint8_t* pa = base_u8_.data() + (static_cast<size_t>(a) * dim_);
            const uint8_t* pb = base_u8_.data() + (static_cast<size_t>(b) * dim_);
            return u8_l2_raw(pa, pb);
        }
        if (!config_.graph_distance_use_norm_dot) {
            return euclidean_sqr_fast(
                base_.data() + (static_cast<size_t>(a) * dim_),
                base_.data() + (static_cast<size_t>(b) * dim_),
                dim_
            );
        }
        const size_t a_offset = static_cast<size_t>(a) * dim_;
        const size_t b_offset = static_cast<size_t>(b) * dim_;
        float dist = base_sq_norms_[a] + base_sq_norms_[b] -
                     (2.0F * dot_product(
                                 base_.data() + a_offset,
                                 base_.data() + b_offset,
                                 dim_
                             ));
        return std::max(dist, 0.0F);
    }

    float query_l1_norm(const float* query) const {
        float sum = 0.0F;
        for (size_t d = 0; d < dim_; ++d) {
            sum += std::fabs(query[d]);
        }
        return sum;
    }

    size_t effective_ef_for_query(
        const float* query,
        float center_margin = std::numeric_limits<float>::quiet_NaN()
    ) const {
        if (config_.graph_query_adaptive_center_margin &&
            config_.graph_query_adaptive_ef_min > 0 &&
            config_.graph_query_adaptive_ef_min < config_.ef_search &&
            std::isfinite(center_margin)) {
            size_t low_budget = config_.graph_query_adaptive_ef_min;
            size_t mid_budget = config_.graph_query_adaptive_ef_mid == 0
                                    ? (low_budget + config_.ef_search) / 2
                                    : config_.graph_query_adaptive_ef_mid;
            mid_budget = std::max(low_budget, std::min(mid_budget, config_.ef_search));
            if (center_margin >= config_.graph_query_adaptive_easy_margin) {
                return low_budget;
            }
            if (center_margin >= config_.graph_query_adaptive_hard_margin) {
                return mid_budget;
            }
            return config_.ef_search;
        }
        if (!query_adaptive_l1_ready_ ||
            config_.graph_query_adaptive_ef_min == 0 ||
            config_.graph_query_adaptive_ef_min >= config_.ef_search) {
            return config_.ef_search;
        }
        size_t low_budget = config_.graph_query_adaptive_ef_min;
        size_t mid_budget = config_.graph_query_adaptive_ef_mid == 0
                                ? (low_budget + config_.ef_search) / 2
                                : config_.graph_query_adaptive_ef_mid;
        mid_budget = std::max(low_budget, std::min(mid_budget, config_.ef_search));

        float l1 = query_l1_norm(query);
        if (l1 <= query_adaptive_l1_low_) {
            return low_budget;
        }
        if (l1 <= query_adaptive_l1_high_) {
            return mid_budget;
        }
        return config_.ef_search;
    }

    PID external_id(PID id) const {
        return external_ids_.empty() ? id : external_ids_[id];
    }

    static void keep_smallest(std::vector<ScoredPid>& items, size_t keep);
    static void push_topk_smallest(
        std::vector<ScoredPid>& heap, ScoredPid item, size_t keep
    );
    static void finish_topk_smallest(std::vector<ScoredPid>& heap);
};

#include "ATMMG/index/ATMMG_graph/detail/build_core.ipp"
#include "ATMMG/index/ATMMG_graph/detail/search.ipp"
#include "ATMMG/index/ATMMG_graph/detail/routing.ipp"

}  // namespace ATMMG::graph
