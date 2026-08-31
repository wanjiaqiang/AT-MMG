inline std::vector<PID> ATMMGGraphIndex::search(
    const float* query, size_t k, ATMMGGraphQueryStats* stats
) const {
    if (query == nullptr || k == 0) {
        return {};
    }
    if (num_ == 0) {
        throw std::runtime_error("ATMMGGraphIndex is not constructed");
    }
    if (stats == nullptr) {
        return search_fast(query, k);
    }

    using ProfileClock = std::chrono::high_resolution_clock;
    auto profile_now = []() { return ProfileClock::now(); };
    auto elapsed_ms = [](ProfileClock::time_point begin, ProfileClock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    auto total_begin = profile_now();

    ATMMGGraphQueryStats local_stats;
    local_stats.topk = k;
    auto prepare_begin = profile_now();
    float query_sq_norm = squared_norm(query, dim_);
    std::vector<float> rotated_query;
    auto ensure_rotated_query = [&]() -> const float* {
        if (rotated_query.empty()) {
            rotated_query.assign(padded_dim_, 0.0F);
            rotator_->rotate(query, rotated_query.data());
        }
        return rotated_query.data();
    };
    std::unique_ptr<SplitSingleQuery<float>> query_wrapper;
    auto ensure_query_wrapper = [&]() -> SplitSingleQuery<float>& {
        if (query_wrapper == nullptr) {
            query_wrapper = std::make_unique<SplitSingleQuery<float>>(
                ensure_rotated_query(), padded_dim_, ex_bits_, query_config_, METRIC_L2
            );
        }
        return *query_wrapper;
    };
    auto prepare_end = profile_now();
    local_stats.query_prepare_ms = elapsed_ms(prepare_begin, prepare_end);

    auto route_begin = profile_now();
    PID routed_center = 0;
    if (config_.center_entry_mode != CenterEntryMode::QuantOnly) {
        routed_center = route_entry_center(query);
    }
    auto route_end = profile_now();
    local_stats.route_ms = elapsed_ms(route_begin, route_end);

    auto center_refine_begin = profile_now();
    PID best_center = routed_center;
    std::vector<PID> candidate_centers;
    float center_margin = std::numeric_limits<float>::quiet_NaN();
    if (config_.center_entry_mode == CenterEntryMode::QuantOnly) {
        best_center =
            refine_center(
                query, ensure_query_wrapper(), routed_center, false, local_stats,
                &candidate_centers, &center_margin
            );
    } else if (config_.center_entry_mode == CenterEntryMode::TreeThenQuant) {
        best_center =
            refine_center(
                query, ensure_query_wrapper(), routed_center, true, local_stats,
                &candidate_centers, &center_margin
            );
    }
    if (candidate_centers.empty()) {
        candidate_centers.push_back(best_center);
    }
    auto center_refine_end = profile_now();
    local_stats.center_refine_ms =
        elapsed_ms(center_refine_begin, center_refine_end);

    auto seed_begin = profile_now();
    float best_center_dist2 = l2_to_center(query, best_center);
    ++local_stats.exact_distance_evals;

    local_stats.routed_center = routed_center;
    local_stats.best_center = best_center;

    std::vector<PID> seeds;
    local_stats.mode = SeedMode::CenterRealPool;
    local_stats.init_mode = SeedMode::CenterRealPool;
    local_stats.trigger_pass = true;
    local_stats.trigger_pass_count = 1;
    if (best_center < center_real_pool_.size()) {
        const auto& pool = center_real_pool_[best_center];
        size_t take = std::min(config_.center_real_pool_take, pool.size());
        local_stats.entry_points_raw = pool.size();
        local_stats.entry_points_before_cap = pool.size();
        local_stats.entry_points_after_cap = take;
        local_stats.init_size = take;
        seeds.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(take));
    }
    if (seeds.empty()) {
        seeds.push_back(0);
        local_stats.entry_points_after_cap = 1;
        local_stats.init_size = 1;
    }
    auto seed_end = profile_now();
    local_stats.seed_select_ms = elapsed_ms(seed_begin, seed_end);

    if (can_use_exact_l2_light_fast_path()) {
        std::vector<PID> result =
            search_exact_l2_light_with_stats(query, k, seeds, best_center, local_stats);
        auto total_end = profile_now();
        local_stats.query_total_ms = elapsed_ms(total_begin, total_end);
        if (stats != nullptr) {
            *stats = local_stats;
        }
        for (PID& id : result) {
            id = external_id(id);
        }
        return result;
    }

    auto graph_prepare_begin = profile_now();
    std::vector<float> center_query_dist2;
    std::vector<float> center_query_norm;
    bool use_admission_bound =
        config_.graph_admission_bound && !point_center_residual_norms_.empty();
    bool needs_center_query_distance =
        config_.graph_search_use_quant || use_admission_bound;
    if (needs_center_query_distance) {
        size_t centers_count = num_centers();
        center_query_dist2.resize(centers_count);
        center_query_norm.resize(centers_count);
        if (config_.graph_lazy_center_distance || use_admission_bound ||
            !config_.graph_search_use_quant) {
            std::fill(center_query_dist2.begin(), center_query_dist2.end(), -1.0F);
            std::fill(center_query_norm.begin(), center_query_norm.end(), 0.0F);
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        } else {
            for (size_t c = 0; c < centers_count; ++c) {
                float d2 = l2_to_center(query, static_cast<PID>(c));
                center_query_dist2[c] = d2;
                center_query_norm[c] = std::sqrt(std::max(d2, 0.0F));
            }
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
            local_stats.graph_center_exact_scores = centers_count;
            local_stats.exact_distance_evals += centers_count;
        }
    }
    std::vector<uint8_t> query_u8;
    if (config_.graph_search_use_u8_l2 && !base_u8_.empty() &&
        !config_.graph_search_use_quant) {
        encode_query_u8(query, query_u8);
    }
    const bool residual_filter_ready = residual_hash_ready();
    std::array<float, 512> residual_prepared_query{};
    bool residual_query_prepared = false;
    auto graph_prepare_end = profile_now();
    local_stats.graph_prepare_ms =
        elapsed_ms(graph_prepare_begin, graph_prepare_end);

    auto ensure_center_query_distance = [&](PID center_id) {
        if (center_query_dist2.empty()) {
            return;
        }
        if (center_query_dist2[center_id] < 0.0F) {
            float d2 = l2_to_center(query, center_id);
            center_query_dist2[center_id] = d2;
            center_query_norm[center_id] = std::sqrt(std::max(d2, 0.0F));
            ++local_stats.graph_center_exact_scores;
            ++local_stats.exact_distance_evals;
        }
    };

    auto graph_distance = [&](PID id) {
        ++local_stats.graph_distance_evals;
        if (!config_.graph_search_use_quant) {
            if (!query_u8.empty()) {
                return u8_l2_to_point(query_u8.data(), id);
            }
            ++local_stats.exact_distance_evals;
            return l2_to_point(query, query_sq_norm, id);
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        ++local_stats.graph_est_evals;
        if (config_.graph_search_full_quant) {
            return estimate_code(
                point_codes_.bin(id),
                point_codes_.ex(id),
                ensure_query_wrapper(),
                center_query_dist2[center_id],
                center_query_norm[center_id]
            );
        }
        return estimate_code_onebit(
            point_codes_.bin(id),
            ensure_query_wrapper(),
            center_query_dist2[center_id],
            center_query_norm[center_id]
        );
    };

    auto graph_search_begin = profile_now();
    const size_t effective_ef_max = effective_ef_for_query(query, center_margin);
    local_stats.graph_query_ef_budget = effective_ef_max;
    const bool hard_query = is_hard_query(center_margin);
    const size_t effective_neighbor_cap =
        hard_query && config_.hard_query_neighbor_cap > 0
            ? config_.hard_query_neighbor_cap
            : config_.graph_search_neighbor_cap;
    const size_t effective_late_neighbor_cap =
        hard_query && config_.hard_query_late_neighbor_cap > 0
            ? config_.hard_query_late_neighbor_cap
            : config_.graph_late_neighbor_cap;
    const size_t effective_late_neighbor_after =
        hard_query && config_.hard_query_late_neighbor_after > 0
            ? config_.hard_query_late_neighbor_after
            : config_.graph_late_neighbor_after;
    const bool use_adaptive_ef =
        config_.graph_adaptive_ef_min > 0 &&
        config_.graph_adaptive_ef_min < effective_ef_max;
    uint32_t visit_epoch = next_visit_epoch();
    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, effective_ef_max * config_.graph_degree));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, effective_ef_max * config_.graph_degree));
    size_t window_keep = k;
    std::priority_queue<ScoredPid> best_window;
    bool track_window =
        config_.graph_early_stop || config_.graph_cold_neighbor_count > 0 ||
        use_admission_bound;
    float best_seen_distance = std::numeric_limits<float>::infinity();
    auto push_seen = [&](ScoredPid item) {
        seen.push_back(item);
        best_seen_distance = std::min(best_seen_distance, item.distance);
        if (track_window) {
            best_window.push(item);
            if (best_window.size() > window_keep) {
                best_window.pop();
            }
        }
    };

    for (PID id : seeds) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            continue;
        }
        visit_marks_[id] = visit_epoch;
        ++local_stats.visited_nodes;
        float dist = graph_distance(id);
        candidate_queue.push({dist, id});
        push_seen({dist, id});
        ++local_stats.seed_pushes;
        ++local_stats.heap_pushes;
    }

    if (candidate_queue.empty()) {
        ++local_stats.fallback_count;
        local_stats.fallback = true;
        local_stats.mode = SeedMode::Fallback;
        PID fallback = center_real_pool_[best_center].empty() ? 0 : center_real_pool_[best_center][0];
        visit_marks_[fallback] = visit_epoch;
        ++local_stats.visited_nodes;
        float dist = graph_distance(fallback);
        candidate_queue.push({dist, fallback});
        push_seen({dist, fallback});
        ++local_stats.seed_pushes;
        ++local_stats.heap_pushes;
    }

    bool use_hot_cold = !graph_hot_indices_.empty();
    const std::vector<size_t>& search_offsets =
        use_hot_cold ? graph_hot_offsets_ : graph_offsets_;
    const std::vector<PID>& search_indices =
        use_hot_cold ? graph_hot_indices_ : graph_indices_;
    bool use_cold_fallback =
        use_hot_cold && config_.graph_cold_neighbor_count > 0 &&
        config_.graph_cold_max_expansions > 0 && !graph_cold_indices_.empty();
    size_t cold_expansions = 0;

    auto should_scan_cold = [&](const ScoredPid& current) {
        if (!use_cold_fallback) {
            return false;
        }
        if (cold_expansions >= config_.graph_cold_max_expansions) {
            return false;
        }
        if (best_window.size() < window_keep) {
            return true;
        }
        return current.distance <=
               best_window.top().distance * config_.graph_cold_search_slack;
    };

    auto reject_by_admission_bound = [&](PID id) {
        if (!use_admission_bound || best_window.size() < window_keep) {
            return false;
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        float center_norm = center_query_norm[center_id];
        float residual_norm = point_center_residual_norms_[id];
        float lower = std::fabs(center_norm - residual_norm);
        float lower_sq = lower * lower;
        if (lower_sq <= best_window.top().distance * config_.graph_admission_slack) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ++local_stats.graph_admission_rejects;
        return true;
    };

    bool use_neighbor_prefilter =
        config_.graph_neighbor_prefilter_keep > 0 && !graph_prefilter_dims_.empty();
    bool use_result_margin_stop =
        config_.graph_result_margin_stop && !config_.graph_search_use_quant &&
        query_u8.empty();
    std::vector<ScoredPid> neighbor_prefilter;
    if (use_neighbor_prefilter) {
        size_t reserve_cap = effective_neighbor_cap == 0
                                 ? config_.graph_degree
                                 : effective_neighbor_cap;
        neighbor_prefilter.reserve(std::min(config_.graph_neighbor_prefilter_keep, reserve_cap));
    }
    std::unique_ptr<BatchQuery<float>> graph_batch_query;
    auto ensure_graph_batch_query = [&]() -> BatchQuery<float>& {
        if (graph_batch_query == nullptr) {
            graph_batch_query =
                std::make_unique<BatchQuery<float>>(ensure_rotated_query(), padded_dim_);
        }
        return *graph_batch_query;
    };
    auto push_graph_neighbor_with_distance = [&](PID nb, float dist) {
        visit_marks_[nb] = visit_epoch;
        ++local_stats.visited_nodes;
        candidate_queue.push({dist, nb});
        ++local_stats.heap_pushes;
        bool improved = dist < best_seen_distance;
        push_seen({dist, nb});
        return improved;
    };
    auto push_graph_neighbor = [&](PID nb) {
        return push_graph_neighbor_with_distance(nb, graph_distance(nb));
    };

    const bool use_fixed_search_neighbors =
        !graph_scan_ready() && !use_hot_cold && !use_neighbor_prefilter &&
        !residual_filter_ready &&
        graph_fixed_neighbor_count_ > 0 && !graph_fixed_indices_.empty();

    auto scan_fixed_neighbors = [&](const PID* neighbors, size_t limit) {
        for (size_t ni = 0; ni < limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < limit) {
                    prefetch_point(neighbors[pf]);
                }
            }
            PID nb = neighbors[ni];
            ++local_stats.edges_scanned;
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            if (reject_by_admission_bound(nb)) {
                continue;
            }
            push_graph_neighbor(nb);
        }
    };

    auto scan_graph_neighbors = [&](
                                    PID center_id,
                                    const std::vector<PID>& indices,
                                    size_t begin,
                                    size_t limit,
                                    bool allow_prefilter,
                                    bool count_cold_edges
                                ) {
        bool prefilter =
            allow_prefilter && use_neighbor_prefilter &&
            config_.graph_neighbor_prefilter_keep < limit;
        if (prefilter) {
            neighbor_prefilter.clear();
            size_t keep = std::min(config_.graph_neighbor_prefilter_keep, limit);
            for (size_t ni = 0; ni < limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < limit) {
                        prefetch_point(indices[begin + pf]);
                    }
                }
                PID nb = indices[begin + ni];
                ++local_stats.edges_scanned;
                if (count_cold_edges) {
                    ++local_stats.graph_cold_edges_scanned;
                }
                if (visit_marks_[nb] == visit_epoch) {
                    continue;
                }
                if (reject_by_admission_bound(nb)) {
                    continue;
                }
                ++local_stats.graph_prefilter_evals;
                push_topk_smallest(
                    neighbor_prefilter, {graph_prefilter_l2_to_point(query, nb), nb}, keep
                );
            }
            finish_topk_smallest(neighbor_prefilter);
            for (const auto& cand : neighbor_prefilter) {
                if (visit_marks_[cand.id] == visit_epoch) {
                    continue;
                }
                push_graph_neighbor(cand.id);
            }
            return;
        }

        std::array<PID, 128> accepted{};
        size_t accepted_count = 0;
        bool used_residual_filter =
            residual_filter_ready &&
            (config_.residual_hash_filter_after == 0 ||
             local_stats.node_expansions > config_.residual_hash_filter_after) &&
            !use_hot_cold &&
            indices.data() == graph_indices_.data() && limit > 0 &&
            limit <= accepted.size();
        if (used_residual_filter) {
            if (!residual_query_prepared) {
                residual_hash_prepare_query(query, residual_prepared_query.data());
                residual_query_prepared = true;
            }
            const float* center =
                base_.data() + (static_cast<size_t>(center_id) * dim_);
            std::array<uint64_t, 8> query_code{};
            residual_hash_code_from_prepared_query(
                center, residual_prepared_query.data(), query_code.data()
            );
            for (size_t ni = 0; ni < limit; ++ni) {
                size_t pos = begin + ni;
                PID nb = indices[pos];
                ++local_stats.edges_scanned;
                if (visit_marks_[nb] == visit_epoch) {
                    continue;
                }
                if (reject_by_admission_bound(nb)) {
                    continue;
                }
                const uint64_t* edge_code =
                    residual_hash_codes_.data() + (pos * residual_hash_words_);
                if (residual_hash_pass(query_code.data(), edge_code)) {
                    accepted[accepted_count++] = nb;
                }
            }
            if (accepted_count > 0) {
                for (size_t i = 0; i < accepted_count; ++i) {
                    push_graph_neighbor(accepted[i]);
                }
                return;
            }
        }

        bool can_scan_row =
            graph_scan_ready() && limit > 0 && !used_residual_filter &&
            center_id < num_ && indices.data() == graph_indices_.data() &&
            begin == graph_offsets_[center_id];
        if (can_scan_row) {
            BatchQuery<float>& q_obj = ensure_graph_batch_query();
            q_obj.set_g_add(l2_to_point(query, query_sq_norm, center_id));
            ++local_stats.exact_distance_evals;

            std::array<float, scan::kBatchSize> est_dist{};
            size_t batch_base = graph_edge_batch_offsets_[center_id];
            for (size_t offset = 0; offset < limit; offset += scan::kBatchSize) {
                size_t count = std::min(scan::kBatchSize, limit - offset);
                size_t batch_id = batch_base + (offset / scan::kBatchSize);
                qg_batch_estdist(
                    graph_edge_batch_codes_.batch(batch_id),
                    q_obj,
                    padded_dim_,
                    est_dist.data()
                );
                ++local_stats.graph_scan_batches;
                local_stats.graph_scan_evals += count;
                local_stats.graph_distance_evals += count;
                local_stats.graph_est_evals += count;

                for (size_t j = 0; j < count; ++j) {
                    size_t ni = offset + j;
                    if (config_.graph_neighbor_prefetch > 0) {
                        size_t pf = ni + config_.graph_neighbor_prefetch;
                        if (pf < limit) {
                            prefetch_point(indices[begin + pf]);
                        }
                    }
                    PID nb = indices[begin + ni];
                    ++local_stats.edges_scanned;
                    if (visit_marks_[nb] == visit_epoch) {
                        continue;
                    }
                    if (reject_by_admission_bound(nb)) {
                        continue;
                    }
                    push_graph_neighbor_with_distance(nb, est_dist[j]);
                }
            }
            return;
        }

        for (size_t ni = 0; ni < limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < limit) {
                    prefetch_point(indices[begin + pf]);
                }
            }
            PID nb = indices[begin + ni];
            if (!used_residual_filter) {
                ++local_stats.edges_scanned;
            }
            if (count_cold_edges && !used_residual_filter) {
                ++local_stats.graph_cold_edges_scanned;
            }
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            if (reject_by_admission_bound(nb)) {
                continue;
            }
            push_graph_neighbor(nb);
        }
    };

    auto should_stop_adaptive_ef = [&]() {
        if (!use_adaptive_ef ||
            local_stats.node_expansions < config_.graph_adaptive_ef_min ||
            seen.size() < window_keep || candidate_queue.empty()) {
            return false;
        }
        size_t after_min = local_stats.node_expansions - config_.graph_adaptive_ef_min;
        if ((after_min % config_.graph_adaptive_ef_check_interval) != 0) {
            return false;
        }
        auto nth = seen.begin() + static_cast<std::ptrdiff_t>(window_keep - 1);
        std::nth_element(seen.begin(), nth, seen.end());
        float bound = nth->distance * config_.graph_adaptive_ef_slack;
        if (candidate_queue.top().distance <= bound) {
            return false;
        }
        local_stats.graph_adaptive_ef_stop = 1;
        return true;
    };

    auto should_stop_result_margin = [&]() {
        if (!use_result_margin_stop || k == 0 ||
            local_stats.node_expansions < config_.graph_result_margin_min_expansions ||
            seen.size() <= k) {
            return false;
        }
        size_t after_min =
            local_stats.node_expansions - config_.graph_result_margin_min_expansions;
        if ((after_min % config_.graph_result_margin_check_interval) != 0) {
            return false;
        }
        auto next = seen.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(seen.begin(), next, seen.end());
        auto kth = seen.begin() + static_cast<std::ptrdiff_t>(k - 1);
        std::nth_element(seen.begin(), kth, next);
        float denom = std::max(kth->distance, 1e-6F);
        float margin = (next->distance - kth->distance) / denom;
        if (margin < config_.graph_result_margin) {
            return false;
        }
        local_stats.graph_result_margin_stop_count = 1;
        return true;
    };

    while (!candidate_queue.empty() && local_stats.node_expansions < effective_ef_max) {
        if (config_.graph_early_stop &&
            local_stats.node_expansions >= config_.graph_early_stop_min_expansions &&
            best_window.size() >= window_keep) {
            float bound = best_window.top().distance * config_.graph_early_stop_slack;
            if (candidate_queue.top().distance > bound) {
                local_stats.graph_early_stop_count = 1;
                break;
            }
        }
        if (should_stop_adaptive_ef()) {
            break;
        }
        if (should_stop_result_margin()) {
            break;
        }
        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++local_stats.heap_pops;
        ++local_stats.node_expansions;
        bool scan_cold_for_current = should_scan_cold(current);

        size_t neighbor_begin = 0;
        size_t neighbor_count = 0;
        const PID* fixed_neighbors = nullptr;
        if (use_fixed_search_neighbors) {
            fixed_neighbors =
                graph_fixed_indices_.data() +
                static_cast<size_t>(current.id) * graph_fixed_neighbor_count_;
            neighbor_count = graph_fixed_counts_[current.id];
        } else {
            neighbor_begin = search_offsets[current.id];
            size_t neighbor_end = search_offsets[static_cast<size_t>(current.id) + 1];
            neighbor_count = neighbor_end - neighbor_begin;
        }
        size_t neighbor_limit = effective_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(effective_neighbor_cap, neighbor_count);
        if (effective_late_neighbor_cap > 0 &&
            effective_late_neighbor_after > 0 &&
            local_stats.node_expansions > effective_late_neighbor_after) {
            neighbor_limit = std::min(neighbor_limit, effective_late_neighbor_cap);
        }
        if (use_fixed_search_neighbors) {
            scan_fixed_neighbors(fixed_neighbors, neighbor_limit);
        } else {
            scan_graph_neighbors(
                current.id, search_indices, neighbor_begin, neighbor_limit, true, false
            );
        }
        if (scan_cold_for_current) {
            ++cold_expansions;
            ++local_stats.graph_cold_expansions;
            size_t cold_begin = graph_cold_offsets_[current.id];
            size_t cold_end = graph_cold_offsets_[static_cast<size_t>(current.id) + 1];
            size_t cold_count = cold_end - cold_begin;
            size_t cold_limit =
                std::min(config_.graph_cold_neighbor_count, cold_count);
            scan_graph_neighbors(
                current.id, graph_cold_indices_, cold_begin, cold_limit, false, true
            );
        }
    }
    auto graph_search_end = profile_now();
    local_stats.graph_search_ms = elapsed_ms(graph_search_begin, graph_search_end);

    local_stats.visited = seen.size();
    local_stats.jumps = local_stats.node_expansions;
    local_stats.graph_adaptive_ef = local_stats.node_expansions;

    auto final_begin = profile_now();
    std::vector<PID> result;
    bool needs_exact_rerank = config_.graph_search_use_quant || !query_u8.empty();
    if (needs_exact_rerank) {
        size_t rerank_keep = std::min(seen.size(), k);
        keep_smallest(seen, rerank_keep);

        std::vector<ScoredPid> exact_scores;
        exact_scores.reserve(rerank_keep);
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores.push_back(
                {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
            );
        }
        local_stats.exact_rerank_scores = rerank_keep;
        local_stats.final_candidates = rerank_keep;
        local_stats.exact_distance_evals += rerank_keep;
        keep_smallest(exact_scores, std::min(k, exact_scores.size()));

        result.reserve(std::min(k, exact_scores.size()));
        for (size_t i = 0; i < std::min(k, exact_scores.size()); ++i) {
            result.push_back(exact_scores[i].id);
        }
    } else {
        size_t rerank_keep = std::min(seen.size(), k);
        keep_smallest(seen, rerank_keep);
        local_stats.final_candidates = rerank_keep;

        result.reserve(std::min(k, seen.size()));
        for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
            result.push_back(seen[i].id);
        }
    }
    auto final_end = profile_now();
    local_stats.final_select_ms = elapsed_ms(final_begin, final_end);
    local_stats.query_total_ms = elapsed_ms(total_begin, final_end);

    if (stats != nullptr) {
        *stats = local_stats;
    }
    for (PID& id : result) {
        id = external_id(id);
    }
    return result;
}

inline size_t ATMMGGraphIndex::search_into(
    const float* query, size_t k, PID* out_ids
) const {
    if (query == nullptr || out_ids == nullptr || k == 0) {
        return 0;
    }
    if (num_ == 0) {
        throw std::runtime_error("ATMMGGraphIndex is not constructed");
    }
    return search_fast_into(query, k, out_ids);
}

inline bool ATMMGGraphIndex::uses_center_quant_refine() const {
    if (config_.center_flat_exact_scan || config_.center_super_level_scan ||
        config_.center_coarse_cascade_scan) {
        return false;
    }
    return config_.center_entry_mode == CenterEntryMode::QuantOnly ||
           config_.center_entry_mode == CenterEntryMode::TreeThenQuant;
}

inline bool ATMMGGraphIndex::can_use_exact_l2_light_fast_path() const {
    return !config_.graph_search_use_quant &&
           !config_.graph_search_use_u8_l2 && !config_.graph_distance_use_norm_dot &&
           !config_.graph_early_stop && config_.graph_adaptive_ef_min == 0 &&
           config_.graph_query_adaptive_ef_min == 0 &&
           !config_.hard_query_fallback &&
           !config_.graph_result_margin_stop && !config_.graph_admission_bound &&
           config_.graph_neighbor_prefilter_keep == 0 &&
           config_.graph_cold_neighbor_count == 0 && graph_hot_indices_.empty();
}

inline bool ATMMGGraphIndex::can_use_u8_l2_light_fast_path() const {
    return config_.graph_search_use_u8_l2 && !base_u8_.empty() &&
           !config_.graph_search_use_quant && !config_.graph_early_stop &&
           config_.graph_adaptive_ef_min == 0 && !config_.graph_result_margin_stop &&
           !config_.graph_admission_bound && config_.graph_neighbor_prefilter_keep == 0 &&
           config_.graph_cold_neighbor_count == 0 && graph_hot_indices_.empty() &&
           graph_fixed_neighbor_count_ > 0 && !graph_fixed_indices_.empty();
}

inline std::vector<PID> ATMMGGraphIndex::search_exact_l2_light_with_stats(
    const float* query,
    size_t k,
    const std::vector<PID>& seeds,
    PID best_center,
    ATMMGGraphQueryStats& local_stats
) const {
    using ProfileClock = std::chrono::high_resolution_clock;
    auto profile_now = []() { return ProfileClock::now(); };
    auto elapsed_ms = [](ProfileClock::time_point begin, ProfileClock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };

    auto graph_search_begin = profile_now();
    size_t neighbor_base_cap =
        config_.graph_search_neighbor_cap == 0 ? config_.graph_degree
                                               : config_.graph_search_neighbor_cap;
    size_t expected_visits =
        seeds.size() + (config_.ef_search * std::max<size_t>(1, neighbor_base_cap)) + 64;
    uint32_t visit_epoch = next_visit_epoch();

    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, expected_visits));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    const size_t top_capacity = std::max(k, config_.ef_search);
    std::priority_queue<ScoredPid> top_candidates;
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, expected_visits));
    const bool fixed_neighbor_ready =
        graph_fixed_neighbor_count_ > 0 &&
        !graph_fixed_indices_.empty();
    float lower_bound = std::numeric_limits<float>::max();
    float best_seen_distance = std::numeric_limits<float>::infinity();
    auto accept_candidate = [&](const ScoredPid& item) {
        if (top_candidates.size() < top_capacity) {
            top_candidates.push(item);
            lower_bound = top_candidates.top().distance;
            return true;
        }
        if (item.distance < lower_bound) {
            top_candidates.push(item);
            top_candidates.pop();
            lower_bound = top_candidates.top().distance;
            return true;
        }
        return false;
    };

    auto push_id = [&](PID id, bool is_seed) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ++local_stats.visited_nodes;
        ++local_stats.graph_distance_evals;
        ++local_stats.exact_distance_evals;
        ScoredPid item{l2_to_point(query, id), id};
        if (accept_candidate(item)) {
            bool improved = item.distance < best_seen_distance;
            if (improved) {
                best_seen_distance = item.distance;
            }
            candidate_queue.push(item);
            seen.push_back(item);
            ++local_stats.heap_pushes;
            if (is_seed) {
                ++local_stats.seed_pushes;
            }
            return improved;
        }
        return false;
    };

    for (PID id : seeds) {
        push_id(id, true);
    }
    if (candidate_queue.empty()) {
        ++local_stats.fallback_count;
        local_stats.fallback = true;
        local_stats.mode = SeedMode::Fallback;
        PID fallback = center_real_pool_[best_center].empty()
                           ? 0
                           : center_real_pool_[best_center][0];
        push_id(fallback, true);
    }

    while (!candidate_queue.empty() && local_stats.node_expansions < config_.ef_search) {
        ScoredPid current = candidate_queue.top();
        if (top_candidates.size() >= top_capacity && current.distance > lower_bound) {
            break;
        }
        candidate_queue.pop();
        ++local_stats.heap_pops;
        ++local_stats.node_expansions;

        size_t neighbor_begin = 0;
        size_t neighbor_count = 0;
        const PID* fixed_neighbors = nullptr;
        if (fixed_neighbor_ready) {
            fixed_neighbors =
                graph_fixed_indices_.data() +
                static_cast<size_t>(current.id) * graph_fixed_neighbor_count_;
            neighbor_count = graph_fixed_counts_[current.id];
        } else {
            neighbor_begin = graph_offsets_[current.id];
            size_t neighbor_end = graph_offsets_[static_cast<size_t>(current.id) + 1];
            neighbor_count = neighbor_end - neighbor_begin;
        }
        size_t neighbor_limit = config_.graph_search_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(config_.graph_search_neighbor_cap, neighbor_count);
        if (config_.graph_late_neighbor_cap > 0 &&
            config_.graph_late_neighbor_after > 0 &&
            local_stats.node_expansions > config_.graph_late_neighbor_after) {
            neighbor_limit = std::min(neighbor_limit, config_.graph_late_neighbor_cap);
        }
        for (size_t ni = 0; ni < neighbor_limit; ++ni) {
            ++local_stats.edges_scanned;
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < neighbor_limit) {
                    prefetch_point(
                        fixed_neighbor_ready ? fixed_neighbors[pf]
                                             : graph_indices_[neighbor_begin + pf]
                    );
                }
            }
            PID nb = fixed_neighbor_ready ? fixed_neighbors[ni]
                                          : graph_indices_[neighbor_begin + ni];
            push_id(nb, false);
        }
    }

    auto graph_search_end = profile_now();
    local_stats.graph_search_ms = elapsed_ms(graph_search_begin, graph_search_end);
    local_stats.visited = seen.size();
    local_stats.jumps = local_stats.node_expansions;
    local_stats.graph_adaptive_ef = local_stats.node_expansions;

    auto final_begin = profile_now();
    size_t rerank_keep = std::min(seen.size(), k);
    keep_smallest(seen, rerank_keep);
    local_stats.final_candidates = rerank_keep;

    std::vector<PID> result;
    result.reserve(std::min(k, seen.size()));
    for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
        result.push_back(seen[i].id);
    }
    auto final_end = profile_now();
    local_stats.final_select_ms = elapsed_ms(final_begin, final_end);
    return result;
}

inline std::vector<PID> ATMMGGraphIndex::search_exact_l2_light_fast(
    const float* query,
    size_t k,
    const std::vector<PID>& seeds,
    PID best_center
) const {
    size_t neighbor_base_cap =
        config_.graph_search_neighbor_cap == 0 ? config_.graph_degree
                                               : config_.graph_search_neighbor_cap;
    size_t expected_visits =
        seeds.size() + (config_.ef_search * std::max<size_t>(1, neighbor_base_cap)) + 64;
    uint32_t visit_epoch = next_visit_epoch();

    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, expected_visits));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    const size_t top_capacity = std::max(k, config_.ef_search);
    std::priority_queue<ScoredPid> top_candidates;
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, expected_visits));
    const bool fixed_neighbor_ready =
        graph_fixed_neighbor_count_ > 0 &&
        !graph_fixed_indices_.empty();
    float lower_bound = std::numeric_limits<float>::max();
    float best_seen_distance = std::numeric_limits<float>::infinity();
    auto accept_candidate = [&](const ScoredPid& item) {
        if (top_candidates.size() < top_capacity) {
            top_candidates.push(item);
            lower_bound = top_candidates.top().distance;
            return true;
        }
        if (item.distance < lower_bound) {
            top_candidates.push(item);
            top_candidates.pop();
            lower_bound = top_candidates.top().distance;
            return true;
        }
        return false;
    };

    auto push_id = [&](PID id) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ScoredPid item{l2_to_point(query, id), id};
        if (accept_candidate(item)) {
            bool improved = item.distance < best_seen_distance;
            if (improved) {
                best_seen_distance = item.distance;
            }
            candidate_queue.push(item);
            seen.push_back(item);
            return improved;
        }
        return false;
    };

    for (PID id : seeds) {
        push_id(id);
    }
    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty()
                           ? 0
                           : center_real_pool_[best_center][0];
        push_id(fallback);
    }

    size_t expansions = 0;
    while (!candidate_queue.empty() && expansions < config_.ef_search) {
        ScoredPid current = candidate_queue.top();
        if (top_candidates.size() >= top_capacity && current.distance > lower_bound) {
            break;
        }
        candidate_queue.pop();
        ++expansions;

        size_t neighbor_begin = 0;
        size_t neighbor_count = 0;
        const PID* fixed_neighbors = nullptr;
        if (fixed_neighbor_ready) {
            fixed_neighbors =
                graph_fixed_indices_.data() +
                static_cast<size_t>(current.id) * graph_fixed_neighbor_count_;
            neighbor_count = graph_fixed_counts_[current.id];
        } else {
            neighbor_begin = graph_offsets_[current.id];
            size_t neighbor_end = graph_offsets_[static_cast<size_t>(current.id) + 1];
            neighbor_count = neighbor_end - neighbor_begin;
        }
        size_t neighbor_limit = config_.graph_search_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(config_.graph_search_neighbor_cap, neighbor_count);
        if (config_.graph_late_neighbor_cap > 0 &&
            config_.graph_late_neighbor_after > 0 &&
            expansions > config_.graph_late_neighbor_after) {
            neighbor_limit = std::min(neighbor_limit, config_.graph_late_neighbor_cap);
        }
        for (size_t ni = 0; ni < neighbor_limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < neighbor_limit) {
                    prefetch_point(
                        fixed_neighbor_ready ? fixed_neighbors[pf]
                                             : graph_indices_[neighbor_begin + pf]
                    );
                }
            }
            PID nb = fixed_neighbor_ready ? fixed_neighbors[ni]
                                          : graph_indices_[neighbor_begin + ni];
            push_id(nb);
        }
    }

    size_t rerank_keep = std::min(seen.size(), k);
    keep_smallest(seen, rerank_keep);

    std::vector<PID> result;
    result.reserve(std::min(k, seen.size()));
    for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
        result.push_back(external_id(seen[i].id));
    }
    return result;
}

inline size_t ATMMGGraphIndex::search_exact_l2_light_fast_into(
    const float* query,
    size_t k,
    const std::vector<PID>& seeds,
    PID best_center,
    PID* out_ids
) const {
    size_t neighbor_base_cap =
        config_.graph_search_neighbor_cap == 0 ? config_.graph_degree
                                               : config_.graph_search_neighbor_cap;
    size_t expected_visits =
        seeds.size() + (config_.ef_search * std::max<size_t>(1, neighbor_base_cap)) + 64;
    uint32_t visit_epoch = next_visit_epoch();

    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, expected_visits));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    const size_t top_capacity = std::max(k, config_.ef_search);
    std::priority_queue<ScoredPid> top_candidates;
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, expected_visits));
    const bool fixed_neighbor_ready =
        graph_fixed_neighbor_count_ > 0 &&
        !graph_fixed_indices_.empty();
    float lower_bound = std::numeric_limits<float>::max();
    float best_seen_distance = std::numeric_limits<float>::infinity();
    auto accept_candidate = [&](const ScoredPid& item) {
        if (top_candidates.size() < top_capacity) {
            top_candidates.push(item);
            lower_bound = top_candidates.top().distance;
            return true;
        }
        if (item.distance < lower_bound) {
            top_candidates.push(item);
            top_candidates.pop();
            lower_bound = top_candidates.top().distance;
            return true;
        }
        return false;
    };

    auto push_id = [&](PID id) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ScoredPid item{l2_to_point(query, id), id};
        if (accept_candidate(item)) {
            bool improved = item.distance < best_seen_distance;
            if (improved) {
                best_seen_distance = item.distance;
            }
            candidate_queue.push(item);
            seen.push_back(item);
            return improved;
        }
        return false;
    };

    for (PID id : seeds) {
        push_id(id);
    }
    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty()
                           ? 0
                           : center_real_pool_[best_center][0];
        push_id(fallback);
    }

    size_t expansions = 0;
    while (!candidate_queue.empty() && expansions < config_.ef_search) {
        ScoredPid current = candidate_queue.top();
        if (top_candidates.size() >= top_capacity && current.distance > lower_bound) {
            break;
        }
        candidate_queue.pop();
        ++expansions;

        size_t neighbor_begin = 0;
        size_t neighbor_count = 0;
        const PID* fixed_neighbors = nullptr;
        if (fixed_neighbor_ready) {
            fixed_neighbors =
                graph_fixed_indices_.data() +
                static_cast<size_t>(current.id) * graph_fixed_neighbor_count_;
            neighbor_count = graph_fixed_counts_[current.id];
        } else {
            neighbor_begin = graph_offsets_[current.id];
            size_t neighbor_end = graph_offsets_[static_cast<size_t>(current.id) + 1];
            neighbor_count = neighbor_end - neighbor_begin;
        }
        size_t neighbor_limit = config_.graph_search_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(config_.graph_search_neighbor_cap, neighbor_count);
        if (config_.graph_late_neighbor_cap > 0 &&
            config_.graph_late_neighbor_after > 0 &&
            expansions > config_.graph_late_neighbor_after) {
            neighbor_limit = std::min(neighbor_limit, config_.graph_late_neighbor_cap);
        }
        for (size_t ni = 0; ni < neighbor_limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < neighbor_limit) {
                    prefetch_point(
                        fixed_neighbor_ready ? fixed_neighbors[pf]
                                             : graph_indices_[neighbor_begin + pf]
                    );
                }
            }
            PID nb = fixed_neighbor_ready ? fixed_neighbors[ni]
                                          : graph_indices_[neighbor_begin + ni];
            push_id(nb);
        }
    }

    size_t rerank_keep = std::min(seen.size(), k);
    keep_smallest(seen, rerank_keep);

    size_t out_count = std::min(k, seen.size());
    for (size_t i = 0; i < out_count; ++i) {
        out_ids[i] = external_id(seen[i].id);
    }
    return out_count;
}

inline std::vector<PID> ATMMGGraphIndex::search_u8_l2_light_fast(
    const float* query,
    float query_sq_norm,
    size_t k,
    const std::vector<PID>& seeds,
    PID best_center,
    float center_margin,
    size_t ef_override
) const {
    const bool hard_query = is_hard_query(center_margin);
    size_t effective_ef_max = effective_ef_for_query(query, center_margin);
    if (!hard_query && ef_override > 0) {
        effective_ef_max = std::max<size_t>(
            1,
            std::min(effective_ef_max, ef_override)
        );
    }
    const size_t effective_neighbor_cap =
        hard_query && config_.hard_query_neighbor_cap > 0
            ? config_.hard_query_neighbor_cap
            : config_.graph_search_neighbor_cap;
    const size_t effective_late_neighbor_cap =
        hard_query && config_.hard_query_late_neighbor_cap > 0
            ? config_.hard_query_late_neighbor_cap
            : config_.graph_late_neighbor_cap;
    const size_t effective_late_neighbor_after =
        hard_query && config_.hard_query_late_neighbor_after > 0
            ? config_.hard_query_late_neighbor_after
            : config_.graph_late_neighbor_after;

    size_t base_neighbor_limit = effective_neighbor_cap == 0
                                     ? graph_fixed_neighbor_count_
                                     : effective_neighbor_cap;
    base_neighbor_limit = std::min(base_neighbor_limit, graph_fixed_neighbor_count_);
    if (base_neighbor_limit == 0) {
        return {};
    }

    std::array<uint8_t, 256> query_u8_stack{};
    std::vector<uint8_t> query_u8_heap;
    const uint8_t* query_u8_data = nullptr;
    if (dim_ <= query_u8_stack.size()) {
        encode_query_u8_to_buffer(query, query_u8_stack.data());
        query_u8_data = query_u8_stack.data();
    } else {
        encode_query_u8(query, query_u8_heap);
        query_u8_data = query_u8_heap.data();
    }

    size_t expected_visits =
        seeds.size() + (effective_ef_max * std::max<size_t>(1, base_neighbor_limit)) + 64;
    uint32_t visit_epoch = next_visit_epoch();

    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, expected_visits));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, expected_visits));
    const bool residual_filter_ready = residual_hash_ready();
    std::array<float, 512> residual_prepared_query{};
    bool residual_query_prepared = false;

    size_t expansions = 0;
    const bool use_dual_scale =
        config_.graph_dual_scale_search &&
        graph_dual_short_neighbor_count_ > 0 &&
        graph_dual_long_neighbor_count_ > 0 &&
        !graph_dual_short_indices_.empty() &&
        !graph_dual_long_indices_.empty() &&
        graph_dual_short_radius_.size() == num_;
    size_t long_no_improve = 0;
    float best_seen_distance = std::numeric_limits<float>::infinity();

    auto push_id = [&](PID id) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ScoredPid item{u8_l2_to_point(query_u8_data, id), id};
        bool improved = item.distance < best_seen_distance;
        if (improved) {
            best_seen_distance = item.distance;
        }
        candidate_queue.push(item);
        seen.push_back(item);
        return improved;
    };

    for (PID id : seeds) {
        push_id(id);
    }
    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty()
                           ? 0
                           : center_real_pool_[best_center][0];
        push_id(fallback);
    }

    const size_t fixed_count = graph_fixed_neighbor_count_;
    while (!candidate_queue.empty() && expansions < effective_ef_max) {
        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++expansions;

        const PID* neighbors =
            graph_fixed_indices_.data() + (static_cast<size_t>(current.id) * fixed_count);
        size_t available_neighbors = graph_fixed_counts_[current.id];
        bool scan_long_edges = false;
        if (use_dual_scale && graph_dual_long_counts_[current.id] > 0 &&
            long_no_improve <= config_.graph_dual_long_no_improve_limit) {
            float radius = graph_dual_short_radius_[current.id];
            float u8_radius = graph_dual_short_u8_radius_.empty()
                                  ? 0.0F
                                  : graph_dual_short_u8_radius_[current.id];
            if (u8_radius > 0.0F) {
                scan_long_edges =
                    current.distance > u8_radius * config_.graph_dual_query_beta;
            } else if (radius > 0.0F) {
                float exact_current = l2_to_point(query, query_sq_norm, current.id);
                scan_long_edges =
                    exact_current > radius * config_.graph_dual_query_beta;
            }
        }
        float before_best = best_seen_distance;
        size_t neighbor_budget = base_neighbor_limit;
        if (effective_late_neighbor_cap > 0 &&
            effective_late_neighbor_after > 0 &&
            expansions > effective_late_neighbor_after) {
            neighbor_budget = std::min(neighbor_budget, effective_late_neighbor_cap);
        }
        size_t neighbor_limit = std::min<size_t>(available_neighbors, neighbor_budget);
        std::array<PID, 128> accepted{};
        size_t accepted_count = 0;
        bool used_residual_filter =
            !use_dual_scale &&
            residual_filter_ready &&
            (config_.residual_hash_filter_after == 0 ||
             expansions > config_.residual_hash_filter_after) &&
            !graph_offsets_.empty() && neighbor_limit > 0 &&
            neighbor_limit <= accepted.size();
        size_t edge_begin = 0;
        if (used_residual_filter) {
            if (!residual_query_prepared) {
                residual_hash_prepare_query(query, residual_prepared_query.data());
                residual_query_prepared = true;
            }
            edge_begin = graph_offsets_[current.id];
            const float* center =
                base_.data() + (static_cast<size_t>(current.id) * dim_);
            std::array<uint64_t, 8> query_code{};
            residual_hash_code_from_prepared_query(
                center, residual_prepared_query.data(), query_code.data()
            );
            for (size_t ni = 0; ni < neighbor_limit; ++ni) {
                size_t pos = edge_begin + ni;
                const uint64_t* edge_code =
                    residual_hash_codes_.data() + (pos * residual_hash_words_);
                if (residual_hash_pass(query_code.data(), edge_code)) {
                    accepted[accepted_count++] = neighbors[ni];
                }
            }
        }
        if (used_residual_filter && accepted_count > 0) {
            for (size_t i = 0; i < accepted_count; ++i) {
                push_id(accepted[i]);
            }
        } else {
            for (size_t ni = 0; ni < neighbor_limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < neighbor_limit) {
                        prefetch_u8_point(neighbors[pf]);
                    }
                }
                push_id(neighbors[ni]);
            }
        }
        if (scan_long_edges && use_dual_scale &&
            graph_dual_long_counts_[current.id] > 0) {
            const PID* long_neighbors =
                graph_dual_long_indices_.data() +
                (static_cast<size_t>(current.id) * graph_dual_long_neighbor_count_);
            size_t long_limit = graph_dual_long_counts_[current.id];
            for (size_t ni = 0; ni < long_limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < long_limit) {
                        prefetch_u8_point(long_neighbors[pf]);
                    }
                }
                push_id(long_neighbors[ni]);
            }
        }
        if (scan_long_edges) {
            if (best_seen_distance < before_best) {
                long_no_improve = 0;
            } else {
                ++long_no_improve;
            }
        }
    }

    size_t rerank_keep = std::min(seen.size(), k);
    keep_smallest(seen, rerank_keep);
    rerank_keep = apply_final_prefilter(query, k, seen, rerank_keep);

    constexpr size_t kStackExactScoresLimit = 256;
    if (rerank_keep <= kStackExactScoresLimit) {
        std::array<ScoredPid, kStackExactScoresLimit> exact_scores{};
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores[i] = {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id};
        }
        size_t out_count = std::min(k, rerank_keep);
        if (out_count < rerank_keep) {
            auto nth = exact_scores.begin() + static_cast<std::ptrdiff_t>(out_count);
            std::nth_element(exact_scores.begin(), nth, exact_scores.begin() + rerank_keep);
        }
        std::sort(exact_scores.begin(), exact_scores.begin() + out_count);
        if (!hard_query && config_.hard_query_fallback &&
            config_.hard_query_result_margin > 0.0F && out_count == k &&
            rerank_keep > k && k > 0) {
            float kth = std::max(exact_scores[out_count - 1].distance, 1e-6F);
            float margin = (exact_scores[out_count].distance -
                            exact_scores[out_count - 1].distance) /
                           kth;
            if (margin < config_.hard_query_result_margin) {
                return search_u8_l2_light_fast(
                    query, query_sq_norm, k, seeds, best_center, -1.0F
                );
            }
        }
        std::vector<PID> result;
        result.reserve(out_count);
        for (size_t i = 0; i < out_count; ++i) {
            result.push_back(external_id(exact_scores[i].id));
        }
        return result;
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(rerank_keep);
    for (size_t i = 0; i < rerank_keep; ++i) {
        exact_scores.push_back(
            {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
        );
    }
    if (!hard_query && config_.hard_query_fallback &&
        config_.hard_query_result_margin > 0.0F && exact_scores.size() > k &&
        k > 0) {
        auto next = exact_scores.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(exact_scores.begin(), next, exact_scores.end());
        auto kth = exact_scores.begin() + static_cast<std::ptrdiff_t>(k - 1);
        std::nth_element(exact_scores.begin(), kth, next);
        float denom = std::max(kth->distance, 1e-6F);
        float margin = (next->distance - kth->distance) / denom;
        if (margin < config_.hard_query_result_margin) {
            return search_u8_l2_light_fast(
                query, query_sq_norm, k, seeds, best_center, -1.0F
            );
        }
    }
    keep_smallest(exact_scores, std::min(k, exact_scores.size()));

    std::vector<PID> result;
    result.reserve(std::min(k, exact_scores.size()));
    for (size_t i = 0; i < std::min(k, exact_scores.size()); ++i) {
        result.push_back(external_id(exact_scores[i].id));
    }
    return result;
}

inline size_t ATMMGGraphIndex::search_u8_l2_light_fast_into(
    const float* query,
    float query_sq_norm,
    size_t k,
    const std::vector<PID>& seeds,
    PID best_center,
    float center_margin,
    PID* out_ids,
    size_t ef_override
) const {
    const bool hard_query = is_hard_query(center_margin);
    size_t effective_ef_max = effective_ef_for_query(query, center_margin);
    if (!hard_query && ef_override > 0) {
        effective_ef_max = std::max<size_t>(
            1,
            std::min(effective_ef_max, ef_override)
        );
    }
    const size_t effective_neighbor_cap =
        hard_query && config_.hard_query_neighbor_cap > 0
            ? config_.hard_query_neighbor_cap
            : config_.graph_search_neighbor_cap;
    const size_t effective_late_neighbor_cap =
        hard_query && config_.hard_query_late_neighbor_cap > 0
            ? config_.hard_query_late_neighbor_cap
            : config_.graph_late_neighbor_cap;
    const size_t effective_late_neighbor_after =
        hard_query && config_.hard_query_late_neighbor_after > 0
            ? config_.hard_query_late_neighbor_after
            : config_.graph_late_neighbor_after;

    size_t base_neighbor_limit = effective_neighbor_cap == 0
                                     ? graph_fixed_neighbor_count_
                                     : effective_neighbor_cap;
    base_neighbor_limit = std::min(base_neighbor_limit, graph_fixed_neighbor_count_);
    if (base_neighbor_limit == 0) {
        return 0;
    }

    std::array<uint8_t, 256> query_u8_stack{};
    std::vector<uint8_t> query_u8_heap;
    const uint8_t* query_u8_data = nullptr;
    if (dim_ <= query_u8_stack.size()) {
        encode_query_u8_to_buffer(query, query_u8_stack.data());
        query_u8_data = query_u8_stack.data();
    } else {
        encode_query_u8(query, query_u8_heap);
        query_u8_data = query_u8_heap.data();
    }

    size_t expected_visits =
        seeds.size() + (effective_ef_max * std::max<size_t>(1, base_neighbor_limit)) + 64;
    uint32_t visit_epoch = next_visit_epoch();

    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, expected_visits));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, expected_visits));
    const bool residual_filter_ready = residual_hash_ready();
    std::array<float, 512> residual_prepared_query{};
    bool residual_query_prepared = false;

    size_t expansions = 0;
    const bool use_dual_scale =
        config_.graph_dual_scale_search &&
        graph_dual_short_neighbor_count_ > 0 &&
        graph_dual_long_neighbor_count_ > 0 &&
        !graph_dual_short_indices_.empty() &&
        !graph_dual_long_indices_.empty() &&
        graph_dual_short_radius_.size() == num_;
    size_t long_no_improve = 0;
    float best_seen_distance = std::numeric_limits<float>::infinity();

    auto push_id = [&](PID id) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        ScoredPid item{u8_l2_to_point(query_u8_data, id), id};
        bool improved = item.distance < best_seen_distance;
        if (improved) {
            best_seen_distance = item.distance;
        }
        candidate_queue.push(item);
        seen.push_back(item);
        return improved;
    };

    for (PID id : seeds) {
        push_id(id);
    }
    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty()
                           ? 0
                           : center_real_pool_[best_center][0];
        push_id(fallback);
    }

    const size_t fixed_count = graph_fixed_neighbor_count_;
    while (!candidate_queue.empty() && expansions < effective_ef_max) {
        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++expansions;

        const PID* neighbors =
            graph_fixed_indices_.data() + (static_cast<size_t>(current.id) * fixed_count);
        size_t available_neighbors = graph_fixed_counts_[current.id];
        bool scan_long_edges = false;
        if (use_dual_scale && graph_dual_long_counts_[current.id] > 0 &&
            long_no_improve <= config_.graph_dual_long_no_improve_limit) {
            float radius = graph_dual_short_radius_[current.id];
            float u8_radius = graph_dual_short_u8_radius_.empty()
                                  ? 0.0F
                                  : graph_dual_short_u8_radius_[current.id];
            if (u8_radius > 0.0F) {
                scan_long_edges =
                    current.distance > u8_radius * config_.graph_dual_query_beta;
            } else if (radius > 0.0F) {
                float exact_current = l2_to_point(query, query_sq_norm, current.id);
                scan_long_edges =
                    exact_current > radius * config_.graph_dual_query_beta;
            }
        }
        float before_best = best_seen_distance;
        size_t neighbor_budget = base_neighbor_limit;
        if (effective_late_neighbor_cap > 0 &&
            effective_late_neighbor_after > 0 &&
            expansions > effective_late_neighbor_after) {
            neighbor_budget = std::min(neighbor_budget, effective_late_neighbor_cap);
        }
        size_t neighbor_limit = std::min<size_t>(available_neighbors, neighbor_budget);

        std::array<PID, 128> accepted{};
        size_t accepted_count = 0;
        bool used_residual_filter =
            !use_dual_scale &&
            residual_filter_ready &&
            (config_.residual_hash_filter_after == 0 ||
             expansions > config_.residual_hash_filter_after) &&
            !graph_offsets_.empty() && neighbor_limit > 0 &&
            neighbor_limit <= accepted.size();
        size_t edge_begin = 0;
        if (used_residual_filter) {
            if (!residual_query_prepared) {
                residual_hash_prepare_query(query, residual_prepared_query.data());
                residual_query_prepared = true;
            }
            edge_begin = graph_offsets_[current.id];
            const float* center =
                base_.data() + (static_cast<size_t>(current.id) * dim_);
            std::array<uint64_t, 8> query_code{};
            residual_hash_code_from_prepared_query(
                center, residual_prepared_query.data(), query_code.data()
            );
            for (size_t ni = 0; ni < neighbor_limit; ++ni) {
                size_t pos = edge_begin + ni;
                const uint64_t* edge_code =
                    residual_hash_codes_.data() + (pos * residual_hash_words_);
                if (residual_hash_pass(query_code.data(), edge_code)) {
                    accepted[accepted_count++] = neighbors[ni];
                }
            }
        }
        if (used_residual_filter && accepted_count > 0) {
            for (size_t i = 0; i < accepted_count; ++i) {
                push_id(accepted[i]);
            }
        } else {
            for (size_t ni = 0; ni < neighbor_limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < neighbor_limit) {
                        prefetch_u8_point(neighbors[pf]);
                    }
                }
                push_id(neighbors[ni]);
            }
        }
        if (scan_long_edges && use_dual_scale &&
            graph_dual_long_counts_[current.id] > 0) {
            const PID* long_neighbors =
                graph_dual_long_indices_.data() +
                (static_cast<size_t>(current.id) * graph_dual_long_neighbor_count_);
            size_t long_limit = graph_dual_long_counts_[current.id];
            for (size_t ni = 0; ni < long_limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < long_limit) {
                        prefetch_u8_point(long_neighbors[pf]);
                    }
                }
                push_id(long_neighbors[ni]);
            }
        }
        if (scan_long_edges) {
            if (best_seen_distance < before_best) {
                long_no_improve = 0;
            } else {
                ++long_no_improve;
            }
        }
    }

    size_t rerank_keep = std::min(seen.size(), k);
    keep_smallest(seen, rerank_keep);
    rerank_keep = apply_final_prefilter(query, k, seen, rerank_keep);

    constexpr size_t kStackExactScoresLimit = 256;
    if (rerank_keep <= kStackExactScoresLimit) {
        std::array<ScoredPid, kStackExactScoresLimit> exact_scores{};
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores[i] = {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id};
        }
        size_t out_count = std::min(k, rerank_keep);
        if (out_count < rerank_keep) {
            auto nth = exact_scores.begin() + static_cast<std::ptrdiff_t>(out_count);
            std::nth_element(exact_scores.begin(), nth, exact_scores.begin() + rerank_keep);
        }
        std::sort(exact_scores.begin(), exact_scores.begin() + out_count);
        if (!hard_query && config_.hard_query_fallback &&
            config_.hard_query_result_margin > 0.0F && out_count == k &&
            rerank_keep > k && k > 0) {
            float kth = std::max(exact_scores[out_count - 1].distance, 1e-6F);
            float margin = (exact_scores[out_count].distance -
                            exact_scores[out_count - 1].distance) /
                           kth;
            if (margin < config_.hard_query_result_margin) {
                return search_u8_l2_light_fast_into(
                    query, query_sq_norm, k, seeds, best_center, -1.0F, out_ids
                );
            }
        }
        for (size_t i = 0; i < out_count; ++i) {
            out_ids[i] = external_id(exact_scores[i].id);
        }
        return out_count;
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(rerank_keep);
    for (size_t i = 0; i < rerank_keep; ++i) {
        exact_scores.push_back(
            {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
        );
    }
    if (!hard_query && config_.hard_query_fallback &&
        config_.hard_query_result_margin > 0.0F && exact_scores.size() > k &&
        k > 0) {
        auto next = exact_scores.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(exact_scores.begin(), next, exact_scores.end());
        auto kth = exact_scores.begin() + static_cast<std::ptrdiff_t>(k - 1);
        std::nth_element(exact_scores.begin(), kth, next);
        float denom = std::max(kth->distance, 1e-6F);
        float margin = (next->distance - kth->distance) / denom;
        if (margin < config_.hard_query_result_margin) {
            return search_u8_l2_light_fast_into(
                query, query_sq_norm, k, seeds, best_center, -1.0F, out_ids
            );
        }
    }
    keep_smallest(exact_scores, std::min(k, exact_scores.size()));

    size_t out_count = std::min(k, exact_scores.size());
    for (size_t i = 0; i < out_count; ++i) {
        out_ids[i] = external_id(exact_scores[i].id);
    }
    return out_count;
}

inline std::vector<PID> ATMMGGraphIndex::search_fast(
    const float* query, size_t k
) const {
    float query_sq_norm = squared_norm(query, dim_);
    std::vector<float> rotated_query;
    auto ensure_rotated_query = [&]() -> const float* {
        if (rotated_query.empty()) {
            rotated_query.assign(padded_dim_, 0.0F);
            rotator_->rotate(query, rotated_query.data());
        }
        return rotated_query.data();
    };
    std::unique_ptr<SplitSingleQuery<float>> query_wrapper;
    auto ensure_query_wrapper = [&]() -> SplitSingleQuery<float>& {
        if (query_wrapper == nullptr) {
            query_wrapper = std::make_unique<SplitSingleQuery<float>>(
                ensure_rotated_query(), padded_dim_, ex_bits_, query_config_, METRIC_L2
            );
        }
        return *query_wrapper;
    };

    PID routed_center = 0;
    if (config_.center_entry_mode != CenterEntryMode::QuantOnly) {
        routed_center = route_entry_center(query);
    }
    PID best_center = routed_center;
    std::vector<PID> candidate_centers;
    float center_margin = std::numeric_limits<float>::quiet_NaN();
    if (config_.center_entry_mode == CenterEntryMode::QuantOnly) {
        best_center = uses_center_quant_refine()
                          ? refine_center_fast(
                                query, ensure_query_wrapper(), routed_center, false,
                                &candidate_centers, &center_margin
                            )
                          : refine_center_fast_without_quant(
                                query, routed_center, false, &candidate_centers,
                                &center_margin
                            );
    } else if (config_.center_entry_mode == CenterEntryMode::TreeThenQuant) {
        best_center = uses_center_quant_refine()
                          ? refine_center_fast(
                                query, ensure_query_wrapper(), routed_center, true,
                                &candidate_centers, &center_margin
                            )
                          : refine_center_fast_without_quant(
                                query, routed_center, true, &candidate_centers,
                                &center_margin
                            );
    }
    if (candidate_centers.empty()) {
        candidate_centers.push_back(best_center);
    }
    float best_center_dist2 = l2_to_center(query, best_center);

    std::vector<PID> seeds;
    if (best_center < center_real_pool_.size()) {
        const auto& pool = center_real_pool_[best_center];
        size_t take = std::min(config_.center_real_pool_take, pool.size());
        seeds.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(take));
    }
    if (seeds.empty()) {
        seeds.push_back(0);
    }
    const size_t bucket_ef_override = 0;
    if (can_use_exact_l2_light_fast_path()) {
        return search_exact_l2_light_fast(query, k, seeds, best_center);
    }
    if (can_use_u8_l2_light_fast_path()) {
        return search_u8_l2_light_fast(
            query, query_sq_norm, k, seeds, best_center, center_margin,
            bucket_ef_override
        );
    }

    std::vector<float> center_query_dist2;
    std::vector<float> center_query_norm;
    bool use_admission_bound =
        config_.graph_admission_bound && !point_center_residual_norms_.empty();
    bool needs_center_query_distance =
        config_.graph_search_use_quant || use_admission_bound;
    if (needs_center_query_distance) {
        size_t centers_count = num_centers();
        center_query_dist2.resize(centers_count);
        center_query_norm.resize(centers_count);
        if (config_.graph_lazy_center_distance || use_admission_bound ||
            !config_.graph_search_use_quant) {
            std::fill(center_query_dist2.begin(), center_query_dist2.end(), -1.0F);
            std::fill(center_query_norm.begin(), center_query_norm.end(), 0.0F);
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        } else {
            for (size_t c = 0; c < centers_count; ++c) {
                float d2 = l2_to_center(query, static_cast<PID>(c));
                center_query_dist2[c] = d2;
                center_query_norm[c] = std::sqrt(std::max(d2, 0.0F));
            }
            center_query_dist2[best_center] = best_center_dist2;
            center_query_norm[best_center] = std::sqrt(std::max(best_center_dist2, 0.0F));
        }
    }
    std::vector<uint8_t> query_u8;
    if (config_.graph_search_use_u8_l2 && !base_u8_.empty() &&
        !config_.graph_search_use_quant) {
        encode_query_u8(query, query_u8);
    }

    auto ensure_center_query_distance = [&](PID center_id) {
        if (center_query_dist2.empty()) {
            return;
        }
        if (center_query_dist2[center_id] < 0.0F) {
            float d2 = l2_to_center(query, center_id);
            center_query_dist2[center_id] = d2;
            center_query_norm[center_id] = std::sqrt(std::max(d2, 0.0F));
        }
    };

    auto graph_distance = [&](PID id) {
        if (!config_.graph_search_use_quant) {
            if (!query_u8.empty()) {
                return u8_l2_to_point(query_u8.data(), id);
            }
            return l2_to_point(query, query_sq_norm, id);
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        if (config_.graph_search_full_quant) {
            return estimate_code(
                point_codes_.bin(id),
                point_codes_.ex(id),
                ensure_query_wrapper(),
                center_query_dist2[center_id],
                center_query_norm[center_id]
            );
        }
        return estimate_code_onebit(
            point_codes_.bin(id),
            ensure_query_wrapper(),
            center_query_dist2[center_id],
            center_query_norm[center_id]
        );
    };

    const size_t effective_ef_max = effective_ef_for_query(query, center_margin);
    const bool hard_query = is_hard_query(center_margin);
    const size_t effective_neighbor_cap =
        hard_query && config_.hard_query_neighbor_cap > 0
            ? config_.hard_query_neighbor_cap
            : config_.graph_search_neighbor_cap;
    const size_t effective_late_neighbor_cap =
        hard_query && config_.hard_query_late_neighbor_cap > 0
            ? config_.hard_query_late_neighbor_cap
            : config_.graph_late_neighbor_cap;
    const size_t effective_late_neighbor_after =
        hard_query && config_.hard_query_late_neighbor_after > 0
            ? config_.hard_query_late_neighbor_after
            : config_.graph_late_neighbor_after;
    const bool use_adaptive_ef =
        config_.graph_adaptive_ef_min > 0 &&
        config_.graph_adaptive_ef_min < effective_ef_max;
    uint32_t visit_epoch = next_visit_epoch();
    std::vector<ScoredPid> queue_storage;
    queue_storage.reserve(std::min(num_, effective_ef_max * config_.graph_degree));
    std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
        candidate_queue(std::greater<ScoredPid>(), std::move(queue_storage));
    std::vector<ScoredPid> seen;
    seen.reserve(std::min(num_, effective_ef_max * config_.graph_degree));
    size_t window_keep = k;
    std::priority_queue<ScoredPid> best_window;
    bool track_window =
        config_.graph_early_stop || config_.graph_cold_neighbor_count > 0 ||
        use_admission_bound;
    float best_seen_distance = std::numeric_limits<float>::infinity();

    auto push_seen = [&](ScoredPid item) {
        seen.push_back(item);
        best_seen_distance = std::min(best_seen_distance, item.distance);
        if (track_window) {
            best_window.push(item);
            if (best_window.size() > window_keep) {
                best_window.pop();
            }
        }
    };

    for (PID id : seeds) {
        if (id >= num_ || visit_marks_[id] == visit_epoch) {
            continue;
        }
        visit_marks_[id] = visit_epoch;
        float dist = graph_distance(id);
        candidate_queue.push({dist, id});
        push_seen({dist, id});
    }

    if (candidate_queue.empty()) {
        PID fallback = center_real_pool_[best_center].empty() ? 0 : center_real_pool_[best_center][0];
        visit_marks_[fallback] = visit_epoch;
        float dist = graph_distance(fallback);
        candidate_queue.push({dist, fallback});
        push_seen({dist, fallback});
    }

    bool use_hot_cold = !graph_hot_indices_.empty();
    const std::vector<size_t>& search_offsets =
        use_hot_cold ? graph_hot_offsets_ : graph_offsets_;
    const std::vector<PID>& search_indices =
        use_hot_cold ? graph_hot_indices_ : graph_indices_;
    bool use_cold_fallback =
        use_hot_cold && config_.graph_cold_neighbor_count > 0 &&
        config_.graph_cold_max_expansions > 0 && !graph_cold_indices_.empty();
    size_t cold_expansions = 0;

    auto should_scan_cold = [&](const ScoredPid& current) {
        if (!use_cold_fallback) {
            return false;
        }
        if (cold_expansions >= config_.graph_cold_max_expansions) {
            return false;
        }
        if (best_window.size() < window_keep) {
            return true;
        }
        return current.distance <=
               best_window.top().distance * config_.graph_cold_search_slack;
    };

    auto reject_by_admission_bound = [&](PID id) {
        if (!use_admission_bound || best_window.size() < window_keep) {
            return false;
        }
        PID center_id = point_center_[id];
        ensure_center_query_distance(center_id);
        float center_norm = center_query_norm[center_id];
        float residual_norm = point_center_residual_norms_[id];
        float lower = std::fabs(center_norm - residual_norm);
        float lower_sq = lower * lower;
        if (lower_sq <= best_window.top().distance * config_.graph_admission_slack) {
            return false;
        }
        visit_marks_[id] = visit_epoch;
        return true;
    };

    bool use_neighbor_prefilter =
        config_.graph_neighbor_prefilter_keep > 0 && !graph_prefilter_dims_.empty();
    bool use_result_margin_stop =
        config_.graph_result_margin_stop && !config_.graph_search_use_quant &&
        query_u8.empty();
    std::vector<ScoredPid> neighbor_prefilter;
    if (use_neighbor_prefilter) {
        size_t reserve_cap = effective_neighbor_cap == 0
                                 ? config_.graph_degree
                                 : effective_neighbor_cap;
        neighbor_prefilter.reserve(std::min(config_.graph_neighbor_prefilter_keep, reserve_cap));
    }
    size_t expansions = 0;
    std::unique_ptr<BatchQuery<float>> graph_batch_query;
    auto ensure_graph_batch_query = [&]() -> BatchQuery<float>& {
        if (graph_batch_query == nullptr) {
            graph_batch_query =
                std::make_unique<BatchQuery<float>>(ensure_rotated_query(), padded_dim_);
        }
        return *graph_batch_query;
    };
    auto push_graph_neighbor_with_distance = [&](PID nb, float dist) {
        visit_marks_[nb] = visit_epoch;
        candidate_queue.push({dist, nb});
        bool improved = dist < best_seen_distance;
        push_seen({dist, nb});
        return improved;
    };
    auto push_graph_neighbor = [&](PID nb) {
        return push_graph_neighbor_with_distance(nb, graph_distance(nb));
    };

    const bool residual_filter_ready = residual_hash_ready();
    std::array<float, 512> residual_prepared_query{};
    bool residual_query_prepared = false;
    const bool use_fixed_search_neighbors =
        !graph_scan_ready() && !use_hot_cold && !use_neighbor_prefilter &&
        !residual_filter_ready &&
        graph_fixed_neighbor_count_ > 0 && !graph_fixed_indices_.empty();

    auto scan_fixed_neighbors = [&](const PID* neighbors, size_t limit) {
        for (size_t ni = 0; ni < limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < limit) {
                    prefetch_point(neighbors[pf]);
                }
            }
            PID nb = neighbors[ni];
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            if (reject_by_admission_bound(nb)) {
                continue;
            }
            push_graph_neighbor(nb);
        }
    };

    auto scan_graph_neighbors = [&](
                                    PID center_id,
                                    const std::vector<PID>& indices,
                                    size_t begin,
                                    size_t limit,
                                    bool allow_prefilter
                                ) {
        bool prefilter =
            allow_prefilter && use_neighbor_prefilter &&
            config_.graph_neighbor_prefilter_keep < limit;
        if (prefilter) {
            neighbor_prefilter.clear();
            size_t keep = std::min(config_.graph_neighbor_prefilter_keep, limit);
            for (size_t ni = 0; ni < limit; ++ni) {
                if (config_.graph_neighbor_prefetch > 0) {
                    size_t pf = ni + config_.graph_neighbor_prefetch;
                    if (pf < limit) {
                        prefetch_point(indices[begin + pf]);
                    }
                }
                PID nb = indices[begin + ni];
                if (visit_marks_[nb] == visit_epoch) {
                    continue;
                }
                if (reject_by_admission_bound(nb)) {
                    continue;
                }
                push_topk_smallest(
                    neighbor_prefilter, {graph_prefilter_l2_to_point(query, nb), nb}, keep
                );
            }
            finish_topk_smallest(neighbor_prefilter);
            for (const auto& cand : neighbor_prefilter) {
                if (visit_marks_[cand.id] == visit_epoch) {
                    continue;
                }
                push_graph_neighbor(cand.id);
            }
            return;
        }

        std::array<PID, 128> accepted{};
        size_t accepted_count = 0;
        bool used_residual_filter =
            residual_filter_ready &&
            (config_.residual_hash_filter_after == 0 ||
             expansions > config_.residual_hash_filter_after) &&
            !use_hot_cold &&
            indices.data() == graph_indices_.data() && limit > 0 &&
            limit <= accepted.size();
        if (used_residual_filter) {
            if (!residual_query_prepared) {
                residual_hash_prepare_query(query, residual_prepared_query.data());
                residual_query_prepared = true;
            }
            const float* center =
                base_.data() + (static_cast<size_t>(center_id) * dim_);
            std::array<uint64_t, 8> query_code{};
            residual_hash_code_from_prepared_query(
                center, residual_prepared_query.data(), query_code.data()
            );
            for (size_t ni = 0; ni < limit; ++ni) {
                size_t pos = begin + ni;
                PID nb = indices[pos];
                if (visit_marks_[nb] == visit_epoch) {
                    continue;
                }
                if (reject_by_admission_bound(nb)) {
                    continue;
                }
                const uint64_t* edge_code =
                    residual_hash_codes_.data() + (pos * residual_hash_words_);
                if (residual_hash_pass(query_code.data(), edge_code)) {
                    accepted[accepted_count++] = nb;
                }
            }
            if (accepted_count > 0) {
                for (size_t i = 0; i < accepted_count; ++i) {
                    push_graph_neighbor(accepted[i]);
                }
                return;
            }
        }

        bool can_scan_row =
            graph_scan_ready() && limit > 0 && !used_residual_filter &&
            center_id < num_ && indices.data() == graph_indices_.data() &&
            begin == graph_offsets_[center_id];
        if (can_scan_row) {
            BatchQuery<float>& q_obj = ensure_graph_batch_query();
            q_obj.set_g_add(l2_to_point(query, query_sq_norm, center_id));

            std::array<float, scan::kBatchSize> est_dist{};
            size_t batch_base = graph_edge_batch_offsets_[center_id];
            for (size_t offset = 0; offset < limit; offset += scan::kBatchSize) {
                size_t count = std::min(scan::kBatchSize, limit - offset);
                size_t batch_id = batch_base + (offset / scan::kBatchSize);
                qg_batch_estdist(
                    graph_edge_batch_codes_.batch(batch_id),
                    q_obj,
                    padded_dim_,
                    est_dist.data()
                );

                for (size_t j = 0; j < count; ++j) {
                    size_t ni = offset + j;
                    if (config_.graph_neighbor_prefetch > 0) {
                        size_t pf = ni + config_.graph_neighbor_prefetch;
                        if (pf < limit) {
                            prefetch_point(indices[begin + pf]);
                        }
                    }
                    PID nb = indices[begin + ni];
                    if (visit_marks_[nb] == visit_epoch) {
                        continue;
                    }
                    if (reject_by_admission_bound(nb)) {
                        continue;
                    }
                    push_graph_neighbor_with_distance(nb, est_dist[j]);
                }
            }
            return;
        }

        for (size_t ni = 0; ni < limit; ++ni) {
            if (config_.graph_neighbor_prefetch > 0) {
                size_t pf = ni + config_.graph_neighbor_prefetch;
                if (pf < limit) {
                    prefetch_point(indices[begin + pf]);
                }
            }
            PID nb = indices[begin + ni];
            if (visit_marks_[nb] == visit_epoch) {
                continue;
            }
            if (reject_by_admission_bound(nb)) {
                continue;
            }
            push_graph_neighbor(nb);
        }
    };

    auto should_stop_adaptive_ef = [&]() {
        if (!use_adaptive_ef || expansions < config_.graph_adaptive_ef_min ||
            seen.size() < window_keep || candidate_queue.empty()) {
            return false;
        }
        size_t after_min = expansions - config_.graph_adaptive_ef_min;
        if ((after_min % config_.graph_adaptive_ef_check_interval) != 0) {
            return false;
        }
        auto nth = seen.begin() + static_cast<std::ptrdiff_t>(window_keep - 1);
        std::nth_element(seen.begin(), nth, seen.end());
        return candidate_queue.top().distance >
               nth->distance * config_.graph_adaptive_ef_slack;
    };

    auto should_stop_result_margin = [&]() {
        if (!use_result_margin_stop || k == 0 ||
            expansions < config_.graph_result_margin_min_expansions ||
            seen.size() <= k) {
            return false;
        }
        size_t after_min = expansions - config_.graph_result_margin_min_expansions;
        if ((after_min % config_.graph_result_margin_check_interval) != 0) {
            return false;
        }
        auto next = seen.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(seen.begin(), next, seen.end());
        auto kth = seen.begin() + static_cast<std::ptrdiff_t>(k - 1);
        std::nth_element(seen.begin(), kth, next);
        float denom = std::max(kth->distance, 1e-6F);
        return ((next->distance - kth->distance) / denom) >=
               config_.graph_result_margin;
    };

    while (!candidate_queue.empty() && expansions < effective_ef_max) {
        if (config_.graph_early_stop &&
            expansions >= config_.graph_early_stop_min_expansions &&
            best_window.size() >= window_keep) {
            float bound = best_window.top().distance * config_.graph_early_stop_slack;
            if (candidate_queue.top().distance > bound) {
                break;
            }
        }
        if (should_stop_adaptive_ef()) {
            break;
        }
        if (should_stop_result_margin()) {
            break;
        }

        ScoredPid current = candidate_queue.top();
        candidate_queue.pop();
        ++expansions;
        bool scan_cold_for_current = should_scan_cold(current);

        size_t neighbor_begin = 0;
        size_t neighbor_count = 0;
        const PID* fixed_neighbors = nullptr;
        if (use_fixed_search_neighbors) {
            fixed_neighbors =
                graph_fixed_indices_.data() +
                static_cast<size_t>(current.id) * graph_fixed_neighbor_count_;
            neighbor_count = graph_fixed_counts_[current.id];
        } else {
            neighbor_begin = search_offsets[current.id];
            size_t neighbor_end = search_offsets[static_cast<size_t>(current.id) + 1];
            neighbor_count = neighbor_end - neighbor_begin;
        }
        size_t neighbor_limit = effective_neighbor_cap == 0
                                    ? neighbor_count
                                    : std::min(effective_neighbor_cap, neighbor_count);
        if (effective_late_neighbor_cap > 0 &&
            effective_late_neighbor_after > 0 &&
            expansions > effective_late_neighbor_after) {
            neighbor_limit = std::min(neighbor_limit, effective_late_neighbor_cap);
        }
        if (use_fixed_search_neighbors) {
            scan_fixed_neighbors(fixed_neighbors, neighbor_limit);
        } else {
            scan_graph_neighbors(current.id, search_indices, neighbor_begin, neighbor_limit, true);
        }
        if (scan_cold_for_current) {
            ++cold_expansions;
            size_t cold_begin = graph_cold_offsets_[current.id];
            size_t cold_end = graph_cold_offsets_[static_cast<size_t>(current.id) + 1];
            size_t cold_count = cold_end - cold_begin;
            size_t cold_limit =
                std::min(config_.graph_cold_neighbor_count, cold_count);
            scan_graph_neighbors(current.id, graph_cold_indices_, cold_begin, cold_limit, false);
        }
    }

    std::vector<PID> result;
    bool needs_exact_rerank = config_.graph_search_use_quant || !query_u8.empty();
    if (needs_exact_rerank) {
        size_t rerank_keep = std::min(seen.size(), k);
        keep_smallest(seen, rerank_keep);

        std::vector<ScoredPid> exact_scores;
        exact_scores.reserve(rerank_keep);
        for (size_t i = 0; i < rerank_keep; ++i) {
            exact_scores.push_back(
                {l2_to_point(query, query_sq_norm, seen[i].id), seen[i].id}
            );
        }
        keep_smallest(exact_scores, std::min(k, exact_scores.size()));

        result.reserve(std::min(k, exact_scores.size()));
        for (size_t i = 0; i < std::min(k, exact_scores.size()); ++i) {
            result.push_back(exact_scores[i].id);
        }
    } else {
        size_t rerank_keep = std::min(seen.size(), k);
        keep_smallest(seen, rerank_keep);

        result.reserve(std::min(k, seen.size()));
        for (size_t i = 0; i < std::min(k, seen.size()); ++i) {
            result.push_back(seen[i].id);
        }
    }

    for (PID& id : result) {
        id = external_id(id);
    }
    return result;
}

inline size_t ATMMGGraphIndex::search_fast_into(
    const float* query, size_t k, PID* out_ids
) const {
    const bool use_exact_l2_light = can_use_exact_l2_light_fast_path();
    const bool use_u8_l2_light = can_use_u8_l2_light_fast_path();
    if (!use_exact_l2_light && !use_u8_l2_light) {
        std::vector<PID> ids = search_fast(query, k);
        size_t out_count = std::min(k, ids.size());
        for (size_t i = 0; i < out_count; ++i) {
            out_ids[i] = ids[i];
        }
        return out_count;
    }

    float query_sq_norm = squared_norm(query, dim_);
    std::vector<float> rotated_query;
    auto ensure_rotated_query = [&]() -> const float* {
        if (rotated_query.empty()) {
            rotated_query.assign(padded_dim_, 0.0F);
            rotator_->rotate(query, rotated_query.data());
        }
        return rotated_query.data();
    };
    std::unique_ptr<SplitSingleQuery<float>> query_wrapper;
    auto ensure_query_wrapper = [&]() -> SplitSingleQuery<float>& {
        if (query_wrapper == nullptr) {
            query_wrapper = std::make_unique<SplitSingleQuery<float>>(
                ensure_rotated_query(), padded_dim_, ex_bits_, query_config_, METRIC_L2
            );
        }
        return *query_wrapper;
    };

    PID routed_center = 0;
    if (config_.center_entry_mode != CenterEntryMode::QuantOnly) {
        routed_center = route_entry_center(query);
    }
    PID best_center = routed_center;
    std::vector<PID> candidate_centers;
    float center_margin = std::numeric_limits<float>::quiet_NaN();
    if (config_.center_entry_mode == CenterEntryMode::QuantOnly) {
        best_center = uses_center_quant_refine()
                          ? refine_center_fast(
                                query, ensure_query_wrapper(), routed_center, false,
                                &candidate_centers, &center_margin
                            )
                          : refine_center_fast_without_quant(
                                query, routed_center, false, &candidate_centers,
                                &center_margin
                            );
    } else if (config_.center_entry_mode == CenterEntryMode::TreeThenQuant) {
        best_center = uses_center_quant_refine()
                          ? refine_center_fast(
                                query, ensure_query_wrapper(), routed_center, true,
                                &candidate_centers, &center_margin
                            )
                          : refine_center_fast_without_quant(
                                query, routed_center, true, &candidate_centers,
                                &center_margin
                            );
    }
    if (candidate_centers.empty()) {
        candidate_centers.push_back(best_center);
    }

    std::vector<PID> seeds;
    if (best_center < center_real_pool_.size()) {
        const auto& pool = center_real_pool_[best_center];
        size_t take = std::min(config_.center_real_pool_take, pool.size());
        seeds.assign(pool.begin(), pool.begin() + static_cast<std::ptrdiff_t>(take));
    }
    if (seeds.empty()) {
        seeds.push_back(0);
    }
    const size_t bucket_ef_override = 0;
    if (use_exact_l2_light) {
        return search_exact_l2_light_fast_into(query, k, seeds, best_center, out_ids);
    }
    if (use_u8_l2_light) {
        return search_u8_l2_light_fast_into(
            query, query_sq_norm, k, seeds, best_center, center_margin, out_ids,
            bucket_ef_override
        );
    }
    return 0;
}
