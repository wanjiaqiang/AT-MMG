inline size_t ATMMGGraphIndex::adaptive_center_budget(
    size_t default_budget,
    size_t easy_budget,
    size_t mid_budget,
    float margin
) const {
    if (!config_.center_adaptive_refine || default_budget == 0 ||
        !std::isfinite(margin)) {
        return default_budget;
    }
    size_t easy = easy_budget == 0 ? default_budget : easy_budget;
    size_t mid = mid_budget == 0 ? ((easy + default_budget) / 2) : mid_budget;
    easy = std::max<size_t>(1, std::min(easy, default_budget));
    mid = std::max(easy, std::min(mid, default_budget));
    if (margin >= config_.center_adaptive_easy_margin) {
        return easy;
    }
    if (margin >= config_.center_adaptive_hard_margin) {
        return mid;
    }
    return default_budget;
}

inline bool ATMMGGraphIndex::is_hard_query(float center_margin) const {
    if (!config_.hard_query_fallback || !std::isfinite(center_margin)) {
        return false;
    }
    float margin = config_.hard_query_center_margin > 0.0F
                       ? config_.hard_query_center_margin
                       : config_.graph_query_adaptive_hard_margin;
    return center_margin < margin;
}

inline std::vector<ATMMGGraphIndex::ScoredPid>
ATMMGGraphIndex::coarse_center_candidates(
    const float* query,
    size_t centers_count,
    size_t coarse_keep,
    float* coarse_margin
) const {
    coarse_keep = std::min(coarse_keep, centers_count);
    size_t full_dims = center_coarse_dims_.size();
    size_t prefilter_dims =
        std::min(config_.center_coarse_prefilter_dims, full_dims);
    size_t prefilter_keep =
        std::min(config_.center_coarse_prefilter_keep, centers_count);
    bool use_prefilter =
        prefilter_dims > 0 && prefilter_dims < full_dims &&
        prefilter_keep > coarse_keep && prefilter_keep < centers_count;
    std::vector<float> query_coarse_values;
    if (!center_coarse_values_.empty() && full_dims > 0) {
        query_coarse_values.resize(full_dims);
        for (size_t i = 0; i < full_dims; ++i) {
            query_coarse_values[i] = query[center_coarse_dims_[i]];
        }
    }
    auto coarse_distance = [&](PID center_id, size_t dims_count) {
        if (query_coarse_values.empty()) {
            return coarse_l2_to_center(query, center_id, dims_count);
        }
        dims_count = std::min(dims_count, full_dims);
        const float* center_vec = center_coarse_values_.data() +
                                  (static_cast<size_t>(center_id) * full_dims);
        float sum = 0.0F;
        for (size_t i = 0; i < dims_count; ++i) {
            float diff = query_coarse_values[i] - center_vec[i];
            sum += diff * diff;
        }
        return sum;
    };

    std::vector<ScoredPid> candidates;
    float coarse_best = std::numeric_limits<float>::infinity();
    float coarse_second = std::numeric_limits<float>::infinity();
    auto push_full_score = [&](PID center_id) {
        float coarse = coarse_distance(center_id, full_dims);
        candidates.push_back({coarse, center_id});
        if (coarse < coarse_best) {
            coarse_second = coarse_best;
            coarse_best = coarse;
        } else if (coarse < coarse_second) {
            coarse_second = coarse;
        }
    };

    if (use_prefilter) {
        std::vector<ScoredPid> prefilter_scores;
        prefilter_scores.reserve(centers_count);
        for (size_t c = 0; c < centers_count; ++c) {
            prefilter_scores.push_back(
                {coarse_distance(static_cast<PID>(c), prefilter_dims), static_cast<PID>(c)}
            );
        }
        keep_smallest(prefilter_scores, prefilter_keep);
        candidates.reserve(prefilter_scores.size());
        for (const auto& cand : prefilter_scores) {
            push_full_score(cand.id);
        }
    } else {
        candidates.reserve(centers_count);
        for (size_t c = 0; c < centers_count; ++c) {
            push_full_score(static_cast<PID>(c));
        }
    }

    if (coarse_margin != nullptr) {
        if (std::isfinite(coarse_second)) {
            float best = std::max(coarse_best, 1e-6F);
            *coarse_margin = (coarse_second - coarse_best) / best;
        } else {
            *coarse_margin = std::numeric_limits<float>::quiet_NaN();
        }
    }
    keep_smallest(candidates, coarse_keep);
    return candidates;
}

inline PID ATMMGGraphIndex::refine_center_exact_scan(
    const float* query,
    ATMMGGraphQueryStats* stats,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    const size_t centers_count = num_centers();
    const size_t keep = std::min(
        centers_count,
        std::max<size_t>(2, config_.exact_center_keep)
    );
    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(keep);
    for (size_t c = 0; c < centers_count; ++c) {
        PID center_id = static_cast<PID>(c);
        push_topk_smallest(
            exact_scores,
            {l2_to_center(query, center_id), center_id},
            keep
        );
    }
    finish_topk_smallest(exact_scores);

    if (stats != nullptr) {
        stats->center_estimates = 0;
        stats->center_exact_scores = centers_count;
        stats->exact_distance_evals += centers_count;
    }
    if (center_margin != nullptr) {
        if (exact_scores.size() >= 2) {
            float best = std::max(exact_scores[0].distance, 1e-6F);
            *center_margin = (exact_scores[1].distance - exact_scores[0].distance) / best;
        } else {
            *center_margin = std::numeric_limits<float>::infinity();
        }
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        size_t out_keep = std::min(config_.exact_center_keep, exact_scores.size());
        exact_centers->reserve(out_keep);
        for (size_t i = 0; i < out_keep; ++i) {
            exact_centers->push_back(exact_scores[i].id);
        }
    }
    return exact_scores.empty() ? 0 : exact_scores.front().id;
}

inline PID ATMMGGraphIndex::refine_center_super_level_scan(
    const float* query,
    PID routed_center,
    bool use_routed_center,
    ATMMGGraphQueryStats* stats,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    const size_t centers_count = num_centers();
    const size_t super_count = center_super_offsets_.empty()
                                   ? 0
                                   : center_super_offsets_.size() - 1;
    if (centers_count == 0) {
        return 0;
    }
    if (super_count == 0 || center_super_centers_.empty() ||
        center_super_fine_centers_.empty()) {
        if (config_.center_coarse_cascade_scan) {
            return refine_center_coarse_cascade_scan(
                query, routed_center, use_routed_center, stats, exact_centers,
                center_margin
            );
        }
        return refine_center_exact_scan(query, stats, exact_centers, center_margin);
    }

    size_t super_probe =
        std::max<size_t>(1, std::min(config_.center_super_probe, super_count));
    static thread_local std::vector<ScoredPid> super_scores;
    static thread_local std::vector<PID> fine_candidates;
    static thread_local std::vector<ScoredPid> exact_scores;
    static thread_local std::vector<uint32_t> fine_seen;
    static thread_local uint32_t fine_seen_epoch = 1;

    super_scores.clear();
    if (super_scores.capacity() < super_probe) {
        super_scores.reserve(super_probe);
    }
    for (size_t s = 0; s < super_count; ++s) {
        push_topk_smallest(
            super_scores,
            {euclidean_sqr_fast(
                 query,
                 center_super_centers_.data() + (s * dim_),
                 dim_
             ),
             static_cast<PID>(s)},
            super_probe
        );
    }
    finish_topk_smallest(super_scores);

    size_t avg_fine_hint =
        center_super_fine_centers_.size() / std::max<size_t>(1, super_count);
    fine_candidates.clear();
    size_t fine_reserve = super_probe * avg_fine_hint + 4;
    if (fine_candidates.capacity() < fine_reserve) {
        fine_candidates.reserve(fine_reserve);
    }
    if (fine_seen.size() < centers_count) {
        fine_seen.assign(centers_count, 0);
        fine_seen_epoch = 1;
    } else {
        ++fine_seen_epoch;
        if (fine_seen_epoch == 0) {
            std::fill(fine_seen.begin(), fine_seen.end(), 0);
            fine_seen_epoch = 1;
        }
    }
    auto append_fine_unique = [&](PID center_id) {
        if (center_id >= centers_count) {
            return;
        }
        uint32_t& seen = fine_seen[center_id];
        if (seen != fine_seen_epoch) {
            seen = fine_seen_epoch;
            fine_candidates.push_back(center_id);
        }
    };

    for (const auto& super : super_scores) {
        size_t s = super.id;
        size_t begin = center_super_offsets_[s];
        size_t end = center_super_offsets_[s + 1];
        for (size_t i = begin; i < end; ++i) {
            append_fine_unique(center_super_fine_centers_[i]);
        }
    }
    if (use_routed_center) {
        append_fine_unique(routed_center);
    }
    if (fine_candidates.empty()) {
        return routed_center;
    }

    const size_t exact_keep = std::min(
        centers_count,
        std::max<size_t>(2, config_.exact_center_keep)
    );
    exact_scores.clear();
    if (exact_scores.capacity() < exact_keep) {
        exact_scores.reserve(exact_keep);
    }
    for (PID center_id : fine_candidates) {
        push_topk_smallest(
            exact_scores,
            {l2_to_center(query, center_id), center_id},
            exact_keep
        );
    }
    finish_topk_smallest(exact_scores);

    if (stats != nullptr) {
        stats->center_estimates = super_count;
        stats->center_exact_scores = fine_candidates.size();
        stats->exact_distance_evals += super_count + fine_candidates.size();
    }
    if (center_margin != nullptr) {
        if (exact_scores.size() >= 2) {
            float best = std::max(exact_scores[0].distance, 1e-6F);
            *center_margin = (exact_scores[1].distance - exact_scores[0].distance) / best;
        } else {
            *center_margin = std::numeric_limits<float>::infinity();
        }
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        size_t out_keep = std::min(config_.exact_center_keep, exact_scores.size());
        exact_centers->reserve(out_keep);
        for (size_t i = 0; i < out_keep; ++i) {
            exact_centers->push_back(exact_scores[i].id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline PID ATMMGGraphIndex::refine_center_coarse_cascade_scan(
    const float* query,
    PID routed_center,
    bool use_routed_center,
    ATMMGGraphQueryStats* stats,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    const size_t centers_count = num_centers();
    const size_t full_dims = center_coarse_dims_.size();
    if (centers_count == 0) {
        return 0;
    }
    if (full_dims == 0 || center_coarse_values_.empty()) {
        return refine_center_exact_scan(query, stats, exact_centers, center_margin);
    }

    constexpr size_t kStackCoarseDimLimit = 256;
    std::array<float, kStackCoarseDimLimit> query_coarse_values{};
    if (full_dims > query_coarse_values.size()) {
        return refine_center_exact_scan(query, stats, exact_centers, center_margin);
    }
    for (size_t i = 0; i < full_dims; ++i) {
        query_coarse_values[i] = query[center_coarse_dims_[i]];
    }

    auto coarse_distance = [&](PID center_id, size_t dims_count) {
        dims_count = std::min(dims_count, full_dims);
        const float* center_vec =
            center_coarse_values_.data() +
            (static_cast<size_t>(center_id) * full_dims);
        return euclidean_sqr_projected_fast(
            query_coarse_values.data(), center_vec, dims_count
        );
    };

    size_t low_dims = std::min(config_.center_cascade_low_dims, full_dims);
    if (low_dims == 0) {
        low_dims = std::min(config_.center_neighbor_prefilter_dims, full_dims);
    }
    if (low_dims == 0) {
        low_dims = std::min(config_.center_coarse_prefilter_dims, full_dims);
    }
    if (low_dims == 0) {
        low_dims = std::min<size_t>(8, full_dims);
    }

    size_t low_keep = config_.center_cascade_low_keep;
    if (low_keep == 0) {
        low_keep = config_.center_neighbor_prefilter_keep;
    }
    if (low_keep == 0) {
        low_keep = config_.center_coarse_prefilter_keep;
    }
    if (low_keep == 0) {
        low_keep = std::min<size_t>(256, centers_count);
    }
    low_keep = std::min(
        centers_count,
        std::max({low_keep, config_.center_scan_keep, config_.exact_center_keep})
    );

    size_t full_keep = config_.center_cascade_mid_keep;
    if (full_keep == 0) {
        full_keep = config_.center_neighbor_global_guard_keep;
    }
    if (full_keep == 0) {
        full_keep = config_.center_scan_keep;
    }
    full_keep = std::min(
        low_keep,
        std::max({full_keep, config_.center_scan_keep, config_.exact_center_keep})
    );

    static thread_local std::vector<ScoredPid> low_scores;
    static thread_local std::vector<ScoredPid> full_scores;
    static thread_local std::vector<ScoredPid> exact_scores;
    auto reset_score_buffer = [](std::vector<ScoredPid>& buffer, size_t reserve_size) {
        buffer.clear();
        if (buffer.capacity() < reserve_size) {
            buffer.reserve(reserve_size);
        }
    };

    if (config_.center_cascade_use_nth) {
        reset_score_buffer(low_scores, centers_count);
        for (size_t c = 0; c < centers_count; ++c) {
            PID center_id = static_cast<PID>(c);
            low_scores.push_back({coarse_distance(center_id, low_dims), center_id});
        }
        keep_smallest(low_scores, low_keep);
    } else {
        reset_score_buffer(low_scores, low_keep);
        for (size_t c = 0; c < centers_count; ++c) {
            PID center_id = static_cast<PID>(c);
            push_topk_smallest(
                low_scores,
                {coarse_distance(center_id, low_dims), center_id},
                low_keep
            );
        }
        finish_topk_smallest(low_scores);
    }

    float low_margin = std::numeric_limits<float>::quiet_NaN();
    if (low_scores.size() >= 2) {
        float best = std::max(low_scores[0].distance, 1e-6F);
        low_margin = (low_scores[1].distance - low_scores[0].distance) / best;
    }
    size_t adaptive_low_keep = adaptive_center_budget(
        low_keep,
        config_.center_adaptive_coarse_keep_easy,
        config_.center_adaptive_coarse_keep_mid,
        low_margin
    );
    adaptive_low_keep = std::min(
        centers_count,
        std::max({adaptive_low_keep, config_.center_scan_keep, config_.exact_center_keep})
    );
    if (adaptive_low_keep < low_scores.size()) {
        low_scores.resize(adaptive_low_keep);
        low_keep = adaptive_low_keep;
    }
    size_t adaptive_full_keep = adaptive_center_budget(
        full_keep,
        config_.center_adaptive_scan_keep_easy,
        config_.center_adaptive_scan_keep_mid,
        low_margin
    );
    full_keep = std::min(
        low_keep,
        std::max({adaptive_full_keep, config_.center_scan_keep, config_.exact_center_keep})
    );

    if (config_.center_cascade_use_nth) {
        reset_score_buffer(full_scores, low_scores.size());
        for (const auto& cand : low_scores) {
            full_scores.push_back({coarse_distance(cand.id, full_dims), cand.id});
        }
        keep_smallest(full_scores, full_keep);
    } else {
        reset_score_buffer(full_scores, full_keep);
        for (const auto& cand : low_scores) {
            push_topk_smallest(
                full_scores,
                {coarse_distance(cand.id, full_dims), cand.id},
                full_keep
            );
        }
        finish_topk_smallest(full_scores);
    }

    const size_t exact_keep = std::min(
        centers_count,
        std::max<size_t>(2, config_.exact_center_keep)
    );
    reset_score_buffer(exact_scores, exact_keep);
    auto has_exact = [&](PID center_id) {
        return std::any_of(
            exact_scores.begin(),
            exact_scores.end(),
            [center_id](const ScoredPid& item) { return item.id == center_id; }
        );
    };
    auto add_exact = [&](PID center_id) {
        if (!has_exact(center_id)) {
            push_topk_smallest(
                exact_scores,
                {l2_to_center(query, center_id), center_id},
                exact_keep
            );
        }
    };
    for (const auto& cand : full_scores) {
        add_exact(cand.id);
    }
    if (use_routed_center && routed_center < centers_count) {
        add_exact(routed_center);
    }
    finish_topk_smallest(exact_scores);

    if (stats != nullptr) {
        stats->center_estimates = full_scores.size();
        stats->center_exact_scores = full_scores.size() +
                                     ((use_routed_center && routed_center < centers_count) ? 1 : 0);
        stats->exact_distance_evals += stats->center_exact_scores;
    }
    if (center_margin != nullptr) {
        if (exact_scores.size() >= 2) {
            float best = std::max(exact_scores[0].distance, 1e-6F);
            *center_margin = (exact_scores[1].distance - exact_scores[0].distance) / best;
        } else {
            *center_margin = std::numeric_limits<float>::infinity();
        }
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        size_t out_keep = std::min(config_.exact_center_keep, exact_scores.size());
        exact_centers->reserve(out_keep);
        for (size_t i = 0; i < out_keep; ++i) {
            exact_centers->push_back(exact_scores[i].id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline PID ATMMGGraphIndex::refine_center(
    const float* query,
    const SplitSingleQuery<float>& query_wrapper,
    PID routed_center,
    bool use_routed_center,
    ATMMGGraphQueryStats& stats,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    if (config_.center_flat_exact_scan) {
        return refine_center_exact_scan(query, &stats, exact_centers, center_margin);
    }
    if (config_.center_super_level_scan) {
        return refine_center_super_level_scan(
            query,
            routed_center,
            use_routed_center,
            &stats,
            exact_centers,
            center_margin
        );
    }
    if (config_.center_coarse_cascade_scan) {
        return refine_center_coarse_cascade_scan(
            query,
            routed_center,
            use_routed_center,
            &stats,
            exact_centers,
            center_margin
        );
    }

    std::vector<ScoredPid> center_scores;
    size_t centers_count = num_centers();
    bool use_neighbor_scan =
        use_routed_center && config_.center_refine_neighbor_scan > 0 &&
        routed_center < center_neighbors_.size();
    size_t scan_limit = use_neighbor_scan
                            ? std::min(
                                  config_.center_refine_neighbor_scan,
                                  center_neighbors_[routed_center].size()
                              )
                            : centers_count;
    size_t center_scan_keep = config_.center_scan_keep;
    center_scores.reserve(scan_limit + 1 + config_.center_neighbor_global_guard_keep);
    constexpr size_t kStackCoarseDimLimit = 256;
    std::array<float, kStackCoarseDimLimit> query_coarse_values{};
    const float* query_coarse = nullptr;
    size_t full_coarse_dims = center_coarse_dims_.size();
    if (!center_coarse_values_.empty() && full_coarse_dims <= query_coarse_values.size()) {
        for (size_t i = 0; i < full_coarse_dims; ++i) {
            query_coarse_values[i] = query[center_coarse_dims_[i]];
        }
        query_coarse = query_coarse_values.data();
    }
    auto coarse_center_distance = [&](PID center_id, size_t dims_count) {
        if (query_coarse == nullptr) {
            return coarse_l2_to_center(query, center_id, dims_count);
        }
        dims_count = std::min(dims_count, full_coarse_dims);
        const float* center_vec = center_coarse_values_.data() +
                                  (static_cast<size_t>(center_id) * full_coarse_dims);
        float sum = 0.0F;
        for (size_t i = 0; i < dims_count; ++i) {
            float diff = query_coarse[i] - center_vec[i];
            sum += diff * diff;
        }
        return sum;
    };

    auto add_center_estimate = [&](PID c) {
        float est = estimate_code(center_codes_.bin(c), center_codes_.ex(c), query_wrapper, 0, 0);
        center_scores.push_back({est, static_cast<PID>(c)});
    };
    auto has_center_estimate = [&](PID c) {
        return std::any_of(center_scores.begin(), center_scores.end(),
                           [c](const ScoredPid& item) { return item.id == c; });
    };
    auto add_center_estimate_if_new = [&](PID c) {
        if (!has_center_estimate(c)) {
            add_center_estimate(c);
        }
    };
    auto add_global_guard_centers = [&]() {
        size_t guard_keep = std::min(config_.center_neighbor_global_guard_keep, centers_count);
        if (guard_keep == 0 || center_coarse_dims_.empty()) {
            return;
        }
        size_t guard_dims =
            std::min(config_.center_neighbor_prefilter_dims, center_coarse_dims_.size());
        if (guard_dims == 0) {
            guard_dims = center_coarse_dims_.size();
        }
        std::vector<ScoredPid> guard_scores;
        guard_scores.reserve(guard_keep);
        for (size_t c = 0; c < centers_count; ++c) {
            PID center_id = static_cast<PID>(c);
            push_topk_smallest(
                guard_scores,
                {coarse_center_distance(center_id, guard_dims), center_id},
                guard_keep
            );
        }
        finish_topk_smallest(guard_scores);
        for (const auto& cand : guard_scores) {
            add_center_estimate_if_new(cand.id);
        }
    };

    bool use_coarse_scan =
        !use_neighbor_scan && !center_coarse_dims_.empty() &&
        config_.center_coarse_keep > 0 && config_.center_coarse_keep < centers_count;

    if (use_neighbor_scan) {
        add_center_estimate(routed_center);
        size_t prefilter_dims =
            std::min(config_.center_neighbor_prefilter_dims, center_coarse_dims_.size());
        bool use_neighbor_prefilter =
            prefilter_dims > 0 &&
            config_.center_neighbor_prefilter_keep > 0 &&
            config_.center_neighbor_prefilter_keep < scan_limit + 1;
        if (use_neighbor_prefilter) {
            size_t neighbor_keep =
                std::min(scan_limit, config_.center_neighbor_prefilter_keep - 1);
            std::vector<ScoredPid> neighbor_scores;
            neighbor_scores.reserve(scan_limit);
            for (size_t i = 0; i < scan_limit; ++i) {
                PID center_id = center_neighbors_[routed_center][i];
                neighbor_scores.push_back(
                    {coarse_center_distance(center_id, prefilter_dims), center_id}
                );
            }
            keep_smallest(neighbor_scores, neighbor_keep);
            for (const auto& cand : neighbor_scores) {
                add_center_estimate(cand.id);
            }
        } else {
            for (size_t i = 0; i < scan_limit; ++i) {
                add_center_estimate(center_neighbors_[routed_center][i]);
            }
        }
        add_global_guard_centers();
    } else if (use_coarse_scan) {
        size_t coarse_keep_default = std::min(
            centers_count,
            std::max(
                {config_.center_coarse_keep, config_.center_scan_keep,
                 config_.exact_center_keep}
            )
        );
        float coarse_margin = std::numeric_limits<float>::quiet_NaN();
        std::vector<ScoredPid> coarse_scores =
            coarse_center_candidates(query, centers_count, coarse_keep_default, &coarse_margin);
        center_scan_keep = adaptive_center_budget(
            config_.center_scan_keep,
            config_.center_adaptive_scan_keep_easy,
            config_.center_adaptive_scan_keep_mid,
            coarse_margin
        );
        size_t coarse_keep = adaptive_center_budget(
            coarse_keep_default,
            config_.center_adaptive_coarse_keep_easy,
            config_.center_adaptive_coarse_keep_mid,
            coarse_margin
        );
        coarse_keep = std::min(
            centers_count,
            std::max({coarse_keep, center_scan_keep, config_.exact_center_keep})
        );
        if (coarse_keep < coarse_scores.size()) {
            keep_smallest(coarse_scores, coarse_keep);
        }
        for (const auto& cand : coarse_scores) {
            add_center_estimate(cand.id);
        }
    } else {
        for (size_t c = 0; c < centers_count; ++c) {
            add_center_estimate(static_cast<PID>(c));
        }
    }
    stats.center_estimates = center_scores.size();
    stats.quant_center_est_evals += center_scores.size();
    keep_smallest(center_scores, std::min(center_scan_keep, center_scores.size()));

    bool has_routed = false;
    for (const auto& cand : center_scores) {
        has_routed = has_routed || cand.id == routed_center;
    }
    if (use_routed_center && !has_routed) {
        ++stats.exact_distance_evals;
        center_scores.push_back(
            {l2_to_center(query, routed_center), static_cast<PID>(routed_center)}
        );
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(center_scores.size());
    for (const auto& cand : center_scores) {
        bool duplicate = false;
        for (const auto& prev : exact_scores) {
            if (prev.id == cand.id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) {
            continue;
        }
        exact_scores.push_back({l2_to_center(query, cand.id), cand.id});
    }
    stats.center_exact_scores = exact_scores.size();
    stats.exact_distance_evals += exact_scores.size();
    keep_smallest(exact_scores, std::min(config_.exact_center_keep, exact_scores.size()));
    if (center_margin != nullptr) {
        if (exact_scores.size() >= 2) {
            float best = std::max(exact_scores[0].distance, 1e-6F);
            *center_margin = (exact_scores[1].distance - exact_scores[0].distance) / best;
        } else {
            *center_margin = std::numeric_limits<float>::infinity();
        }
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        exact_centers->reserve(exact_scores.size());
        for (const auto& cand : exact_scores) {
            exact_centers->push_back(cand.id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline PID ATMMGGraphIndex::refine_center_fast(
    const float* query,
    const SplitSingleQuery<float>& query_wrapper,
    PID routed_center,
    bool use_routed_center,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    if (config_.center_flat_exact_scan) {
        return refine_center_exact_scan(query, nullptr, exact_centers, center_margin);
    }
    if (config_.center_super_level_scan) {
        return refine_center_super_level_scan(
            query,
            routed_center,
            use_routed_center,
            nullptr,
            exact_centers,
            center_margin
        );
    }
    if (config_.center_coarse_cascade_scan) {
        return refine_center_coarse_cascade_scan(
            query,
            routed_center,
            use_routed_center,
            nullptr,
            exact_centers,
            center_margin
        );
    }

    std::vector<ScoredPid> center_scores;
    size_t centers_count = num_centers();
    bool use_neighbor_scan =
        use_routed_center && config_.center_refine_neighbor_scan > 0 &&
        routed_center < center_neighbors_.size();
    size_t scan_limit = use_neighbor_scan
                            ? std::min(
                                  config_.center_refine_neighbor_scan,
                                  center_neighbors_[routed_center].size()
                              )
                            : centers_count;
    size_t center_scan_keep = config_.center_scan_keep;
    center_scores.reserve(scan_limit + 1 + config_.center_neighbor_global_guard_keep);
    constexpr size_t kStackCoarseDimLimit = 256;
    std::array<float, kStackCoarseDimLimit> query_coarse_values{};
    const float* query_coarse = nullptr;
    size_t full_coarse_dims = center_coarse_dims_.size();
    if (!center_coarse_values_.empty() && full_coarse_dims <= query_coarse_values.size()) {
        for (size_t i = 0; i < full_coarse_dims; ++i) {
            query_coarse_values[i] = query[center_coarse_dims_[i]];
        }
        query_coarse = query_coarse_values.data();
    }
    auto coarse_center_distance = [&](PID center_id, size_t dims_count) {
        if (query_coarse == nullptr) {
            return coarse_l2_to_center(query, center_id, dims_count);
        }
        dims_count = std::min(dims_count, full_coarse_dims);
        const float* center_vec = center_coarse_values_.data() +
                                  (static_cast<size_t>(center_id) * full_coarse_dims);
        float sum = 0.0F;
        for (size_t i = 0; i < dims_count; ++i) {
            float diff = query_coarse[i] - center_vec[i];
            sum += diff * diff;
        }
        return sum;
    };

    auto add_center_estimate = [&](PID c) {
        float est = estimate_code(center_codes_.bin(c), center_codes_.ex(c), query_wrapper, 0, 0);
        center_scores.push_back({est, static_cast<PID>(c)});
    };
    auto has_center_estimate = [&](PID c) {
        return std::any_of(center_scores.begin(), center_scores.end(),
                           [c](const ScoredPid& item) { return item.id == c; });
    };
    auto add_center_estimate_if_new = [&](PID c) {
        if (!has_center_estimate(c)) {
            add_center_estimate(c);
        }
    };
    auto add_global_guard_centers = [&]() {
        size_t guard_keep = std::min(config_.center_neighbor_global_guard_keep, centers_count);
        if (guard_keep == 0 || center_coarse_dims_.empty()) {
            return;
        }
        size_t guard_dims =
            std::min(config_.center_neighbor_prefilter_dims, center_coarse_dims_.size());
        if (guard_dims == 0) {
            guard_dims = center_coarse_dims_.size();
        }
        std::vector<ScoredPid> guard_scores;
        guard_scores.reserve(guard_keep);
        for (size_t c = 0; c < centers_count; ++c) {
            PID center_id = static_cast<PID>(c);
            push_topk_smallest(
                guard_scores,
                {coarse_center_distance(center_id, guard_dims), center_id},
                guard_keep
            );
        }
        finish_topk_smallest(guard_scores);
        for (const auto& cand : guard_scores) {
            add_center_estimate_if_new(cand.id);
        }
    };

    bool use_coarse_scan =
        !use_neighbor_scan && !center_coarse_dims_.empty() &&
        config_.center_coarse_keep > 0 && config_.center_coarse_keep < centers_count;

    if (use_neighbor_scan) {
        add_center_estimate(routed_center);
        size_t prefilter_dims =
            std::min(config_.center_neighbor_prefilter_dims, center_coarse_dims_.size());
        bool use_neighbor_prefilter =
            prefilter_dims > 0 &&
            config_.center_neighbor_prefilter_keep > 0 &&
            config_.center_neighbor_prefilter_keep < scan_limit + 1;
        if (use_neighbor_prefilter) {
            size_t neighbor_keep =
                std::min(scan_limit, config_.center_neighbor_prefilter_keep - 1);
            std::vector<ScoredPid> neighbor_scores;
            neighbor_scores.reserve(scan_limit);
            for (size_t i = 0; i < scan_limit; ++i) {
                PID center_id = center_neighbors_[routed_center][i];
                neighbor_scores.push_back(
                    {coarse_center_distance(center_id, prefilter_dims), center_id}
                );
            }
            keep_smallest(neighbor_scores, neighbor_keep);
            for (const auto& cand : neighbor_scores) {
                add_center_estimate(cand.id);
            }
        } else {
            for (size_t i = 0; i < scan_limit; ++i) {
                add_center_estimate(center_neighbors_[routed_center][i]);
            }
        }
        add_global_guard_centers();
    } else if (use_coarse_scan) {
        size_t coarse_keep_default = std::min(
            centers_count,
            std::max(
                {config_.center_coarse_keep, config_.center_scan_keep,
                 config_.exact_center_keep}
            )
        );
        float coarse_margin = std::numeric_limits<float>::quiet_NaN();
        std::vector<ScoredPid> coarse_scores =
            coarse_center_candidates(query, centers_count, coarse_keep_default, &coarse_margin);
        center_scan_keep = adaptive_center_budget(
            config_.center_scan_keep,
            config_.center_adaptive_scan_keep_easy,
            config_.center_adaptive_scan_keep_mid,
            coarse_margin
        );
        size_t coarse_keep = adaptive_center_budget(
            coarse_keep_default,
            config_.center_adaptive_coarse_keep_easy,
            config_.center_adaptive_coarse_keep_mid,
            coarse_margin
        );
        coarse_keep = std::min(
            centers_count,
            std::max({coarse_keep, center_scan_keep, config_.exact_center_keep})
        );
        if (coarse_keep < coarse_scores.size()) {
            keep_smallest(coarse_scores, coarse_keep);
        }
        for (const auto& cand : coarse_scores) {
            add_center_estimate(cand.id);
        }
    } else {
        for (size_t c = 0; c < centers_count; ++c) {
            add_center_estimate(static_cast<PID>(c));
        }
    }
    keep_smallest(center_scores, std::min(center_scan_keep, center_scores.size()));

    bool has_routed = false;
    for (const auto& cand : center_scores) {
        has_routed = has_routed || cand.id == routed_center;
    }
    if (use_routed_center && !has_routed) {
        center_scores.push_back(
            {l2_to_center(query, routed_center), static_cast<PID>(routed_center)}
        );
    }

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(center_scores.size());
    for (const auto& cand : center_scores) {
        bool duplicate = false;
        for (const auto& prev : exact_scores) {
            if (prev.id == cand.id) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            exact_scores.push_back({l2_to_center(query, cand.id), cand.id});
        }
    }
    keep_smallest(exact_scores, std::min(config_.exact_center_keep, exact_scores.size()));
    if (center_margin != nullptr) {
        if (exact_scores.size() >= 2) {
            float best = std::max(exact_scores[0].distance, 1e-6F);
            *center_margin = (exact_scores[1].distance - exact_scores[0].distance) / best;
        } else {
            *center_margin = std::numeric_limits<float>::infinity();
        }
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        exact_centers->reserve(exact_scores.size());
        for (const auto& cand : exact_scores) {
            exact_centers->push_back(cand.id);
        }
    }
    return exact_scores.empty() ? routed_center : exact_scores.front().id;
}

inline PID ATMMGGraphIndex::refine_center_fast_without_quant(
    const float* query,
    PID routed_center,
    bool use_routed_center,
    std::vector<PID>* exact_centers,
    float* center_margin
) const {
    if (config_.center_flat_exact_scan) {
        return refine_center_exact_scan(query, nullptr, exact_centers, center_margin);
    }
    if (config_.center_super_level_scan) {
        return refine_center_super_level_scan(
            query,
            routed_center,
            use_routed_center,
            nullptr,
            exact_centers,
            center_margin
        );
    }
    if (config_.center_coarse_cascade_scan) {
        return refine_center_coarse_cascade_scan(
            query,
            routed_center,
            use_routed_center,
            nullptr,
            exact_centers,
            center_margin
        );
    }
    if (exact_centers != nullptr) {
        exact_centers->clear();
        exact_centers->push_back(routed_center);
    }
    if (center_margin != nullptr) {
        *center_margin = std::numeric_limits<float>::infinity();
    }
    return routed_center;
}

inline void ATMMGGraphIndex::append_hash_spectrum_seeds(
    const float* query,
    float query_sq_norm,
    const float* rotated_query,
    PID best_center,
    const std::vector<PID>& candidate_centers,
    std::vector<PID>& seeds,
    ATMMGGraphQueryStats* stats
) const {
    if (!config_.use_hash_neighborhood_spectrum ||
        config_.hash_spectrum_size == 0 || hash_spectrum_ids_.empty() ||
        best_center >= hash_spectrum_ids_.size()) {
        return;
    }

    std::vector<PID> seed_centers;
    seed_centers.reserve(config_.hash_spectrum_center_keep);
    auto append_center_unique = [&](PID center_id) {
        if (center_id >= hash_spectrum_ids_.size()) {
            return;
        }
        if (std::find(seed_centers.begin(), seed_centers.end(), center_id) ==
            seed_centers.end()) {
            seed_centers.push_back(center_id);
        }
    };

    append_center_unique(best_center);
    for (PID center_id : candidate_centers) {
        append_center_unique(center_id);
        if (seed_centers.size() >= config_.hash_spectrum_center_keep) {
            break;
        }
    }

    const size_t hamming_keep =
        std::max<size_t>(1, config_.hash_spectrum_candidates * seed_centers.size());
    std::vector<ScoredPid> hamming_scores;
    hamming_scores.reserve(hamming_keep);

    for (PID center_id : seed_centers) {
        const auto& ids = hash_spectrum_ids_[center_id];
        const auto& codes = hash_spectrum_codes_[center_id];
        if (ids.empty()) {
            continue;
        }
        if (stats != nullptr) {
            stats->entry_points_raw += ids.size();
        }
        uint64_t query_code = hash_spectrum_code(center_id, rotated_query);
        for (size_t i = 0; i < ids.size(); ++i) {
            uint32_t full = popcount64(query_code ^ codes[i]);
            uint32_t segment = hash_spectrum_segment_min(query_code, codes[i]);
            float score = static_cast<float>(full);
            if (segment > config_.hash_spectrum_segment_radius) {
                score += static_cast<float>(config_.hash_spectrum_segment_bits);
            }
            push_topk_smallest(hamming_scores, {score, ids[i]}, hamming_keep);
        }
    }

    if (hamming_scores.empty()) {
        return;
    }
    finish_topk_smallest(hamming_scores);

    std::vector<ScoredPid> exact_scores;
    exact_scores.reserve(std::min(config_.hash_spectrum_entry_take, hamming_scores.size()));
    auto already_seeded = [&](PID id) {
        if (std::find(seeds.begin(), seeds.end(), id) != seeds.end()) {
            return true;
        }
        for (const auto& cand : exact_scores) {
            if (cand.id == id) {
                return true;
            }
        }
        return false;
    };

    for (const auto& cand : hamming_scores) {
        if (already_seeded(cand.id)) {
            continue;
        }
        exact_scores.push_back({l2_to_point(query, query_sq_norm, cand.id), cand.id});
    }
    if (stats != nullptr) {
        stats->exact_distance_evals += exact_scores.size();
        stats->entry_points_before_cap += exact_scores.size();
    }
    keep_smallest(exact_scores, std::min(config_.hash_spectrum_entry_take, exact_scores.size()));
    for (const auto& cand : exact_scores) {
        seeds.push_back(cand.id);
    }
}

inline float ATMMGGraphIndex::estimate_code(
    const char* bin_data,
    const char* ex_data,
    const SplitSingleQuery<float>& query_wrapper,
    float g_add,
    float g_error
) const {
    float ip_x0_qr = 0;
    float est_dist = 0;
    float low_dist = 0;
    if (ex_bits_ == 0) {
        split_single_estdist(
            bin_data, query_wrapper, padded_dim_, ip_x0_qr, est_dist, low_dist, g_add, g_error
        );
    } else {
        split_single_fulldist(
            bin_data,
            ex_data,
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            est_dist,
            low_dist,
            ip_x0_qr,
            g_add,
            g_error
        );
    }
    return est_dist;
}

inline float ATMMGGraphIndex::estimate_code_onebit(
    const char* bin_data,
    const SplitSingleQuery<float>& query_wrapper,
    float g_add,
    float g_error
) const {
    float ip_x0_qr = 0;
    float est_dist = 0;
    float low_dist = 0;
    split_single_estdist(
        bin_data, query_wrapper, padded_dim_, ip_x0_qr, est_dist, low_dist, g_add, g_error
    );
    return est_dist;
}

inline bool ATMMGGraphIndex::graph_scan_ready() const {
    return config_.graph_search_use_quant && !config_.graph_search_full_quant &&
           !graph_edge_batch_offsets_.empty() &&
           graph_edge_batch_offsets_.size() == num_ + 1 &&
           graph_edge_batch_codes_.batch_count > 0 &&
           graph_edge_batch_codes_.batch_bytes >=
               QGBatchDataMap<float>::data_bytes(padded_dim_);
}

inline uint32_t ATMMGGraphIndex::next_visit_epoch() const {
    ++visit_epoch_;
    if (visit_epoch_ == 0) {
        std::fill(visit_marks_.begin(), visit_marks_.end(), 0);
        visit_epoch_ = 1;
    }
    return visit_epoch_;
}

inline void ATMMGGraphIndex::keep_smallest(
    std::vector<ScoredPid>& items, size_t keep
) {
    if (keep >= items.size()) {
        std::sort(items.begin(), items.end());
        return;
    }
    if (keep == 0) {
        items.clear();
        return;
    }
    auto nth = items.begin() + static_cast<std::ptrdiff_t>(keep);
    std::nth_element(items.begin(), nth, items.end());
    items.resize(keep);
    std::sort(items.begin(), items.end());
}

inline void ATMMGGraphIndex::push_topk_smallest(
    std::vector<ScoredPid>& heap, ScoredPid item, size_t keep
) {
    if (keep == 0) {
        return;
    }
    if (heap.size() < keep) {
        heap.push_back(item);
        std::push_heap(heap.begin(), heap.end());
        return;
    }
    if (item.distance < heap.front().distance) {
        std::pop_heap(heap.begin(), heap.end());
        heap.back() = item;
        std::push_heap(heap.begin(), heap.end());
    }
}

inline void ATMMGGraphIndex::finish_topk_smallest(std::vector<ScoredPid>& heap) {
    std::sort_heap(heap.begin(), heap.end());
}
