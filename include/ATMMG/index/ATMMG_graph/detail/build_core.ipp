inline void ATMMGGraphIndex::construct(const float* base, size_t num, size_t dim) {
    if (base == nullptr || num == 0 || dim == 0) {
        throw std::invalid_argument("ATMMGGraphIndex::construct got empty data");
    }

    num_ = num;
    dim_ = dim;
    ex_bits_ = config_.quant_total_bits - 1;
    base_.assign(base, base + (num_ * dim_));
    external_ids_.resize(num_);
    std::iota(external_ids_.begin(), external_ids_.end(), static_cast<PID>(0));
    base_sq_norms_.clear();
    query_adaptive_l1_ready_ = false;
    query_adaptive_l1_low_ = 0.0F;
    query_adaptive_l1_high_ = 0.0F;
    if (config_.graph_distance_use_norm_dot) {
        base_sq_norms_.resize(num_);
        for (size_t i = 0; i < num_; ++i) {
            base_sq_norms_[i] = squared_norm(base_.data() + (i * dim_), dim_);
        }
    }

    tree_.clear();
    annoy_route_nodes_.clear();
    annoy_route_roots_.clear();
    cluster_route_nodes_.clear();
    cluster_route_root_ = 0;
    leaf_ids_.clear();
    centers_.clear();
    center_sq_norms_.clear();
    point_center_.assign(num_, 0);
    graph_offsets_.clear();
    graph_indices_.clear();
    graph_edge_batch_codes_ = BatchCodeBank{};
    graph_edge_batch_offsets_.clear();
    graph_hot_offsets_.clear();
    graph_hot_indices_.clear();
    graph_cold_offsets_.clear();
    graph_cold_indices_.clear();
    graph_bridge_edges_added_ = 0;
    graph_post_nnd_edges_before_ = 0;
    graph_post_nnd_edges_after_ = 0;
    graph_post_nnd_candidate_total_ = 0;
    graph_post_nnd_candidate_sources_ = 0;
    graph_reordered_by_center_ = false;
    center_coarse_dims_.clear();
    center_coarse_values_.clear();
    center_super_centers_.clear();
    center_super_offsets_.clear();
    center_super_fine_centers_.clear();
    graph_prefilter_dims_.clear();
    point_center_residual_norms_.clear();
    center_portal_pool_.clear();
    point_is_portal_.clear();
    hash_spectrum_ids_.clear();
    hash_spectrum_codes_.clear();
    hash_spectrum_dims_.clear();
    residual_hash_dims_.clear();
    residual_hash_signs_.clear();
    residual_hash_codes_.clear();
    residual_hash_words_ = 0;
    residual_bucket_indices_.clear();
    residual_bucket_counts_.clear();
    residual_bucket_center_thresholds_.clear();
    residual_bucket_code_bits_ = 0;
    residual_query_bucket_indices_.clear();
    residual_query_bucket_counts_.clear();
    residual_query_bucket_width_ = 0;
    residual_query_bucket_stride_ = 0;
    residual_bucket_count_ = 0;
    residual_bucket_take_ = 0;
    residual_bucket_stride_ = 0;

    size_t feasible_leaves = std::max<size_t>(1, num_ / config_.center_leaf_min_size);
    size_t target_leaves = std::min(config_.n_centers, feasible_leaves);
    std::vector<PID> all_ids(num_);
    std::iota(all_ids.begin(), all_ids.end(), static_cast<PID>(0));
    root_ = build_tree(std::move(all_ids), target_leaves);
    build_center_sq_norms();
    build_annoy_route_forest();
    build_cluster_route_tree();

    rotate_base_and_centers();
    index_config_ = quant::make_quant_config(padded_dim_, config_.quant_total_bits);
    query_config_ =
        quant::make_quant_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
    ip_func_ = select_excode_ipfunc(ex_bits_);

    build_center_pools();
    build_center_neighbors();
    if (config_.graph_admission_bound) {
        build_point_center_residual_norms();
    }
    build_center_coarse_projection();
    build_center_super_layer();
    build_graph_prefilter_projection();
    build_query_adaptive_l1_thresholds();
    build_center_portals();
    build_hash_neighborhood_spectrum();
    if (uses_center_quant_refine()) {
        quantize_center_codes();
    } else {
        center_codes_ = CodeBank{};
    }
    if (!config_.graph_reorder_by_center) {
        quantize_topn_codes();
    } else {
        topn_codes_.clear();
        topn_batch_codes_.clear();
    }
    if (config_.graph_search_use_quant && !config_.graph_reorder_by_center) {
        quantize_point_codes();
    }
    build_graph();
    apply_graph_post_nnd_refine();
    build_residual_hash_spectrum();
    build_residual_hash_bucket_graph();
    visit_marks_.assign(num_, 0);
    visit_epoch_ = 0;
}

inline size_t ATMMGGraphIndex::build_tree(
    std::vector<PID> ids, size_t target_leaves
) {
    size_t node_id = tree_.size();
    tree_.push_back(TreeNode());

    size_t feasible_leaves =
        std::max<size_t>(1, ids.size() / config_.center_leaf_min_size);
    target_leaves = std::min(target_leaves, feasible_leaves);

    if (target_leaves <= 1 || ids.size() <= config_.center_leaf_min_size) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t left_target = std::max<size_t>(1, target_leaves / 2);
    size_t right_target = target_leaves - left_target;
    if (right_target == 0) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t split_count = (ids.size() * left_target) / target_leaves;
    size_t left_min = left_target * config_.center_leaf_min_size;
    size_t right_min = right_target * config_.center_leaf_min_size;
    split_count = std::max(split_count, left_min);
    split_count = std::min(split_count, ids.size() - right_min);

    if (split_count == 0 || split_count >= ids.size()) {
        tree_[node_id].leaf = true;
        tree_[node_id].center = make_leaf(ids);
        return node_id;
    }

    size_t split_dim = choose_split_dim(ids);
    auto mid = ids.begin() + static_cast<std::ptrdiff_t>(split_count);
    std::nth_element(
        ids.begin(),
        mid,
        ids.end(),
        [&](PID a, PID b) {
            return base_[static_cast<size_t>(a) * dim_ + split_dim] <
                   base_[static_cast<size_t>(b) * dim_ + split_dim];
        }
    );

    float left_max = -std::numeric_limits<float>::infinity();
    float right_min_value = std::numeric_limits<float>::infinity();
    for (auto it = ids.begin(); it != mid; ++it) {
        left_max = std::max(left_max, base_[static_cast<size_t>(*it) * dim_ + split_dim]);
    }
    for (auto it = mid; it != ids.end(); ++it) {
        right_min_value =
            std::min(right_min_value, base_[static_cast<size_t>(*it) * dim_ + split_dim]);
    }

    std::vector<PID> left(ids.begin(), mid);
    std::vector<PID> right(mid, ids.end());

    tree_[node_id].split_dim = split_dim;
    tree_[node_id].split_value = (left_max + right_min_value) * 0.5F;
    tree_[node_id].left = build_tree(std::move(left), left_target);
    tree_[node_id].right = build_tree(std::move(right), right_target);
    return node_id;
}

inline PID ATMMGGraphIndex::make_leaf(const std::vector<PID>& ids) {
    PID center_id = static_cast<PID>(leaf_ids_.size());
    leaf_ids_.push_back(ids);

    centers_.resize((static_cast<size_t>(center_id) + 1) * dim_, 0);
    float* center = centers_.data() + (static_cast<size_t>(center_id) * dim_);
    for (PID id : ids) {
        point_center_[id] = center_id;
        const float* point = base_.data() + (static_cast<size_t>(id) * dim_);
        for (size_t d = 0; d < dim_; ++d) {
            center[d] += point[d];
        }
    }
    float inv = 1.0F / static_cast<float>(ids.size());
    for (size_t d = 0; d < dim_; ++d) {
        center[d] *= inv;
    }
    return center_id;
}

inline size_t ATMMGGraphIndex::choose_split_dim(const std::vector<PID>& ids) const {
    size_t best_dim = 0;
    double best_var = -1.0;
    for (size_t d = 0; d < dim_; ++d) {
        double sum = 0;
        double sum_sq = 0;
        for (PID id : ids) {
            double v = base_[static_cast<size_t>(id) * dim_ + d];
            sum += v;
            sum_sq += v * v;
        }
        double mean = sum / static_cast<double>(ids.size());
        double var = (sum_sq / static_cast<double>(ids.size())) - (mean * mean);
        if (var > best_var) {
            best_var = var;
            best_dim = d;
        }
    }
    return best_dim;
}

inline PID ATMMGGraphIndex::route_to_center(const float* query) const {
    size_t node_id = root_;
    while (!tree_[node_id].leaf) {
        const TreeNode& node = tree_[node_id];
        node_id = (query[node.split_dim] <= node.split_value) ? node.left : node.right;
    }
    return tree_[node_id].center;
}

inline void ATMMGGraphIndex::build_center_sq_norms() {
    size_t centers_count = num_centers();
    center_sq_norms_.assign(centers_count, 0.0F);
    for (size_t c = 0; c < centers_count; ++c) {
        center_sq_norms_[c] = squared_norm(centers_.data() + (c * dim_), dim_);
    }
}

inline void ATMMGGraphIndex::build_annoy_route_forest() {
    annoy_route_nodes_.clear();
    annoy_route_roots_.clear();
    size_t centers_count = num_centers();
    if (centers_count == 0) {
        return;
    }

    std::vector<PID> center_ids(centers_count);
    std::iota(center_ids.begin(), center_ids.end(), static_cast<PID>(0));
    std::mt19937 rng(static_cast<uint32_t>(config_.random_seed ^ 0xA11D00DU));
    annoy_route_roots_.reserve(config_.annoy_route_trees);
    for (size_t t = 0; t < config_.annoy_route_trees; ++t) {
        std::vector<PID> ids = center_ids;
        std::shuffle(ids.begin(), ids.end(), rng);
        annoy_route_roots_.push_back(build_annoy_route_tree(std::move(ids), rng));
    }
}

inline size_t ATMMGGraphIndex::build_annoy_route_tree(
    std::vector<PID> center_ids, std::mt19937& rng
) {
    size_t node_id = annoy_route_nodes_.size();
    annoy_route_nodes_.push_back(AnnoyRouteNode());

    if (center_ids.size() <= config_.annoy_route_leaf_size) {
        annoy_route_nodes_[node_id].leaf = true;
        annoy_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    std::uniform_int_distribution<size_t> pick(0, center_ids.size() - 1);
    PID pivot_a = center_ids[pick(rng)];
    PID pivot_b = center_ids[pick(rng)];
    for (size_t tries = 0; tries < 16 && pivot_a == pivot_b; ++tries) {
        pivot_b = center_ids[pick(rng)];
    }
    if (pivot_a == pivot_b) {
        annoy_route_nodes_[node_id].leaf = true;
        annoy_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    const float* a = centers_.data() + (static_cast<size_t>(pivot_a) * dim_);
    const float* b = centers_.data() + (static_cast<size_t>(pivot_b) * dim_);
    float threshold =
        0.5F * (center_sq_norms_[pivot_a] - center_sq_norms_[pivot_b]);

    std::vector<PID> left;
    std::vector<PID> right;
    left.reserve(center_ids.size() / 2 + 1);
    right.reserve(center_ids.size() / 2 + 1);
    for (PID center_id : center_ids) {
        const float* c = centers_.data() + (static_cast<size_t>(center_id) * dim_);
        float projection = 0.0F;
        for (size_t d = 0; d < dim_; ++d) {
            projection += c[d] * (a[d] - b[d]);
        }
        if (projection <= threshold) {
            left.push_back(center_id);
        } else {
            right.push_back(center_id);
        }
    }

    if (left.empty() || right.empty()) {
        std::shuffle(center_ids.begin(), center_ids.end(), rng);
        size_t mid = center_ids.size() / 2;
        left.assign(center_ids.begin(), center_ids.begin() + static_cast<std::ptrdiff_t>(mid));
        right.assign(center_ids.begin() + static_cast<std::ptrdiff_t>(mid), center_ids.end());
    }
    if (left.empty() || right.empty()) {
        annoy_route_nodes_[node_id].leaf = true;
        annoy_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    size_t left_id = build_annoy_route_tree(std::move(left), rng);
    size_t right_id = build_annoy_route_tree(std::move(right), rng);
    AnnoyRouteNode& node = annoy_route_nodes_[node_id];
    node.pivot_a = pivot_a;
    node.pivot_b = pivot_b;
    node.threshold = threshold;
    node.left = left_id;
    node.right = right_id;
    return node_id;
}

inline PID ATMMGGraphIndex::route_annoy_to_center(const float* query) const {
    if (annoy_route_roots_.empty()) {
        return route_to_center(query);
    }

    size_t centers_count = num_centers();
    std::vector<uint8_t> seen(centers_count, 0);
    PID best_center = 0;
    float best_d2 = std::numeric_limits<float>::infinity();
    bool found = false;

    for (size_t root : annoy_route_roots_) {
        size_t node_id = root;
        while (!annoy_route_nodes_[node_id].leaf) {
            const AnnoyRouteNode& node = annoy_route_nodes_[node_id];
            const float* a =
                centers_.data() + (static_cast<size_t>(node.pivot_a) * dim_);
            const float* b =
                centers_.data() + (static_cast<size_t>(node.pivot_b) * dim_);
            float projection = 0.0F;
            for (size_t d = 0; d < dim_; ++d) {
                projection += query[d] * (a[d] - b[d]);
            }
            node_id = (projection <= node.threshold) ? node.left : node.right;
        }

        for (PID center_id : annoy_route_nodes_[node_id].centers) {
            if (center_id >= centers_count || seen[center_id]) {
                continue;
            }
            seen[center_id] = 1;
            float d2 = l2_to_center(query, center_id);
            if (d2 < best_d2) {
                best_d2 = d2;
                best_center = center_id;
                found = true;
            }
        }
    }

    return found ? best_center : route_to_center(query);
}

inline void ATMMGGraphIndex::build_cluster_route_tree() {
    cluster_route_nodes_.clear();
    cluster_route_root_ = 0;
    size_t centers_count = num_centers();
    if (centers_count == 0) {
        return;
    }
    std::vector<PID> center_ids(centers_count);
    std::iota(center_ids.begin(), center_ids.end(), static_cast<PID>(0));
    cluster_route_root_ = build_cluster_route_tree(std::move(center_ids));
}

inline size_t ATMMGGraphIndex::build_cluster_route_tree(std::vector<PID> center_ids) {
    size_t node_id = cluster_route_nodes_.size();
    cluster_route_nodes_.push_back(ClusterRouteNode());

    if (center_ids.size() <= config_.cluster_route_leaf_size) {
        cluster_route_nodes_[node_id].leaf = true;
        cluster_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    auto center_ptr = [&](PID center_id) {
        return centers_.data() + (static_cast<size_t>(center_id) * dim_);
    };
    auto center_center_l2 = [&](PID a, PID b) {
        const float* av = center_ptr(a);
        const float* bv = center_ptr(b);
        float d2 = 0.0F;
        for (size_t d = 0; d < dim_; ++d) {
            float diff = av[d] - bv[d];
            d2 += diff * diff;
        }
        return d2;
    };
    auto point_centroid_l2 = [&](PID center_id, const std::vector<float>& centroid) {
        const float* c = center_ptr(center_id);
        float d2 = 0.0F;
        for (size_t d = 0; d < dim_; ++d) {
            float diff = c[d] - centroid[d];
            d2 += diff * diff;
        }
        return d2;
    };
    auto recompute_centroid = [&](const std::vector<PID>& ids, std::vector<float>& centroid) {
        std::fill(centroid.begin(), centroid.end(), 0.0F);
        if (ids.empty()) {
            return;
        }
        for (PID center_id : ids) {
            const float* c = center_ptr(center_id);
            for (size_t d = 0; d < dim_; ++d) {
                centroid[d] += c[d];
            }
        }
        float inv = 1.0F / static_cast<float>(ids.size());
        for (size_t d = 0; d < dim_; ++d) {
            centroid[d] *= inv;
        }
    };

    PID pivot_a = center_ids.front();
    PID pivot_b = pivot_a;
    float farthest = -1.0F;
    for (PID center_id : center_ids) {
        float d2 = center_center_l2(pivot_a, center_id);
        if (d2 > farthest) {
            farthest = d2;
            pivot_b = center_id;
        }
    }
    pivot_a = pivot_b;
    farthest = -1.0F;
    for (PID center_id : center_ids) {
        float d2 = center_center_l2(pivot_a, center_id);
        if (d2 > farthest) {
            farthest = d2;
            pivot_b = center_id;
        }
    }
    if (pivot_a == pivot_b) {
        cluster_route_nodes_[node_id].leaf = true;
        cluster_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    std::vector<float> left_centroid(center_ptr(pivot_a), center_ptr(pivot_a) + dim_);
    std::vector<float> right_centroid(center_ptr(pivot_b), center_ptr(pivot_b) + dim_);
    std::vector<PID> left;
    std::vector<PID> right;
    left.reserve(center_ids.size() / 2 + 1);
    right.reserve(center_ids.size() / 2 + 1);

    for (size_t iter = 0; iter < config_.cluster_route_iters; ++iter) {
        left.clear();
        right.clear();
        for (PID center_id : center_ids) {
            float dl = point_centroid_l2(center_id, left_centroid);
            float dr = point_centroid_l2(center_id, right_centroid);
            if (dl <= dr) {
                left.push_back(center_id);
            } else {
                right.push_back(center_id);
            }
        }
        if (left.empty() || right.empty()) {
            break;
        }
        recompute_centroid(left, left_centroid);
        recompute_centroid(right, right_centroid);
    }

    if (left.empty() || right.empty()) {
        size_t mid = center_ids.size() / 2;
        left.assign(center_ids.begin(), center_ids.begin() + static_cast<std::ptrdiff_t>(mid));
        right.assign(center_ids.begin() + static_cast<std::ptrdiff_t>(mid), center_ids.end());
        recompute_centroid(left, left_centroid);
        recompute_centroid(right, right_centroid);
    }
    if (left.empty() || right.empty()) {
        cluster_route_nodes_[node_id].leaf = true;
        cluster_route_nodes_[node_id].centers = std::move(center_ids);
        return node_id;
    }

    size_t left_id = build_cluster_route_tree(std::move(left));
    size_t right_id = build_cluster_route_tree(std::move(right));
    ClusterRouteNode& node = cluster_route_nodes_[node_id];
    node.left = left_id;
    node.right = right_id;
    node.left_centroid = std::move(left_centroid);
    node.right_centroid = std::move(right_centroid);
    return node_id;
}

inline PID ATMMGGraphIndex::route_cluster_to_center(const float* query) const {
    if (cluster_route_nodes_.empty()) {
        return route_to_center(query);
    }
    size_t node_id = cluster_route_root_;
    while (!cluster_route_nodes_[node_id].leaf) {
        const ClusterRouteNode& node = cluster_route_nodes_[node_id];
        float dl = 0.0F;
        float dr = 0.0F;
        for (size_t d = 0; d < dim_; ++d) {
            float diff_l = query[d] - node.left_centroid[d];
            float diff_r = query[d] - node.right_centroid[d];
            dl += diff_l * diff_l;
            dr += diff_r * diff_r;
        }
        node_id = (dl <= dr) ? node.left : node.right;
    }

    PID best_center = 0;
    float best_d2 = std::numeric_limits<float>::infinity();
    for (PID center_id : cluster_route_nodes_[node_id].centers) {
        float d2 = l2_to_center(query, center_id);
        if (d2 < best_d2) {
            best_d2 = d2;
            best_center = center_id;
        }
    }
    return best_d2 < std::numeric_limits<float>::infinity() ? best_center
                                                            : route_to_center(query);
}

inline PID ATMMGGraphIndex::route_entry_center(const float* query) const {
    if (config_.center_entry_mode == CenterEntryMode::AnnoyTreeOnly) {
        return route_annoy_to_center(query);
    }
    if (config_.center_entry_mode == CenterEntryMode::ClusterTreeOnly) {
        return route_cluster_to_center(query);
    }
    return route_to_center(query);
}

inline void ATMMGGraphIndex::rotate_base_and_centers() {
    rotator_.reset(choose_rotator<float>(
        dim_, rotator_type_, rotator_impl::padding_requirement(dim_, rotator_type_)
    ));
    padded_dim_ = rotator_->size();

    rotated_base_.assign(num_ * padded_dim_, 0);
    for (size_t i = 0; i < num_; ++i) {
        rotator_->rotate(base_.data() + (i * dim_), rotated_base_.data() + (i * padded_dim_));
    }

    size_t centers_count = num_centers();
    rotated_centers_.assign(centers_count * padded_dim_, 0);
    for (size_t c = 0; c < centers_count; ++c) {
        rotator_->rotate(
            centers_.data() + (c * dim_), rotated_centers_.data() + (c * padded_dim_)
        );
    }
}

inline void ATMMGGraphIndex::build_center_pools() {
    size_t centers_count = num_centers();
    center_topn_.assign(centers_count, {});

    size_t max_keep = std::max(
        {config_.graph_portal_pool_size, config_.center_topn_scan,
         config_.center_topn_coarse_keep}
    );
    max_keep = std::min(max_keep, num_);

    for (size_t c = 0; c < centers_count; ++c) {
        const float* center = centers_.data() + (c * dim_);
        std::vector<ScoredPid> scored(num_);
        for (size_t i = 0; i < num_; ++i) {
            scored[i] = {
                euclidean_sqr<float>(center, base_.data() + (i * dim_), dim_),
                static_cast<PID>(i)
            };
        }
        keep_smallest(scored, max_keep);

        size_t topn_keep = std::min(config_.center_topn_scan, scored.size());
        center_topn_[c].reserve(topn_keep);
        for (size_t i = 0; i < topn_keep; ++i) {
            center_topn_[c].push_back(scored[i].id);
        }
    }
}

inline void ATMMGGraphIndex::build_center_neighbors() {
    size_t centers_count = num_centers();
    center_neighbors_.assign(centers_count, {});
    if (centers_count <= 1) {
        return;
    }

    size_t keep = std::max(
        {config_.graph_build_center_neighbors,
         config_.graph_bridge_center_neighbors,
         config_.center_refine_neighbor_scan}
    );
    keep = std::min(keep, centers_count - 1);
    for (size_t c = 0; c < centers_count; ++c) {
        std::vector<ScoredPid> scored;
        scored.reserve(centers_count - 1);
        const float* center = centers_.data() + (c * dim_);
        for (size_t other = 0; other < centers_count; ++other) {
            if (other == c) {
                continue;
            }
            scored.push_back(
                {euclidean_sqr<float>(center, centers_.data() + (other * dim_), dim_),
                 static_cast<PID>(other)}
            );
        }
        keep_smallest(scored, keep);
        center_neighbors_[c].reserve(scored.size());
        for (const auto& cand : scored) {
            center_neighbors_[c].push_back(cand.id);
        }
    }
}

inline void ATMMGGraphIndex::build_point_center_residual_norms() {
    point_center_residual_norms_.assign(num_, 0.0F);
    for (PID id = 0; id < static_cast<PID>(num_); ++id) {
        PID center_id = point_center_[id];
        const float* point = base_.data() + (static_cast<size_t>(id) * dim_);
        const float* center = centers_.data() + (static_cast<size_t>(center_id) * dim_);
        float d2 = euclidean_sqr_fast(point, center, dim_);
        point_center_residual_norms_[id] = std::sqrt(std::max(d2, 0.0F));
    }
}

inline void ATMMGGraphIndex::rebuild_portal_marks() {
    point_is_portal_.assign(num_, 0);
    for (const auto& portals : center_portal_pool_) {
        for (PID id : portals) {
            if (id < num_) {
                point_is_portal_[id] = 1;
            }
        }
    }
}

inline void ATMMGGraphIndex::build_center_portals() {
    size_t centers_count = num_centers();
    center_portal_pool_.assign(centers_count, {});
    point_is_portal_.assign(num_, 0);
    if (config_.graph_portal_pool_size == 0 || centers_count == 0) {
        return;
    }

    for (PID c = 0; c < static_cast<PID>(centers_count); ++c) {
        const auto& topn = center_topn_[c];
        size_t keep = std::min(config_.graph_portal_pool_size, topn.size());
        auto& portals = center_portal_pool_[c];
        portals.reserve(keep);
        if (keep == 0) {
            continue;
        }

        std::vector<ScoredPid> items;
        items.reserve(topn.size());
        const float* center = centers_.data() + (static_cast<size_t>(c) * dim_);
        for (PID id : topn) {
            items.push_back(
                {euclidean_sqr_fast(
                     center, base_.data() + (static_cast<size_t>(id) * dim_), dim_
                 ),
                 id}
            );
        }
        std::sort(items.begin(), items.end());

        for (const auto& cand : items) {
            bool ok = true;
            for (PID selected : portals) {
                if (point_point_l2(selected, cand.id) < cand.distance) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                portals.push_back(cand.id);
                if (portals.size() >= keep) {
                    break;
                }
            }
        }
        if (portals.size() < keep) {
            for (const auto& cand : items) {
                if (std::find(portals.begin(), portals.end(), cand.id) !=
                    portals.end()) {
                    continue;
                }
                portals.push_back(cand.id);
                if (portals.size() >= keep) {
                    break;
                }
            }
        }
    }

    rebuild_portal_marks();
}

inline void ATMMGGraphIndex::build_center_coarse_projection() {
    center_coarse_dims_.clear();
    center_coarse_values_.clear();
    size_t centers_count = num_centers();
    size_t keep_dims = std::min(config_.center_coarse_projection_dims, dim_);
    if (centers_count == 0 || keep_dims == 0 || config_.center_coarse_keep >= centers_count) {
        return;
    }

    std::vector<double> mean(dim_, 0.0);
    std::vector<double> sq_sum(dim_, 0.0);
    for (size_t c = 0; c < centers_count; ++c) {
        const float* center = centers_.data() + (c * dim_);
        for (size_t d = 0; d < dim_; ++d) {
            double value = static_cast<double>(center[d]);
            mean[d] += value;
            sq_sum[d] += value * value;
        }
    }

    double inv_count = 1.0 / static_cast<double>(centers_count);
    std::vector<ScoredPid> dims;
    dims.reserve(dim_);
    for (size_t d = 0; d < dim_; ++d) {
        double m = mean[d] * inv_count;
        double variance = sq_sum[d] * inv_count - (m * m);
        dims.push_back({-static_cast<float>(variance), static_cast<PID>(d)});
    }

    keep_smallest(dims, keep_dims);
    std::sort(dims.begin(), dims.end(), [](const ScoredPid& a, const ScoredPid& b) {
        if (a.distance != b.distance) {
            return a.distance < b.distance;
        }
        return a.id < b.id;
    });

    center_coarse_dims_.reserve(dims.size());
    for (const auto& dim_score : dims) {
        center_coarse_dims_.push_back(static_cast<size_t>(dim_score.id));
    }

    center_coarse_values_.resize(centers_count * center_coarse_dims_.size());
    for (size_t c = 0; c < centers_count; ++c) {
        const float* center = centers_.data() + (c * dim_);
        float* projected =
            center_coarse_values_.data() + (c * center_coarse_dims_.size());
        for (size_t i = 0; i < center_coarse_dims_.size(); ++i) {
            projected[i] = center[center_coarse_dims_[i]];
        }
    }
}

inline void ATMMGGraphIndex::build_center_super_layer() {
    center_super_centers_.clear();
    center_super_offsets_.clear();
    center_super_fine_centers_.clear();
    const size_t centers_count = num_centers();
    if (!config_.center_super_level_scan || centers_count == 0 || dim_ == 0) {
        return;
    }

    size_t super_count = std::min(config_.center_super_count, centers_count);
    if (super_count == 0 || super_count >= centers_count) {
        return;
    }

    center_super_centers_.assign(super_count * dim_, 0.0F);
    std::vector<float> min_dist(centers_count, std::numeric_limits<float>::infinity());
    std::vector<PID> selected;
    selected.reserve(super_count);
    selected.push_back(0);
    std::copy(
        centers_.data(),
        centers_.data() + dim_,
        center_super_centers_.data()
    );
    for (size_t s = 1; s < super_count; ++s) {
        const float* prev =
            center_super_centers_.data() + ((s - 1) * dim_);
        for (size_t c = 0; c < centers_count; ++c) {
            float d2 = euclidean_sqr_fast(
                prev,
                centers_.data() + (c * dim_),
                dim_
            );
            min_dist[c] = std::min(min_dist[c], d2);
        }
        PID best = 0;
        float best_dist = -1.0F;
        for (size_t c = 0; c < centers_count; ++c) {
            if (min_dist[c] > best_dist &&
                std::find(selected.begin(), selected.end(), static_cast<PID>(c)) ==
                    selected.end()) {
                best_dist = min_dist[c];
                best = static_cast<PID>(c);
            }
        }
        selected.push_back(best);
        std::copy(
            centers_.data() + (static_cast<size_t>(best) * dim_),
            centers_.data() + ((static_cast<size_t>(best) + 1) * dim_),
            center_super_centers_.data() + (s * dim_)
        );
    }

    std::vector<PID> assignment(centers_count, 0);
    std::vector<float> accum(super_count * dim_, 0.0F);
    std::vector<size_t> counts(super_count, 0);
    constexpr size_t kSuperRefineIters = 4;
    for (size_t iter = 0; iter < kSuperRefineIters; ++iter) {
        std::fill(accum.begin(), accum.end(), 0.0F);
        std::fill(counts.begin(), counts.end(), 0);
        for (size_t c = 0; c < centers_count; ++c) {
            const float* center = centers_.data() + (c * dim_);
            PID best_super = 0;
            float best_dist = std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < super_count; ++s) {
                float d2 = euclidean_sqr_fast(
                    center,
                    center_super_centers_.data() + (s * dim_),
                    dim_
                );
                if (d2 < best_dist) {
                    best_dist = d2;
                    best_super = static_cast<PID>(s);
                }
            }
            assignment[c] = best_super;
            float* dst = accum.data() + (static_cast<size_t>(best_super) * dim_);
            for (size_t d = 0; d < dim_; ++d) {
                dst[d] += center[d];
            }
            ++counts[best_super];
        }
        for (size_t s = 0; s < super_count; ++s) {
            if (counts[s] == 0) {
                continue;
            }
            float inv = 1.0F / static_cast<float>(counts[s]);
            float* dst = center_super_centers_.data() + (s * dim_);
            const float* src = accum.data() + (s * dim_);
            for (size_t d = 0; d < dim_; ++d) {
                dst[d] = src[d] * inv;
            }
        }
    }

    size_t overlap = std::max<size_t>(1, std::min(config_.center_super_overlap, super_count));
    std::vector<std::vector<PID>> buckets(super_count);
    std::vector<ScoredPid> super_scores;
    super_scores.reserve(super_count);
    for (size_t c = 0; c < centers_count; ++c) {
        super_scores.clear();
        const float* center = centers_.data() + (c * dim_);
        for (size_t s = 0; s < super_count; ++s) {
            super_scores.push_back(
                {euclidean_sqr_fast(
                     center,
                     center_super_centers_.data() + (s * dim_),
                     dim_
                 ),
                 static_cast<PID>(s)}
            );
        }
        keep_smallest(super_scores, overlap);
        for (const auto& cand : super_scores) {
            buckets[cand.id].push_back(static_cast<PID>(c));
        }
    }

    center_super_offsets_.assign(super_count + 1, 0);
    for (size_t s = 0; s < super_count; ++s) {
        auto& fine = buckets[s];
        const float* super_center = center_super_centers_.data() + (s * dim_);
        std::sort(fine.begin(), fine.end(), [&](PID a, PID b) {
            float da = euclidean_sqr_fast(
                super_center,
                centers_.data() + (static_cast<size_t>(a) * dim_),
                dim_
            );
            float db = euclidean_sqr_fast(
                super_center,
                centers_.data() + (static_cast<size_t>(b) * dim_),
                dim_
            );
            if (da != db) {
                return da < db;
            }
            return a < b;
        });
        center_super_offsets_[s + 1] = center_super_offsets_[s] + fine.size();
    }
    center_super_fine_centers_.reserve(center_super_offsets_.back());
    for (const auto& fine : buckets) {
        center_super_fine_centers_.insert(
            center_super_fine_centers_.end(), fine.begin(), fine.end()
        );
    }
}

inline void ATMMGGraphIndex::build_graph_prefilter_projection() {
    graph_prefilter_dims_.clear();
    size_t keep_dims = 0;
    if (config_.graph_neighbor_prefilter_keep > 0 &&
        config_.graph_neighbor_prefilter_dims > 0) {
        keep_dims = std::max(keep_dims, config_.graph_neighbor_prefilter_dims);
    }
    if (config_.graph_final_prefilter_keep > 0 &&
        config_.graph_final_prefilter_dims > 0) {
        keep_dims = std::max(keep_dims, config_.graph_final_prefilter_dims);
    }
    if (keep_dims == 0 || dim_ == 0) {
        return;
    }

    keep_dims = std::min(keep_dims, dim_);
    graph_prefilter_dims_.reserve(keep_dims);
    auto append_dim = [&](size_t dim_id) {
        if (dim_id >= dim_ || graph_prefilter_dims_.size() >= keep_dims) {
            return;
        }
        if (std::find(
                graph_prefilter_dims_.begin(), graph_prefilter_dims_.end(), dim_id
            ) == graph_prefilter_dims_.end()) {
            graph_prefilter_dims_.push_back(dim_id);
        }
    };

    for (size_t dim_id : center_coarse_dims_) {
        append_dim(dim_id);
    }
    for (size_t dim_id = 0; graph_prefilter_dims_.size() < keep_dims && dim_id < dim_;
         ++dim_id) {
        append_dim(dim_id);
    }
}

inline void ATMMGGraphIndex::build_query_adaptive_l1_thresholds() {
    query_adaptive_l1_ready_ = false;
    query_adaptive_l1_low_ = 0.0F;
    query_adaptive_l1_high_ = 0.0F;
    if (config_.graph_query_adaptive_center_margin ||
        config_.graph_query_adaptive_ef_min == 0 ||
        config_.graph_query_adaptive_ef_min >= config_.ef_search || num_ == 0 ||
        dim_ == 0) {
        return;
    }

    std::vector<float> l1_norms(num_);
    for (size_t i = 0; i < num_; ++i) {
        l1_norms[i] = query_l1_norm(base_.data() + (i * dim_));
    }

    auto quantile_value = [&](float quantile) {
        if (l1_norms.empty()) {
            return 0.0F;
        }
        size_t pos = static_cast<size_t>(
            std::round(quantile * static_cast<float>(l1_norms.size() - 1))
        );
        pos = std::min(pos, l1_norms.size() - 1);
        auto nth = l1_norms.begin() + static_cast<std::ptrdiff_t>(pos);
        std::nth_element(l1_norms.begin(), nth, l1_norms.end());
        return *nth;
    };

    query_adaptive_l1_low_ =
        quantile_value(config_.graph_query_adaptive_l1_low_quantile);
    query_adaptive_l1_high_ =
        quantile_value(config_.graph_query_adaptive_l1_high_quantile);
    if (query_adaptive_l1_high_ < query_adaptive_l1_low_) {
        std::swap(query_adaptive_l1_high_, query_adaptive_l1_low_);
    }
    query_adaptive_l1_ready_ = true;
}

inline uint64_t ATMMGGraphIndex::hash_spectrum_code(
    PID center, const float* rotated_vec
) const {
    const size_t bits = std::min(config_.hash_spectrum_bits, padded_dim_);
    const float* center_vec =
        rotated_centers_.data() + (static_cast<size_t>(center) * padded_dim_);
    uint64_t code = 0;
    for (size_t bit = 0; bit < bits; ++bit) {
        size_t d = (bit * 73 + 17) % padded_dim_;
        size_t dim_offset = static_cast<size_t>(center) * config_.hash_spectrum_bits + bit;
        if (!hash_spectrum_dims_.empty() && dim_offset < hash_spectrum_dims_.size()) {
            d = hash_spectrum_dims_[dim_offset];
        }
        if (rotated_vec[d] >= center_vec[d]) {
            code |= (uint64_t{1} << bit);
        }
    }
    return code;
}

inline uint32_t ATMMGGraphIndex::popcount64(uint64_t value) {
#if defined(_MSC_VER) && defined(_M_X64)
    return static_cast<uint32_t>(__popcnt64(value));
#elif defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(value));
#else
    uint32_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

inline uint32_t ATMMGGraphIndex::hash_spectrum_segment_min(
    uint64_t a, uint64_t b
) const {
    const size_t bits = std::min(config_.hash_spectrum_bits, padded_dim_);
    const size_t segment_bits = std::min(config_.hash_spectrum_segment_bits, bits);
    const uint64_t mask =
        segment_bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << segment_bits) - 1);
    uint32_t best = static_cast<uint32_t>(segment_bits);
    for (size_t offset = 0; offset < bits; offset += segment_bits) {
        uint64_t diff = ((a ^ b) >> offset) & mask;
        best = std::min(best, popcount64(diff));
    }
    return best;
}

inline bool ATMMGGraphIndex::residual_hash_ready() const {
    return config_.use_residual_hash_spectrum && residual_hash_words_ > 0 &&
           !residual_hash_dims_.empty() && !residual_hash_codes_.empty();
}

inline bool ATMMGGraphIndex::residual_hash_codes_ready() const {
    return residual_hash_words_ > 0 && !residual_hash_dims_.empty() &&
           !residual_hash_codes_.empty();
}

inline bool ATMMGGraphIndex::residual_hash_bucket_ready() const {
    return config_.use_residual_hash_bucket_graph && residual_hash_codes_ready() &&
           residual_bucket_count_ > 0 && residual_bucket_take_ > 0 &&
           residual_bucket_stride_ > 0 && !residual_bucket_indices_.empty() &&
           !residual_bucket_counts_.empty() && residual_bucket_code_bits_ > 0 &&
           !residual_bucket_center_thresholds_.empty();
}

inline bool ATMMGGraphIndex::residual_hash_query_bucket_ready() const {
    return residual_hash_bucket_ready() && config_.residual_hash_build_query_buckets &&
           residual_query_bucket_width_ > 0 && residual_query_bucket_stride_ > 0 &&
           !residual_query_bucket_indices_.empty() && !residual_query_bucket_counts_.empty();
}

inline size_t ATMMGGraphIndex::residual_hash_bucket_id(
    const uint64_t* code
) const {
    if (residual_bucket_count_ <= 1 || code == nullptr) {
        return 0;
    }
    const uint64_t v = code[0];
    if ((residual_bucket_count_ & (residual_bucket_count_ - 1)) == 0) {
        return static_cast<size_t>(v & static_cast<uint64_t>(residual_bucket_count_ - 1));
    }
    return static_cast<size_t>(v % residual_bucket_count_);
}

inline size_t ATMMGGraphIndex::residual_hash_bucket_id_for_node(
    const float* query,
    PID node_id
) const {
    if (query == nullptr || residual_bucket_count_ <= 1 ||
        residual_bucket_code_bits_ == 0 || node_id >= num_) {
        return 0;
    }
    const float* node = base_.data() + (static_cast<size_t>(node_id) * dim_);
    size_t code = 0;
    for (size_t bit = 0; bit < residual_bucket_code_bits_; ++bit) {
        const size_t d = residual_hash_dims_[bit];
        const float residual = query[d] - node[d];
        const bool positive =
            residual_hash_signs_[bit] >= 0 ? residual >= 0.0F : residual <= 0.0F;
        if (positive) {
            code |= (size_t{1} << bit);
        }
    }
    if ((residual_bucket_count_ & (residual_bucket_count_ - 1)) == 0) {
        return code & (residual_bucket_count_ - 1);
    }
    return code % residual_bucket_count_;
}

inline size_t ATMMGGraphIndex::residual_hash_bucket_id_for_center(
    const float* query,
    PID center_id
) const {
    if (query == nullptr || residual_bucket_count_ <= 1 ||
        residual_bucket_code_bits_ == 0 || num_centers() == 0) {
        return 0;
    }
    const size_t center_count = num_centers();
    size_t center_index = static_cast<size_t>(center_id);
    if (center_index >= center_count) {
        center_index = 0;
    }
    const float* thresholds =
        residual_bucket_center_thresholds_.data() +
        center_index * residual_bucket_code_bits_;
    size_t code = 0;
    for (size_t bit = 0; bit < residual_bucket_code_bits_; ++bit) {
        const size_t d = residual_hash_dims_[bit];
        const float signed_query =
            residual_hash_signs_[bit] >= 0 ? query[d] : -query[d];
        if (signed_query >= thresholds[bit]) {
            code |= (size_t{1} << bit);
        }
    }
    if ((residual_bucket_count_ & (residual_bucket_count_ - 1)) == 0) {
        return code & (residual_bucket_count_ - 1);
    }
    return code % residual_bucket_count_;
}

inline void ATMMGGraphIndex::residual_hash_code(
    const float* vec,
    const float* center,
    uint64_t* out
) const {
    const size_t words = residual_hash_words_;
    const size_t bits = config_.residual_hash_bits;
    for (size_t w = 0; w < words; ++w) {
        const size_t bit_base = w * 64;
        const size_t bit_end = std::min(bit_base + 64, bits);
        uint64_t code = 0;
        for (size_t bit = bit_base; bit < bit_end; ++bit) {
            const size_t d = residual_hash_dims_[bit];
            const float residual = vec[d] - center[d];
            const bool positive =
                residual_hash_signs_[bit] >= 0 ? residual >= 0.0F : residual <= 0.0F;
            if (positive) {
                code |= (uint64_t{1} << (bit - bit_base));
            }
        }
        out[w] = code;
    }
}

inline void ATMMGGraphIndex::residual_hash_prepare_query(
    const float* query,
    float* out
) const {
    const size_t bits = config_.residual_hash_bits;
    for (size_t bit = 0; bit < bits; ++bit) {
        const size_t d = residual_hash_dims_[bit];
        const float value = query[d];
        out[bit] = residual_hash_signs_[bit] >= 0 ? value : -value;
    }
}

inline void ATMMGGraphIndex::residual_hash_code_from_prepared_query(
    const float* center,
    const float* prepared_query,
    uint64_t* out
) const {
    const size_t words = residual_hash_words_;
    const size_t bits = config_.residual_hash_bits;
    for (size_t w = 0; w < words; ++w) {
        const size_t bit_base = w * 64;
        const size_t bit_end = std::min(bit_base + 64, bits);
        uint64_t code = 0;
        for (size_t bit = bit_base; bit < bit_end; ++bit) {
            const size_t d = residual_hash_dims_[bit];
            const float center_value = center[d];
            const float signed_center =
                residual_hash_signs_[bit] >= 0 ? center_value : -center_value;
            if (prepared_query[bit] >= signed_center) {
                code |= (uint64_t{1} << (bit - bit_base));
            }
        }
        out[w] = code;
    }
}

inline bool ATMMGGraphIndex::residual_hash_pass(
    const uint64_t* query_code,
    const uint64_t* edge_code
) const {
    const uint32_t full_radius = static_cast<uint32_t>(config_.residual_radius_full);
    const uint32_t segment_radius =
        static_cast<uint32_t>(config_.residual_radius_segment);
    const size_t words = residual_hash_words_;
    const size_t segments = config_.residual_hash_segments;
    const size_t segment_bits = config_.residual_hash_bits / segments;

    if (words == 1) {
        const uint64_t diff = query_code[0] ^ edge_code[0];
        if (segments <= 1 || segment_bits >= 64) {
            const uint32_t full = popcount64(diff);
            return full <= full_radius || full <= segment_radius;
        }
        uint64_t remaining = diff;
        const uint64_t mask =
            segment_bits == 64 ? ~uint64_t{0} : ((uint64_t{1} << segment_bits) - 1);
        uint32_t full = 0;
        for (size_t s = 0; s < segments; ++s) {
            const uint32_t segment = popcount64(remaining & mask);
            if (segment <= segment_radius) {
                return true;
            }
            full += segment;
            remaining >>= segment_bits;
        }
        return full <= full_radius;
    }

    if (segment_bits == 64 && segments == words) {
        uint32_t full = 0;
        for (size_t w = 0; w < words; ++w) {
            const uint32_t segment = popcount64(query_code[w] ^ edge_code[w]);
            if (segment <= segment_radius) {
                return true;
            }
            full += segment;
        }
        return full <= full_radius;
    }

    uint32_t full = 0;
    for (size_t s = 0; s < config_.residual_hash_segments; ++s) {
        size_t bit_begin = s * segment_bits;
        size_t bit_end = bit_begin + segment_bits;
        size_t word_begin = bit_begin / 64;
        size_t word_end = (bit_end + 63) / 64;
        uint32_t segment = 0;
        for (size_t w = word_begin; w < word_end; ++w) {
            size_t local_begin = w == word_begin ? (bit_begin % 64) : 0;
            size_t local_end = w + 1 == word_end ? ((bit_end - 1) % 64) + 1 : 64;
            uint64_t mask;
            if (local_begin == 0 && local_end == 64) {
                mask = ~uint64_t{0};
            } else {
                size_t width = local_end - local_begin;
                mask = ((uint64_t{1} << width) - 1) << local_begin;
            }
            segment += popcount64((query_code[w] ^ edge_code[w]) & mask);
        }
        if (segment <= config_.residual_radius_segment) {
            return true;
        }
        full += segment;
    }
    return full <= config_.residual_radius_full;
}

inline void ATMMGGraphIndex::build_residual_hash_spectrum() {
    residual_hash_dims_.clear();
    residual_hash_signs_.clear();
    residual_hash_codes_.clear();
    residual_hash_words_ = 0;
    if ((!config_.use_residual_hash_spectrum &&
         !config_.use_residual_hash_bucket_graph) ||
        dim_ == 0 ||
        graph_indices_.empty() || graph_offsets_.empty()) {
        return;
    }

    residual_hash_words_ = config_.residual_hash_bits / 64;
    residual_hash_dims_.resize(config_.residual_hash_bits);
    residual_hash_signs_.resize(config_.residual_hash_bits);

    std::mt19937 rng(static_cast<uint32_t>(config_.residual_hash_seed));
    std::uniform_int_distribution<size_t> dim_dist(0, dim_ - 1);
    std::uniform_int_distribution<int> sign_dist(0, 1);
    for (size_t bit = 0; bit < config_.residual_hash_bits; ++bit) {
        residual_hash_dims_[bit] = static_cast<uint16_t>(dim_dist(rng));
        residual_hash_signs_[bit] = sign_dist(rng) == 0 ? int8_t{-1} : int8_t{1};
    }

    residual_hash_codes_.assign(graph_indices_.size() * residual_hash_words_, uint64_t{0});
    const size_t centers_count = num_centers();
    for (PID node_id = 0; node_id < num_; ++node_id) {
        const float* center =
            base_.data() + (static_cast<size_t>(node_id) * dim_);
        if (centers_count > 0 && !point_center_.empty() &&
            static_cast<size_t>(node_id) < point_center_.size()) {
            PID owner_center = point_center_[node_id];
            if (static_cast<size_t>(owner_center) < centers_count) {
                center = centers_.data() + (static_cast<size_t>(owner_center) * dim_);
            }
        }
        size_t begin = graph_offsets_[node_id];
        size_t end = graph_offsets_[static_cast<size_t>(node_id) + 1];
        for (size_t pos = begin; pos < end; ++pos) {
            PID neighbor_id = graph_indices_[pos];
            const float* neighbor =
                base_.data() + (static_cast<size_t>(neighbor_id) * dim_);
            uint64_t* code =
                residual_hash_codes_.data() + (pos * residual_hash_words_);
            residual_hash_code(neighbor, center, code);
        }
    }
}

inline void ATMMGGraphIndex::build_residual_hash_bucket_graph() {
    residual_bucket_indices_.clear();
    residual_bucket_counts_.clear();
    residual_bucket_center_thresholds_.clear();
    residual_bucket_code_bits_ = 0;
    residual_query_bucket_indices_.clear();
    residual_query_bucket_counts_.clear();
    residual_query_bucket_width_ = 0;
    residual_query_bucket_stride_ = 0;
    residual_bucket_count_ = 0;
    residual_bucket_take_ = 0;
    residual_bucket_stride_ = 0;
    if (!config_.use_residual_hash_bucket_graph || !residual_hash_codes_ready() ||
        graph_indices_.empty() || graph_offsets_.empty() || num_ == 0) {
        return;
    }

    residual_bucket_count_ = std::max<size_t>(1, config_.residual_hash_bucket_count);
    residual_bucket_take_ = std::max<size_t>(1, config_.residual_hash_bucket_take);
    residual_bucket_code_bits_ = 1;
    if (residual_bucket_count_ > 1) {
        residual_bucket_code_bits_ = 0;
        size_t bucket_slots = 1;
        while (bucket_slots < residual_bucket_count_) {
            bucket_slots <<= 1;
            ++residual_bucket_code_bits_;
        }
        residual_bucket_code_bits_ =
            std::max<size_t>(1, std::min(residual_bucket_code_bits_, residual_hash_words_ * 64));
    }
    residual_bucket_stride_ = residual_bucket_count_ * residual_bucket_take_;
    if (residual_bucket_stride_ == 0) {
        return;
    }
    residual_bucket_counts_.assign(num_ * residual_bucket_count_, uint16_t{0});
    residual_bucket_indices_.assign(num_ * residual_bucket_stride_, PID{0});
    const size_t centers_count = num_centers();
    if (centers_count > 0) {
        residual_bucket_center_thresholds_.assign(
            centers_count * residual_bucket_code_bits_,
            0.0F
        );
        for (PID center_id = 0; center_id < static_cast<PID>(centers_count); ++center_id) {
            const float* center =
                centers_.data() + (static_cast<size_t>(center_id) * dim_);
            float* dst =
                residual_bucket_center_thresholds_.data() +
                static_cast<size_t>(center_id) * residual_bucket_code_bits_;
            for (size_t bit = 0; bit < residual_bucket_code_bits_; ++bit) {
                const size_t d = residual_hash_dims_[bit];
                dst[bit] = residual_hash_signs_[bit] >= 0 ? center[d] : -center[d];
            }
        }
    }

    const size_t hot_count = config_.residual_hash_hot_count;
    for (PID node = 0; node < num_; ++node) {
        const size_t begin = graph_offsets_[node];
        const size_t end = graph_offsets_[static_cast<size_t>(node) + 1];
        const size_t count = end - begin;
        const size_t hot = std::min(hot_count, count);
        for (size_t pos = begin + hot; pos < end; ++pos) {
            const uint64_t* code =
                residual_hash_codes_.data() + (pos * residual_hash_words_);
            const size_t bucket = residual_hash_bucket_id(code);
            uint16_t& bucket_count =
                residual_bucket_counts_[static_cast<size_t>(node) * residual_bucket_count_ +
                                        bucket];
            if (bucket_count >= residual_bucket_take_) {
                continue;
            }
            const size_t dst =
                static_cast<size_t>(node) * residual_bucket_stride_ +
                bucket * residual_bucket_take_ + bucket_count;
            residual_bucket_indices_[dst] = graph_indices_[pos];
            ++bucket_count;
        }
    }

    if (!config_.residual_hash_build_query_buckets) {
        return;
    }

    const size_t bucket_probe =
        std::min(config_.residual_hash_bucket_probe, residual_bucket_count_);
    residual_query_bucket_width_ =
        config_.residual_hash_hot_count +
        (bucket_probe * residual_bucket_take_) +
        config_.residual_hash_cold_count;
    residual_query_bucket_width_ =
        std::max<size_t>(1, std::min(residual_query_bucket_width_, config_.graph_degree));
    residual_query_bucket_stride_ = residual_bucket_count_ * residual_query_bucket_width_;
    residual_query_bucket_counts_.assign(num_ * residual_bucket_count_, uint16_t{0});
    residual_query_bucket_indices_.assign(
        num_ * residual_query_bucket_stride_,
        PID{0}
    );

    const bool pow2_bucket =
        (residual_bucket_count_ & (residual_bucket_count_ - 1)) == 0;
    size_t bucket_bits = 1;
    if (pow2_bucket) {
        bucket_bits = 0;
        for (size_t v = residual_bucket_count_; v > 1; v >>= 1) {
            ++bucket_bits;
        }
        bucket_bits = std::max<size_t>(1, bucket_bits);
    }
    std::vector<PID> merged;
    merged.reserve(residual_query_bucket_width_ + residual_bucket_take_);
    auto append_unique = [&](PID id) {
        if (merged.size() >= residual_query_bucket_width_) {
            return;
        }
        if (std::find(merged.begin(), merged.end(), id) == merged.end()) {
            merged.push_back(id);
        }
    };
    for (PID node = 0; node < num_; ++node) {
        const size_t begin = graph_offsets_[node];
        const size_t end = graph_offsets_[static_cast<size_t>(node) + 1];
        const size_t count = end - begin;
        const size_t hot = std::min(config_.residual_hash_hot_count, count);
        for (size_t bucket0 = 0; bucket0 < residual_bucket_count_; ++bucket0) {
            merged.clear();
            for (size_t ni = 0; ni < hot; ++ni) {
                append_unique(graph_indices_[begin + ni]);
            }
            for (size_t probe = 0; probe < bucket_probe; ++probe) {
                size_t bucket = bucket0;
                if (probe > 0) {
                    if (pow2_bucket) {
                        const size_t bit = (probe - 1) % bucket_bits;
                        bucket = bucket0 ^ (size_t{1} << bit);
                    } else {
                        bucket = (bucket0 + probe) % residual_bucket_count_;
                    }
                }
                const size_t count_index =
                    static_cast<size_t>(node) * residual_bucket_count_ + bucket;
                const size_t bucket_count = residual_bucket_counts_[count_index];
                const size_t bucket_begin =
                    static_cast<size_t>(node) * residual_bucket_stride_ +
                    bucket * residual_bucket_take_;
                for (size_t i = 0; i < bucket_count; ++i) {
                    append_unique(residual_bucket_indices_[bucket_begin + i]);
                }
            }
            const size_t cold_begin = hot;
            const size_t cold_end = std::min(count, hot + config_.residual_hash_cold_count);
            for (size_t ni = cold_begin; ni < cold_end; ++ni) {
                append_unique(graph_indices_[begin + ni]);
            }
            const size_t out_count =
                std::min(merged.size(), size_t{std::numeric_limits<uint16_t>::max()});
            const size_t count_index =
                static_cast<size_t>(node) * residual_bucket_count_ + bucket0;
            residual_query_bucket_counts_[count_index] =
                static_cast<uint16_t>(out_count);
            PID* dst =
                residual_query_bucket_indices_.data() +
                static_cast<size_t>(node) * residual_query_bucket_stride_ +
                bucket0 * residual_query_bucket_width_;
            std::copy_n(merged.data(), out_count, dst);
        }
    }
}

inline void ATMMGGraphIndex::build_hash_neighborhood_spectrum() {
    const size_t centers_count = num_centers();
    hash_spectrum_ids_.clear();
    hash_spectrum_codes_.clear();
    hash_spectrum_dims_.clear();
    if (!config_.use_hash_neighborhood_spectrum ||
        config_.hash_spectrum_size == 0 || centers_count == 0) {
        return;
    }

    hash_spectrum_ids_.assign(centers_count, {});
    hash_spectrum_codes_.assign(centers_count, {});
    const size_t bits = std::min(config_.hash_spectrum_bits, padded_dim_);
    hash_spectrum_dims_.assign(centers_count * config_.hash_spectrum_bits, 0);
    std::vector<double> residual_sum(padded_dim_, 0.0);
    std::vector<double> residual_sum_sq(padded_dim_, 0.0);
    std::vector<double> residual_var(padded_dim_, 0.0);
    std::vector<size_t> dim_order(padded_dim_, 0);

    for (PID c = 0; c < static_cast<PID>(centers_count); ++c) {
        const auto& topn = center_topn_[c];
        auto& ids = hash_spectrum_ids_[c];
        auto& codes = hash_spectrum_codes_[c];
        const size_t scan = std::min(config_.hash_spectrum_pool_scan, topn.size());
        const size_t target = std::min(config_.hash_spectrum_size, scan);
        if (scan == 0 || target == 0) {
            continue;
        }
        ids.reserve(target);
        codes.reserve(target);

        std::vector<PID> scanned_ids;
        std::vector<uint64_t> scanned_codes;
        scanned_ids.reserve(scan);
        scanned_codes.reserve(scan);

        std::fill(residual_sum.begin(), residual_sum.end(), 0.0);
        std::fill(residual_sum_sq.begin(), residual_sum_sq.end(), 0.0);
        const float* center_rotated =
            rotated_centers_.data() + (static_cast<size_t>(c) * padded_dim_);
        for (size_t i = 0; i < scan; ++i) {
            PID id = topn[i];
            const float* point_rotated =
                rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_);
            for (size_t d = 0; d < padded_dim_; ++d) {
                double residual =
                    static_cast<double>(point_rotated[d] - center_rotated[d]);
                residual_sum[d] += residual;
                residual_sum_sq[d] += residual * residual;
            }
        }
        const double inv_scan = 1.0 / static_cast<double>(scan);
        for (size_t d = 0; d < padded_dim_; ++d) {
            double mean = residual_sum[d] * inv_scan;
            residual_var[d] = std::max(0.0, residual_sum_sq[d] * inv_scan - mean * mean);
        }
        std::iota(dim_order.begin(), dim_order.end(), size_t{0});
        if (bits < dim_order.size()) {
            std::nth_element(
                dim_order.begin(),
                dim_order.begin() + static_cast<std::ptrdiff_t>(bits),
                dim_order.end(),
                [&](size_t a, size_t b) {
                    if (residual_var[a] == residual_var[b]) {
                        return a < b;
                    }
                    return residual_var[a] > residual_var[b];
                }
            );
            dim_order.resize(bits);
        }
        std::sort(
            dim_order.begin(),
            dim_order.end(),
            [&](size_t a, size_t b) {
                if (residual_var[a] == residual_var[b]) {
                    return a < b;
                }
                return residual_var[a] > residual_var[b];
            }
        );
        for (size_t bit = 0; bit < bits; ++bit) {
            hash_spectrum_dims_[static_cast<size_t>(c) * config_.hash_spectrum_bits + bit] =
                static_cast<uint16_t>(dim_order[bit]);
        }
        if (dim_order.size() != padded_dim_) {
            dim_order.resize(padded_dim_);
        }

        for (size_t i = 0; i < scan; ++i) {
            PID id = topn[i];
            const float* point_rotated =
                rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_);
            uint64_t code = hash_spectrum_code(c, point_rotated);
            scanned_ids.push_back(id);
            scanned_codes.push_back(code);

            bool diverse = true;
            for (uint64_t selected_code : codes) {
                if (popcount64(code ^ selected_code) < config_.hash_spectrum_min_hamming) {
                    diverse = false;
                    break;
                }
            }
            if (diverse) {
                ids.push_back(id);
                codes.push_back(code);
                if (ids.size() >= target) {
                    break;
                }
            }
        }

        if (ids.size() < target) {
            for (size_t i = 0; i < scanned_ids.size() && ids.size() < target; ++i) {
                PID id = scanned_ids[i];
                if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
                    continue;
                }
                ids.push_back(id);
                codes.push_back(scanned_codes[i]);
            }
        }
    }
}

inline void ATMMGGraphIndex::quantize_center_codes() {
    size_t centers_count = num_centers();
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    center_codes_.reset(centers_count, bin_bytes, ex_bytes);

    std::vector<float> zero(padded_dim_, 0);
    for (size_t c = 0; c < centers_count; ++c) {
        quant::quantize_split_single(
            rotated_centers_.data() + (c * padded_dim_),
            zero.data(),
            padded_dim_,
            ex_bits_,
            center_codes_.bin(c),
            center_codes_.ex(c),
            METRIC_L2,
            index_config_
        );
    }
}

inline void ATMMGGraphIndex::quantize_topn_codes() {
    size_t centers_count = num_centers();
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    size_t batch_bytes =
        round_up_to_multiple(BatchDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    topn_codes_.resize(centers_count);
    topn_batch_codes_.resize(centers_count);

    for (size_t c = 0; c < centers_count; ++c) {
        topn_codes_[c].reset(center_topn_[c].size(), bin_bytes, ex_bytes);
        topn_batch_codes_[c].reset(center_topn_[c].size(), batch_bytes);
        const float* center = rotated_centers_.data() + (c * padded_dim_);
        for (size_t i = 0; i < center_topn_[c].size(); ++i) {
            PID id = center_topn_[c][i];
            quant::quantize_split_single(
                rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_),
                center,
                padded_dim_,
                ex_bits_,
                topn_codes_[c].bin(i),
                topn_codes_[c].ex(i),
                METRIC_L2,
                index_config_
            );
        }

        std::vector<float> batch_data(scan::kBatchSize * padded_dim_, 0);
        for (size_t batch_id = 0; batch_id < topn_batch_codes_[c].batch_count; ++batch_id) {
            std::fill(batch_data.begin(), batch_data.end(), 0.0F);
            size_t offset = batch_id * scan::kBatchSize;
            size_t count =
                std::min(scan::kBatchSize, center_topn_[c].size() - offset);
            for (size_t j = 0; j < count; ++j) {
                PID id = center_topn_[c][offset + j];
                std::copy(
                    rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_),
                    rotated_base_.data() + ((static_cast<size_t>(id) + 1) * padded_dim_),
                    batch_data.data() + (j * padded_dim_)
                );
            }
            quant::quantize_one_batch(
                batch_data.data(),
                center,
                count,
                padded_dim_,
                topn_batch_codes_[c].batch(batch_id),
                METRIC_L2
            );
        }
    }
}

inline void ATMMGGraphIndex::quantize_point_codes() {
    size_t bin_bytes =
        round_up_to_multiple(BinDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    size_t ex_bytes = config_.graph_search_full_quant
                          ? ExDataMap<float>::data_bytes(padded_dim_, ex_bits_)
                          : 0;
    point_codes_.reset(num_, bin_bytes, ex_bytes);

    for (size_t i = 0; i < num_; ++i) {
        PID center_id = point_center_[i];
        const float* center =
            rotated_centers_.data() + (static_cast<size_t>(center_id) * padded_dim_);
        if (config_.graph_search_full_quant) {
            quant::quantize_split_single(
                rotated_base_.data() + (i * padded_dim_),
                center,
                padded_dim_,
                ex_bits_,
                point_codes_.bin(i),
                point_codes_.ex(i),
                METRIC_L2,
                index_config_
            );
        } else {
            quant::quantize_compact_one_bit(
                rotated_base_.data() + (i * padded_dim_),
                center,
                padded_dim_,
                point_codes_.bin(i),
                METRIC_L2
            );
        }
    }
}

inline void ATMMGGraphIndex::quantize_graph_edge_batch_codes() {
    graph_edge_batch_codes_ = BatchCodeBank{};
    graph_edge_batch_offsets_.clear();
    if (!config_.graph_search_use_quant || config_.graph_search_full_quant ||
        num_ == 0 || graph_offsets_.empty() || graph_indices_.empty() ||
        rotated_base_.empty()) {
        return;
    }

    size_t batch_bytes =
        round_up_to_multiple(QGBatchDataMap<float>::data_bytes(padded_dim_), sizeof(uint64_t));
    graph_edge_batch_offsets_.assign(num_ + 1, 0);
    for (size_t i = 0; i < num_; ++i) {
        graph_edge_batch_offsets_[i + 1] =
            graph_edge_batch_offsets_[i] +
            div_round_up(graph_offsets_[i + 1] - graph_offsets_[i], scan::kBatchSize);
    }
    size_t total_batches = graph_edge_batch_offsets_[num_];

    graph_edge_batch_codes_.count = graph_indices_.size();
    graph_edge_batch_codes_.batch_count = total_batches;
    graph_edge_batch_codes_.batch_bytes = batch_bytes;
    graph_edge_batch_codes_.storage.assign(
        BatchCodeBank::words_for(total_batches * batch_bytes), 0
    );
    if (total_batches == 0) {
        return;
    }

    std::vector<float> batch_data(scan::kBatchSize * padded_dim_, 0.0F);
    size_t batch_id = 0;
    for (size_t node = 0; node < num_; ++node) {
        size_t begin = graph_offsets_[node];
        size_t end = graph_offsets_[node + 1];
        const float* center = rotated_base_.data() + (node * padded_dim_);
        for (size_t offset = begin; offset < end; offset += scan::kBatchSize) {
            std::fill(batch_data.begin(), batch_data.end(), 0.0F);
            size_t count = std::min(scan::kBatchSize, end - offset);
            for (size_t j = 0; j < count; ++j) {
                PID id = graph_indices_[offset + j];
                std::copy(
                    rotated_base_.data() + (static_cast<size_t>(id) * padded_dim_),
                    rotated_base_.data() + ((static_cast<size_t>(id) + 1) * padded_dim_),
                    batch_data.data() + (j * padded_dim_)
                );
            }
            quant::quantize_qg_batch(
                batch_data.data(),
                center,
                count,
                padded_dim_,
                graph_edge_batch_codes_.batch(batch_id),
                METRIC_L2
            );
            ++batch_id;
        }
    }
}

inline void ATMMGGraphIndex::build_graph_insertion() {
    graph_.assign(num_, {});
    if (num_ <= 1) {
        return;
    }

    size_t centers_count = num_centers();
    constexpr PID kInvalidPid = std::numeric_limits<PID>::max();
    size_t new_degree_limit =
        config_.graph_insert_new_degree == 0
            ? std::max<size_t>(1, config_.graph_degree / 2)
            : config_.graph_insert_new_degree;
    size_t old_degree_limit = config_.graph_degree;
    size_t ef_construction =
        std::max<size_t>(new_degree_limit, config_.graph_build_intra_candidates);
    size_t hint_keep =
        std::max<size_t>(old_degree_limit, config_.graph_build_cross_candidates);

    auto append_unique = [](std::vector<PID>& dst, PID id) {
        if (std::find(dst.begin(), dst.end(), id) == dst.end()) {
            dst.push_back(id);
        }
    };

    auto prune_candidate_pool = [&](PID query_id, std::vector<PID>& candidate_ids, size_t keep) {
        std::sort(candidate_ids.begin(), candidate_ids.end());
        candidate_ids.erase(
            std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end()
        );
        candidate_ids.erase(
            std::remove(candidate_ids.begin(), candidate_ids.end(), query_id),
            candidate_ids.end()
        );
        if (candidate_ids.size() <= keep) {
            return;
        }
        std::vector<ScoredPid> scored;
        scored.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            scored.push_back({point_point_l2(query_id, cand), cand});
        }
        keep_smallest(scored, keep);
        candidate_ids.clear();
        candidate_ids.reserve(scored.size());
        for (const auto& cand : scored) {
            candidate_ids.push_back(cand.id);
        }
    };

    auto heuristic_select = [&](PID query_id, std::vector<PID> candidate_ids, size_t max_neighbors) {
        if (max_neighbors == 0) {
            return std::vector<PID>();
        }
        prune_candidate_pool(query_id, candidate_ids, std::max(max_neighbors, candidate_ids.size()));

        std::vector<ScoredPid> items;
        items.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            items.push_back({point_point_l2(query_id, cand), cand});
        }
        std::sort(items.begin(), items.end());

        std::vector<PID> selected;
        selected.reserve(std::min(max_neighbors, items.size()));
        for (const auto& cand : items) {
            bool ok = true;
            for (PID sid : selected) {
                if (point_point_l2(sid, cand.id) < cand.distance) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }

        if (selected.size() < std::min(max_neighbors, items.size())) {
            for (const auto& cand : items) {
                if (std::find(selected.begin(), selected.end(), cand.id) != selected.end()) {
                    continue;
                }
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }
        return selected;
    };

    auto prune_neighbors = [&](PID id) {
        auto& row = graph_[id];
        if (row.size() <= old_degree_limit) {
            return;
        }
        row = heuristic_select(id, row, old_degree_limit);
    };

    auto mutual_connect = [&](PID id, const std::vector<PID>& selected_in) {
        std::vector<PID> selected = heuristic_select(id, selected_in, new_degree_limit);
        graph_[id] = selected;
        for (PID nb : selected) {
            auto& row = graph_[nb];
            if (std::find(row.begin(), row.end(), id) == row.end()) {
                row.push_back(id);
            }
            prune_neighbors(nb);
        }
    };

    std::vector<uint32_t> build_marks(num_, 0);
    uint32_t build_epoch = 0;
    auto next_build_epoch = [&]() {
        ++build_epoch;
        if (build_epoch == 0) {
            std::fill(build_marks.begin(), build_marks.end(), 0);
            build_epoch = 1;
        }
        return build_epoch;
    };

    auto search_insert_candidates = [&](PID query_id,
                                        const std::vector<PID>& entry_points,
                                        PID insert_limit,
                                        size_t ef) {
        std::vector<PID> empty;
        if (insert_limit == 0) {
            return empty;
        }
        ef = std::max<size_t>(1, std::min(ef, static_cast<size_t>(insert_limit)));
        uint32_t epoch = next_build_epoch();
        std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
            candidate_heap;
        std::priority_queue<ScoredPid> best_heap;

        auto push_entry = [&](PID ep) {
            if (ep >= insert_limit || build_marks[ep] == epoch) {
                return;
            }
            build_marks[ep] = epoch;
            float d = point_point_l2(query_id, ep);
            candidate_heap.push({d, ep});
            best_heap.push({d, ep});
        };

        for (PID ep : entry_points) {
            push_entry(ep);
        }
        if (candidate_heap.empty()) {
            push_entry(0);
        }

        while (!candidate_heap.empty()) {
            ScoredPid current = candidate_heap.top();
            candidate_heap.pop();
            if (best_heap.size() >= ef && current.distance > best_heap.top().distance) {
                break;
            }

            for (PID nb : graph_[current.id]) {
                if (nb >= insert_limit || build_marks[nb] == epoch) {
                    continue;
                }
                build_marks[nb] = epoch;
                float d = point_point_l2(query_id, nb);
                if (best_heap.size() < ef || d < best_heap.top().distance) {
                    candidate_heap.push({d, nb});
                    best_heap.push({d, nb});
                    if (best_heap.size() > ef) {
                        best_heap.pop();
                    }
                }
            }
        }

        std::vector<ScoredPid> scored;
        scored.reserve(best_heap.size());
        while (!best_heap.empty()) {
            scored.push_back(best_heap.top());
            best_heap.pop();
        }
        std::sort(scored.begin(), scored.end());

        std::vector<PID> result;
        result.reserve(scored.size());
        for (const auto& cand : scored) {
            result.push_back(cand.id);
        }
        return result;
    };

    std::vector<PID> center_entry(centers_count, kInvalidPid);
    std::vector<float> center_entry_d2(
        centers_count, std::numeric_limits<float>::max()
    );
    PID global_entry = kInvalidPid;

    for (PID id = 0; id < num_; ++id) {
        PID center_id = point_center_[id];
        if (id == 0) {
            center_entry[center_id] = id;
            center_entry_d2[center_id] = l2_to_center(base_.data(), center_id);
            global_entry = id;
            continue;
        }

        std::vector<PID> entry_points;
        if (center_entry[center_id] != kInvalidPid && center_entry[center_id] < id) {
            append_unique(entry_points, center_entry[center_id]);
        }
        size_t neighbor_entries =
            std::min(config_.graph_build_center_neighbors, center_neighbors_[center_id].size());
        for (size_t i = 0; i < neighbor_entries; ++i) {
            PID nc = center_neighbors_[center_id][i];
            if (center_entry[nc] != kInvalidPid && center_entry[nc] < id) {
                append_unique(entry_points, center_entry[nc]);
            }
        }
        if (global_entry != kInvalidPid && global_entry < id) {
            append_unique(entry_points, global_entry);
        }
        if (entry_points.empty()) {
            entry_points.push_back(0);
        }

        std::vector<PID> candidates =
            search_insert_candidates(id, entry_points, id, ef_construction);

        std::vector<PID> hint_centers;
        hint_centers.reserve(neighbor_entries + 1);
        hint_centers.push_back(center_id);
        for (size_t i = 0; i < neighbor_entries; ++i) {
            hint_centers.push_back(center_neighbors_[center_id][i]);
        }
        size_t per_center_hint =
            std::max<size_t>(1, div_round_up(hint_keep, hint_centers.size()));
        size_t scan_prefix = per_center_hint * 8;
        for (PID hc : hint_centers) {
            const auto& topn = center_topn_[hc];
            size_t scan = std::min(topn.size(), scan_prefix);
            size_t added = 0;
            for (size_t j = 0; j < scan && added < per_center_hint; ++j) {
                PID other = topn[j];
                if (other < id && other != id) {
                    append_unique(candidates, other);
                    ++added;
                }
            }
        }
        prune_candidate_pool(id, candidates, std::max(ef_construction, hint_keep));
        if (candidates.empty() && global_entry != kInvalidPid) {
            candidates.push_back(global_entry);
        }

        std::vector<PID> selected = heuristic_select(id, candidates, new_degree_limit);
        if (selected.empty() && !candidates.empty()) {
            selected.push_back(candidates.front());
        }
        mutual_connect(id, selected);

        float center_d2 = l2_to_center(base_.data() + (static_cast<size_t>(id) * dim_), center_id);
        if (center_d2 < center_entry_d2[center_id]) {
            center_entry_d2[center_id] = center_d2;
            center_entry[center_id] = id;
        }
        global_entry = id;
    }

    if (config_.graph_build_bridge_edges) {
        add_bridge_edges();
    }
    if (config_.graph_query_adjacency_order) {
        order_graph_for_query();
    }
    if (config_.graph_reorder_by_center) {
        reorder_graph_by_center();
        quantize_topn_codes();
        if (config_.graph_search_use_quant) {
            quantize_point_codes();
        }
    }
    finalize_graph_csr();
}

inline void ATMMGGraphIndex::build_graph_nsg() {
    graph_.assign(num_, {});
    if (num_ <= 1) {
        return;
    }

    const size_t centers_count = num_centers();
    const size_t seed_degree = std::max<size_t>(1, config_.graph_degree / 2);
    const size_t row_degree = config_.graph_degree;
    const size_t intra_scan =
        std::max(config_.graph_build_intra_candidates, config_.graph_degree * 4);
    const size_t cross_total =
        std::max(config_.graph_build_cross_candidates, config_.graph_degree * 2);

    std::vector<std::vector<PID>> center_members(centers_count);
    for (PID id = 0; id < static_cast<PID>(num_); ++id) {
        center_members[point_center_[id]].push_back(id);
    }
    for (PID c = 0; c < static_cast<PID>(centers_count); ++c) {
        auto& members = center_members[c];
        std::vector<ScoredPid> scored;
        scored.reserve(members.size());
        for (PID id : members) {
            scored.push_back(
                {l2_to_center(base_.data() + (static_cast<size_t>(id) * dim_), c), id}
            );
        }
        std::sort(scored.begin(), scored.end());
        for (size_t i = 0; i < scored.size(); ++i) {
            members[i] = scored[i].id;
        }
    }

    std::vector<uint32_t> candidate_marks(num_, 0);
    uint32_t candidate_epoch = 0;
    auto next_candidate_epoch = [&]() {
        ++candidate_epoch;
        if (candidate_epoch == 0) {
            std::fill(candidate_marks.begin(), candidate_marks.end(), 0);
            candidate_epoch = 1;
        }
        return candidate_epoch;
    };

    auto append_candidate = [&](std::vector<PID>& dst,
                                PID id,
                                PID query_id,
                                uint32_t epoch) {
        if (id >= num_ || id == query_id || candidate_marks[id] == epoch) {
            return;
        }
        candidate_marks[id] = epoch;
        dst.push_back(id);
    };

    auto prune_candidate_pool = [&](PID query_id, std::vector<PID>& candidate_ids, size_t keep) {
        if (candidate_ids.size() <= keep) {
            return;
        }
        std::vector<ScoredPid> scored;
        scored.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            scored.push_back({point_point_l2(query_id, cand), cand});
        }
        keep_smallest(scored, keep);
        candidate_ids.clear();
        candidate_ids.reserve(scored.size());
        for (const auto& cand : scored) {
            candidate_ids.push_back(cand.id);
        }
    };

    auto heuristic_select = [&](PID query_id, std::vector<PID> candidate_ids, size_t max_neighbors) {
        if (max_neighbors == 0 || candidate_ids.empty()) {
            return std::vector<PID>();
        }
        size_t candidate_keep = std::max(max_neighbors * 8, max_neighbors);
        prune_candidate_pool(query_id, candidate_ids, candidate_keep);

        std::vector<ScoredPid> items;
        items.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            items.push_back({point_point_l2(query_id, cand), cand});
        }
        std::sort(items.begin(), items.end());

        std::vector<PID> selected;
        selected.reserve(std::min(max_neighbors, items.size()));
        for (const auto& cand : items) {
            bool ok = true;
            for (PID sid : selected) {
                if (point_point_l2(sid, cand.id) < cand.distance) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }

        for (const auto& cand : items) {
            if (selected.size() >= max_neighbors) {
                break;
            }
            if (std::find(selected.begin(), selected.end(), cand.id) ==
                selected.end()) {
                selected.push_back(cand.id);
            }
        }
        return selected;
    };

    auto append_mutual = [&](PID a, PID b) {
        append_graph_edge(a, b);
        append_graph_edge(b, a);
    };

    for (PID id = 0; id < static_cast<PID>(num_); ++id) {
        uint32_t epoch = next_candidate_epoch();
        candidate_marks[id] = epoch;

        std::vector<PID> candidates;
        candidates.reserve(intra_scan + cross_total + config_.graph_portal_pool_size + 8);

        PID center_id = point_center_[id];
        const auto& local_topn = center_members[center_id];
        size_t local_scan = local_topn.size();
        for (size_t i = 0; i < local_scan; ++i) {
            append_candidate(candidates, local_topn[i], id, epoch);
        }

        size_t neighbor_centers =
            std::min(config_.graph_build_center_neighbors, center_neighbors_[center_id].size());
        size_t per_neighbor =
            neighbor_centers == 0 ? 0 : std::max<size_t>(1, div_round_up(cross_total, neighbor_centers));
        for (size_t ci = 0; ci < neighbor_centers; ++ci) {
            PID nc = center_neighbors_[center_id][ci];
            const auto& topn = center_members[nc];
            size_t scan = std::min(per_neighbor, topn.size());
            for (size_t j = 0; j < scan; ++j) {
                append_candidate(candidates, topn[j], id, epoch);
            }
        }

        if (!center_portal_pool_.empty()) {
            for (PID portal : center_portal_pool_[center_id]) {
                append_candidate(candidates, portal, id, epoch);
            }
            for (size_t ci = 0; ci < neighbor_centers; ++ci) {
                PID nc = center_neighbors_[center_id][ci];
                for (PID portal : center_portal_pool_[nc]) {
                    append_candidate(candidates, portal, id, epoch);
                }
            }
        }

        if (candidates.empty()) {
            const auto& fallback_pool = center_topn_[center_id];
            for (PID cand : fallback_pool) {
                append_candidate(candidates, cand, id, epoch);
                if (!candidates.empty()) {
                    break;
                }
            }
        }

        std::vector<PID> selected = heuristic_select(id, candidates, seed_degree);
        for (PID nb : selected) {
            append_mutual(id, nb);
        }
    }

    for (auto& row : graph_) {
        std::sort(row.begin(), row.end());
        row.erase(std::unique(row.begin(), row.end()), row.end());
    }

    for (PID id = 0; id < static_cast<PID>(num_); ++id) {
        auto& row = graph_[id];
        if (row.size() > row_degree) {
            row = heuristic_select(id, row, row_degree);
        }
        if (row.empty() && num_ > 1) {
            PID center_id = point_center_[id];
            for (PID cand : center_topn_[center_id]) {
                if (cand != id) {
                    append_graph_edge(id, cand);
                    break;
                }
            }
            if (row.empty()) {
                append_graph_edge(id, id == 0 ? 1 : 0);
            }
        }
    }

    if (config_.graph_build_bridge_edges) {
        add_bridge_edges();
    }
    if (config_.graph_query_adjacency_order) {
        order_graph_for_query();
    }
    if (config_.graph_reorder_by_center) {
        reorder_graph_by_center();
        quantize_topn_codes();
        if (config_.graph_search_use_quant) {
            quantize_point_codes();
        }
    }
    finalize_graph_csr();
}

inline void ATMMGGraphIndex::build_graph_vamana() {
    graph_.assign(num_, {});
    if (num_ <= 1) {
        return;
    }

    const size_t centers_count = num_centers();
    constexpr PID kInvalidPid = std::numeric_limits<PID>::max();
    const size_t row_degree = config_.graph_degree;
    const size_t candidate_limit =
        config_.graph_vamana_candidate_limit == 0
            ? std::max(row_degree * 8, row_degree)
            : config_.graph_vamana_candidate_limit;
    const float alpha2 = config_.graph_vamana_alpha * config_.graph_vamana_alpha;
    const size_t ef_construction =
        std::max<size_t>(row_degree, config_.graph_build_intra_candidates);
    const size_t hint_keep =
        std::max<size_t>(row_degree, config_.graph_build_cross_candidates);

    auto append_unique = [](std::vector<PID>& dst, PID id) {
        if (std::find(dst.begin(), dst.end(), id) == dst.end()) {
            dst.push_back(id);
        }
    };

    auto prune_candidate_pool = [&](PID query_id, std::vector<PID>& candidate_ids, size_t keep) {
        std::sort(candidate_ids.begin(), candidate_ids.end());
        candidate_ids.erase(
            std::unique(candidate_ids.begin(), candidate_ids.end()), candidate_ids.end()
        );
        candidate_ids.erase(
            std::remove(candidate_ids.begin(), candidate_ids.end(), query_id),
            candidate_ids.end()
        );
        if (candidate_ids.size() <= keep) {
            return;
        }
        std::vector<ScoredPid> scored;
        scored.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            scored.push_back({point_point_l2(query_id, cand), cand});
        }
        keep_smallest(scored, keep);
        candidate_ids.clear();
        candidate_ids.reserve(scored.size());
        for (const auto& cand : scored) {
            candidate_ids.push_back(cand.id);
        }
    };

    auto robust_prune_select = [&](PID query_id,
                                   std::vector<PID> candidate_ids,
                                   size_t max_neighbors) {
        if (max_neighbors == 0 || candidate_ids.empty()) {
            return std::vector<PID>();
        }
        prune_candidate_pool(query_id, candidate_ids, candidate_limit);

        std::vector<ScoredPid> items;
        items.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            items.push_back({point_point_l2(query_id, cand), cand});
        }
        std::sort(items.begin(), items.end());

        std::vector<uint8_t> removed(items.size(), 0);
        std::vector<PID> selected;
        selected.reserve(std::min(max_neighbors, items.size()));
        for (size_t i = 0; i < items.size() && selected.size() < max_neighbors; ++i) {
            if (removed[i] != 0) {
                continue;
            }
            PID chosen = items[i].id;
            selected.push_back(chosen);
            for (size_t j = i + 1; j < items.size(); ++j) {
                if (removed[j] != 0) {
                    continue;
                }
                float inter_dist = point_point_l2(chosen, items[j].id);
                if (inter_dist * alpha2 <= items[j].distance) {
                    removed[j] = 1;
                }
            }
        }

        for (const auto& cand : items) {
            if (selected.size() >= max_neighbors) {
                break;
            }
            if (std::find(selected.begin(), selected.end(), cand.id) == selected.end()) {
                selected.push_back(cand.id);
            }
        }
        return selected;
    };

    auto prune_neighbors = [&](PID id) {
        auto& row = graph_[id];
        if (row.size() <= row_degree) {
            return;
        }
        row = robust_prune_select(id, row, row_degree);
    };

    auto mutual_connect = [&](PID id, const std::vector<PID>& selected_in) {
        std::vector<PID> selected = robust_prune_select(id, selected_in, row_degree);
        graph_[id] = selected;
        for (PID nb : selected) {
            auto& row = graph_[nb];
            if (std::find(row.begin(), row.end(), id) == row.end()) {
                row.push_back(id);
            }
            prune_neighbors(nb);
        }
    };

    std::vector<uint32_t> build_marks(num_, 0);
    uint32_t build_epoch = 0;
    auto next_build_epoch = [&]() {
        ++build_epoch;
        if (build_epoch == 0) {
            std::fill(build_marks.begin(), build_marks.end(), 0);
            build_epoch = 1;
        }
        return build_epoch;
    };

    auto search_insert_candidates = [&](PID query_id,
                                        const std::vector<PID>& entry_points,
                                        PID insert_limit,
                                        size_t ef) {
        std::vector<PID> empty;
        if (insert_limit == 0) {
            return empty;
        }
        ef = std::max<size_t>(1, std::min(ef, static_cast<size_t>(insert_limit)));
        uint32_t epoch = next_build_epoch();
        std::priority_queue<ScoredPid, std::vector<ScoredPid>, std::greater<ScoredPid>>
            candidate_heap;
        std::priority_queue<ScoredPid> best_heap;

        auto push_entry = [&](PID ep) {
            if (ep >= insert_limit || build_marks[ep] == epoch) {
                return;
            }
            build_marks[ep] = epoch;
            float d = point_point_l2(query_id, ep);
            candidate_heap.push({d, ep});
            best_heap.push({d, ep});
        };

        for (PID ep : entry_points) {
            push_entry(ep);
        }
        if (candidate_heap.empty()) {
            push_entry(0);
        }

        while (!candidate_heap.empty()) {
            ScoredPid current = candidate_heap.top();
            candidate_heap.pop();
            if (best_heap.size() >= ef && current.distance > best_heap.top().distance) {
                break;
            }

            for (PID nb : graph_[current.id]) {
                if (nb >= insert_limit || build_marks[nb] == epoch) {
                    continue;
                }
                build_marks[nb] = epoch;
                float d = point_point_l2(query_id, nb);
                if (best_heap.size() < ef || d < best_heap.top().distance) {
                    candidate_heap.push({d, nb});
                    best_heap.push({d, nb});
                    if (best_heap.size() > ef) {
                        best_heap.pop();
                    }
                }
            }
        }

        std::vector<ScoredPid> scored;
        scored.reserve(best_heap.size());
        while (!best_heap.empty()) {
            scored.push_back(best_heap.top());
            best_heap.pop();
        }
        std::sort(scored.begin(), scored.end());

        std::vector<PID> result;
        result.reserve(scored.size());
        for (const auto& cand : scored) {
            result.push_back(cand.id);
        }
        return result;
    };

    std::vector<PID> center_entry(centers_count, kInvalidPid);
    std::vector<float> center_entry_d2(
        centers_count, std::numeric_limits<float>::max()
    );
    PID global_entry = kInvalidPid;

    for (PID id = 0; id < num_; ++id) {
        PID center_id = point_center_[id];
        if (id == 0) {
            center_entry[center_id] = id;
            center_entry_d2[center_id] = l2_to_center(base_.data(), center_id);
            global_entry = id;
            continue;
        }

        std::vector<PID> entry_points;
        if (center_entry[center_id] != kInvalidPid && center_entry[center_id] < id) {
            append_unique(entry_points, center_entry[center_id]);
        }
        size_t neighbor_entries =
            std::min(config_.graph_build_center_neighbors, center_neighbors_[center_id].size());
        for (size_t i = 0; i < neighbor_entries; ++i) {
            PID nc = center_neighbors_[center_id][i];
            if (center_entry[nc] != kInvalidPid && center_entry[nc] < id) {
                append_unique(entry_points, center_entry[nc]);
            }
        }
        if (global_entry != kInvalidPid && global_entry < id) {
            append_unique(entry_points, global_entry);
        }
        if (entry_points.empty()) {
            entry_points.push_back(0);
        }

        std::vector<PID> candidates =
            search_insert_candidates(id, entry_points, id, ef_construction);

        std::vector<PID> hint_centers;
        hint_centers.reserve(neighbor_entries + 1);
        hint_centers.push_back(center_id);
        for (size_t i = 0; i < neighbor_entries; ++i) {
            hint_centers.push_back(center_neighbors_[center_id][i]);
        }
        size_t per_center_hint =
            std::max<size_t>(1, div_round_up(hint_keep, hint_centers.size()));
        size_t scan_prefix = per_center_hint * 8;
        for (PID hc : hint_centers) {
            const auto& topn = center_topn_[hc];
            size_t scan = std::min(topn.size(), scan_prefix);
            size_t added = 0;
            for (size_t j = 0; j < scan && added < per_center_hint; ++j) {
                PID other = topn[j];
                if (other < id && other != id) {
                    append_unique(candidates, other);
                    ++added;
                }
            }
        }
        prune_candidate_pool(
            id, candidates, std::max(candidate_limit, std::max(ef_construction, hint_keep))
        );
        if (candidates.empty() && global_entry != kInvalidPid) {
            candidates.push_back(global_entry);
        }

        std::vector<PID> selected = robust_prune_select(id, candidates, row_degree);
        if (selected.empty() && !candidates.empty()) {
            selected.push_back(candidates.front());
        }
        mutual_connect(id, selected);

        float center_d2 = l2_to_center(base_.data() + (static_cast<size_t>(id) * dim_), center_id);
        if (center_d2 < center_entry_d2[center_id]) {
            center_entry_d2[center_id] = center_d2;
            center_entry[center_id] = id;
        }
        global_entry = id;
    }

    if (config_.graph_build_bridge_edges) {
        add_bridge_edges();
    }
    if (config_.graph_query_adjacency_order) {
        order_graph_for_query();
    }
    if (config_.graph_reorder_by_center) {
        reorder_graph_by_center();
        quantize_topn_codes();
        if (config_.graph_search_use_quant) {
            quantize_point_codes();
        }
    }
    finalize_graph_csr();
}

inline void ATMMGGraphIndex::build_graph() {
    if (config_.graph_build_mode == 2) {
        build_graph_vamana();
    } else if (config_.graph_build_mode == 1) {
        build_graph_nsg();
    } else {
        build_graph_insertion();
    }
}

inline void ATMMGGraphIndex::apply_graph_post_nnd_refine() {
    graph_post_nnd_edges_before_ = 0;
    graph_post_nnd_edges_after_ = 0;
    graph_post_nnd_candidate_total_ = 0;
    graph_post_nnd_candidate_sources_ = 0;

    if (!config_.graph_post_nnd_refine || num_ <= 1 ||
        graph_offsets_.size() != num_ + 1 || graph_indices_.empty()) {
        return;
    }

    graph_post_nnd_edges_before_ = graph_indices_.size();
    const size_t row_degree = config_.graph_degree;
    const size_t iterations =
        std::max<size_t>(1, config_.graph_post_nnd_iterations);
    const size_t candidate_limit =
        config_.graph_post_nnd_candidate_limit == 0
            ? std::max(row_degree * 8, row_degree)
            : config_.graph_post_nnd_candidate_limit;
    const float alpha2 =
        config_.graph_post_nnd_alpha * config_.graph_post_nnd_alpha;

    std::vector<std::vector<PID>> current_graph(num_);
    std::vector<size_t> row_limits(num_, row_degree);
    for (size_t i = 0; i < num_; ++i) {
        const size_t begin = graph_offsets_[i];
        const size_t end = graph_offsets_[i + 1];
        current_graph[i].assign(
            graph_indices_.begin() + static_cast<std::ptrdiff_t>(begin),
            graph_indices_.begin() + static_cast<std::ptrdiff_t>(end)
        );
        if (config_.graph_post_nnd_preserve_degree) {
            row_limits[i] = current_graph[i].size();
        }
    }

    std::vector<uint32_t> candidate_marks(num_, 0);
    uint32_t candidate_epoch = 0;
    auto next_candidate_epoch = [&]() {
        ++candidate_epoch;
        if (candidate_epoch == 0) {
            std::fill(candidate_marks.begin(), candidate_marks.end(), 0);
            candidate_epoch = 1;
        }
        return candidate_epoch;
    };

    auto prune_candidate_pool = [&](PID query_id,
                                    std::vector<PID>& candidate_ids,
                                    size_t keep) {
        if (candidate_ids.size() <= keep) {
            return;
        }
        std::vector<ScoredPid> scored;
        scored.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            scored.push_back({point_point_l2(query_id, cand), cand});
        }
        keep_smallest(scored, keep);
        candidate_ids.clear();
        candidate_ids.reserve(scored.size());
        for (const auto& cand : scored) {
            candidate_ids.push_back(cand.id);
        }
    };

    auto robust_select = [&](PID query_id,
                             std::vector<PID> candidate_ids,
                             size_t max_neighbors) {
        if (max_neighbors == 0 || candidate_ids.empty()) {
            return std::vector<PID>();
        }
        prune_candidate_pool(
            query_id,
            candidate_ids,
            std::max(candidate_limit, max_neighbors)
        );

        std::vector<ScoredPid> items;
        items.reserve(candidate_ids.size());
        for (PID cand : candidate_ids) {
            if (cand < num_ && cand != query_id) {
                items.push_back({point_point_l2(query_id, cand), cand});
            }
        }
        std::sort(items.begin(), items.end());

        std::vector<PID> selected;
        selected.reserve(std::min(max_neighbors, items.size()));
        for (const auto& cand : items) {
            bool occluded = false;
            for (PID sid : selected) {
                if (point_point_l2(sid, cand.id) * alpha2 <= cand.distance) {
                    occluded = true;
                    break;
                }
            }
            if (!occluded) {
                selected.push_back(cand.id);
                if (selected.size() >= max_neighbors) {
                    break;
                }
            }
        }

        for (const auto& cand : items) {
            if (selected.size() >= max_neighbors) {
                break;
            }
            if (std::find(selected.begin(), selected.end(), cand.id) ==
                selected.end()) {
                selected.push_back(cand.id);
            }
        }
        return selected;
    };

    auto collect_neighbor_candidates =
        [&](PID query_id, const std::vector<std::vector<PID>>& graph) {
            const uint32_t epoch = next_candidate_epoch();
            candidate_marks[query_id] = epoch;

            std::vector<PID> candidates;
            candidates.reserve(std::max<size_t>(row_degree, candidate_limit));
            auto add_candidate = [&](PID id) {
                if (id >= num_ || candidate_marks[id] == epoch) {
                    return;
                }
                candidate_marks[id] = epoch;
                candidates.push_back(id);
            };

            for (PID nb : graph[query_id]) {
                add_candidate(nb);
                if (nb >= num_) {
                    continue;
                }
                for (PID nn : graph[nb]) {
                    add_candidate(nn);
                }
            }

            prune_candidate_pool(
                query_id,
                candidates,
                std::max(candidate_limit, row_limits[query_id])
            );
            return candidates;
        };

    for (size_t iter = 0; iter < iterations; ++iter) {
        std::vector<std::vector<PID>> next_graph(num_);
        for (PID id = 0; id < static_cast<PID>(num_); ++id) {
            std::vector<PID> candidates =
                collect_neighbor_candidates(id, current_graph);
            graph_post_nnd_candidate_total_ += candidates.size();
            if (!candidates.empty()) {
                ++graph_post_nnd_candidate_sources_;
            }
            const size_t target_degree = config_.graph_post_nnd_preserve_degree
                                             ? row_limits[id]
                                             : row_degree;
            next_graph[id] =
                robust_select(id, std::move(candidates), target_degree);
            if (target_degree > 0 && next_graph[id].empty() &&
                !current_graph[id].empty()) {
                next_graph[id].push_back(current_graph[id].front());
            }
        }
        current_graph.swap(next_graph);
    }

    graph_.swap(current_graph);
    if (config_.graph_query_adjacency_order) {
        order_graph_for_query();
    }
    finalize_graph_csr();
    graph_post_nnd_edges_after_ = graph_indices_.size();
}

inline bool ATMMGGraphIndex::append_graph_edge(PID a, PID b) {
    if (a >= num_ || b >= num_ || a == b) {
        return false;
    }
    auto& row = graph_[a];
    if (std::find(row.begin(), row.end(), b) != row.end()) {
        return false;
    }
    row.push_back(b);
    return true;
}

inline void ATMMGGraphIndex::add_bridge_edges() {
    graph_bridge_edges_added_ = 0;
    size_t centers_count = num_centers();
    if (centers_count <= 1 || graph_.empty()) {
        return;
    }

    size_t center_keep = std::min(
        config_.graph_bridge_center_neighbors,
        centers_count > 0 ? centers_count - 1 : 0
    );
    size_t points_keep = config_.graph_bridge_points_per_center;
    size_t scan_keep = config_.graph_bridge_candidate_scan;

    auto connect = [&](PID a, PID b) {
        if (append_graph_edge(a, b)) {
            ++graph_bridge_edges_added_;
        }
        if (append_graph_edge(b, a)) {
            ++graph_bridge_edges_added_;
        }
    };

    for (PID c = 0; c < centers_count; ++c) {
        const auto& left = center_topn_[c];
        if (left.empty()) {
            continue;
        }
        size_t neighbor_count = std::min(center_keep, center_neighbors_[c].size());
        size_t left_scan = std::min(scan_keep, left.size());
        size_t left_take = std::min(points_keep, left_scan);
        for (size_t ni = 0; ni < neighbor_count; ++ni) {
            PID nc = center_neighbors_[c][ni];
            if (nc >= centers_count || nc <= c) {
                continue;
            }
            const auto& right = center_topn_[nc];
            if (right.empty()) {
                continue;
            }
            size_t right_scan = std::min(scan_keep, right.size());
            size_t right_take =
                std::min(right_scan, std::max(points_keep, points_keep * 4));

            if (!center_portal_pool_.empty()) {
                const auto& left_portals = center_portal_pool_[c];
                const auto& right_portals = center_portal_pool_[nc];
                size_t portal_take = std::min(
                    points_keep,
                    std::max(left_portals.empty() ? size_t{0} : size_t{1},
                             points_keep)
                );
                for (PID a : left_portals) {
                    std::vector<ScoredPid> scored;
                    scored.reserve(right_portals.size());
                    for (PID b : right_portals) {
                        if (a != b) {
                            scored.push_back({point_point_l2(a, b), b});
                        }
                    }
                    keep_smallest(scored, std::min(portal_take, scored.size()));
                    for (const auto& cand : scored) {
                        connect(a, cand.id);
                    }
                }
                for (PID b : right_portals) {
                    std::vector<ScoredPid> scored;
                    scored.reserve(left_portals.size());
                    for (PID a : left_portals) {
                        if (a != b) {
                            scored.push_back({point_point_l2(b, a), a});
                        }
                    }
                    keep_smallest(scored, std::min(portal_take, scored.size()));
                    for (const auto& cand : scored) {
                        connect(b, cand.id);
                    }
                }
            }

            std::vector<ScoredPid> left_boundary;
            left_boundary.reserve(left_scan);
            const float* right_center =
                centers_.data() + (static_cast<size_t>(nc) * dim_);
            for (size_t li = 0; li < left_scan; ++li) {
                PID id = left[li];
                left_boundary.push_back(
                    {euclidean_sqr_fast(
                         right_center, base_.data() + (static_cast<size_t>(id) * dim_), dim_
                     ),
                     id}
                );
            }
            keep_smallest(left_boundary, left_take);

            std::vector<ScoredPid> right_boundary;
            right_boundary.reserve(right_scan);
            const float* left_center = centers_.data() + (static_cast<size_t>(c) * dim_);
            for (size_t ri = 0; ri < right_scan; ++ri) {
                PID id = right[ri];
                right_boundary.push_back(
                    {euclidean_sqr_fast(
                         left_center, base_.data() + (static_cast<size_t>(id) * dim_), dim_
                     ),
                     id}
                );
            }
            keep_smallest(right_boundary, right_take);

            for (const auto& left_item : left_boundary) {
                PID a = left_item.id;
                std::vector<ScoredPid> scored;
                scored.reserve(right_boundary.size());
                for (const auto& right_item : right_boundary) {
                    PID b = right_item.id;
                    if (a != b) {
                        scored.push_back({point_point_l2(a, b), b});
                    }
                }
                keep_smallest(scored, std::min(points_keep, scored.size()));
                for (const auto& cand : scored) {
                    connect(a, cand.id);
                }
            }
        }
    }
}

inline void ATMMGGraphIndex::order_graph_for_query() {
    if (graph_.empty()) {
        return;
    }
    size_t cap_hint = config_.graph_search_neighbor_cap == 0
                          ? config_.graph_degree
                          : config_.graph_search_neighbor_cap;
    if (config_.graph_hot_neighbor_count > 0) {
        cap_hint = std::min(cap_hint, config_.graph_hot_neighbor_count);
    }
    cap_hint = std::max<size_t>(1, cap_hint);

    for (PID id = 0; id < num_; ++id) {
        auto& row = graph_[id];
        if (row.empty()) {
            continue;
        }
        std::vector<PID> deduped;
        deduped.reserve(row.size());
        for (PID nb : row) {
            if (nb == id || nb >= num_) {
                continue;
            }
            if (std::find(deduped.begin(), deduped.end(), nb) == deduped.end()) {
                deduped.push_back(nb);
            }
        }

        if (config_.graph_query_front_prune) {
            std::vector<ScoredPid> items;
            std::vector<ScoredPid> local_items;
            std::vector<ScoredPid> cross_items;
            std::vector<ScoredPid> portal_items;
            items.reserve(deduped.size());
            local_items.reserve(deduped.size());
            cross_items.reserve(deduped.size());
            portal_items.reserve(deduped.size());
            PID center_id = point_center_[id];
            for (PID nb : deduped) {
                ScoredPid item{point_point_l2(id, nb), nb};
                items.push_back(item);
                if (point_center_[nb] == center_id) {
                    local_items.push_back(item);
                } else {
                    cross_items.push_back(item);
                }
                if (!point_is_portal_.empty() && point_is_portal_[nb] != 0) {
                    portal_items.push_back(item);
                }
            }
            std::sort(items.begin(), items.end());
            std::sort(local_items.begin(), local_items.end());
            std::sort(cross_items.begin(), cross_items.end());
            std::sort(portal_items.begin(), portal_items.end());

            size_t front_limit = std::min(cap_hint, items.size());
            std::vector<PID> front;
            front.reserve(front_limit);

            auto has_front = [&](PID candidate) {
                return std::find(front.begin(), front.end(), candidate) != front.end();
            };

            auto append_diverse = [&](const std::vector<ScoredPid>& source,
                                      size_t budget) {
                if (budget == 0 || front.size() >= front_limit) {
                    return;
                }
                size_t target = std::min(front_limit, front.size() + budget);
                for (const auto& cand : source) {
                    if (has_front(cand.id)) {
                        continue;
                    }
                    bool ok = true;
                    for (PID sid : front) {
                        if (point_point_l2(sid, cand.id) < cand.distance) {
                            ok = false;
                            break;
                        }
                    }
                    if (ok) {
                        front.push_back(cand.id);
                        if (front.size() >= target) {
                            break;
                        }
                    }
                }
                for (const auto& cand : source) {
                    if (front.size() >= target) {
                        break;
                    }
                    if (has_front(cand.id)) {
                        continue;
                    }
                    front.push_back(cand.id);
                }
            };

            size_t local_budget = std::min<size_t>(12, front_limit);
            size_t cross_budget =
                std::min<size_t>(6, front_limit - local_budget);
            size_t portal_budget =
                std::min<size_t>(2, front_limit - local_budget - cross_budget);
            append_diverse(local_items, local_budget);
            append_diverse(cross_items, cross_budget);
            append_diverse(portal_items, portal_budget);
            if (front.size() < front_limit) {
                append_diverse(items, front_limit - front.size());
            }

            std::vector<PID> ordered;
            ordered.reserve(items.size());
            ordered.insert(ordered.end(), front.begin(), front.end());
            for (const auto& cand : items) {
                if (!has_front(cand.id)) {
                    ordered.push_back(cand.id);
                }
            }
            row.swap(ordered);
            continue;
        }

        std::vector<ScoredPid> local;
        std::vector<ScoredPid> cross;
        local.reserve(deduped.size());
        cross.reserve(deduped.size());
        PID center_id = point_center_[id];
        for (PID nb : deduped) {
            ScoredPid item{point_point_l2(id, nb), nb};
            if (point_center_[nb] == center_id) {
                local.push_back(item);
            } else {
                cross.push_back(item);
            }
        }
        std::sort(local.begin(), local.end());
        std::sort(cross.begin(), cross.end());

        size_t cross_front_budget =
            std::min(cross.size(), std::max<size_t>(1, cap_hint / 4));
        std::vector<PID> ordered;
        ordered.reserve(local.size() + cross.size());
        size_t li = 0;
        size_t ci = 0;
        while (li < local.size() || ci < cross_front_budget) {
            for (size_t repeat = 0; repeat < 2 && li < local.size(); ++repeat) {
                ordered.push_back(local[li++].id);
            }
            if (ci < cross_front_budget) {
                ordered.push_back(cross[ci++].id);
            }
        }
        while (li < local.size()) {
            ordered.push_back(local[li++].id);
        }
        while (ci < cross.size()) {
            ordered.push_back(cross[ci++].id);
        }
        row.swap(ordered);
    }
}

inline void ATMMGGraphIndex::reorder_graph_by_center() {
    if (num_ == 0 || graph_.empty()) {
        return;
    }

    std::vector<PID> new_to_old(num_);
    std::iota(new_to_old.begin(), new_to_old.end(), static_cast<PID>(0));
    std::stable_sort(new_to_old.begin(), new_to_old.end(), [&](PID a, PID b) {
        PID ca = point_center_[a];
        PID cb = point_center_[b];
        if (ca != cb) {
            return ca < cb;
        }
        return a < b;
    });

    std::vector<PID> old_to_new(num_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        old_to_new[new_to_old[new_id]] = new_id;
    }

    auto remap_ids = [&](std::vector<PID>& ids) {
        for (PID& id : ids) {
            if (id < num_) {
                id = old_to_new[id];
            }
        }
    };

    std::vector<float> new_base(num_ * dim_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        PID old_id = new_to_old[new_id];
        std::copy_n(
            base_.data() + (static_cast<size_t>(old_id) * dim_),
            dim_,
            new_base.data() + (static_cast<size_t>(new_id) * dim_)
        );
    }
    base_.swap(new_base);

    if (!base_sq_norms_.empty()) {
        std::vector<float> new_norms(num_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            new_norms[new_id] = base_sq_norms_[new_to_old[new_id]];
        }
        base_sq_norms_.swap(new_norms);
    }

    if (!point_center_residual_norms_.empty()) {
        std::vector<float> new_residual_norms(num_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            new_residual_norms[new_id] =
                point_center_residual_norms_[new_to_old[new_id]];
        }
        point_center_residual_norms_.swap(new_residual_norms);
    }

    if (!rotated_base_.empty()) {
        std::vector<float> new_rotated(num_ * padded_dim_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            PID old_id = new_to_old[new_id];
            std::copy_n(
                rotated_base_.data() + (static_cast<size_t>(old_id) * padded_dim_),
                padded_dim_,
                new_rotated.data() + (static_cast<size_t>(new_id) * padded_dim_)
            );
        }
        rotated_base_.swap(new_rotated);
    }

    if (!external_ids_.empty()) {
        std::vector<PID> new_external_ids(num_);
        for (PID new_id = 0; new_id < num_; ++new_id) {
            new_external_ids[new_id] = external_ids_[new_to_old[new_id]];
        }
        external_ids_.swap(new_external_ids);
    }

    std::vector<PID> new_point_center(num_);
    for (PID new_id = 0; new_id < num_; ++new_id) {
        new_point_center[new_id] = point_center_[new_to_old[new_id]];
    }
    point_center_.swap(new_point_center);

    for (auto& ids : leaf_ids_) {
        remap_ids(ids);
    }
    for (auto& ids : center_topn_) {
        remap_ids(ids);
    }
    for (auto& ids : center_topn_) {
        remap_ids(ids);
    }
    for (auto& ids : center_portal_pool_) {
        remap_ids(ids);
    }
    rebuild_portal_marks();

    std::vector<std::vector<PID>> new_graph(num_);
    for (PID old_id = 0; old_id < num_; ++old_id) {
        PID new_id = old_to_new[old_id];
        auto& dst = new_graph[new_id];
        dst.reserve(graph_[old_id].size());
        for (PID nb : graph_[old_id]) {
            if (nb < num_) {
                dst.push_back(old_to_new[nb]);
            }
        }
    }
    graph_.swap(new_graph);
    graph_reordered_by_center_ = true;
}

inline void ATMMGGraphIndex::finalize_graph_csr() {
    graph_offsets_.assign(num_ + 1, 0);
    size_t total = 0;
    for (size_t i = 0; i < num_; ++i) {
        auto& row = graph_[i];
        std::vector<PID> deduped;
        deduped.reserve(row.size());
        for (PID id : row) {
            if (id == static_cast<PID>(i)) {
                continue;
            }
            if (std::find(deduped.begin(), deduped.end(), id) == deduped.end()) {
                deduped.push_back(id);
            }
        }
        row.swap(deduped);
        graph_offsets_[i] = total;
        total += row.size();
    }
    graph_offsets_[num_] = total;

    graph_indices_.clear();
    graph_indices_.reserve(total);
    for (const auto& row : graph_) {
        graph_indices_.insert(graph_indices_.end(), row.begin(), row.end());
    }

    graph_hot_offsets_.clear();
    graph_hot_indices_.clear();
    graph_cold_offsets_.clear();
    graph_cold_indices_.clear();
    if (config_.graph_hot_neighbor_count > 0) {
        graph_hot_offsets_.assign(num_ + 1, 0);
        graph_cold_offsets_.assign(num_ + 1, 0);
        size_t hot_total = 0;
        size_t cold_total = 0;
        for (size_t i = 0; i < num_; ++i) {
            const auto& row = graph_[i];
            size_t hot_count = std::min(config_.graph_hot_neighbor_count, row.size());
            graph_hot_offsets_[i] = hot_total;
            graph_cold_offsets_[i] = cold_total;
            hot_total += hot_count;
            cold_total += row.size() - hot_count;
        }
        graph_hot_offsets_[num_] = hot_total;
        graph_cold_offsets_[num_] = cold_total;

        graph_hot_indices_.reserve(hot_total);
        graph_cold_indices_.reserve(cold_total);
        for (const auto& row : graph_) {
            size_t hot_count = std::min(config_.graph_hot_neighbor_count, row.size());
            graph_hot_indices_.insert(
                graph_hot_indices_.end(), row.begin(),
                row.begin() + static_cast<std::ptrdiff_t>(hot_count)
            );
            graph_cold_indices_.insert(
                graph_cold_indices_.end(),
                row.begin() + static_cast<std::ptrdiff_t>(hot_count),
                row.end()
            );
        }
    }

    size_t fixed_need = config_.graph_search_neighbor_cap;
    fixed_need = std::max(fixed_need, config_.hard_query_neighbor_cap);
    fixed_need = std::max(fixed_need, config_.graph_late_neighbor_cap);
    fixed_need = std::max(fixed_need, config_.hard_query_late_neighbor_cap);
    rebuild_fixed_search_neighbors(fixed_need);
    rebuild_dual_scale_neighbors();
    quantize_graph_edge_batch_codes();

    std::vector<std::vector<PID>> empty_graph;
    graph_.swap(empty_graph);
}

inline void ATMMGGraphIndex::rebuild_fixed_search_neighbors(
    size_t min_neighbor_count
) {
    graph_fixed_indices_.clear();
    graph_fixed_counts_.clear();
    graph_fixed_neighbor_count_ = 0;
    if (num_ == 0 || graph_offsets_.empty() || graph_indices_.empty() ||
        config_.graph_hot_neighbor_count > 0 || min_neighbor_count == 0) {
        return;
    }
    size_t fixed_count = std::min(min_neighbor_count, config_.graph_degree);
    if (fixed_count == 0 || fixed_count > std::numeric_limits<uint16_t>::max()) {
        return;
    }
    graph_fixed_neighbor_count_ = fixed_count;
    graph_fixed_counts_.assign(num_, 0);
    graph_fixed_indices_.assign(num_ * fixed_count, PID{0});
    for (size_t i = 0; i < num_; ++i) {
        size_t begin = graph_offsets_[i];
        size_t end = graph_offsets_[i + 1];
        size_t count = std::min(fixed_count, end - begin);
        graph_fixed_counts_[i] = static_cast<uint16_t>(count);
        PID* dst = graph_fixed_indices_.data() + (i * fixed_count);
        std::copy_n(graph_indices_.data() + begin, count, dst);
    }
}

inline void ATMMGGraphIndex::rebuild_dual_scale_neighbors() {
    graph_dual_short_indices_.clear();
    graph_dual_short_counts_.clear();
    graph_dual_long_indices_.clear();
    graph_dual_long_counts_.clear();
    graph_dual_short_radius_.clear();
    graph_dual_short_neighbor_count_ = 0;
    graph_dual_long_neighbor_count_ = 0;

    if (!config_.graph_dual_scale_search || num_ == 0 || graph_offsets_.empty() ||
        graph_indices_.empty() || config_.graph_dual_short_count == 0 ||
        config_.graph_dual_long_count == 0) {
        return;
    }

    size_t short_count =
        std::min(config_.graph_dual_short_count, config_.graph_degree);
    size_t long_count =
        std::min(config_.graph_dual_long_count, config_.graph_degree);
    if (short_count == 0 || long_count == 0 ||
        short_count > std::numeric_limits<uint16_t>::max() ||
        long_count > std::numeric_limits<uint16_t>::max()) {
        return;
    }

    graph_dual_short_neighbor_count_ = short_count;
    graph_dual_long_neighbor_count_ = long_count;
    graph_dual_short_counts_.assign(num_, 0);
    graph_dual_long_counts_.assign(num_, 0);
    graph_dual_short_radius_.assign(num_, 0.0F);
    graph_dual_short_indices_.assign(num_ * short_count, PID{0});
    graph_dual_long_indices_.assign(num_ * long_count, PID{0});

    std::vector<ScoredPid> row_scores;
    for (size_t i = 0; i < num_; ++i) {
        size_t begin = graph_offsets_[i];
        size_t end = graph_offsets_[i + 1];
        size_t row_size = end - begin;
        if (row_size == 0) {
            continue;
        }

        row_scores.clear();
        row_scores.reserve(row_size);
        PID src = static_cast<PID>(i);
        for (size_t pos = begin; pos < end; ++pos) {
            PID nb = graph_indices_[pos];
            row_scores.push_back({point_point_l2(src, nb), nb});
        }
        std::sort(row_scores.begin(), row_scores.end());

        size_t actual_short = std::min(short_count, row_scores.size());
        graph_dual_short_counts_[i] = static_cast<uint16_t>(actual_short);
        PID* short_dst = graph_dual_short_indices_.data() + (i * short_count);
        float radius = 0.0F;
        for (size_t s = 0; s < actual_short; ++s) {
            short_dst[s] = row_scores[s].id;
            radius = std::max(radius, row_scores[s].distance);
        }
        graph_dual_short_radius_[i] = radius;

        if (row_scores.size() <= actual_short) {
            continue;
        }

        float long_threshold = radius * config_.graph_dual_long_alpha;
        PID* long_dst = graph_dual_long_indices_.data() + (i * long_count);
        size_t actual_long = 0;
        auto append_long = [&](PID nb) {
            if (actual_long >= long_count) {
                return;
            }
            for (size_t j = 0; j < actual_long; ++j) {
                if (long_dst[j] == nb) {
                    return;
                }
            }
            long_dst[actual_long++] = nb;
        };

        for (const auto& cand : row_scores) {
            if (actual_long >= long_count) {
                break;
            }
            if (cand.distance > long_threshold) {
                append_long(cand.id);
            }
        }
        for (auto it = row_scores.rbegin();
             it != row_scores.rend() && actual_long < long_count; ++it) {
            if (it->distance > radius) {
                append_long(it->id);
            }
        }
        graph_dual_long_counts_[i] = static_cast<uint16_t>(actual_long);
    }
}

