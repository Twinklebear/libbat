#pragma once

#include <memory>
#include "aggregation_tree.h"
#include "ba_tree.h"

class BATHandle {
    std::shared_ptr<BATree> ba_tree;
    std::shared_ptr<AggregationTree> pba_tree;

public:
    std::vector<AttributeDescription> attributes;

    BATHandle(const std::string &fname);

    BATHandle() = default;

    const Box &bounds() const;

    uint64_t num_points() const;

    uint32_t num_aggregators() const;

    size_t radix_tree_size();

    size_t radix_tree_size_bytes();

    size_t bitmap_dictionary_size();

    size_t bitmap_dictionary_size_bytes();

    size_t bitmap_indices_size();

    size_t bitmap_indices_size_bytes();

    size_t node_attrib_ranges_size();

    size_t node_attrib_ranges_size_bytes();

    size_t num_treelets();

    size_t aggregation_tree_size();

    size_t aggregation_tree_size_bytes();

    size_t total_tree_node_count();

    size_t total_tree_node_bytes();

    std::vector<float> aggregator_bitmap_bin_sizes(const std::string &attr);

    void set_enable_range_filtering(bool enable);

    void set_enable_bitmap_filtering(bool enable);

    template <typename Fn>
    QueryStats query_box_progressive(const Box &b,
                                     std::vector<AttributeQuery> *attrib_queries,
                                     float prev_quality,
                                     float current_quality,
                                     const Fn &callback);

    template <typename Fn>
    QueryStats query_box(const Box &b,
                         std::vector<AttributeQuery> *attrib_queries,
                         float quality,
                         const Fn &callback);

    // For debugging/testing: query the splitting planes of the tree
    void get_splitting_planes(std::vector<Plane> &planes,
                              const Box &query_box,
                              std::vector<AttributeQuery> *attrib_queries,
                              float quality);
};

template <typename Fn>
QueryStats BATHandle::query_box_progressive(const Box &b,
                                            std::vector<AttributeQuery> *attrib_queries,
                                            float prev_quality,
                                            float current_quality,
                                            const Fn &callback)
{
    if (ba_tree) {
        return ba_tree->query_box_progressive(
            b, attrib_queries, prev_quality, current_quality, callback);
    } else {
        return pba_tree->query_box_progressive(
            b, attrib_queries, prev_quality, current_quality, callback);
    }
}

template <typename Fn>
QueryStats BATHandle::query_box(const Box &b,
                                std::vector<AttributeQuery> *attrib_queries,
                                float quality,
                                const Fn &callback)
{
    if (ba_tree) {
        return ba_tree->query_box(b, attrib_queries, quality, callback);
    } else {
        return pba_tree->query_box(b, attrib_queries, quality, callback);
    }
}
