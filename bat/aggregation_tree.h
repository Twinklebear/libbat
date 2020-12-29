#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include <parallel_hashmap/phmap.h>
#include "abstract_array.h"
#include "aggregation_kd_node.h"
#include "attribute.h"
#include "ba_tree.h"
#include "box.h"
#include "owned_array.h"

struct WriteOptions {
    bool find_best_axis = false;
    float max_split_imbalance_ratio = 4.f;
    float max_overfull_aggregator_factor = 1.5f;
    bool build_local_trees = true;
    uint32_t fixed_num_aggregators = 0;
};

struct RankAggregationInfo {
    int rank = -1;
    uint64_t num_particles = -1;

    RankAggregationInfo(const int rank, const uint64_t num_particles);

    RankAggregationInfo() = default;
};

/* A very simple median-split kd tree
 */
struct AggregationTree {
    Box bounds;
    uint32_t num_aggregators;
    uint64_t num_points;

    std::vector<AttributeDescription> attributes;
    phmap::flat_hash_map<std::string, size_t> attrib_ids;

    ArrayHandle<AggregationKdNode> nodes;
    ArrayHandle<uint32_t> leaf_indices;
    ArrayHandle<RankAggregationInfo> primitives;
    ArrayHandle<uint32_t> bitmap_dictionary;
    ArrayHandle<uint16_t> node_bitmap_ids;

    // May be null if not stored in file
    ArrayHandle<glm::vec2> node_attrib_ranges;

    std::string bat_prefix;
    // The trees which have been loaded by the traversal
    phmap::flat_hash_map<uint64_t, std::shared_ptr<BATree>> loaded_trees;

    bool enable_range_filtering = true;
    bool enable_bitmap_filtering = true;

    AggregationTree(const Box &bounds,
                    const uint64_t num_points,
                    const ArrayHandle<AggregationKdNode> &nodes,
                    const ArrayHandle<uint32_t> &leaf_indices,
                    const ArrayHandle<RankAggregationInfo> &primitives);

    // When loaded from disk we no longer have the leaf indices info since it's
    // only needed during aggregation
    AggregationTree(const Box &bounds,
                    const uint32_t num_aggregators,
                    const uint64_t num_points,
                    const std::vector<AttributeDescription> &attributes,
                    const ArrayHandle<AggregationKdNode> &nodes,
                    const ArrayHandle<uint32_t> &bitmap_dictionary,
                    const ArrayHandle<uint16_t> &node_bitmap_ids,
                    const ArrayHandle<glm::vec2> &node_attrib_ranges,
                    const std::string &bat_prefix);

    AggregationTree() = default;

    // Called on the write side to build the attribute bitmaps and ranges for the inner nodes
    void initialize_attributes(const std::vector<AttributeDescription> &attributes,
                               const std::vector<uint32_t> &aggregator_attribute_bitmaps,
                               const std::vector<glm::vec2> &aggregator_attribute_ranges);

    // Query the particles contained in some box, retrieving just those particles
    // which are new for the selected quality level given the previous one
    template <typename Fn>
    QueryStats query_box_progressive(const Box &b,
                                     std::vector<AttributeQuery> *attrib_queries,
                                     float prev_quality,
                                     float current_quality,
                                     const Fn &callback);

    // Query all particles contained in some bounding box
    // The callback should take the point id, position, and list of attributes to read
    // the point's attributes from, if desired. The function signature should be:
    // void (const size_t id, const glm::vec3 &pos, const std::vector<Attribute> &attributes)
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

    // Get the IDs of the BATrees which contain the query box
    std::vector<size_t> get_overlapped_subtree_ids(const Box &b) const;

    // Utility function to iterate through the BATrees and process them with the
    // callback. Mainly for the inspector, to get meta-data about the aggregtor's trees
    template <typename Fn>
    void iterate_aggregator_trees(const Fn &fn);

private:
    void initialize_node_attributes(
        const size_t n,
        const std::vector<uint32_t> &aggregator_attribute_bitmaps,
        const std::vector<glm::vec2> &aggregator_attribute_ranges,
        const phmap::flat_hash_map<int, size_t> &aggregator_indices,
        std::vector<uint32_t> &node_bitmaps,
        OwnedArrayHandle<glm::vec2> &attrib_ranges);

    bool node_overlaps_query(uint32_t n,
                             const std::vector<AttributeQuery> &query,
                             const std::vector<size_t> &query_indices,
                             QueryStats &stats) const;

    std::shared_ptr<BATree> fetch_tree(const size_t tree_id);
};

template <typename Fn>
QueryStats AggregationTree::query_box_progressive(const Box &b,
                                                  std::vector<AttributeQuery> *attrib_queries,
                                                  float prev_quality,
                                                  float current_quality,
                                                  const Fn &callback)
{
    QueryStats stats;
    if (!b.overlaps(bounds)) {
        return stats;
    }
    if (prev_quality == current_quality && current_quality != 0.f) {
        return stats;
    }

    std::vector<size_t> query_indices;
    std::vector<uint32_t> global_bitmasks;
    if (attrib_queries) {
        for (auto &a : *attrib_queries) {
            auto fnd = attrib_ids.find(a.name);
            if (fnd == attrib_ids.end()) {
                throw std::runtime_error("Request for attribute " + a.name +
                                         " which does not exist");
            }
            query_indices.push_back(fnd->second);
            a.data_type = attributes[fnd->second].data_type;
            a.bitmask = a.query_bitmask(attributes[fnd->second].range);
            global_bitmasks.push_back(a.bitmask);
        }
    }

    prev_quality = remap_quality(prev_quality);
    current_quality = remap_quality(current_quality);

    std::array<size_t, 64> node_stack = {0};
    size_t stack_idx = 0;
    size_t current_node = 0;
    while (true) {
        const AggregationKdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            bool traverse_left = b.lower[node.split_axis()] < node.split_pos;
            bool traverse_right = b.upper[node.split_axis()] > node.split_pos;
            if (attrib_queries && traverse_left) {
                traverse_left = traverse_left &&
                                node_overlaps_query(
                                    current_node + 1, *attrib_queries, query_indices, stats);
            }

            if (attrib_queries && traverse_right) {
                traverse_right =
                    traverse_right &&
                    node_overlaps_query(
                        node.right_child_offset(), *attrib_queries, query_indices, stats);
            }
            // If both overlap, descend both children following the left first
            if (traverse_left && traverse_right) {
                node_stack[stack_idx] = node.right_child_offset();
                stack_idx++;

                current_node = current_node + 1;
                continue;
            } else if (traverse_left) {
                current_node = current_node + 1;
                continue;
            } else if (traverse_right) {
                current_node = node.right_child_offset();
                continue;
            }
        } else {
            auto tree = fetch_tree(node.aggregator_rank);

            // Sub-tree ranges are local to the subtree, so we need to remap the query masks
            for (size_t i = 0; i < query_indices.size(); ++i) {
                const size_t attr = query_indices[i];
                const glm::vec2 &global_range = attributes[attr].range;
                const glm::vec2 &subtree_range = tree->attributes[attr].range;
                (*attrib_queries)[i].bitmask =
                    remap_bitmask(global_bitmasks[i], global_range, subtree_range);
            }
            tree->query_box_log(b,
                                attrib_queries,
                                query_indices,
                                prev_quality,
                                current_quality,
                                callback,
                                stats);

            for (size_t i = 0; i < query_indices.size(); ++i) {
                (*attrib_queries)[i].bitmask = global_bitmasks[i];
            }
        }
        // Pop the stack to get the next node to traverse
        if (stack_idx > 0) {
            --stack_idx;
            current_node = node_stack[stack_idx];
        } else {
            break;
        }
    }
    return stats;
}

template <typename Fn>
QueryStats AggregationTree::query_box(const Box &b,
                                      std::vector<AttributeQuery> *attrib_queries,
                                      float quality,
                                      const Fn &callback)
{
    return query_box_progressive(b, attrib_queries, 0.f, quality, callback);
}

template <typename Fn>
void AggregationTree::iterate_aggregator_trees(const Fn &fn)
{
    for (size_t i = 0; i < nodes->size(); ++i) {
        const auto &node = nodes->at(i);
        if (node.is_leaf()) {
            auto tree = fetch_tree(node.aggregator_rank);
            fn(tree);
        }
    }
}
