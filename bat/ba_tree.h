#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <parallel_hashmap/phmap.h>
#include "abstract_array.h"
#include "attribute.h"
#include "ba_treelet.h"
#include "box.h"
#include "owned_array.h"
#include "plane.h"
#include "query_stats.h"
#include "radix_tree_node.h"

struct AggregationTree;

// Scale quality by a log scale so that 0.5 quality = 50% data. Since the
// tree doubles the amount of data at each level we need to apply an inverse
// mapping of this to the quality level
inline float remap_quality(const float q)
{
    return glm::clamp(std::log10(1.f + q * 9.f), 0.f, 1.f);
}

struct BATree {
    friend struct AggregationTree;

    Box bounds;
    std::vector<AttributeDescription> attributes;
    phmap::flat_hash_map<std::string, size_t> attrib_ids;

    ArrayHandle<uint8_t> tree_mem;

    ArrayHandle<RadixTreeNode> radix_tree;
    ArrayHandle<uint64_t> treelet_offsets;
    ArrayHandle<uint32_t> bitmap_dictionary;
    ArrayHandle<uint16_t> node_bitmap_ids;

    // May be null if not stored in file
    ArrayHandle<glm::vec2> node_attrib_ranges;

    // The treelets which have been loaded by the traversal
    phmap::flat_hash_map<uint64_t, std::shared_ptr<BATreelet>> loaded_treelets;

    size_t num_points = 0;

    // The number of LOD primitives stored for each node which has LOD primitives
    uint32_t num_lod_prims = 0;

    // The min/max levels of detail which can be queried from the tree. Since LOD
    // primitives aren't stored in the top-level coarse tree there's a min query
    // depth we have to do to get data (min_lod). max_lod is the max depth of any
    // path down to a leaf in the tree
    uint32_t min_lod = 0;
    uint32_t max_lod = 0;

    bool enable_range_filtering = true;
    bool enable_bitmap_filtering = true;

    BATree(const Box &bounds,
           const std::vector<AttributeDescription> &attributes,
           const ArrayHandle<uint8_t> &tree_mem,
           const ArrayHandle<RadixTreeNode> &radix_tree,
           const ArrayHandle<uint64_t> &treelet_offsets,
           const ArrayHandle<uint32_t> &bitmap_dictionary,
           const ArrayHandle<uint16_t> &node_bitmap_ids,
           const ArrayHandle<glm::vec2> &node_attrib_ranges,
           size_t num_points,
           uint32_t num_lod_prims,
           uint32_t min_lod,
           uint32_t max_lod);

    BATree() = default;

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

    // Utility function to iterate through the BATreelets and process them with the
    // callback. Mainly for the inspector, to get meta-data about the treelets
    template <typename Fn>
    void iterate_treelets(const Fn &fn);

private:
    // Query all particles in the bounding box, using the already provided query indices
    // The query value should have previously had the log scale applied
    template <typename Fn>
    void query_box_log(const Box &b,
                       std::vector<AttributeQuery> *attrib_queries,
                       const std::vector<size_t> &query_indices,
                       float prev_quality,
                       float current_quality,
                       const Fn &callback,
                       QueryStats &stats);

    // Get the splitting plans for the log-scaled quality level
    void get_splitting_planes_log(std::vector<Plane> &planes,
                                  const Box &query_box,
                                  std::vector<AttributeQuery> *attrib_queries,
                                  float quality);

    bool node_overlaps_query(uint32_t n,
                             const std::vector<AttributeQuery> &query,
                             const std::vector<size_t> &query_indices,
                             QueryStats &stats) const;

    std::shared_ptr<BATreelet> fetch_treelet(const size_t treelet_id);
};

template <typename Fn>
QueryStats BATree::query_box_progressive(const Box &b,
                                         std::vector<AttributeQuery> *attrib_queries,
                                         float prev_quality,
                                         float current_quality,
                                         const Fn &callback)
{
    QueryStats stats;
    if (!b.overlaps(bounds)) {
        return stats;
    }

    std::vector<size_t> query_indices;
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
        }
    }

    prev_quality = remap_quality(prev_quality);
    current_quality = remap_quality(current_quality);

    query_box_log(
        b, attrib_queries, query_indices, prev_quality, current_quality, callback, stats);

    return stats;
}

template <typename Fn>
QueryStats BATree::query_box(const Box &b,
                             std::vector<AttributeQuery> *attrib_queries,
                             float quality,
                             const Fn &callback)
{
    return query_box_progressive(b, attrib_queries, 0.f, quality, callback);
}

template <typename Fn>
void BATree::query_box_log(const Box &b,
                           std::vector<AttributeQuery> *attrib_queries,
                           const std::vector<size_t> &query_indices,
                           float prev_quality,
                           float current_quality,
                           const Fn &callback,
                           QueryStats &stats)
{
    if (!b.overlaps(bounds)) {
        return;
    }
    if (prev_quality == current_quality && current_quality != 0.f) {
        return;
    }

    stats.query_depth = glm::clamp(
        uint32_t(current_quality * (max_lod - min_lod) + min_lod), min_lod, max_lod);

    const QueryQuality quality(prev_quality, current_quality, min_lod, max_lod);

    std::array<size_t, 64> node_stack = {0};
    size_t stack_idx = 0;
    size_t current_node = 0;
    while (true) {
        const RadixTreeNode &node = radix_tree->at(current_node);

        bool traverse_left = true;
        if (!node.left_leaf()) {
            traverse_left = b.lower[node.split_axis] < node.split_position;
            if (attrib_queries && traverse_left) {
                traverse_left = traverse_left &&
                                node_overlaps_query(
                                    node.left_child(), *attrib_queries, query_indices, stats);
            }
        } else {
            traverse_left = false;
            auto treelet = fetch_treelet(node.left_child());
            treelet->query_box(b, attrib_queries, query_indices, quality, callback, stats);
        }

        bool traverse_right = true;
        if (!node.right_leaf()) {
            traverse_right = b.upper[node.split_axis] > node.split_position;
            if (attrib_queries && traverse_right) {
                traverse_right =
                    traverse_right &&
                    node_overlaps_query(
                        node.right_child(), *attrib_queries, query_indices, stats);
            }
        } else {
            traverse_right = false;
            auto treelet = fetch_treelet(node.right_child());
            treelet->query_box(b, attrib_queries, query_indices, quality, callback, stats);
        }

        // If both overlap, descend both children following the left first
        if (traverse_left && traverse_right) {
            node_stack[stack_idx] = node.right_child();
            stack_idx++;

            current_node = node.left_child();
        } else if (traverse_left) {
            current_node = node.left_child();
        } else if (traverse_right) {
            current_node = node.right_child();
        } else {
            // Pop the stack to get the next node to traverse
            if (stack_idx > 0) {
                --stack_idx;
                current_node = node_stack[stack_idx];
            } else {
                break;
            }
        }
    }
}

template <typename Fn>
void BATree::iterate_treelets(const Fn &fn)
{
    for (size_t i = 0; i < treelet_offsets->size(); ++i) {
        auto treelet = fetch_treelet(i);
        fn(treelet);
    }
}
