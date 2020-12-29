#pragma once

#include <memory>
#include <ostream>
#include <unordered_map>
#include <vector>
#include "abstract_array.h"
#include "attribute.h"
#include "box.h"
#include "kd_node.h"
#include "plane.h"
#include "query_stats.h"

#pragma pack(push, 1)
struct BATreeletHeader {
    uint32_t num_nodes = 0;
    uint32_t num_points = 0;

    BATreeletHeader(uint32_t n_nodes, uint32_t n_points);

    BATreeletHeader() = default;
};
#pragma pack(pop)

struct BATreelet {
    // The memory containing the treelet, kept referenced here to ensure a valid lifetime
    ArrayHandle<uint8_t> tree_mem;

    ArrayHandle<uint32_t> bitmap_dictionary;
    uint32_t num_lod_prims = 0;

    ArrayHandle<KdNode> nodes;
    ArrayHandle<uint16_t> node_bitmap_ids;

    // May be null if not stored in file
    ArrayHandle<glm::vec2> node_attrib_ranges;

    ArrayHandle<glm::vec3> points;
    std::vector<Attribute> attribs;

    bool enable_range_filtering = true;
    bool enable_bitmap_filtering = true;

    // Initialize the treelet stored at the passed address. buf[offset] should contain
    // the BATreeletHeader for the treelet, which is followed by the treelet data.
    BATreelet(const ArrayHandle<uint8_t> &buf,
              const size_t offset,
              const ArrayHandle<uint32_t> &bitmap_dictionary,
              const std::vector<AttributeDescription> &attrib_descs,
              uint32_t num_lod_prims,
              bool has_attrib_ranges);

    BATreelet() = default;

    template <typename Fn>
    void query_box(const Box &b,
                   std::vector<AttributeQuery> *attrib_queries,
                   const std::vector<size_t> &query_indices,
                   const QueryQuality &quality,
                   const Fn &callback,
                   QueryStats &stats) const;

    // For debugging/testing: query the splitting planes of the tree
    void get_splitting_planes(std::vector<Plane> &planes,
                              const Box &query_box,
                              std::vector<AttributeQuery> *attrib_queries,
                              const std::vector<size_t> &query_indices,
                              uint32_t max_depth,
                              const Box &treelet_bounds,
                              QueryStats &stats) const;

private:
    bool node_overlaps_query(uint32_t n,
                             const std::vector<size_t> &query_indices,
                             const std::vector<AttributeQuery> &query,
                             QueryStats &query_stats) const;

    template <typename Fn>
    void process_queried_points(const size_t offset,
                                const size_t count,
                                const Box &b,
                                const uint32_t current_depth,
                                std::vector<AttributeQuery> *attrib_queries,
                                const std::vector<size_t> &query_indices,
                                const Fn &callback,
                                const QueryQuality &quality,
                                QueryStats &stats) const;
};

template <typename Fn>
void BATreelet::query_box(const Box &b,
                          std::vector<AttributeQuery> *attrib_queries,
                          const std::vector<size_t> &query_indices,
                          const QueryQuality &quality,
                          const Fn &callback,
                          QueryStats &stats) const
{
    std::array<size_t, 64> node_stack = {0};
    std::array<uint32_t, 64> depth_stack = {0};
    size_t stack_idx = 0;
    size_t current_node = 0;
    uint32_t current_depth = 0;
    while (true) {
        const KdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            // Collect LOD prims down the tree as we traverse the query
            // TODO: Later these LOD prims be stored packed in a separate up front region
            // so they're faster to access
            if (node.has_lod_prims() && current_depth >= quality.prev_depth) {
                process_queried_points(node.lod_prims,
                                       num_lod_prims,
                                       b,
                                       current_depth,
                                       attrib_queries,
                                       query_indices,
                                       callback,
                                       quality,
                                       stats);
            }

            if (current_depth < quality.current_depth) {
                // Interior node, descend into children if they're contained in the box
                bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
                bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;
                if (attrib_queries) {
                    left_overlaps =
                        left_overlaps &&
                        node_overlaps_query(
                            current_node + 1, query_indices, *attrib_queries, stats);

                    right_overlaps =
                        right_overlaps &&
                        node_overlaps_query(
                            node.right_child_offset(), query_indices, *attrib_queries, stats);
                }

                // If both overlap, descend both children following the left first
                if (left_overlaps && right_overlaps) {
                    depth_stack[stack_idx] = current_depth + 1;
                    node_stack[stack_idx] = node.right_child_offset();
                    stack_idx++;

                    current_node = current_node + 1;
                    current_depth++;
                    continue;
                } else if (left_overlaps) {
                    current_node = current_node + 1;
                    current_depth++;
                    continue;
                } else if (right_overlaps) {
                    current_node = node.right_child_offset();
                    current_depth++;
                    continue;
                }
            }
        } else if (current_depth >= quality.prev_depth) {
            // Leaf node, collect points contained in the box
            process_queried_points(node.prim_indices_offset,
                                   node.get_num_prims(),
                                   b,
                                   current_depth,
                                   attrib_queries,
                                   query_indices,
                                   callback,
                                   quality,
                                   stats);
        }

        // Pop the stack to get the next node to traverse
        if (stack_idx > 0) {
            --stack_idx;
            current_node = node_stack[stack_idx];
            current_depth = depth_stack[stack_idx];
        } else {
            break;
        }
    }
}

template <typename Fn>
void BATreelet::process_queried_points(const size_t offset,
                                       const size_t count,
                                       const Box &b,
                                       const uint32_t current_depth,
                                       std::vector<AttributeQuery> *attrib_queries,
                                       const std::vector<size_t> &query_indices,
                                       const Fn &callback,
                                       const QueryQuality &quality,
                                       QueryStats &stats) const
{
    const size_t start =
        quality.prev_depth < current_depth ? 0 : count * quality.prev_fraction;

    const size_t end =
        quality.current_depth > current_depth ? count : count * quality.current_fraction;

    if (start == end || start > end) {
        return;
    }

    stats.n_particles_tested += end - start;
    for (size_t i = start; i < end; ++i) {
        const auto &pt = points->at(offset + i);
        if (b.contains_point(pt)) {
            bool in_query = true;
            if (attrib_queries) {
                for (size_t j = 0; j < attrib_queries->size(); ++j) {
                    AttributeQuery &q = (*attrib_queries)[j];
                    const size_t attr = query_indices[j];
                    const float fval = attribs[attr].as_float(offset + i);
                    if (!q.contains_value(fval)) {
                        in_query = false;
                        break;
                    }
                }
            }
            if (in_query) {
                stats.n_particles_returned++;
                callback(offset + i, pt, attribs);
            }
        }
    }
}
