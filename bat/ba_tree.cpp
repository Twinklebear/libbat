#include "ba_tree.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <tbb/task_group.h>
#include "owned_array.h"
#include "util.h"

BATree::BATree(const Box &bounds,
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
               uint32_t max_lod)
    : bounds(bounds),
      attributes(attributes),
      tree_mem(tree_mem),
      radix_tree(radix_tree),
      treelet_offsets(treelet_offsets),
      bitmap_dictionary(bitmap_dictionary),
      node_bitmap_ids(node_bitmap_ids),
      node_attrib_ranges(node_attrib_ranges),
      num_points(num_points),
      num_lod_prims(num_lod_prims),
      min_lod(min_lod),
      max_lod(max_lod)
{
    for (size_t i = 0; i < attributes.size(); ++i) {
        attrib_ids[attributes[i].name] = i;
    }
}

void BATree::get_splitting_planes(std::vector<Plane> &planes,
                                  const Box &b,
                                  std::vector<AttributeQuery> *attrib_queries,
                                  float quality)
{
    quality = remap_quality(quality);
    get_splitting_planes_log(planes, b, attrib_queries, quality);
}

void BATree::get_splitting_planes_log(std::vector<Plane> &planes,
                                      const Box &b,
                                      std::vector<AttributeQuery> *attrib_queries,
                                      float quality)
{
    if (!b.overlaps(bounds)) {
        return;
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

    QueryStats stats;
    stats.query_depth =
        glm::clamp(uint32_t(quality * (max_lod - min_lod) + min_lod), min_lod, max_lod);
    const uint32_t max_depth = stats.query_depth - min_lod;

    std::array<size_t, 64> node_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    Box current_bounds = bounds;

    while (true) {
        const RadixTreeNode &node = radix_tree->at(current_node);

        glm::vec3 plane_origin = current_bounds.center();
        plane_origin[node.split_axis] = node.split_position;

        glm::vec3 plane_half_vecs = current_bounds.diagonal() / 2.f;
        plane_half_vecs[node.split_axis] = 0.f;

        planes.emplace_back(plane_origin, plane_half_vecs);

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

            treelet->get_splitting_planes(
                planes, b, attrib_queries, query_indices, max_depth, current_bounds, stats);
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

            treelet->get_splitting_planes(
                planes, b, attrib_queries, query_indices, max_depth, current_bounds, stats);
        }

        // If both overlap, descend both children following the left first
        if (traverse_left && traverse_right) {
            node_stack[stack_idx] = node.right_child();
            bounds_stack[stack_idx] = current_bounds;
            bounds_stack[stack_idx].lower[node.split_axis] = node.split_position;
            stack_idx++;

            current_node = node.left_child();
            current_bounds.upper[node.split_axis] = node.split_position;
        } else if (traverse_left) {
            current_node = node.left_child();
            current_bounds.upper[node.split_axis] = node.split_position;
        } else if (traverse_right) {
            current_node = node.right_child();
            current_bounds.lower[node.split_axis] = node.split_position;
        } else {
            // Pop the stack to get the next node to traverse
            if (stack_idx > 0) {
                --stack_idx;
                current_node = node_stack[stack_idx];
                current_bounds = bounds_stack[stack_idx];
            } else {
                break;
            }
        }
    }
}

bool BATree::node_overlaps_query(uint32_t n,
                                 const std::vector<AttributeQuery> &query,
                                 const std::vector<size_t> &query_indices,
                                 QueryStats &stats) const
{
    ++stats.n_total_tested;
    for (size_t i = 0; i < query.size(); ++i) {
        const size_t attr = query_indices[i];
        const uint16_t bmap_id = node_bitmap_ids->at(n * attributes.size() + attr);
        const uint32_t node_bitmap = bitmap_dictionary->at(bmap_id);

        if (node_bitmap == 0) {
            std::cout << "node " << n << " has a bitmap of 0!?\n";
        }

        // These are kept around for testing the bitmaps vs. ranges option
        bool would_range_filter = false;
        glm::vec2 node_range(0);
        if (node_attrib_ranges && enable_range_filtering) {
            node_range = node_attrib_ranges->at(n * attributes.size() + attr);
            would_range_filter = true;
            for (const auto &r : query[i].ranges) {
                const glm::vec2 overlap(glm::max(r.x, node_range.x),
                                        glm::min(r.y, node_range.y));
                if (overlap.x <= overlap.y) {
                    would_range_filter = false;
                    break;
                }
            }
        }

        bool would_mask_filter = false;
        const uint32_t query_mask = query[i].bitmask;
        if (!(query_mask & node_bitmap) && enable_bitmap_filtering) {
            would_mask_filter = true;
        }

#if 0
        if (!would_range_filter && would_mask_filter) {
            std::cout << "=====\nNode " << n << " was mask filtered but not range filtered!\n"
                      << "  node range: " << glm::to_string(node_range)
                      << "  query range: " << glm::to_string(query[i].ranges[0]) << "\n";
            printf("  query mask 0x%08x, node mask: 0x%08x\n", query_mask, node_bitmap);
            std::cout << "=====\n";
        }
#endif

#if 1
        if (would_range_filter) {
            ++stats.n_range_filtered;
            return false;
        }
#endif
#if 1
        if (would_mask_filter) {
            ++stats.n_map_filtered;
            return false;
        }
#endif
    }
    return true;
}

std::shared_ptr<BATreelet> BATree::fetch_treelet(const size_t treelet_id)
{
    auto fnd = loaded_treelets.find(treelet_id);
    std::shared_ptr<BATreelet> treelet = nullptr;
    if (fnd == loaded_treelets.end()) {
        loaded_treelets[treelet_id] =
            std::make_shared<BATreelet>(tree_mem,
                                        treelet_offsets->at(treelet_id),
                                        bitmap_dictionary,
                                        attributes,
                                        num_lod_prims,
                                        node_attrib_ranges != nullptr);
        treelet = loaded_treelets[treelet_id];
    } else {
        treelet = fnd->second;
    }
    treelet->enable_range_filtering = enable_range_filtering;
    treelet->enable_bitmap_filtering = enable_bitmap_filtering;
    return treelet;
}
