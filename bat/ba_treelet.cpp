#include "ba_treelet.h"
#include <iostream>
#include <glm/ext.hpp>
#include "borrowed_array.h"
#include "byte_cursor.h"

BATreeletHeader::BATreeletHeader(uint32_t n_nodes, uint32_t n_points)
    : num_nodes(n_nodes), num_points(n_points)
{
}

BATreelet::BATreelet(const ArrayHandle<uint8_t> &buf,
                     const size_t offset,
                     const ArrayHandle<uint32_t> &bitmap_dictionary,
                     const std::vector<AttributeDescription> &attrib_descs,
                     uint32_t num_lod_prims,
                     bool has_attrib_ranges)
    : tree_mem(buf), bitmap_dictionary(bitmap_dictionary), num_lod_prims(num_lod_prims)
{
    ByteCursor cursor(tree_mem->data() + offset);
    BATreeletHeader header;
    cursor.read(header);

    nodes = std::dynamic_pointer_cast<AbstractArray<KdNode>>(
        std::make_shared<BorrowedArray<KdNode>>(reinterpret_cast<KdNode *>(cursor.position()),
                                                header.num_nodes));
    cursor.advance(nodes->size_bytes());

    // TODO: I think throughout the rest of the code it's assumed the data has at least one
    // attribute
    if (!attrib_descs.empty()) {
        node_bitmap_ids = std::dynamic_pointer_cast<AbstractArray<uint16_t>>(
            std::make_shared<BorrowedArray<uint16_t>>(
                reinterpret_cast<uint16_t *>(cursor.position()),
                header.num_nodes * attrib_descs.size()));
        cursor.advance(node_bitmap_ids->size_bytes());

        if (has_attrib_ranges) {
            node_attrib_ranges = std::dynamic_pointer_cast<AbstractArray<glm::vec2>>(
                std::make_shared<BorrowedArray<glm::vec2>>(
                    reinterpret_cast<glm::vec2 *>(cursor.position()),
                    header.num_nodes * attrib_descs.size()));
            cursor.advance(node_attrib_ranges->size_bytes());
        }
    }

    points = std::dynamic_pointer_cast<AbstractArray<glm::vec3>>(
        std::make_shared<BorrowedArray<glm::vec3>>(
            reinterpret_cast<glm::vec3 *>(cursor.position()), header.num_points));
    cursor.advance(points->size_bytes());

    for (const auto &desc : attrib_descs) {
        auto data = std::dynamic_pointer_cast<AbstractArray<uint8_t>>(
            std::make_shared<BorrowedArray<uint8_t>>(
                reinterpret_cast<uint8_t *>(cursor.position()),
                header.num_points * desc.stride()));
        cursor.advance(data->size_bytes());

        attribs.emplace_back(desc, data);
    }
}

void BATreelet::get_splitting_planes(std::vector<Plane> &planes,
                                     const Box &b,
                                     std::vector<AttributeQuery> *attrib_queries,
                                     const std::vector<size_t> &query_indices,
                                     uint32_t max_depth,
                                     const Box &treelet_bounds,
                                     QueryStats &stats) const
{
    std::array<size_t, 64> node_stack = {0};
    std::array<uint32_t, 64> depth_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    uint32_t current_depth = 0;
    Box current_bounds = treelet_bounds;
    while (true) {
        const KdNode &node = nodes->at(current_node);

        if (!node.is_leaf()) {
            glm::vec3 plane_origin = current_bounds.center();
            plane_origin[node.split_axis()] = node.split_pos;

            glm::vec3 plane_half_vecs = current_bounds.diagonal() / 2.f;
            plane_half_vecs[node.split_axis()] = 0.f;

            planes.emplace_back(plane_origin, plane_half_vecs);

            if (current_depth < max_depth) {
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
                    bounds_stack[stack_idx] = current_bounds;
                    bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                    stack_idx++;

                    current_node = current_node + 1;
                    current_depth++;
                    current_bounds.upper[node.split_axis()] = node.split_pos;
                    continue;
                } else if (left_overlaps) {
                    current_node = current_node + 1;
                    current_depth++;
                    current_bounds.upper[node.split_axis()] = node.split_pos;
                    continue;
                } else if (right_overlaps) {
                    current_node = node.right_child_offset();
                    current_depth++;
                    current_bounds.lower[node.split_axis()] = node.split_pos;
                    continue;
                }
            }
        }
        // Pop the stack to get the next node to traverse
        if (stack_idx > 0) {
            --stack_idx;
            current_node = node_stack[stack_idx];
            current_bounds = bounds_stack[stack_idx];
            current_depth = depth_stack[stack_idx];
        } else {
            break;
        }
    }
}

bool BATreelet::node_overlaps_query(uint32_t n,
                                    const std::vector<size_t> &query_indices,
                                    const std::vector<AttributeQuery> &query,
                                    QueryStats &stats) const
{
    ++stats.n_total_tested;
    for (size_t i = 0; i < query.size(); ++i) {
        const size_t attr = query_indices[i];
        const uint16_t bmap_id = node_bitmap_ids->at(n * attribs.size() + attr);
        const uint32_t node_bitmap = bitmap_dictionary->at(bmap_id);

        // TODO: On the scivis data set it seems like i get some 0 masks which shouldn't be 0
        if (node_bitmap == 0) {
            std::cout << "node " << n << " has a bitmap of 0!?\n";
        }

        // These are kept around for testing the bitmaps vs. ranges option
        bool would_range_filter = false;
        glm::vec2 node_range(0);
        if (node_attrib_ranges && enable_range_filtering) {
            node_range = node_attrib_ranges->at(n * attribs.size() + attr);
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

/*
void BATree::get_bounding_boxes(std::vector<Box> &boxes, const Box &b) const
{
    std::array<size_t, 64> node_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    Box current_bounds = tree_bounds;

    while (true) {
        const KdNode &node = nodes->at(current_node);
        boxes.push_back(current_bounds);
        if (!node.is_leaf()) {
            // Interior node, descend into children if they're contained in the box
            const bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
            const bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;

            if (left_overlaps && right_overlaps) {
                node_stack[stack_idx] = node.right_child_offset();
                bounds_stack[stack_idx] = current_bounds;
                bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                ++stack_idx;

                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else if (left_overlaps) {
                current_node = current_node + 1;
                current_bounds.upper[node.split_axis()] = node.split_pos;
            } else {
                current_node = node.right_child_offset();
                current_bounds.lower[node.split_axis()] = node.split_pos;
            }
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

void BATree::get_splitting_planes(std::vector<Plane> &planes,
                                  const Box &b,
                                  std::vector<AttributeQuery> *attrib_queries,
                                  float quality) const
{
    const uint32_t query_depth =
        glm::clamp(uint32_t(quality * (max_lod - min_lod) + min_lod), min_lod, max_lod);

    std::array<size_t, 64> node_stack = {0};
    std::array<uint32_t, 64> depth_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    uint32_t current_depth = 0;
    Box current_bounds = tree_bounds;
    QueryStats query_stats;

    std::vector<size_t> query_indices;
    if (attrib_queries) {
        for (auto &a : *attrib_queries) {
            auto fnd = attrib_ids.find(a.name);
            if (fnd == attrib_ids.end()) {
                throw std::runtime_error("Request for attribute " + a.name +
                                         " which does not exist");
            }
            query_indices.push_back(fnd->second);
            a.data_type = attribs[fnd->second].data_type;
            a.bitmask = a.query_bitmask(attribs[fnd->second].range);

            for (const auto &r : a.ranges) {
                if (r.x == r.y) {
                    std::cout << "Not querying empty range\n";
                    return;
                }
            }
        }
    }

    while (true) {
        const KdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            glm::vec3 plane_origin = current_bounds.center();
            plane_origin[node.split_axis()] = node.split_pos;

            glm::vec3 plane_half_vecs = current_bounds.diagonal() / 2.f;

            plane_half_vecs[node.split_axis()] = 0.f;

            planes.emplace_back(plane_origin, plane_half_vecs);

            if (current_depth < query_depth) {
                // Interior node, descend into children if they're contained in the box
                bool left_overlaps = b.lower[node.split_axis()] <= node.split_pos;
                bool right_overlaps = b.upper[node.split_axis()] >= node.split_pos;
                if (attrib_queries) {
                    left_overlaps =
                        left_overlaps &&
                        node_overlaps_query(
                            current_node + 1, query_indices, *attrib_queries, query_stats);
                    right_overlaps =
                        right_overlaps && node_overlaps_query(node.right_child_offset(),
                                                              query_indices,
                                                              *attrib_queries,
                                                              query_stats);
                }

                // If both overlap, descend both children following the left first
                if (left_overlaps && right_overlaps) {
                    node_stack[stack_idx] = node.right_child_offset();
                    depth_stack[stack_idx] = current_depth + 1;
                    bounds_stack[stack_idx] = current_bounds;
                    bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                    ++stack_idx;

                    current_node = current_node + 1;
                    current_depth++;
                    current_bounds.upper[node.split_axis()] = node.split_pos;
                    continue;
                } else if (left_overlaps) {
                    current_node = current_node + 1;
                    current_depth++;
                    current_bounds.upper[node.split_axis()] = node.split_pos;
                    continue;
                } else if (right_overlaps) {
                    current_node = node.right_child_offset();
                    current_depth++;
                    current_bounds.lower[node.split_axis()] = node.split_pos;
                    continue;
                }
            }
        }
        // Pop the stack to get the next node to traverse
        if (stack_idx > 0) {
            --stack_idx;
            current_node = node_stack[stack_idx];
            current_bounds = bounds_stack[stack_idx];
            current_depth = depth_stack[stack_idx];
        } else {
            break;
        }
    }
}
*/
