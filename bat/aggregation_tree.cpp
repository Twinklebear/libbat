#include "aggregation_tree.h"
#include <glm/ext.hpp>
#include <tbb/parallel_sort.h>
#include "bat_file.h"
#include "fixed_array.h"
#include "profiling.h"

AggregationTree::AggregationTree(const Box &bounds,
                                 const uint64_t num_points,
                                 const ArrayHandle<AggregationKdNode> &nodes,
                                 const ArrayHandle<uint32_t> &leaf_indices,
                                 const ArrayHandle<RankAggregationInfo> &primitives)

    : bounds(bounds),
      num_aggregators(leaf_indices->size()),
      num_points(num_points),
      nodes(nodes),
      leaf_indices(leaf_indices),
      primitives(primitives)
{
}

AggregationTree::AggregationTree(const Box &bounds,
                                 const uint32_t num_aggregators,
                                 const uint64_t num_points,
                                 const std::vector<AttributeDescription> &attribs,
                                 const ArrayHandle<AggregationKdNode> &nodes,
                                 const ArrayHandle<uint32_t> &bitmap_dictionary,
                                 const ArrayHandle<uint16_t> &node_bitmap_ids,
                                 const ArrayHandle<glm::vec2> &node_attrib_ranges,
                                 const std::string &bat_prefix)
    : bounds(bounds),
      num_aggregators(num_aggregators),
      num_points(num_points),
      attributes(attribs),
      nodes(nodes),
      bitmap_dictionary(bitmap_dictionary),
      node_bitmap_ids(node_bitmap_ids),
      node_attrib_ranges(node_attrib_ranges),
      bat_prefix(bat_prefix)
{
    for (size_t i = 0; i < attributes.size(); ++i) {
        attrib_ids[attributes[i].name] = i;
    }
}

void AggregationTree::initialize_attributes(
    const std::vector<AttributeDescription> &attribute_info,
    const std::vector<uint32_t> &aggregator_attribute_bitmaps,
    const std::vector<glm::vec2> &aggregator_attribute_ranges)
{
    attributes = attribute_info;
    std::vector<uint32_t> remapped_bitmaps = aggregator_attribute_bitmaps;
    for (size_t i = 0; i < attributes.size(); ++i) {
        auto &attr = attributes[i];
        attrib_ids[attr.name] = i;
        attr.range = glm::vec2(std::numeric_limits<float>::infinity(),
                               -std::numeric_limits<float>::infinity());

        for (size_t j = 0; j < num_aggregators; ++j) {
            attr.range.x =
                std::min(attr.range.x, aggregator_attribute_ranges[i * num_aggregators + j].x);
            attr.range.y =
                std::max(attr.range.y, aggregator_attribute_ranges[i * num_aggregators + j].y);
        }
        for (size_t j = 0; j < num_aggregators; ++j) {
            remapped_bitmaps[i * num_aggregators + j] =
                remap_bitmask(aggregator_attribute_bitmaps[i * num_aggregators + j],
                              aggregator_attribute_ranges[i * num_aggregators + j],
                              attr.range);
        }
    }

#ifdef BAT_STORE_NODE_RANGES
    auto attr_ranges =
        std::make_shared<OwnedArray<glm::vec2>>(nodes->size() * attributes.size());
#else
    OwnedArrayHandle<glm::vec2> attr_ranges = nullptr;
#endif

    // Build the mapping from global aggregator rank to aggregator ID
    phmap::flat_hash_map<int, size_t> aggregator_indices;
    {
        std::vector<int> aggregator_ranks;
        for (size_t i = 0; i < nodes->size(); ++i) {
            const auto &node = nodes->at(i);
            if (node.is_leaf()) {
                aggregator_ranks.push_back(node.aggregator_rank);
            }
        }
        tbb::parallel_sort(aggregator_ranks.begin(), aggregator_ranks.end());
        for (size_t i = 0; i < aggregator_ranks.size(); ++i) {
            aggregator_indices[aggregator_ranks[i]] = i;
        }
    }

    std::vector<uint32_t> node_bitmaps(nodes->size() * attributes.size(), 0);
    initialize_node_attributes(0,
                               remapped_bitmaps,
                               aggregator_attribute_ranges,
                               aggregator_indices,
                               node_bitmaps,
                               attr_ranges);
    node_attrib_ranges = attr_ranges;

    // Build the dictionary of all bitmap indices
    const ProfilingPoint start_dict;
    phmap::flat_hash_map<uint32_t, uint32_t> bitmap_map;

    // Add coarse level bitmaps to the map
    for (const uint32_t &m : node_bitmaps) {
        bitmap_map[m]++;
    }

    bitmap_dictionary = build_bitmap_dictionary(bitmap_map);

    const ProfilingPoint end_dict;

    auto ids = std::make_shared<FixedArray<uint16_t>>(node_bitmaps.size());
    for (size_t i = 0; i < node_bitmaps.size(); ++i) {
        ids->at(i) = bitmap_map[node_bitmaps.at(i)];
    }
    node_bitmap_ids = ids;
}

void AggregationTree::get_splitting_planes(std::vector<Plane> &planes,
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

    // Scale quality by a log scale so that 0.5 quality = 50% data. Since the
    // tree doubles the amount of data at each level we need to apply an inverse
    // mapping of this to the quality level
    quality = remap_quality(quality);

    std::array<size_t, 64> node_stack = {0};
    std::array<uint32_t, 64> depth_stack = {0};
    std::array<Box, 64> bounds_stack;
    size_t stack_idx = 0;
    size_t current_node = 0;
    uint32_t current_depth = 0;
    Box current_bounds = bounds;
    while (true) {
        const AggregationKdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            glm::vec3 plane_origin = current_bounds.center();
            plane_origin[node.split_axis()] = node.split_pos;

            glm::vec3 plane_half_vecs = current_bounds.diagonal() / 2.f;
            plane_half_vecs[node.split_axis()] = 0.f;

            planes.emplace_back(plane_origin, plane_half_vecs);

            // Interior node, descend into children if they're contained in the box
            bool traverse_left = b.lower[node.split_axis()] <= node.split_pos;
            bool traverse_right = b.upper[node.split_axis()] >= node.split_pos;
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
            if (traverse_right && traverse_left) {
                depth_stack[stack_idx] = current_depth + 1;
                node_stack[stack_idx] = node.right_child_offset();
                bounds_stack[stack_idx] = current_bounds;
                bounds_stack[stack_idx].lower[node.split_axis()] = node.split_pos;
                stack_idx++;

                current_node = current_node + 1;
                current_depth++;
                current_bounds.upper[node.split_axis()] = node.split_pos;
                continue;
            } else if (traverse_left) {
                current_node = current_node + 1;
                current_depth++;
                current_bounds.upper[node.split_axis()] = node.split_pos;
                continue;
            } else if (traverse_right) {
                current_node = node.right_child_offset();
                current_depth++;
                current_bounds.lower[node.split_axis()] = node.split_pos;
                continue;
            }
        } else {
            auto tree = fetch_tree(node.aggregator_rank);
            tree->get_splitting_planes_log(planes, b, attrib_queries, quality);
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

std::vector<size_t> AggregationTree::get_overlapped_subtree_ids(const Box &b) const
{
    std::vector<size_t> subtree_ids;
    std::array<size_t, 64> node_stack = {0};
    size_t stack_idx = 0;
    size_t current_node = 0;
    while (true) {
        const AggregationKdNode &node = nodes->at(current_node);
        if (!node.is_leaf()) {
            const bool traverse_left = b.lower[node.split_axis()] < node.split_pos;
            const bool traverse_right = b.upper[node.split_axis()] > node.split_pos;

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
            subtree_ids.push_back(node.aggregator_rank);
        }

        // Pop the stack to get the next node to traverse
        if (stack_idx > 0) {
            --stack_idx;
            current_node = node_stack[stack_idx];
        } else {
            break;
        }
    }
    return subtree_ids;
}

void AggregationTree::initialize_node_attributes(
    const size_t n,
    const std::vector<uint32_t> &aggregator_attribute_bitmaps,
    const std::vector<glm::vec2> &aggregator_attribute_ranges,
    const phmap::flat_hash_map<int, size_t> &aggregator_indices,
    std::vector<uint32_t> &node_bitmaps,
    OwnedArrayHandle<glm::vec2> &attrib_ranges)
{
    const auto &node = nodes->at(n);
    if (node.is_leaf()) {
        const size_t aggregator_index = aggregator_indices.find(node.aggregator_rank)->second;
        for (size_t i = 0; i < attributes.size(); ++i) {
            node_bitmaps[n * attributes.size() + i] =
                aggregator_attribute_bitmaps[i * num_aggregators + aggregator_index];
#ifdef BAT_STORE_NODE_RANGES
            attrib_ranges->at(n * attributes.size() + i) =
                aggregator_attribute_ranges[i * num_aggregators + aggregator_index];
#endif
        }
    } else {
        // Build the bitmaps and ranges for the children, then combine them for the parent
        initialize_node_attributes(n + 1,
                                   aggregator_attribute_bitmaps,
                                   aggregator_attribute_ranges,
                                   aggregator_indices,
                                   node_bitmaps,
                                   attrib_ranges);

        initialize_node_attributes(node.right_child_offset(),
                                   aggregator_attribute_bitmaps,
                                   aggregator_attribute_ranges,
                                   aggregator_indices,
                                   node_bitmaps,
                                   attrib_ranges);

        for (size_t i = 0; i < attributes.size(); ++i) {
            const uint32_t lmask = node_bitmaps[(n + 1) * attributes.size() + i];
            const uint32_t rmask =
                node_bitmaps[node.right_child_offset() * attributes.size() + i];

            const uint32_t parent_mask = lmask | rmask;
            node_bitmaps[n * attributes.size() + i] = parent_mask;

#ifdef BAT_STORE_NODE_RANGES
            const glm::vec2 lrange = attrib_ranges->at((n + 1) * attributes.size() + i);
            const glm::vec2 rrange =
                attrib_ranges->at(node.right_child_offset() * attributes.size() + i);

            const glm::vec2 parent_range(std::min(lrange.x, rrange.x),
                                         std::max(lrange.y, rrange.y));
            attrib_ranges->at(n * attributes.size() + i) = parent_range;
#endif
        }
    }
}

bool AggregationTree::node_overlaps_query(uint32_t n,
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

std::shared_ptr<BATree> AggregationTree::fetch_tree(const size_t tree_id)
{
    auto fnd = loaded_trees.find(tree_id);
    std::shared_ptr<BATree> tree = nullptr;
    if (fnd == loaded_trees.end()) {
        const std::string fname = bat_prefix + std::to_string(tree_id) + ".bat";
        loaded_trees[tree_id] = std::make_shared<BATree>(map_ba_tree(fname));
        tree = loaded_trees[tree_id];
    } else {
        tree = fnd->second;
    }
    tree->enable_range_filtering = enable_range_filtering;
    tree->enable_bitmap_filtering = enable_bitmap_filtering;
    return tree;
}
