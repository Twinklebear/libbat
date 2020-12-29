#include "lba_tree_builder.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <numeric>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <parallel_hashmap/phmap.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>
#include "borrowed_array.h"
#include "byte_cursor.h"
#include "fixed_array.h"
#include "owned_array.h"
#include "profiling.h"
#include "util.h"

LBuildPoint::LBuildPoint(const glm::vec3 &pos, size_t id, uint32_t morton_code)
    : pos(pos), id(id), morton_code(morton_code)
{
}

LBATreelet::LBATreelet(LBuildPoint *points,
                       uint32_t treelet_prim_offset,
                       uint32_t treelet_prims,
                       const std::vector<Attribute> *in_attributes,
                       uint32_t num_lod_prims)
    : points(points),
      in_attribs(in_attributes),
      treelet_prim_offset(treelet_prim_offset),
      treelet_prims(treelet_prims),
      max_depth(8 + 1.3 * std::log2(treelet_prims)),
      num_lod_prims(num_lod_prims),
      rng(treelet_prim_offset)
{
    if (in_attribs) {
        for (const auto &a : *in_attribs) {
            auto arr = std::make_shared<FixedArray<uint8_t>>(a.desc.stride() * treelet_prims);
            attributes.push_back(Attribute(a.desc, arr));
        }
    }
    build(treelet_prim_offset, treelet_prim_offset + treelet_prims, 0);
}

uint32_t LBATreelet::build(size_t lo, size_t hi, const uint32_t depth)
{
    const size_t num_prims = hi - lo;
    treelet_depth = std::max(treelet_depth, depth);

    // If taking the LOD prims for the inner node would push us under the leaf threshold, just
    // make a leaf
    if (num_prims <= min_prims + num_lod_prims || depth >= max_depth) {
        return build_leaf(lo, hi);
    }

    Box centroid_bounds;
    for (auto it = points + lo; it != points + hi; ++it) {
        centroid_bounds.extend(it->pos);
    }

    uint32_t inner_node_lod_offset = -1;
    if (num_lod_prims > 0) {
        // Now we pick out N LOD points to use from the [lo, hi) set of points at this
        // inner node, and set those aside to provide the LOD representatives for this node.
        // Then recurse on constructing the child nodes over the [lo + N, hi) range and store
        // the offset to the inner node's LOD particles with the node. Note: we know num_prims
        // > min_prims + num_lod_prims, otherwise we would have made a leaf
        const size_t lod_point_stride = num_prims / num_lod_prims;
        // Treat the lod_point_stride as a set of buckets on the points and
        // randomly choose our LOD points from those within the buckets, to get a
        // stratified sampling
        for (size_t i = 0; i < num_lod_prims; ++i) {
            size_t l = rng.randomf() * lod_point_stride + i * lod_point_stride + lo;
            l = clamp(l, lo, hi - 1);
            std::swap(points[lo + i], points[l]);
        }
        inner_node_lod_offset = lo;

        // The child nodes will partition the remainder points
        lo += num_lod_prims;
    }

    // Find the median-split position, retrying if we have some weird particle configuration.
    // It seems like in very rare cases this can happen (e.g., the Uintah data set), where a
    // lot particles are all clumped right on the boundary so the median is actually at the
    // lower/upper edge of the box. Try the next longest axis, then the smallest
    const glm::uvec3 axis_order = centroid_bounds.axis_ordering();
    int split_axis = axis_order.x;
    float split_pos = 0;
    for (uint32_t i = 0; i < 3; ++i) {
        split_axis = axis_order[i];

        std::sort(points + lo, points + hi, [&](const LBuildPoint &a, const LBuildPoint &b) {
            return a.pos[split_axis] < b.pos[split_axis];
        });
        split_pos = points[lo + num_prims / 2].pos[split_axis];

        if (split_pos != centroid_bounds.lower[split_axis] &&
            split_pos != centroid_bounds.upper[split_axis]) {
            break;
        }
    }

    auto right_start = std::upper_bound(
        points + lo, points + hi, split_pos, [&](const float &split, const LBuildPoint &p) {
            return split <= p.pos[split_axis];
        });

    // Ranges within points array for the left/right child
    const size_t left_lo = lo;
    const size_t left_hi = std::distance(points, right_start);

    const size_t right_lo = left_hi;
    const size_t right_hi = hi;

    const uint32_t inner_idx = nodes.size();
    // Node primitive references should be local to the treelet
    nodes.push_back(KdNode::inner(split_pos,
                                  static_cast<AXIS>(split_axis),
                                  inner_node_lod_offset - treelet_prim_offset));

    // Reserve the attribute range indices in the array for the inner node's ranges, and fill
    // with the values from our LOD prims, if we have them
    if (in_attribs) {
        for (size_t i = 0; i < in_attribs->size(); ++i) {
            // const glm::vec2 range = (*in_attribs)[i].range;
            glm::vec2 range;
            uint32_t bitmap;
            if (num_lod_prims > 0) {
                copy_attributes(inner_node_lod_offset, lo, i, range, bitmap);
            }
#ifdef BAT_STORE_NODE_RANGES
            attribute_ranges.push_back(range);
#endif
            attribute_bitmaps.push_back(bitmap);
        }
    }

    // Build the left child following the inner node, and the right node after the left subtree
    build(left_lo, left_hi, depth + 1);
    const uint32_t right_child = build(right_lo, right_hi, depth + 1);
    nodes[inner_idx].set_right_child(right_child);

    // Merge the ranges of the child nodes to the parent
    if (in_attribs) {
        // The inner node stores additional LOD primitives which the leaves don't
        // know about, so we have to merge their values in with those from the children
        const size_t parent_start = inner_idx * in_attribs->size();
        const size_t left_start = (inner_idx + 1) * in_attribs->size();
        const size_t right_start = right_child * in_attribs->size();
        for (size_t i = 0; i < in_attribs->size(); ++i) {
#ifdef BAT_STORE_NODE_RANGES
            glm::vec2 &r = attribute_ranges[parent_start + i];

            const glm::vec2 &left = attribute_ranges[left_start + i];
            const glm::vec2 &right = attribute_ranges[right_start + i];

            r.x = std::min(left.x, std::min(right.x, r.x));
            r.y = std::max(left.y, std::max(right.y, r.y));
#endif

            const uint32_t bitmap = attribute_bitmaps[left_start + i] |
                                    attribute_bitmaps[right_start + i] |
                                    attribute_bitmaps[parent_start + i];
            attribute_bitmaps[parent_start + i] = bitmap;
        }
    }
    return inner_idx;
}

uint32_t LBATreelet::build_leaf(const size_t lo, const size_t hi)
{
    const uint32_t num_prims = hi - lo;
    const uint32_t index = nodes.size();
    // We won't merge the points back together, so the leaf node offsets should
    // be local to this treelet's set
    nodes.push_back(KdNode::leaf(num_prims, lo - treelet_prim_offset));

    // Now re-order the attributes and compute our min/max range info for the leaf, which
    // will be propagated up the treelet to the parent.
    if (in_attribs) {
        for (size_t i = 0; i < in_attribs->size(); ++i) {
            glm::vec2 range;
            uint32_t bitmap;
            copy_attributes(lo, hi, i, range, bitmap);
#ifdef BAT_STORE_NODE_RANGES
            attribute_ranges.push_back(range);
#endif
            attribute_bitmaps.push_back(bitmap);
        }
    }
    return index;
}

void LBATreelet::copy_attributes(const size_t lo,
                                 const size_t hi,
                                 const size_t attrib_id,
                                 glm::vec2 &range,
                                 uint32_t &bitmap)
{
    range = glm::vec2(std::numeric_limits<float>::infinity(),
                      -std::numeric_limits<float>::infinity());
    bitmap = 0;

    const auto &inattr = *in_attribs;
    const size_t stride = inattr[attrib_id].desc.stride();
    for (uint32_t j = lo; j < hi; ++j) {
        std::memcpy(attributes[attrib_id].at(j - treelet_prim_offset),
                    inattr[attrib_id].at(points[j].id),
                    stride);

        const float x = attributes[attrib_id].as_float(j - treelet_prim_offset);
        range.x = std::min(static_cast<float>(x), range.x);
        range.y = std::max(static_cast<float>(x), range.y);
    }

    for (uint32_t j = lo; j < hi; ++j) {
        bitmap |= attributes[attrib_id].bitmap(j - treelet_prim_offset);
    }
}

LBATreeBuilder::LBATreeBuilder(std::vector<glm::vec3> points,
                               std::vector<Attribute> in_attributes,
                               uint32_t num_lod_prims)
    : attribs(in_attributes), num_lod_prims(num_lod_prims)
{
    using namespace std::chrono;
    const ProfilingPoint start;

    tree_bounds = compute_bounds(points.begin(), points.end());

    // Pre-compute the global range, which will be used by all nodes to compute the bitmap
    // indices over. This lets us discard the range at each node, and use a shared dictionary
    // of compressed bitmap indices (though we could also do the latter based on the bit
    // pattern even if we had different ranges).
    const ProfilingPoint start_attr_range_comp;
    tbb::parallel_for(size_t(0), attribs.size(), [&](const size_t i) {
        auto &attr = attribs[i];
        if (!attr.desc.has_range()) {
            attr.desc.range = attr.compute_range();
        }
    });
    const ProfilingPoint end_attr_range_comp;
#if 0
    std::cout << "attribute range computation took: "
              << elapsed_time_ms(start_attr_range_comp, end_attr_range_comp)
              << "ms, CPU: " << cpu_utilization(start_attr_range_comp, end_attr_range_comp)
              << "%\n";
#endif

    // Compute Morton codes for each point
    const uint32_t morton_grid_size = (1 << morton_bits) - 1;
    // TODO: maybe use fixed array to cut init cost
    build_points.resize(points.size(), LBuildPoint());
    tbb::parallel_for(size_t(0), points.size(), [&](const size_t i) {
        const glm::vec3 p =
            morton_grid_size * ((points[i] - tree_bounds.lower) / tree_bounds.diagonal());
        const uint32_t morton_code = encode_morton32(p.x, p.y, p.z);
        build_points[i] = LBuildPoint(points[i], i, morton_code);
    });

    tbb::parallel_sort(build_points.begin(),
                       build_points.end(),
                       [&](const LBuildPoint &a, const LBuildPoint &b) {
                           return a.morton_code < b.morton_code;
                       });

    // We build the tree over fewer morton bits to build a coarse tree bottom-up, then build
    // better kd-treelets within the leaves of the tree in parallel
    const ProfilingPoint start_unique_find;
    // TODO: This could be a parallel compaction, but doesn't seem to be very expensive even
    // serially
    kd_morton_mask = 0xffffffff << (morton_bits - kd_morton_bits) * 3;
    morton_codes.push_back(build_points[0].morton_code & kd_morton_mask);
    for (size_t i = 1; i < build_points.size(); ++i) {
        const uint32_t key = build_points[i].morton_code & kd_morton_mask;
        if (morton_codes.back() != key) {
            morton_codes.push_back(key);
        }
    }

    // Check that we have a good balance of coarseness in the coarse tree and depth in the
    // treelets and if not coarsen the top-level further
    float avg_prims_per_treelet =
        static_cast<float>(build_points.size()) / morton_codes.size();
    if (build_points.size() > 2.f * LBATREELET_MIN_PRIMS &&
        avg_prims_per_treelet < 2.f * LBATREELET_MIN_PRIMS) {
        for (int i = kd_morton_bits - 1; i > 0; --i) {
            kd_morton_bits = i;
            kd_morton_mask = 0xffffffff << (morton_bits - kd_morton_bits) * 3;
            auto end = std::unique(morton_codes.begin(),
                                   morton_codes.end(),
                                   [&](const uint32_t a, const uint32_t b) {
                                       return (a & kd_morton_mask) == (b & kd_morton_mask);
                                   });
            morton_codes.erase(end, morton_codes.end());
            avg_prims_per_treelet =
                static_cast<float>(build_points.size()) / morton_codes.size();
            if (avg_prims_per_treelet > 2.f * LBATREELET_MIN_PRIMS) {
                break;
            }
        }
    }
    const ProfilingPoint end_unique_find;

    const size_t num_unique_keys = morton_codes.size();
    if (num_unique_keys == 1) {
        std::cout << "[ERROR] Too few particles for tree! Write will succeed but file will be "
                     "unreadable\n";
    }
    const size_t num_inner_nodes = num_unique_keys == 1 ? 1 : num_unique_keys - 1;

#if 0
    std::cout << "# of treelets: " << num_unique_keys << " with " << kd_morton_bits
              << " coarse morton bits\n"
              << "unique find step took: "
              << elapsed_time_ms(start_unique_find, end_unique_find)
              << "ms, CPU: " << cpu_utilization(start_unique_find, end_unique_find) << "%\n"
              << "Avg. # prims/treelet: "
              << static_cast<float>(build_points.size()) / num_unique_keys << "\n";
#endif

    const auto delta = [&](const int i, const int j) {
        if (j < 0 || j > num_inner_nodes) {
            return -1;
        }
        return int(longest_prefix(morton_codes[i], morton_codes[j]));
    };

    // Build the treelets for each leaf node we'll produce of the coarse k-d tree, in
    // parallel with the coarse k-d tree construction
    auto treelets_task = std::async(std::launch::async, [&]() { build_treelets(); });

    // TODO: make fixedarray?
    inner_nodes.resize(num_inner_nodes, RadixTreeNode{});

    // TODO: maybe use fixedarray to cut init cost
    std::vector<uint32_t> parent_pointers(num_inner_nodes,
                                          std::numeric_limits<uint32_t>::max());
    std::vector<uint32_t> leaf_parent_pointers(num_unique_keys,
                                               std::numeric_limits<uint32_t>::max());

    // Build the inner nodes of our coarse k-d tree
    tbb::parallel_for(size_t(0), num_inner_nodes, [&](const size_t i) {
        const int dir = (delta(i, i + 1) - delta(i, i - 1)) >= 0 ? 1 : -1;
        const int delta_min = delta(i, i - dir);

        // Find upper bound for the length of the range
        int l_max = 2;
        while (delta(i, i + l_max * dir) > delta_min) {
            l_max *= 2;
        }

        // Find the other end of the range using binary search
        int l = 0;
        for (int t = l_max / 2; t > 0; t /= 2) {
            if (delta(i, i + (l + t) * dir) > delta_min) {
                l += t;
            }
        }
        const size_t j = i + l * dir;

        // Find the split position of the node in the array
        const int delta_node = delta(i, j);

        int s = 0;
        for (int t = std::ceil(l / 2.f); t > 0; t = std::ceil(t / 2.f)) {
            if (delta(i, i + (s + t) * dir) > delta_node) {
                s += t;
            }
            if (t == 1) {
                break;
            }
        }
        const size_t gamma = i + s * dir + std::min(dir, 0);

        // Write the child pointers for this inner node
        inner_nodes[i].set_gamma(gamma);
        if (std::min(i, j) == gamma) {
            inner_nodes[i].set_left_leaf(true);
            leaf_parent_pointers[gamma] = i;
        } else {
            inner_nodes[i].set_left_leaf(false);
            parent_pointers[gamma] = i;
        }

        if (std::max(i, j) == gamma + 1) {
            inner_nodes[i].set_right_leaf(true);
            leaf_parent_pointers[gamma + 1] = i;
        } else {
            inner_nodes[i].set_right_leaf(false);
            parent_pointers[gamma + 1] = i;
        }

        // This split plane computation is definitely wrong
        const uint32_t node_prefix = morton_codes[i] & (0xffffffff << (32 - delta_node));
        const uint32_t partition_prefix = node_prefix | (0x1 << (32 - delta_node - 1));

        // Compute split plane position based on the prefix
        uint32_t sx, sy, sz;
        decode_morton32(partition_prefix, sx, sy, sz);
        uint32_t morton_split_pos = 0;
        if (delta_node % 3 == 0) {
            morton_split_pos = sy;
            inner_nodes[i].split_axis = Y;
        } else if (delta_node % 3 == 1) {
            morton_split_pos = sx;
            inner_nodes[i].split_axis = X;
        } else if (delta_node % 3 == 2) {
            morton_split_pos = sz;
            inner_nodes[i].split_axis = Z;
        }

        inner_nodes[i].split_position =
            tree_bounds.diagonal()[inner_nodes[i].split_axis] *
                (morton_split_pos / static_cast<float>(morton_grid_size)) +
            tree_bounds.lower[inner_nodes[i].split_axis];
    });

    // Sync for the treelets to finish
    treelets_task.wait();

    // Now propagate the attribute min/max values up the tree to the root
    std::vector<std::atomic<uint32_t>> node_touched(num_inner_nodes);
    // Can't do a copy ctor for the atomics, we've got to go init them all
    for (auto &v : node_touched) {
        v = 0;
    }

    const uint32_t num_attribs = attribs.size();
#ifdef BAT_STORE_NODE_RANGES
    attribute_ranges.resize(num_attribs * inner_nodes.size());
#endif
    attribute_bitmaps.resize(num_attribs * inner_nodes.size());
    tbb::parallel_for(size_t(0), leaf_parent_pointers.size(), [&](const size_t i) {
        uint32_t cur_node = i;
        uint32_t parent = leaf_parent_pointers[cur_node];
        do {
            uint32_t count = node_touched[parent].fetch_add(1);
            // If we're the second to reach this node, compute its attribute ranges by
            // combining the children
            if (count == 1) {
                const RadixTreeNode &node = inner_nodes[parent];
                for (uint32_t j = 0; j < num_attribs; ++j) {
                    glm::vec2 left_range, right_range;
                    uint32_t left_bitmap, right_bitmap;
                    if (node.left_leaf()) {
                        left_bitmap = treelets[node.left_child()].attribute_bitmaps[j];
#ifdef BAT_STORE_NODE_RANGES
                        left_range = treelets[node.left_child()].attribute_ranges[j];
#endif
                    } else {
                        left_bitmap = attribute_bitmaps[node.left_child() * num_attribs + j];
#ifdef BAT_STORE_NODE_RANGES
                        left_range = attribute_ranges[node.left_child() * num_attribs + j];
#endif
                    }
                    if (node.right_leaf()) {
                        right_bitmap = treelets[node.right_child()].attribute_bitmaps[j];
#ifdef BAT_STORE_NODE_RANGES
                        right_range = treelets[node.right_child()].attribute_ranges[j];
#endif
                    } else {
                        right_bitmap = attribute_bitmaps[node.right_child() * num_attribs + j];
#ifdef BAT_STORE_NODE_RANGES
                        right_range = attribute_ranges[node.right_child() * num_attribs + j];
#endif
                    }

#ifdef BAT_STORE_NODE_RANGES
                    glm::vec2 parent_range(std::min(left_range.x, right_range.x),
                                           std::max(left_range.y, right_range.y));
                    attribute_ranges[parent * num_attribs + j].x = parent_range.x;
                    attribute_ranges[parent * num_attribs + j].y = parent_range.y;
#endif

                    const uint32_t bitmap = left_bitmap | right_bitmap;
                    attribute_bitmaps[parent * num_attribs + j] = bitmap;
                }
                cur_node = parent;
                parent = parent_pointers[cur_node];
            } else {
                // If we're the first to reach the node the work below this one is incomplete
                // and we terminate
                break;
            }
            // The root will have no parent, so terminate once we've processed it
        } while (parent != std::numeric_limits<uint32_t>::max());
    });

#if 0
    float dbg_avg_treelet_depth = std::accumulate(
        treelets.begin(), treelets.end(), 0.f, [](float f, const LBATreelet &t) {
            return f + t.treelet_depth;
        });
    const uint32_t max_treelet_depth = std::accumulate(
        treelets.begin(), treelets.end(), 0, [](uint32_t d, const LBATreelet &t) {
            return std::max(d, t.treelet_depth);
        });

    const size_t coarse_tree_depth =
        std::ceil(std::log2(static_cast<float>(num_unique_keys))) + 1;

    std::cout << "Avg. # prims/treelet: "
              << static_cast<float>(build_points.size()) / num_unique_keys << "\n"
              << "Avg. treelet depth: " << dbg_avg_treelet_depth / treelets.size() << "\n"
              << "Max treelet depth: " << max_treelet_depth << "\n"
              << "# unique keys (i.e., leaves, treelets): " << num_unique_keys << "\n"
              << "Coarse tree depth (Min LOD): " << coarse_tree_depth << "\n"
              << "Max tree depth (Max LOD): " << coarse_tree_depth + max_treelet_depth + 1
              << "\n";
#endif

#if 0
    const ProfilingPoint end;
    const size_t build_duration = elapsed_time_ms(start, end);
    std::cout << "BA Tree build on " << build_points.size() << " points took "
              << build_duration << "ms (" << build_points.size() / (build_duration * 1e-3f)
              << "prim/s), CPU: " << cpu_utilization(start, end) << "%\n";
#endif
}

BATree LBATreeBuilder::compact()
{
    const ProfilingPoint start;

    size_t total_tree_mem = inner_nodes.size() * sizeof(RadixTreeNode) +
                            treelets.size() * sizeof(uint64_t) +
                            attribute_bitmaps.size() * sizeof(uint16_t);

#ifdef BAT_STORE_NODE_RANGES
    total_tree_mem += attribute_ranges.size() * sizeof(glm::vec2);
#endif

    // Build the dictionary of all bitmap indices
    const ProfilingPoint start_dict;
    phmap::flat_hash_map<uint32_t, uint32_t> bitmap_map;

    // Add coarse level bitmaps to the map
    for (const uint32_t &m : attribute_bitmaps) {
        bitmap_map[m]++;
    }
    // Add the treelet node's bitmaps
    for (const auto &t : treelets) {
        for (const uint32_t &m : t.attribute_bitmaps) {
            bitmap_map[m]++;
        }
    }

    auto bitmap_dictionary = build_bitmap_dictionary(bitmap_map);

    const ProfilingPoint end_dict;
#if 0
    std::cout << "Bitmap Dictionary build took " << elapsed_time_ms(start_dict, end_dict)
              << "ms\n";
#endif

    total_tree_mem += bitmap_dictionary->size_bytes();

    auto treelet_offsets = std::make_shared<FixedArray<uint64_t>>(treelets.size());
    for (size_t i = 0; i < treelets.size(); ++i) {
        const auto &t = treelets[i];
        // Treelets are aligned to 4k pages
        total_tree_mem = align_to(total_tree_mem, 4096);

        treelet_offsets->at(i) = total_tree_mem;

        size_t treelet_size = sizeof(BATreeletHeader) + t.nodes.size() * sizeof(KdNode) +
                              t.attribute_bitmaps.size() * sizeof(uint16_t) +
                              t.treelet_prims * sizeof(glm::vec3);

#ifdef BAT_STORE_NODE_RANGES
        treelet_size += t.attribute_ranges.size() * sizeof(glm::vec2);
#endif

        for (const auto &attr : t.attributes) {
            treelet_size += attr.data->size_bytes();
        }
        total_tree_mem += treelet_size;
    }

    // TODO: On the write side how this is setup we will make a second copy of the data,
    // instead we could write it directly from the lbatreebuilder. I could build the BATree
    // with a null tree_mem, where the different buffers actually own the memory referenced,
    // instead of being views into the single buffer
#if 0
    std::cout << "Tree occupies a total of " << format_bytes(total_tree_mem) << " ("
              << total_tree_mem << ")\n";
#endif
    auto tree_buffer = std::make_shared<FixedArray<uint8_t>>(total_tree_mem);

    ArrayHandle<RadixTreeNode> radix_nodes_view = nullptr;
    ArrayHandle<uint64_t> treelet_offsets_view = nullptr;
    ArrayHandle<uint16_t> radix_tree_bitmap_ids_view = nullptr;
    ArrayHandle<uint32_t> bitmap_dictionary_view = nullptr;
    ArrayHandle<glm::vec2> radix_tree_ranges_view = nullptr;
    uint32_t max_treelet_depth = 0;

    tbb::task_group root_compact;
    root_compact.run([&]() {
        ByteCursor cursor(tree_buffer->data());

        // Copy the coarse radix tree nodes and make the view for the tree
        radix_nodes_view = std::make_shared<BorrowedArray<RadixTreeNode>>(
            reinterpret_cast<RadixTreeNode *>(cursor.position()), inner_nodes.size());
        cursor.write(inner_nodes.data(), inner_nodes.size());

        // Copy the leaves of the coarse tree, pointers to treelets in the buffer
        // and make a view of them for the tree we'll return
        treelet_offsets_view = std::make_shared<BorrowedArray<uint64_t>>(
            reinterpret_cast<uint64_t *>(cursor.position()), treelet_offsets->size());
        cursor.write(treelet_offsets->begin(), treelet_offsets->size());

        // Copy the bitmap dictionary into the buffer
        bitmap_dictionary_view = std::make_shared<BorrowedArray<uint32_t>>(
            reinterpret_cast<uint32_t *>(cursor.position()), bitmap_dictionary->size());
        cursor.write(bitmap_dictionary->data(), bitmap_dictionary->size());

        // Copy the bitmap index ids into the buffer
        radix_tree_bitmap_ids_view = std::make_shared<BorrowedArray<uint16_t>>(
            reinterpret_cast<uint16_t *>(cursor.position()), attribute_bitmaps.size());
        for (size_t i = 0; i < attribute_bitmaps.size(); ++i) {
            const uint16_t id = bitmap_map[attribute_bitmaps[i]];
            cursor.write(id);
        }

#ifdef BAT_STORE_NODE_RANGES
        // Copy the min/max ranges into the buffer if we have them
        radix_tree_ranges_view = std::make_shared<BorrowedArray<glm::vec2>>(
            reinterpret_cast<glm::vec2 *>(cursor.position()), attribute_ranges.size());
        cursor.write(attribute_ranges.data(), attribute_ranges.size());
#endif

        max_treelet_depth = std::accumulate(
            treelets.begin(), treelets.end(), 0, [](uint32_t d, const LBATreelet &t) {
                return std::max(d, t.treelet_depth);
            });
    });

    // Copy the treelets into the buffer
    tbb::parallel_for(size_t(0), treelets.size(), [&](const size_t i) {
        const auto &t = treelets.at(i);
        ByteCursor cursor(tree_buffer->begin() + treelet_offsets->at(i));

        // Write the nodes/prims header
        const BATreeletHeader header(t.nodes.size(), t.treelet_prims);
        cursor.write(header);

        // Copy the treelet's kd-nodes
        cursor.write(t.nodes.data(), t.nodes.size());

        // Build and copy the treelet's bitmap ids
        for (size_t i = 0; i < t.attribute_bitmaps.size(); ++i) {
            const uint16_t id = bitmap_map[t.attribute_bitmaps[i]];
            cursor.write(id);
        }

#ifdef BAT_STORE_NODE_RANGES
        // Copy the min/max ranges into the buffer if we have them
        cursor.write(t.attribute_ranges.data(), t.attribute_ranges.size());
#endif

        // Copy the treelet primitives (TODO: re-order the LOD primitives?)
        for (size_t j = 0; j < t.treelet_prims; ++j) {
            cursor.write(t.points[j + t.treelet_prim_offset].pos);
        }

        for (const auto &attr : t.attributes) {
            cursor.write(attr.data->data(), attr.data->size());
        }
    });
    root_compact.wait();

    const size_t coarse_tree_depth =
        std::ceil(std::log2(static_cast<float>(treelets.size()))) + 1;
    const size_t max_tree_depth = coarse_tree_depth + max_treelet_depth;

    std::vector<AttributeDescription> attribute_descriptors;
    std::transform(attribs.begin(),
                   attribs.end(),
                   std::back_inserter(attribute_descriptors),
                   [](const Attribute &a) { return a.desc; });

#if 0
    const ProfilingPoint end;
    const size_t compact_duration = elapsed_time_ms(start, end);
    std::cout << "BA Tree compaction on took " << compact_duration
              << "ms, CPU: " << cpu_utilization(start, end) << "%\n";
#endif

    return BATree(tree_bounds,
                  attribute_descriptors,
                  std::dynamic_pointer_cast<AbstractArray<uint8_t>>(tree_buffer),
                  radix_nodes_view,
                  treelet_offsets_view,
                  bitmap_dictionary_view,
                  radix_tree_bitmap_ids_view,
                  radix_tree_ranges_view,
                  build_points.size(),
                  num_lod_prims,
                  coarse_tree_depth,
                  max_tree_depth);
}

void LBATreeBuilder::build_treelets()
{
    treelets.resize(morton_codes.size());
    tbb::parallel_for(size_t(0), morton_codes.size(), [&](const size_t i) {
        // Find the points falling into the leaf's bin
        const uint32_t leaf_bin = morton_codes[i] & kd_morton_mask;
        auto prims = std::equal_range(build_points.begin(),
                                      build_points.end(),
                                      leaf_bin,
                                      [&](const uint32_t &a, const uint32_t &b) {
                                          return (a & kd_morton_mask) < (b & kd_morton_mask);
                                      });

        const size_t treelet_prims = std::distance(prims.first, prims.second);
        const size_t lo = std::distance(build_points.begin(), prims.first);

        treelets[i] =
            LBATreelet(build_points.data(), lo, treelet_prims, &attribs, num_lod_prims);
    });
}
