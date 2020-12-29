#include "aggregation_tree_builder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_sort.h>
#include <tbb/task_group.h>
#include "profiling.h"

AggregationKdBuildNode AggregationKdBuildNode::inner(float split_pos, AXIS split_axis)
{
    AggregationKdBuildNode n;
    n.split_pos = split_pos;
    n.right_child = split_axis;
    return n;
}

AggregationKdBuildNode AggregationKdBuildNode::leaf(uint32_t nprims, uint32_t prim_offset)
{
    AggregationKdBuildNode n;
    n.prim_indices_offset = prim_offset;
    n.num_prims = 3 | (nprims << 2);
    return n;
}

AggregationKdBuildNode::AggregationKdBuildNode(const AggregationKdBuildNode &n)
    : left_child(n.left_child)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
}

AggregationKdBuildNode::AggregationKdBuildNode()
    : prim_indices_offset(0), num_prims(0), left_child(-1)
{
}

AggregationKdBuildNode &AggregationKdBuildNode::operator=(const AggregationKdBuildNode &n)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
    left_child = n.left_child;
    return *this;
}

void AggregationKdBuildNode::set_right_child(uint32_t r)
{
    right_child = right_child & 0x3;
    right_child |= (r << 2);
}

uint32_t AggregationKdBuildNode::get_num_prims() const
{
    return num_prims >> 2;
}

uint32_t AggregationKdBuildNode::right_child_offset() const
{
    return right_child >> 2;
}

AXIS AggregationKdBuildNode::split_axis() const
{
    return static_cast<AXIS>(num_prims & 3);
}

bool AggregationKdBuildNode::is_leaf() const
{
    return (num_prims & 3) == 3;
}

RankPoint::RankPoint(const int rank, const glm::vec3 &pos, const uint64_t num_particles)
    : rank(rank), pos(pos), num_particles(num_particles)
{
}

AggregationTreeBuilder::AggregationTreeBuilder(std::vector<RankPoint> &inpoints,
                                               const uint64_t min_prims,
                                               const WriteOptions &options)
    : primitives(std::move(inpoints)), min_prims(min_prims), options(options)
{
    build_nodes.reserve(2 * std::log2(primitives.size()));
    split_test_positions.resize(primitives.size(), 0.f);
    build_aggre_tree(0, 0, primitives.size());
}

AggregationTree AggregationTreeBuilder::compact() const
{
    auto nodes = std::make_shared<OwnedArray<AggregationKdNode>>();
    nodes->reserve(build_nodes.size());
    auto leaf_indices = std::make_shared<OwnedArray<uint32_t>>();

    tbb::task_group task_group;
    task_group.run([&]() { compact_tree(0, nodes, leaf_indices); });

    auto compact_prims = std::make_shared<OwnedArray<RankAggregationInfo>>(primitives.size());
    tbb::parallel_for(size_t(0), primitives.size(), [&](const size_t i) {
        compact_prims->at(i).rank = primitives[i].rank;
        compact_prims->at(i).num_particles = primitives[i].num_particles;
    });

    const uint64_t total_points = std::accumulate(
        primitives.begin(),
        primitives.end(),
        uint64_t(0),
        [](const uint64_t c, const RankPoint &prim) { return c + prim.num_particles; });

    task_group.wait();

    return AggregationTree(
        Box(),
        total_points,
        std::dynamic_pointer_cast<AbstractArray<AggregationKdNode>>(nodes),
        std::dynamic_pointer_cast<AbstractArray<uint32_t>>(leaf_indices),
        std::dynamic_pointer_cast<AbstractArray<RankAggregationInfo>>(compact_prims));
}

uint32_t AggregationTreeBuilder::build_aggre_tree(const uint32_t depth,
                                                  const size_t lo,
                                                  const size_t hi)
{
    // We've hit max depth or the prim threshold, so make a leaf
    const size_t num_prims = hi - lo;
    const size_t num_particles_total = std::accumulate(
        primitives.begin() + lo,
        primitives.begin() + hi,
        size_t(0),
        [](const size_t acc, const RankPoint &p) { return acc + p.num_particles; });

    bool make_leaf = false;
    if (options.fixed_num_aggregators == 0) {
        const size_t ranks_with_particles =
            std::accumulate(primitives.begin() + lo,
                            primitives.begin() + hi,
                            size_t(0),
                            [](const size_t acc, const RankPoint &p) {
                                return acc + (p.num_particles > 0 ? 1 : 0);
                            });
        make_leaf =
            num_particles_total <= min_prims || num_prims == 1 || ranks_with_particles == 1;
    } else {
        // Note: Because the input points are in a grid forcing a fixed number of prims
        // per-leaf will produce a larger grid, giving us a fixed aggregation pattern
        const size_t ranks_per_agg =
            std::ceil(static_cast<float>(primitives.size()) / options.fixed_num_aggregators);
        make_leaf = num_prims <= ranks_per_agg;
    }

    if (make_leaf) {
        std::lock_guard<std::mutex> lock(mutex);

        const uint32_t node_index = build_nodes.size();
        build_nodes.push_back(AggregationKdBuildNode::leaf(num_prims, lo));

        return node_index;
    }

    // We're making an interior node, find the median point and split the objects
    Box centroid_bounds;
    if (num_prims > 1024) {
        using range_type = tbb::blocked_range<std::vector<RankPoint>::iterator>;
        centroid_bounds = tbb::parallel_reduce(
            range_type(primitives.begin() + lo, primitives.begin() + hi),
            Box{},
            [&](const range_type &r, const Box &b) {
                Box res = b;
                for (auto it = r.begin(); it != r.end(); ++it) {
                    if (it->num_particles > 0 || options.fixed_num_aggregators) {
                        res.extend(it->pos);
                    }
                }
                return res;
            },
            [](const Box &a, const Box &b) { return box_union(a, b); });
    } else {
        for (size_t i = lo; i < hi; ++i) {
            if (primitives[i].num_particles > 0 || options.fixed_num_aggregators) {
                centroid_bounds.extend(primitives[i].pos);
            }
        }
    }

    // Find the point along the split axis where we can partition the rank points
    // such that each half has a roughly even number of particles
    const glm::uvec3 axis_order = centroid_bounds.axis_ordering();
    uint32_t split_axis = axis_order.x;
    size_t right_start_idx = 0;
    float current_split_cost = std::numeric_limits<float>::infinity();
    float split_pos = std::numeric_limits<float>::infinity();
    uint64_t left_particles = 0;
    uint64_t right_particles = 0;
    if (options.fixed_num_aggregators == 0) {
        const uint32_t test_axes = options.find_best_axis ? 3 : 1;
        for (uint32_t i = 0; i < test_axes; ++i) {
            const int test_split_axis = axis_order[i];

            tbb::parallel_sort(primitives.begin() + lo,
                               primitives.begin() + hi,
                               [&](const RankPoint &a, const RankPoint &b) {
                                   return a.pos[test_split_axis] < b.pos[test_split_axis];
                               });

            // Find the unique possible split positions we're going to test
            float last_pos = std::numeric_limits<float>::infinity();
            size_t last_split_idx = lo;
            for (size_t i = lo; i < hi; ++i) {
                const float test_split = primitives[i].pos[test_split_axis];
                if (last_pos != test_split) {
                    split_test_positions[last_split_idx++] = test_split;
                    last_pos = test_split;
                }
            }

            // TODO NOTE: This could be parallelized as a parallel reduction
            // Test the different possible split positions to find the best one
            for (size_t i = 0; i < last_split_idx - lo; ++i) {
                const float test_split = split_test_positions[i + lo];
                auto right_start =
                    std::upper_bound(primitives.begin() + lo,
                                     primitives.begin() + hi,
                                     test_split,
                                     [&](const float &split, const RankPoint &p) {
                                         return split <= p.pos[test_split_axis];
                                     });

                const uint64_t test_left_particles = std::accumulate(
                    primitives.begin() + lo,
                    right_start,
                    uint64_t(0),
                    [](const uint64_t n, const RankPoint &p) { return n + p.num_particles; });

                const uint64_t test_right_particles = std::accumulate(
                    right_start,
                    primitives.begin() + hi,
                    uint64_t(0),
                    [](const uint64_t n, const RankPoint &p) { return n + p.num_particles; });

                const float test_cost =
                    std::abs(0.5f - static_cast<float>(test_left_particles) /
                                        (test_left_particles + test_right_particles));

#if 0
                std::cout << std::string(depth, ' ') << "Depth " << depth << ": test split ("
                          << i << ") at " << test_split << " on " << test_split_axis
                          << " has cost " << test_cost << " w/ left, right: "
                          << " [" << test_left_particles << ", " << test_right_particles << "], {" << lo
                          << ", " << std::distance(primitives.begin(), right_start) << "}\n";
#endif

                if (test_left_particles == 0 || test_right_particles == 0) {
                    continue;
                }

                // We want a split which has the most even ratio of particles on each side
                if (test_cost < current_split_cost) {
                    current_split_cost = test_cost;
                    split_axis = test_split_axis;
                    right_start_idx = std::distance(primitives.begin(), right_start);
                    split_pos = test_split;
                    left_particles = test_left_particles;
                    right_particles = test_right_particles;
#if 0
                    std::cout << std::string(depth, ' ') << "Depth " << depth
                              << ": best split so far at " << test_split << " on "
                              << test_split_axis << " has cost " << test_cost << " ["
                              << left_particles << ", " << right_particles << "]\n";
#endif
                }
            }
        }
    } else {
        tbb::parallel_sort(primitives.begin() + lo,
                           primitives.begin() + hi,
                           [&](const RankPoint &a, const RankPoint &b) {
                               return a.pos[split_axis] < b.pos[split_axis];
                           });

        split_pos = primitives[lo + num_prims / 2].pos[split_axis];

        auto right_start = std::upper_bound(primitives.begin() + lo,
                                            primitives.begin() + hi,
                                            split_pos,
                                            [&](const float &split, const RankPoint &p) {
                                                return split <= p.pos[split_axis];
                                            });
        right_start_idx = std::distance(primitives.begin(), right_start);
        current_split_cost = 0.f;
    }

    // If we'd make a really bad split and are within some threshold of the target
    // leaf size, just make a leaf
    const bool force_leaf =
        (current_split_cost >= options.max_split_imbalance_ratio &&
         num_particles_total <= min_prims * options.max_overfull_aggregator_factor) ||
        left_particles == 1 || right_particles == 1;

    if (options.fixed_num_aggregators == 0 && force_leaf) {
        std::lock_guard<std::mutex> lock(mutex);

        const uint32_t node_index = build_nodes.size();
        build_nodes.push_back(AggregationKdBuildNode::leaf(num_prims, lo));
        return node_index;
    }

    if (options.find_best_axis && split_axis != axis_order[2]) {
        // Now we need to re-sort along the best split axis we found
        tbb::parallel_sort(primitives.begin() + lo,
                           primitives.begin() + hi,
                           [&](const RankPoint &a, const RankPoint &b) {
                               return a.pos[split_axis] < b.pos[split_axis];
                           });
    }

    // Ranges within points array for the left/right child
    const size_t left_lo = lo;
    const size_t left_hi = right_start_idx;

    const size_t right_lo = left_hi;
    const size_t right_hi = hi;

    uint32_t inner_idx = -1;
    {
        std::lock_guard<std::mutex> lock(mutex);

        inner_idx = build_nodes.size();
        build_nodes.push_back(AggregationKdBuildNode::inner(split_pos, AXIS(split_axis)));
    }

    // Spawn off task to build the right side and do the left on this thread
    uint32_t right_child = -1;
    tbb::task_group task_group;
    task_group.run([&]() { right_child = build_aggre_tree(depth + 1, right_lo, right_hi); });
    const uint32_t left_child = build_aggre_tree(depth + 1, left_lo, left_hi);
    task_group.wait();

    {
        std::lock_guard<std::mutex> lock(mutex);
        build_nodes[inner_idx].left_child = left_child;
        build_nodes[inner_idx].set_right_child(right_child);
    }

    return inner_idx;
}

uint32_t AggregationTreeBuilder::compact_tree(const uint32_t n,
                                              OwnedArrayHandle<AggregationKdNode> &nodes,
                                              OwnedArrayHandle<uint32_t> &leaf_indices) const
{
    const auto &bn = build_nodes[n];
    const uint32_t index = nodes->size();

    if (!bn.is_leaf()) {
        nodes->push_back(AggregationKdNode::inner(bn.split_pos, bn.split_axis()));

        compact_tree(bn.left_child, nodes, leaf_indices);

        const uint32_t right_child =
            compact_tree(bn.right_child_offset(), nodes, leaf_indices);
        nodes->at(index).set_right_child(right_child);
    } else {
        nodes->push_back(AggregationKdNode::leaf(bn.get_num_prims(), bn.prim_indices_offset));
        leaf_indices->push_back(index);
    }
    return index;
}

