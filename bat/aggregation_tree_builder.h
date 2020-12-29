#pragma once

#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>
#include "abstract_array.h"
#include "aggregation_tree.h"
#include "box.h"
#include "owned_array.h"

struct AggregationKdBuildNode {
    union {
        // Interior node, splitting position along the axis
        float split_pos;
        // Leaf node, offset in 'primitive_indices' to its contained prims
        uint32_t prim_indices_offset;
    };
    // Used by inner and leaf, lower 2 bits used by both inner and leaf
    // nodes, for inner nodes the lower bits track the split axis,
    // for leaf nodes they indicate it's a leaf
    union {
        // Interior node, offset to its right child (with elements above
        // the splitting plane)
        uint32_t right_child;
        // Leaf node, number of primitives in the leaf
        uint32_t num_prims;
    };
    uint32_t left_child;

    static AggregationKdBuildNode inner(float split_pos, AXIS split_axis);

    static AggregationKdBuildNode leaf(uint32_t nprims, uint32_t prim_offset);

    AggregationKdBuildNode(const AggregationKdBuildNode &n);

    AggregationKdBuildNode();

    AggregationKdBuildNode &operator=(const AggregationKdBuildNode &n);

    void set_right_child(uint32_t right_child);

    uint32_t get_num_prims() const;

    uint32_t right_child_offset() const;

    AXIS split_axis() const;

    bool is_leaf() const;
};

struct RankPoint {
    int rank = -1;
    glm::vec3 pos = glm::vec3(std::numeric_limits<float>::infinity());
    uint64_t num_particles = -1;

    RankPoint(const int rank, const glm::vec3 &pos, const uint64_t num_particles);

    RankPoint() = default;
};

/* A very simple median-split kd tree
 */
struct AggregationTreeBuilder {
    std::vector<AggregationKdBuildNode> build_nodes;
    std::vector<RankPoint> primitives;
    // Scratch space used for storing the unique split test positions
    std::vector<float> split_test_positions;

    std::mutex mutex;

    uint64_t min_prims;
    const WriteOptions &options;

    AggregationTreeBuilder(std::vector<RankPoint> &points,
                           const uint64_t min_prims,
                           const WriteOptions &options);

    AggregationTree compact() const;

private:
    uint32_t build_aggre_tree(const uint32_t depth, const size_t lo, const size_t hi);

    uint32_t compact_tree(const uint32_t n,
                          OwnedArrayHandle<AggregationKdNode> &nodes,
                          OwnedArrayHandle<uint32_t> &leaf_indices) const;
};
