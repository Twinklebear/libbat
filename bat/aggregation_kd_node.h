#pragma once

#include <cstddef>
#include <cstdint>
#include "box.h"

#pragma pack(push, 1)
struct AggregationKdNode {
    union {
        // Interior node, splitting position along the axis
        float split_pos;
        // Leaf node, offset in 'primitive_indices' to its contained prims
        uint32_t prim_indices_offset;
        // Leaf node without prims, this is directly the aggregator ID
        uint32_t aggregator_rank;
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

    static AggregationKdNode inner(float split_pos, AXIS split_axis);

    static AggregationKdNode leaf(uint32_t nprims, uint32_t prim_offset);

    AggregationKdNode(const AggregationKdNode &n);

    AggregationKdNode();

    AggregationKdNode &operator=(const AggregationKdNode &n);

    void set_right_child(uint32_t right_child);

    uint32_t get_num_prims() const;

    uint32_t right_child_offset() const;

    AXIS split_axis() const;

    bool is_leaf() const;
};
#pragma pack(pop)

