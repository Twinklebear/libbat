#pragma once

#include <cstdint>
#include <cstddef>
#include "box.h"

#pragma pack(push, 1)
struct KdNode {
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

    uint32_t lod_prims = -1;

    static KdNode inner(float split_pos, AXIS split_axis, uint32_t lod_prim_offset = -1);
    static KdNode leaf(uint32_t nprims, uint32_t prim_offset);

    KdNode(const KdNode &n);

    KdNode();

    KdNode &operator=(const KdNode &n);

    void set_right_child(uint32_t right_child);

    uint32_t get_num_prims() const;

    uint32_t right_child_offset() const;

    AXIS split_axis() const;

    bool is_leaf() const;

    bool has_lod_prims() const;
};
#pragma pack(pop)

