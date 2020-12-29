#include <iostream>
#include "aggregation_kd_node.h"

AggregationKdNode AggregationKdNode::inner(float split_pos, AXIS split_axis)
{
    AggregationKdNode n;
    n.split_pos = split_pos;
    n.right_child = split_axis;
    return n;
}

AggregationKdNode AggregationKdNode::leaf(uint32_t nprims, uint32_t prim_offset)
{
    AggregationKdNode n;
    n.prim_indices_offset = prim_offset;
    n.num_prims = 3 | (nprims << 2);
    return n;
}

AggregationKdNode::AggregationKdNode(const AggregationKdNode &n)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
}

AggregationKdNode::AggregationKdNode() : prim_indices_offset(-1), num_prims(-1) {}

AggregationKdNode &AggregationKdNode::operator=(const AggregationKdNode &n)
{
    if (n.is_leaf()) {
        prim_indices_offset = n.prim_indices_offset;
        num_prims = n.num_prims;
    } else {
        split_pos = n.split_pos;
        right_child = n.right_child;
    }
    return *this;
}

void AggregationKdNode::set_right_child(uint32_t r)
{
    // Clear the previous right child bits before setting the value
    right_child = right_child & 0x3;
    right_child |= (r << 2);
}

uint32_t AggregationKdNode::get_num_prims() const
{
    return num_prims >> 2;
}

uint32_t AggregationKdNode::right_child_offset() const
{
    return right_child >> 2;
}

AXIS AggregationKdNode::split_axis() const
{
    return static_cast<AXIS>(num_prims & 3);
}

bool AggregationKdNode::is_leaf() const
{
    return (num_prims & 3) == 3;
}

