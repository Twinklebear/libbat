#include "radix_tree_node.h"
#include <iostream>

std::ostream &operator<<(std::ostream &os, const RadixTreeNode &n)
{
    os << "{left: " << n.left_child() << (n.left_leaf() ? " (leaf) " : " (inner) ")
       << ", right: " << n.right_child() << (n.right_leaf() ? " (leaf) " : " (inner) ")
       << ", split axis: " << n.split_axis << ", split pos: " << n.split_position << "}";
    return os;
}

