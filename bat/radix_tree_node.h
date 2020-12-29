#pragma once

#include <iostream>
#include "box.h"

class RadixTreeNode {
    // low 2 bits store if the left/right child is a leaf (bits 0 and 1 respectively)
    uint64_t node_info = 0;

public:
    float split_position = 0;
    AXIS split_axis = X;

    inline uint64_t gamma() const
    {
        return node_info >> 2;
    }

    inline void set_gamma(uint64_t g)
    {
#ifdef DEBUG
        if (g & 0xc0000000) {
            std::cout << "high 2 bits of gamma are set!\n";
            throw std::runtime_error("High 2 bits of gamma are set, tree won't be valid!");
        }
#endif
        node_info = (node_info & 0x3) | (g << 2);
    }

    inline uint64_t left_child() const
    {
        return gamma();
    }

    inline uint64_t right_child() const
    {
        return gamma() + 1;
    }

    inline bool left_leaf() const
    {
        return node_info & 0x1;
    }

    inline bool right_leaf() const
    {
        return node_info & 0x2;
    }

    inline void set_left_leaf(bool is_leaf)
    {
        if (is_leaf) {
            node_info = node_info | 0x1;
        } else {
            node_info = node_info & 0xfffffffe;
        }
    }

    inline void set_right_leaf(bool is_leaf)
    {
        if (is_leaf) {
            node_info = node_info | 0x2;
        } else {
            node_info = node_info & 0xfffffffd;
        }
    }
};

std::ostream &operator<<(std::ostream &os, const RadixTreeNode &n);
