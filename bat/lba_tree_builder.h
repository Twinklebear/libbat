#pragma once

#include <memory>
#include <mutex>
#include <ostream>
#include <vector>
#include <parallel_hashmap/phmap.h>
#include "abstract_array.h"
#include "attribute.h"
#include "ba_tree.h"
#include "ba_treelet.h"
#include "box.h"
#include "owned_array.h"
#include "pcg_rng.h"
#include "radix_tree_node.h"

#define LBATREELET_MIN_PRIMS 128

/* We track the original ID of the particle so that we can re-order the attribute arrays
 * to match the kd tree once the build is done.
 */
struct LBuildPoint {
    glm::vec3 pos = glm::vec3(0);
    size_t id = -1;
    uint32_t morton_code = 0;

    LBuildPoint(const glm::vec3 &pos, size_t id, uint32_t morton_code);

    LBuildPoint() = default;

    inline operator uint32_t() const
    {
        return morton_code;
    }
};

/* Within each leaf of the coarse bottom-up built tree, we construct
 * a better kd-tree using a standard median-split approach. Each treelet
 * is built serially, however the builds are run in parallel
 */
struct LBATreelet {
    LBuildPoint *points;
    const std::vector<Attribute> *in_attribs = nullptr;
    std::vector<Attribute> attributes;
    std::vector<KdNode> nodes;
#ifdef BAT_STORE_NODE_RANGES
    std::vector<glm::vec2> attribute_ranges;
#endif
    std::vector<uint32_t> attribute_bitmaps;

    uint32_t treelet_prim_offset = 0;
    uint32_t treelet_prims = 0;
    uint32_t min_prims = LBATREELET_MIN_PRIMS;
    uint32_t max_depth;

    uint32_t treelet_depth = 0;
    uint32_t num_lod_prims = 8;

    PCGRand rng;

    LBATreelet(LBuildPoint *points,
               uint32_t treelet_prim_offset,
               uint32_t treelet_prims,
               const std::vector<Attribute> *attributes = nullptr,
               uint32_t num_lod_prims = 0);
    LBATreelet() = default;

private:
    uint32_t build(size_t lo, size_t hi, const uint32_t depth);

    uint32_t build_leaf(const size_t lo, const size_t hi);

    // Copy the attribute "attrib_id" of the primitives in [lo, hi) into the treelet's
    // attributes buffers, and return the range of the particle attributes and the bitmap for
    // that range
    void copy_attributes(const size_t lo,
                         const size_t hi,
                         const size_t attrib_id,
                         glm::vec2 &range,
                         uint32_t &bitmap);
};

/* A hybrid bottom-up/treelet tree. A coarse top-level tree is built
 * bottom-up based on `kd_morton_bits` to produce a coarse hierarchy.
 * Within each leaf cell of this tree we build a spatial median-split
 * k-d tree.
 */
struct LBATreeBuilder {
    Box tree_bounds;
    std::vector<Attribute> attribs;
    std::vector<LBuildPoint> build_points;
    // The inner nodes of the coarse bottom-up k-d tree
    std::vector<RadixTreeNode> inner_nodes;
    // Morton codes for each leaf
    std::vector<uint32_t> morton_codes;
    // The treelets built within each leaf of the coarse k-d tree
    std::vector<LBATreelet> treelets;
#ifdef BAT_STORE_NODE_RANGES
    // Attribute ranges of the coarse k-d tree nodes
    std::vector<glm::vec2> attribute_ranges;
#endif
    // Attribute bitmaps of the coarse k-d tree nodes
    std::vector<uint32_t> attribute_bitmaps;

    uint32_t kd_morton_bits = 4;
    uint32_t kd_morton_mask = 0;
    uint32_t morton_bits = 10;
    uint32_t num_lod_prims = 8;

    LBATreeBuilder(std::vector<glm::vec3> points,
                   std::vector<Attribute> attributes = std::vector<Attribute>{},
                   uint32_t num_lod_prims = 8);

    BATree compact();

private:
    void build_treelets();
};
