#include "bat_handle.h"
#include "bat_file.h"
#include "pbat_file.h"
#include "util.h"

BATHandle::BATHandle(const std::string &fname)
{
    if (get_file_extension(fname) == "bat") {
        ba_tree = std::make_shared<BATree>(map_ba_tree(fname));
        attributes = ba_tree->attributes;
    } else {
        pba_tree = std::make_shared<AggregationTree>(map_pba_tree(fname));
        attributes = pba_tree->attributes;
    }
}

const Box &BATHandle::bounds() const
{
    if (ba_tree) {
        return ba_tree->bounds;
    }
    return pba_tree->bounds;
}

uint64_t BATHandle::num_points() const
{
    if (ba_tree) {
        return ba_tree->num_points;
    }
    return pba_tree->num_points;
}

uint32_t BATHandle::num_aggregators() const
{
    if (ba_tree) {
        return 0;
    }
    return pba_tree->num_aggregators;
}

size_t BATHandle::radix_tree_size()
{
    if (ba_tree) {
        return ba_tree->radix_tree->size();
    }

    size_t total_radix_size = 0;
    pba_tree->iterate_aggregator_trees(
        [&](const std::shared_ptr<BATree> &t) { total_radix_size += t->radix_tree->size(); });
    return total_radix_size;
}

size_t BATHandle::radix_tree_size_bytes()
{
    return radix_tree_size() * sizeof(RadixTreeNode);
}

size_t BATHandle::bitmap_dictionary_size()
{
    if (ba_tree) {
        return ba_tree->bitmap_dictionary->size();
    }

    size_t aggregator_dictionary_size = 0;
    pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &t) {
        aggregator_dictionary_size += t->bitmap_dictionary->size();
    });
    return aggregator_dictionary_size + pba_tree->bitmap_dictionary->size();
}

size_t BATHandle::bitmap_dictionary_size_bytes()
{
    return bitmap_dictionary_size() * sizeof(uint32_t);
}

size_t BATHandle::bitmap_indices_size()
{
    if (ba_tree) {
        size_t treelet_ids_size = 0;
        ba_tree->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
            treelet_ids_size += t->node_bitmap_ids->size();
        });
        return treelet_ids_size + ba_tree->node_bitmap_ids->size();
    }

    size_t aggregator_bitmap_indices_size = 0;
    pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &bt) {
        aggregator_bitmap_indices_size += bt->node_bitmap_ids->size();
        bt->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
            aggregator_bitmap_indices_size += t->node_bitmap_ids->size();
        });
    });
    return aggregator_bitmap_indices_size + pba_tree->node_bitmap_ids->size();
}

size_t BATHandle::bitmap_indices_size_bytes()
{
    return bitmap_indices_size() * sizeof(uint16_t);
}

size_t BATHandle::node_attrib_ranges_size()
{
    if (ba_tree) {
        if (!ba_tree->node_attrib_ranges) {
            return 0;
        }
        size_t treelet_ranges_size = 0;
        ba_tree->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
            treelet_ranges_size += t->node_attrib_ranges->size();
        });
        return treelet_ranges_size + ba_tree->node_attrib_ranges->size();
    }

    if (!pba_tree->node_attrib_ranges) {
        return 0;
    }

    size_t aggregator_ranges_size = 0;
    pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &bt) {
        aggregator_ranges_size += bt->node_attrib_ranges->size();
        bt->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
            aggregator_ranges_size += t->node_attrib_ranges->size();
        });
    });
    return aggregator_ranges_size + pba_tree->node_attrib_ranges->size();
}

size_t BATHandle::node_attrib_ranges_size_bytes()
{
    return node_attrib_ranges_size() * sizeof(glm::vec2);
}

size_t BATHandle::num_treelets()
{
    if (ba_tree) {
        return ba_tree->treelet_offsets->size();
    }

    size_t total_treelets = 0;
    pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &t) {
        total_treelets += t->treelet_offsets->size();
    });
    return total_treelets;
}

size_t BATHandle::aggregation_tree_size()
{
    if (ba_tree) {
        return 0;
    }

    return pba_tree->nodes->size();
}

size_t BATHandle::aggregation_tree_size_bytes()
{
    return aggregation_tree_size() * sizeof(AggregationKdNode);
}

size_t BATHandle::total_tree_node_count()
{
    size_t num_nodes = radix_tree_size() + aggregation_tree_size();
    // Now sum up the number of treelet nodes
    if (ba_tree) {
        ba_tree->iterate_treelets(
            [&](const std::shared_ptr<BATreelet> &t) { num_nodes += t->nodes->size(); });
    } else {
        pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &bt) {
            bt->iterate_treelets(
                [&](const std::shared_ptr<BATreelet> &t) { num_nodes += t->nodes->size(); });
        });
    }
    return num_nodes;
}

size_t BATHandle::total_tree_node_bytes()
{
    size_t total_size = radix_tree_size_bytes() + aggregation_tree_size_bytes();
    // Now sum up the treelet memory
    if (ba_tree) {
        ba_tree->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
            total_size += t->nodes->size_bytes();
        });
    } else {
        pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &bt) {
            bt->iterate_treelets([&](const std::shared_ptr<BATreelet> &t) {
                total_size += t->nodes->size_bytes();
            });
        });
    }
    return total_size;
}

std::vector<float> BATHandle::aggregator_bitmap_bin_sizes(const std::string &attr)
{
    std::vector<float> bin_sizes;
    if (ba_tree) {
        return bin_sizes;
    }

    const size_t attr_id = pba_tree->attrib_ids[attr];
    pba_tree->iterate_aggregator_trees([&](const std::shared_ptr<BATree> &t) {
        const glm::vec2 agg_range = t->attributes[attr_id].range;
        bin_sizes.push_back((agg_range.y - agg_range.x) / 32.f);
    });
    return bin_sizes;
}

void BATHandle::set_enable_range_filtering(bool enable)
{
    if (ba_tree) {
        ba_tree->enable_range_filtering = enable;
    } else {
        pba_tree->enable_range_filtering = enable;
    }
}

void BATHandle::set_enable_bitmap_filtering(bool enable)
{
    if (ba_tree) {
        ba_tree->enable_bitmap_filtering = enable;
    } else {
        pba_tree->enable_bitmap_filtering = enable;
    }
}

void BATHandle::get_splitting_planes(std::vector<Plane> &planes,
                                     const Box &query_box,
                                     std::vector<AttributeQuery> *attrib_queries,
                                     float quality)
{
    if (ba_tree) {
        ba_tree->get_splitting_planes(planes, query_box, attrib_queries, quality);
    } else {
        pba_tree->get_splitting_planes(planes, query_box, attrib_queries, quality);
    }
}
