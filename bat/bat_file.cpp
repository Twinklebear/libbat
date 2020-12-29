#include "bat_file.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <glm/ext.hpp>
#include "abstract_array.h"
#include "attribute.h"
#include "data_type.h"
#include "file_mapping.h"
#include "json.hpp"
#include "owned_array.h"
#include "util.h"

using json = nlohmann::json;

size_t write_ba_tree(const std::string &fname, const BATree &tree)
{
    std::ofstream fout(fname.c_str(), std::ios::binary);

    json header;
    header["bounds"] = {tree.bounds.lower.x,
                        tree.bounds.lower.y,
                        tree.bounds.lower.z,
                        tree.bounds.upper.x,
                        tree.bounds.upper.y,
                        tree.bounds.upper.z};
    header["num_nodes"] = tree.radix_tree->size();
    header["num_treelets"] = tree.treelet_offsets->size();
    header["bitmap_dict_size"] = tree.bitmap_dictionary->size();
    header["num_points"] = tree.num_points;
    header["num_lod_prims"] = tree.num_lod_prims;
    header["min_lod"] = tree.min_lod;
    header["max_lod"] = tree.max_lod;

    if (tree.node_attrib_ranges) {
        header["has_ranges"] = true;
    }

    for (size_t i = 0; i < tree.attributes.size(); ++i) {
        const auto &a = tree.attributes[i];
        header["attributes"][i]["name"] = a.name;
        header["attributes"][i]["dtype"] = static_cast<int>(a.data_type);
        header["attributes"][i]["range"][0] = a.range.x;
        header["attributes"][i]["range"][1] = a.range.y;
    }

    const std::string json_header = header.dump();

    // We don't include the null terminator
    const uint64_t json_header_size = json_header.size();
    fout.write(reinterpret_cast<const char *>(&json_header_size), sizeof(uint64_t));
    fout.write(json_header.c_str(), json_header_size);

    // Place the tree at the next 4k page following the header
    fout.seekp(align_to(fout.tellp(), 4096));
    fout.write(reinterpret_cast<const char *>(tree.tree_mem->data()),
               tree.tree_mem->size_bytes());

    return json_header_size + sizeof(uint64_t) + tree.tree_mem->size_bytes();
}

BATree map_ba_tree(const std::string &fname)
{
    auto mapping = std::make_shared<FileMapping>(fname);

    const uint64_t json_header_size = *reinterpret_cast<const uint64_t *>(mapping->data());
    const uint64_t total_header_size = json_header_size + sizeof(uint64_t);

    json header =
        json::parse(mapping->data() + sizeof(uint64_t), mapping->data() + total_header_size);

    // The tree starts at the next 4k page following the header
    uint64_t view_offset = align_to(total_header_size, 4096);

    auto tree_mem = std::make_shared<FileView<uint8_t>>(
        mapping, view_offset, mapping->nbytes() - view_offset);

    const uint64_t num_nodes = header["num_nodes"].get<uint64_t>();
    auto radix_nodes =
        std::make_shared<FileView<RadixTreeNode>>(mapping, view_offset, num_nodes);
    view_offset += radix_nodes->size_bytes();

    const uint64_t num_treelets = header["num_treelets"].get<uint64_t>();
    auto treelet_offsets =
        std::make_shared<FileView<uint64_t>>(mapping, view_offset, num_treelets);
    view_offset += treelet_offsets->size_bytes();

    const uint64_t bitmap_dict_size = header["bitmap_dict_size"].get<uint64_t>();
    auto bitmap_dictionary =
        std::make_shared<FileView<uint32_t>>(mapping, view_offset, bitmap_dict_size);
    view_offset += bitmap_dictionary->size_bytes();

    const uint64_t num_attribs = header["attributes"].size();
    auto radix_tree_bitmap_ids =
        std::make_shared<FileView<uint16_t>>(mapping, view_offset, num_nodes * num_attribs);
    view_offset += radix_tree_bitmap_ids->size_bytes();

    std::shared_ptr<AbstractArray<glm::vec2>> node_attrib_ranges = nullptr;
    if (header.find("has_ranges") != header.end()) {
        auto node_attrib_ranges_view = std::make_shared<FileView<glm::vec2>>(
            mapping, view_offset, num_nodes * num_attribs);
        view_offset += node_attrib_ranges_view->size_bytes();

        node_attrib_ranges = node_attrib_ranges_view;
    }

    std::vector<AttributeDescription> attributes;
    for (size_t i = 0; i < num_attribs; ++i) {
        AttributeDescription a;
        a.name = header["attributes"][i]["name"].get<std::string>();
        a.data_type = static_cast<DTYPE>(header["attributes"][i]["dtype"].get<int>());
        a.range.x = header["attributes"][i]["range"][0].get<float>();
        a.range.y = header["attributes"][i]["range"][1].get<float>();

        attributes.push_back(a);
    }

    json bounds_array = header["bounds"];
    Box bounds(glm::vec3(bounds_array[0].get<float>(),
                         bounds_array[1].get<float>(),
                         bounds_array[2].get<float>()),
               glm::vec3(bounds_array[3].get<float>(),
                         bounds_array[4].get<float>(),
                         bounds_array[5].get<float>()));

    return BATree(bounds,
                  attributes,
                  tree_mem,
                  radix_nodes,
                  treelet_offsets,
                  bitmap_dictionary,
                  radix_tree_bitmap_ids,
                  node_attrib_ranges,
                  header["num_points"].get<uint64_t>(),
                  header["num_lod_prims"].get<uint64_t>(),
                  header["min_lod"].get<uint64_t>(),
                  header["max_lod"].get<uint64_t>());
}
