#include "pbat_file.h"
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

size_t write_pba_tree(const std::string &fname, const AggregationTree &tree)
{
    std::ofstream fout(fname.c_str(), std::ios::binary);

    json header;
    header["bounds"] = {tree.bounds.lower.x,
                        tree.bounds.lower.y,
                        tree.bounds.lower.z,
                        tree.bounds.upper.x,
                        tree.bounds.upper.y,
                        tree.bounds.upper.z};
    header["num_nodes"] = tree.nodes->size();
    header["num_aggregators"] =
        std::accumulate(tree.nodes->cbegin(),
                        tree.nodes->cend(),
                        size_t(0),
                        [](const size_t n, const AggregationKdNode &node) {
                            return n + (node.is_leaf() ? 1 : 0);
                        });
    header["num_sim_ranks"] = tree.primitives->size();
    header["num_points"] = tree.num_points;
    header["bat_prefix"] = tree.bat_prefix;
    header["bitmap_dict_size"] = tree.bitmap_dictionary->size();
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

    // TODO: Add some padding to ensure the tree mem buffer starts at a 4k page boundary
    // We don't include the null terminator
    const uint64_t json_header_size = json_header.size();
    fout.write(reinterpret_cast<const char *>(&json_header_size), sizeof(uint64_t));
    fout.write(json_header.c_str(), json_header_size);

    fout.write(reinterpret_cast<const char *>(tree.nodes->data()), tree.nodes->size_bytes());
    fout.write(reinterpret_cast<const char *>(tree.bitmap_dictionary->data()),
               tree.bitmap_dictionary->size_bytes());
    fout.write(reinterpret_cast<const char *>(tree.node_bitmap_ids->data()),
               tree.node_bitmap_ids->size_bytes());
    size_t bytes_written = json_header_size + sizeof(uint64_t) + tree.nodes->size_bytes() +
                           tree.bitmap_dictionary->size_bytes() +
                           tree.node_bitmap_ids->size_bytes();

    if (tree.node_attrib_ranges) {
        fout.write(reinterpret_cast<const char *>(tree.node_attrib_ranges->data()),
                   tree.node_attrib_ranges->size_bytes());
        bytes_written += tree.node_attrib_ranges->size_bytes();
    }
    return bytes_written;
}

AggregationTree map_pba_tree(const std::string &fname)
{
    auto mapping = std::make_shared<FileMapping>(fname);

    const uint64_t json_header_size = *reinterpret_cast<const uint64_t *>(mapping->data());
    const uint64_t total_header_size = json_header_size + sizeof(uint64_t);

    json header =
        json::parse(mapping->data() + sizeof(uint64_t), mapping->data() + total_header_size);

    uint64_t view_offset = total_header_size;

    const uint64_t num_nodes = header["num_nodes"].get<uint64_t>();
    auto nodes =
        std::make_shared<FileView<AggregationKdNode>>(mapping, view_offset, num_nodes);
    view_offset += nodes->size_bytes();

    const uint64_t bitmap_dict_size = header["bitmap_dict_size"].get<uint64_t>();
    auto bitmap_dictionary =
        std::make_shared<FileView<uint32_t>>(mapping, view_offset, bitmap_dict_size);
    view_offset += bitmap_dictionary->size_bytes();

    const uint64_t num_attribs = header["attributes"].size();
    auto node_bitmap_ids =
        std::make_shared<FileView<uint16_t>>(mapping, view_offset, num_nodes * num_attribs);
    view_offset += node_bitmap_ids->size_bytes();

    std::shared_ptr<AbstractArray<glm::vec2>> node_attrib_ranges = nullptr;
    if (header.find("has_ranges") != header.end()) {
        auto node_attrib_ranges_view = std::make_shared<FileView<glm::vec2>>(
            mapping, view_offset, num_nodes * num_attribs);
        view_offset += node_attrib_ranges_view->size_bytes();

        node_attrib_ranges =
            std::dynamic_pointer_cast<AbstractArray<glm::vec2>>(node_attrib_ranges_view);
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

    const std::string base_path = get_file_basepath(canonicalize_path(fname));
    const std::string bat_prefix = base_path + "/" + header["bat_prefix"].get<std::string>();
    const uint32_t num_aggregators = header["num_aggregators"].get<uint32_t>();
    const uint64_t num_points = header["num_points"].get<uint64_t>();

    return AggregationTree(bounds,
                           num_aggregators,
                           num_points,
                           attributes,
                           nodes,
                           bitmap_dictionary,
                           node_bitmap_ids,
                           node_attrib_ranges,
                           bat_prefix);
}

