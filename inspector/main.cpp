#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "aggregation_tree.h"
#include "bat_handle.h"
#include "json.hpp"
#include "util.h"

/* A query should be JSON in the form:
 * {
 *   "attributes": [
 *      {
 *          "name": "<name>",
 *          "range": [
 *              [lo, hi],
 *              [lo, hi]
 *              ...
 *          ]
 *      }
 *    "bounds": [[lo_x, lo_y, lo_z], [hi_x, hi_y, hi_z]]
 *   ]
 * }
 */

using json = nlohmann::json;
using namespace std::chrono;

const std::string USAGE = R"(Usage: ./bat_inspector [file.bat...] [options]

Options:
    -file <file.json>       Specify the JSON file containing the query to run.
    -json <json string>     Specify a JSON string containing the query to run.
    -quality [0-1]          Specify the LOD quality level, from 0 to 1. Can be passed repeatedly
                            in which case the quality levels will be queried in order
    -disable-range          Disable the use of range-based filtering (if available in the file).
    -disable-bitmap         Disable the use of bitmap-based filtering
    -n <N>                  Run the query <N> times
)";

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv, argv + argc);
    if (args.size() == 1) {
        std::cout << USAGE << "\n";
        return 1;
    }

    std::vector<float> quality;
    bool enable_range_filtering = true;
    bool enable_bitmap_filtering = true;
    size_t num_tests = 1;
    BATHandle tree;
    json query_cfg;
    json benchmark_log;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i][0] == '-') {
            if (args[i] == "-file") {
                std::ifstream fin(args[++i].c_str());
                fin >> query_cfg;
            } else if (args[i] == "-json") {
                query_cfg = json::parse(args[++i]);
            } else if (args[i] == "-quality") {
                quality.push_back(std::stof(args[++i]));
            } else if (args[i] == "-disable-range") {
                enable_range_filtering = false;
            } else if (args[i] == "-disable-bitmap") {
                enable_bitmap_filtering = false;
            } else if (args[i] == "-n") {
                num_tests = std::stoull(args[++i]);
            }
        } else {
            tree = BATHandle(args[i]);

            auto &tree_info = benchmark_log["tree"];
            tree_info["file"] = args[i];
            tree_info["bounds"] = {
                {tree.bounds().lower.x, tree.bounds().lower.y, tree.bounds().lower.z},
                {tree.bounds().upper.x, tree.bounds().upper.y, tree.bounds().upper.z}};
            tree_info["#attribs"] = tree.attributes.size();
            tree_info["#treelets"] = tree.num_treelets();
            tree_info["#points"] = tree.num_points();

            tree_info["#radix_nodes"] = tree.radix_tree_size();
            tree_info["radix_nodes_bytes"] = tree.radix_tree_size_bytes();
            tree_info["#bitmaps"] = tree.bitmap_dictionary_size();
            tree_info["bitmaps_bytes"] = tree.bitmap_dictionary_size_bytes();

            tree_info["#total_bitmap_indices"] = tree.bitmap_indices_size();
            tree_info["total_bitmap_indices_bytes"] = tree.bitmap_indices_size_bytes();
            tree_info["#total_attrib_ranges"] = tree.node_attrib_ranges_size();
            tree_info["total_attrib_ranges_bytes"] = tree.node_attrib_ranges_size_bytes();

            tree_info["#aggregators"] = tree.num_aggregators();
            tree_info["#aggregator_nodes"] = tree.aggregation_tree_size();
            tree_info["aggregator_nodes_bytes"] = tree.aggregation_tree_size_bytes();

            tree_info["total_tree_nodes"] = tree.total_tree_node_count();
            tree_info["total_tree_node_bytes"] = tree.total_tree_node_bytes();

            size_t base_data_size = tree.num_points() * sizeof(glm::vec3);
            for (const auto &a : tree.attributes) {
                auto &attr_info = tree_info["attribs"][a.name];
                attr_info["name"] = a.name;
                attr_info["range"] = {a.range.x, a.range.y};
                attr_info["bitmap_bin_size"] = (a.range.y - a.range.x) / 32.f;
                attr_info["dtype"] = print_data_type(a.data_type);
                base_data_size += a.stride() * tree.num_points();

                // If tree has aggregators, fetch all their bin sizes and include those too
                if (tree.num_aggregators() > 0) {
                    attr_info["aggregator_bitmap_bin_sizes"] =
                        tree.aggregator_bitmap_bin_sizes(a.name);
                }
            }
            tree_info["base_data_size"] = base_data_size;

            size_t total_size =
                tree.total_tree_node_bytes() + tree.bitmap_indices_size_bytes() +
                tree.bitmap_dictionary_size_bytes() + tree.node_attrib_ranges_size_bytes();
            tree_info["bat_total_size"] = total_size;
            tree_info["overhead(%)"] =
                (static_cast<float>(total_size) / base_data_size) * 100.f;
            tree_info["nodes_overhead(%)"] =
                (static_cast<float>(tree.total_tree_node_bytes()) / base_data_size) * 100.f;

            size_t bitmaps_only = tree.total_tree_node_bytes() +
                                  tree.bitmap_indices_size_bytes() +
                                  tree.bitmap_dictionary_size_bytes();
            tree_info["bitmap_tree_only_overhead(%)"] =
                (static_cast<float>(bitmaps_only) / base_data_size) * 100.f;

            size_t ranges_only =
                tree.total_tree_node_bytes() + tree.node_attrib_ranges_size_bytes();
            // If not storing ranges, compute what the overhead would be if we did store them
            if (tree.node_attrib_ranges_size_bytes() == 0) {
                ranges_only = tree.total_tree_node_bytes() + sizeof(glm::vec2) *
                                                                 tree.total_tree_node_count() *
                                                                 tree.attributes.size();
            }
            tree_info["ranges_tree_only_overhead(%)"] =
                (static_cast<float>(ranges_only) / base_data_size) * 100.f;
        }
    }

    if (query_cfg.is_null() && quality.empty()) {
        std::cout << benchmark_log.dump(4) << "\n";
        return 0;
    }

    if (!enable_range_filtering) {
        benchmark_log["range_filtering"] = false;
    }
    if (!enable_bitmap_filtering) {
        benchmark_log["bitmap_filtering"] = false;
    }

    tree.set_enable_range_filtering(enable_range_filtering);
    tree.set_enable_bitmap_filtering(enable_bitmap_filtering);

    benchmark_log["query"]["quality"] = quality;
    std::vector<AttributeQuery> attribute_queries;
    benchmark_log["query"] = query_cfg;
    if (query_cfg.find("attributes") != query_cfg.end()) {
        json attrib_cfg = query_cfg["attributes"];
        for (size_t i = 0; i < attrib_cfg.size(); ++i) {
            const std::string attrib = attrib_cfg[i]["name"].get<std::string>();

            std::vector<glm::vec2> ranges;
            for (size_t j = 0; j < attrib_cfg[i]["range"].size(); ++j) {
                ranges.push_back(get_vec<float, 2>(attrib_cfg[i]["range"][j]));
            }
            attribute_queries.push_back(AttributeQuery(attrib, ranges));
        }
    }

    Box query_bounds = tree.bounds();
    if (query_cfg.find("bounds") != query_cfg.end()) {
        query_bounds.lower = get_vec<float, 3>(query_cfg["bounds"][0]);
        query_bounds.upper = get_vec<float, 3>(query_cfg["bounds"][1]);
    }

    float last_quality = 0.f;
    for (size_t j = 0; j < quality.size(); ++j) {
        auto &query_quality_perf = benchmark_log["query_perf"][j];
        query_quality_perf["quality"] = quality[j];
        if (last_quality > quality[j]) {
            last_quality = 0.f;
        }
        for (size_t i = 0; i < num_tests; ++i) {
            std::vector<glm::vec3> pts;
            auto start = steady_clock::now();
            auto stats = tree.query_box_progressive(
                query_bounds,
                !attribute_queries.empty() ? &attribute_queries : nullptr,
                last_quality,
                quality[j],
                [](const size_t i,
                   const glm::vec3 &pos,
                   const std::vector<Attribute> &attribs) {});
            auto end = steady_clock::now();
            auto dur = duration_cast<milliseconds>(end - start).count();

            auto &query_perf = query_quality_perf["queries"][i];
            query_perf["nodes_tested"] = stats.n_total_tested.load();

            query_perf["nodes_range_filtered"] = stats.n_range_filtered.load();
            query_perf["percent_range_filtered"] =
                (100.f * stats.n_range_filtered.load()) / stats.n_total_tested.load();

            query_perf["nodes_bitmap_filtered"] = stats.n_map_filtered.load();
            query_perf["percent_bitmap_filtered"] =
                (100.f * stats.n_map_filtered.load()) / stats.n_total_tested.load();

            query_perf["points_returned"] = stats.n_particles_returned.load();
            query_perf["points_tested"] = stats.n_particles_tested.load();

            const size_t particles_discarded =
                stats.n_particles_tested.load() - stats.n_particles_returned.load();
            query_perf["points_discarded"] = particles_discarded;
            query_perf["percent_tested_and_discarded"] =
                (100.f * particles_discarded) / stats.n_particles_tested.load();

            query_perf["time_ms"] = dur;
        }
        last_quality = quality[j];
    }

    std::cout << benchmark_log.dump(4) << "\n";

    return 0;
}
