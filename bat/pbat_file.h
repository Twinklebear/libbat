#pragma once

#include <string>
#include "aggregation_tree.h"

/* The PBATree file format is inspired by GLB, and consists of a header
 * followed by arrays containing the tree nodes, points, and their attributes.
 * The file prefix is the prefix for the bat files containing the data
 * for the leaves of the aggregation tree
 *
 * Header:
 * json_header_size: uint64_t
 * json_header: [char...]
 *
 * The JSON header contains the following information:
 * {
 *   "bounds": [tree bounding box],
 *   "num_nodes": # of kd nodes in the file,
 *   "num_points": # of points in the file,
 *   "attributes": [
 *      {
 *        "name": attrib name,
 *        "dtype": attrib_dtype,
 *        "offset": offset from the end of the header to the attribute array,
 *        "size": size in bytes of the array
 *      },
 *    ],
 *    "bat_prefix": "file"
 * }
 *
 * Each leaf cell references on
 */

// Write the Aggregation tree to the file
size_t write_pba_tree(const std::string &fname, const AggregationTree &tree);

// Map the file containing the tree
AggregationTree map_pba_tree(const std::string &fname);

