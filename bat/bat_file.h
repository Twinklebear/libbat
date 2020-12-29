#pragma once

#include <string>
#include "ba_tree.h"

/* The BATree file format is inspired by GLB, and consists of a header
 * followed by arrays containing the tree nodes, points, and their attributes.
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
 *    ]
 * }
 */

// Write the BATree to the file, returns the number of bytes written
size_t write_ba_tree(const std::string &fname, const BATree &tree);

// Map the file containing the tree
BATree map_ba_tree(const std::string &fname);
