#pragma once

#include <memory>
#include <string>
#include <glm/glm.hpp>
#include <parallel_hashmap/phmap.h>
#include "abstract_array.h"
#include "data_type.h"
#include "owned_array.h"

struct AttributeDescription {
    std::string name;
    DTYPE data_type = UNKNOWN;
    glm::vec2 range = glm::vec2(-std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity());

    AttributeDescription(const std::string &name, DTYPE data_type);

    AttributeDescription(const std::string &name, DTYPE data_type, const glm::vec2 &range);

    AttributeDescription() = default;

    bool has_range() const;

    size_t stride() const;

    uint32_t bitmap(const float val) const;

    float bitmap_bin_size() const;
};

struct Attribute {
    AttributeDescription desc;
    ArrayHandle<uint8_t> data = nullptr;

    Attribute(const AttributeDescription &desc, const ArrayHandle<uint8_t> &data);

    Attribute() = default;

    glm::vec2 compute_range() const;

    size_t size() const;

    // Get a raw pointer to the attribute element at index i, applying the data
    // type stride
    uint8_t *at(size_t i);

    const uint8_t *at(size_t i) const;

    float as_float(size_t i) const;

    // Compute the bitmap for a specific element in the array
    uint32_t bitmap(size_t i) const;
};

struct AttributeQuery {
    std::string name;
    DTYPE data_type = UNKNOWN;
    std::vector<glm::vec2> ranges;
    uint32_t bitmask = 0xffffffff;

    AttributeQuery(const std::string &name);

    AttributeQuery(const std::string &name, const glm::vec2 &range);

    AttributeQuery(const std::string &name, const std::vector<glm::vec2> &ranges);

    AttributeQuery() = default;

    // Compute the bitmask for the query when run over the specified sub-range
    uint32_t query_bitmask(const glm::vec2 &sub_range) const;

    bool contains_value(const float val) const;
};

uint32_t query_bitmask(const glm::vec2 &query_range, const glm::vec2 &value_range);

// Adjust the bitmask to be usable for queries over the new range, by either stretching
// or squashing the input bitmask
uint32_t remap_bitmask(uint32_t mask, const glm::vec2 &old_range, const glm::vec2 &new_range);

// Build the bitmap dictionary based on the map of bitmap use counts. The use counts
// in the map will then be replaced with the bitmap index in the returned dictionary
ArrayHandle<uint32_t> build_bitmap_dictionary(
    phmap::flat_hash_map<uint32_t, uint32_t> &bitmap_map);

