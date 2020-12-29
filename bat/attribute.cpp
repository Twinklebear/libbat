#include "attribute.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <glm/ext.hpp>
#include <tbb/parallel_sort.h>
#include "abstract_array.h"
#include "data_type.h"
#include "fixed_array.h"
#include "util.h"

AttributeDescription::AttributeDescription(const std::string &name, DTYPE data_type)
    : name(name), data_type(data_type)
{
}

AttributeDescription::AttributeDescription(const std::string &name,
                                           DTYPE data_type,
                                           const glm::vec2 &range)
    : name(name), data_type(data_type), range(range)
{
}

bool AttributeDescription::has_range() const
{
    return std::isfinite(range.x) && std::isfinite(range.y);
}

size_t AttributeDescription::stride() const
{
    return dtype_stride(data_type);
}

uint32_t AttributeDescription::bitmap(const float val) const
{
    const float x = (val - range.x) / (range.y - range.x);
    if (x < 0.f || x > 1.f) {
        return 0;
    }

    const uint32_t bit = clamp(x * 32.f, 0.f, 31.f);
    return 1 << bit;
}

float AttributeDescription::bitmap_bin_size() const
{
    return (range.y - range.x) / 32.f;
}

Attribute::Attribute(const AttributeDescription &desc, const ArrayHandle<uint8_t> &data)
    : desc(desc), data(data)
{
}

glm::vec2 Attribute::compute_range() const
{
    switch (desc.data_type) {
    case INT_8:
        return ::compute_range(reinterpret_cast<const int8_t *>(at(0)),
                               reinterpret_cast<const int8_t *>(at(size() - 1)));
    case UINT_8:
        return ::compute_range(reinterpret_cast<const uint8_t *>(at(0)),
                               reinterpret_cast<const uint8_t *>(at(size() - 1)));
    case INT_16:
        return ::compute_range(reinterpret_cast<const int16_t *>(at(0)),
                               reinterpret_cast<const int16_t *>(at(size() - 1)));
    case UINT_16:
        return ::compute_range(reinterpret_cast<const uint16_t *>(at(0)),
                               reinterpret_cast<const uint16_t *>(at(size() - 1)));
    case INT_32:
        return ::compute_range(reinterpret_cast<const int32_t *>(at(0)),
                               reinterpret_cast<const int32_t *>(at(size() - 1)));
    case UINT_32:
        return ::compute_range(reinterpret_cast<const uint32_t *>(at(0)),
                               reinterpret_cast<const uint32_t *>(at(size() - 1)));
    case FLOAT_32:
        return ::compute_range(reinterpret_cast<const float *>(at(0)),
                               reinterpret_cast<const float *>(at(size() - 1)));
    case FLOAT_64:
        return ::compute_range(reinterpret_cast<const double *>(at(0)),
                               reinterpret_cast<const double *>(at(size() - 1)));
    case INT_64:
        return ::compute_range(reinterpret_cast<const int64_t *>(at(0)),
                               reinterpret_cast<const int64_t *>(at(size() - 1)));
    case UINT_64:
        return ::compute_range(reinterpret_cast<const uint64_t *>(at(0)),
                               reinterpret_cast<const uint64_t *>(at(size() - 1)));
    default:
        throw std::runtime_error("Invalid or unknown data type in Attribute::compute_range");
        return glm::vec2(std::numeric_limits<float>::signaling_NaN());
    }
}

size_t Attribute::size() const
{
    return data->size() / desc.stride();
}

uint8_t *Attribute::at(size_t i)
{
    return data->data() + i * desc.stride();
}

const uint8_t *Attribute::at(size_t i) const
{
    return data->data() + i * desc.stride();
}

float Attribute::as_float(size_t i) const
{
    switch (desc.data_type) {
    case INT_8:
        return *reinterpret_cast<const int8_t *>(at(i));
    case UINT_8:
        return *reinterpret_cast<const uint8_t *>(at(i));
    case INT_16:
        return *reinterpret_cast<const int16_t *>(at(i));
    case UINT_16:
        return *reinterpret_cast<const uint16_t *>(at(i));
    case INT_32:
        return *reinterpret_cast<const int32_t *>(at(i));
    case UINT_32:
        return *reinterpret_cast<const uint32_t *>(at(i));
    case FLOAT_32:
        return *reinterpret_cast<const float *>(at(i));
    case FLOAT_64:
        return *reinterpret_cast<const double *>(at(i));
    case INT_64:
        return *reinterpret_cast<const int64_t *>(at(i));
    case UINT_64:
        return *reinterpret_cast<const uint64_t *>(at(i));
    default:
        throw std::runtime_error("Invalid or unknown data type in Attribute::as_float");
        return 0;
    }
}

uint32_t Attribute::bitmap(size_t i) const
{
    return desc.bitmap(as_float(i));
}

AttributeQuery::AttributeQuery(const std::string &name) : name(name) {}

AttributeQuery::AttributeQuery(const std::string &name, const glm::vec2 &range)
    : name(name), ranges({range})
{
}

AttributeQuery::AttributeQuery(const std::string &name, const std::vector<glm::vec2> &ranges)
    : name(name), ranges(ranges)
{
}

uint32_t AttributeQuery::query_bitmask(const glm::vec2 &sub_range) const
{
    if (ranges.empty()) {
        return 0xffffffff;
    }

    uint32_t query_mask = 0;
    for (const auto &r : ranges) {
        query_mask |= ::query_bitmask(r, sub_range);
    }
    return query_mask;
}

bool AttributeQuery::contains_value(const float val) const
{
    bool val_in_range = false;
    for (const auto &r : ranges) {
        if (val >= r.x && val <= r.y) {
            val_in_range = true;
        }
    }
    return val_in_range;
}

uint32_t query_bitmask(const glm::vec2 &query_range, const glm::vec2 &value_range)
{
    if (query_range.y < value_range.x || query_range.x > value_range.y) {
        return 0;
    }

    const glm::vec2 clamped_query_range(std::max(query_range.x, value_range.x),
                                        std::min(query_range.y, value_range.y));

    const float denom = 1.f / (value_range.y - value_range.x);
    const uint32_t lo_bit =
        clamp((clamped_query_range.x - value_range.x) * denom * 32.f, 0.f, 31.f);
    uint32_t hi_bit =
        clamp(std::ceil((clamped_query_range.y - value_range.x) * denom * 32.f), 0.f, 32.f);
    if (hi_bit == lo_bit) {
        hi_bit++;
    }

    const uint32_t lo_mask = 0xffffffff << lo_bit;
    uint32_t hi_mask = 0xffffffff;
    if (hi_bit < 32) {
        hi_mask = ~(hi_mask << hi_bit);
    }

    return lo_mask & hi_mask;
}

uint32_t remap_bitmask(uint32_t mask, const glm::vec2 &old_range, const glm::vec2 &new_range)
{
    if (old_range.y < new_range.x || old_range.x > new_range.y) {
        return 0;
    }

    if (old_range == new_range) {
        return mask;
    }

    if (old_range.x == old_range.y || popcnt(mask) == 32) {
        return query_bitmask(old_range, new_range);
    }

    // Either the bin size for each bit here will grow, squishing the bitmask together
    // (old_range is contained in new_range), or the bin size is smaller and the bitmask
    // stretches (new_range is contined in old_range). The bit mask can also shift, in
    // the case that the ranges are not a perfect scale of each other (which is likely)
    // TODO: This is easy to do w/ a for loop, but is it possible to do something a bit better?
    // looping through is like going along and making a bunch of query bitmasks to port the
    // old range mask to the new range
    // Note: what's also important to note is that as we go up the tree the bin size can
    // only stay the same or get bigger, and as we descend the bin size can only stay
    // the same or shrink
    const float old_bin_size = (old_range.y - old_range.x) / 32.f;
    uint32_t new_mask = 0;
    uint32_t chunk_start = -1;
    uint32_t chunk_end = -1;
    for (uint32_t i = 0; i < 32; ++i) {
        if (mask & (1 << i)) {
            if (chunk_start == -1) {
                chunk_start = i;
            }
            chunk_end = i;
        } else if (chunk_start != -1) {
            const glm::vec2 chunk_range(chunk_start * old_bin_size + old_range.x,
                                        (chunk_end + 0.99) * old_bin_size + old_range.x);

            new_mask |= query_bitmask(chunk_range, new_range);

            chunk_start = -1;
            chunk_end = -1;
        }
    }

    // The last chunk which may go to the final bit of the bitmask
    if (chunk_start != -1) {
        const glm::vec2 chunk_range(chunk_start * old_bin_size + old_range.x,
                                    (chunk_end + 0.99) * old_bin_size + old_range.x);

        new_mask |= query_bitmask(chunk_range, new_range);
    }
    return new_mask;
}

ArrayHandle<uint32_t> build_bitmap_dictionary(
    phmap::flat_hash_map<uint32_t, uint32_t> &bitmap_map)
{
    // std::cout << "# of unique bitmap indices: " << bitmap_map.size() << "\n";
    // Curious if we ever have a data set which hits this threshold
    if (bitmap_map.size() > std::numeric_limits<uint16_t>::max()) {
        std::cout << "[ERROR]: # of bitmaps exceeds uint16 max!\n";
        throw std::runtime_error(
            "TODO WILL: Too many bitmaps, need to implement some form of merging!");
    }

    auto bitmap_dictionary = std::make_shared<FixedArray<uint32_t>>(bitmap_map.size());
    uint32_t total_use_counts = 0;
    uint32_t max_use_count = 0;
    {
        size_t idx = 0;
        for (const auto &b : bitmap_map) {
            max_use_count = std::max(b.second, max_use_count);
            total_use_counts += b.second;
            (*bitmap_dictionary)[idx++] = b.first;
        }
    }
#if 0
    std::cout << "Most used bitmap is used " << max_use_count << " times, avg. use count: "
              << total_use_counts / static_cast<float>(bitmap_map.size()) << "\n";
#endif

    // TODO: Is it worth sorting the bitmaps by frequency of use?
    tbb::parallel_sort(
        bitmap_dictionary->begin(),
        bitmap_dictionary->end(),
        [&](const uint32_t &a, const uint32_t &b) { return bitmap_map[a] > bitmap_map[b]; });

    for (size_t i = 0; i < bitmap_dictionary->size(); ++i) {
        const uint32_t m = (*bitmap_dictionary)[i];
        bitmap_map[m] = i;
    }
    return bitmap_dictionary;
}
