#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <glm/glm.hpp>
#ifdef __AVX2__
#include <immintrin.h>
#include <nmmintrin.h>
#endif
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include "box.h"
#include "json.hpp"
#include "tinyxml2.h"

// Format the bytes as XKB, XMB, etc. depending on the size
std::string format_bytes(size_t nbytes);

// Format the count as #G, #M, #K, depending on its magnitude
std::string pretty_print_count(const double count);

uint64_t align_to(uint64_t val, uint64_t align);

std::string canonicalize_path(const std::string &path);

std::string get_cpu_brand();

// Read the contents of a file into the string
std::string get_file_content(const std::string &fname);

bool compute_divisor(uint32_t x, uint32_t &divisor);

// Compute a 3D grid which has num grid cells
glm::uvec3 compute_grid3d(uint32_t num);

// Compute a 2D grid which has num grid cells
glm::uvec2 compute_grid2d(uint32_t num);

std::string get_file_extension(const std::string &fname);

std::string get_file_basename(const std::string &path);

std::string get_file_basepath(const std::string &path);

bool starts_with(const std::string &str, const std::string &prefix);

bool is_big_endian();

template <typename RandomIt>
Box compute_bounds(RandomIt begin, RandomIt end)
{
    using range_type = tbb::blocked_range<RandomIt>;
    return tbb::parallel_reduce(
        range_type(begin, end),
        Box{},
        [](const range_type &r, const Box &b) {
            Box res = b;
            for (auto it = r.begin(); it != r.end(); ++it) {
                res.extend(*it);
            }
            return res;
        },
        [](const Box &a, const Box &b) { return box_union(a, b); });
}

template <typename RandomIt, typename GetPos>
Box compute_bounds(RandomIt begin, RandomIt end, GetPos get_pos)
{
    using range_type = tbb::blocked_range<RandomIt>;
    return tbb::parallel_reduce(
        range_type(begin, end),
        Box{},
        [&](const range_type &r, const Box &b) {
            Box res = b;
            for (auto it = r.begin(); it != r.end(); ++it) {
                res.extend(get_pos(*it));
            }
            return res;
        },
        [](const Box &a, const Box &b) { return box_union(a, b); });
}

template <typename T>
inline T clamp(const T &x, const T &lo, const T &hi)
{
    if (x < lo) {
        return lo;
    } else if (x > hi) {
        return hi;
    }
    return x;
}

inline float srgb_to_linear(const float x)
{
    if (x <= 0.04045f) {
        return x / 12.92f;
    } else {
        return std::pow((x + 0.055f) / 1.055f, 2.4f);
    }
}

template <typename T>
glm::vec2 compute_range(const T *begin, const T *end)
{
    using range_type = tbb::blocked_range<const T *>;
    auto minmax = tbb::parallel_reduce(
        range_type(begin, end),
        std::make_pair(*begin, *begin),
        [](const range_type &r, const std::pair<T, T> &tmp) {
            auto sub_minmax = std::minmax_element(r.begin(), r.end());
            return std::make_pair(std::min(tmp.first, *sub_minmax.first),
                                  std::max(tmp.second, *sub_minmax.second));
        },
        [](const std::pair<T, T> &a, const std::pair<T, T> &b) {
            return std::make_pair(std::min(a.first, b.first), std::max(a.second, b.second));
        });
    return glm::vec2(minmax.first, minmax.second);
}

#ifdef __AVX2__

inline uint32_t encode_morton32(uint32_t x, uint32_t y, uint32_t z)
{
    // 32   28   24   20   16   12   8    4
    // 00zy xzyx zyxz yxzy xzyx zyxz yxzy xzyx
    // x  0    9    2    4    9    2    4    9
    // y  1    2    4    9    2    4    9    2
    // z  2    4    9    2    4    9    2    4
#define X_MASK 0x09249249
#define Z_MASK 0x24924924
#define Y_MASK 0x12492492
    return _pdep_u32(x, X_MASK) | _pdep_u32(y, Y_MASK) | _pdep_u32(z, Z_MASK);
}

inline void decode_morton32(uint32_t code, uint32_t &x, uint32_t &y, uint32_t &z)
{
#define X_MASK 0x09249249
#define Y_MASK 0x12492492
#define Z_MASK 0x24924924
    x = _pext_u32(code, X_MASK);
    y = _pext_u32(code, Y_MASK);
    z = _pext_u32(code, Z_MASK);
}

inline uint32_t leading_zeros(uint32_t x)
{
    return _lzcnt_u32(x);
}

#else

// See https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
inline uint32_t part1by2(uint32_t x)
{
    x &= 0x000003ff;                   // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) & 0xff0000ff;  // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) & 0x0300f00f;   // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) & 0x030c30c3;   // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) & 0x09249249;   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

inline uint32_t encode_morton32(uint32_t x, uint32_t y, uint32_t z)
{
    return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x);
}

inline uint32_t compact1by2(uint32_t x)
{
    x &= 0x09249249;                   // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) & 0x030c30c3;   // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) & 0x0300f00f;   // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) & 0xff0000ff;   // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff;  // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

inline void decode_morton32(uint32_t code, uint32_t &x, uint32_t &y, uint32_t &z)
{
    x = compact1by2(code);
    y = compact1by2(code >> 1);
    z = compact1by2(code >> 2);
}

inline uint32_t leading_zeros(uint32_t x)
{
    uint32_t count = 0;
    for (int32_t i = 31; i >= 0; --i, ++count) {
        if ((x & (0x1 << i))) {
            break;
        }
    }
    return count;
}

#endif

inline uint32_t longest_prefix(uint32_t a, uint32_t b)
{
    return leading_zeros(a ^ b);
}

inline uint32_t popcnt(uint32_t x)
{
#ifndef __PPC__
    return _mm_popcnt_u32(x);
#else
    return __builtin_popcount(x);
#endif
}

// More of a debug/statistical thing for testing how well the bitmap index
// can be used to refine queries faster than just min/max range. Counts the
// number of distinct bit ranges which are set and the number of bits within these
std::vector<uint32_t> count_bitmap_chunks(uint32_t m);

template <typename T, size_t N>
inline glm::vec<N, T> get_vec(const nlohmann::json &j)
{
    glm::vec<N, T> v;
    for (size_t i = 0; i < N; ++i) {
        v[i] = j[i].get<T>();
    }
    return v;
}

std::string tinyxml_error_string(const tinyxml2::XMLError e);
