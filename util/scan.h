#pragma once

#include <vector>
#include <tbb/parallel_for.h>
#include <tbb/tbb.h>

template <typename T, typename Op>
T inclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op)
{
    out.resize(in.size(), id);
    using range_type = tbb::blocked_range<size_t>;
    T sum = tbb::parallel_scan(
        range_type(0, in.size()),
        id,
        [&](const range_type &r, T sum, bool is_final_scan) {
            T tmp = sum;
            for (size_t i = r.begin(); i < r.end(); ++i) {
                tmp = op(tmp, in[i]);
                if (is_final_scan) {
                    out[i] = tmp;
                }
            }
            return tmp;
        },
        [&](const T &a, const T &b) { return op(a, b); });
    return sum;
}

template <typename T, typename Op>
T exclusive_scan(const std::vector<T> &in, const T &id, std::vector<T> &out, Op op)
{
    // Exclusive scan is the same as inclusive, but shifted by one
    out.resize(in.size() + 1, id);
    using range_type = tbb::blocked_range<size_t>;
    T sum = tbb::parallel_scan(
        range_type(0, in.size()),
        id,
        [&](const range_type &r, T sum, bool is_final_scan) {
            T tmp = sum;
            for (size_t i = r.begin(); i < r.end(); ++i) {
                tmp = op(tmp, in[i]);
                if (is_final_scan) {
                    out[i + 1] = tmp;
                }
            }
            return tmp;
        },
        [&](const T &a, const T &b) { return op(a, b); });
    out.pop_back();
    return sum;
}
