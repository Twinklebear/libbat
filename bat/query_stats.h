#pragma once

#include <atomic>
#include <ostream>

struct QueryStats {
    std::atomic_size_t n_total_tested;
    std::atomic_size_t n_range_filtered;
    std::atomic_size_t n_map_filtered;
    std::atomic_size_t n_particles_tested;
    std::atomic_size_t n_particles_returned;
    std::atomic<uint32_t> query_depth;

    QueryStats();

    QueryStats(const QueryStats &s);

    QueryStats &operator=(const QueryStats &s);
};

struct QueryQuality {
    uint32_t prev_depth = 0;
    uint32_t current_depth = 0;
    float prev_fraction = 0.f;
    float current_fraction = 0.f;

    QueryQuality(const float prev_quality,
                 const float current_quality,
                 const uint32_t min_lod,
                 const uint32_t max_lod);
};

std::ostream &operator<<(std::ostream &os, const QueryStats &stats);

std::ostream &operator<<(std::ostream &os, const QueryQuality &quality);
