#include "query_stats.h"
#include <glm/glm.hpp>

QueryStats::QueryStats()
    : n_total_tested(0),
      n_range_filtered(0),
      n_map_filtered(0),
      n_particles_tested(0),
      n_particles_returned(0),
      query_depth(0)
{
}

QueryStats::QueryStats(const QueryStats &s)
    : n_total_tested(s.n_total_tested.load()),
      n_range_filtered(s.n_range_filtered.load()),
      n_map_filtered(s.n_map_filtered.load()),
      n_particles_tested(s.n_particles_tested.load()),
      n_particles_returned(s.n_particles_returned.load()),
      query_depth(s.query_depth.load())
{
}

QueryStats &QueryStats::operator=(const QueryStats &s)
{
    if (this == &s) {
        return *this;
    }

    n_total_tested = s.n_total_tested.load();
    n_range_filtered = s.n_range_filtered.load();
    n_map_filtered = s.n_map_filtered.load();
    n_particles_tested = s.n_particles_tested.load();
    n_particles_returned = s.n_particles_returned.load();
    query_depth = s.query_depth.load();
    return *this;
}

QueryQuality::QueryQuality(const float prev_quality,
                           const float current_quality,
                           const uint32_t min_lod,
                           const uint32_t max_lod)
{
    const uint32_t lod_range = max_lod - min_lod + 1.f;
    const float prev_lod =
        glm::clamp(prev_quality * lod_range + min_lod, float(min_lod), float(max_lod) + 1.f);
    const float cur_lod =
        glm::clamp(current_quality * lod_range + min_lod, float(min_lod), float(max_lod) + 1.f);

    prev_depth = static_cast<uint32_t>(prev_lod);
    prev_fraction = prev_lod - prev_depth;
    // Depths are relative to treelet depths
    prev_depth -= min_lod;

    current_depth = static_cast<uint32_t>(cur_lod);
    current_fraction = cur_lod - current_depth;
    // Depths are relative to treelet depths
    current_depth -= min_lod;
}

std::ostream &operator<<(std::ostream &os, const QueryStats &stats)
{
    os << "{\n"
       << "\ttotal nodes tested: " << stats.n_total_tested << "\n"
       << "\tnodes range filtered: " << stats.n_range_filtered << "\n"
       << "\tnodes bitmap filtered: " << stats.n_map_filtered << "\n"
       << "\tpercentage range filtered: "
       << (100.f * stats.n_range_filtered) / stats.n_total_tested << "%\n"
       << "\tpercentage bitmap filtered: "
       << (100.f * stats.n_map_filtered) / stats.n_total_tested << "%\n"
       << "\tparticles returned: " << stats.n_particles_returned << "\n"
       << "\tparticles tested: " << stats.n_particles_tested << "\n"
       << "\tpercentage particles tested and discarded: "
       << (100.f * (stats.n_particles_tested - stats.n_particles_returned)) /
              stats.n_particles_tested
       << "%\n"
       << "\tquery depth: " << stats.query_depth << "\n"
       << "}";
    return os;
}

std::ostream &operator<<(std::ostream &os, const QueryQuality &q)
{
    os << "(prev: {" << q.prev_depth << ", " << q.prev_fraction << "}; current: {"
       << q.current_depth << ", " << q.current_fraction << "})";
    return os;
}
