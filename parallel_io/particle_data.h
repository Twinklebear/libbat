#pragma once

#include <parallel_hashmap/phmap.h>
#include "abstract_array.h"
#include "aggregation_tree.h"
#include "attribute.h"

struct WritePerformance {
    // Uniform info across all ranks
    int mpi_size = -1;
    uint64_t total_bytes_written = 0;
    uint32_t node_core_count = 0;

    // On all ranks, some basic info
    int mpi_rank = -1;
    uint64_t local_particle_count = 0;
    uint32_t assigned_aggregator = 0;
    uint32_t aggregator_send_time = 0;

    // On rank 0, stats about the aggregation tree setup
    uint32_t agg_tree_time = 0;
    float agg_tree_cpu_use = 0.f;
    uint32_t agg_assign_distrib_time = 0;
    uint32_t agg_info_distrib_time = 0;
    uint32_t agg_list_distrib_time = 0;
    uint32_t total_agg_assign_time = 0;
    uint32_t pbat_data_gather_time = 0;
    uint32_t pbat_write_time = 0;
    uint64_t pbat_file_size = 0;

    // For ranks which are aggregators, stats about the aggregation process
    // and BAT tree build/write
    uint32_t aggregator_num_clients = 0;
    uint64_t aggregator_particle_count = 0;
    uint32_t aggregator_recv_time = 0;
    uint32_t aggregator_num_cores = 0;
    uint32_t bat_tree_time = 0;
    uint32_t bat_tree_compact_time = 0;
    float bat_tree_cpu_use = 0.f;
    uint32_t data_write_time = 0;
    uint64_t aggregator_file_size = 0;

    WritePerformance(int mpi_rank, int mpi_size, uint64_t local_particle_count);

    WritePerformance() = default;
};

struct ParticleData {
    int mpi_rank = -1;
    int mpi_size = -1;

    Box local_bounds;
    uint64_t num_particles = 0;
    ArrayHandle<glm::vec3> positions;
    std::vector<Attribute> attributes;
    phmap::flat_hash_map<std::string, size_t> attribute_indices;

    uint64_t max_bytes_per_subfile = -1;
    uint64_t max_particles_per_aggregator = -1;

    WriteOptions options;

    WritePerformance perf;

    // On rank 0, the write performance stats aggregated from the other ranks
    std::string perf_json;
};

struct ParticleFile {
    int mpi_rank = -1;
    int mpi_size = -1;

    AggregationTree tree;
    // The trees which this rank is assigned to be the read aggregator for
    phmap::flat_hash_map<uint64_t, std::shared_ptr<BATree>> ba_trees;

    Box read_bounds;
    uint64_t num_particles = 0;
    OwnedArrayHandle<glm::vec3> positions;
    std::vector<Attribute> attributes;
};
