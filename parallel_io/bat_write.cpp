#include "bat_write.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <json.hpp>
#include <mpi.h>
#include <sched.h>
#include <stdlib.h>
#include <sys/types.h>
#include <tbb/global_control.h>
#include <unistd.h>
#include "aggregator_assignment.h"
#include "bat_file.h"
#include "binary_particle_file.h"
#include "borrowed_array.h"
#include "lba_tree_builder.h"
#include "particle_data.h"
#include "pbat_file.h"
#include "profiling.h"
#include "util.h"

extern "C" BATParticleState bat_io_allocate(void)
{
    auto *pd = new ParticleData;
    MPI_Comm_rank(MPI_COMM_WORLD, &pd->mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &pd->mpi_size);
    return reinterpret_cast<BATParticleState>(pd);
}

extern "C" void bat_io_free(BATParticleState state)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    delete pd;
}

extern "C" void bat_io_set_positions(BATParticleState state,
                                     void *positions,
                                     const uint64_t size,
                                     BATDataType btd_type)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);

    if (btd_type != BTD_VEC3_FLOAT) {
        std::cerr << "[ERROR] only vec3f positions are supported right now\n";
        throw std::runtime_error("only vec3f positions are supported right now");
    }
    const DTYPE type = static_cast<DTYPE>(btd_type);
    pd->positions = std::make_shared<BorrowedArray<glm::vec3>>(
        reinterpret_cast<glm::vec3 *>(positions), size * dtype_stride(type));
    pd->num_particles = size;
}

extern "C" void bat_io_set_attribute(BATParticleState state,
                                     const char *name,
                                     void *data,
                                     const uint64_t size,
                                     BATDataType btd_type)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);

    const DTYPE type = static_cast<DTYPE>(btd_type);
    auto array = std::make_shared<BorrowedArray<uint8_t>>(reinterpret_cast<uint8_t *>(data),
                                                          size * dtype_stride(type));
    const std::string attrib_name = name;
    auto fnd = pd->attribute_indices.find(attrib_name);
    if (fnd != pd->attribute_indices.end()) {
        pd->attributes[fnd->second] =
            Attribute(AttributeDescription(attrib_name, type), array);
    } else {
        pd->attribute_indices[attrib_name] = pd->attributes.size();
        pd->attributes.emplace_back(AttributeDescription(attrib_name, type), array);
    }
}

extern "C" void bat_io_set_local_bounds(BATParticleState state, const float *bounds)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->local_bounds = Box(bounds);
}

extern "C" void bat_io_set_bytes_per_subfile(BATParticleState state, const uint64_t size)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->max_bytes_per_subfile = size;
}

extern "C" void bat_io_set_find_best_axis(BATParticleState state, const uint32_t find_best)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->options.find_best_axis = find_best != 0;
}

extern "C" void bat_io_set_max_split_imbalance_ratio(BATParticleState state, const float ratio)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->options.max_split_imbalance_ratio = ratio;
}

extern "C" void bat_io_set_max_overfull_aggregator_factor(BATParticleState state,
                                                          const float factor)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->options.max_overfull_aggregator_factor = factor;
}

extern "C" void bat_io_set_build_local_trees(BATParticleState state,
                                             const uint32_t build_local)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->options.build_local_trees = build_local != 0;
}

extern "C" void bat_io_set_fixed_aggregation(BATParticleState state,
                                             const uint32_t num_aggregators)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->options.fixed_num_aggregators = num_aggregators;
}

extern "C" uint64_t bat_io_write(BATParticleState state, const char *cfname)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    pd->perf = WritePerformance(pd->mpi_rank, pd->mpi_size, pd->num_particles);

    const bool summit_cores = getenv("LIBBAT_SUMMIT_CORES") != nullptr;
    const int node_cores =
        summit_cores ? 168 : static_cast<int>(std::thread::hardware_concurrency());
    pd->perf.node_core_count = node_cores;

    const uint64_t bytes_per_particle = std::accumulate(
        pd->attributes.begin(),
        pd->attributes.end(),
        sizeof(glm::vec3),
        [](const uint64_t sum, const Attribute &attr) { return sum + attr.desc.stride(); });

    pd->max_particles_per_aggregator =
        std::ceil(static_cast<float>(pd->max_bytes_per_subfile) / bytes_per_particle);

    // Rank 0 needs to do some multi-threaded work here, so adjust our affinity
    // to get all cores
    std::unique_ptr<tbb::global_control> tbb_thread_control;
    cpu_set_t prev_affinity;
    {
        const int rc = sched_getaffinity(getpid(), sizeof(cpu_set_t), &prev_affinity);
        if (rc == -1) {
            perror("[ERROR] Failed to get schedular affinity for agg tree build");
        }
    }
    if (pd->mpi_rank == 0) {
        cpu_set_t all_cores;
        CPU_ZERO(&all_cores);
        for (int i = 0; i < node_cores; ++i) {
            // Note: on Summit cores 84-87 don't exist or are reserved. Same with 172-176
            if (summit_cores && i > 83 && i < 88) {
                continue;
            }
            CPU_SET(i, &all_cores);
        }
        const int rc = sched_setaffinity(getpid(), sizeof(cpu_set_t), &all_cores);
        if (rc == -1) {
            perror("[ERROR] Failed to set schedular affinity for agg tree build");
            std::cout << "[ERROR] Failed to set schedular affinity for agg tree build"
                      << "\n";
        }
        tbb_thread_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, node_cores);
    }

    const ProfilingPoint start_aggregator_assignment;
    // Compute our aggregator rank and the clients (if any) which will send
    // data to this rank. The client list is empty if this rank is not an aggregator
    auto aggregator_info = compute_aggregator_assignment(pd);
    const ProfilingPoint end_aggregator_assignment;
    if (pd->mpi_rank == 0) {
        pd->perf.total_agg_assign_time =
            elapsed_time_ms(start_aggregator_assignment, end_aggregator_assignment);
    }

    const ProfilingPoint start_agg_data;

    // TODO: Should we make subcomms for each group of aggregator and assigned ranks?
    // TODO: Need to handle chunking for Irecv/Isend if we want to send more than 2GB
    ParticleData aggregated_data;
    std::vector<Box> aggregated_bounds;
    MPI_Request req = MPI_REQUEST_NULL;
    std::vector<MPI_Request> requests;
    const bool is_aggregator = !aggregator_info.clients.empty();
    if (is_aggregator) {
        aggregated_bounds.resize(aggregator_info.clients.size());
        aggregated_data.mpi_rank = pd->mpi_rank;
        aggregated_data.mpi_size = pd->mpi_size;
        aggregated_data.num_particles =
            std::accumulate(aggregator_info.client_particle_counts.begin(),
                            aggregator_info.client_particle_counts.end(),
                            uint64_t(0),
                            [](const uint64_t acc, const int n) { return acc + uint64_t(n); });

        pd->perf.aggregator_num_clients = aggregator_info.clients.size();
        pd->perf.aggregator_particle_count = aggregated_data.num_particles;

        aggregated_data.positions =
            std::make_shared<OwnedArray<glm::vec3>>(aggregated_data.num_particles);
        for (const auto &attr : pd->attributes) {
            auto array = std::make_shared<OwnedArray<uint8_t>>(aggregated_data.num_particles *
                                                               attr.desc.stride());
            aggregated_data.attributes.emplace_back(attr.desc, array);
        }

        // Post irecvs for all incoming client data, into a single set of buffers
        // Offset in # of particles
        size_t recv_offset = 0;
        for (size_t i = 0; i < aggregator_info.clients.size(); ++i) {
            const int client = aggregator_info.clients[i];
            const int num_client_particles = aggregator_info.client_particle_counts[i];

            MPI_Irecv(
                &aggregated_bounds[i], sizeof(Box), MPI_BYTE, client, 0, MPI_COMM_WORLD, &req);
            requests.push_back(req);

            if (num_client_particles > 0) {
                MPI_Irecv(aggregated_data.positions->data() + recv_offset,
                          sizeof(glm::vec3) * num_client_particles,
                          MPI_BYTE,
                          client,
                          0,
                          MPI_COMM_WORLD,
                          &req);
                requests.push_back(req);

                for (auto &attr : aggregated_data.attributes) {
                    MPI_Irecv(attr.at(recv_offset),
                              attr.desc.stride() * num_client_particles,
                              MPI_BYTE,
                              client,
                              0,
                              MPI_COMM_WORLD,
                              &req);
                    requests.push_back(req);
                }
                recv_offset += num_client_particles;
            }
        }
    }

    // Send our data to our assigned aggregator
    MPI_Isend(&pd->local_bounds,
              sizeof(Box),
              MPI_BYTE,
              aggregator_info.aggregator_rank,
              0,
              MPI_COMM_WORLD,
              &req);
    requests.push_back(req);

    if (pd->num_particles > 0) {
        MPI_Isend(pd->positions->data(),
                  sizeof(glm::vec3) * pd->num_particles,
                  MPI_BYTE,
                  aggregator_info.aggregator_rank,
                  0,
                  MPI_COMM_WORLD,
                  &req);
        requests.push_back(req);

        for (const auto &attr : pd->attributes) {
            MPI_Isend(attr.at(0),
                      attr.desc.stride() * pd->num_particles,
                      MPI_BYTE,
                      aggregator_info.aggregator_rank,
                      0,
                      MPI_COMM_WORLD,
                      &req);
            requests.push_back(req);
        }
    }

    MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    const ProfilingPoint end_agg_data;
    pd->perf.assigned_aggregator = aggregator_info.aggregator_rank;
    pd->perf.aggregator_send_time = elapsed_time_ms(start_agg_data, end_agg_data);
    if (is_aggregator) {
        pd->perf.aggregator_recv_time = elapsed_time_ms(start_agg_data, end_agg_data);
    }

    if (pd->mpi_rank == 0 && !is_aggregator) {
        throw std::runtime_error("Rank 0 is not an aggregator!?");
    }

    // Split a comm for the aggregators to communicate with rank 0
    MPI_Comm aggregator_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, is_aggregator ? 1 : 0, pd->mpi_rank, &aggregator_comm);
    int num_aggregators = 0;
    MPI_Comm_size(aggregator_comm, &num_aggregators);

    // For each aggregator, determine how many others are on its node so we
    // can adjust the thread affinity to share the cores effectively between
    // the local aggregators.
    MPI_Comm local_aggregator_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(aggregator_comm,
                        MPI_COMM_TYPE_SHARED,
                        pd->mpi_rank,
                        MPI_INFO_NULL,
                        &local_aggregator_comm);
    int local_aggregator_rank = 0;
    int num_local_aggregators = 0;
    MPI_Comm_rank(local_aggregator_comm, &local_aggregator_rank);
    MPI_Comm_size(local_aggregator_comm, &num_local_aggregators);

    if (is_aggregator) {
        int extra_core_offset = 0;
        int threads_per_aggregator = node_cores / num_local_aggregators;
        const int remainder_threads = node_cores % num_local_aggregators;
        if (remainder_threads != 0) {
            extra_core_offset += std::min(local_aggregator_rank, remainder_threads);
            if (remainder_threads - local_aggregator_rank > 0) {
                threads_per_aggregator++;
            }
        }
        int start_core = local_aggregator_rank * threads_per_aggregator + extra_core_offset;
        int end_core = local_aggregator_rank * threads_per_aggregator + extra_core_offset +
                       threads_per_aggregator;
        // Note: on Summit cores 84-87 don't exist or are reserved. Same with 172-176
        if (summit_cores) {
            if (start_core > 83) {
                start_core += 4;
            }
            if (end_core > 83) {
                end_core += 4;
            }
        }

        cpu_set_t agg_cores;
        CPU_ZERO(&agg_cores);
        for (int i = start_core; i < end_core; ++i) {
            CPU_SET(i, &agg_cores);
        }
        const int rc = sched_setaffinity(getpid(), sizeof(cpu_set_t), &agg_cores);
        if (rc == -1) {
            perror("[ERROR] Failed to set schedular affinity for aggregator sharing");
        }

        tbb_thread_control = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, threads_per_aggregator);
        pd->perf.aggregator_num_cores = threads_per_aggregator;
    }
    MPI_Comm_free(&local_aggregator_comm);

    // Global bounds and attribute info for the top-level metadata
    Box global_bounds;
    std::vector<AttributeDescription> attribute_info;
    std::vector<uint32_t> aggregator_attribute_bitmaps;
    std::vector<glm::vec2> aggregator_attribute_ranges;

    size_t aggregator_tree_bytes = 0;
    const std::string file_name = std::string(cfname);
    if (is_aggregator) {
        std::shared_ptr<OwnedArray<glm::vec3>> positions =
            std::dynamic_pointer_cast<OwnedArray<glm::vec3>>(aggregated_data.positions);

        if (pd->options.build_local_trees) {
            BATree tree;
            if (aggregated_data.num_particles > 0) {
                const ProfilingPoint start_build_tree;
                auto builder =
                    LBATreeBuilder(std::move(positions->array), aggregated_data.attributes);
                const ProfilingPoint end_build_tree;
                tree = builder.compact();
                const ProfilingPoint end_compact_tree;

                pd->perf.bat_tree_time = elapsed_time_ms(start_build_tree, end_build_tree);
                pd->perf.bat_tree_compact_time =
                    elapsed_time_ms(end_build_tree, end_compact_tree);
                pd->perf.bat_tree_cpu_use = cpu_utilization(start_build_tree, end_build_tree);

                const std::string fname = file_name + std::to_string(pd->mpi_rank) + ".bat";

                const ProfilingPoint start_write_file;
                aggregator_tree_bytes = write_ba_tree(fname, tree);
                const ProfilingPoint end_write_file;

                pd->perf.data_write_time = elapsed_time_ms(start_write_file, end_write_file);
                pd->perf.aggregator_file_size = aggregator_tree_bytes;
            }

            // Send our bounding box to rank 0 and do a min/max across the ranks to compute
            // the global data bounds
            const Box aggregator_bounds =
                std::accumulate(aggregated_bounds.begin(),
                                aggregated_bounds.end(),
                                Box(),
                                [](const Box &b, const Box &a) { return box_union(b, a); });

            MPI_Barrier(aggregator_comm);
            const ProfilingPoint start_pbat_gather;
            MPI_Reduce(&aggregator_bounds.lower.x,
                       &global_bounds.lower.x,
                       3,
                       MPI_FLOAT,
                       MPI_MIN,
                       0,
                       aggregator_comm);
            MPI_Reduce(&aggregator_bounds.upper.x,
                       &global_bounds.upper.x,
                       3,
                       MPI_FLOAT,
                       MPI_MAX,
                       0,
                       aggregator_comm);

            if (pd->mpi_rank == 0) {
                std::transform(pd->attributes.begin(),
                               pd->attributes.end(),
                               std::back_inserter(attribute_info),
                               [](const Attribute &a) { return a.desc; });
                aggregator_attribute_ranges.resize(attribute_info.size() * num_aggregators);
                aggregator_attribute_bitmaps.resize(attribute_info.size() * num_aggregators);
            }

            const glm::vec2 empty_range(std::numeric_limits<float>::infinity(),
                                        -std::numeric_limits<float>::infinity());
            for (size_t i = 0; i < pd->attributes.size(); ++i) {
                const glm::vec2 *r = aggregated_data.num_particles > 0
                                         ? &tree.attributes[i].range
                                         : &empty_range;
                MPI_Gather(r,
                           2,
                           MPI_FLOAT,
                           &aggregator_attribute_ranges[i * num_aggregators],
                           2,
                           MPI_FLOAT,
                           0,
                           aggregator_comm);

                const uint32_t root_bitmap =
                    aggregated_data.num_particles > 0
                        ? tree.bitmap_dictionary->at(tree.node_bitmap_ids->at(i))
                        : 0;
                MPI_Gather(&root_bitmap,
                           1,
                           MPI_UNSIGNED,
                           &aggregator_attribute_bitmaps[i * num_aggregators],
                           1,
                           MPI_UNSIGNED,
                           0,
                           aggregator_comm);
            }
            const ProfilingPoint end_pbat_gather;
            pd->perf.pbat_data_gather_time =
                elapsed_time_ms(start_pbat_gather, end_pbat_gather);
        } else if (aggregated_data.num_particles > 0) {
            BinaryParticleDump dump;
            dump.points = positions;
            dump.attribs = aggregated_data.attributes;

            const std::string fname = file_name + std::to_string(pd->mpi_rank) + ".bp";

            const ProfilingPoint start_write_file;
            aggregator_tree_bytes = write_binary_particle_file(fname, dump);
            const ProfilingPoint end_write_file;

            pd->perf.data_write_time = elapsed_time_ms(start_write_file, end_write_file);
            pd->perf.aggregator_file_size = aggregator_tree_bytes;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    size_t pbat_bytes = 0;
    if (pd->mpi_rank == 0 && pd->options.build_local_trees) {
        const std::string pbat_name = file_name + ".pbat";
        aggregator_info.tree.bat_prefix = get_file_basename(file_name);

        aggregator_info.tree.bounds = global_bounds;
        aggregator_info.tree.initialize_attributes(
            attribute_info, aggregator_attribute_bitmaps, aggregator_attribute_ranges);

        const ProfilingPoint pbat_write_start;
        pbat_bytes = write_pba_tree(pbat_name, aggregator_info.tree);
        const ProfilingPoint pbat_write_end;

        pd->perf.pbat_write_time = elapsed_time_ms(pbat_write_start, pbat_write_end);
        pd->perf.pbat_file_size = pbat_bytes;
    }

    // Compute the total number of bytes we wrote
    uint64_t local_bytes = pbat_bytes + aggregator_tree_bytes;
    uint64_t global_bytes = 0;
    if (is_aggregator) {
        MPI_Reduce(&local_bytes,
                   &global_bytes,
                   1,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM,
                   0,
                   aggregator_comm);
    }
    MPI_Comm_free(&aggregator_comm);

    // Restore the original thread affinity being used by the simulation
    sched_setaffinity(getpid(), sizeof(cpu_set_t), &prev_affinity);
    tbb_thread_control = nullptr;

    MPI_Bcast(&global_bytes, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    pd->perf.total_bytes_written = global_bytes;
    return global_bytes;
}

const char *bat_io_get_performance_statistics(BATParticleState state)
{
    ParticleData *pd = reinterpret_cast<ParticleData *>(state);
    std::vector<WritePerformance> perf_stats;
    if (pd->mpi_rank == 0) {
        perf_stats.resize(pd->mpi_size, WritePerformance());
    }
    MPI_Gather(&pd->perf,
               sizeof(WritePerformance),
               MPI_BYTE,
               perf_stats.data(),
               sizeof(WritePerformance),
               MPI_BYTE,
               0,
               MPI_COMM_WORLD);

    if (pd->mpi_rank == 0) {
        using json = nlohmann::json;
        json info;
        info["mpi_size"] = pd->perf.mpi_size;
        info["node_core_count"] = pd->perf.node_core_count;
        info["total_bytes_written"] = pd->perf.total_bytes_written;

        json &rank_info = info["ranks"];
        for (const auto &perf : perf_stats) {
            json &rank = rank_info[perf.mpi_rank];
            rank["mpi_rank"] = perf.mpi_rank;
            rank["local_particle_count"] = perf.local_particle_count;
            rank["assigned_aggregator"] = perf.assigned_aggregator;
            rank["aggregator_send_time"] = perf.aggregator_send_time;

            if (perf.mpi_rank == 0) {
                rank["agg_tree_time"] = perf.agg_tree_time;
                rank["agg_tree_cpu_use"] = perf.agg_tree_cpu_use;
                rank["agg_assign_distrib_time"] = perf.agg_assign_distrib_time;
                rank["agg_info_distrib_time"] = perf.agg_info_distrib_time;
                rank["agg_list_distrib_time"] = perf.agg_list_distrib_time;
                rank["total_agg_assign_time"] = perf.total_agg_assign_time;
                rank["pbat_write_time"] = perf.pbat_write_time;
                rank["pbat_file_size"] = perf.pbat_file_size;
            }
            if (perf.aggregator_num_clients > 0) {
                rank["aggregator_num_clients"] = perf.aggregator_num_clients;
                rank["aggregator_particle_count"] = perf.aggregator_particle_count;
                rank["aggregator_recv_time"] = perf.aggregator_recv_time;
                rank["aggregator_num_cores"] = perf.aggregator_num_cores;
                rank["bat_tree_time"] = perf.bat_tree_time;
                rank["bat_tree_compact_time"] = perf.bat_tree_compact_time;
                rank["bat_tree_cpu_use"] = perf.bat_tree_cpu_use;
                rank["data_write_time"] = perf.data_write_time;
                rank["aggregator_file_size"] = perf.aggregator_file_size;
                rank["syncd_pbat_data_gather_time"] = perf.pbat_data_gather_time;
            }
        }

        pd->perf_json = info.dump();
        return pd->perf_json.c_str();
    }
    return NULL;
}
