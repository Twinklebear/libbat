#include "bat_read.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <mpi.h>
#include "aggregator_assignment.h"
#include "bat_file.h"
#include "binary_particle_file.h"
#include "borrowed_array.h"
#include "lba_tree_builder.h"
#include "mpi_send_recv.h"
#include "particle_data.h"
#include "pbat_file.h"
#include "profiling.h"

extern "C" BATParticleFile bat_io_open(const char *file_name)
{
    auto *pf = new ParticleFile;
    MPI_Comm_rank(MPI_COMM_WORLD, &pf->mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &pf->mpi_size);
    pf->tree = map_pba_tree(file_name);
    return reinterpret_cast<BATParticleFile>(pf);
}

extern "C" void bat_io_close(BATParticleFile file)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    delete pf;
}

extern "C" const float *bat_io_get_global_bounds(BATParticleFile file)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    return &pf->tree.bounds.lower.x;
}

extern "C" uint64_t bat_io_get_num_attributes(BATParticleFile file)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    return pf->tree.attributes.size();
}

extern "C" uint64_t bat_io_get_num_points(BATParticleFile file)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    return pf->tree.num_points;
}

// Query the description of one of the attributes (name, value range, data type)
extern "C" void bat_io_get_attribute_description(
    BATParticleFile file, uint64_t id, const char **name, float *range, BATDataType *type)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    const AttributeDescription &desc = pf->tree.attributes[id];
    *name = desc.name.c_str();
    range[0] = desc.range.x;
    range[1] = desc.range.y;
    *type = static_cast<BATDataType>(desc.data_type);
}

// bounds should be [lower_x, lower_y, lower_z, upper_x, upper_y, upper_z]
extern "C" void bat_io_set_read_bounds(BATParticleFile file, const float *bounds)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    pf->read_bounds = Box(bounds);
}

// Read the file, the loaded particles can then be fetched from the file state
// returns the number of particles read on this rank
// NOTE: This does a checkpoint-restart style read only right now
extern "C" uint64_t bat_io_read(BATParticleFile file)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    // Determine the mapping of bat file(s) (aggregator) to ranks to assign
    // each rank zero or more bat files which it's responsible for reading (basically
    // the inverse of being an aggregator). These ranks open their assigned files
    // and receive box queries from the ranks whose query region overlaps theirs. This
    // can be done by traversing the aggregation tree to find which sub files we would
    // need to open, and instead of opening them sending the query to the owner.
    // This will need a new traversal mode added to the aggregation tree.
    // Each rank sends its query out and receives back results from each read aggregator
    std::vector<std::vector<uint32_t>> read_aggregator_assignments(pf->mpi_size,
                                                                   std::vector<uint32_t>{});
    // Map of subtree ID to rank assigned to its file
    phmap::flat_hash_map<uint32_t, uint32_t> read_subtree_ranks;
    if (pf->mpi_size < pf->tree.num_aggregators) {
        const uint32_t aggregators_per_rank = pf->tree.num_aggregators / pf->mpi_size;
        const int remainder_aggregators_per_rank = pf->tree.num_aggregators % pf->mpi_size;
        if (pf->mpi_rank == 0) {
            std::cout << "Read with " << pf->tree.num_aggregators << " aggregators to "
                      << pf->mpi_size << " ranks, agg per-rank: " << aggregators_per_rank
                      << ", remainder " << remainder_aggregators_per_rank << "\n";
        }
        size_t node = 0;
        for (int rank = 0; rank < pf->mpi_size; ++rank) {
            auto &aggregators = read_aggregator_assignments[rank];
            size_t aggs_for_rank = aggregators_per_rank;
            if (remainder_aggregators_per_rank - rank > 0) {
                aggs_for_rank++;
            }
            for (; node < pf->tree.nodes->size(); ++node) {
                const auto &n = pf->tree.nodes->at(node);
                if (!n.is_leaf()) {
                    continue;
                }
                aggregators.push_back(n.aggregator_rank);
                read_subtree_ranks[n.aggregator_rank] = rank;
                if (aggregators.size() == aggs_for_rank) {
                    ++node;
                    break;
                }
            }
        }
    } else {
        const uint32_t ranks_per_aggregator = pf->mpi_size / pf->tree.num_aggregators;
        if (pf->mpi_rank == 0) {
            std::cout << "Read with " << pf->tree.num_aggregators << " aggregators to "
                      << pf->mpi_size << " ranks, ranks per-agg: " << ranks_per_aggregator
                      << "\n";
        }

        size_t agg_id = 0;
        for (size_t i = 0; i < pf->tree.nodes->size(); ++i) {
            const auto &n = pf->tree.nodes->at(i);
            if (!n.is_leaf()) {
                continue;
            }

            read_aggregator_assignments[agg_id * ranks_per_aggregator].push_back(
                n.aggregator_rank);
            read_subtree_ranks[n.aggregator_rank] = agg_id * ranks_per_aggregator;
            ++agg_id;
        }
    }
    const bool is_read_aggregator = !read_aggregator_assignments[pf->mpi_rank].empty();

    // For missing bat files (only the case when using non-adaptive agg for testing) we want
    // to catch the error and not insert it. Then if a rank requests a non-existant
    // file from us we just return 0 particles
    for (const auto &ag : read_aggregator_assignments[pf->mpi_rank]) {
        const std::string fname = pf->tree.bat_prefix + std::to_string(ag) + ".bat";
        try {
            pf->ba_trees[ag] = std::make_shared<BATree>(map_ba_tree(fname));
        } catch (const std::runtime_error &e) {
            std::cout << "[ERROR] Could not open " << fname << ": " << e.what()
                      << " (did you disable adaptive aggregation?)\n";
        }
    }

    const auto subtree_ids = pf->tree.get_overlapped_subtree_ids(pf->read_bounds);

    // Each rank will wait for requests coming from those who need data from
    // its subregion. Once a rank has gotten all of its data it will call the Ibarrier,
    // when the Ibarrier is completed we know all ranks have their data and can exit the loop
    // How would the scalability of this compare to a simpler strategy where each rank
    // set its overlapped subtree ids to rank 0, which then sent that ranks ID out
    // to the corresponding aggregators? With the latter approach it'd also be possible to
    // build subcommunicators for each aggregator read group
    const int box_request_tag = 7532;
    const int query_result_tag = box_request_tag + 1;
    // Send our box query to the ranks assigned to read the file we need
    std::vector<ISend> send_query;
    phmap::flat_hash_map<int, size_t> queried_ranks;
    for (const auto &id : subtree_ids) {
        int rank = read_subtree_ranks[id];
        auto fnd = queried_ranks.find(rank);
        if (fnd == queried_ranks.end()) {
            queried_ranks[rank] = send_query.size();
            if (rank != pf->mpi_rank) {
                send_query.emplace_back(&pf->read_bounds.lower.x,
                                        6,
                                        MPI_FLOAT,
                                        rank,
                                        box_request_tag,
                                        MPI_COMM_WORLD,
                                        true);
            } else {
                send_query.push_back(ISend());
            }
        }
    }

    // Queue the recv for the particle count returned by the rank
    std::vector<IRecv> recv_particle_counts(send_query.size(), IRecv());
    for (const auto &qr : queried_ranks) {
        if (qr.first != pf->mpi_rank) {
            recv_particle_counts[qr.second] = IRecv::recv<uint64_t>(
                1, MPI_UNSIGNED_LONG_LONG, qr.first, query_result_tag, MPI_COMM_WORLD);
        }
    }
    std::vector<int> completed_queries(send_query.size(), 0);
    std::vector<ISend> query_responses;

    std::vector<OwnedArrayHandle<uint8_t>> attribute_data;
    std::vector<IRecv> recv_particle_data;

    bool query_sends_complete = false;
    bool count_recvs_complete = false;
    bool data_receive_complete = false;

    // Receive and handle box queries if we're a read aggregator, otherwise wait for our data
    // to come back
    MPI_Request all_ranks_done_barrier = MPI_REQUEST_NULL;
    int all_ranks_done = 0;
    while (!all_ranks_done) {
        // Check for our data to come back if we haven't received it already
        if (!query_sends_complete) {
            query_sends_complete = std::all_of(
                send_query.begin(), send_query.end(), [](ISend &s) { return s.complete(); });
        }

        if (!count_recvs_complete) {
            count_recvs_complete = std::all_of(recv_particle_counts.begin(),
                                               recv_particle_counts.end(),
                                               [](IRecv &r) { return r.complete(); });

            if (count_recvs_complete) {
                const uint64_t total_recv_particles = std::accumulate(
                    recv_particle_counts.begin(),
                    recv_particle_counts.end(),
                    uint64_t(0),
                    [](const uint64_t &acc, const IRecv &r) {
                        if (r.data) {
                            return acc + *reinterpret_cast<uint64_t *>(r.data->data());
                        }
                        return acc;
                    });
                data_receive_complete = total_recv_particles == 0;
                // Allocate space for the data we'll receive and start receiving it.
                // Any data from the rank's own file will be read and appended at the end of
                // the function
                pf->positions = std::make_shared<OwnedArray<glm::vec3>>(total_recv_particles);
                pf->attributes.clear();
                for (const auto &attr : pf->tree.attributes) {
                    attribute_data.push_back(std::make_shared<OwnedArray<uint8_t>>(
                        total_recv_particles * attr.stride()));
                    pf->attributes.emplace_back(attr, attribute_data.back());
                }

                size_t recv_count_offset = 0;
                for (const auto &r : recv_particle_counts) {
                    if (!r.data) {
                        continue;
                    }

                    const uint64_t recv_count = *reinterpret_cast<uint64_t *>(r.data->data());
                    recv_particle_data.emplace_back(pf->positions->data() + recv_count_offset,
                                                    3 * recv_count,
                                                    MPI_FLOAT,
                                                    r.src,
                                                    query_result_tag,
                                                    MPI_COMM_WORLD);
                    for (size_t i = 0; i < pf->attributes.size(); ++i) {
                        auto &data = attribute_data[i];
                        const auto &desc = pf->attributes[i].desc;
                        recv_particle_data.emplace_back(
                            data->data() + recv_count_offset * desc.stride(),
                            desc.stride() * recv_count,
                            MPI_BYTE,
                            r.src,
                            query_result_tag,
                            MPI_COMM_WORLD);
                    }
                    recv_count_offset += recv_count;
                }
            }
        }

        if (!recv_particle_data.empty()) {
            // Check if data receives are done
            data_receive_complete = std::all_of(recv_particle_data.begin(),
                                                recv_particle_data.end(),
                                                [](IRecv &r) { return r.complete(); });
        }

        const bool rank_done =
            query_sends_complete && count_recvs_complete && data_receive_complete;
        if (rank_done && all_ranks_done_barrier == MPI_REQUEST_NULL) {
            MPI_Ibarrier(MPI_COMM_WORLD, &all_ranks_done_barrier);
        }

        // Check for any incoming queries to this rank, if we're a read aggregator
        if (is_read_aggregator) {
            MPI_Status status;
            int got_query = 0;
            MPI_Iprobe(MPI_ANY_SOURCE, box_request_tag, MPI_COMM_WORLD, &got_query, &status);
            // If we got a query receive the box and handle it
            if (got_query) {
                Box query_box;
                MPI_Recv(&query_box.lower.x,
                         6,
                         MPI_FLOAT,
                         status.MPI_SOURCE,
                         box_request_tag,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);

                uint64_t num_queried = 0;
                auto positions = std::make_shared<OwnedArray<uint8_t>>();
                std::vector<OwnedArrayHandle<uint8_t>> send_attribute_data;
                for (size_t i = 0; i < pf->tree.attributes.size(); ++i) {
                    send_attribute_data.push_back(std::make_shared<OwnedArray<uint8_t>>());
                }

                for (auto &t : pf->ba_trees) {
                    t.second->query_box(query_box,
                                        nullptr,
                                        1.f,
                                        [&](const size_t id,
                                            const glm::vec3 &pos,
                                            const std::vector<Attribute> &attr) {
                                            num_queried++;
                                            positions->push_back(&pos, sizeof(glm::vec3));
                                            for (size_t i = 0; i < attr.size(); ++i) {
                                                send_attribute_data[i]->push_back(
                                                    attr[i].at(id), attr[i].desc.stride());
                                            }
                                        });
                }
                query_responses.emplace_back(&num_queried,
                                             1,
                                             MPI_UNSIGNED_LONG_LONG,
                                             status.MPI_SOURCE,
                                             query_result_tag,
                                             MPI_COMM_WORLD,
                                             false);

                query_responses.emplace_back(positions,
                                             positions->size() / sizeof(float),
                                             MPI_FLOAT,
                                             status.MPI_SOURCE,
                                             query_result_tag,
                                             MPI_COMM_WORLD);
                for (const auto &attr : send_attribute_data) {
                    query_responses.emplace_back(attr,
                                                 attr->size_bytes(),
                                                 MPI_BYTE,
                                                 status.MPI_SOURCE,
                                                 query_result_tag,
                                                 MPI_COMM_WORLD);
                }
            }
        }

        if (!query_responses.empty()) {
            auto end = std::partition(query_responses.begin(),
                                      query_responses.end(),
                                      [](ISend &s) { return !s.complete(); });
            query_responses.erase(end, query_responses.end());
        }

        if (all_ranks_done_barrier != MPI_REQUEST_NULL) {
            MPI_Test(&all_ranks_done_barrier, &all_ranks_done, MPI_STATUS_IGNORE);
        }
    }

    // std::vector<uint64_t> particle_counts(send_query.size(), 0);
    // If we're querying some of our local subtrees, query them now
    auto query_local = queried_ranks.find(pf->mpi_rank);
    if (query_local != queried_ranks.end()) {
        for (auto &t : pf->ba_trees) {
            t.second->query_box(pf->read_bounds,
                                nullptr,
                                1.f,
                                [&](const size_t id,
                                    const glm::vec3 &pos,
                                    const std::vector<Attribute> &attr) {
                                    // particle_counts[query_local->second]++;
                                    pf->positions->push_back(pos);
                                    for (size_t i = 0; i < attr.size(); ++i) {
                                        attribute_data[i]->push_back(attr[i].at(id),
                                                                     attr[i].desc.stride());
                                    }
                                });
        }
    }

#if 0
    // TODO: For debugging only
    for (const auto &qr : queried_ranks) {
        if (qr.first != pf->mpi_rank) {
            particle_counts[qr.second] =
                *reinterpret_cast<uint64_t *>(recv_particle_counts[qr.second].data->data());
        }

        std::cout << "Rank " << pf->mpi_rank << " queried from rank " << qr.first << " "
                  << particle_counts[qr.second] << " particles\n";
    }
#endif

    return pf->positions->size();
}

// Get the pointer to the positions data read from the file
extern "C" void bat_io_get_positions(BATParticleFile file,
                                     const void **positions,
                                     uint64_t *size,
                                     BATDataType *type)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    if (pf->positions) {
        *positions = reinterpret_cast<void *>(pf->positions->data());
        *size = pf->positions->size();
        *type = BTD_VEC3_FLOAT;
    } else {
        *positions = NULL;
        *size = 0;
        *type = BTD_UNKNOWN;
    }
}

// Get the pointer to the attribute data read from the file
extern "C" void bat_io_get_attribute(BATParticleFile file,
                                     const char *name,
                                     const void **data,
                                     uint64_t *size,
                                     BATDataType *type)
{
    auto *pf = reinterpret_cast<ParticleFile *>(file);
    auto fnd = pf->tree.attrib_ids.find(std::string(name));
    if (fnd != pf->tree.attrib_ids.end()) {
        const Attribute &attr = pf->attributes[fnd->second];
        *data = reinterpret_cast<void *>(attr.data->data());
        *size = attr.data->size();
        *type = static_cast<BATDataType>(attr.desc.data_type);
    } else {
        *data = NULL;
        *size = 0;
        *type = BTD_UNKNOWN;
    }
}
