#include "aggregator_assignment.h"
#include <numeric>
#include <mpi.h>
#include <tbb/parallel_for.h>
#include "aggregation_tree_builder.h"
#include "profiling.h"

AggregatorInfo::AggregatorInfo(const int rank) : aggregator_rank(rank) {}

AggregatorInfo compute_aggregator_assignment(ParticleData *data)
{
    // Collect the number of particles each rank has onto rank 0, and
    // the bottom corner of each rank's bounds for building the aggregation tree
    // TODO this assumes each rank's bounds are the same, which is not the case for AMR
    // TODO each rank should also send its bounding box (we send this later anyways)
    const RankPoint rank_point(data->mpi_rank, data->local_bounds.lower, data->num_particles);
    std::vector<RankPoint> rank_points;
    if (data->mpi_rank == 0) {
        rank_points.resize(data->mpi_size);
    }
    MPI_Gather(&rank_point,
               sizeof(RankPoint),
               MPI_BYTE,
               rank_points.data(),
               sizeof(RankPoint),
               MPI_BYTE,
               0,
               MPI_COMM_WORLD);

    // The lists of clients for each aggregator on rank 0
    std::vector<AggregatorInfo> aggregator_info;
    AggregationTree aggregation_tree;

    // [our aggregator rank, num clients we'll receive from]
    std::array<int, 2> aggregator_assignment = {0, 0};
    if (data->mpi_rank == 0) {
        const ProfilingPoint start_agg_tree_build;
        aggregation_tree = AggregationTreeBuilder(
                               rank_points, data->max_particles_per_aggregator, data->options)
                               .compact();
        const ProfilingPoint end_agg_tree_build;
        data->perf.agg_tree_time = elapsed_time_ms(start_agg_tree_build, end_agg_tree_build);
        data->perf.agg_tree_cpu_use =
            cpu_utilization(start_agg_tree_build, end_agg_tree_build);

        // Scatter aggregator rank assignments to the ranks
        std::vector<int> aggregator_assignments(data->mpi_size * 2, 0);
        aggregator_info.resize(aggregation_tree.leaf_indices->size());

        auto tree_nodes =
            std::dynamic_pointer_cast<OwnedArray<AggregationKdNode>>(aggregation_tree.nodes);

        const ProfilingPoint start_agg_info_distrib;
        tbb::parallel_for(size_t(0), aggregation_tree.leaf_indices->size(), [&](size_t i) {
            AggregationKdNode &leaf = tree_nodes->at(aggregation_tree.leaf_indices->at(i));
            int aggregator_rank = (data->mpi_size / aggregation_tree.leaf_indices->size()) * i;

            // If we know we're doing FPP just have each rank be its own aggregator
            if (aggregation_tree.leaf_indices->size() == data->mpi_size) {
                aggregator_rank =
                    aggregation_tree.primitives->at(leaf.prim_indices_offset).rank;
            }

            aggregator_assignments[aggregator_rank * 2 + 1] = leaf.get_num_prims();
            aggregator_info[i] = AggregatorInfo(aggregator_rank);

            for (uint32_t j = 0; j < leaf.get_num_prims(); ++j) {
                const auto rank_point =
                    aggregation_tree.primitives->at(leaf.prim_indices_offset + j);

                aggregator_assignments[rank_point.rank * 2] = aggregator_rank;
                aggregator_info[i].clients.push_back(rank_point.rank);
                aggregator_info[i].client_particle_counts.push_back(rank_point.num_particles);
            }

            // Now, replace the prim ID offset with the aggregator rank to produce
            // the aggregator list tree
            leaf.aggregator_rank = aggregator_rank;
        });

        MPI_Scatter(aggregator_assignments.data(),
                    2,
                    MPI_INT,
                    aggregator_assignment.data(),
                    2,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);
        const ProfilingPoint end_agg_info_distrib;
        data->perf.agg_info_distrib_time =
            elapsed_time_ms(start_agg_info_distrib, end_agg_info_distrib);
    } else {
        MPI_Scatter(
            nullptr, 2, MPI_INT, aggregator_assignment.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Our local aggregator info
    AggregatorInfo my_aggregator_info(aggregator_assignment[0]);
    my_aggregator_info.tree = aggregation_tree;

    const ProfilingPoint start_agg_list_send;
    std::vector<MPI_Request> requests;
    if (aggregator_assignment[1] != 0) {
        my_aggregator_info.clients.resize(aggregator_assignment[1], 0);
        my_aggregator_info.client_particle_counts.resize(aggregator_assignment[1], 0);

        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Irecv(my_aggregator_info.clients.data(),
                  my_aggregator_info.clients.size(),
                  MPI_INT,
                  0,
                  0,
                  MPI_COMM_WORLD,
                  &req);
        requests.push_back(req);

        MPI_Irecv(my_aggregator_info.client_particle_counts.data(),
                  my_aggregator_info.client_particle_counts.size(),
                  MPI_INT,
                  0,
                  0,
                  MPI_COMM_WORLD,
                  &req);
        requests.push_back(req);
    }

    if (data->mpi_rank == 0) {
        // Send the aggregator client lists to the aggregators
        for (const auto &agg : aggregator_info) {
            MPI_Request req = MPI_REQUEST_NULL;
            MPI_Isend(agg.clients.data(),
                      agg.clients.size(),
                      MPI_INT,
                      agg.aggregator_rank,
                      0,
                      MPI_COMM_WORLD,
                      &req);
            requests.push_back(req);

            MPI_Isend(agg.client_particle_counts.data(),
                      agg.client_particle_counts.size(),
                      MPI_INT,
                      agg.aggregator_rank,
                      0,
                      MPI_COMM_WORLD,
                      &req);
            requests.push_back(req);
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    } else if (!requests.empty()) {
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const ProfilingPoint end_agg_list_send;
    if (data->mpi_rank == 0) {
        data->perf.agg_list_distrib_time =
            elapsed_time_ms(start_agg_list_send, end_agg_list_send);
    }
    return my_aggregator_info;
}
