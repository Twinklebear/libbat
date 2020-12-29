#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <mpi.h>
#include "bat_file.h"
#include "bat_handle.h"
#include "bat_io.h"
#include "binary_particle_file.h"
#include "data_type.h"
#include "generate_data.h"
#include "util.h"

using namespace std::chrono;

const static std::string USAGE =
    R"(Usage: ./adaptive_read_test <input.pbat> [options]

Options:
    -n <N>          Read the file <N> times
    -print-bounds   Print each rank's bounds
)";

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cout << USAGE << "\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    const glm::uvec3 grid_dim = compute_grid3d(world_size);
    const std::vector<std::string> args(argv, argv + argc);

    BATParticleFile file = bat_io_open(args[1].c_str());

    Box world_bounds(bat_io_get_global_bounds(file));

    const glm::vec3 bbox_lo = world_bounds.lower;
    const glm::vec3 size = world_bounds.diagonal();
    const glm::vec3 rank_brick_lower(
        rank % grid_dim.x * (size.x / grid_dim.x) + bbox_lo.x,
        ((rank / grid_dim.x) % grid_dim.y) * (size.y / grid_dim.y) + bbox_lo.y,
        rank / (grid_dim.x * grid_dim.y) * (size.z / grid_dim.z) + bbox_lo.z);
    const Box rank_bounds(rank_brick_lower, rank_brick_lower + size / glm::vec3(grid_dim));

    bool print_bounds = false;
    size_t num_reads = 1;
    for (size_t i = 2; i < args.size(); ++i) {
        if (args[i] == "-n") {
            num_reads = std::stoull(args[++i]);
        } else if (args[i] == "-print-bounds") {
            print_bounds = true;
        } else {
            std::cout << "Unrecognized command line option " << args[i] << "\n";
            std::exit(1);
        }
    }

    if (rank == 0) {
        std::cout << "World bounds: " << world_bounds << std::endl;
    }
    if (print_bounds) {
        for (int i = 0; i < world_size; ++i) {
            if (i == rank) {
                std::cout << "Rank " << rank << " bounds " << rank_bounds << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    bat_io_set_read_bounds(file, &rank_bounds.lower.x);
    for (uint32_t i = 0; i < num_reads; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = steady_clock::now();
        const uint64_t particles_read = bat_io_read(file);
        auto end = steady_clock::now();

        uint64_t total_particles_read = 0;
        MPI_Reduce(&particles_read,
                   &total_particles_read,
                   1,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        if (rank == 0) {
            if (total_particles_read != bat_io_get_num_points(file)) {
                std::cout << "[ERROR] Read did not return all points in the file, got "
                          << total_particles_read << "/" << bat_io_get_num_points(file)
                          << "\n";
            }
            const size_t read_time = duration_cast<milliseconds>(end - start).count();
            // TODO: Need to compute the total number of bytes read
            size_t bytes_read = total_particles_read * sizeof(glm::vec3);
            for (size_t j = 0; j < bat_io_get_num_attributes(file); ++j) {
                const char *attr_name = nullptr;
                glm::vec2 range;
                BATDataType dtype;
                bat_io_get_attribute_description(file, j, &attr_name, &range.x, &dtype);

                bytes_read += total_particles_read * dtype_stride((DTYPE)dtype);
            }
            const float bandwidth = (bytes_read * 1e-6f) / (read_time * 1e-3f);
            std::cout << "Total read time: " << read_time << "ms\n"
                      << "Total bytes read: " << bytes_read << "b\n"
                      << "Read bandwidth: " << bandwidth << "MB/s\n=======\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const glm::vec3 *positions = nullptr;
        uint64_t n_positions = 0;
        BATDataType pos_type = BTD_UNKNOWN;
        bat_io_get_positions(
            file, reinterpret_cast<const void **>(&positions), &n_positions, &pos_type);
        for (size_t i = 0; i < n_positions; ++i) {
            if (!rank_bounds.contains_point(positions[i])) {
                std::cout << "[ERROR] Rank " << rank << " with bounds " << rank_bounds
                          << " got not contained point [" << i
                          << "] = " << glm::to_string(positions[i]) << "\n";
            }
        }
    }

    bat_io_close(file);

    MPI_Finalize();
    return 0;
}
