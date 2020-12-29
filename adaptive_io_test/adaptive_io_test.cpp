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
#include "generate_data.h"
#include "util.h"

using namespace std::chrono;

const static std::string USAGE =
    R"(Usage: ./adaptive_io_test <MB per file> <output> [options]

Options:
    -bat <file.bat>                 Use the data from file.bat as input

    -uniform <N>                    Generate <N> uniformly distributed particles per-rank

    -gradient <N>                   Generate a gradient distribution along x with <N> total particles

    -exp-gradient <N>               Generate an exponential gradient distribution along x with <N> total particles

    -spheres <#spheres> <N>         Generate a set of Gaussian sphere "emitters". Uniformly places #spheres spheres
                                    in the domain each with a Gaussian falloff density, and generates a total of N
                                    particles distributed among the spheres.

    -cut-box <w> <h> <d> <N>        Generate a uniform distribution of N particles with some WxHxD box cut out of
                                    the data. The box size should be specified in normalized coordinates [0, 1].

    -double                         Generate double precision attributes

    -non-spatial-attrib <type> <N>  Generate N non-spatially correlated attributes using distribution <type>. Type
                                    can be uniform or normal.

    -spatial-attrib <type> <N>      Generate N spatially correlated attributes using type <type> (gradient or sphere). 

    -inject <N>                     Inject N particles after each timestep, using the selected generators. Note
                                    that the uniform generator will inject N per-rank, the others are global.

    -advect <x> <y> <z>             Advect the existing particles by the vector <x> <y> <z> each timestep.

    -best-split                     Check all possible split axes to select the best split
                                    when building the aggregation tree.

    -n <N>                          Write the file <N> times, to a different timestep path
                                    each time.

    -seed <x>                       Set the seed for the RNG used by the generators, to allow
                                    consistent/reproducible data layouts for benchmarks.

    -fixed-aggregation              Disable the data-adaptive aggregation and use a fixed number of aggregators

    -raw                            Skip building local trees and dump raw particles only.

    -grid_mode (x/y/z)              Specify the axes to partition on as a string of xyz (e.g., x, xy, yz, xyz)
)";

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << USAGE << "\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    int rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::vector<std::string> args(argv, argv + argc);
    const uint64_t max_bytes_per_file = std::stof(args[1]) * 1000000;
    const std::string file_prefix = args[2];

    // Check if we're setting a global seed, if so remove the args
    {
        auto global_seed = std::find(args.begin(), args.end(), std::string("-seed"));
        if (global_seed != args.end()) {
            set_global_seed(std::stoi(*(global_seed + 1)));
            args.erase(global_seed, global_seed + 2);
        }
    }

    // Check if we're setting a custom grid mode
    glm::uvec3 grid_dim = compute_grid3d(world_size);
    {
        auto grid_mode = std::find(args.begin(), args.end(), std::string("-grid_mode"));
        if (grid_mode != args.end()) {
            const std::string grid_str = *(grid_mode + 1);
            args.erase(grid_mode, grid_mode + 2);

            std::vector<int> axes;
            for (size_t i = 0; i < grid_str.size(); ++i) {
                switch (grid_str[i]) {
                case 'x':
                    axes.push_back(0);
                    break;
                case 'y':
                    axes.push_back(1);
                    break;
                case 'z':
                    axes.push_back(2);
                    break;
                default:
                    std::cout << "Invalid grid str: " << grid_str << "\n";
                    return 1;
                }
            }

            if (axes.size() == 1) {
                grid_dim = glm::uvec3(1);
                grid_dim[axes[0]] = world_size;
            } else if (axes.size() == 2) {
                grid_dim = glm::uvec3(1);
                glm::uvec2 grid2d = compute_grid2d(world_size);
                grid_dim[axes[0]] = grid2d[0];
                grid_dim[axes[1]] = grid2d[1];
            } else if (axes.size() > 3) {
                std::cout << "Grid str is too long! Str: " << grid_str << "\n";
                return 1;
            }
            if (rank == 0) {
                std::cout << "Grid mode str " << grid_str
                          << ", grid: " << glm::to_string(grid_dim) << "\n";
            }
        }
    }

    Box world_bounds = Box(glm::vec3(0), glm::vec3(grid_dim));
    Box rank_bounds;
    std::vector<std::shared_ptr<Generator>> generators;
    std::vector<std::shared_ptr<AttributeGenerator>> attrib_generators;
    std::vector<glm::vec3> block;
    std::vector<Attribute> block_attributes;
    std::vector<OwnedArrayHandle<uint8_t>> attr_data;
    std::vector<size_t> particles_generated_per_generator;

    glm::vec3 advect(0.f);
    bool advect_particles = false;
    bool got_bat_file = false;
    bool best_split = false;
    size_t num_writes = 1;
    size_t inject = 0;
    bool dump_raw = false;
    bool double_precision_attribs = false;
    bool fixed_aggregation = 0;
    for (size_t i = 3; i < args.size(); ++i) {
        if (args[i] == "-bat") {
            got_bat_file = true;
            BATHandle tree = BATHandle(args[++i]);
            world_bounds = tree.bounds();
            const glm::vec3 bbox_lo = world_bounds.lower;
            const glm::vec3 bbox_hi = world_bounds.upper;
            const glm::vec3 size = world_bounds.diagonal();
            const glm::vec3 rank_brick_lower(
                rank % grid_dim.x * (size.x / grid_dim.x) + bbox_lo.x,
                ((rank / grid_dim.x) % grid_dim.y) * (size.y / grid_dim.y) + bbox_lo.y,
                rank / (grid_dim.x * grid_dim.y) * (size.z / grid_dim.z) + bbox_lo.z);

            rank_bounds = Box(rank_brick_lower, rank_brick_lower + size / glm::vec3(grid_dim));

            // Read the block of particles for this rank
            for (const auto &attr : tree.attributes) {
                block_attributes.push_back(
                    Attribute(attr, std::make_shared<OwnedArray<uint8_t>>()));
                attr_data.push_back(std::dynamic_pointer_cast<OwnedArray<uint8_t>>(
                    block_attributes.back().data));
            }

            tree.query_box(rank_bounds,
                           nullptr,
                           1.f,
                           [&](const size_t id,
                               const glm::vec3 &pos,
                               const std::vector<Attribute> &attribs) {
                               block.emplace_back(pos);
                               for (size_t i = 0; i < attribs.size(); ++i) {
                                   const uint8_t *val = attribs[i].at(id);
                                   // Not the best perf way to copy these out, but meh
                                   for (size_t j = 0; j < attribs[i].desc.stride(); ++j) {
                                       attr_data[i]->push_back(val[j]);
                                   }
                               }
                           });
        } else if (args[i] == "-uniform") {
            const glm::vec3 rank_brick_lower(rank % grid_dim.x,
                                             (rank / grid_dim.x) % grid_dim.y,
                                             rank / (grid_dim.x * grid_dim.y));
            rank_bounds = Box(rank_brick_lower, rank_brick_lower + glm::vec3(1.f));

            auto gen = std::make_shared<UniformGenerator>(world_bounds, rank_bounds);
            const size_t generated =
                gen->generate(std::stoull(args[++i]), glm::vec3(0.f), block);
            generators.push_back(gen);
            particles_generated_per_generator.push_back(generated);
        } else if (args[i] == "-gradient") {
            const glm::vec3 rank_brick_lower(rank % grid_dim.x,
                                             (rank / grid_dim.x) % grid_dim.y,
                                             rank / (grid_dim.x * grid_dim.y));
            rank_bounds = Box(rank_brick_lower, rank_brick_lower + glm::vec3(1.f));

            auto gen = std::make_shared<LinearGradientGenerator>(world_bounds, rank_bounds);
            const size_t generated =
                gen->generate(std::stoull(args[++i]), glm::vec3(0.f), block);
            generators.push_back(gen);
            particles_generated_per_generator.push_back(generated);
        } else if (args[i] == "-exp-gradient") {
            const glm::vec3 rank_brick_lower(rank % grid_dim.x,
                                             (rank / grid_dim.x) % grid_dim.y,
                                             rank / (grid_dim.x * grid_dim.y));
            rank_bounds = Box(rank_brick_lower, rank_brick_lower + glm::vec3(1.f));

            auto gen =
                std::make_shared<ExponentialGradientGenerator>(world_bounds, rank_bounds);
            const size_t generated =
                gen->generate(std::stoull(args[++i]), glm::vec3(0.f), block);
            generators.push_back(gen);
            particles_generated_per_generator.push_back(generated);
        } else if (args[i] == "-spheres") {
            const glm::vec3 rank_brick_lower(rank % grid_dim.x,
                                             (rank / grid_dim.x) % grid_dim.y,
                                             rank / (grid_dim.x * grid_dim.y));
            rank_bounds = Box(rank_brick_lower, rank_brick_lower + glm::vec3(1.f));

            const size_t n_spheres = std::stoull(args[++i]);
            auto gen = std::make_shared<SphereGenerator>(world_bounds, rank_bounds, n_spheres);
            const size_t total_particles = std::stoull(args[++i]);
            const size_t generated = gen->generate(total_particles, glm::vec3(0.f), block);
            generators.push_back(gen);
            particles_generated_per_generator.push_back(generated);
        } else if (args[i] == "-cut-box") {
            const glm::vec3 rank_brick_lower(rank % grid_dim.x,
                                             (rank / grid_dim.x) % grid_dim.y,
                                             rank / (grid_dim.x * grid_dim.y));
            rank_bounds = Box(rank_brick_lower, rank_brick_lower + glm::vec3(1.f));
            glm::vec3 box_size;
            box_size.x = std::stof(args[++i]);
            box_size.y = std::stof(args[++i]);
            box_size.z = std::stof(args[++i]);

            auto gen = std::make_shared<CutBoxGenerator>(world_bounds, rank_bounds, box_size);
            const size_t total_particles = std::stoull(args[++i]);
            const size_t generated = gen->generate(total_particles, glm::vec3(0.f), block);
            generators.push_back(gen);
            particles_generated_per_generator.push_back(generated);
        } else if (args[i] == "-non-spatial-attrib") {
            const std::string type = args[++i];
            const size_t n_attribs = std::stoull(args[++i]);
            for (size_t j = 0; j < n_attribs; ++j) {
                attrib_generators.push_back(std::make_shared<NonSpatialAttribute>(type));
            }
        } else if (args[i] == "-spatial-attrib") {
            const std::string type = args[++i];
            const size_t n_attribs = std::stoull(args[++i]);
            for (size_t j = 0; j < n_attribs; ++j) {
                attrib_generators.push_back(
                    std::make_shared<SpatialAttribute>(type, world_bounds));
            }
        } else if (args[i] == "-inject") {
            inject = std::stoull(args[++i]);
        } else if (args[i] == "-advect") {
            advect_particles = true;
            advect.x = std::stof(args[++i]);
            advect.y = std::stof(args[++i]);
            advect.z = std::stof(args[++i]);
        } else if (args[i] == "-best-split") {
            best_split = true;
        } else if (args[i] == "-n") {
            num_writes = std::stoull(args[++i]);
        } else if (args[i] == "-raw") {
            dump_raw = true;
        } else if (args[i] == "-double") {
            double_precision_attribs = true;
        } else if (args[i] == "-fixed-aggregation") {
            fixed_aggregation = true;
        } else {
            std::cout << "Unrecognized command line option " << args[i] << "\n";
            std::exit(1);
        }
    }

    if (inject > 0 && generators.empty()) {
        std::cout
            << "Error: Invalid configuration: cannot inject particles without generators!\n";
        throw std::runtime_error(
            "Error: Invalid configuration: cannot inject particles without generators!");
    }

    if (got_bat_file && (!generators.empty() || !attrib_generators.empty())) {
        std::cout << "Error: BAT file cannot be combined with generators\n";
        throw std::runtime_error("Error: BAT file cannot be combined with generators");
    }

    if (advect_particles && generators.empty()) {
        std::cout << "Error: Particle generation must be used for particle advection\n";
        throw std::runtime_error(
            "Error: Particle generation must be used for particle advection");
    }

    if (rank == 0) {
        std::cout << "Run with bytes per-file: " << max_bytes_per_file << "\n"
                  << "Computation Grid: " << glm::to_string(grid_dim) << "\n";
        if (fixed_aggregation != 0) {
            std::cout << "Adaptive aggregation disabled, using fixed aggregation\n";
        }
        if (double_precision_attribs) {
            std::cout << "Using double precision attribs\n";
        }
    }

    if (!attrib_generators.empty()) {
        for (size_t i = 0; i < attrib_generators.size(); ++i) {
            const std::string attr_name = attrib_generators[i]->name() + std::to_string(i);

            const DTYPE type = double_precision_attribs ? FLOAT_64 : FLOAT_32;
            block_attributes.push_back(Attribute(AttributeDescription(attr_name, type),
                                                 std::make_shared<OwnedArray<uint8_t>>()));
            auto data =
                std::dynamic_pointer_cast<OwnedArray<uint8_t>>(block_attributes.back().data);
            attr_data.push_back(data);

            data->resize(block.size() * dtype_stride(type), 0);
            if (double_precision_attribs) {
                attrib_generators[i]->generate(
                    block.data(), block.size(), reinterpret_cast<double *>(data->begin()));
            } else {
                attrib_generators[i]->generate(
                    block.data(), block.size(), reinterpret_cast<float *>(data->begin()));
            }
        }
    }

    if (rank == 0) {
        std::cout << "Run with " << block_attributes.size() << " attributes per-particle\n";
    }

    BATParticleState state = bat_io_allocate();

    bat_io_set_local_bounds(state, &rank_bounds.lower.x);
    bat_io_set_bytes_per_subfile(state, max_bytes_per_file);

    if (best_split) {
        bat_io_set_find_best_axis(state, 1);
    }

    if (dump_raw) {
        bat_io_set_build_local_trees(state, 0);
    }

    glm::vec3 offset(0.f);
    for (uint32_t i = 0; i < num_writes; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);

        bat_io_set_positions(state, block.data(), block.size(), BTD_VEC3_FLOAT);
        for (const auto &attr : block_attributes) {
            bat_io_set_attribute(state,
                                 attr.desc.name.c_str(),
                                 attr.data->data(),
                                 attr.data->size(),
                                 BATDataType(attr.desc.data_type));
        }

        // Compute the # aggs based on the target file size and the global particle count
        if (fixed_aggregation) {
            // Compute avg. # of bytes per-rank and assign a fixed number of aggregators
            // to achieve the target file size
            const uint64_t local_particle_count = block.size();
            uint64_t global_particle_count = 0;
            MPI_Allreduce(&local_particle_count,
                          &global_particle_count,
                          1,
                          MPI_UNSIGNED_LONG_LONG,
                          MPI_SUM,
                          MPI_COMM_WORLD);

            uint64_t total_bytes = global_particle_count * sizeof(glm::vec3);
            for (const auto &attr : block_attributes) {
                total_bytes += global_particle_count * attr.desc.stride();
            }

            uint32_t num_aggregators = 0;
            if (max_bytes_per_file > 0) {
                num_aggregators = glm::clamp(uint32_t(total_bytes / max_bytes_per_file),
                                             uint32_t(1),
                                             uint32_t(world_size));
            } else {
                num_aggregators = world_size;
            }
            if (rank == 0) {
                std::cout << "Fixed aggregation with " << num_aggregators
                          << " target aggregators\n";
            }
            bat_io_set_fixed_aggregation(state, num_aggregators);
        }

        const std::string fname = file_prefix + "-t" + std::to_string(i);
        auto start = steady_clock::now();
        const uint64_t bytes_written = bat_io_write(state, fname.c_str());
        auto end = steady_clock::now();
        const char *perf_stats = bat_io_get_performance_statistics(state);
        if (rank == 0) {
            const size_t write_time = duration_cast<milliseconds>(end - start).count();
            const float bandwidth = (bytes_written * 1e-6f) / (write_time * 1e-3f);
            std::cout << "Total write time: " << write_time << "ms\n"
                      << "Total bytes written: " << bytes_written << "b\n"
                      << "Write bandwidth: " << bandwidth << "MB/s\n"
                      << "Perf Stats: " << perf_stats << "\n"
                      << "=======\n"
                      << std::flush;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // Now read the data back
        if (!dump_raw) {
            const std::string read_fname = fname + ".pbat";
            BATParticleFile file = bat_io_open(read_fname.c_str());
            bat_io_set_read_bounds(file, &rank_bounds.lower.x);
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
            bat_io_close(file);
        }

        // If we're advecting, clear the existing particles and re-create them with the
        // offset applied. Afterwards inject any new particles
        if (advect_particles) {
            block.clear();
            offset += advect;
            for (size_t j = 0; j < generators.size(); ++j) {
                const size_t generated = generators[j]->generate(
                    particles_generated_per_generator[j], offset, block);
                particles_generated_per_generator[j] = generated;
            }

            // Update attributes for the particles
            for (size_t j = 0; j < attrib_generators.size(); ++j) {
                auto &data = attr_data[j];
                data->resize(block.size() * block_attributes[j].desc.stride(), 0);
                if (double_precision_attribs) {
                    attrib_generators[j]->generate(
                        block.data(), block.size(), reinterpret_cast<double *>(data->begin()));
                } else {
                    attrib_generators[j]->generate(
                        block.data(), block.size(), reinterpret_cast<float *>(data->begin()));
                }
            }
        }

        if (inject > 0) {
            if (rank == 0) {
                std::cout << "Injecting " << inject << " particles\n";
            }
            const size_t prev_count = block.size();
            const size_t particles_per_generator =
                std::max(inject / generators.size(), size_t(1));
            for (size_t j = 0; j < generators.size(); ++j) {
                const size_t generated =
                    generators[j]->generate(particles_per_generator, glm::vec3(0.f), block);
                particles_generated_per_generator[j] += generated;
            }

            for (size_t j = 0; j < attrib_generators.size(); ++j) {
                auto &data = attr_data[j];
                data->resize(block.size() * block_attributes[j].desc.stride(), 0);
                if (double_precision_attribs) {
                    attrib_generators[j]->generate(
                        block.data() + prev_count,
                        block.size() - prev_count,
                        reinterpret_cast<double *>(data->begin()) + prev_count);
                } else {
                    attrib_generators[j]->generate(
                        block.data() + prev_count,
                        block.size() - prev_count,
                        reinterpret_cast<float *>(data->begin()) + prev_count);
                }
            }
        }

        // Check that some ranks still have particles to output, otherwise terminate early
        const uint32_t local_count = block.size();
        uint32_t global_max = 0;
        MPI_Allreduce(&local_count, &global_max, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
        if (global_max == 0) {
            if (rank == 0) {
                std::cout << "All particle sources have been advected outside the domain, "
                             "terminating early at timestep "
                          << i << "\n";
            }
            break;
        }
    }

    bat_io_free(state);

    MPI_Finalize();
    return 0;
}
