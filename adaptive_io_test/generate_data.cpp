#include "generate_data.h"
#include <mpi.h>

static int override_global_seed = -1;

void set_global_seed(int seed)
{
    override_global_seed = seed;
}

int get_global_seed()
{
    if (override_global_seed >= 0) {
        return override_global_seed;
    }
    // We need a consistent seed for the rng so each rank places the sphere centers at the same
    // location. Rank 0 generates the seed and bcasts it out
    int rank = 0;
    int seed = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::random_device rd;
        seed = rd();
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return seed;
}

Generator::Generator(const Box &world_bounds, const Box &rank_bounds)
    : world_bounds(world_bounds), rank_bounds(rank_bounds)
{
    rng.seed(get_global_seed());
}

UniformGenerator::UniformGenerator(const Box &world_bounds, const Box &rank_bounds)
    : Generator(world_bounds, rank_bounds)
{
}

size_t UniformGenerator::generate(const size_t n_particles,
                                  const glm::vec3 &,
                                  std::vector<glm::vec3> &out)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> x_distrib(rank_bounds.lower.x, rank_bounds.upper.x);
    std::uniform_real_distribution<float> y_distrib(rank_bounds.lower.y, rank_bounds.upper.y);
    std::uniform_real_distribution<float> z_distrib(rank_bounds.lower.z, rank_bounds.upper.z);

    for (size_t i = 0; i < n_particles; ++i) {
        const glm::vec3 p = glm::vec3(x_distrib(rng), y_distrib(rng), z_distrib(rng));
        out.emplace_back(p);
    }
    return n_particles;
}

LinearGradientGenerator::LinearGradientGenerator(const Box &world_bounds,
                                                 const Box &rank_bounds)
    : Generator(world_bounds, rank_bounds)
{
}

size_t LinearGradientGenerator::generate(const size_t n_particles,
                                         const glm::vec3 &offset,
                                         std::vector<glm::vec3> &out)
{
    if (!world_bounds.contains_point(offset)) {
        return 0;
    }

    // Kind of a hacky thing to make some linear gradient density distribution
    const size_t n_steps = 100;
    float sum = 0.f;
    for (size_t i = 0; i < n_steps; ++i) {
        sum += static_cast<float>(i + 1) / n_steps;
    }
    const float base_particle_count = n_particles / sum;

    std::random_device rd;
    std::uniform_real_distribution<float> y_distrib(world_bounds.lower.y,
                                                    world_bounds.upper.y);
    std::uniform_real_distribution<float> z_distrib(world_bounds.lower.z,
                                                    world_bounds.upper.z);

    size_t generated = 0;
    const float x_step = (world_bounds.upper.x - world_bounds.lower.x) / n_steps;
    for (size_t i = 0; i < n_steps; ++i) {
        const size_t gradient_step_particles =
            base_particle_count * static_cast<float>(i + 1) / n_steps;
        generated += gradient_step_particles;

        std::uniform_real_distribution<float> x_distrib(x_step * i, x_step * (i + 1));
        if (rank_bounds.lower.x > x_distrib.max() || rank_bounds.upper.x < x_distrib.min()) {
            continue;
        }

        for (size_t j = 0; j < gradient_step_particles; ++j) {
            const glm::vec3 p =
                glm::vec3(x_distrib(rng), y_distrib(rng), z_distrib(rng)) + offset;
            if (rank_bounds.contains_point(p)) {
                out.push_back(p);
            }
        }
    }
    return generated;
}

ExponentialGradientGenerator::ExponentialGradientGenerator(const Box &world_bounds,
                                                           const Box &rank_bounds)
    : Generator(world_bounds, rank_bounds)
{
}

size_t ExponentialGradientGenerator::generate(const size_t n_particles,
                                              const glm::vec3 &offset,
                                              std::vector<glm::vec3> &out)
{
    if (!world_bounds.contains_point(offset)) {
        return 0;
    }
    std::random_device rd;
    std::exponential_distribution<float> x_distrib;
    std::uniform_real_distribution<float> y_distrib(world_bounds.lower.y,
                                                    world_bounds.upper.y);
    std::uniform_real_distribution<float> z_distrib(world_bounds.lower.z,
                                                    world_bounds.upper.z);
    for (size_t i = 0; i < n_particles; ++i) {
        // The exponential distribution will be nearly 0 at 5 anyway so just clamp
        // between that range
        float x = glm::clamp(x_distrib(rng), 0.f, 5.f);
        x = (world_bounds.upper.x - world_bounds.lower.x) * x / 5.f + world_bounds.lower.x;

        const glm::vec3 p = glm::vec3(x, y_distrib(rng), z_distrib(rng)) + offset;
        if (rank_bounds.contains_point(p)) {
            out.push_back(p);
        }
    }
    return n_particles;
}

SphereGenerator::Source::Source(const glm::vec3 &c, float r) : center(c), radius(r) {}

SphereGenerator::SphereGenerator(const Box &world_bounds,
                                 const Box &rank_bounds,
                                 const size_t num_spheres)
    : Generator(world_bounds, rank_bounds)
{
    std::uniform_real_distribution<float> x_distrib(world_bounds.lower.x,
                                                    world_bounds.upper.x);
    std::uniform_real_distribution<float> y_distrib(world_bounds.lower.y,
                                                    world_bounds.upper.y);
    std::uniform_real_distribution<float> z_distrib(world_bounds.lower.z,
                                                    world_bounds.upper.z);
    const float world_diag = glm::length(world_bounds.diagonal());
    std::uniform_real_distribution<float> r_distrib(0.05f * world_diag, 0.6f * world_diag);
    for (size_t i = 0; i < num_spheres; ++i) {
        sources.emplace_back(glm::vec3(x_distrib(rng), y_distrib(rng), z_distrib(rng)),
                             r_distrib(rng));
    }
}

size_t SphereGenerator::generate(const size_t n_particles,
                                 const glm::vec3 &offset,
                                 std::vector<glm::vec3> &out)
{
    std::normal_distribution<float> x_distrib;
    std::normal_distribution<float> y_distrib;
    std::normal_distribution<float> z_distrib;
    std::uniform_real_distribution<float> r_distrib(0.f, 1.f);
    std::uniform_int_distribution<uint32_t> source_selector(0, sources.size() - 1);
    size_t generated = 0;
    for (size_t i = 0; i < n_particles; ++i) {
        // Repeat until we get a point inside the world bounds so we reach the desired # of
        // points
        glm::vec3 p;
        do {
            const Source &src = sources[sources.size() > 1 ? source_selector(rng) : 0];
            // Check if it's possible for this source to still generate points in the domain
            if (distance_to_box(world_bounds, src.center + offset) >= src.radius) {
                break;
            }
            const float r = r_distrib(rng) * src.radius;
            p = src.center +
                r * glm::normalize(glm::vec3(x_distrib(rng), y_distrib(rng), z_distrib(rng))) +
                offset;
            if (world_bounds.contains_point(p)) {
                ++generated;
            }
        } while (!world_bounds.contains_point(p));

        if (rank_bounds.contains_point(p)) {
            out.push_back(p);
        }
    }
    return generated;
}

CutBoxGenerator::CutBoxGenerator(const Box &world_bounds,
                                 const Box &rank_bounds,
                                 const glm::vec3 &box_size)
    : Generator(world_bounds, rank_bounds), cut_box(world_bounds)
{
    cut_box.upper = cut_box.lower + box_size * world_bounds.diagonal();
}

size_t CutBoxGenerator::generate(const size_t n_particles,
                                 const glm::vec3 &offset,
                                 std::vector<glm::vec3> &out)
{
    if (!world_bounds.contains_point(offset)) {
        return 0;
    }

    if (cut_box.contains_box(rank_bounds)) {
        return n_particles;
    }

    std::uniform_real_distribution<float> x_distrib(world_bounds.lower.x,
                                                    world_bounds.upper.x);
    std::uniform_real_distribution<float> y_distrib(world_bounds.lower.y,
                                                    world_bounds.upper.y);
    std::uniform_real_distribution<float> z_distrib(world_bounds.lower.z,
                                                    world_bounds.upper.z);
    for (size_t i = 0; i < n_particles; ++i) {
        glm::vec3 p;
        do {
            p = glm::vec3(x_distrib(rng), y_distrib(rng), z_distrib(rng)) + offset;
        } while (cut_box.contains_point(p));

        if (rank_bounds.contains_point(p)) {
            out.push_back(p);
        }
    }
    return n_particles;
}

AttributeGenerator::AttributeGenerator()
{
    rng.seed(get_global_seed());
    std::uniform_real_distribution<float> attr_min(-10000.f, 10000.f);
    std::uniform_real_distribution<float> attr_width(0.5f, 5000.f);
    attr_range.x = attr_min(rng);
    attr_range.y = attr_range.x + attr_width(rng);
}

NonSpatialAttribute::NonSpatialAttribute(const std::string &type) : type(type)
{
    if (type != "uniform" && type != "normal") {
        throw std::runtime_error("Invalid non-spatial attribute type " + type);
    }
}

void NonSpatialAttribute::generate(const glm::vec3 *, const size_t n_particles, float *out)
{
    if (type == "uniform") {
        std::uniform_real_distribution<float> attr_distrib(attr_range.x, attr_range.y);
        for (size_t i = 0; i < n_particles; ++i) {
            out[i] = attr_distrib(rng);
        }
    } else {
        std::normal_distribution<float> attr_distrib;
        for (size_t i = 0; i < n_particles; ++i) {
            // Clamp the distribution between [-4.f, 4.f]
            float x = glm::clamp(attr_distrib(rng), -4.f, 4.f);
            out[i] = (attr_range.y - attr_range.x) * (x + 4.f) / 8.f + attr_range.x;
        }
    }
}

void NonSpatialAttribute::generate(const glm::vec3 *, const size_t n_particles, double *out)
{
    if (type == "uniform") {
        std::uniform_real_distribution<float> attr_distrib(attr_range.x, attr_range.y);
        for (size_t i = 0; i < n_particles; ++i) {
            out[i] = attr_distrib(rng);
        }
    } else {
        std::normal_distribution<float> attr_distrib;
        for (size_t i = 0; i < n_particles; ++i) {
            // Clamp the distribution between [-4.f, 4.f]
            float x = glm::clamp(attr_distrib(rng), -4.f, 4.f);
            out[i] = (attr_range.y - attr_range.x) * (x + 4.f) / 8.f + attr_range.x;
        }
    }
}

const std::string NonSpatialAttribute::name() const
{
    return "non-spatial-" + type;
}

SpatialAttribute::SpatialAttribute(const std::string &type, const Box &world_bounds)
    : type(type), world_bounds(world_bounds)
{
    if (type == "gradient") {
        std::uniform_int_distribution<uint32_t> axis_distrib(0, 2);
        gradient_axis = axis_distrib(rng);
    } else if (type == "sphere") {
        std::uniform_real_distribution<float> x_distrib(world_bounds.lower.x,
                                                        world_bounds.upper.x);
        std::uniform_real_distribution<float> y_distrib(world_bounds.lower.y,
                                                        world_bounds.upper.y);
        std::uniform_real_distribution<float> z_distrib(world_bounds.lower.z,
                                                        world_bounds.upper.z);
        std::uniform_real_distribution<float> r_distrib(0.5f,
                                                        glm::length(world_bounds.diagonal()));
        sphere_center.x = x_distrib(rng);
        sphere_center.y = y_distrib(rng);
        sphere_center.z = z_distrib(rng);
        sphere_radius = r_distrib(rng);
    } else {
        throw std::runtime_error("Invalid type for spatial attribute " + type);
    }
}

void SpatialAttribute::generate(const glm::vec3 *particles,
                                const size_t n_particles,
                                float *out)
{
    const glm::vec3 diag = world_bounds.diagonal();
    if (type == "gradient") {
        for (size_t i = 0; i < n_particles; ++i) {
            float v = (particles[i][gradient_axis] - world_bounds.lower[gradient_axis]) /
                      diag[gradient_axis];
            v = glm::clamp(v, 0.f, 1.f);
            out[i] = (attr_range.y - attr_range.x) * v + attr_range.x;
        }
    } else {
        for (size_t i = 0; i < n_particles; ++i) {
            float v =
                1.f - glm::clamp(
                          glm::length(particles[i] - sphere_center) / sphere_radius, 0.f, 1.f);
            out[i] = (attr_range.y - attr_range.x) * v + attr_range.x;
        }
    }
}

void SpatialAttribute::generate(const glm::vec3 *particles,
                                const size_t n_particles,
                                double *out)
{
    const glm::vec3 diag = world_bounds.diagonal();
    if (type == "gradient") {
        for (size_t i = 0; i < n_particles; ++i) {
            float v = (particles[i][gradient_axis] - world_bounds.lower[gradient_axis]) /
                      diag[gradient_axis];
            v = glm::clamp(v, 0.f, 1.f);
            out[i] = (attr_range.y - attr_range.x) * v + attr_range.x;
        }
    } else {
        for (size_t i = 0; i < n_particles; ++i) {
            float v =
                1.f - glm::clamp(
                          glm::length(particles[i] - sphere_center) / sphere_radius, 0.f, 1.f);
            out[i] = (attr_range.y - attr_range.x) * v + attr_range.x;
        }
    }
}

const std::string SpatialAttribute::name() const
{
    return "spatial-" + type;
}
