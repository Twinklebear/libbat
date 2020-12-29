#pragma once

#include <random>
#include <string>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "util.h"

// Allow setting the global seed so we can run consistent
// benchmarks on the same generated data sets
void set_global_seed(int seed);

struct Generator {
    Box world_bounds;
    Box rank_bounds;
    std::mt19937 rng;

    Generator(const Box &world_bounds, const Box &rank_bounds);

    Generator() = default;

    virtual ~Generator() {}

    virtual size_t generate(const size_t n_particles,
                            const glm::vec3 &offset,
                            std::vector<glm::vec3> &out) = 0;
};

struct UniformGenerator : Generator {
    UniformGenerator(const Box &world_bounds, const Box &rank_bounds);

    size_t generate(const size_t n_particles,
                    const glm::vec3 &offset,
                    std::vector<glm::vec3> &out) override;
};

struct LinearGradientGenerator : Generator {
    LinearGradientGenerator(const Box &world_bounds, const Box &rank_bounds);

    size_t generate(const size_t n_particles,
                    const glm::vec3 &offset,
                    std::vector<glm::vec3> &out) override;
};

struct ExponentialGradientGenerator : Generator {
    ExponentialGradientGenerator(const Box &world_bounds, const Box &rank_bounds);

    size_t generate(const size_t n_particles,
                    const glm::vec3 &offset,
                    std::vector<glm::vec3> &out) override;
};

struct SphereGenerator : Generator {
    struct Source {
        glm::vec3 center;
        float radius;

        Source(const glm::vec3 &c, float r);

        Source() = default;
    };

    std::vector<Source> sources;

    SphereGenerator(const Box &world_bounds, const Box &rank_bounds, const size_t num_spheres);

    size_t generate(const size_t n_particles,
                    const glm::vec3 &offset,
                    std::vector<glm::vec3> &out) override;
};

struct CutBoxGenerator : Generator {
    Box cut_box;

    CutBoxGenerator(const Box &world_bounds,
                    const Box &rank_bounds,
                    const glm::vec3 &box_size);

    size_t generate(const size_t n_particles,
                    const glm::vec3 &offset,
                    std::vector<glm::vec3> &out) override;
};

struct AttributeGenerator {
    std::mt19937 rng;
    glm::vec2 attr_range;

    AttributeGenerator();

    virtual ~AttributeGenerator() {}

    virtual void generate(const glm::vec3 *particles,
                          const size_t n_particles,
                          float *out) = 0;

    virtual void generate(const glm::vec3 *particles,
                          const size_t n_particles,
                          double *out) = 0;

    virtual const std::string name() const = 0;
};

struct NonSpatialAttribute : AttributeGenerator {
    std::string type;

    NonSpatialAttribute(const std::string &type);

    void generate(const glm::vec3 *particles, const size_t n_particles, float *out) override;

    void generate(const glm::vec3 *particles, const size_t n_particles, double *out) override;

    const std::string name() const override;
};

struct SpatialAttribute : AttributeGenerator {
    std::string type;
    Box world_bounds;

    uint32_t gradient_axis;

    glm::vec3 sphere_center;
    float sphere_radius;

    SpatialAttribute(const std::string &type, const Box &world_bounds);

    void generate(const glm::vec3 *particles, const size_t n_particles, float *out) override;

    void generate(const glm::vec3 *particles, const size_t n_particles, double *out) override;

    const std::string name() const override;
};
