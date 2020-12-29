#pragma once

#include <glm/glm.hpp>

/* A plane is defined by its origin, and the half vectors
 * spanning the plane
 */
struct Plane {
    glm::vec3 origin;
    glm::vec3 half_vectors;

    Plane() = default;
    Plane(const glm::vec3 &origin, const glm::vec3 &half_vectors);
};
