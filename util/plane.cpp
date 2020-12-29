#include "plane.h"

Plane::Plane(const glm::vec3 &origin, const glm::vec3 &half_vectors)
    : origin(origin), half_vectors(half_vectors)
{
}
