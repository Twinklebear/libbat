#include "box.h"
#include <cmath>
#include <numeric>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

Box::Box()
    : lower(std::numeric_limits<float>::infinity()),
      upper(-std::numeric_limits<float>::infinity())
{
}

Box::Box(const glm::vec3 &lower, const glm::vec3 &upper) : lower(lower), upper(upper) {}

Box::Box(const float *vals) : lower(glm::make_vec3(vals)), upper(glm::make_vec3(vals + 3)) {}

void Box::extend(const glm::vec3 &p)
{
    lower = glm::min(lower, p);
    upper = glm::max(upper, p);
}

void Box::box_union(const Box &b)
{
    extend(b.lower);
    extend(b.upper);
}

bool Box::overlaps(const Box &b) const
{
    if (lower.x > b.upper.x || b.lower.x > upper.x) {
        return false;
    }
    if (lower.y > b.upper.y || b.lower.y > upper.y) {
        return false;
    }
    if (lower.z > b.upper.z || b.lower.z > upper.z) {
        return false;
    }
    return true;
}

bool Box::contains_point(const glm::vec3 &p) const
{
    if (p.x < lower.x || p.x > upper.x) {
        return false;
    }
    if (p.y < lower.y || p.y > upper.y) {
        return false;
    }
    if (p.z < lower.z || p.z > upper.z) {
        return false;
    }
    return true;
}

bool Box::contains_box(const Box &b) const
{
    return glm::all(glm::lessThanEqual(lower, b.lower)) &&
           glm::all(glm::greaterThanEqual(upper, b.upper));
}

AXIS Box::longest_axis() const
{
    const glm::vec3 diag = diagonal();
    if (diag.x >= diag.y && diag.x >= diag.z) {
        return X;
    }
    if (diag.y >= diag.z) {
        return Y;
    }
    return Z;
}

glm::uvec3 Box::axis_ordering() const
{
    const glm::vec3 diag = diagonal();
    glm::uvec3 order(0);
    if (diag.x >= diag.y && diag.x >= diag.z) {
        order.x = 0;
    } else if (diag.y >= diag.z) {
        order.x = 1;
    } else {
        order.x = 2;
    }

    order.y = (order.x + 1) % 3;
    order.z = (order.x + 2) % 3;
    if (diag[(order.x + 1) % 3] < diag[(order.x + 2) % 3]) {
        std::swap(order.y, order.z);
    }
    return order;
}

glm::vec3 Box::center() const
{
    return lower + diagonal() * 0.5f;
}

glm::vec3 Box::diagonal() const
{
    return upper - lower;
}

std::ostream &operator<<(std::ostream &os, const Box &b)
{
    os << "Box [" << glm::to_string(b.lower) << ", " << glm::to_string(b.upper) << "]";
    return os;
}

float distance_to_box(const Box &b, const glm::vec3 &p)
{
    const glm::vec3 v(glm::max(b.lower.x - p.x, glm::max(0.f, p.x - b.upper.x)),
                      glm::max(b.lower.y - p.y, glm::max(0.f, p.y - b.upper.y)),
                      glm::max(b.lower.z - p.z, glm::max(0.f, p.z - b.upper.z)));
    return glm::length(v);
}
