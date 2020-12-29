#include "frustum.h"
#include <glm/gtc/matrix_access.hpp>

Frustum::Frustum(const glm::mat4 &proj_view)
{
    const std::array<glm::vec4, 4> rows = {glm::row(proj_view, 0),
                                           glm::row(proj_view, 1),
                                           glm::row(proj_view, 2),
                                           glm::row(proj_view, 3)};

    // -x plane
    planes[0] = rows[3] + rows[0];
    // +x plane
    planes[1] = rows[3] - rows[0];
    // -y plane
    planes[2] = rows[3] + rows[1];
    // +y plane
    planes[3] = rows[3] - rows[1];
    // -z plane
    planes[4] = rows[3] + rows[2];
    // +z plane
    planes[5] = rows[3] - rows[2];

    // Normalize the planes
    for (auto &p : planes) {
        const float s = 1.f / glm::length(glm::vec3(p));
        p.x *= s;
        p.y *= s;
        p.z *= s;
        p.w *= s;
    }

    // Compute the frustum points
    const auto inv_proj_view = glm::inverse(proj_view);
    // x_l, y_l, z_l
    points[0] = glm::vec4(-1, -1, -1, 1);
    // x_h, y_l, z_l
    points[1] = glm::vec4(1, -1, -1, 1);
    // x_l, y_h, z_l
    points[2] = glm::vec4(-1, 1, -1, 1);
    // x_h, y_h, z_l
    points[3] = glm::vec4(1, 1, -1, 1);
    // x_l, y_l, z_h
    points[4] = glm::vec4(-1, -1, 1, 1);
    // x_h, y_l, z_h
    points[5] = glm::vec4(1, -1, 1, 1);
    // x_l, y_h, z_h
    points[6] = glm::vec4(-1, 1, 1, 1);
    // x_h, y_h, z_h
    points[7] = glm::vec4(1, 1, 1, 1);

    for (auto &p : points) {
        p = inv_proj_view * p;
        p.x /= p.w;
        p.y /= p.w;
        p.z /= p.w;
        p.w = 1.f;
    }
}
bool Frustum::contains(const Box &box) const
{
    // Test the box against each plane
    int out = 0;
    for (const auto &p : planes) {
        out = 0;
        // x_l, y_l, z_l
        glm::vec4 v = glm::vec4(box.lower, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_h, y_l, z_l
        v = glm::vec4(box.upper.x, box.lower.y, box.lower.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_l, y_h, z_l
        v = glm::vec4(box.lower.x, box.upper.y, box.lower.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_h, y_h, z_l
        v = glm::vec4(box.upper.x, box.upper.y, box.lower.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_l, y_l, z_h
        v = glm::vec4(box.lower.x, box.lower.y, box.upper.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_h, y_l, z_h
        v = glm::vec4(box.upper.x, box.lower.y, box.upper.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_l, y_h, z_h
        v = glm::vec4(box.lower.x, box.upper.y, box.upper.z, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        // x_h, y_h, z_h
        v = glm::vec4(box.upper, 1.f);
        out += glm::dot(p, v) < 0.f ? 1 : 0;

        if (out == 8) {
            return false;
        }
    }

    // Test the frustum against the box
    out = 0;
    for (const auto &p : points) {
        out += p[0] > box.upper.x ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }

    out = 0;
    for (const auto &p : points) {
        out += p[0] < box.lower.x ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }

    out = 0;
    for (const auto &p : points) {
        out += p[1] > box.upper.y ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }

    out = 0;
    for (const auto &p : points) {
        out += p[1] < box.lower.y ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }

    out = 0;
    for (const auto &p : points) {
        out += p[2] > box.upper.z ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }

    out = 0;
    for (const auto &p : points) {
        out += p[2] < box.lower.z ? 1 : 0;
    }
    if (out == 8) {
        return false;
    }
    return true;
}
