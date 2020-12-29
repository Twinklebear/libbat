#pragma once

#include <array>
#include <glm/glm.hpp>
#include "box.h"

struct Frustum {
    /* Planes are the frustum planes stored in order:
     * -x, +x, -y, +y, -z, +z
     */
    std::array<glm::vec4, 6> planes;

    /* Points are the frustum's corner points, in order:
     * (x_l, y_l, z_l), (x_h, y_l, z_l), (x_l, y_h, z_l),
     * (x_h, y_h, z_l), (x_l, y_l, z_h), (x_h, y_l, z_h),
     * (x_l, y_h, z_h), (x_h, y_h, z_h)
     */
    std::array<glm::vec4, 8> points;

    /* Construct the frustum by decomposing the
     * passed proj_view matrix
     */
    Frustum(const glm::mat4 &proj_view);
    /* Check if the box is contained inside the frustum
     * (i.e., partially or fully contained)
     * This is done using Inigo Quilez's approach to help with large
     * bounds: https://www.iquilezles.org/www/articles/frustumcorrect/frustumcorrect.htm
     */
    bool contains(const Box &box) const;
};
