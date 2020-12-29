#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <lasreader.hpp>
#include "util.h"

enum LIDAR_CLASSIFICATION {
    CREATED = 0,
    UNCLASSIFIED,
    GROUND,
    LOW_VEGETATION,
    MEDIUM_VEGETATION,
    HIGH_VEGETATION,
    BUILDING,
    NOISE,
    MODEL_KEY_POINT,
    WATER,
    OVERLAP_POINT,
    RESERVED
};
LIDAR_CLASSIFICATION classify_point(uint8_t class_attrib);

int main(int argc, char **argv)
{
    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " <input.las/laz>\n";
        return 0;
    }

    LASreadOpener read_opener;
    read_opener.set_file_name(argv[1]);
    LASreader *reader = read_opener.open();
    if (!reader) {
        std::cout << "Failed to open: " << argv[1] << "\n";
        return 1;
    }

    float scale_factor = 1.0f;
    if (argc == 4) {
        scale_factor = std::atof(argv[3]);
    }

    const bool has_color = reader->header.point_data_format == 2 ||
                           reader->header.point_data_format == 3 ||
                           reader->header.point_data_format == 5;

    std::cout << "LiDAR file '" << argv[1] << "' contains " << reader->npoints << " points "
              << (has_color ? "with" : "without") << " color attributes\n"
              << "min: ( " << reader->get_min_x() << ", " << reader->get_min_y() << ", "
              << reader->get_min_z() << " )\n"
              << "max: ( " << reader->get_max_x() << ", " << reader->get_max_y() << ", "
              << reader->get_max_z() << " )\n";

    const glm::vec3 min_pt(reader->get_min_x(), reader->get_min_y(), reader->get_min_z());
    const glm::vec3 max_pt(reader->get_max_x(), reader->get_max_y(), reader->get_max_z());
    const glm::vec3 diagonal = max_pt - min_pt;

    size_t n_points = 0;
    size_t n_noise = 0;
    const float inv_max_color = 1.0f / std::numeric_limits<uint16_t>::max();
    while (reader->read_point()) {
        // Points classified as low point are noise and should be discarded
        if (classify_point(reader->point.get_classification()) == NOISE) {
            ++n_noise;
            continue;
        }
        ++n_points;
        reader->point.compute_coordinates();
        // Re-scale points to a better precision range for floats
        const glm::vec3 p = glm::vec3(reader->point.coordinates[0],
                                      reader->point.coordinates[1],
                                      reader->point.coordinates[2]) -
                            min_pt - diagonal * 0.5f;

        const uint16_t *rgba = reader->point.get_rgb();
        glm::vec3 c;
        if (has_color) {
            c = glm::vec3(
                rgba[0] * inv_max_color, rgba[1] * inv_max_color, rgba[2] * inv_max_color);
            c.x = srgb_to_linear(c.x);
            c.y = srgb_to_linear(c.y);
            c.z = srgb_to_linear(c.z);
        } else {
            c = glm::vec3(1.0);
        }
    }
    reader->close();
    delete reader;

    std::cout << "Read " << n_points << " points from " << argv[1] << "\n"
              << "Discarded " << n_noise << " noise classified points\n"
              << "Translated bounds to " << glm::to_string(diagonal * -0.5f) << ", "
              << glm::to_string(max_pt - min_pt - diagonal * 0.5f) << "\n";

    return 0;
}

LIDAR_CLASSIFICATION classify_point(uint8_t class_attrib)
{
    switch (class_attrib) {
    case 0:
        return CREATED;
    case 1:
        return UNCLASSIFIED;
    case 2:
        return GROUND;
    case 3:
        return LOW_VEGETATION;
    case 4:
        return MEDIUM_VEGETATION;
    case 5:
        return HIGH_VEGETATION;
    case 6:
        return BUILDING;
    case 7:
        return NOISE;
    case 8:
        return MODEL_KEY_POINT;
    case 9:
        return WATER;
    case 10:
        return RESERVED;
    case 11:
        return RESERVED;
    case 12:
        return OVERLAP_POINT;
    default:
        return RESERVED;
    }
}
