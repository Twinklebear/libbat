#pragma once

#include <memory>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "abstract_array.h"
#include "attribute.h"

struct BinaryParticleDump {
    std::shared_ptr<AbstractArray<glm::vec3>> points = nullptr;
    std::vector<Attribute> attribs;
};

size_t write_binary_particle_file(const std::string &fname,
                                  const BinaryParticleDump &particles);

BinaryParticleDump read_binary_particle_file(const std::string &fname);
