#include "binary_particle_file.h"
#include <fstream>
#include "data_type.h"
#include "file_mapping.h"
#include "json.hpp"
#include "owned_array.h"

using json = nlohmann::json;

size_t write_binary_particle_file(const std::string &fname,
                                  const BinaryParticleDump &particles)
{
    std::ofstream fout(fname.c_str(), std::ios::binary);

    json header;
    header["num_points"] = particles.points->size();

    size_t offset = sizeof(glm::vec3) * particles.points->size();
    for (size_t i = 0; i < particles.attribs.size(); ++i) {
        const auto &a = particles.attribs[i];
        header["attributes"][i]["name"] = a.desc.name;
        header["attributes"][i]["dtype"] = static_cast<int>(a.desc.data_type);
        header["attributes"][i]["offset"] = offset;
        header["attributes"][i]["size"] = a.data->size();

        offset += a.data->size();
    }

    const std::string json_header = header.dump();

    // We don't include the null terminator
    const uint64_t json_header_size = json_header.size();
    fout.write(reinterpret_cast<const char *>(&json_header_size), sizeof(uint64_t));
    fout.write(json_header.c_str(), json_header_size);

    fout.write(reinterpret_cast<const char *>(particles.points->data()),
               particles.points->size() * sizeof(glm::vec3));
    size_t bytes_written =
        json_header_size + sizeof(uint64_t) + particles.points->size() * sizeof(glm::vec3);
    for (const auto &a : particles.attribs) {
        fout.write(reinterpret_cast<const char *>(a.data->data()), a.data->size_bytes());
        bytes_written += a.data->size_bytes();
    }
    return bytes_written;
}

BinaryParticleDump read_binary_particle_file(const std::string &fname)
{
    std::ifstream fin(fname.c_str(), std::ios::binary);

    uint64_t json_header_size = 0;
    fin.read(reinterpret_cast<char *>(&json_header_size), sizeof(uint64_t));
    const uint64_t total_header_size = json_header_size + sizeof(uint64_t);

    std::vector<uint8_t> json_header(json_header_size, 0);
    fin.read(reinterpret_cast<char *>(json_header.data()), json_header.size());

    json header = json::parse(json_header);

    BinaryParticleDump particles;
    auto points = std::make_shared<OwnedArray<glm::vec3>>();
    points->resize(header["num_points"].get<uint64_t>());
    fin.read(reinterpret_cast<char *>(points->data()), points->size_bytes());
    particles.points = points;

    for (size_t i = 0; i < header["attributes"].size(); ++i) {
        Attribute a;
        a.desc.name = header["attributes"][i]["name"].get<std::string>();
        a.desc.data_type = static_cast<DTYPE>(header["attributes"][i]["dtype"].get<int>());

        const size_t offset = header["attributes"][i]["offset"].get<uint64_t>();
        const size_t size = header["attributes"][i]["size"].get<uint64_t>();
        a.data = std::make_shared<OwnedArray<uint8_t>>(size);
        fin.seekg(offset + total_header_size);
        fin.read(reinterpret_cast<char *>(a.data->data()), size);

        particles.attribs.push_back(a);
    }

    return particles;
}
