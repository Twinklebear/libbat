#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "file_mapping.h"
#include "lba_tree_builder.h"
#include "tinyxml2.h"
#include "util.h"

using namespace tinyxml2;

const std::string USAGE = "Usage: ./uintah_converter <uda.xml> <out.bat>";

struct UintahPatch {
    glm::vec3 lower;
};

struct ParticleModel {
    std::vector<glm::vec3> positions;

    std::vector<Attribute> attributes;

    size_t attrib_index(const std::string &name)
    {
        auto fnd = std::find_if(attributes.begin(), attributes.end(), [&](const Attribute &a) {
            return a.desc.name == name;
        });
        if (fnd == attributes.end()) {
            return std::numeric_limits<size_t>::max();
        }
        return std::distance(attributes.begin(), fnd);
    }
};

bool uintah_is_big_endian = false;

template <typename T>
T ntoh(const T &t)
{
    T ret = T{};
    const uint8_t *in = reinterpret_cast<const uint8_t *>(&t);
    uint8_t *out = reinterpret_cast<uint8_t *>(&ret);
    for (size_t i = 0; i < sizeof(T); ++i) {
        out[i] = in[sizeof(T) - 1 - i];
    }
    return ret;
}

void read_particle_positions(const std::string &file_name,
                             std::vector<glm::vec3> &positions,
                             const size_t start,
                             const size_t end);

template <typename In, typename Out = In>
void read_particle_attribute(const std::string &file_name,
                             std::shared_ptr<OwnedArray<uint8_t>> &attrib,
                             const size_t start,
                             const size_t end);

bool read_uintah_particle_variable(const std::string &base_path,
                                   XMLElement *elem,
                                   ParticleModel &model);

bool read_uintah_datafile(const std::string &file_name,
                          tinyxml2::XMLDocument &doc,
                          ParticleModel &model);

bool read_uintah_timestep_meta(XMLNode *node);

bool read_uintah_patches(XMLNode *node, std::vector<UintahPatch> &patches);

bool read_uintah_timestep_data(const std::string &base_path,
                               XMLNode *node,
                               ParticleModel &model);

bool read_uintah_timestep(const std::string &file_name,
                          XMLElement *node,
                          ParticleModel &model);

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cout << USAGE << "\n";
        return 1;
    }

    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h") {
            std::cout << USAGE << "\n";
            return 0;
        }
    }

    const std::string uda_file = canonicalize_path(args[1]);
    std::cout << "Importing Uintah data from " << uda_file << "\n";

    ParticleModel model;
    tinyxml2::XMLDocument doc;
    XMLError err = doc.LoadFile(uda_file.c_str());
    if (err != XML_SUCCESS) {
        std::cout << "Error loading Uintah data file '" << uda_file
                  << "': " << tinyxml_error_string(err) << "\n";
        return 1;
    }

    if (doc.FirstChildElement("Uintah_timestep")) {
        read_uintah_timestep(uda_file, doc.FirstChildElement("Uintah_timestep"), model);
    } else if (doc.FirstChildElement("Uintah_Output")) {
        read_uintah_datafile(uda_file, doc, model);
    } else {
        std::cout << "Unrecognized UDA XML file!\n";
        return 1;
    }

    std::cout << "Read " << model.positions.size() << " particles\n"
              << "# attributes: " << model.attributes.size() << "\n";
    for (const auto &a : model.attributes) {
        std::cout << "attrib: " << a.desc.name << "\n";
    }

    BATree tree =
        LBATreeBuilder(std::move(model.positions), std::move(model.attributes)).compact();
    write_ba_tree(args[2], tree);

    return 0;
}

void read_particle_positions(const std::string &file_name,
                             std::vector<glm::vec3> &positions,
                             const size_t start,
                             const size_t end)
{
    FileMapping map(file_name);
    const glm::dvec3 *map_start = reinterpret_cast<const glm::dvec3 *>(map.data() + start);
    const glm::dvec3 *map_end = reinterpret_cast<const glm::dvec3 *>(map.data() + end);
    std::transform(map_start, map_end, std::back_inserter(positions), [](const glm::dvec3 &p) {
        if (uintah_is_big_endian) {
            return glm::vec3(ntoh(p.x), ntoh(p.y), ntoh(p.z));
        }
        return glm::vec3(p);
    });
}

template <typename In, typename Out>
void read_particle_attribute(const std::string &file_name,
                             std::shared_ptr<OwnedArray<uint8_t>> &attrib,
                             const size_t start,
                             const size_t end)
{
    FileMapping map(file_name);
    const In *map_start = reinterpret_cast<const In *>(map.data() + start);
    const In *map_end = reinterpret_cast<const In *>(map.data() + end);
    for (const In *it = map_start; it != map_end; ++it) {
        Out o;
        if (uintah_is_big_endian) {
            o = static_cast<Out>(ntoh(*it));
        } else {
            o = static_cast<Out>(*it);
        }
        uint8_t *x = reinterpret_cast<uint8_t *>(&o);
        for (size_t i = 0; i < sizeof(Out); ++i) {
            attrib->push_back(x[i]);
        }
    };
}

bool read_uintah_particle_variable(const std::string &base_path,
                                   XMLElement *elem,
                                   ParticleModel &model)
{
    std::string type;
    {
        const char *type_attrib = elem->Attribute("type");
        if (!type_attrib) {
            std::cout << "Variable missing type attribute\n";
            return false;
        }
        type = std::string(type_attrib);
    }
    std::string variable;
    std::string file_name;
    size_t index = std::numeric_limits<size_t>::max();
    size_t start = std::numeric_limits<size_t>::max();
    size_t end = std::numeric_limits<size_t>::max();
    size_t patch = std::numeric_limits<size_t>::max();
    size_t num_particles = 0;
    for (XMLNode *c = elem->FirstChild(); c; c = c->NextSibling()) {
        XMLElement *e = c->ToElement();
        if (!e) {
            std::cout << "Failed to parse Uintah variable element\n";
            return false;
        }
        std::string name = e->Value();
        const char *text = e->GetText();
        if (!text) {
            std::cout << "Invalid variable '" << name << "', missing value\n";
            return false;
        }
        if (name == "variable") {
            variable = text;
        } else if (name == "filename") {
            file_name = text;
        } else if (name == "index") {
            try {
                index = std::strtoul(text, NULL, 10);
            } catch (const std::range_error &r) {
                std::cout << "Invalid index value specified\n";
                return false;
            }
        } else if (name == "start") {
            try {
                start = std::strtoul(text, NULL, 10);
            } catch (const std::range_error &r) {
                std::cout << "Invalid start value specified\n";
                return false;
            }
        } else if (name == "end") {
            try {
                end = std::strtoul(text, NULL, 10);
            } catch (const std::range_error &r) {
                std::cout << "Invalid end value specified\n";
                return false;
            }
        } else if (name == "patch") {
            try {
                patch = std::strtoul(text, NULL, 10);
            } catch (const std::range_error &r) {
                std::cout << "Invalid patch value specified\n";
                return false;
            }
        } else if (name == "numParticles") {
            try {
                num_particles = std::strtoul(text, NULL, 10);
            } catch (const std::range_error &r) {
                std::cout << "Invalid numParticles value specified\n";
                return false;
            }
        }
    }
    if (num_particles > 0) {
        file_name = base_path + "/" + file_name;

        if (variable == "p.x") {
            read_particle_positions(file_name, model.positions, start, end);
        } else if (starts_with(type, "ParticleVariable")) {
            if (type != "ParticleVariable<double>" && type != "ParticleVariable<float>" &&
                type != "ParticleVariable<long64>") {
                std::cout << "Skipping variable " << variable << ": Unsupported type " << type
                          << "\n";
            } else {
                size_t attrib_index = model.attrib_index(variable);
                if (attrib_index == std::numeric_limits<size_t>::max()) {
                    attrib_index = model.attributes.size();
                    model.attributes.emplace_back(AttributeDescription(variable, UNKNOWN),
                                                  std::make_shared<OwnedArray<uint8_t>>());
                }
                Attribute &attrib = model.attributes[attrib_index];
                auto arr = std::dynamic_pointer_cast<OwnedArray<uint8_t>>(attrib.data);

                if (type == "ParticleVariable<double>") {
                    attrib.desc.data_type = FLOAT_64;
                    read_particle_attribute<double>(file_name, arr, start, end);
                } else if (type == "ParticleVariable<float>") {
                    attrib.desc.data_type = FLOAT_32;
                    read_particle_attribute<float>(file_name, arr, start, end);
                } else if (type == "ParticleVariable<long64>") {
                    attrib.desc.data_type = INT_64;
                    read_particle_attribute<int64_t>(file_name, arr, start, end);
                }
            }
        }
    }
    return true;
}

bool read_uintah_datafile(const std::string &file_name,
                          tinyxml2::XMLDocument &doc,
                          ParticleModel &model)
{
    const std::string base_path = get_file_basepath(file_name);
    XMLElement *node = doc.FirstChildElement("Uintah_Output");
    const static std::string VAR_TYPE = "ParticleVariable";
    for (XMLNode *c = node->FirstChild(); c; c = c->NextSibling()) {
        if (std::string(c->Value()) != "Variable") {
            std::cout << "Invalid XML node encountered, expected <Variable...>\n";
            return false;
        }
        XMLElement *e = c->ToElement();
        if (!e || !e->Attribute("type")) {
            std::cout << "Invalid variable element found\n";
            return false;
        }
        std::string var_type = e->Attribute("type");
        if (var_type.substr(0, VAR_TYPE.size()) == VAR_TYPE) {
            if (!read_uintah_particle_variable(base_path, e, model)) {
                return false;
            }
        }
    }
    return true;
}

bool read_uintah_timestep_meta(XMLNode *node)
{
    for (XMLNode *c = node->FirstChild(); c; c = c->NextSibling()) {
        XMLElement *e = c->ToElement();
        if (!e) {
            std::cout << "Error parsing Uintah timestep meta\n";
            return false;
        }
        if (std::string(e->Value()) == "endianness") {
            if (std::string(e->GetText()) == "big_endian") {
                std::cout << "Uintah parser switching to big endian\n";
                uintah_is_big_endian = true;
            }
        }
    }
    return true;
}

bool read_uintah_patches(XMLNode *node, std::vector<UintahPatch> &patches)
{
    // TODO: How to deal with multiple levels?
    for (XMLNode *c = node->FirstChild(); c; c = c->NextSibling()) {
        if (std::string(c->Value()) == "Level") {
            for (XMLNode *p = c->FirstChild(); p; p = p->NextSibling()) {
                if (std::string(p->Value()) == "Patch") {
                    UintahPatch patch;
                    XMLElement *lower_elem = p->FirstChildElement("lower");
                    std::sscanf(lower_elem->GetText(),
                                "[%f, %f, %f]",
                                &patch.lower.x,
                                &patch.lower.y,
                                &patch.lower.z);
                    patches.push_back(patch);
                }
            }
        }
    }
    return true;
}

bool read_uintah_timestep_data(const std::string &base_path,
                               XMLNode *node,
                               ParticleModel &model)
{
    for (XMLNode *c = node->FirstChild(); c; c = c->NextSibling()) {
        if (std::string(c->Value()) == "Datafile") {
            XMLElement *e = c->ToElement();
            if (!e) {
                std::cout << "Error parsing Uintah timestep data\n";
                return false;
            }
            const char *href = e->Attribute("href");
            if (!href) {
                std::cout << "Error parsing Uintah timestep data: Missing file href\n";
                return false;
            }
            const std::string data_file = base_path + "/" + std::string(href);
            std::cout << "Reading " << data_file << "\n";
            tinyxml2::XMLDocument doc;
            XMLError err = doc.LoadFile(data_file.c_str());
            if (err != XML_SUCCESS) {
                std::cout << "Error loading Uintah data file '" << data_file
                          << "': " << tinyxml_error_string(err) << "\n";
                return false;
            }
            if (!read_uintah_datafile(data_file, doc, model)) {
                std::cout << "Error reading Uintah data file " << data_file << "\n";
                return false;
            }
        }
    }
    return true;
}

bool read_uintah_timestep(const std::string &file_name, XMLElement *node, ParticleModel &model)
{
    const std::string base_path = get_file_basepath(file_name);
    std::cout << "base path : " << base_path << "\n";

    std::vector<UintahPatch> patches;
    for (XMLNode *c = node->FirstChild(); c; c = c->NextSibling()) {
        const std::string node_type = c->Value();
        if (node_type == "Meta") {
            if (!read_uintah_timestep_meta(c)) {
                return false;
            }
        }
    }
    XMLNode *c = node->FirstChildElement("Data");
    if (!c || !read_uintah_timestep_data(base_path, c, model)) {
        return false;
    }
    return true;
}
