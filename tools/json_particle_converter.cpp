#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "borrowed_array.h"
#include "json.hpp"
#include "lba_tree_builder.h"

using json = nlohmann::json;

const std::string USAGE = "Usage: ./json_particle_converter <file.json> <out.bat>";

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cout << USAGE << "\n";
        return 1;
    }
    std::string attr;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h") {
            std::cout << USAGE << "\n";
            return 0;
        }
        if (args[i] == "-attr") {
            attr = args[++i];
        }
    }

    std::cout << "Converting " << args[1] << "\n";
    std::ifstream fin(args[1].c_str());
    json content;
    fin >> content;

    std::vector<glm::vec3> points;
    points.reserve(content["points"].size() / 3);
    for (size_t i = 0; i < content["points"].size() / 3; ++i) {
        glm::vec3 v;
        v.x = content["points"][i * 3].get<float>();
        v.y = content["points"][i * 3 + 1].get<float>();
        v.z = content["points"][i * 3 + 2].get<float>();
        points.push_back(v);
    }

    std::vector<Attribute> attributes;
    std::vector<float> attribute_data;
    if (content.find("attribs") != content.end()) {
        json::iterator j = content["attribs"].begin();
        if (!attr.empty()) {
            for (auto it = content["attribs"].begin(); it != content["attribs"].end(); ++it) {
                if (it.value()["name"].get<std::string>() == attr) {
                    j = it;
                    break;
                }
            }
        }

        attr = j.value()["name"].get<std::string>();
        std::cout << "Loading attribute: " << attr << "\n";
        attribute_data.reserve(j.value()["data"].size());
        for (size_t i = 0; i < j.value()["data"].size(); ++i) {
            attribute_data.push_back(j.value()["data"][i].get<float>());
        }

        auto data_arr = std::make_shared<BorrowedArray<uint8_t>>(
            reinterpret_cast<uint8_t *>(attribute_data.data()),
            sizeof(float) * attribute_data.size());
        attributes.push_back(Attribute(AttributeDescription(attr, FLOAT_32), data_arr));
    }

    BATree tree = LBATreeBuilder(std::move(points), std::move(attributes)).compact();
    write_ba_tree(args[2], tree);

    return 0;
}

