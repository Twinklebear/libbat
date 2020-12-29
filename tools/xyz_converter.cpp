#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "borrowed_array.h"
#include "lba_tree_builder.h"

const std::string USAGE = "Usage: ./xyz_converter <file.xyz> <out.bat>";

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

    std::cout << "Converting " << args[1] << "\n";
    std::ifstream fin(args[1].c_str());

    // First line has the number of atoms, second has the name of the data
    std::string line;
    std::getline(fin, line);
    const size_t num_atoms = std::stoi(line);
    std::cout << "Expecting: " << num_atoms << " atoms\n";

    std::getline(fin, line);
    std::cout << "Molecule name: " << line << "\n";

    // XYZ format is assumed to be TYPE X Y Z
    std::vector<glm::vec3> points;
    std::vector<int> atom_ids;
    int next_atom_id = 0;
    std::unordered_map<std::string, int> atom_id_map;
    while (std::getline(fin, line)) {
        if (line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        std::string type;
        float x, y, z;
        ss >> type >> x >> y >> z;

        int atom_id = -1;
        auto fnd = atom_id_map.find(type);
        if (fnd != atom_id_map.end()) {
            atom_id = fnd->second;
        } else {
            atom_id = next_atom_id++;
            atom_id_map[type] = atom_id;
        }

        points.emplace_back(x, y, z);
        atom_ids.push_back(atom_id);
        if (points.size() == num_atoms) {
            break;
        }
    }

    auto atom_arr = std::make_shared<BorrowedArray<uint8_t>>(
        reinterpret_cast<uint8_t *>(atom_ids.data()), sizeof(int) * atom_ids.size());

    std::vector<Attribute> attributes = {
        Attribute(AttributeDescription("atom_id", INT_32), atom_arr)};

    BATree tree = LBATreeBuilder(std::move(points), std::move(attributes)).compact();
    write_ba_tree(args[2], tree);

    return 0;
}
