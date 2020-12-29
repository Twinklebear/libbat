#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "binary_particle_file.h"
#include "lba_tree_builder.h"

const std::string USAGE = "Usage: ./binary_particle_converter <file.bp> <out.bat>";

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

    BinaryParticleDump particles = read_binary_particle_file(args[1]);

    std::vector<glm::vec3> &points =
        std::dynamic_pointer_cast<OwnedArray<glm::vec3>>(particles.points)->array;
    BATree tree = LBATreeBuilder(std::move(points), particles.attribs).compact();
    write_ba_tree(args[2], tree);

    return 0;
}
