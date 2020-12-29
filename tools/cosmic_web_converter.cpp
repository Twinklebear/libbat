#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "borrowed_array.h"
#include "file_mapping.h"
#include "lba_tree_builder.h"
#include "util.h"

const std::string USAGE = "Usage: ./cosmic_web_converter [files.dat] <out.bat>";

#pragma pack(push, 1)
struct CosmicWebHeader {
    // The number of particles in this dat file
    int np_local;
    float a, t, tau;
    int nts;
    float dt_f_acc, dt_pp_acc, dt_c_acc;
    int cur_checkpoint, cur_projection, cur_halofind;
    float massp;
};
#pragma pack(pop)

std::ostream &operator<<(std::ostream &os, const CosmicWebHeader &h)
{
    os << "{\n"
       << "  np_local = " << h.np_local << "\n"
       << "  a = " << h.a << "\n"
       << "  t = " << h.t << "\n"
       << "  tau = " << h.tau << "\n"
       << "  nts = " << h.nts << "\n"
       << "  dt_f_acc = " << h.dt_f_acc << "\n"
       << "  dt_pp_acc = " << h.dt_pp_acc << "\n"
       << "  dt_c_acc = " << h.dt_c_acc << "\n"
       << "  cur_checkpoint = " << h.cur_checkpoint << "\n"
       << "  cur_halofind = " << h.cur_halofind << "\n"
       << "  massp = " << h.massp << "\n}";
    return os;
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv, argv + argc);
    if (args.size() < 3) {
        std::cout << USAGE << "\n";
        return 1;
    }

    std::vector<std::string> dat_files;
    std::string output_file;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "-h") {
            std::cout << USAGE << "\n";
            return 0;
        } else if (get_file_extension(args[i]) == "dat") {
            dat_files.push_back(canonicalize_path(args[i]));
        } else {
            output_file = canonicalize_path(args[i]);
        }
    }

    if (dat_files.empty() || output_file.empty()) {
        std::cout
            << "Error: at least on Cosmic Web (.dat) file is required along with an output "
               "file name (.bat)\n";
        return 1;
    }

    // Each cell is 768x768x768 units
    const float step = 768.f;

    std::vector<glm::vec3> points;
    // Cosmic web has the x/y/z velocity components as attributes
    std::vector<std::vector<float>> velocities = {
        std::vector<float>(), std::vector<float>(), std::vector<float>()};
    for (const auto &dat : dat_files) {
        glm::ivec3 brick(0);
        const std::string basename = get_file_basename(dat);

        std::sscanf(basename.c_str(), "0.000xv%1d%1d%1d.dat", &brick.x, &brick.y, &brick.z);
        std::cout << "brick " << glm::to_string(brick) << "\n";

        FileMapping mapping(dat);

        const CosmicWebHeader header =
            *reinterpret_cast<const CosmicWebHeader *>(mapping.data());
        std::cout << "Cosmic Web File '" << dat << "'\n"
                  << "Brick: " << glm::to_string(brick) << "\n"
                  << header << "\n";

        const glm::vec3 offset(step * brick.x, step * brick.y, step * brick.z);
        std::cout << "brick offset: " << glm::to_string(offset) << "\n";

        points.reserve(points.size() + header.np_local);

        // Each particle stores the position and velocity as a pair of vec3f
        const glm::vec3 *vecs =
            reinterpret_cast<const glm::vec3 *>(mapping.data() + sizeof(CosmicWebHeader));
        for (int i = 0; i < header.np_local; ++i) {
            const glm::vec3 pos = vecs[i * 2] + offset;
            const glm::vec3 &vel = vecs[i * 2 + 1];
            points.push_back(pos);
            for (size_t j = 0; j < 3; ++j) {
                velocities[j].push_back(vel[j]);
            }
        }
    }

    auto v_x = std::dynamic_pointer_cast<AbstractArray<uint8_t>>(
        std::make_shared<BorrowedArray<uint8_t>>(
            reinterpret_cast<uint8_t *>(velocities[0].data()), sizeof(float) * points.size()));

    auto v_y = std::dynamic_pointer_cast<AbstractArray<uint8_t>>(
        std::make_shared<BorrowedArray<uint8_t>>(
            reinterpret_cast<uint8_t *>(velocities[1].data()), sizeof(float) * points.size()));

    auto v_z = std::dynamic_pointer_cast<AbstractArray<uint8_t>>(
        std::make_shared<BorrowedArray<uint8_t>>(
            reinterpret_cast<uint8_t *>(velocities[2].data()), sizeof(float) * points.size()));

    std::vector<Attribute> attributes = {
        Attribute(AttributeDescription("v_x", FLOAT_32), v_x),
        Attribute(AttributeDescription("v_y", FLOAT_32), v_y),
        Attribute(AttributeDescription("v_z", FLOAT_32), v_z)};

    BATree tree = LBATreeBuilder(std::move(points), std::move(attributes)).compact();
    write_ba_tree(output_file, tree);

    return 0;
}
