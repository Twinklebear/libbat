#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "bat_file.h"
#include "borrowed_array.h"
#include "lba_tree_builder.h"

int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << "Usage: ./bitmap_tester <count> <lo> <hi>\n";
        return 1;
    }

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> distrib(-2.f, 2.f);

    const size_t num_vals = std::atoi(argv[1]);
    std::vector<float> vals;
    for (size_t i = 0; i < num_vals; ++i) {
        vals.push_back(distrib(rng));
    }

    std::sort(vals.begin(), vals.end());
    glm::vec2 range(vals[0], vals.back());

    auto arr = std::make_shared<BorrowedArray<uint8_t>>(reinterpret_cast<uint8_t *>(vals.data()),
                                                        vals.size() * sizeof(float));

    AttributeDescription desc("test", DTYPE::FLOAT_32, range);
    Attribute attr(desc, std::dynamic_pointer_cast<AbstractArray<uint8_t>>(arr));

    std::cout << "data range: " << glm::to_string(desc.range) << "\n";
    std::cout << "Data:\n";
    for (size_t i = 0; i < attr.size(); ++i) {
        std::cout << "[" << i << "]:";
        printf("(%f, 0x%08x)\n", vals[i], attr.bitmap(i));
    }

    glm::vec2 query_range(std::atof(argv[2]), std::atof(argv[3]));
    uint32_t mask = query_bitmask(query_range, desc.range);
    std::cout << "For query " << glm::to_string(query_range) << " mask: ";
    printf("0x%08x\n", mask);

    std::cout << "Data matching query:\n";
    for (size_t i = 0; i < attr.size(); ++i) {
        if (attr.bitmap(i) & mask) {
            std::cout << "[" << i << "]:";
            printf("(%f, 0x%08x)\n", vals[i], attr.bitmap(i));
        }
    }
    /*
    uint32_t test_mask = 0xffffffff >> 15;
    uint32_t result_mask = mask_for_range(test_mask, glm::vec2(0.f, 1.f), glm::vec2(0.5f, 0.5f));
    printf("test mask: 0x%08x\n", test_mask);
    printf("result mask: 0x%08x\n", result_mask);
    */

    return 0;
}

