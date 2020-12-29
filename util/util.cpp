#include "util.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <glm/ext.hpp>
#include "tinyxml2.h"

#ifndef __PPC__
#ifdef _WIN32
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::string format_bytes(size_t nbytes)
{
    const size_t giga = 1024 * 1024 * 1024;
    const size_t mega = 1024 * 1024;
    const size_t kilo = 1024;
    if (nbytes >= giga) {
        return std::to_string(static_cast<float>(nbytes) / giga) + "GB";
    } else if (nbytes >= mega) {
        return std::to_string(static_cast<float>(nbytes) / mega) + "MB";
    } else if (nbytes >= kilo) {
        return std::to_string(static_cast<float>(nbytes) / kilo) + "KB";
    } else {
        return std::to_string(nbytes) + "B";
    }
}

std::string pretty_print_count(const double count)
{
    const double giga = 1000000000;
    const double mega = 1000000;
    const double kilo = 1000;
    if (count > giga) {
        return std::to_string(count / giga) + " G";
    } else if (count > mega) {
        return std::to_string(count / mega) + " M";
    } else if (count > kilo) {
        return std::to_string(count / kilo) + " K";
    }
    return std::to_string(count);
}

uint64_t align_to(uint64_t val, uint64_t align)
{
    return ((val + align - 1) / align) * align;
}

std::string canonicalize_path(const std::string &path)
{
    std::string p = path;
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

std::string get_cpu_brand()
{
#ifdef __PPC__
    return "PowerPC";
#else
    std::string brand = "Unspecified";
    std::array<int, 4> regs;
#ifdef _WIN32
    __cpuid(regs.data(), 0x80000000);
#else
    __cpuid(0x80000000, regs[0], regs[1], regs[2], regs[3]);
#endif
    if (regs[0] >= 0x80000004) {
        char b[64] = {0};
        for (int i = 0; i < 3; ++i) {
#ifdef _WIN32
            __cpuid(regs.data(), 0x80000000 + i + 2);
#else
            __cpuid(0x80000000 + i + 2, regs[0], regs[1], regs[2], regs[3]);
#endif
            std::memcpy(b + i * sizeof(regs), regs.data(), sizeof(regs));
        }
        brand = b;
    }
    return brand;
#endif
}

std::string get_file_content(const std::string &fname)
{
    std::ifstream file{fname};
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << fname << std::endl;
        return "";
    }
    return std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

bool compute_divisor(uint32_t x, uint32_t &divisor)
{
    uint32_t upper_bound = uint32_t(std::sqrt(x));
    for (uint32_t i = 2; i <= upper_bound; ++i) {
        if (x % i == 0) {
            divisor = i;
            return true;
        }
    }
    return false;
}

glm::uvec3 compute_grid3d(uint32_t num)
{
    glm::uvec3 grid(1);
    uint32_t axis = 0;
    uint32_t divisor = 0;
    while (compute_divisor(num, divisor)) {
        grid[axis] *= divisor;
        num /= divisor;
        axis = (axis + 1) % 3;
    }
    if (num != 1) {
        grid[axis] *= num;
    }
    return grid;
}

glm::uvec2 compute_grid2d(uint32_t num)
{
    glm::uvec2 grid(1);
    uint32_t axis = 0;
    uint32_t divisor = 0;
    while (compute_divisor(num, divisor)) {
        grid[axis] *= divisor;
        num /= divisor;
        axis = (axis + 1) % 2;
    }
    if (num != 1) {
        grid[axis] *= num;
    }
    return grid;
}

std::string get_file_extension(const std::string &fname)
{
    const size_t fnd = fname.find_last_of('.');
    if (fnd == std::string::npos) {
        return "";
    }
    return fname.substr(fnd + 1);
}

std::string get_file_basename(const std::string &path)
{
    size_t fname_offset = path.find_last_of('/');
    if (fname_offset == std::string::npos) {
        return path;
    }
    return path.substr(fname_offset + 1);
}

std::string get_file_basepath(const std::string &path)
{
    size_t end = path.find_last_of('/');
    if (end == std::string::npos) {
        return ".";
    }
    return path.substr(0, end);
}

bool starts_with(const std::string &str, const std::string &prefix)
{
    return std::strncmp(str.c_str(), prefix.c_str(), prefix.size()) == 0;
}

bool is_big_endian()
{
    const uint32_t x = 0x01020304;
    const uint8_t *v = reinterpret_cast<const uint8_t *>(&x);
    return v[0] == 0x01;
}

std::vector<uint32_t> count_bitmap_chunks(uint32_t m)
{
    std::vector<uint32_t> chunks;
    uint32_t current_size = 0;
    // TODO: We could do some neat faster thing w/ leading zero count and shifts
    // based on that
    for (uint32_t i = 0; i < 32; ++i) {
        if (m & (1 << i)) {
            ++current_size;
        } else if (current_size != 0) {
            chunks.push_back(current_size);
            current_size = 0;
        }
    }
    if (current_size != 0) {
        chunks.push_back(current_size);
    }
    return chunks;
}

std::string tinyxml_error_string(const tinyxml2::XMLError e)
{
    using namespace tinyxml2;

    switch (e) {
    case XML_NO_ATTRIBUTE:
        return "XML_NO_ATTRIBUTE";
    case XML_WRONG_ATTRIBUTE_TYPE:
        return "XML_WRONG_ATTRIBUTE_TYPE";
    case XML_ERROR_FILE_NOT_FOUND:
        return "XML_ERROR_FILE_NOT_FOUND";
    case XML_ERROR_FILE_COULD_NOT_BE_OPENED:
        return "XML_ERROR_FILE_COULD_NOT_BE_OPENED";
    case XML_ERROR_FILE_READ_ERROR:
        return "XML_ERROR_FILE_READ_ERROR";
    case XML_ERROR_PARSING_ELEMENT:
        return "XML_ERROR_PARSING_ELEMENT";
    case XML_ERROR_PARSING_ATTRIBUTE:
        return "XML_ERROR_PARSING_ATTRIBUTE";
    case XML_ERROR_PARSING_TEXT:
        return "XML_ERROR_PARSING_TEXT";
    case XML_ERROR_PARSING_CDATA:
        return "XML_ERROR_PARSING_CDATA";
    case XML_ERROR_PARSING_COMMENT:
        return "XML_ERROR_PARSING_COMMENT";
    case XML_ERROR_PARSING_DECLARATION:
        return "XML_ERROR_PARSING_DECLARATION";
    case XML_ERROR_PARSING_UNKNOWN:
        return "XML_ERROR_PARSING_UNKNOWN";
    case XML_ERROR_EMPTY_DOCUMENT:
        return "XML_ERROR_EMPTY_DOCUMENT";
    case XML_ERROR_MISMATCHED_ELEMENT:
        return "XML_ERROR_MISMATCHED_ELEMENT";
    case XML_ERROR_PARSING:
        return "XML_ERROR_PARSING";
    case XML_CAN_NOT_CONVERT_TEXT:
        return "XML_CAN_NOT_CONVERT_TEXT";
    case XML_NO_TEXT_NODE:
        return "XML_NO_TEXT_NODE";
    case XML_ELEMENT_DEPTH_EXCEEDED:
        return "XML_ELEMENT_DEPTH_EXCEEDED";
    default:
        return "XML_SUCCESS";
    }
}
