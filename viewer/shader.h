#pragma once

#include <string>
#include <unordered_map>
#include "gl_core_4_5.h"

struct Shader {
    GLuint program;
    std::unordered_map<std::string, GLint> uniforms;

    Shader(const std::string &vert_file, const std::string &frag_file);
    template <typename T>
    void uniform(const std::string &unif, const T &t);

    void use();

private:
    // Parse the uniform variable declarations in the src file and
    // add them to the uniforms map
    void parse_uniforms(const std::string &src);
};
