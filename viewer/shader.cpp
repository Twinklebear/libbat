#include "shader.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <glm/ext.hpp>
#include <glm/glm.hpp>
#include "util.h"

// Load a file's content and its includes returning the file with includes inserted
// and #line directives for better GLSL error messages within the included files
// the vector of file names will be filled with the file name for each file name number
// in the #line directive
std::string load_shader_file(const std::string &fname, std::vector<std::string> &file_names);

// Load a GLSL shader from the file. Returns -1 if loading fails and prints
// out the compilation errors
GLint compile_shader(GLenum type,
                     const std::string &src,
                     const std::vector<std::string> &file_names);

// Load a GLSL shader program from the shader files specified. The pair
// to specify a shader is { shader type, shader file }
// Returns -1 if program creation fails
GLint load_program(const std::vector<std::pair<GLenum, std::string>> &shader_files);

Shader::Shader(const std::string &vert_file, const std::string &frag_file)
{
    std::vector<std::string> vert_fnames;
    const std::string vert_src = load_shader_file(vert_file, vert_fnames);
    GLint vert = compile_shader(GL_VERTEX_SHADER, vert_src, vert_fnames);
    if (vert == -1) {
        throw std::runtime_error("Failed to compile vertex shader");
    }

    std::vector<std::string> frag_fnames;
    const std::string frag_src = load_shader_file(frag_file, frag_fnames);
    GLint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src, frag_fnames);
    if (frag == -1) {
        throw std::runtime_error("Failed to compile fragment shader");
    }

    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);
    GLint status;
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (status == GL_FALSE) {
        std::cout << "Error loading shader program: Program failed to link, log:\n";
        GLint len;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len, '\0');
        log.resize(len);
        glGetProgramInfoLog(program, log.size(), 0, log.data());
        std::cout << log.data() << "\n";
    }

    glDetachShader(program, vert);
    glDetachShader(program, frag);
    glDeleteShader(vert);
    glDeleteShader(frag);

    if (status == GL_FALSE) {
        glDeleteProgram(program);
        throw std::runtime_error("Failed to link program");
    }

    parse_uniforms(vert_src);
    parse_uniforms(frag_src);
}

/*
Shader::~Shader()
{
    glDeleteProgram(program);
}
*/

template <>
void Shader::uniform<bool>(const std::string &unif, const bool &t)
{
    glUniform1i(uniforms[unif], t ? 1 : 0);
}

template <>
void Shader::uniform<int>(const std::string &unif, const int &t)
{
    glUniform1i(uniforms[unif], t);
}

template <>
void Shader::uniform<unsigned int>(const std::string &unif, const unsigned int &t)
{
    glUniform1ui(uniforms[unif], t);
}

template <>
void Shader::uniform<float>(const std::string &unif, const float &t)
{
    glUniform1f(uniforms[unif], t);
}

template <>
void Shader::uniform<glm::vec2>(const std::string &unif, const glm::vec2 &t)
{
    glUniform2fv(uniforms[unif], 1, &t.x);
}

template <>
void Shader::uniform<glm::vec3>(const std::string &unif, const glm::vec3 &t)
{
    glUniform3fv(uniforms[unif], 1, &t.x);
}

template <>
void Shader::uniform<glm::vec4>(const std::string &unif, const glm::vec4 &t)
{
    glUniform4fv(uniforms[unif], 1, &t.x);
}

template <>
void Shader::uniform<glm::mat4>(const std::string &unif, const glm::mat4 &t)
{
    glUniformMatrix4fv(uniforms[unif], 1, GL_FALSE, glm::value_ptr(t));
}

void Shader::use()
{
    glUseProgram(program);
}

void Shader::parse_uniforms(const std::string &src)
{
    const std::regex regex_unif("uniform[^;]+[ ](\\w+);");
    for (auto it = std::sregex_iterator(src.begin(), src.end(), regex_unif);
         it != std::sregex_iterator();
         ++it) {
        const std::smatch &m = *it;
        uniforms[m[1]] = glGetUniformLocation(program, m[1].str().c_str());
    }
}

std::string load_shader_file(const std::string &fname, std::vector<std::string> &file_names)
{
    if (std::find(file_names.begin(), file_names.end(), fname) != file_names.end()) {
        std::cout << "Multiple includes of file " << fname << " detected, dropping this include\n";
        return "";
    }
    std::string content = get_file_content(fname);
    file_names.push_back(fname);
    // Insert the current file name index and line number for this file to preserve error logs
    // before inserting the included file contents
    size_t inc = content.rfind("#include");
    if (inc != std::string::npos) {
        size_t line_no = std::count_if(
            content.begin(), content.begin() + inc, [](const char &c) { return c == '\n'; });
        content.insert(content.find("\n", inc) + 1,
                       "#line " + std::to_string(line_no + 2) + " " +
                           std::to_string(file_names.size() - 1) + "\n");
    } else if (file_names.size() > 1) {
        content.insert(0, "#line 1 " + std::to_string(file_names.size() - 1) + "\n");
    }
    std::string dir = fname.substr(0, fname.rfind('/') + 1);
    // Insert includes backwards so we don't waste time parsing through the inserted file after
    // inserting
    for (; inc != std::string::npos; inc = content.rfind("#include", inc - 1)) {
        size_t open = content.find("\"", inc + 8);
        size_t close = content.find("\"", open + 1);
        std::string included = content.substr(open + 1, close - open - 1);
        content.erase(inc, close - inc + 2);
        std::string include_content = load_shader_file(dir + included, file_names);
        if (!include_content.empty()) {
            content.insert(inc, include_content);
        }
    }
    return content;
}

// Extract the file number the error occured in from the log message
// it's expected that the message begins with file_no(line_no)
// TODO: Is this always the form of the compilation errors?
// TODO it's not always the form of the compilation errors, need to
// handle intel and possible AMD differences properly
int get_file_num(const std::vector<char> &log)
{
    auto paren = std::find(log.begin(), log.end(), '(');
    std::string file_no{log.begin(), paren};
    // return std::stoi(file_no);
    return 0;
}
GLint compile_shader(GLenum type,
                     const std::string &src,
                     const std::vector<std::string> &file_names)
{
    GLuint shader = glCreateShader(type);
    const char *csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, 0);
    glCompileShader(shader);
    GLint status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
        std::cout << "Shader compilation error, ";
        switch (type) {
        case GL_VERTEX_SHADER:
            std::cout << "Vertex shader: ";
            break;
        case GL_FRAGMENT_SHADER:
            std::cout << "Fragment shader: ";
            break;
        case GL_GEOMETRY_SHADER:
            std::cout << "Geometry shader: ";
            break;
        case GL_COMPUTE_SHADER:
            std::cout << "Compute shader: ";
            break;
        case GL_TESS_CONTROL_SHADER:
            std::cout << "Tessellation Control shader: ";
            break;
        case GL_TESS_EVALUATION_SHADER:
            std::cout << "Tessellation Evaluation shader: ";
            break;
        default:
            std::cout << "Unknown shader type: ";
        }
        std::cout << file_names[0] << " failed to compile. Compilation log:\n";
        GLint len;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len, '\0');
        log.resize(len);
        glGetShaderInfoLog(shader, log.size(), 0, log.data());
        std::cout << "In file: " << file_names[get_file_num(log)] << ":\n" << log.data() << "\n";
        glDeleteShader(shader);
        return -1;
    }
    return shader;
}
