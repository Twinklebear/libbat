#include "debug.h"
#include <cassert>
#include <iomanip>
#include <iostream>
#include <SDL.h>
#include "gl_core_4_5.h"

void log_debug_msg(
    GLenum src, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar *msg);

void register_debug_callback()
{
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(debug_callback, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
    glDebugMessageControl(GL_DEBUG_SOURCE_API,
                          GL_DEBUG_TYPE_OTHER,
                          GL_DEBUG_SEVERITY_NOTIFICATION,
                          0,
                          NULL,
                          GL_FALSE);
}
#ifdef _WIN32
void APIENTRY debug_callback(GLenum src,
                             GLenum type,
                             GLuint id,
                             GLenum severity,
                             GLsizei len,
                             const GLchar *msg,
                             const GLvoid *)
{
    log_debug_msg(src, type, id, severity, len, msg);
}
#else
void debug_callback(GLenum src,
                    GLenum type,
                    GLuint id,
                    GLenum severity,
                    GLsizei len,
                    const GLchar *msg,
                    const GLvoid *)
{
    log_debug_msg(src, type, id, severity, len, msg);
}
#endif
void log_debug_msg(
    GLenum src, GLenum type, GLuint, GLenum severity, GLsizei, const GLchar *msg)
{
    // Print a time stamp for the message
    float sec = SDL_GetTicks() / 1000.f;
    int min = static_cast<int>(sec / 60.f);
    sec -= sec / 60.f;
    std::cout << "[" << min << ":" << std::setprecision(3) << sec << "] OpenGL Debug -";
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        std::cout << " High severity";
        break;
    case GL_DEBUG_SEVERITY_MEDIUM:
        std::cout << " Medium severity";
        break;
    case GL_DEBUG_SEVERITY_LOW:
        std::cout << " Low severity";
        break;
    case GL_DEBUG_SEVERITY_NOTIFICATION:
        std::cout << " Notification";
        break;
    }
    switch (src) {
    case GL_DEBUG_SOURCE_API:
        std::cout << " API";
        break;
    case GL_DEBUG_SOURCE_WINDOW_SYSTEM:
        std::cout << " Window system";
        break;
    case GL_DEBUG_SOURCE_SHADER_COMPILER:
        std::cout << " Shader compiler";
        break;
    case GL_DEBUG_SOURCE_THIRD_PARTY:
        std::cout << " Third party";
        break;
    case GL_DEBUG_SOURCE_APPLICATION:
        std::cout << " Application";
        break;
    default:
        std::cout << " Other";
    }
    switch (type) {
    case GL_DEBUG_TYPE_ERROR:
        std::cout << " Error";
        break;
    case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
        std::cout << " Deprecated behavior";
        break;
    case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
        std::cout << " Undefined behavior";
        break;
    case GL_DEBUG_TYPE_PORTABILITY:
        std::cout << " Portability";
        break;
    case GL_DEBUG_TYPE_PERFORMANCE:
        std::cout << " Performance";
        break;
    default:
        std::cout << " Other";
    }
    std::cout << ":\n\t" << msg << "\n";

    // Break for a stack trace of sorts
    if (severity == GL_DEBUG_SEVERITY_HIGH || type == GL_DEBUG_TYPE_ERROR) {
        throw std::runtime_error("Severe OpenGL error!");
    }
}
