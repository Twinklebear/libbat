add_subdirectory(imgui)
add_subdirectory(shaders)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(bat_viewer main.cpp
    arcball_camera.cpp
    debug.cpp
    app_util.cpp
    shader.cpp
    transfer_function_widget.cpp
    imgui_impl_sdl.cpp
    imgui_impl_opengl3.cpp
    gl_core_4_5.c)

set_target_properties(bat_viewer PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(bat_viewer PUBLIC
    imgui
    bat
    util
    ${SDL2_LIBRARY}
    ${OPENGL_LIBRARIES}
    TBB::tbb)

target_compile_definitions(bat_viewer PUBLIC
    SDL_MAIN_HANDLED
    TFN_WIDGET_NO_STB_IMAGE_IMPL
    $<$<CONFIG:Debug>:DEBUG_GL>)

target_include_directories(bat_viewer PUBLIC
    $<BUILD_INTERFACE:${SDL2_INCLUDE_DIR}>
    $<BUILD_INTERFACE:${OPENGL_INCLUDE_DIR}>)

