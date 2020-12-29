#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <SDL.h>
#include <glm/gtx/color_space.hpp>
#include "aggregation_tree.h"
#include "app_util.h"
#include "arcball_camera.h"
#include "ba_tree.h"
#include "bat_file.h"
#include "bat_handle.h"
#include "debug.h"
#include "file_mapping.h"
#include "gl_core_4_5.h"
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_sdl.h"
#include "pbat_file.h"
#include "shader.h"
#include "transfer_function_widget.h"
#include "util.h"

const std::string USAGE = "Usage: ./bat_viewer [file.bat...]";

struct TreeData {
    GLuint points_vbo = -1;
    GLuint planes_vbo = -1;
    GLuint attrib_vbo = -1;

    std::vector<glm::vec3> query_pts;
    std::vector<std::vector<uint8_t>> query_attributes;

    std::vector<glm::vec3> plane_verts;
    std::vector<std::string> attrib_names;
    std::vector<AttributeQuery> attribute_queries;
    QueryStats query_stats;
    uint64_t query_time_ms = 0;

    int drop_down_attrib = 0;
    std::vector<int> selected_attributes;
    std::vector<std::vector<glm::vec2>> query_ranges;
    int shader = 0;
    bool display = true;
    bool compute_magnitude = false;
    glm::vec2 magnitude_range;
    std::string file;
};

int win_width = 1280;
int win_height = 720;

void run_app(const std::vector<std::string> &args, SDL_Window *window);

std::vector<float> compute_attribute_magnitude(TreeData &data)
{
    std::vector<float> magnitudes;
    const size_t num_points = data.query_pts.size();
    magnitudes.reserve(num_points);
    data.magnitude_range.x = std::numeric_limits<float>::infinity();
    data.magnitude_range.y = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < num_points; ++i) {
        float x = 0.f;
        for (size_t j = 0; j < data.query_attributes.size(); ++j) {
            const auto &q = data.attribute_queries[j];
            const uint8_t *attr_val =
                data.query_attributes[j].data() + i * dtype_stride(q.data_type);
            float v = 0.f;
            if (q.data_type == INT_32) {
                v = *reinterpret_cast<const int32_t *>(attr_val);
            } else if (q.data_type == FLOAT_32) {
                v = *reinterpret_cast<const float *>(attr_val);
            } else if (q.data_type == FLOAT_64) {
                v = *reinterpret_cast<const double *>(attr_val);
            }
            x += v * v;
        }
        x = std::sqrt(x);
        data.magnitude_range.x = std::min(x, data.magnitude_range.x);
        data.magnitude_range.y = std::max(x, data.magnitude_range.y);
        magnitudes.push_back(x);
    }
    std::cout << "magnitude range: " << glm::to_string(data.magnitude_range) << "\n";
    return magnitudes;
}

glm::vec2 transform_mouse(glm::vec2 in)
{
    return glm::vec2(in.x * 2.f / win_width - 1.f, 1.f - 2.f * in.y / win_height);
}

std::vector<glm::vec3> make_box_verts(const Box &b);
std::vector<glm::vec3> make_plane_verts(const Plane &p);

int main(int argc, char **argv)
{
    const std::vector<std::string> args(argv, argv + argc);
    if (args.size() == 1) {
        std::cout << USAGE << "\n";
        return 1;
    }

    if (SDL_Init(SDL_INIT_EVERYTHING) != 0) {
        std::cerr << "Failed to init SDL: " << SDL_GetError() << "\n";
        return -1;
    }

    const char *glsl_version = "#version 450 core";
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 5);
#ifdef DEBUG_GL
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif

    // Create window with graphics context
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window *window = SDL_CreateWindow("BAT Viewer",
                                          SDL_WINDOWPOS_CENTERED,
                                          SDL_WINDOWPOS_CENTERED,
                                          win_width,
                                          win_height,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    SDL_GL_SetSwapInterval(1);
    SDL_GL_MakeCurrent(window, gl_context);

    if (ogl_LoadFunctions() == ogl_LOAD_FAILED) {
        std::cerr << "Failed to initialize OpenGL\n";
        return 1;
    }
#ifdef DEBUG_GL
    register_debug_callback();
#endif

    // Setup Dear ImGui context
    ImGui::CreateContext();

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init(glsl_version);

    run_app(args, window);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}

void run_app(const std::vector<std::string> &args, SDL_Window *window)
{
    ImGuiIO &io = ImGui::GetIO();

    const std::string base_path = get_base_path();

    std::vector<Shader> shaders = {
        Shader(base_path + "shaders/vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/int_attrib_vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/float_attrib_vert.glsl", base_path + "shaders/frag.glsl"),
        Shader(base_path + "shaders/double_attrib_vert.glsl",
               base_path + "shaders/frag.glsl")};

    Box world_bounds;

    std::vector<BATHandle> trees;
    std::vector<TreeData> tree_data;
    bool enable_range_filtering = true;
    bool enable_bitmap_filtering = true;
    for (size_t i = 1; i < args.size(); ++i) {
        if (args[i][0] == '-') {
            if (args[i] == "-disable-range") {
                enable_range_filtering = false;
            } else if (args[i] == "-disable-bitmap") {
                enable_bitmap_filtering = false;
            }
            continue;
        }
        trees.push_back(BATHandle(args[i]));

        const auto &t = trees.back();
        std::cout << "Tree '" << args[i] << "' info:\n"
                  << "bounds: " << t.bounds() << "\n"
                  << "# points: " << t.num_points() << "\n"
                  << "# attribs: " << t.attributes.size() << "\n";

        TreeData data;
        data.file = args[i];
        world_bounds.box_union(t.bounds());

        for (size_t j = 0; j < t.attributes.size(); ++j) {
            const auto &a = t.attributes[j];
            data.attrib_names.push_back(a.name);

            std::cout << "  attrib: " << a.name << ", "
                      << "range: " << glm::to_string(a.range) << "\n";
        }

        glCreateBuffers(1, &data.points_vbo);
        glCreateBuffers(1, &data.planes_vbo);
        glCreateBuffers(1, &data.attrib_vbo);
        tree_data.push_back(data);
    }
    std::cout << "Total world bounds: " << world_bounds << "\n";

    const float world_diag_len = glm::length(world_bounds.diagonal());
    ArcballCamera camera(world_bounds.center() - glm::vec3(0.f, 0.f, world_diag_len * 1.5),
                         world_bounds.center(),
                         glm::vec3(0, 1, 0));
    glm::mat4 proj = glm::perspective(
        glm::radians(65.f), static_cast<float>(win_width) / win_height, 0.1f, 5000.f);

    if (!enable_range_filtering) {
        std::cout << "Range filtering disabled\n";
    }
    if (!enable_bitmap_filtering) {
        std::cout << "Bitmap filtering disabled\n";
    }

    for (auto &t : trees) {
        t.set_enable_range_filtering(enable_range_filtering);
        t.set_enable_bitmap_filtering(enable_bitmap_filtering);
    }

    Box query_box = world_bounds;
    std::vector<glm::vec3> query_box_verts;

    GLuint vao;
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLuint query_box_vbo;
    glCreateBuffers(1, &query_box_vbo);

    glEnableVertexAttribArray(0);

    TransferFunctionWidget tfn_widget;
    // Texture for sampling the transfer function data
    GLuint colormap_texture;
    glGenTextures(1, &colormap_texture);
    glBindTexture(GL_TEXTURE_1D, colormap_texture);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MIN_LOD, 0);
    glTexParameterf(GL_TEXTURE_1D, GL_TEXTURE_MAX_LOD, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    {
        auto colormap = tfn_widget.get_colormap();
        glTexImage1D(GL_TEXTURE_1D,
                     0,
                     GL_RGBA8,
                     colormap.size() / 4,
                     0,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     colormap.data());
    }

    const float max_point_size = 8.f;
    float point_size = max_point_size;
    glPointSize(point_size);
    glClearColor(0.1f, 0.1f, 0.1f, 1.f);
    glClearDepth(1.f);
    glEnable(GL_DEPTH_TEST);

    glm::vec2 prev_mouse(-2.f);
    size_t total_points = 0;
    size_t displayed_points = 0;
    float prev_quality = 0.f;
    float query_quality = 0.01f;
    bool done = false;
    bool auto_point_size = true;
    bool draw_splitting_planes = false;
    bool draw_query_box = false;
    bool query_changed = true;
    bool quality_changed = true;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT) {
                done = true;
            }
            if (event.type == SDL_WINDOWEVENT) {
                if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
                    done = true;
                } else if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    win_width = event.window.data1;
                    win_height = event.window.data2;

                    proj = glm::perspective(glm::radians(65.f),
                                            static_cast<float>(win_width) / win_height,
                                            0.1f,
                                            500.f);
                }
            }
            if (!io.WantCaptureKeyboard) {
                if (event.type == SDL_KEYDOWN) {
                    switch (event.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        done = true;
                        break;
                    default:
                        break;
                    }
                }
            }
            if (!io.WantCaptureMouse) {
                if (event.type == SDL_MOUSEMOTION) {
                    const glm::vec2 cur_mouse =
                        transform_mouse(glm::vec2(event.motion.x, event.motion.y));
                    if (prev_mouse != glm::vec2(-2.f)) {
                        if (event.motion.state & SDL_BUTTON_LMASK) {
                            camera.rotate(prev_mouse, cur_mouse);
                        } else if (event.motion.state & SDL_BUTTON_RMASK) {
                            camera.pan(cur_mouse - prev_mouse);
                        }
                    }
                    prev_mouse = cur_mouse;
                } else if (event.type == SDL_MOUSEWHEEL) {
                    camera.zoom(event.wheel.y * 0.05 * world_diag_len);
                }
            }
        }

        if (tfn_widget.changed()) {
            auto colormap = tfn_widget.get_colormap();
            glTexImage1D(GL_TEXTURE_1D,
                         0,
                         GL_RGBA8,
                         colormap.size() / 4,
                         0,
                         GL_RGBA,
                         GL_UNSIGNED_BYTE,
                         colormap.data());
        }

        const glm::mat4 proj_view = proj * camera.transform();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame(window);
        ImGui::NewFrame();

        if (ImGui::Begin("Transfer Function")) {
            tfn_widget.draw_ui();
        }
        ImGui::End();

        ImGui::Begin("Debug Panel");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate,
                    ImGui::GetIO().Framerate);

        if (ImGui::SliderFloat("Point Size", &point_size, 1.f, max_point_size)) {
            glPointSize(point_size);
        }

        ImGui::Checkbox("Automatic Point Size", &auto_point_size);

        // We need a separate slider for each b/c we have different ranges for each
        std::array<float, 2> x_range = {query_box.lower.x, query_box.upper.x};
        std::array<float, 2> y_range = {query_box.lower.y, query_box.upper.y};
        std::array<float, 2> z_range = {query_box.lower.z, query_box.upper.z};

        const std::string pretty_total = pretty_print_count(total_points);
        const std::string pretty_displayed = pretty_print_count(displayed_points);
        const size_t dataset_points = 0;
        std::accumulate(trees.begin(), trees.end(), 0, [](size_t s, const BATHandle &t) {
            return s + t.num_points();
        });
        const std::string pretty_dataset = pretty_print_count(dataset_points);
        ImGui::Text("Query Box");
        ImGui::Text("Queried Points: %s", pretty_total.c_str());
        ImGui::Text("Displayed Points: %s", pretty_displayed.c_str());
        ImGui::Text("Data Set Points: %s", pretty_dataset.c_str());

        quality_changed = ImGui::SliderFloat("Quality", &query_quality, 0.01f, 1.f);

        query_changed |= ImGui::SliderFloat2(
            "x range", x_range.data(), world_bounds.lower.x, world_bounds.upper.x);
        query_changed |= ImGui::SliderFloat2(
            "y range", y_range.data(), world_bounds.lower.y, world_bounds.upper.y);
        query_changed |= ImGui::SliderFloat2(
            "z range", z_range.data(), world_bounds.lower.z, world_bounds.upper.z);

        ImGui::Checkbox("Draw splitting planes", &draw_splitting_planes);
        ImGui::Checkbox("Draw query box", &draw_query_box);

        for (size_t i = 0; i < tree_data.size(); ++i) {
            auto &td = tree_data[i];
            ImGui::PushID(i);

            ImGui::Separator();
            ImGui::Text("Tree %s", td.file.c_str());
            const std::string pretty_tree_count = pretty_print_count(td.query_pts.size());
            const std::string total_tree_count = pretty_print_count(trees[i].num_points());
            ImGui::Text("Queried Points: %s", pretty_tree_count.c_str());
            ImGui::Text("Total Points: %s", total_tree_count.c_str());
            ImGui::Checkbox("Display", &td.display);

            std::stringstream query_stats_text;
            query_stats_text << "Query took: " << td.query_time_ms << "ms\n" << td.query_stats;
            ImGui::Text("%s", query_stats_text.str().c_str());

            if (!td.attrib_names.empty()) {
                std::vector<const char *> strs;
                std::transform(td.attrib_names.begin(),
                               td.attrib_names.end(),
                               std::back_inserter(strs),
                               [](const std::string &s) { return s.c_str(); });

                ImGui::Combo("Attributes", &td.drop_down_attrib, strs.data(), strs.size());
                if (ImGui::Checkbox("Compute Magnitude", &td.compute_magnitude)) {
                    query_changed = true;
                }

                ImGui::Text("Attribute Range: [%f, %f]",
                            trees[i].attributes[td.drop_down_attrib].range.x,
                            trees[i].attributes[td.drop_down_attrib].range.y);

                if (ImGui::Button("+")) {
                    auto fnd = std::find(td.selected_attributes.begin(),
                                         td.selected_attributes.end(),
                                         td.drop_down_attrib);
                    if (fnd == td.selected_attributes.end()) {
                        td.selected_attributes.push_back(td.drop_down_attrib);
                        td.query_ranges.push_back(std::vector<glm::vec2>{
                            trees[i].attributes[td.drop_down_attrib].range});
                        query_changed = true;
                    } else {
                        const size_t id = std::distance(td.selected_attributes.begin(), fnd);
                        td.query_ranges[id].push_back(
                            trees[i].attributes[td.drop_down_attrib].range);
                    }
                }

                if (!td.selected_attributes.empty()) {
                    ImGui::SameLine();
                    if (ImGui::Button("-")) {
                        td.selected_attributes.pop_back();
                        td.query_ranges.pop_back();
                        query_changed = true;
                    }
                }

                for (size_t j = 0; j < td.selected_attributes.size(); ++j) {
                    const int attr_id = td.selected_attributes[j];
                    ImGui::PushID(j);

                    ImGui::Text("Attribute '%s' range: [%f, %f]",
                                td.attrib_names[attr_id].c_str(),
                                trees[i].attributes[attr_id].range.x,
                                trees[i].attributes[attr_id].range.y);

                    for (size_t k = 0; k < td.query_ranges[j].size(); ++k) {
                        ImGui::PushID(k);
                        glm::vec2 &query_range = td.query_ranges[j][k];
                        query_changed |=
                            ImGui::SliderFloat2("Query Range",
                                                &query_range.x,
                                                trees[i].attributes[attr_id].range.x,
                                                trees[i].attributes[attr_id].range.y + 0.001);

                        if (query_range.x > query_range.y) {
                            std::swap(query_range.x, query_range.y);
                        }
                        ImGui::PopID();
                    }

                    ImGui::PopID();
                }
            }

            ImGui::PopID();
        }

        ImGui::End();

        if (query_changed || quality_changed) {
            total_points = 0;
            displayed_points = 0;

            glm::vec3 lower(x_range[0], y_range[0], z_range[0]);
            glm::vec3 upper(x_range[1], y_range[1], z_range[1]);
            query_box = Box(glm::min(lower, upper), glm::max(lower, upper));

            bool data_reset = false;
            if (query_changed || prev_quality > query_quality) {
                prev_quality = 0.f;
                data_reset = true;
            }

            for (size_t i = 0; i < trees.size(); ++i) {
                using namespace std::chrono;

                auto &tree = trees[i];
                TreeData &data = tree_data[i];

                if (data_reset) {
                    data.query_pts.clear();
                    data.query_attributes.clear();
                }
                std::vector<size_t> query_attribute_indices;
                if (!data.attrib_names.empty() && !data.selected_attributes.empty()) {
                    data.attribute_queries.clear();
                    for (size_t j = 0; j < data.selected_attributes.size(); ++j) {
                        if ((data.compute_magnitude && j < 3) || j == 0) {
                            query_attribute_indices.push_back(data.selected_attributes[j]);
                        }
                        data.attribute_queries.emplace_back(
                            data.attrib_names[data.selected_attributes[j]],
                            data.query_ranges[j]);
                    }
                    data.query_attributes.resize(query_attribute_indices.size());
                }

                auto start = steady_clock::now();
                data.query_stats = tree.query_box_progressive(
                    query_box,
                    &data.attribute_queries,
                    prev_quality,
                    query_quality,
                    [&](const size_t id,
                        const glm::vec3 &pos,
                        const std::vector<Attribute> &attribs) {
                        data.query_pts.push_back(pos);

                        for (size_t i = 0; i < data.query_attributes.size(); ++i) {
                            const Attribute &attr = attribs[query_attribute_indices[i]];
                            std::copy(attr.at(id),
                                      attr.at(id) + attr.desc.stride(),
                                      std::back_inserter(data.query_attributes[i]));
                        }
                    });
                auto end = steady_clock::now();
                auto dur = duration_cast<milliseconds>(end - start);
                data.query_time_ms = dur.count();
                std::cout << "Query took " << data.query_time_ms << "ms\n";
                std::cout << data.query_stats << "\n";

                total_points += data.query_pts.size();
                if (data.display) {
                    displayed_points += data.query_pts.size();
                }

                if (!data.query_pts.empty()) {
                    glBindBuffer(GL_ARRAY_BUFFER, data.points_vbo);
                    glBufferData(GL_ARRAY_BUFFER,
                                 data.query_pts.size() * sizeof(glm::vec3),
                                 data.query_pts.data(),
                                 GL_STATIC_DRAW);

                    data.shader = 0;
                    if (!data.attribute_queries.empty()) {
                        if (data.compute_magnitude) {
                            data.shader = 2;
                        } else if (data.attribute_queries[0].data_type == INT_32) {
                            data.shader = 1;
                        } else if (data.attribute_queries[0].data_type == FLOAT_32) {
                            data.shader = 2;
                        } else if (data.attribute_queries[0].data_type == FLOAT_64) {
                            data.shader = 3;
                        } else {
                            std::cout
                                << "Warning: Unhandled attribute/shader type combination\n";
                        }
                        std::cout << "# particles: " << data.query_pts.size() << "\n";
                        for (size_t i = 0; i < data.query_attributes.size(); ++i) {
                            const AttributeDescription &attr =
                                tree.attributes[query_attribute_indices[i]];
                            std::cout << "attrib type: " << print_data_type(attr.data_type)
                                      << ", # attrib queried: "
                                      << data.query_attributes[i].size() / attr.stride()
                                      << ", stride: " << attr.stride() << "\n";
                        }

                        if (!query_attribute_indices.empty()) {
                            glBindBuffer(GL_ARRAY_BUFFER, data.attrib_vbo);
                            if (data.compute_magnitude) {
                                auto magnitude = compute_attribute_magnitude(data);
                                glBufferData(GL_ARRAY_BUFFER,
                                             magnitude.size() * sizeof(float),
                                             magnitude.data(),
                                             GL_STATIC_DRAW);
                            } else {
                                glBufferData(GL_ARRAY_BUFFER,
                                             data.query_attributes[0].size(),
                                             data.query_attributes[0].data(),
                                             GL_STATIC_DRAW);
                            }
                        }
                    }
                }

                // For debug vis: query the splitting planes of the tree down this path
                std::vector<Plane> splitting_planes;
                tree.get_splitting_planes(
                    splitting_planes, query_box, &data.attribute_queries, query_quality);
                data.plane_verts.clear();
                for (const auto &p : splitting_planes) {
                    const auto plane_verts = make_plane_verts(p);
                    std::copy(plane_verts.begin(),
                              plane_verts.end(),
                              std::back_inserter(data.plane_verts));
                }
                glBindBuffer(GL_ARRAY_BUFFER, data.planes_vbo);
                glBufferData(GL_ARRAY_BUFFER,
                             data.plane_verts.size() * sizeof(glm::vec3),
                             data.plane_verts.data(),
                             GL_STATIC_DRAW);
            }
            prev_quality = query_quality;

            query_box_verts = make_box_verts(query_box);
            glBindBuffer(GL_ARRAY_BUFFER, query_box_vbo);
            glBufferData(GL_ARRAY_BUFFER,
                         query_box_verts.size() * sizeof(glm::vec3),
                         query_box_verts.data(),
                         GL_STATIC_DRAW);
        }

        // Rendering
        ImGui::Render();
        glViewport(0, 0, (int)io.DisplaySize.x, (int)io.DisplaySize.y);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_1D, colormap_texture);

        if (auto_point_size && quality_changed) {
            point_size = glm::lerp(max_point_size, 1.f, query_quality);
            glPointSize(point_size);
        }

        for (size_t i = 0; i < tree_data.size(); ++i) {
            const TreeData &td = tree_data[i];
            const auto &tree = trees[i];

            if (!td.display) {
                continue;
            }

            shaders[td.shader].use();
            shaders[td.shader].uniform("proj_view", proj_view);
            shaders[td.shader].uniform(
                "fcolor", glm::rgbColor(glm::vec3((360.f * i) / tree_data.size(), 1.f, 0.7f)));

            if (td.shader != 0) {
                if (td.query_attributes.empty()) {
                    shaders[td.shader].uniform("range", tree.attributes[0].range);
                } else if (!td.compute_magnitude) {
                    shaders[td.shader].uniform(
                        "range", tree.attributes[td.selected_attributes[0]].range);
                } else {
                    shaders[td.shader].uniform("range", td.magnitude_range);
                }
            }

            glBindBuffer(GL_ARRAY_BUFFER, td.points_vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

            glBindBuffer(GL_ARRAY_BUFFER, td.attrib_vbo);
            if (!td.selected_attributes.empty()) {
                glEnableVertexAttribArray(1);
                if (td.compute_magnitude) {
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
                } else if (td.attribute_queries[0].data_type == INT_32) {
                    glVertexAttribIPointer(1, 1, GL_INT, 0, 0);
                } else if (td.attribute_queries[0].data_type == FLOAT_32) {
                    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, 0);
                } else if (td.attribute_queries[0].data_type == FLOAT_64) {
                    glVertexAttribLPointer(1, 1, GL_DOUBLE, 0, 0);
                }
            }
            glDrawArrays(GL_POINTS, 0, td.query_pts.size());
            glDisableVertexAttribArray(1);

            if (draw_splitting_planes) {
                glBindBuffer(GL_ARRAY_BUFFER, td.planes_vbo);
                glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

                shaders[0].use();
                for (size_t j = 0; j < td.plane_verts.size() / 4; ++j) {
                    shaders[0].uniform("proj_view", proj_view);
                    shaders[0].uniform(
                        "fcolor",
                        glm::rgbColor(
                            glm::vec3((4.f * 360.f * j) / td.plane_verts.size(), 0.7f, 0.7f)));
                    glDrawArrays(GL_LINE_LOOP, j * 4, 4);
                }
            }
        }
        if (draw_query_box) {
            glBindBuffer(GL_ARRAY_BUFFER, query_box_vbo);
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);

            shaders[0].use();
            shaders[0].uniform("proj_view", proj_view);
            shaders[0].uniform("fcolor", glm::vec3(1.f, 0.75f, 0.f));
            glDrawArrays(GL_LINES, 0, query_box_verts.size());
        }

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(window);
        query_changed = false;
        quality_changed = false;
    }
}

std::vector<glm::vec3> make_box_verts(const Box &b)
{
    return std::vector<glm::vec3>{b.lower,
                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),
                                  b.lower,

                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),
                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z),
                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),

                                  b.lower,
                                  glm::vec3(b.lower.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.lower.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.lower.y, b.upper.z),

                                  glm::vec3(b.upper.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.upper.x, b.upper.y, b.upper.z),

                                  glm::vec3(b.lower.x, b.upper.y, b.lower.z),
                                  glm::vec3(b.lower.x, b.upper.y, b.upper.z)};
}

std::vector<glm::vec3> make_plane_verts(const Plane &p)
{
    glm::vec3 dir = glm::vec3(-1.f);
    if (p.half_vectors.x == 0.f) {
        dir = glm::vec3(0.f, 1.f, -1.f);
    } else if (p.half_vectors.y == 0.f) {
        dir = glm::vec3(1.f, 0.f, -1.f);
    } else {
        dir = glm::vec3(1.f, -1.f, 0.f);
    }

    return std::vector<glm::vec3>{p.origin - p.half_vectors,
                                  p.origin + dir * p.half_vectors,
                                  p.origin + p.half_vectors,
                                  p.origin - dir * p.half_vectors};
}
