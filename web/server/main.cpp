#include <array>
#include <chrono>
#include <string>
#include <thread>
#include <vector>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#ifdef _WIN32
#include <Windows.h>
#endif
#include "bat_handle.h"
#include "fixed_array.h"
#include "httplib.h"
#include "json.hpp"
#include "util.h"

const size_t SEND_CHUNK_SIZE = 8192;

int main(int argc, char **argv)
{
    using namespace httplib;
    using namespace std::chrono;
    using json = nlohmann::json;

    if (argc == 1) {
        std::cout << "Usage: " << argv[0] << " <bat/pbat files\n";
        return 1;
    }

    std::vector<std::string> bat_files(argv + 1, argv + argc);
    std::vector<std::string> dataset_names;
    for (auto &f : bat_files) {
        f = canonicalize_path(f);

        std::string name = get_file_basename(f);
        name = name.substr(0, name.find('.'));
        dataset_names.push_back(name);
    }

    httplib::Server server;

    // /datasets endpoint returns a JSON array listing the data sets and
    // their metadata
    server.Get("/datasets", [&](const Request &req, Response &response) {
        response.set_header("Access-Control-Allow-Origin", "*");
        response.status = 200;

        json dataset_info;
        for (size_t i = 0; i < bat_files.size(); ++i) {
            auto bat_handle = BATHandle(bat_files[i]);
            dataset_info[i]["name"] = dataset_names[i];
            dataset_info[i]["num_points"] = bat_handle.num_points();

            for (size_t j = 0; j < bat_handle.attributes.size(); ++j) {
                auto &desc = bat_handle.attributes[j];
                auto &attr_json = dataset_info[i]["attributes"][j];
                attr_json["name"] = desc.name;
                attr_json["type"] = print_data_type(desc.data_type);
                attr_json["range"][0] = desc.range.x;
                attr_json["range"][1] = desc.range.y;
            }
        }

        const std::string info = dataset_info.dump();
        response.set_content(info, "application/json");
    });

    server.Post(
        R"(/dataset/([^/]+))",
        [&](const Request &req, Response &response, const ContentReader &content_reader) {
            response.set_header("Access-Control-Allow-Origin", "*");
            std::cout << "Request for data set " << req.matches[1] << "\n";

            auto fnd = std::find(dataset_names.begin(), dataset_names.end(), req.matches[1]);
            if (fnd == dataset_names.end()) {
                response.status = 404;
                std::cout << "Dataset " << req.matches[1] << " does not exist\n";
                response.set_content("Dataset does not exist", "text/plain");
                return;
            }

            response.status = 200;
            response.set_header("Content-Type", "arraybuffer");

            json query_params;
            {
                std::string post_body;
                content_reader([&](const char *data, size_t len) {
                    post_body.append(data, len);
                    return true;
                });
                query_params = json::parse(post_body);
            }
            std::cout << "query params: " << query_params.dump(4) << "\n";

            const size_t bat_id = std::distance(dataset_names.begin(), fnd);
            auto bat_handle = BATHandle(bat_files[bat_id]);

            float quality = 0.1f;
            if (query_params.contains("quality")) {
                quality = query_params["quality"].get<float>();
            }
      
            float prev_quality = 0.0f;
            if (query_params.contains("prev_quality")) {
                prev_quality = query_params["prev_quality"].get<float>();
            }

            size_t attrib_id = 0;
            glm::vec2 query_range(std::numeric_limits<float>::infinity());
            if (query_params.contains("attribute")) {
                const std::string attrib_name = query_params["attribute"].get<std::string>();
                auto fnd = std::find_if(bat_handle.attributes.begin(),
                                        bat_handle.attributes.end(),
                                        [&](const AttributeDescription &desc) {
                                            return desc.name == attrib_name;
                                        });
                if (fnd != bat_handle.attributes.end()) {
                    attrib_id = std::distance(bat_handle.attributes.begin(), fnd);
                    std::cout << "Querying attrib " << attrib_name << " id: " << attrib_id
                              << "\n";
                    if (query_params.contains("range_min") &&
                        query_params.contains("range_max")) {
                        query_range = glm::vec2(query_params["range_min"].get<float>(),
                                                query_params["range_max"].get<float>());
                    }
                }
            }
            if (!std::isfinite(query_range.x) || !std::isfinite(query_range.y)) {
                query_range = bat_handle.attributes[attrib_id].range;
            }

            std::vector<AttributeQuery> attrib_queries = {
                AttributeQuery(bat_handle.attributes[attrib_id].name, query_range)};

            std::vector<glm::vec3> query_points;
            std::vector<float> query_attribute;

            auto start = steady_clock::now();
            const auto query_stats =
                bat_handle.query_box_progressive(bat_handle.bounds(),
                                     &attrib_queries,
                                     prev_quality,
                                     quality,
                                     [&](const size_t id,
                                         const glm::vec3 &pos,
                                         const std::vector<Attribute> &attribs) {
                                         query_points.push_back(pos);

                                         const Attribute &attr = attribs[attrib_id];
                                         query_attribute.push_back(attr.as_float(id));
                                     });
            auto end = steady_clock::now();
            auto dur = duration_cast<milliseconds>(end - start).count();
            std::cout << "Query took " << dur << "ms\n" << query_stats << "\n" << std::flush;

            const size_t response_size = sizeof(uint32_t) +
                                         query_points.size() * sizeof(glm::vec3) +
                                         query_attribute.size() * sizeof(float);

            auto test_buf = std::make_shared<FixedArray<char>>(response_size);
            {
                const size_t points_buf_size = query_points.size() * sizeof(glm::vec3);
                const size_t attrib_buf_size = query_attribute.size() * sizeof(float);
                const uint32_t num_particles = query_points.size();
                std::memcpy(test_buf->data(), &num_particles, sizeof(num_particles));

                size_t offs = sizeof(num_particles);
                std::memcpy(test_buf->data() + offs, query_points.data(), points_buf_size);

                offs += points_buf_size;
                std::memcpy(test_buf->data() + offs, query_attribute.data(), attrib_buf_size);
            }

            response.set_content_provider(
                test_buf->size(), [test_buf](size_t offset, size_t len, DataSink &sink) {
                    sink.write(test_buf->data(), test_buf->size());
                });
        });

    server.listen("localhost", 1234);

    return 0;
}
