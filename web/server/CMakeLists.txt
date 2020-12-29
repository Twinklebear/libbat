add_executable(bat_server main.cpp)

set_target_properties(bat_server PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

if (WIN32)
    target_compile_definitions(bat_server PRIVATE
        -DWIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(bat_server PRIVATE
    bat
    util
    TBB::tbb)

