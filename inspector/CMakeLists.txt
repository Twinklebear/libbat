add_executable(bat_inspector main.cpp)

set_target_properties(bat_inspector PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_link_libraries(bat_inspector PUBLIC
    bat
    util
    TBB::tbb)

