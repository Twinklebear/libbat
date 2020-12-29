option(BAT_STORE_NODE_RANGES "Store per-node min/max attribute ranges" OFF)

add_library(bat
    attribute.cpp
    radix_tree_node.cpp
    kd_node.cpp
    ba_treelet.cpp
    ba_tree.cpp
    lba_tree_builder.cpp
    bat_file.cpp
    binary_particle_file.cpp
    query_stats.cpp
    aggregation_tree_builder.cpp
    aggregation_tree.cpp
    aggregation_kd_node.cpp
    pbat_file.cpp
    bat_handle.cpp)

if (BAT_STORE_NODE_RANGES)
    target_compile_definitions(bat
        PUBLIC
        -DBAT_STORE_NODE_RANGES)
endif()

set_target_properties(bat PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(bat PUBLIC
    $<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(bat PUBLIC
    TBB::tbb
    util
    parallel_hashmap)

install(TARGETS bat EXPORT libbatConfig
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
