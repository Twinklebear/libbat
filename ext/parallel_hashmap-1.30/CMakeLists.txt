option(BAT_STORE_NODE_RANGES "Store per-node min/max attribute ranges" OFF)

add_library(parallel_hashmap INTERFACE)

target_include_directories(parallel_hashmap INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

# TODO: Work around CMake bullshit that forces this to be exported
install(TARGETS parallel_hashmap EXPORT libbatConfig
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
