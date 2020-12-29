add_library(util
    util.cpp
    byte_cursor.cpp
    data_type.cpp
    frustum.cpp
    file_mapping.cpp
    plane.cpp
    box.cpp
    profiling.cpp
    tinyxml2.cpp)

set_target_properties(util PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(util PUBLIC
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)

target_link_libraries(util PUBLIC TBB::tbb Threads::Threads)

target_compile_options(util PUBLIC
    $<$<CONFIG:Debug>:-DDEBUG=1>)

if (NOT ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
    target_compile_options(util PUBLIC
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
        $<$<CXX_COMPILER_ID:GNU,AppleClang,Clang,Intel>:-march=core-avx2>)
endif()

# TODO: Work around CMake bullshit that forces this to be exported
install(TARGETS util EXPORT libbatConfig
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
