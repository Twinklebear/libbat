cmake_minimum_required(VERSION 3.15)
project(libbat)

if (NOT WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
endif()

add_definitions(-DNOMINMAX -DGLM_ENABLE_EXPERIMENTAL)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

option(BUILD_VIEWER "Build the viewer app" ON)
option(BUILD_TOOLS "Build the converter tools" ON)
option(BUILD_SERVER "Build the BAT webserver" ON)
if (NOT WIN32)
    option(BUILD_PARALLEL_IO_LIBRARY "Build the parallel IO library and test app" ON)
else()
    set(BUILD_PARALLEL_IO_LIBRARY OFF)
endif()
option(BUILD_INSPECTOR "Build the inspector and query test app" ON)

find_package(Threads REQUIRED)
find_package(glm REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(ext)
add_subdirectory(util)
add_subdirectory(bat)

if (BUILD_TOOLS)
    add_subdirectory(tools)
endif()

if (BUILD_VIEWER)
    add_subdirectory(viewer)
endif()

if (BUILD_PARALLEL_IO_LIBRARY)
    add_subdirectory(adaptive_io_test)
    add_subdirectory(parallel_io)
endif()

if (BUILD_INSPECTOR)
    add_subdirectory(inspector)
endif()

if (BUILD_SERVER)
    add_subdirectory(web/server)
endif()

