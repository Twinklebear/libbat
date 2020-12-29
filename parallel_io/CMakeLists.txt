find_package(MPI REQUIRED)

add_library(bat_io STATIC
    bat_write.cpp
    bat_read.cpp
    aggregator_assignment.cpp
    mpi_send_recv.cpp
    particle_data.cpp)

set_target_properties(bat_io PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)

target_include_directories(bat_io
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE
    $<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(bat_io
    PUBLIC
    TBB::tbb
    MPI::MPI_CXX
    PRIVATE 
    bat
    util)

install(TARGETS bat_io EXPORT libbatConfig
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)

install(FILES
    bat_io.h
    bat_write.h
    bat_read.h
    bat_data_type.h
    DESTINATION include)

install(EXPORT libbatConfig DESTINATION lib/cmake/libbat)

