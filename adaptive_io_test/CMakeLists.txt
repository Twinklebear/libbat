find_package(MPI REQUIRED)

add_executable(adaptive_io_test 
	adaptive_io_test.cpp
    generate_data.cpp)

set_target_properties(adaptive_io_test PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(adaptive_io_test PUBLIC
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(adaptive_io_test PUBLIC
    util bat bat_io TBB::tbb MPI::MPI_CXX)

add_executable(adaptive_read_test 
	adaptive_read_test.cpp)

set_target_properties(adaptive_read_test PROPERTIES
	CXX_STANDARD 14
	CXX_STANDARD_REQUIRED ON)

target_include_directories(adaptive_read_test PUBLIC
	$<BUILD_INTERFACE:${GLM_INCLUDE_DIRS}>)

target_link_libraries(adaptive_read_test PUBLIC
    util bat bat_io TBB::tbb MPI::MPI_CXX)

