add_executable(xyz_converter xyz_converter.cpp)
set_target_properties(xyz_converter PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(xyz_converter PUBLIC bat)

add_executable(cosmic_web_converter cosmic_web_converter.cpp)
set_target_properties(cosmic_web_converter PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(cosmic_web_converter PUBLIC bat)

add_executable(uintah_converter uintah_converter.cpp)
set_target_properties(uintah_converter PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(uintah_converter PUBLIC bat)

add_executable(binary_particle_converter binary_particle_converter.cpp)
set_target_properties(binary_particle_converter PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(binary_particle_converter PUBLIC bat)

add_executable(json_particle_converter json_particle_converter.cpp)
set_target_properties(json_particle_converter PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(json_particle_converter PUBLIC bat)

add_executable(bitmap_tester bitmap_tester.cpp)
set_target_properties(bitmap_tester PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON)
target_link_libraries(bitmap_tester PUBLIC bat)

find_package(VTK)
if (${VTK_FOUND})
    add_executable(vtu_converter vtu_converter.cpp)
    set_target_properties(vtu_converter PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON)
    target_link_libraries(vtu_converter PUBLIC bat ${VTK_LIBRARIES})
    target_include_directories(vtu_converter PUBLIC ${VTK_INCLUDE_DIRS})
    target_compile_definitions(vtu_converter PUBLIC ${VTK_DEFINITIONS})
endif()

find_package(LASTools)
if (${LASTOOLS_FOUND})
	add_executable(lidar_converter lidar_converter.cpp)
    target_link_libraries(lidar_converter PUBLIC
		bat ${LASTOOLS_LIBRARIES})
	set_target_properties(lidar_converter PROPERTIES
		CXX_STANDARD 14
		CXX_STANDARD_REQUIRED ON)
    target_include_directories(lidar_converter PUBLIC
        ${LASTOOLS_INCLUDE_DIRS})
endif()

