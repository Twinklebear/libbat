function(add_glsl_library)
    cmake_parse_arguments(PARSE_ARGV 1 GLSL "" "" "")

    set(GLSL_OUTPUTS "")
    foreach (shader ${GLSL_UNPARSED_ARGUMENTS})
        get_filename_component(FNAME ${shader} NAME)
        set(COPY_DST ${CMAKE_CURRENT_BINARY_DIR}/${FNAME})
        if (MSVC)
            # We have to copy to two places on windows, since the output dir
            # is dependent on the config
            add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FNAME}
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/${FNAME}
                    ${CMAKE_CURRENT_BINARY_DIR}/../$<CONFIG>/shaders/${FNAME}
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/${FNAME} ${COPY_DST}
                DEPENDS ${CMAKE_CURRENT_LIST_DIR}/${FNAME}
                COMMENT "Copying ${CMAKE_CURRENT_LIST_DIR}/${FNAME} to build output directory.")
        else()
            add_custom_command(OUTPUT ${COPY_DST}
                COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_LIST_DIR}/${FNAME} ${COPY_DST}
                DEPENDS ${CMAKE_CURRENT_LIST_DIR}/${FNAME}
                COMMENT "Copying ${CMAKE_CURRENT_LIST_DIR}/${FNAME} to ${COPY_DST}")
        endif()

        list(APPEND GLSL_OUTPUTS ${COPY_DST})
    endforeach()

    add_custom_target(${ARGV0} ALL DEPENDS ${GLSL_OUTPUTS})
endfunction()

file(GLOB SHADER_SRCS *.glsl)

add_glsl_library(glsl_shaders ${SHADER_SRCS})

