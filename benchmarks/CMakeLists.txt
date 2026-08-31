set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(ATMMG_fvecs_benchmark fvecs_benchmark.cpp)
target_link_libraries(ATMMG_fvecs_benchmark PRIVATE ATMMG)

find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND)
    add_executable(ATMMG_hdf5_benchmark hdf5_benchmark.cpp)
    target_link_libraries(
        ATMMG_hdf5_benchmark
        PRIVATE
            ATMMG
            ${HDF5_LIBRARIES}
    )
    target_include_directories(ATMMG_hdf5_benchmark PRIVATE ${HDF5_INCLUDE_DIRS})
    target_compile_definitions(ATMMG_hdf5_benchmark PRIVATE ${HDF5_DEFINITIONS})

    if(WIN32 AND HDF5_DIR)
        get_filename_component(HDF5_RUNTIME_DIR "${HDF5_DIR}/../bin" ABSOLUTE)
        file(GLOB ATMMG_HDF5_RUNTIME_DLLS
            "${HDF5_RUNTIME_DIR}/hdf5*.dll"
            "${HDF5_RUNTIME_DIR}/zlib*.dll"
            "${HDF5_RUNTIME_DIR}/zstd*.dll"
            "${HDF5_RUNTIME_DIR}/libzstd*.dll"
            "${HDF5_RUNTIME_DIR}/deflate*.dll"
            "${HDF5_RUNTIME_DIR}/*aec*.dll"
            "${HDF5_RUNTIME_DIR}/szip*.dll"
        )
        if(ATMMG_HDF5_RUNTIME_DLLS)
            add_custom_command(TARGET ATMMG_hdf5_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ATMMG_HDF5_RUNTIME_DLLS}
                    $<TARGET_FILE_DIR:ATMMG_hdf5_benchmark>
            )
        endif()
    endif()
else()
    message(STATUS "HDF5 C++ library not found; skipping ATMMG_hdf5_benchmark")
endif()
