cmake_minimum_required(VERSION 3.16)

project(ATMMG LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ATMMG_BUILD_BENCHMARKS "Build local benchmark programs" ON)

add_library(ATMMG INTERFACE)
target_include_directories(
    ATMMG
    INTERFACE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(ATMMG INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(ATMMG INTERFACE /W4 /EHsc /arch:AVX2)
else()
    target_compile_options(ATMMG INTERFACE -Wall -Wextra -march=native -fexceptions)
endif()

if(ATMMG_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
