cmake_minimum_required(VERSION 3.27.2)

if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file"
    )
endif()

project(taosim CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

file(WRITE "${CMAKE_BINARY_DIR}/.gitignore" "*")

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(OVERRIDE_NEW_DELETE "Override the C++ canonical allocation interface with mimalloc alternatives" ON)
option(BUILD_TESTS "Build test suite" OFF)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_ASAN "Enable address sanitizer" OFF)

if(ENABLE_ASAN)
    if(NOT UNIX)
        message(FATAL_ERROR "ASan requires Unix")
    endif()
    string(APPEND CMAKE_CXX_FLAGS " -fsanitize=address")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    #string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wpedantic")
endif()

set(SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp" CACHE STRING "C++ source directory")

#-------------------------------------------------------------------------

find_package(Boost REQUIRED)
find_package(CLI11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(bdl REQUIRED)
find_package(bsl REQUIRED)
find_package(date REQUIRED)
find_package(fmt REQUIRED)
find_package(mimalloc REQUIRED)
find_package(pugixml REQUIRED)
find_package(pybind11 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

#-------------------------------------------------------------------------

add_subdirectory("${SRC_DIR}")

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(test/cpp-tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

add_custom_target(check-compiler
    COMMAND ${CMAKE_COMMAND} -E echo
        "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
        "at ${CMAKE_CXX_COMPILER}"
)

#-------------------------------------------------------------------------
