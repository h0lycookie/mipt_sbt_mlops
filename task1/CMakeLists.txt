cmake_minimum_required(VERSION 3.12)
project(gauss_jordan_cpp LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 REQUIRED)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

if(NOT pybind11_FOUND)
    FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(gauss_jordan_cpp
    bindings.cpp
    gauss_jordan.cpp
)

target_compile_options(gauss_jordan_cpp PRIVATE -O3)

install(TARGETS gauss_jordan_cpp
    LIBRARY DESTINATION .
)