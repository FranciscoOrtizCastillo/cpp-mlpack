cmake_minimum_required(VERSION 3.15)

project(mlpack_simple CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(mlpack_simple cpp_quickstart_1.cpp)

find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")
target_include_directories(mlpack_simple PRIVATE ${MLPACK_INCLUDE_DIRS})