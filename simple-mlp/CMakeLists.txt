cmake_minimum_required(VERSION 3.15)

project(mlp_mlpack)

set(CMAKE_CXX_STANDARD 17)

#add_executable(mlp_mlpack src/mlp_mlpack.cpp)
#add_executable(mlp_mlpack src/ffn_example.cpp)
add_executable(mlp_mlpack src/rnn_example.cpp)

find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")
target_include_directories(mlp_mlpack PRIVATE ${MLPACK_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
target_include_directories(mlp_mlpack PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(mlp_mlpack PRIVATE ${ARMADILLO_LIBRARIES})
