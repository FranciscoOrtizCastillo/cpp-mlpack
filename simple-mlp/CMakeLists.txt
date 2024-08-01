cmake_minimum_required(VERSION 3.15)

project(mlp_mlpack)

set(CMAKE_CXX_STANDARD 17)

#add_executable(${PROJECT_NAME}  src/mlp_mlpack.cpp)
#add_executable(${PROJECT_NAME}  src/ffn_example.cpp)
add_executable(${PROJECT_NAME}  src/rnn_example.cpp)

find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")
target_include_directories(${PROJECT_NAME}  PRIVATE ${MLPACK_INCLUDE_DIRS})

find_package(Armadillo REQUIRED)
target_include_directories(${PROJECT_NAME}  PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}  PRIVATE ${ARMADILLO_LIBRARIES})
