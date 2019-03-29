cmake_minimum_required(VERSION 2.8.9)
project(yaml_alexnet)
set (CMAKE_CXX_STANDARD 11)

add_executable(yaml_alexnet readYAML.cpp)
target_link_libraries(yaml_alexnet yaml-cpp)
