cmake_minimum_required(VERSION 2.8.9)
project(csv_alexnet)
set (CMAKE_CXX_STANDARD 11)

add_executable(csv_alexnet readCSV.cpp)
