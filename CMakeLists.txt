cmake_minimum_required(VERSION 3.22.1)
project(ODEPrototyping)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
find_package(GTest 1.14.0 REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_subdirectory(src)