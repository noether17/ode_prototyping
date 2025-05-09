cmake_minimum_required(VERSION 3.29.5)
project(ODEPrototyping)
include(CheckLanguage)

set_property(GLOBAL PROPERTY RULE_MESSAGES OFF)

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  message(STATUS "CUDA enabled")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
  set(CMAKE_CUDA_ARCHITECTURES native)

  # prevent use of rsp files, which are incompatible with clangd
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES 0)
  set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS 0)

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -Xcompiler -Wall \
    -Xcompiler -Werror \
    --expt-relaxed-constexpr \
    --extended-lambda")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror")

enable_testing()
find_package(GTest 1.14.0 REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

find_package(benchmark 1.8.2 REQUIRED)

if (CMAKE_COMPILER_IS_GNUCXX AND COVERAGE)
  message(STATUS "Code coverage enabled")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  add_custom_target(coverage
    COMMAND lcov --directory . --capture --output-file coverage.info
    COMMAND lcov --remove coverage.info '/usr/*' --output-file coverage.info
    COMMAND lcov --list coverage.info
    COMMAND genhtml -o coverage_report coverage.info
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating code coverage report"
  )
endif()

add_subdirectory(src)
