include_directories(
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
  ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)
add_library(ParallelThreadPool INTERFACE ParallelThreadPool.hpp)
add_executable(parallel_n_body n_body.cpp)
