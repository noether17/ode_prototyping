include_directories(
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
  ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)
add_library(ParallelThreadPool INTERFACE ParallelThreadPool.hpp)
add_library(RKEmbeddedParallel INTERFACE RKEmbeddedParallel.hpp)
add_library(CudaExecutor INTERFACE CudaExecutor.cuh)
add_library(CudaState INTERFACE CudaState.cuh)
add_executable(parallel_n_body n_body.cpp)
#add_executable(cuda_n_body cuda_n_body.cu)

add_subdirectory(test)
