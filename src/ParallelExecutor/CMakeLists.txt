include_directories(
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
  ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)
add_library(RKEmbeddedParallel INTERFACE RKEmbeddedParallel.hpp)
add_library(CudaState INTERFACE CudaState.cuh)

add_executable(parallel_n_body n_body.cpp)
target_link_libraries(parallel_n_body ParallelThreadPool)

add_executable(cuda_n_body cuda_n_body.cu)
target_link_libraries(cuda_n_body CudaExecutor)

add_subdirectory(cuda_util)
add_subdirectory(ode)
add_subdirectory(par_exec)
add_subdirectory(util)
add_subdirectory(test)
