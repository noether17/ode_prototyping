include_directories(
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
  ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)

add_library(CudaState INTERFACE CudaState.cuh)

add_executable(parallel_n_body n_body.cpp)
target_link_libraries(parallel_n_body ParallelThreadPool RKEmbeddedParallel)

add_executable(cuda_n_body cuda_n_body.cu)
target_link_libraries(cuda_n_body CudaExecutor RKEmbeddedParallel)

add_subdirectory(cuda_util)
add_subdirectory(integrator)
add_subdirectory(ode)
add_subdirectory(par_exec)
add_subdirectory(util)
add_subdirectory(test)
