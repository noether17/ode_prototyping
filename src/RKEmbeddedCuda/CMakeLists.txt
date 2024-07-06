add_library(CudaUtils INTERFACE CudaUtils.cuh)

add_library(CudaState INTERFACE CudaState.cuh)

add_library(RKEmbeddedCuda INTERFACE RKEmbeddedCuda.cuh)
target_link_libraries(RKEmbeddedCuda INTERFACE CudaState)

add_executable(n_body_cuda n_body.cu)
target_link_libraries(n_body_cuda INTERFACE RKEmbeddedCuda)

add_subdirectory(test)
