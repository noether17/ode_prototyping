add_library(CudaUtils INTERFACE CudaUtils.cuh)

add_library(CudaState INTERFACE CudaState.cuh)

add_library(CUDAExpODE INTERFACE CUDAExpODE.cuh)

add_library(HostUtils INTERFACE HostUtils.hpp)

add_library(RKEmbeddedCuda INTERFACE RKEmbeddedCuda.cuh)

add_library(RawCudaOutput INTERFACE RawCudaOutput.cuh)

add_executable(n_body_cuda n_body.cu)
target_link_libraries(n_body_cuda INTERFACE RKEmbeddedCuda)

add_subdirectory(test)
