add_library(CudaUtils INTERFACE CudaUtils.cuh)

add_library(CudaState INTERFACE CudaState.cuh)

add_library(CUDAExpODE INTERFACE CUDAExpODE.cuh)

add_library(HostUtils INTERFACE HostUtils.hpp)

add_library(RKEmbeddedCuda INTERFACE RKEmbeddedCuda.cuh)

# Butcher Tableau libraries
add_library(BTHE21 INTERFACE BTHE21.hpp)
add_library(BTRKF45 INTERFACE BTRKF45.hpp)
add_library(BTDOPRI5 INTERFACE BTDOPRI5.hpp)
add_library(BTDVERK INTERFACE BTDVERK.hpp)
add_library(BTRKF78 INTERFACE BTRKF78.hpp)

add_library(CUDANBodyODE INTERFACE CUDANBodyODE.cuh)
target_link_libraries(CUDANBodyODE INTERFACE RKEmbeddedCuda)

add_library(RawCudaOutput INTERFACE RawCudaOutput.cuh)

add_executable(n_body_cuda n_body.cu)
target_link_libraries(n_body_cuda INTERFACE RKEmbeddedCuda BTRKF78)

add_subdirectory(test)
