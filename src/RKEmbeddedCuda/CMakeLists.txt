add_library(CudaUtils INTERFACE CudaUtils.cuh)

add_library(CudaState INTERFACE CudaState.cuh)

add_library(CudaExpOde INTERFACE CudaExpOde.cuh)

add_library(HostUtils INTERFACE HostUtils.hpp)

add_library(RKEmbeddedCuda INTERFACE RKEmbeddedCuda.cuh)

add_library(CudaNBodyOde INTERFACE CudaNBodyOde.cuh)
target_link_libraries(CudaNBodyOde INTERFACE RKEmbeddedCuda)

add_library(RawCudaOutput INTERFACE RawCudaOutput.cuh)

add_executable(n_body_cuda n_body.cu)
target_link_libraries(n_body_cuda INTERFACE RKEmbeddedCuda BTRKF78 CudaNBodyOde)
target_include_directories(n_body_cuda
  PUBLIC ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)

add_subdirectory(test)
