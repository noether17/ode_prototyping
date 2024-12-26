add_library(CudaErrorCheck INTERFACE CudaErrorCheck.cuh)
target_include_directories(CudaErrorCheck INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
