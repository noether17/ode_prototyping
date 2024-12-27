add_library(CudaState INTERFACE CudaState.cuh)
target_include_directories(CudaState INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(HeapState INTERFACE HeapState.hpp)
target_include_directories(HeapState INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
