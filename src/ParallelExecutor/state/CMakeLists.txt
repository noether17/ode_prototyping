add_library(CudaState INTERFACE CudaState.cuh)
target_include_directories(CudaState INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(HeapState INTERFACE HeapState.hpp)
target_include_directories(HeapState INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(StateUtils INTERFACE StateUtils.hpp)
target_include_directories(StateUtils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(StateUtils INTERFACE ParallelExecutor)
