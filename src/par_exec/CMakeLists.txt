add_library(KernelConcepts INTERFACE KernelConcepts.hpp)
target_include_directories(KernelConcepts INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(ParallelExecutor INTERFACE ParallelExecutor.hpp)
target_include_directories(ParallelExecutor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ParallelExecutor INTERFACE KernelConcepts)

add_library(CudaExecutor INTERFACE CudaExecutor.cuh)
target_include_directories(CudaExecutor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(CudaExecutor INTERFACE CudaErrorCheck KernelConcepts)

add_library(ThreadPoolExecutor INTERFACE ThreadPoolExecutor.hpp)
target_include_directories(ThreadPoolExecutor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ThreadPoolExecutor INTERFACE KernelConcepts)

add_library(SingleThreadedExecutor INTERFACE SingleThreadedExecutor.hpp)
target_include_directories(SingleThreadedExecutor INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SingleThreadedExecutor INTERFACE KernelConcepts)
