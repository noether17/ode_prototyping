add_library(ExponentialODE INTERFACE ExponentialODE.hpp)
target_include_directories(ExponentialODE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(NBodyODE INTERFACE NBodyODE.hpp)
target_link_libraries(NBodyODE INTERFACE AtomicUtil ParallelExecutor)
target_include_directories(NBodyODE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(VanDerPolODE INTERFACE VanDerPolODE.hpp)
target_include_directories(VanDerPolODE INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
