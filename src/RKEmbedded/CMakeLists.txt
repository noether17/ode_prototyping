add_library(ODEState INTERFACE ODEState.hpp)

add_library(RKEmbedded INTERFACE RKEmbedded.hpp)

add_library(RawOutput INTERFACE RawOutput.hpp)

add_library(AllocatedState INTERFACE AllocatedState.hpp)
target_link_libraries(AllocatedState INTERFACE ODEState)

add_executable(main main.cpp)
target_include_directories(main PUBLIC ${CMAKE_SOURCE_DIR}/src/ButcherTableaus)
target_link_libraries(main
  RKEmbedded
  RawOutput
  BTRKF45
  BTDOPRI5
  BTRKF78
  BTDVERK
  AllocatedState
)

add_executable(n_body_cpu n_body.cpp)
target_include_directories(n_body_cpu PUBLIC ${CMAKE_SOURCE_DIR}/src/ButcherTableaus)
target_link_libraries(n_body_cpu
  RKEmbedded
  RawOutput
  BTRKF78
  AllocatedState
)

add_subdirectory(test)
