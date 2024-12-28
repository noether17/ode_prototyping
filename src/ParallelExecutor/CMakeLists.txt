include_directories(
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
  ${CMAKE_SOURCE_DIR}/src/ButcherTableaus
)

add_subdirectory(cuda_util)
add_subdirectory(demos)
add_subdirectory(integrator)
add_subdirectory(ode)
add_subdirectory(output)
add_subdirectory(par_exec)
add_subdirectory(state)
add_subdirectory(test)
add_subdirectory(util)
