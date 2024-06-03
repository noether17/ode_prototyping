add_library(ODEState INTERFACE ODEState.hpp)

add_library(RKEmbedded INTERFACE RKEmbedded.hpp)

add_library(HE21 INTERFACE HE21.hpp)
add_library(RKF45 INTERFACE RKF45.hpp)
add_library(DOPRI5 INTERFACE DOPRI5.hpp)
add_library(RKF78 INTERFACE RKF78.hpp)
add_library(DVERK INTERFACE DVERK.hpp)

add_library(VectorState INTERFACE VectorState.hpp)
target_link_libraries(VectorState INTERFACE ODEState)

add_executable(main main.cpp)
target_link_libraries(main RKEmbedded RKF45 DOPRI5 RKF78 DVERK VectorState)

add_subdirectory(test)
