include_directories(${CMAKE_SOURCE_DIR}/src/ButcherTableaus
  ${CMAKE_SOURCE_DIR}/src/ParallelExecutor
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
)

add_executable(SingleThreadedRKEmbeddedTest SingleThreadedRKEmbeddedTest.cpp)
target_link_libraries(SingleThreadedRKEmbeddedTest
  GTest::GTest
  GTest::Main
  RKEmbeddedParallel
)
add_test(NAME SingleThreadedRKEmbeddedTest COMMAND SingleThreadedRKEmbeddedTest)

add_executable(ParallelRKEmbeddedTest ParallelRKEmbeddedTest.cpp)
target_link_libraries(ParallelRKEmbeddedTest
  GTest::GTest
  GTest::Main
  RKEmbeddedParallel
)
add_test(NAME ParallelRKEmbeddedTest COMMAND ParallelRKEmbeddedTest)
