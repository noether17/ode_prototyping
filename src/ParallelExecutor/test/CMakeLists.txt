include_directories(${CMAKE_SOURCE_DIR}/src/ButcherTableaus
  ${CMAKE_SOURCE_DIR}/src/ParallelExecutor
  ${CMAKE_SOURCE_DIR}/src/RKEmbedded
)

add_executable(SingleThreadedRKEmbeddedTest SingleThreadedRKEmbeddedTest.cpp)
target_link_libraries(SingleThreadedRKEmbeddedTest
  GTest::GTest
  GTest::Main
  ExponentialODE
  NBodyODE
  VanDerPolODE
  RKEmbeddedParallel
)
add_test(NAME SingleThreadedRKEmbeddedTest COMMAND SingleThreadedRKEmbeddedTest)

add_executable(ParallelRKEmbeddedTest ParallelRKEmbeddedTest.cpp)
target_link_libraries(ParallelRKEmbeddedTest
  GTest::GTest
  GTest::Main
  ExponentialODE
  NBodyODE
  VanDerPolODE
  RKEmbeddedParallel
)
add_test(NAME ParallelRKEmbeddedTest COMMAND ParallelRKEmbeddedTest)

add_executable(CudaStateTest CudaStateTest.cu)
target_link_libraries(CudaStateTest
  GTest::GTest
  GTest::Main
  CudaState
)
add_test(NAME CudaStateTest COMMAND CudaStateTest)

add_executable(CudaExecutorTest CudaExecutorTest.cu)
target_link_libraries(CudaExecutorTest
  GTest::GTest
  GTest::Main
  CudaState
  CudaExecutor
)
add_test(NAME CudaExecutorTest COMMAND CudaExecutorTest)

add_executable(CudaRKEmbeddedTest CudaRKEmbeddedTest.cu)
target_link_libraries(CudaRKEmbeddedTest
  GTest::GTest
  GTest::Main
  CudaState
  CudaExecutor
  ExponentialODE
  NBodyODE
  VanDerPolODE
  RKEmbeddedParallel
)
add_test(NAME CudaRKEmbeddedTest COMMAND CudaRKEmbeddedTest)
