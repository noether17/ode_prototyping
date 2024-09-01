include_directories(${CMAKE_SOURCE_DIR}/src/RKEmbeddedCuda)

add_executable(CudaUtilsTest CudaUtilsTest.cu)
target_link_libraries(CudaUtilsTest
  GTest::GTest
  GTest::Main
  CudaUtils
)
add_test(NAME CudaUtilsTest COMMAND CudaUtilsTest)

add_executable(CudaStateTest CudaStateTest.cu)
target_link_libraries(CudaStateTest
  GTest::GTest
  GTest::Main
  CudaState
)
add_test(NAME CudaStateTest COMMAND CudaStateTest)

add_executable(RKEmbeddedCudaTest RKEmbeddedCudaTest.cu)
target_link_libraries(RKEmbeddedCudaTest
  GTest::GTest
  GTest::Main
  BTHE21
  CUDAExpODE
  HostUtils
  RKEmbeddedCuda
)
add_test(NAME RKEmbeddedCudaTest COMMAND RKEmbeddedCudaTest)

add_executable(CudaRKEmbeddedConsistencyTest CudaRKEmbeddedConsistencyTest.cu)
target_link_libraries(CudaRKEmbeddedConsistencyTest
  GTest::GTest
  GTest::Main
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  CUDAExpODE
  RKEmbeddedCuda
  RawCudaOutput
)
add_test(NAME CudaRKEmbeddedConsistencyTest
  COMMAND CudaRKEmbeddedConsistencyTest
)

add_executable(CudaRKEmbeddedCPUComparisonTest
  CudaRKEmbeddedCPUComparisonTest.cu
)
target_link_libraries(CudaRKEmbeddedCPUComparisonTest
  GTest::GTest
  GTest::Main
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  CUDAExpODE
  RKEmbeddedCuda
  RawCudaOutput
)
add_test(NAME CudaRKEmbeddedCPUComparisonTest
  COMMAND CudaRKEmbeddedCPUComparisonTest
)

add_executable(CudaNBodyTest CudaNBodyTest.cu)
target_link_libraries(CudaNBodyTest
  GTest::GTest
  GTest::Main
  BTRKF78
  CUDANBodyODE
  RKEmbeddedCuda
  RawCudaOutput
)
add_test(NAME CudaNBodyTest COMMAND CudaNBodyTest)
