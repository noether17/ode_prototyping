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
  CUDAExpODE
  CUDANBodyODE
  HostUtils
  RKEmbeddedCuda
  RawCudaOutput
)
add_test(NAME RKEmbeddedCudaTest COMMAND RKEmbeddedCudaTest)
