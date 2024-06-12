include_directories(${CMAKE_SOURCE_DIR}/src/RKEmbeddedCuda)

add_executable(CudaUtilsTest CudaUtilsTest.cu)
target_link_libraries(CudaUtilsTest
  GTest::GTest
  GTest::Main
  CudaUtils
)
add_test(NAME CudaUtilsTest COMMAND CudaUtilsTest)
