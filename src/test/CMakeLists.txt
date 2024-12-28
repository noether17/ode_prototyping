add_library(ExponentialTest INTERFACE ExponentialTest.hpp)
target_include_directories(ExponentialTest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ExponentialTest INTERFACE
  ExponentialODE
  HeapState
  RawOutput
)

add_library(NBodyTest INTERFACE NBodyTest.hpp)
target_include_directories(NBodyTest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(NBodyTest INTERFACE
  HeapState
  NBodyODE
  RawOutput
)

add_library(VanDerPolTest INTERFACE VanDerPolTest.hpp)
target_include_directories(VanDerPolTest INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(VanDerPolTest INTERFACE
  HeapState
  RawOutput
  VanDerPolODE
)

add_executable(SingleThreadedRKEmbeddedTest SingleThreadedRKEmbeddedTest.cpp)
target_link_libraries(SingleThreadedRKEmbeddedTest
  GTest::GTest
  GTest::Main
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  ExponentialTest
  NBodyTest
  VanDerPolTest
  RKEmbeddedParallel
  SingleThreadedExecutor
)
add_test(NAME SingleThreadedRKEmbeddedTest COMMAND SingleThreadedRKEmbeddedTest)

add_executable(ThreadPoolRKEmbeddedTest ThreadPoolRKEmbeddedTest.cpp)
target_link_libraries(ThreadPoolRKEmbeddedTest
  GTest::GTest
  GTest::Main
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  ExponentialTest
  NBodyTest
  VanDerPolTest
  RKEmbeddedParallel
  ThreadPoolExecutor
)
add_test(NAME ThreadPoolRKEmbeddedTest COMMAND ThreadPoolRKEmbeddedTest)

add_executable(CudaStateTest CudaStateTest.cu)
target_link_libraries(CudaStateTest
  GTest::GTest
  GTest::Main
  CudaState
  CudaExecutor
  StateUtils
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
  BTHE21
  BTRKF45
  BTDOPRI5
  BTDVERK
  BTRKF78
  CudaState
  CudaExecutor
  ExponentialTest
  NBodyTest
  VanDerPolTest
  RKEmbeddedParallel
)
add_test(NAME CudaRKEmbeddedTest COMMAND CudaRKEmbeddedTest)
