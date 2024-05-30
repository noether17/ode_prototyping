include_directories(${CMAKE_SOURCE_DIR}/src/RKEmbedded)

add_executable(RKF45Test RKF45Test.cpp)
target_link_libraries(RKF45Test
  GTest::GTest
  GTest::Main
  RKEmbedded
  RKF45
  VectorState
)
add_test(NAME RKF45Test COMMAND RKF45Test)

add_executable(DOPRI5Test DOPRI5Test.cpp)
target_link_libraries(DOPRI5Test
  GTest::GTest
  GTest::Main
  RKEmbedded
  DOPRI5
  VectorState
)
add_test(NAME DOPRI5Test COMMAND DOPRI5Test)

add_executable(RKF78Test RKF78Test.cpp)
target_link_libraries(RKF78Test
  GTest::GTest
  GTest::Main
  RKEmbedded
  RKF78
  VectorState
)
add_test(NAME RKF78Test COMMAND RKF78Test)

add_executable(DVERKTest DVERKTest.cpp)
target_link_libraries(DVERKTest
  GTest::GTest
  GTest::Main
  RKEmbedded
  DVERK
  VectorState
)
add_test(NAME DVERKTest COMMAND DVERKTest)
