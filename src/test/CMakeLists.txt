include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(ode_test ode_test.cpp)
add_executable(nr_17_0_4_test nr_17_0_4_test.cpp)
target_link_libraries(ode_test GTest::GTest GTest::Main ode)
target_link_libraries(nr_17_0_4_test
    GTest::GTest
    GTest::Main
    ODEIntegrator
    Output
    StepperBase
    StepperDopr5
)
add_test(NAME ode_test COMMAND ode_test)
add_test(NAME nr_17_0_4_test COMMAND nr_17_0_4_test)