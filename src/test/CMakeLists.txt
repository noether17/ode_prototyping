include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(ode_test ode_test.cpp)
target_link_libraries(ode_test GTest::GTest GTest::Main ode)
add_test(NAME ode_test COMMAND ode_test)

add_executable(VanDerPolTest VanDerPolTest.cpp)
target_link_libraries(VanDerPolTest
    GTest::GTest
    GTest::Main
    ODEIntegrator
    Output
    StepperBase
    StepperDopr5
)
add_test(NAME VanDerPolTest COMMAND VanDerPolTest)

add_executable(SimpleOscillatorTest SimpleOscillatorTest.cpp)
target_link_libraries(SimpleOscillatorTest
    GTest::GTest
    GTest::Main
    ODEIntegrator
    Output
    StepperBase
    StepperDopr5
)
add_test(NAME SimpleOscillatorTest COMMAND SimpleOscillatorTest)