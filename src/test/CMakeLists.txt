add_executable(ode_test ode_test.cpp)
target_link_libraries(ode_test GTest::GTest GTest::Main)
add_test(NAME ode_test COMMAND ode_test)