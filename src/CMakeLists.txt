add_library(ode INTERFACE ode.hpp)
add_library(ODEIntegrator INTERFACE ODEIntegrator.hpp)
add_library(Output INTERFACE Output.hpp)
add_library(StepperBase INTERFACE StepperBase.hpp)
add_library(StepperDopr5 INTERFACE StepperDopr5.hpp)

add_subdirectory(test)