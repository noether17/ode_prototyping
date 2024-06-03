#add_library(ode INTERFACE ode.hpp)
#
#add_library(ODEIntegrator INTERFACE ODEIntegrator.hpp)
#
#add_library(Output INTERFACE Output.hpp)
#
#add_library(StepperBase INTERFACE StepperBase.hpp)
#
#add_library(StepperDopr5 INTERFACE StepperDopr5.hpp)
#target_link_libraries(StepperDopr5 INTERFACE StepperBase)
#
#add_library(StepperDopr853 INTERFACE StepperDopr853.hpp Dopr853_constants.hpp)
#target_link_libraries(StepperDopr853 INTERFACE StepperBase)

add_subdirectory(RKEmbedded)
#add_subdirectory(test)
