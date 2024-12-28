#Introduction
This repository is for testing parallel ODE integrators. Currently, parallelism is achieved via a thread pool and via CUDA (a single-threaded implementation is included for completeness). Integration is performed using one of several embedded Runge-Kutta methods, each method being differentiated by its Butcher tableau.

#Building
```
mkdir build
cd build
cmake ..
cmake --build .
```

#Testing
After building the project, use CTest to run the unit tests.
`ctest`
