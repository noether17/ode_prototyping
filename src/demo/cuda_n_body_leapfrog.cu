#include <array>
#include <fstream>
#include <vector>

#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "LeapfrogParallel.hpp"
#include "NBodyODE.hpp"
#include "RawOutput.hpp"

int main() {
  auto x0_data =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto x0 = CudaState{x0_data};
  constexpr auto n_var = x0_data.size();

  auto cuda_exe = CudaExecutor{};
  auto integrator = ParallelLeapfrogIntegrator{};
  auto output = RawOutput<HeapState<double, n_var>>{};

  auto t0 = 0.0;
  auto tf = 6.3;
  auto n_steps = 1.0e4;
  auto dt = (tf - t0) / n_steps;

  integrator.integrate(x0, t0, tf, dt, NBodyODE<double, n_var>{}, output,
                       cuda_exe);

  auto output_file = std::ofstream{"leapfrog_cuda_n_body_output.txt"};
  for (std::size_t i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (std::size_t j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  return 0;
}
