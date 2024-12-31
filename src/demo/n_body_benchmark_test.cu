#include <array>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "ParticlesInBox.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"

void output_to_file(std::string const& filename, auto& output) {
  auto output_file = std::ofstream{filename};
  for (std::size_t i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (std::size_t j = 0; j < output.states[i].size(); ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }
}

int main() {
  constexpr auto N = 64;
  auto scenario = ParticlesInBox<N, CudaState, double>{};
  constexpr auto n_var = scenario.n_var;

  auto cuda_exe = CudaExecutor{};
  auto integrator =
      RKEmbeddedParallel<CudaState, double, n_var, BTRKF78,
                         NBodyODE<double, n_var>,
                         RawOutput<HeapState<double, n_var>>, CudaExecutor>{};
  auto output = RawOutput<HeapState<double, n_var>>{};

  auto t0 = 0.0;
  auto tf = scenario.tf;

  integrator.integrate(scenario.initial_state, t0, tf, scenario.tolerance_array,
                       scenario.tolerance_array, NBodyODE<double, n_var>{},
                       output, cuda_exe);

  auto filename = std::string{"n_body_benchmark_test_cuda_RKF78_"} +
                  std::to_string(scenario.tolerance_value) + ".txt";
  output_to_file(filename, output);

  return 0;
}
