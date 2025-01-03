#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "ParticlesInBox.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"

template <typename IntegrationMethod, typename ParallelizationMethod>
auto generate_filename(auto const& scenario) {
  // scenario name and number of particles
  auto filename = scenario.name + '_' + std::to_string(scenario.n_particles);

  // integration method
  if constexpr (std::is_same_v<IntegrationMethod, BTRKF78>) {
    filename += "_RKF78";
  } else {
    throw "Unrecognized integration method!\n";
  }

  // parallelization method
  if constexpr (std::is_same_v<ParallelizationMethod, CudaExecutor>) {
    filename += "_CUDA";
  } else {
    throw "Unrecognized parallelization method!\n";
  }

  // current time
  auto now = std::chrono::system_clock::now();
  auto tt = std::chrono::system_clock::to_time_t(now);
  auto tss = std::ostringstream{};
  tss << std::put_time(std::localtime(&tt), "_%Y%m%d_%H%M%S");
  filename += tss.str();

  // softening and tolerance
  filename += "_sof_" + std::to_string(scenario.softening) + "_tol_" +
              std::to_string(scenario.tolerance_value);

  // extension
  filename += ".txt";

  return filename;
}

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

  // auto filename = std::string{"n_body_benchmark_test_cuda_RKF78_"} +
  //                 std::to_string(scenario.tolerance_value) + ".txt";
  auto filename = generate_filename<BTRKF78, CudaExecutor>(scenario);
  output_to_file(filename, output);

  return 0;
}
