#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "HeapState.hpp"
#include "NBodyODE.hpp"
#include "ParticlesInBox.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"
#include "nbody_io.hpp"

int main() {
  constexpr auto N = 64;
  auto scenario = ParticlesInBox<N, CudaState, double>{};
  constexpr auto n_var = scenario.n_var;

  auto cuda_exe = CudaExecutor{};
  auto output = RawOutput<HeapState<double, n_var>>{};

  auto t0 = 0.0;
  auto tf = scenario.tf;

  RKEmbeddedParallel<BTRKF78>::integrate(
      scenario.initial_state, t0, tf, scenario.tolerance_array,
      scenario.tolerance_array, NBodyODE<double, n_var>{}, output, cuda_exe);

  constexpr auto filename = "RKF78_cuda_n_body_box_output.bin";
  output_to_file(filename, output);

  return 0;
}
