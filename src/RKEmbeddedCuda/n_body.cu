#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include "BTRKF78.hpp"
#include "CudaNBodyOde.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

auto constexpr N = 16;
auto constexpr L = 1.0;
auto constexpr n_var = N * 6;
auto init_state_and_tol() -> std::pair<double*, double*>;

template <typename Output>
void write_to_file(Output const& output, std::string const& filename) {
  auto output_file = std::ofstream{filename};
  for (auto i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (auto j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }
}

int main() {
  auto t0 = 0.0;
  auto tf = std::sqrt(L * L * L / N);
  std::cout << "End time = " << tf << '\n';
  auto [dev_x0, dev_tol] = init_state_and_tol();
  auto ode = CudaNBodyOde<n_var>{1.0e-3};
  auto output = RawCudaOutputWithProgress<n_var>{};

  cuda_integrate<n_var, BTRKF78, CudaNBodyOde<n_var>,
                 RawCudaOutputWithProgress<n_var>>(dev_x0, t0, tf, dev_tol,
                                                   dev_tol, ode, output);

  write_to_file(output, "n_body_output.txt");

  cudaFree(dev_tol);
  cudaFree(dev_x0);

  return 0;
}

auto init_state_and_tol() -> std::pair<double*, double*> {
  auto host_x0 = std::vector<double>(n_var);
  auto gen = std::mt19937{0};
  auto dist = std::uniform_real_distribution<double>(0.0, L);
  for (auto i = 0; i < host_x0.size() / 2; ++i) {
    host_x0[i] = dist(gen);
  }
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  return {dev_x0, dev_tol};
}
