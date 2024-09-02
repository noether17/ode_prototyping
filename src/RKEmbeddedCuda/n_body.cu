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

  auto output_file = std::ofstream{"n_body_output.txt"};
  for (auto i = 0; i < output.times.size(); ++i) {
    output_file << output.times[i];
    for (auto j = 0; j < n_var; ++j) {
      output_file << ',' << output.states[i][j];
    }
    output_file << '\n';
  }

  cudaFree(dev_tol);
  cudaFree(dev_x0);

  return 0;
}

auto init_state_and_tol() -> std::pair<double*, double*> {
  // Simple Two-Body Orbit
  // auto host_x0 =
  //     std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5,
  //     0.0};
  // Three-Body Figure-8
  // auto host_x0 =
  //    std::array{0.9700436,   -0.24308753, 0.0, -0.9700436,  0.24308753,  0.0,
  //               0.0,         0.0,         0.0, 0.466203685, 0.43236573,  0.0,
  //               0.466203685, 0.43236573,  0.0, -0.93240737, -0.86473146,
  //               0.0};
  // Pythagorean Three-Body
  // auto host_x0 = std::array{1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0,
  //                           0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0};
  // Five-Body Double Figure-8
  // auto host_x0 =
  //    std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
  //               -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
  //               0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
  //               1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
  //               -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  // auto const n_var = host_x0.size();
  // 1024-Body Cube
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
