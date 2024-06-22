#pragma once

#include <array>
#include <cmath>

#include "CudaState.cuh"

auto static constexpr block_size = 256;

template <int N>
auto consteval num_blocks() {
  return (N + block_size - 1) / block_size;
}

// struct HE21 {
//   auto static constexpr a = std::array<std::array<double, 1>, 1>{{1.0}};
//   auto static constexpr b = std::array{1.0 / 2.0, 1.0 / 2.0};
//   auto static constexpr bt = std::array{1.0, 0.0};
//   auto static constexpr p = 2;
//   auto static constexpr pt = 1;
//   auto static constexpr n_stages = static_cast<int>(b.size());
// };

__global__ void cuda_compute_error_target(double const* x, double const* rtol,
                                          double const* atol,
                                          double* error_target, int n_var) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < n_var) {
    error_target[i] = atol[i] + rtol[i] * std::abs(x[i]);
    i += blockDim.x * gridDim.x;
  }
}

template <int n_var, typename ODE>
void cuda_estimate_initial_step(double* dev_x0, double* dev_atol,
                                double* dev_rtol, double* dev_dt) {
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));
  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x0, dev_rtol, dev_atol, dev_error_target, n_var);

  // TODO

  cudaFree(dev_error_target);
}

template <int n_var, typename RKMethod, typename ODE>
void cuda_integrate(double* dev_x0, double* dev_t0, double* dev_tf,
                    double* dev_atol, double* dev_rtol) {
  double* dev_ks = nullptr;
  cudaMalloc(&dev_ks, n_var * RKMethod::n_stages * sizeof(double));
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cuda_estimate_initial_step<n_var, ODE>(dev_x0, dev_atol, dev_rtol, dev_dt);

  // TODO

  cudaFree(dev_dt);
  cudaFree(dev_ks);
}
