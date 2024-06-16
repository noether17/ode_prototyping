#pragma once

#include <array>
#include <cmath>

#include "CudaState.cuh"

auto static constexpr block_size = 256;

template <int N>
auto consteval num_blocks() {
  return (N + block_size - 1) / block_size;
}

struct HE21 {
  auto static constexpr a = std::array<std::array<double, 1>, 1>{{1.0}};
  auto static constexpr b = std::array{1.0 / 2.0, 1.0 / 2.0};
  auto static constexpr bt = std::array{1.0, 0.0};
  auto static constexpr p = 2;
  auto static constexpr pt = 1;
  auto static constexpr n_stages = static_cast<int>(b.size());
};

template <int n_var>
void rk_norm(double const* v, double const* scale, double* temp,
             double* result) {
  elementwise_binary_op_kernel<<<num_blocks<n_var>(), block_size>>>(
      v, scale, temp, n_var,
      [] __device__(auto const& v, auto const& scale) { return v / scale; });
  inner_product_kernel<<<num_blocks<n_var>(), block_size>>>(temp, temp, result,
                                                            n_var);
  elementwise_unary_op_kernel<<<1, 1>>>(
      result, 1, [] __device__(auto& x) { x = std::sqrt(x / n_var); });
}

template <int n_var>
void compute_error_target(double const* x, double const* rtol,
                          double const* atol, double* error_target) {
  elementwise_binary_op_kernel<<<num_blocks<n_var>(), block_size>>>(
      x, rtol, error_target, n_var,
      [] __device__(auto const& x, auto const& rtol) {
        return rtol * std::abs(x);
      });
  elementwise_binary_op_kernel<<<num_blocks<n_var>(), block_size>>>(
      atol, error_target, n_var,
      [] __device__(auto const& atol, auto& error_target) {
        error_target += atol;
      });
}

//__global__ void compute_error_target(double* error_target, double* atol,
//                                     double* rtol, double* x0) {
//  auto i = blockIdx.x * blockDim.x + threadIdx.x;
//  while (i < n_var) {
//    error_target[i] = atol[i] + rtol[i] * std::abs(x0[i]);
//    i += blockDim.x * gridDim.x;
//  }
//}
//
// template <int n_var>
//__global__ void rk_norm_kernel(double* v, double* scale, double* result) {
//  __shared__ double cache[block_size];
//  auto i = blockIdx.x * blockDim.x + threadIdx.x;
//  auto cache_index = threadIdx.x;
//  auto temp = 0.0;
//  while (i < n_var) {
//    auto scaled_component = v[i] / scale[i];
//    temp += scaled_component * scaled_component;
//    i += blockDim.x * gridDim.x;
//  }
//  cache[cache_index] = temp;
//
//  __syncthreads();
//  auto reduction_size = blockDim.x;
//  while (reduction_size != 0) {
//    if (cache_index < reduction_size) {
//      cache[cache_index] += cache[cache_index + reduction_size];
//    }
//    __syncthreads();
//    reduction_size /= 2;
//  }
//  if (cache_index == 0) {
//    atomicAdd(result, std::sqrt(cache[0]));
//  }
//}
//
// template <int n_var>
// auto rk_norm(double* v, double* scale) {
//  auto result_dev = static_cast<double*>(nullptr);
//  cudaMalloc(&result_dev, sizeof(double));
//  cudaMemset(result_dev, 0, sizeof(double));
//  rk_norm_kernel<<<num_blocks<n_var>(), block_size>>>(v, scale, result_dev);
//  auto result = 0.0;
//  cudaMemcpy(&result, result_dev, sizeof(double), cudaMemcpyDeviceToHost);
//  cudaFree(result_dev);
//  return result;
//}
//
// template <int n_var, typename RKMethod, typename ODE>
// double estimate_initial_step(double* x0_dev, double* atol_dev, double*
// rtol_dev,
//                             ODE ode) {
//  auto error_target_dev = static_cast<double*>(nullptr);
//  cudaMalloc(&error_target_dev, n_var * sizeof(double));
//  compute_error_target<<<num_blocks<n_var>(), block_size>>>(
//      error_target_dev, atol_dev, rtol_dev, x0_dev);
//
//  auto f0_dev = static_cast<double*>(nullptr);
//  cudaMalloc(&f0_dev, n_var * sizeof(double));
//  ode<<<num_blocks<n_var>(), block_size>>>(x0_dev, f0_dev);
//  auto d0 = rk_norm(x0, error_target);
//  auto d1 = rk_norm(f0, error_target);
//  auto dt0 = (d0 < 1.0e-5 or d1 < 1.0e-5) ? 1.0e-6 : 0.01 * (d0 / d1);
//
//  auto x1 = State<n_var>{};
//  for (int i = 0; i < n_var; ++i) {
//    x1[i] = x0[i] + dt0 * f0[i];
//  }
//  auto f1 = State<n_var>{};
//  ode(x1, f1);
//  auto df = State<n_var>{};
//  for (int i = 0; i < n_var; ++i) {
//    df[i] = f1[i] - f0[i];
//  }
//  auto d2 = rk_norm(df, error_target) / dt0;
//
//  auto dt1 =
//      (std::max(d1, d2) <= 1.0e-15)
//          ? std::max(1.0e-6, dt0 * 1.0e-3)
//          : std::pow(0.01 / std::max(d1, d2), (1.0 / (1.0 + RKMethod::p)));
//  return std::min(100.0 * dt0, dt1);
//}
//
// template <int n_var, typename RKMethod, typename ODE>
// void RKEmbeddedIntegrate(double* x0, double t0, double tf, double* atol,
//                         double* rtol, ODE ode) {
//  auto constexpr a = RKMethod::a;
//  auto constexpr b = RKMethod::b;
//  auto constexpr bt = RKMethod::bt;
//  auto constexpr p = RKMethod::p;
//  auto constexpr pt = RKMethod::pt;
//  auto constexpr n_stages = RKMethod::n_stages;
//
//  auto ks = std::array<std::array<double, n_var>, n_stages>{};
//
//  auto dt = estimate_initial_step(x0, atol, rtol, ode);
//}
