#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <numeric>

#include "CudaState.cuh"
#include "RKEmbeddedCuda.cuh"

template <int n_var>
auto host_rk_norm(std::array<double, n_var> const& v,
                  std::array<double, n_var> const& scale) {
  auto scaled_mag = std::inner_product(v.begin(), v.end(), scale.begin(), 0.0,
                                       std::plus<>{}, [](auto v, auto scale) {
                                         auto scaled_v = v / scale;
                                         return scaled_v * scaled_v;
                                       });
  return std::sqrt(scaled_mag / n_var);
}

TEST(RKEmbeddedCudaTest, RKNormTestSmall) {
  auto constexpr n_var = 10;
  auto host_v = std::array<double, n_var>{};
  std::iota(host_v.begin(), host_v.end(), 0.0);
  auto host_scale = std::array<double, n_var>{};
  std::iota(host_scale.begin(), host_scale.end(), 1.0);
  auto dev_v = CudaState<n_var>{host_v};
  auto dev_scale = CudaState<n_var>{host_scale};

  auto dev_result = CudaState<1>{};
  auto dev_temp = CudaState<n_var>{};
  rk_norm<n_var>(dev_v, dev_scale, dev_temp, dev_result);

  auto host_result = host_rk_norm<n_var>(host_v, host_scale);
  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);
}

TEST(RKEmbeddedCudaTest, RKNormTestLarge) {
  auto constexpr n_var = 1 << 20;
  auto const tolerance =
      std::numeric_limits<double>::epsilon() * std::log2(n_var);
  auto host_v = std::make_unique<std::array<double, n_var>>();
  std::iota(host_v->begin(), host_v->end(), 0.0);
  auto host_scale = std::make_unique<std::array<double, n_var>>();
  std::iota(host_scale->begin(), host_scale->end(), 1.0);
  auto dev_v = CudaState<n_var>{*host_v};
  auto dev_scale = CudaState<n_var>{*host_scale};

  auto dev_result = CudaState<1>{};
  auto dev_temp = CudaState<n_var>{};
  rk_norm<n_var>(dev_v, dev_scale, dev_temp, dev_result);

  auto host_result = host_rk_norm<n_var>(*host_v, *host_scale);
  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(host_result, host_cuda_result, tolerance);
}

template <int n_var>
void host_compute_error_target(std::array<double, n_var> const& x,
                               std::array<double, n_var> const& rtol,
                               std::array<double, n_var> const& atol,
                               std::array<double, n_var>& error_target) {
  for (auto i = 0; i < n_var; ++i) {
    error_target[i] = atol[i] + rtol[i] * std::abs(x[i]);
  }
}

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::array<double, n_var>{};
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_rtol = std::array<double, n_var>{};
  std::iota(host_rtol.begin(), host_rtol.end(), 1.0);
  auto host_atol = std::array<double, n_var>{};
  std::iota(host_atol.begin(), host_atol.end(), 2.0);
  auto dev_x = CudaState<n_var>{host_x};
  auto dev_rtol = CudaState<n_var>{host_rtol};
  auto dev_atol = CudaState<n_var>{host_atol};

  auto dev_result = CudaState<n_var>{};
  compute_error_target<n_var>(dev_x, dev_rtol, dev_atol, dev_result);

  auto host_result = std::array<double, n_var>();
  host_compute_error_target<n_var>(host_x, host_rtol, host_atol, host_result);
  auto host_cuda_result = std::array<double, n_var>{};
  dev_result.to_host(host_cuda_result);
  for (auto i = 0; i < n_var; ++i) {
    EXPECT_DOUBLE_EQ(host_result[i], host_cuda_result[i]);
  }
}

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestLarge) {
  auto constexpr n_var = 1 << 20;
  // auto const tolerance =
  //     std::numeric_limits<double>::epsilon() * std::log2(n_var);
  auto host_x = std::make_unique<std::array<double, n_var>>();
  std::iota(host_x->begin(), host_x->end(), 0.0);
  auto host_rtol = std::make_unique<std::array<double, n_var>>();
  std::iota(host_rtol->begin(), host_rtol->end(), 1.0);
  auto host_atol = std::make_unique<std::array<double, n_var>>();
  std::iota(host_atol->begin(), host_atol->end(), 2.0);
  auto dev_x = CudaState<n_var>{*host_x};
  auto dev_rtol = CudaState<n_var>{*host_rtol};
  auto dev_atol = CudaState<n_var>{*host_atol};

  auto dev_result = CudaState<n_var>{};
  compute_error_target<n_var>(dev_x, dev_rtol, dev_atol, dev_result);

  auto host_result = std::make_unique<std::array<double, n_var>>();
  host_compute_error_target<n_var>(*host_x, *host_rtol, *host_atol,
                                   *host_result);
  auto host_cuda_result = std::make_unique<std::array<double, n_var>>();
  dev_result.to_host(*host_cuda_result);
  for (auto i = 0; i < n_var; ++i) {
    // EXPECT_NEAR(host_result[i], host_cuda_result[i], tolerance);
    EXPECT_DOUBLE_EQ((*host_result)[i], (*host_cuda_result)[i]);
  }
}
