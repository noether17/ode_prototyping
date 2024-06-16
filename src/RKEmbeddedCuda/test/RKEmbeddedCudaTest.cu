#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>
#include <numeric>

#include "CudaState.cuh"
#include "RKEmbeddedCuda.cuh"

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

  auto host_result =
      std::sqrt(std::inner_product(host_v.begin(), host_v.end(),
                                   host_scale.begin(), 0.0, std::plus<>{},
                                   [](auto v, auto scale) {
                                     auto scaled_v = v / scale;
                                     return scaled_v * scaled_v;
                                   }) /
                n_var);
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

  auto host_result =
      std::sqrt(std::inner_product(host_v->begin(), host_v->end(),
                                   host_scale->begin(), 0.0, std::plus<>{},
                                   [](auto v, auto scale) {
                                     auto scaled_v = v / scale;
                                     return scaled_v * scaled_v;
                                   }) /
                n_var);
  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  EXPECT_NEAR(host_result, host_cuda_result, tolerance);
}
