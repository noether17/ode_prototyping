#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <numeric>

#include "RKEmbeddedCuda.cuh"

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 1.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 2.0);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));

  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x, dev_rtol, dev_atol, dev_error_target, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_error_target, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_atol[i] + host_rtol[i] * std::abs(host_x[i]);
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_error_target);
  cudaFree(dev_atol);
  cudaFree(dev_rtol);
  cudaFree(dev_x);
}

TEST(RKEmbeddedCudaTest, ComputeErrorTargetTestLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_rtol = std::vector<double>(n_var);
  std::iota(host_rtol.begin(), host_rtol.end(), 1.0);
  auto host_atol = std::vector<double>(n_var);
  std::iota(host_atol.begin(), host_atol.end(), 2.0);
  double* dev_x = nullptr;
  cudaMalloc(&dev_x, n_var * sizeof(double));
  cudaMemcpy(dev_x, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_rtol = nullptr;
  cudaMalloc(&dev_rtol, n_var * sizeof(double));
  cudaMemcpy(dev_rtol, host_rtol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_atol = nullptr;
  cudaMalloc(&dev_atol, n_var * sizeof(double));
  cudaMemcpy(dev_atol, host_atol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_error_target = nullptr;
  cudaMalloc(&dev_error_target, n_var * sizeof(double));

  cuda_compute_error_target<<<num_blocks<n_var>(), block_size>>>(
      dev_x, dev_rtol, dev_atol, dev_error_target, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_error_target, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_atol[i] + host_rtol[i] * std::abs(host_x[i]);
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_error_target);
  cudaFree(dev_atol);
  cudaFree(dev_rtol);
  cudaFree(dev_x);
}

TEST(RKEmbeddedCudaTest, RKNormTestSmall) {
  auto constexpr n_var = 10;
  auto host_v = std::vector<double>(n_var);
  std::iota(host_v.begin(), host_v.end(), 0.0);
  auto host_scale = std::vector<double>(n_var);
  std::iota(host_scale.begin(), host_scale.end(), 1.0);
  double* dev_v = nullptr;
  cudaMalloc(&dev_v, n_var * sizeof(double));
  cudaMemcpy(dev_v, host_v.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_scale = nullptr;
  cudaMalloc(&dev_scale, n_var * sizeof(double));
  cudaMemcpy(dev_scale, host_scale.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_result = nullptr;
  cudaMalloc(&dev_result, sizeof(double));

  cuda_rk_norm<n_var>(dev_v, dev_scale, dev_result);

  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_result);
  cudaFree(dev_scale);
  cudaFree(dev_v);
}

TEST(RKEmbeddedCudaTest, RKNormTestLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_v = std::vector<double>(n_var);
  std::iota(host_v.begin(), host_v.end(), 0.0);
  auto host_scale = std::vector<double>(n_var);
  std::iota(host_scale.begin(), host_scale.end(), 1.0);
  double* dev_v = nullptr;
  cudaMalloc(&dev_v, n_var * sizeof(double));
  cudaMemcpy(dev_v, host_v.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_scale = nullptr;
  cudaMalloc(&dev_scale, n_var * sizeof(double));
  cudaMemcpy(dev_scale, host_scale.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_result = nullptr;
  cudaMalloc(&dev_result, sizeof(double));

  cuda_rk_norm<n_var>(dev_v, dev_scale, dev_result);

  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result, sizeof(double),
             cudaMemcpyDeviceToHost);
  auto host_result = std::sqrt(
      std::inner_product(host_v.begin(), host_v.end(), host_scale.begin(), 0.0,
                         std::plus<>{},
                         [](auto a, auto b) { return (a / b) * (a / b); }) /
      n_var);
  EXPECT_DOUBLE_EQ(host_result, host_cuda_result);

  cudaFree(dev_result);
  cudaFree(dev_scale);
  cudaFree(dev_v);
}

TEST(RKEmbeddedCudaTest, EulerStepSmall) {
  auto constexpr n_var = 10;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto host_dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cudaMemcpy(dev_dt, &host_dt, sizeof(double), cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dev_dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + host_dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_dt);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}

TEST(RKEmbeddedCudaTest, EulerStepLarge) {
  auto constexpr n_var = 1 << 20;
  auto host_x = std::vector<double>(n_var);
  std::iota(host_x.begin(), host_x.end(), 0.0);
  auto host_f = std::vector<double>(n_var);
  std::iota(host_f.begin(), host_f.end(), 1.0);
  auto host_dt = 0.1;
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_f0 = nullptr;
  cudaMalloc(&dev_f0, n_var * sizeof(double));
  cudaMemcpy(dev_f0, host_f.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_dt = nullptr;
  cudaMalloc(&dev_dt, sizeof(double));
  cudaMemcpy(dev_dt, &host_dt, sizeof(double), cudaMemcpyHostToDevice);
  double* dev_x1 = nullptr;
  cudaMalloc(&dev_x1, n_var * sizeof(double));

  cuda_euler_step<<<num_blocks<n_var>(), block_size>>>(dev_x0, dev_f0, dev_dt,
                                                       dev_x1, n_var);

  auto host_cuda_result = std::vector<double>(n_var);
  cudaMemcpy(host_cuda_result.data(), dev_x1, n_var * sizeof(double),
             cudaMemcpyDeviceToHost);
  for (auto i = 0; i < n_var; ++i) {
    auto host_result = host_x[i] + host_dt * host_f[i];
    EXPECT_DOUBLE_EQ(host_result, host_cuda_result[i]);
  }

  cudaFree(dev_x1);
  cudaFree(dev_dt);
  cudaFree(dev_f0);
  cudaFree(dev_x0);
}
