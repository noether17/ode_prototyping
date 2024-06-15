#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "CudaUtils.cuh"

TEST(CudaUtilsTest, ElementWiseAdd) {
  auto constexpr n = 10;
  auto a = std::vector<double>(n);
  auto b = std::vector<double>(n);
  auto c = std::vector<double>(n);
  std::iota(a.begin(), a.end(), 0.0);
  std::iota(b.begin(), b.end(), 1.0);

  auto a_dev = static_cast<double*>(nullptr);
  auto b_dev = static_cast<double*>(nullptr);
  auto c_dev = static_cast<double*>(nullptr);
  cudaMalloc(&a_dev, n * sizeof(double));
  cudaMalloc(&b_dev, n * sizeof(double));
  cudaMalloc(&c_dev, n * sizeof(double));
  cudaMemcpy(a_dev, a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  elementwise_add<<<1, 1>>>(a_dev, b_dev, c_dev, n);
  auto c_dev_host = std::vector<double>(n);
  cudaMemcpy(c_dev_host.data(), c_dev, n * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);

  std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<double>());
  ASSERT_EQ(c, c_dev_host);
}

void cuda_square(double* v, int n) {
  elementwise_unary_op_kernel<<<1, 1>>>(
      v, n, [] __device__(auto x) { return x * x; });
}

TEST(CudaUtilsTest, ElementWiseUnaryOp) {
  auto constexpr n = 10;
  auto v = std::vector<double>(n);
  std::iota(v.begin(), v.end(), 0.0);

  auto v_dev = static_cast<double*>(nullptr);
  cudaMalloc(&v_dev, n * sizeof(double));
  cudaMemcpy(v_dev, v.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  cuda_square(v_dev, n);
  auto v_dev_host = std::vector<double>(n);
  cudaMemcpy(v_dev_host.data(), v_dev, n * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(v_dev);

  std::transform(v.begin(), v.end(), v.begin(), [](auto x) { return x * x; });
  ASSERT_EQ(v, v_dev_host);
}
