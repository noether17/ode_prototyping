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
  std::transform(a.begin(), a.end(), b.begin(), c.begin(), std::plus<double>());

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

  ASSERT_EQ(c, c_dev_host);
}
