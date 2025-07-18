#include <gtest/gtest.h>

#include <functional>
#include <numeric>

#include "CudaExecutor.cuh"
#include "CudaState.cuh"

class CudaExecutorTest : public ::testing::Test {
 protected:
  static constexpr auto N = 10;
  CudaExecutor cuda_exec{};

  static constexpr auto host_state1 = [] {
    auto a = std::array<double, N>{};
    std::iota(a.begin(), a.end(), 0.0);
    return a;
  }();

  static constexpr auto host_state2 = [] {
    auto a = std::array<double, N>{};
    std::iota(a.begin(), a.end(), 1.0);
    return a;
  }();
};

void constexpr add_kernel(int i, double* c, double const* a, double const* b) {
  c[i] = a[i] + b[i];
}

TEST_F(CudaExecutorTest, ElementWiseAdd) {
  auto cuda_state1 = CudaState{host_state1};
  auto cuda_state2 = CudaState{host_state2};
  auto cuda_result = decltype(cuda_state1){};

  cuda_exec.call_parallel_kernel<add_kernel>(
      N, cuda_result.data(), cuda_state1.data(), cuda_state2.data());

  auto host_result = std::array<double, N>{};
  cuda_result.copy_to_span(host_result);
  for (auto i = 0; auto const& x : host_result) {
    EXPECT_DOUBLE_EQ(2 * (i++) + 1, x);
  }
}

constexpr auto binary_add(double a, double b) { return a + b; }

constexpr auto element_wise_multiply(int i, double const* a, double const* b) {
  return a[i] * b[i];
}

TEST_F(CudaExecutorTest, InnerProduct) {
  auto cuda_state1 = CudaState{host_state1};
  auto cuda_state2 = CudaState{host_state2};

  auto result =
      cuda_exec.transform_reduce<double, binary_add, element_wise_multiply>(
          0.0, N, cuda_state1.data(), cuda_state2.data());

  EXPECT_DOUBLE_EQ(330.0, result);
}
