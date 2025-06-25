#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "StateUtils.hpp"

class CudaStateTest : public testing::Test {
 protected:
  static constexpr auto N = 10;
  static constexpr auto threads_per_block = 64;
  static constexpr auto blocks_per_grid =
      (N + threads_per_block - 1) / threads_per_block;
};

TEST_F(CudaStateTest, CudaStateCanBeDefaultConstructed) {
  auto cuda_state = CudaState<double, N>{};

  auto host_copy = std::array<double, N>{};
  cuda_state.copy_to_span(host_copy);
  for (auto const& x : host_copy) {
    ASSERT_DOUBLE_EQ(0.0, x);
  }
}

TEST_F(CudaStateTest, CudaStateCanBeConstructedFromHostState) {
  auto host_state = std::array<double, N>{};
  std::iota(host_state.begin(), host_state.end(), 0.0);

  auto cuda_state = CudaState<double, N>{host_state};

  auto host_copy = std::array<double, N>{};
  cuda_state.copy_to_span(host_copy);
  ASSERT_EQ(host_state, host_copy);
}

TEST_F(CudaStateTest, CudaStateCanBeCopied) {
  auto host_state = std::array<double, N>{};
  std::iota(host_state.begin(), host_state.end(), 0.0);
  auto cuda_state1 = CudaState<double, N>{host_state};

  auto cuda_state2 = cuda_state1;

  auto host_copy1 = std::array<double, N>{};
  cuda_state1.copy_to_span(host_copy1);
  auto host_copy2 = std::array<double, N>{};
  cuda_state2.copy_to_span(host_copy2);
  ASSERT_EQ(host_copy1, host_copy2);
}

__global__ void cuda_iota(double* array, double value, int N) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  while (i < N) {
    array[i] = value + i;
    i += blockDim.x * gridDim.x;
  }
}

TEST_F(CudaStateTest, CudaStateCopyIsIndependent) {
  auto cuda_state1 = CudaState<double, N>{};

  // Modifying cuda_state2 should not affect cuda_state1.
  auto cuda_state2 = cuda_state1;
  cuda_iota<<<blocks_per_grid, threads_per_block>>>(cuda_state2.data(), 0.0, N);

  // cuda_state1 == 0, cuda_state2 == iota.
  auto host_copy1 = std::array<double, N>{};
  cuda_state1.copy_to_span(host_copy1);
  for (auto const& x : host_copy1) {
    ASSERT_DOUBLE_EQ(0.0, x);
  }
  auto host_copy2 = std::array<double, N>{};
  cuda_state2.copy_to_span(host_copy2);
  for (auto i = 0; auto const& x : host_copy2) {
    ASSERT_DOUBLE_EQ(i++, x);
  }
}

TEST_F(CudaStateTest, CudaStateCanBeFilledWithValue) {
  auto cuda_state = CudaState<double, N>{};
  auto cuda_exe = CudaExecutor{};

  fill(cuda_exe, cuda_state, 3.0);

  auto host_copy = std::array<double, N>{};
  cuda_state.copy_to_span(host_copy);
  for (auto const& x : host_copy) {
    ASSERT_DOUBLE_EQ(3.0, x);
  }
}
