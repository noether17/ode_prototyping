#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <numeric>

#include "CudaState.cuh"

TEST(CudaStateTest, CudaStateCanBeDefaultConstructed) {
  auto constexpr n = 10;

  auto cuda_state = CudaState<n>{};

  auto host_state = std::array<double, n>{};
  cuda_state.to_host(host_state);
  for (auto const& x : host_state) {
    ASSERT_DOUBLE_EQ(0.0, x);
  }
}

TEST(CudaStateTest, CudaStateCanBeConstructedFromHostState) {
  auto constexpr n = 10;
  auto host_state = std::array<double, n>{};
  std::iota(host_state.begin(), host_state.end(), 0.0);

  auto cuda_state = CudaState<n>{host_state};

  auto host_cuda_state = std::array<double, n>{};
  cuda_state.to_host(host_cuda_state);
  ASSERT_EQ(host_state, host_cuda_state);
}

TEST(CudaStateTest, CudaStatesCanBeAdded) {
  auto constexpr n = 10;
  auto u = std::array<double, n>{};
  std::iota(u.begin(), u.end(), 0.0);
  auto v = std::array<double, n>{};
  std::iota(v.begin(), v.end(), 1.0);
  auto cuda_u = CudaState<n>{u};
  auto cuda_v = CudaState<n>{v};
  auto cuda_sum = CudaState<n>{};

  elementwise_binary_op(cuda_u, cuda_v, cuda_sum, std::plus<>{});

  auto host_cuda_result = std::array<double, n>{};
  cuda_sum.to_host(host_cuda_result);
  for (auto i = 0; i < n; ++i) {
    ASSERT_DOUBLE_EQ(u[i] + v[i], host_cuda_result[i]);
  }
}

TEST(CudaStateTest, InnerProductSmallState) {
  auto constexpr n = 10;
  auto u = std::array<double, n>{};
  std::iota(u.begin(), u.end(), 0.0);
  auto v = std::array<double, n>{};
  std::iota(v.begin(), v.end(), 1.0);
  auto cuda_u = CudaState<n>{u};
  auto cuda_v = CudaState<n>{v};

  auto dev_result = inner_product(cuda_u, cuda_v);
  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result.get(), sizeof(double),
             cudaMemcpyDeviceToHost);

  auto host_result = std::inner_product(u.begin(), u.end(), v.begin(), 0.0);
  ASSERT_DOUBLE_EQ(host_result, host_cuda_result);
}

TEST(CudaStateTest, InnerProductLargeState) {
  auto constexpr n = 1 << 20;
  auto u_ptr = std::make_unique<std::array<double, n>>();
  std::iota(u_ptr->begin(), u_ptr->end(), 0.0);
  auto v_ptr = std::make_unique<std::array<double, n>>();
  std::iota(v_ptr->begin(), v_ptr->end(), 1.0);
  auto cuda_u = CudaState<n>{*u_ptr};
  auto cuda_v = CudaState<n>{*v_ptr};

  auto dev_result = inner_product(cuda_u, cuda_v);
  auto host_cuda_result = 0.0;
  cudaMemcpy(&host_cuda_result, dev_result.get(), sizeof(double),
             cudaMemcpyDeviceToHost);

  auto host_result =
      std::inner_product(u_ptr->begin(), u_ptr->end(), v_ptr->begin(), 0.0);
  ASSERT_DOUBLE_EQ(host_result, host_cuda_result);
}
