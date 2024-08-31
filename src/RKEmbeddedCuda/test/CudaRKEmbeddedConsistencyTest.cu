#include <gtest/gtest.h>

#include <array>
#include <numeric>
#include <vector>

#include "CUDAExpODE.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

class CudaRKEmbeddedConsistencyTest : public testing::Test {
 protected:
  auto static constexpr n_var = 10;
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 10.0;
  auto static constexpr host_x0 = []() {
    auto x0 = std::array<double, n_var>{};
    std::iota(x0.begin(), x0.end(), 0.0);
    return x0;
  }();
  auto static constexpr host_tol = []() {
    auto tol = std::array<double, n_var>{};
    std::fill(tol.begin(), tol.end(), 1.0e-6);
    return tol;
  }();

  double* dev_x0{};
  double* dev_tol{};
  CUDAExpODE<n_var> ode{};
  RawCudaOutput<n_var> output{};

  CudaRKEmbeddedConsistencyTest() {
    cudaMalloc(&dev_x0, n_var * sizeof(double));
    cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMalloc(&dev_tol, n_var * sizeof(double));
    cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  ~CudaRKEmbeddedConsistencyTest() {
    cudaFree(dev_tol);
    cudaFree(dev_x0);
  }
};

TEST_F(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestHE21) {
  cuda_integrate<n_var, HE21, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(11026, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.0636296424493379, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(11026, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(790.81720024690105,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1423.4709604444204,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.17777934109, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.92000281301, output.states.back().back());
}

TEST_F(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestRKF45) {
  cuda_integrate<n_var, RKF45, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(50, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.088863184252939, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(50, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(811.03335805947813,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1459.8600445070604,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110134.06230636714, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198241.31215146082, output.states.back().back());
}

TEST_F(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestDOPRI5) {
  cuda_integrate<n_var, DOPRI5, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(45, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(4.9896555948055186, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(45, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(734.42960847365703,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1321.9732952525828,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.46804595242, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.44248271445, output.states.back().back());
}

TEST_F(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestDVERK) {
  cuda_integrate<n_var, DVERK, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(32, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.1178634782992676, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(32, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(834.89108275966021,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1502.8039489673879,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110132.30512693951, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.14922849112, output.states.back().back());
}

TEST_F(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestRKF78) {
  cuda_integrate<n_var, RKF78, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(13, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(4.448177963590588, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(13, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front().front());
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front().back());
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2].front());
  EXPECT_DOUBLE_EQ(427.35448789137757,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(769.23807820447973,
                   output.states[output.states.size() / 2].back());
  EXPECT_DOUBLE_EQ(0.0, output.states.back().front());
  EXPECT_DOUBLE_EQ(110131.78807367239, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.21853261034, output.states.back().back());
}
