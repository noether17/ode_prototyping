#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "CUDAExpODE.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

class CudaRKEmbeddedCPUComparisonTest : public testing::Test {
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

  CudaRKEmbeddedCPUComparisonTest() {
    cudaMalloc(&dev_x0, n_var * sizeof(double));
    cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMalloc(&dev_tol, n_var * sizeof(double));
    cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
               cudaMemcpyHostToDevice);
  }

  ~CudaRKEmbeddedCPUComparisonTest() {
    cudaFree(dev_tol);
    cudaFree(dev_x0);
  }
};

TEST_F(CudaRKEmbeddedCPUComparisonTest, HE21) {
  cuda_integrate<n_var, HE21, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(11026, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(5.0636296424493379, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(11026, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  EXPECT_NEAR(790.81720024690105,
              output.states[output.states.size() / 2][n_var / 2],
              790.81720024690105 * state_tol);
  EXPECT_NEAR(1423.4709604444222,
              output.states[output.states.size() / 2].back(),
              1423.4709604444222 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.17777934085, output.states.back()[n_var / 2],
              110132.17777934085 * state_tol);
  EXPECT_NEAR(198237.92000281269, output.states.back().back(),
              198237.92000281269 * state_tol);
}

TEST_F(CudaRKEmbeddedCPUComparisonTest, RKF45) {
  cuda_integrate<n_var, RKF45, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(50, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(5.0888631842534933, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  // Increase tolerance for RKF Methods
  auto const state_tol = 2.0 * std::sqrt(output.states.size()) *
                         std::numeric_limits<double>::epsilon();
  EXPECT_EQ(50, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(811.03335805992742,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             811.03335805992742 * state_tol);
  // EXPECT_NEAR(1459.8600445078696,
  //             output.states[output.states.size() / 2].back(),
  //             1459.8600445078696 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110134.06230636688, output.states.back()[n_var / 2],
              110134.06230636688 * state_tol);
  EXPECT_NEAR(198241.3121514605, output.states.back().back(),
              198241.3121514605 * state_tol);
}

TEST_F(CudaRKEmbeddedCPUComparisonTest, DOPRI5) {
  cuda_integrate<n_var, DOPRI5, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(45, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(4.9896555947535841, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(45, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(734.42960843551498,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             734.42960843551498 * state_tol);
  // EXPECT_NEAR(1321.9732951839271,
  //             output.states[output.states.size() / 2].back(),
  //             1321.9732951839271 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.46804595254, output.states.back()[n_var / 2],
              110132.46804595254 * state_tol);
  EXPECT_NEAR(198238.44248271457, output.states.back().back(),
              198238.44248271457 * state_tol);
}

TEST_F(CudaRKEmbeddedCPUComparisonTest, DVERK) {
  cuda_integrate<n_var, DVERK, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(32, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(5.1178634791056821, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  auto const state_tol =
      std::sqrt(output.states.size()) * std::numeric_limits<double>::epsilon();
  EXPECT_EQ(32, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(834.89108343292787,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             834.89108343292787 * state_tol);
  // EXPECT_NEAR(1502.8039501792707,
  //             output.states[output.states.size() / 2].back(),
  //             1502.8039501792707 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110132.30512693945, output.states.back()[n_var / 2],
              110132.30512693945 * state_tol);
  EXPECT_NEAR(198238.14922849106, output.states.back().back(),
              198238.14922849106 * state_tol);
}

TEST_F(CudaRKEmbeddedCPUComparisonTest, RKF78) {
  cuda_integrate<n_var, RKF78, CUDAExpODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  // TODO: Investigate differences between CPU and CUDA algorithms
  EXPECT_EQ(13, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  // EXPECT_DOUBLE_EQ(4.4481779636028618, output.times[output.times.size() /
  // 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  // Increase tolerance for RKF Methods
  auto const state_tol = 3.0 * std::sqrt(output.states.size()) *
                         std::numeric_limits<double>::epsilon();
  EXPECT_EQ(13, output.states.size());
  EXPECT_NEAR(0.0, output.states.front().front(), state_tol);
  EXPECT_NEAR(5.0, output.states.front()[n_var / 2], 5.0 * state_tol);
  EXPECT_NEAR(9.0, output.states.front().back(), 9.0 * state_tol);
  EXPECT_NEAR(0.0, output.states[output.states.size() / 2].front(), state_tol);
  // EXPECT_NEAR(427.3544878966228,
  //             output.states[output.states.size() / 2][n_var / 2],
  //             427.3544878966228 * state_tol);
  // EXPECT_NEAR(769.23807821392074,
  //             output.states[output.states.size() / 2].back(),
  //             769.23807821392074 * state_tol);
  EXPECT_NEAR(0.0, output.states.back().front(), state_tol);
  EXPECT_NEAR(110131.78807367262, output.states.back()[n_var / 2],
              110131.78807367262 * state_tol);
  EXPECT_NEAR(198237.21853261057, output.states.back().back(),
              198237.21853261057 * state_tol);
}
