#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "CUDAExpODE.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

TEST(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestHE21) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  auto ode = CUDAExpODE<n_var>{};
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

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestRKF45) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  auto ode = CUDAExpODE<n_var>{};
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

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestDOPRI5) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  auto ode = CUDAExpODE<n_var>{};
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

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestDVERK) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  auto ode = CUDAExpODE<n_var>{};
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

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaRKEmbeddedConsistencyTest, CUDAIntegrateConsistencyTestRKF78) {
  auto constexpr n_var = 10;
  auto host_x0 = std::vector<double>(n_var);
  std::iota(host_x0.begin(), host_x0.end(), 0.0);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto host_tol = std::vector<double>(n_var);
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-6);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto output = RawCudaOutput<n_var>{};

  auto ode = CUDAExpODE<n_var>{};
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

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}
