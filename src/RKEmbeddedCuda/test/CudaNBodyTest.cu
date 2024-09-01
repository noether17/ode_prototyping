#include <gtest/gtest.h>

#include "BTRKF78.hpp"
#include "CUDANBodyODE.cuh"
#include "RKEmbeddedCuda.cuh"
#include "RawCudaOutput.cuh"

// The following tests are n-body scenarios from Roa, et al.

TEST(CudaNBodyTest, SimpleTwoBodyOrbit) {
  auto host_x0 =
      std::array{1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0};
  auto constexpr n_var = host_x0.size();
  auto t0 = 0.0;
  auto tf = 200.0;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto masses = std::array{1.0, 1.0};
  auto ode = CUDANBodyODE<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CUDANBodyODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(463, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(99.715523343935061, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(200.0, output.times.back());

  EXPECT_EQ(463, output.states.size());
  EXPECT_DOUBLE_EQ(0.91802735646432221,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.39651697893821747,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-0.91802735646432221,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(0.39651697893821747,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(0.86232100807629852, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-0.50636188936069804, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(-0.86232100807629852, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(0.50636188936069804, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaNBodyTest, ThreeBodyFigureEight) {
  auto host_x0 =
      std::array{0.9700436,   -0.24308753, 0.0, -0.9700436,  0.24308753,  0.0,
                 0.0,         0.0,         0.0, 0.466203685, 0.43236573,  0.0,
                 0.466203685, 0.43236573,  0.0, -0.93240737, -0.86473146, 0.0};
  auto constexpr n_var = host_x0.size();
  auto t0 = 0.0;
  auto tf = 100.0;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto masses = std::array{1.0, 1.0, 1.0};
  auto ode = CUDANBodyODE<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CUDANBodyODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(1075, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(50.02129573636411, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(100.0, output.times.back());

  EXPECT_EQ(1075, output.states.size());
  EXPECT_DOUBLE_EQ(0.469521336317547,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.32583157658428957,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-1.0796243523729612,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.030038192685493133,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(0.61010301605537665,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(0.35586976926976155,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(-0.16245212599824632, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(0.14465331754063407, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(-0.87109309893576614, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-0.31070363737495921, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(1.033545224933937, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(0.16605031983430527, output.states.back()[7]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[8]);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaNBodyTest, PythagoreanThreeBody) {
  auto host_x0 = std::array{1.0, 3.0, 0.0, -2.0, -1.0, 0.0, 1.0, -1.0, 0.0,
                            0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0,  0.0};
  auto constexpr n_var = host_x0.size();
  auto t0 = 0.0;
  auto tf = 70.0;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto masses = std::array{3.0, 4.0, 5.0};
  auto ode = CUDANBodyODE<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CUDANBodyODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(2608, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(41.373537271832326, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(70.0, output.times.back());

  EXPECT_EQ(2608, output.states.size());
  EXPECT_DOUBLE_EQ(0.51631736847579068,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(0.91874707313820136,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(0.23355039999576899,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.71452918385082398,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(-0.49663074108252825,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(0.020375103199220784,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(-1.2489670606635803, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(15.220501544849084, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(0.60680181315361892, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-4.4770240956622382, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(0.26393878587425518, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(-5.5506816503766165, output.states.back()[7]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[8]);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}

TEST(CudaNBodyTest, FiveBodyDoubleFigureEight) {
  auto host_x0 =
      std::array{1.657666,  0.0,       0.0, 0.439775,  -0.169717, 0.0,
                 -1.268608, -0.267651, 0.0, -1.268608, 0.267651,  0.0,
                 0.439775,  0.169717,  0.0, 0.0,       -0.593786, 0.0,
                 1.822785,  0.128248,  0.0, 1.271564,  0.168645,  0.0,
                 -1.271564, 0.168645,  0.0, -1.822785, 0.128248,  0.0};
  auto constexpr n_var = host_x0.size();
  auto t0 = 0.0;
  auto tf = 6.3;
  auto host_tol = std::array<double, n_var>{};
  std::fill(host_tol.begin(), host_tol.end(), 1.0e-10);
  double* dev_x0 = nullptr;
  cudaMalloc(&dev_x0, n_var * sizeof(double));
  cudaMemcpy(dev_x0, host_x0.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  double* dev_tol = nullptr;
  cudaMalloc(&dev_tol, n_var * sizeof(double));
  cudaMemcpy(dev_tol, host_tol.data(), n_var * sizeof(double),
             cudaMemcpyHostToDevice);
  auto masses = std::array{1.0, 1.0, 1.0, 1.0, 1.0};
  auto ode = CUDANBodyODE<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CUDANBodyODE<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(172, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(3.1699354056933444, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(6.2999999999999998, output.times.back());

  EXPECT_EQ(172, output.states.size());
  EXPECT_DOUBLE_EQ(-1.657049151854644,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.016818850570985221,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-0.4911121472955573,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.16256448825335423,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(1.232320992323289,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(-0.26143646571614138,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(1.3042855406517948,
                   output.states[output.states.size() / 2][9]);
  EXPECT_DOUBLE_EQ(0.2709593682488175,
                   output.states[output.states.size() / 2][10]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][11]);
  EXPECT_DOUBLE_EQ(-0.38844523382488511,
                   output.states[output.states.size() / 2][12]);
  EXPECT_DOUBLE_EQ(0.16986043629166359,
                   output.states[output.states.size() / 2][13]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][14]);
  EXPECT_DOUBLE_EQ(1.657321312127696, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-0.013882068867849373, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(0.46802604978556767, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-0.17063101334421135, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(-1.2481480178200384, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(-0.26313782979034916, output.states.back()[7]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[8]);
  EXPECT_DOUBLE_EQ(-1.2883624086146654, output.states.back()[9]);
  EXPECT_DOUBLE_EQ(0.27516687291849024, output.states.back()[10]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[11]);
  EXPECT_DOUBLE_EQ(0.41116306452143053, output.states.back()[12]);
  EXPECT_DOUBLE_EQ(0.17248403908392046, output.states.back()[13]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[14]);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}
