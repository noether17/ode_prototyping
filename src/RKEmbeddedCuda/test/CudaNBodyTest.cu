#include <gtest/gtest.h>

#include "BTRKF78.hpp"
#include "CudaNBodyOde.cuh"
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
  auto ode = CudaNBodyOde<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CudaNBodyOde<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(463, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(99.715523343591755, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(200.0, output.times.back());

  EXPECT_EQ(463, output.states.size());
  EXPECT_DOUBLE_EQ(0.91802735639633759,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.3965169790956089,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-0.91802735639633759,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(0.3965169790956089,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(0.86232100807655865, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-0.5063618893602444, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(-0.86232100807655865, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(0.5063618893602444, output.states.back()[4]);
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
  auto ode = CudaNBodyOde<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CudaNBodyOde<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(1075, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(50.02129542712639, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(100.0, output.times.back());

  EXPECT_EQ(1075, output.states.size());
  EXPECT_DOUBLE_EQ(0.46952099381759138,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.32583146164166943,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-1.0796243381461423,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.030038337177189005,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(0.61010334432863123,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(0.35586979881881536,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(-0.16245212599957398, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(0.14465331754168231, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(-0.87109309893462383, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-0.31070363737572371, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(1.0335452249343748, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(0.16605031983390861, output.states.back()[7]);
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
  auto ode = CudaNBodyOde<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CudaNBodyOde<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(2604, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(41.321564513004127, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(70.0, output.times.back());

  EXPECT_EQ(2604, output.states.size());
  EXPECT_DOUBLE_EQ(0.55469046517182663,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(0.94268382752096957,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(0.13962544401628571,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.60967695039750425,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(-0.44451463431543442,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(-0.077868736195350563,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(-1.214551018340075, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(15.280237924896923, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(0.57477281292641713, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-4.5176537944611113, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(0.26891236066430807, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(-5.5540197193710243, output.states.back()[7]);
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
  auto ode = CudaNBodyOde<n_var>{masses};
  auto output = RawCudaOutput<n_var>{};

  cuda_integrate<n_var, BTRKF78, CudaNBodyOde<n_var>, RawCudaOutput<n_var>>(
      dev_x0, t0, tf, dev_tol, dev_tol, ode, output);

  EXPECT_EQ(172, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(3.1699354065630785, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(6.2999999999999998, output.times.back());

  EXPECT_EQ(172, output.states.size());
  EXPECT_DOUBLE_EQ(-1.6570491518165833,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.016818851087433022,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][2]);
  EXPECT_DOUBLE_EQ(-0.49111214885088855,
                   output.states[output.states.size() / 2][3]);
  EXPECT_DOUBLE_EQ(-0.16256448793085904,
                   output.states[output.states.size() / 2][4]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][5]);
  EXPECT_DOUBLE_EQ(1.2323209912038431,
                   output.states[output.states.size() / 2][6]);
  EXPECT_DOUBLE_EQ(-0.2614364654810421,
                   output.states[output.states.size() / 2][7]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][8]);
  EXPECT_DOUBLE_EQ(1.304285541733357,
                   output.states[output.states.size() / 2][9]);
  EXPECT_DOUBLE_EQ(0.27095936830732198,
                   output.states[output.states.size() / 2][10]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][11]);
  EXPECT_DOUBLE_EQ(-0.38844523226972694,
                   output.states[output.states.size() / 2][12]);
  EXPECT_DOUBLE_EQ(0.16986043619201119,
                   output.states[output.states.size() / 2][13]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][14]);
  EXPECT_DOUBLE_EQ(1.6573213121278003, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-0.013882068864692296, output.states.back()[1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[2]);
  EXPECT_DOUBLE_EQ(0.46802604978741491, output.states.back()[3]);
  EXPECT_DOUBLE_EQ(-0.17063101334085282, output.states.back()[4]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[5]);
  EXPECT_DOUBLE_EQ(-1.2481480178191757, output.states.back()[6]);
  EXPECT_DOUBLE_EQ(-0.26313782979120093, output.states.back()[7]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[8]);
  EXPECT_DOUBLE_EQ(-1.2883624086159355, output.states.back()[9]);
  EXPECT_DOUBLE_EQ(0.27516687291430969, output.states.back()[10]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[11]);
  EXPECT_DOUBLE_EQ(0.41116306451990181, output.states.back()[12]);
  EXPECT_DOUBLE_EQ(0.17248403908243198, output.states.back()[13]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[14]);

  cudaFree(dev_tol);
  cudaFree(dev_x0);
}
