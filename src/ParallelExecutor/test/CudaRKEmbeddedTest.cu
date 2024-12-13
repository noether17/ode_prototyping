#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTHE21.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "CudaExecutor.cuh"
#include "CudaState.cuh"
#include "ExponentialTest.hpp"
#include "HeapState.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"
#include "VanDerPolTest.hpp"

class CudaRKEmbeddedTest : public testing::Test {
 protected:
  template <typename ButcherTableau, typename ODE>
  using Integrator =
      RKEmbeddedParallel<CudaState, double, ODE::n_var, ButcherTableau, ODE,
                         RawOutput<HeapState<double, ODE::n_var>>,
                         CudaExecutor>;
  CudaExecutor executor{};
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(CudaRKEmbeddedTest, HE21VanDerPolConsistencyTest) {
  auto test = VanDerPolTest<CudaState>{};
  auto integrator = Integrator<BTHE21, VanDerPolTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(183012, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.92749583938957947,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(183012, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.5633864957649997,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.74391761303142623,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666707212042, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679629208, test.output.states.back()[1]);
}

// This test contains small, but nonzero differences from single-threaded.
TEST_F(CudaRKEmbeddedTest, HE21ExponentialConsistencyTest) {
  auto test = ExponentialTest<CudaState>{};
  auto integrator = Integrator<BTHE21, ExponentialTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(11026, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(5.0636296424493379,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(11026, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      790.81720024690196,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1423.4709604444181,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.17777934125,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.92000281205,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(CudaRKEmbeddedTest, RKF45VanDerPolConsistencyTest) {
  auto test = VanDerPolTest<CudaState>{};
  auto integrator = Integrator<BTRKF45, VanDerPolTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(83, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.65364467697913309,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(83, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7493089059389617,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.61420762229567305,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.3233166649781048, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745707118286, test.output.states.back()[1]);
}

// This test contains small, but nonzero differences from single-threaded.
TEST_F(CudaRKEmbeddedTest, RKF45ExponentialConsistencyTest) {
  auto test = ExponentialTest<CudaState>{};
  auto integrator = Integrator<BTRKF45, ExponentialTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(50, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(5.0888631842530918,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(50, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      811.03335805960103,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1459.8600445072816,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110134.06230636722,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198241.312151461, test.output.states.back()[test.n_var - 1]);
}

TEST_F(CudaRKEmbeddedTest, DOPRI5VanDerPolConsistencyTest) {
  auto test = VanDerPolTest<CudaState>{};
  auto integrator = Integrator<BTDOPRI5, VanDerPolTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(76, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.6714969756893634,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(76, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7382670271628891,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.62280413462679141,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666704309209, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679718148, test.output.states.back()[1]);
}

// This test contains small, but nonzero differences from single-threaded.
TEST_F(CudaRKEmbeddedTest, DOPRI5ExponentialConsistencyTest) {
  auto test = ExponentialTest<CudaState>{};
  auto integrator = Integrator<BTDOPRI5, ExponentialTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(45, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(4.9896555947727883,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(45, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      734.42960844961965,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1321.9732952093148,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.46804595251,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.44248271445,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(CudaRKEmbeddedTest, DVERKVanDerPolConsistencyTest) {
  auto test = VanDerPolTest<CudaState>{};
  auto integrator = Integrator<BTDVERK, VanDerPolTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(42, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.6633134538972123,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(42, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7433476889102977,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.61887401627929084,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666651502444, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745684699745, test.output.states.back()[1]);
}

TEST_F(CudaRKEmbeddedTest, DVERKExponentialConsistencyTest) {
  auto test = ExponentialTest<CudaState>{};
  auto integrator = Integrator<BTDVERK, ExponentialTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(32, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(5.1178634788750603,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(32, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      834.89108324038477,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1502.8039498326918,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.30512693933,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.14922849069,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(CudaRKEmbeddedTest, RKF78VanDerPolConsistencyTest) {
  auto test = VanDerPolTest<CudaState>{};
  auto integrator = Integrator<BTRKF78, VanDerPolTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(15, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.72815440043062785,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(15, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7022176470385422,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.64964274849420467,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666665543596, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745686289227, test.output.states.back()[1]);
}

TEST_F(CudaRKEmbeddedTest, RKF78ExponentialConsistencyTest) {
  auto test = ExponentialTest<CudaState>{};
  auto integrator = Integrator<BTRKF78, ExponentialTest<CudaState>>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(13, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(4.4481779636137997,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(13, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      427.35448790129726,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      769.23807822233516,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110131.78807367253,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.21853261057,
                   test.output.states.back()[test.n_var - 1]);
}
