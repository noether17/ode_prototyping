#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "AllocatedState.hpp"
#include "RKF45.hpp"

class RKF45VanDerPolTest : public testing::Test {
 protected:
  auto static constexpr eps = 1.0;
  auto static constexpr ode_van = [](AllocatedState<2> const& x,
                                     AllocatedState<2>& dxdt) {
    dxdt[0] = x[1];
    dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
  };
  auto static inline const x0 = AllocatedState<2>(std::array{2.0, 0.0});
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 2.0;
  auto static constexpr tol = 1.0e-10;
  auto static inline const atol = AllocatedState<2>(std::array{tol, tol});
  auto static inline const rtol = atol;

  RKF45<decltype(ode_van), AllocatedState<2>> integrator{ode_van};
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(RKF45VanDerPolTest, IntegrationStepsAreConsistent) {
  integrator.integrate(x0, t0, tf, atol, rtol);

  EXPECT_EQ(83, integrator.times.size());
  EXPECT_DOUBLE_EQ(0.0, integrator.times.front());
  EXPECT_DOUBLE_EQ(0.65364467640860291,
                   integrator.times[integrator.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, integrator.times.back());

  EXPECT_EQ(83, integrator.states.size());
  EXPECT_DOUBLE_EQ(2.0, integrator.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, integrator.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7493089062893852,
                   integrator.states[integrator.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.61420762201954171,
                   integrator.states[integrator.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666497810646, integrator.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745707118268, integrator.states.back()[1]);
}

TEST(RKF45ExpTest, IntegrationStepsAreConsistent) {
  auto constexpr n_var = 10;
  auto constexpr ode_exp = [](AllocatedState<n_var> const& x,
                              AllocatedState<n_var>& dxdt) { dxdt = x; };
  auto x0_data = std::array<double, n_var>{};
  std::iota(x0_data.begin(), x0_data.end(), 0.0);
  auto x0 = AllocatedState<n_var>(x0_data);
  auto t0 = 0.0;
  auto tf = 10.0;
  auto tol = AllocatedState<n_var>{};
  fill(tol, 1.0e-6);
  auto integrator = RKF45<decltype(ode_exp), AllocatedState<n_var>>{ode_exp};

  integrator.integrate(x0, t0, tf, tol, tol);

  EXPECT_EQ(50, integrator.times.size());
  EXPECT_DOUBLE_EQ(0.0, integrator.times.front());
  EXPECT_DOUBLE_EQ(5.0888631842534933,
                   integrator.times[integrator.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, integrator.times.back());

  EXPECT_EQ(50, integrator.states.size());
  EXPECT_DOUBLE_EQ(0.0, integrator.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, integrator.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, integrator.states.front()[n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, integrator.states[integrator.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(811.03335805992742,
                   integrator.states[integrator.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1459.8600445078696,
                   integrator.states[integrator.states.size() / 2][n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, integrator.states.back()[0]);
  EXPECT_DOUBLE_EQ(110134.06230636688, integrator.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198241.3121514605, integrator.states.back()[n_var - 1]);
}
