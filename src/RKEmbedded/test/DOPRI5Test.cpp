#include <gtest/gtest.h>

#include <array>

#include "DOPRI5.hpp"
#include "VectorState.hpp"

class DOPRI5VanDerPolTest : public testing::Test {
 protected:
  auto static constexpr eps = 1.0;
  auto static constexpr ode_van = [](VectorState<2> const& x,
                                     VectorState<2>& dxdt) {
    dxdt[0] = x[1];
    dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
  };
  auto static constexpr x0 = std::array{2.0, 0.0};
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 2.0;
  auto static constexpr tol = 1.0e-10;
  auto static constexpr atol = std::array{tol, tol};
  auto static constexpr rtol = atol;

  DOPRI5<decltype(ode_van), VectorState<2>> integrator{ode_van};
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(DOPRI5VanDerPolTest, IntegrationStepsAreConsistent) {
  integrator.integrate(x0, t0, tf, atol, rtol);

  EXPECT_EQ(76, integrator.times.size());
  EXPECT_DOUBLE_EQ(0.0, integrator.times.front());
  EXPECT_DOUBLE_EQ(0.67149697578171041,
                   integrator.times[integrator.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, integrator.times.back());

  EXPECT_EQ(76, integrator.states.size());
  EXPECT_DOUBLE_EQ(2.0, integrator.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, integrator.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7382670271053748,
                   integrator.states[integrator.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.62280413467104634,
                   integrator.states[integrator.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666704309109, integrator.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679718159, integrator.states.back()[1]);
}
