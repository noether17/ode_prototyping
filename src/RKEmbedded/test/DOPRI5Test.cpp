#include <gtest/gtest.h>

#include <array>
#include <numeric>

#include "AllocatedState.hpp"
#include "BTDOPRI5.hpp"
#include "RKEmbedded.hpp"
#include "RawOutput.hpp"

class DOPRI5VanDerPolTest : public testing::Test {
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

  RKEmbeddedSingleThreaded<AllocatedState<2>, BTDOPRI5, decltype(ode_van),
                           RawOutput<AllocatedState<2>>>
      integrator{};
  RawOutput<AllocatedState<2>> output{};
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(DOPRI5VanDerPolTest, IntegrationStepsAreConsistent) {
  integrator.integrate(x0, t0, tf, atol, rtol, ode_van, output);

  EXPECT_EQ(76, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(0.67149697578171041, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, output.times.back());

  EXPECT_EQ(76, output.states.size());
  EXPECT_DOUBLE_EQ(2.0, output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7382670271053748,
                   output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.62280413467104634,
                   output.states[output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666704309109, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679718159, output.states.back()[1]);
}

TEST(DOPRI5ExpTest, IntegrationStepsAreConsistent) {
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
  auto integrator =
      RKEmbeddedSingleThreaded<AllocatedState<n_var>, BTDOPRI5,
                               decltype(ode_exp),
                               RawOutput<AllocatedState<n_var>>>{};
  auto output = RawOutput<AllocatedState<n_var>>{};

  integrator.integrate(x0, t0, tf, tol, tol, ode_exp, output);

  EXPECT_EQ(45, output.times.size());
  EXPECT_DOUBLE_EQ(0.0, output.times.front());
  EXPECT_DOUBLE_EQ(4.9896555947535841, output.times[output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, output.times.back());

  EXPECT_EQ(45, output.states.size());
  EXPECT_DOUBLE_EQ(0.0, output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, output.states.front()[n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, output.states.front()[n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, output.states[output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(734.42960843551498,
                   output.states[output.states.size() / 2][n_var / 2]);
  EXPECT_DOUBLE_EQ(1321.9732951839271,
                   output.states[output.states.size() / 2][n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.46804595254, output.states.back()[n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.44248271457, output.states.back()[n_var - 1]);
}
