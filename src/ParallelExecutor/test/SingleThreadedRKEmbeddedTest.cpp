#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <numeric>

#include "BTDOPRI5.hpp"
#include "BTDVERK.hpp"
#include "BTHE21.hpp"
#include "BTRKF45.hpp"
#include "BTRKF78.hpp"
#include "HeapState.hpp"
#include "RKEmbeddedParallel.hpp"
#include "RawOutput.hpp"
#include "SingleThreadedExecutor.hpp"

struct VanDerPolTest {
  auto static constexpr n_var = 2;
  auto static inline const x0 = HeapState<double, n_var>(std::array{2.0, 0.0});
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 2.0;
  auto static constexpr tol = 1.0e-10;
  auto static inline const atol =
      HeapState<double, n_var>(std::array{tol, tol});
  auto static inline const rtol = atol;
  auto constexpr operator()(auto&, auto const& x, auto& dxdt) {
    auto constexpr eps = 1.0;
    dxdt[0] = x[1];
    dxdt[1] = eps * (1.0 - x[0] * x[0]) * x[1] - x[0];
  }
  RawOutput<HeapState<double, n_var>> output{};
};

struct ExponentialTest {
  auto static constexpr n_var = 10;
  auto static constexpr x0_data = [] {
    auto temp = std::array<double, n_var>{};
    std::iota(temp.begin(), temp.end(), 0.0);
    return temp;
  }();
  auto static inline const x0 = HeapState<double, n_var>(x0_data);
  auto static constexpr t0 = 0.0;
  auto static constexpr tf = 10.0;
  auto static constexpr tol = 1.0e-6;
  auto static inline const atol = [] {
    auto temp = HeapState<double, n_var>{};
    std::fill(temp.data(), temp.data() + n_var, tol);
    return temp;
  }();
  auto static inline const rtol = atol;
  auto constexpr operator()(auto&, auto const& x, auto& dxdt) {
    for (auto i = 0; i < std::ssize(x); ++i) {
      dxdt[i] = x[i];
    }
  }
  RawOutput<HeapState<double, n_var>> output{};
};

class SingleThreadedRKEmbeddedTest : public testing::Test {
 protected:
  template <typename ButcherTableau, typename ODE>
  using Integrator =
      RKEmbeddedParallel<HeapState, double, ODE::n_var, ButcherTableau, ODE,
                         RawOutput<HeapState<double, ODE::n_var>>,
                         SingleThreadedExecutor>;
  SingleThreadedExecutor executor{};
};

/* Consistency tests (testing for double equality) are to ensure no accidental
 * algorithm changes are made during refactoring. These tests are far stricter
 * than the actual requirements. If an intentional change in algorithm results
 * in small differences in output, these values may be updated. */
TEST_F(SingleThreadedRKEmbeddedTest, HE21VanDerPolConsistencyTest) {
  auto test = VanDerPolTest{};
  auto integrator = Integrator<BTHE21, VanDerPolTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(183012, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.9274958393895415,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(183012, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.5633864957650419,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.74391761303139381,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666707211976, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679629066, test.output.states.back()[1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, HE21ExponentialConsistencyTest) {
  auto test = ExponentialTest{};
  auto integrator = Integrator<BTHE21, ExponentialTest>{};

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
      790.81720024690105,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1423.4709604444222,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.17777934085,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.92000281269,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, RKF45VanDerPolConsistencyTest) {
  auto test = VanDerPolTest{};
  auto integrator = Integrator<BTRKF45, VanDerPolTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(83, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.65364467640860291,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(83, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7493089062893852,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.61420762201954171,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666497810646, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745707118268, test.output.states.back()[1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, RKF45ExponentialConsistencyTest) {
  auto test = ExponentialTest{};
  auto integrator = Integrator<BTRKF45, ExponentialTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(50, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(5.0888631842534933,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(50, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      811.03335805992742,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1459.8600445078696,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110134.06230636688,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198241.3121514605,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, DOPRI5VanDerPolConsistencyTest) {
  auto test = VanDerPolTest{};
  auto integrator = Integrator<BTDOPRI5, VanDerPolTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(76, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.67149697578171041,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(76, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7382670271053748,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.62280413467104634,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666704309109, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745679718159, test.output.states.back()[1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, DOPRI5ExponentialConsistencyTest) {
  auto test = ExponentialTest{};
  auto integrator = Integrator<BTDOPRI5, ExponentialTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(45, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(4.9896555947535841,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(45, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      734.42960843551498,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1321.9732951839271,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.46804595254,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.44248271457,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, DVERKVanDerPolConsistencyTest) {
  auto test = VanDerPolTest{};
  auto integrator = Integrator<BTDVERK, VanDerPolTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(42, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.66331345368999683,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(42, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7433476890385384,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.61887401617955773,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.32331666651502594, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745684699728, test.output.states.back()[1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, DVERKExponentialConsistencyTest) {
  auto test = ExponentialTest{};
  auto integrator = Integrator<BTDVERK, ExponentialTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(32, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(5.1178634791056821,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(32, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      834.89108343292787,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      1502.8039501792707,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110132.30512693945,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198238.14922849106,
                   test.output.states.back()[test.n_var - 1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, RKF78VanDerPolConsistencyTest) {
  auto test = VanDerPolTest{};
  auto integrator = Integrator<BTRKF78, VanDerPolTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(15, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(0.72814921744061389,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(2.0, test.output.times.back());

  EXPECT_EQ(15, test.output.states.size());
  EXPECT_DOUBLE_EQ(2.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[1]);
  EXPECT_DOUBLE_EQ(1.7022210141241134,
                   test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(-0.64964031512382325,
                   test.output.states[test.output.states.size() / 2][1]);
  EXPECT_DOUBLE_EQ(0.3233166666554449, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(-1.8329745686289092, test.output.states.back()[1]);
}

TEST_F(SingleThreadedRKEmbeddedTest, RKF78ExponentialConsistencyTest) {
  auto test = ExponentialTest{};
  auto integrator = Integrator<BTRKF78, ExponentialTest>{};

  integrator.integrate(test.x0, test.t0, test.tf, test.atol, test.rtol, test,
                       test.output, executor);

  EXPECT_EQ(13, test.output.times.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.times.front());
  EXPECT_DOUBLE_EQ(4.4481779636028618,
                   test.output.times[test.output.times.size() / 2]);
  EXPECT_DOUBLE_EQ(10.0, test.output.times.back());

  EXPECT_EQ(13, test.output.states.size());
  EXPECT_DOUBLE_EQ(0.0, test.output.states.front()[0]);
  EXPECT_DOUBLE_EQ(5.0, test.output.states.front()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(9.0, test.output.states.front()[test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states[test.output.states.size() / 2][0]);
  EXPECT_DOUBLE_EQ(
      427.3544878966228,
      test.output.states[test.output.states.size() / 2][test.n_var / 2]);
  EXPECT_DOUBLE_EQ(
      769.23807821392074,
      test.output.states[test.output.states.size() / 2][test.n_var - 1]);
  EXPECT_DOUBLE_EQ(0.0, test.output.states.back()[0]);
  EXPECT_DOUBLE_EQ(110131.78807367262,
                   test.output.states.back()[test.n_var / 2]);
  EXPECT_DOUBLE_EQ(198237.21853261057,
                   test.output.states.back()[test.n_var - 1]);
}
